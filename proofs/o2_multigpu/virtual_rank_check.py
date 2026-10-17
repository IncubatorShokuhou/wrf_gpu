"""M3 virtual-rank check: the REAL single-domain step decomposed over P CPU devices vs the single-device run.

Loads a real case (default PROD d01 120x70x44, wrfinput+wrfbdy) through the product loader, optionally crops a
window / mirror-tiles it, runs N steps once on one device (chunks of k steps, the same compiled loop the tiles
use) and then every requested ARM decomposed (px x py tiles, deep halo, halo exchange every k steps), comparing
every carry leaf (bitwise count, max abs/rel diff, spatial profile: halo errors sit near the internal owned
boundaries, per-shape compile round-off does not). One process: load + reference once, arms sequentially.

ARM syntax  PXxPY[:hH][:no_corners|:bdy_shift][:live]   (h default: analytic required_halo)
  e.g. --arms 2x1,1x1:h0,2x1:h20,2x2:no_corners
Mutants (must FAIL): short halo (:hH below the radius), :no_corners (2-D split without diagonal messages),
:bdy_shift (true-edge boundary strips of every tile but rank 0 sliced one cell off).

  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 GPUWRF_WRF_ROOT=... PYTHONPATH=src \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 python virtual_rank_check.py --arms 2x1,2x2 --steps 1 --out DIR
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, default=Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"))
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--arms", default="2x1", help="comma-separated arms, see module doc")
ap.add_argument("--steps", type=int, default=1)
ap.add_argument("--start-step", type=int, default=1)
ap.add_argument("--exchange-every", type=int, default=1)
ap.add_argument("--crop", default=None, help="x0:x1,y0:y1 mass-cell window of the loaded domain")
ap.add_argument("--mirror", default=None, help="MXxMY mirror tiling of the (cropped) domain")
ap.add_argument("--save-states", action="store_true", help="save each arm's gathered state (npz) for analysis")
ap.add_argument("--reference-npz", type=Path, help="preserved plain-loop reference; fresh 1x1 byte control is mandatory")
ap.add_argument("--expect-mismatch", default="", help="comma-separated negative arm names; actual finite byte mismatch required")
ap.add_argument("--release", action="store_true", help="release fast paths (needs GPUWRF_FAST_DEFAULTS=1): Pallas "
                "kernels in interpret mode + pure-JAX tridiagonal on CPU (scripts/gpu_smoke_v034._cpu_patches)")
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu", "CPU only"


def parse_arm(text: str) -> dict:
    parts = text.split(":")
    m = re.fullmatch(r"(\d+)x(\d+)", parts[0])
    if not m:
        raise SystemExit(f"bad arm {text!r}")
    arm = {"name": text, "px": int(m.group(1)), "py": int(m.group(2)), "halo": None, "mutant": "none", "live": False}
    for p in parts[1:]:
        if p.startswith("h"):
            arm["halo"] = int(p[1:])
        elif p in ("no_corners", "bdy_shift"):
            arm["mutant"] = p
        elif p == "live":
            arm["live"] = True
        else:
            raise SystemExit(f"bad arm option {p!r} in {text!r}")
    return arm


arms = [parse_arm(a) for a in args.arms.split(",") if a]
nproc = max(a["px"] * a["py"] for a in arms)
assert f"--xla_force_host_platform_device_count={nproc}" in os.environ.get("XLA_FLAGS", "") or nproc == 1, \
    f"set XLA_FLAGS=--xla_force_host_platform_device_count={nproc}"

t_start = time.perf_counter()
import gpuwrf  # noqa: F401,E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
if args.release:
    assert os.environ.get("GPUWRF_FAST_DEFAULTS", "1") != "0", "--release needs GPUWRF_FAST_DEFAULTS=1"
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("gpu_smoke_v034", Path(__file__).resolve().parents[2] / "scripts/gpu_smoke_v034.py")
    _smoke = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_smoke)
    _smoke._cpu_patches()
from gpuwrf.parallel import decomposition as dc  # noqa: E402
from gpuwrf.parallel.harness import compare, fresh_copy, load_case  # noqa: E402
from gpuwrf.parallel.tile_driver import TileRuntime, release_step_live_leaves, required_halo  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert len(jax.devices()) >= nproc and jax.devices()[0].platform == "cpu"
args.out.mkdir(parents=True, exist_ok=True)
mirror = tuple(int(v) for v in args.mirror.lower().split("x")) if args.mirror else None
carry, nl, meta = load_case(args.input, args.out / "scratch", crop=args.crop, mirror=mirror, device=jax.devices()[0])
nx, ny = nl.grid.nx, nl.grid.ny
k = args.exchange_every
cadence = int(nl.radiation_cadence_steps)
clock = om.build_clock_base(nl)
carry0 = [np.asarray(x) for x in jax.tree_util.tree_leaves(carry)]
base = {"args": {kk: str(v) for kk, v in vars(args).items()}, "case": meta, "release": bool(args.release),
        "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "xla_flags": os.environ.get("XLA_FLAGS")}

tag = f"s{args.start_step}_n{args.steps}_k{k}{'_release' if args.release else ''}"
ref_path = args.reference_npz or args.out / f"reference_{tag}.npz"
if ref_path.exists():
    with np.load(ref_path) as z:
        assert set(z.files) == {f"l{i}" for i in range(len(carry0))}, "reference leaf schema differs"
        ref_leaves = [z[f"l{i}"] for i in range(len(carry0))]
    base["reference"] = f"reused {ref_path.name}"
else:
    t0 = time.perf_counter()
    c = fresh_copy(carry)  # the release loop donates its carry (GPUWRF_CARRY_DONATE=1 under --release)
    for i in range(0, args.steps, k):
        c = om._advance_chunk_fori(c, nl, np.int32(args.start_step + i), clock, n_steps=k, cadence=cadence)
    jax.block_until_ready(c)
    ref_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(c)]
    np.savez(ref_path, **{f"l{i}": x for i, x in enumerate(ref_leaves)})
    base["reference_s"] = round(time.perf_counter() - t0, 1)
    del c

# Same-program control: a 1x1 TileRuntime (tile == domain, no messages) runs the step inside the identical tile
# wrapper; arms are compared with it (decomposition effect only) AND with the plain reference (wrapper effect).
ctrl_path = args.out / f"control1x1_{tag}.npz"
if ctrl_path.exists():
    with np.load(ctrl_path) as z:
        ctrl_leaves = [z[f"l{i}"] for i in range(len(carry0))]
else:
    t0 = time.perf_counter()
    rt0 = TileRuntime.create(carry, nl, dc.DecompositionSpec.build(nx, ny, 1, 1, 0), jax.devices()[:1], cadence=cadence,
                             exchange_every=k, clock_base=clock, start_step=args.start_step)
    rt0.advance(args.start_step, args.steps)
    rt0.block()
    ctrl_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(rt0.gather())]
    np.savez(ctrl_path, **{f"l{i}": x for i, x in enumerate(ctrl_leaves)})
    base["control_s"] = round(time.perf_counter() - t0, 1)
    del rt0
    gc.collect()
    jax.clear_caches()
ctrl_vs_ref = compare(ctrl_leaves, ref_leaves, [dc.classify(np.shape(x), dc.DecompositionSpec.build(nx, ny, 1, 1, 0))
                                               for x in carry0], carry0)
base["control_vs_plain_reference"] = {kk: ctrl_vs_ref[kk] for kk in ("verdict", "bitwise_leaves", "leaves")}
print(json.dumps({"control_1x1_vs_plain_reference": base["control_vs_plain_reference"]}), flush=True)
if ctrl_vs_ref["verdict"] != "BITWISE":
    (args.out / "CONTROL_FAILURE.json").write_text(json.dumps(ctrl_vs_ref, indent=1) + "\n")
    raise SystemExit("fresh 1x1 control failed strict byte/finite gate; preserved reference reuse refused")

live_mask = None
real_bdy = dc._slice_bdy
expected_mismatch = set(filter(None, args.expect_mismatch.split(",")))
assert expected_mismatch <= {arm["name"] for arm in arms}, "unknown expected-negative arm"
outcomes = []
for arm in arms:
    print(json.dumps({"starting_arm": arm["name"], "source": "frozen snapshot"}), flush=True)
    report = dict(base)
    halo = arm["halo"] if arm["halo"] is not None else required_halo(nl.acoustic_substeps, k)
    spec = dc.DecompositionSpec.build(nx, ny, arm["px"], arm["py"], halo)
    report.update(arm=arm, halo=halo, spec=spec.describe())
    plan_filter = None
    if arm["mutant"] == "no_corners":
        def plan_filter(plan, _spec=spec):
            return {(s, d): m for (s, d), m in plan.items()
                    if _spec.tiles[s].ix == _spec.tiles[d].ix or _spec.tiles[s].iy == _spec.tiles[d].iy}
    if arm["mutant"] == "bdy_shift":
        def shifted(a, t, sp):
            if t.rank == 0:
                return real_bdy(a, t, sp)
            moved = dc.Tile(t.rank, t.ix, t.iy, t.own_x, t.own_y, (t.ext_x[0] + 1, t.ext_x[1] + 1),
                            (t.ext_y[0] + 1, t.ext_y[1] + 1))
            return real_bdy(a, moved, sp)
        dc._slice_bdy = shifted
    live = None
    if arm["live"]:
        if live_mask is None:
            t0 = time.perf_counter()
            live_mask = release_step_live_leaves(carry, nl, clock, cadence)
            base["live_analysis_s"] = round(time.perf_counter() - t0, 1)
        live = live_mask
        report["live"] = {"leaves": int(sum(live)), "of": len(live), "dead_leaf_ids": [i for i, v in enumerate(live) if not v]}
    try:
        t0 = time.perf_counter()
        rt = TileRuntime.create(carry, nl, spec, jax.devices()[: arm["px"] * arm["py"]], cadence=cadence,
                                exchange_every=k, clock_base=clock, plan_filter=plan_filter, live=live,
                                start_step=args.start_step)
        report["create_s"] = round(time.perf_counter() - t0, 1)
        report["message_bytes"] = {f"{s}->{d}": b for (s, d), b in rt.message_bytes().items()}
        t0 = time.perf_counter()
        rt.advance(args.start_step, args.steps)
        rt.block()
        report["decomposed_s"] = round(time.perf_counter() - t0, 1)
        got = jax.tree_util.tree_leaves(rt.gather())
        kinds = rt.kinds
    finally:
        dc._slice_bdy = real_bdy
    cmp = compare(got, ref_leaves, kinds, carry0)
    vs_ctrl = compare(got, ctrl_leaves, kinds, carry0)
    report["vs_control"] = {kk: vs_ctrl[kk] for kk in ("verdict", "bitwise_leaves", "leaves")}
    report["vs_control_mismatch_rows"] = vs_ctrl["mismatch_rows"]
    # Spatial structure per leaf: max |diff| within +-8 cells of an internal owned boundary ("near") vs elsewhere
    # ("far"). A halo/exchange error enters at the boundaries (near >> far); per-shape compiled round-off has no
    # such structure (near ~ far). Also the max distance in ULPs of the reference value.
    cuts_x = sorted({t.own_x[0] for t in spec.tiles if t.own_x[0] > 0})
    cuts_y = sorted({t.own_y[0] for t in spec.tiles if t.own_y[0] > 0})
    leaf_stats, worst_ratio, max_ulps = [], 0.0, 0.0
    for i, (a, b, kind) in enumerate(zip(got, ref_leaves, kinds)):
        if kind[0] != "h" or not np.issubdtype(np.asarray(b).dtype, np.floating):
            continue
        a, b = np.asarray(a), np.asarray(b)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        d = np.where(np.isnan(d) & ~(np.isnan(a) & np.isnan(b)), np.inf, d)  # a NaN difference is a maximal difference
        if not (d > 0).any():
            continue
        d2 = d.reshape(-1, *d.shape[-2:]).max(axis=0)
        mask = np.zeros(d2.shape, dtype=bool)
        for c in cuts_x:
            mask[:, max(0, c - 8):c + 8] = True
        for c in cuts_y:
            mask[max(0, c - 8):c + 8, :] = True
        near_v = float(d2[mask].max()) if mask.any() else 0.0
        far_v = float(d2[~mask].max()) if (~mask).any() else 0.0
        nz = (np.abs(b) > np.finfo(b.dtype).tiny) & np.isfinite(d)
        ulps = float(np.nanmax(d[nz] / np.spacing(np.abs(b[nz])).astype(np.float64))) if nz.any() else 0.0
        ratio = near_v / far_v if far_v > 0 else (float("inf") if near_v > 0 else 0.0)
        worst_ratio = max(worst_ratio, ratio)
        max_ulps = max(max_ulps, ulps)
        leaf_stats.append({"leaf": i, "near_max_abs": near_v, "far_max_abs": far_v, "near_far_ratio": ratio,
                           "field_max_abs": float(np.nanmax(np.abs(b))), "max_ulps": ulps})
    report["spatial"] = {"worst_near_far_ratio": worst_ratio, "max_ulps": max_ulps, "leaves": leaf_stats}
    report["compare"] = cmp
    # Invalid schema/nonfinite values are never a successful negative numeric control (E200/E202).
    wanted = "MISMATCH" if arm["name"] in expected_mismatch else "BITWISE"
    report["gate"] = "PASS" if cmp["verdict"] == wanted else "FAIL"
    report["expected_verdict"] = wanted
    outcomes.append({"arm": arm["name"], "gate": report["gate"], "verdict": cmp["verdict"], "expected": wanted})
    report["process_s"] = round(time.perf_counter() - t_start, 1)
    report["ref_hash"] = hashlib.sha256(b"".join(x.tobytes() for x in ref_leaves)).hexdigest()[:16]
    report["got_hash"] = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in got)).hexdigest()[:16]
    name = f"arm_{arm['name'].replace(':', '_')}_h{halo}_{tag}.json"
    if args.save_states:
        np.savez(args.out / name.replace(".json", ".npz"), **{f"l{i}": np.asarray(x) for i, x in enumerate(got)})
    (args.out / name).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"arm": arm["name"], "halo": halo, "vs_control": vs_ctrl["verdict"],
                      "vs_control_bitwise": f"{vs_ctrl['bitwise_leaves']}/{vs_ctrl['leaves']}",
                      "vs_reference": cmp["verdict"], "bitwise": f"{cmp['bitwise_leaves']}/{cmp['leaves']}",
                      "non_finite": cmp["non_finite"], "worst_near_far_ratio": report["spatial"]["worst_near_far_ratio"],
                      "max_ulps": report["spatial"]["max_ulps"],
                      "create_s": report["create_s"], "decomposed_s": report["decomposed_s"]}), flush=True)
    del rt, got
    gc.collect()
    jax.clear_caches()
summary = {"verdict": "PASS" if outcomes and all(x["gate"] == "PASS" for x in outcomes) else "FAIL", "arms": outcomes}
(args.out / "GATE.json").write_text(json.dumps(summary, indent=1) + "\n")
raise SystemExit(0 if summary["verdict"] == "PASS" else 1)
