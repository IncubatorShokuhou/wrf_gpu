"""Read-only full-grid operands of the port's final-RK scalar chain (CPU, release defaults).

Runs the real Swiss case through the production step (``_advance_chunk`` -> ``_advance_chunk_fori``, one
step per call, jit) and taps the scalar-chain call sites with ``jax.debug.callback``:
RK1 horizontal diffusion (xkhh, muts), sixth-order sc_tend, QV boundary sc_tend,
rk_update_scalar_pd (moist_old, physics/other sc_tend, preload), the native
advect_scalar_pd kernel (all operands + tendency), the stage tendencies, the large-step
update and flow_dep_bdy.  Operands and results come from ONE compiled program, so a
pristine replay on them is a same-operand comparison.  The taps only copy values to the
host; they change no operand, result, cadence or equation of the scalar chain.
Disclosed program change (critic F5): the CPU memory cap needs the RRTMG column tile at 128
(GPUWRF_RRTMG_COLUMN_TILE_COLS=128 and om._M9_RRTMG_COLUMN_TILE_COLS; release 4096). That is a
radiation-only change, byte-neutral per A43 [I], and it cannot affect a same-program
operand/result comparison.  NOT a GPU claim.

Per-step negative census of preload/update/final for every step (summary.jsonl);
full capture (npz) at --capture steps and at steps that set a new per-species minimum.
"""
from __future__ import annotations

import argparse
import ctypes
import functools
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proofs/o1_nlbind")]
assert os.environ.get("JAX_PLATFORMS") == "cpu", "CPU only (E50)"

import cpu_safe  # noqa: E402  (E105 LAPACK gtsv deadlock workaround; harness-wide)

cpu_safe.install()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf  # noqa: E402,F401  (applies release defaults before submodules)
from gpuwrf._fast_defaults import FAST_DEFAULTS_STATUS  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.kernels import dyn_pd_fp32  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

CAP = {"buf": [], "on": True}
META: dict[int, tuple[str, dict]] = {}
_SEQ = [0]


def tap(tag: str, static: dict, payload) -> None:
    seq = _SEQ[0]
    _SEQ[0] += 1
    META[seq] = (tag, static)
    leaves, treedef = jax.tree.flatten(payload)

    def cb(*vals):
        if CAP["on"]:
            CAP["buf"].append((seq, jax.tree.unflatten(treedef, [np.array(v) for v in vals])))

    jax.debug.callback(cb, *leaves)


def wrap(module, name, recorder):
    original = getattr(module, name)

    @functools.wraps(original)
    def tapped(*args, **kwargs):
        result = original(*args, **kwargs)
        recorder(args, kwargs, result)
        return result

    setattr(module, name, tapped)


def species_dict(state, names):
    return {n: getattr(state, n) for n in names if getattr(state, n, None) is not None}


def install():
    def hdiff(args, kwargs, result):
        ref, xkhh, muts, _ = args
        tap("hdiff", {"names": tuple(result)}, {"xkhh": xkhh, "muts": muts, "out": dict(result)})

    def diff6(args, kwargs, result):
        tap("diff6", {"dt": float(kwargs["dt"])}, {"field": args[0], "mu": args[1], "out": result})

    def bdy(args, kwargs, result):
        tap("bdy", {"names": tuple(result)}, {"out": dict(result)})

    def pd(args, kwargs, result):
        origin, species, sources, dt_rk, _ = args
        tap("pd", {"species": tuple(species), "dt_rk": float(dt_rk),
                   "sources": tuple(None if s is None else tuple(s) for s in sources)},
            {"q_old": species_dict(origin, species), "mu_old": origin.mu_total,
             "src": tuple({} if s is None else dict(s) for s in sources), "out": species_dict(result, species)})

    def pdk(args, kwargs, result):
        names = ("fields", "fields_old", "ru", "rv", "rom", "mut", "mu_old", "c1", "c2", "msftx", "msfty",
                 "rdzw", "fzm", "fzp")
        static = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in kwargs.items()
                  if k in ("rdx", "rdy", "dt")}
        tap("pdk", static, {**dict(zip(names, args)), "out": result})

    def stage(args, kwargs, result):
        tap("stage", {"species": tuple(args[1])}, {"out": tuple(result)})

    def upd(args, kwargs, result):
        state, origin = args[:2]
        species = tuple(kwargs.get("species", om._MOISTURE_SPECIES))
        tap("upd", {"species": species, "dt_rk": float(kwargs["dt_rk"])},
            {"mu_new": state.mu_total, "mu_old": origin.mu_total, "q_old": species_dict(origin, species),
             "q_tend": tuple(kwargs["q_tendencies"]), "out": species_dict(result, species)})

    def flow(args, kwargs, result):
        state, names, ru_m, rv_m = args[:4]
        names = tuple(names)
        tap("flow", {"names": names}, {"before": species_dict(state, names), "ru": ru_m, "rv": rv_m,
                                       "after": species_dict(result, names)})

    wrap(om, "_diffopt1_scalar_horizontal_diffusion", hdiff)
    wrap(om, "wrf_sixth_order_scalar_tendf", diff6)
    wrap(om, "root_scalar_boundary_tendencies", bdy)
    wrap(om, "_rk_update_scalar_pd", pd)
    wrap(dyn_pd_fp32, "advect_scalar_pd_fp32", pdk)
    wrap(om, "_root_scalar_stage_tendencies", stage)
    wrap(om, "_apply_moisture_large_step", upd)
    wrap(om, "_root_flow_dep", flow)


def neg(a):
    a = np.asarray(a)
    m = a < 0
    if not m.any():
        return {"count": 0}
    flat = int(np.argmin(np.where(m, a, np.inf)))
    k, j, i = np.unravel_index(flat, a.shape)
    ny, nx = a.shape[-2:]
    ring = int(min(j, ny - 1 - j, i, nx - 1 - i))
    interior = int((m[:, 1:-1, 1:-1]).sum())
    return {"count": int(m.sum()), "interior": interior, "min": float(a.reshape(-1)[flat]), "kji": [int(k), int(j), int(i)],
            "ring_of_min": ring}


def analyse_step(buf, dt, it, wall):
    """Group one step's taps; per-species negative census of preload / update / final."""
    buf = sorted(buf, key=lambda r: r[0])
    by = {}
    for seq, payload in buf:
        tag, static = META[seq]
        by.setdefault(tag, []).append((static, payload))
    final_upd = [x for x in by.get("upd", []) if abs(x[0]["dt_rk"] - dt) < 1e-9]
    # flow_dep_bdy runs after every RK stage and once more post-physics (opmode ~11114):
    # the final-RK call is the one whose input equals the final-stage update result.
    final_flow = []
    if final_upd:
        ref = final_upd[-1][1]["out"]
        final_flow = [x for x in by.get("flow", []) if x[1]["before"]
                      and all(np.array_equal(v, ref[k]) for k, v in x[1]["before"].items() if k in ref)][:1]
    row = {"step": it, "wall_s": wall, "taps": {k: len(v) for k, v in by.items()}}
    stats = {}
    if by.get("pd"):
        s, pl = by["pd"][-1]
        stats["pre"] = {nme: neg(v) for nme, v in pl["out"].items()}
    if final_upd:
        s, pl = final_upd[-1]
        stats["upd"] = {nme: neg(v) for nme, v in pl["out"].items()}
    if final_flow:
        s, pl = final_flow[-1]
        fin = dict(final_upd[-1][1]["out"]) if final_upd else {}
        fin.update(pl["after"])
        stats["fin"] = {nme: neg(v) for nme, v in fin.items()}
    row["neg"] = stats
    return row, stats, by, final_upd, final_flow


def build_arrays(by, final_upd, final_flow, row):
    """Full-grid operand/result arrays of one captured step (npz payload)."""
    arrays = {}
    s, pl = by["pd"][-1]
    species = s["species"]
    arrays["mu1"] = pl["mu_old"]
    for nme in species:
        arrays[f"qold_{nme}"] = pl["q_old"][nme]
        arrays[f"pre_{nme}"] = pl["out"][nme]
        for i, src in enumerate(pl["src"]):
            if nme in src:
                arrays[f"src{i}_{nme}"] = src[nme]
    if by.get("hdiff"):
        s2, ph = by["hdiff"][-1]
        arrays["xkhh"], arrays["muts1"] = ph["xkhh"], ph["muts"]
        for nme, v in ph["out"].items():
            arrays[f"hdiff_{nme}"] = v
    for j, (s6, p6) in enumerate(by.get("diff6", [])):
        arrays[f"diff6call{j}_field"] = p6["field"]
        arrays[f"diff6call{j}_mu"] = p6["mu"]
        arrays[f"diff6call{j}_out"] = p6["out"]
        arrays[f"diff6call{j}_dt"] = np.float64(s6["dt"])
    for sb, pb in by.get("bdy", [])[-1:]:
        for nme, v in pb["out"].items():
            arrays[f"bdy_{nme}"] = v
    pk_static, pk = by["pdk"][-1]
    for key in ("ru", "rv", "rom", "mut", "mu_old"):
        arrays[f"pdk_{key}"] = pk[key]
    arrays["pdk_rdx"], arrays["pdk_rdy"], arrays["pdk_dt"] = (np.float64(pk_static[k]) for k in ("rdx", "rdy", "dt"))
    pdk_names = species  # one stacked launch in species order (asserted below)
    assert pk["fields"].shape[0] == len(pdk_names), (pk["fields"].shape, pdk_names)
    for i, nme in enumerate(pdk_names):
        arrays[f"qcur_{nme}"] = pk["fields"][i]
        arrays[f"qoldpd_{nme}"] = pk["fields_old"][i]
        arrays[f"adv_{nme}"] = pk["out"][i]
    # caller-wiring cross-check operands: RK1/RK2 stage masses and the RK2 scalar result
    stage_upd = sorted(by.get("upd", []), key=lambda x: x[0]["dt_rk"])
    if len(stage_upd) == 3:
        (s1u, p1u), (s2u, p2u) = stage_upd[0], stage_upd[1]
        arrays["upd1_mu_new"], arrays["upd1_mu_old"] = p1u["mu_new"], p1u["mu_old"]
        arrays["upd2_mu_new"] = p2u["mu_new"]
        for nme, v in p2u["out"].items():
            arrays[f"upd2_{nme}"] = v
        for nme, v in p1u["q_old"].items():
            arrays[f"upd1qold_{nme}"] = v
        ref2 = p2u["out"]
        flow2 = [x for x in by.get("flow", []) if x[1]["before"]
                 and all(np.array_equal(v, ref2[k]) for k, v in x[1]["before"].items() if k in ref2)][:1]
        for sf2, pf2 in flow2:
            for nme, v in pf2["after"].items():
                arrays[f"flow2_{nme}"] = v
    su, pu = final_upd[-1]
    arrays["mu2"] = pu["mu_new"]
    for nme, tq in zip(su["species"], pu["q_tend"]):
        arrays[f"qtend_{nme}"] = tq
        arrays[f"upd_{nme}"] = pu["out"][nme]
    if final_flow:
        sf, pf = final_flow[-1]
        arrays["flow_ru"], arrays["flow_rv"] = pf["ru"], pf["rv"]
        for nme, v in pf["after"].items():
            arrays[f"fin_{nme}"] = v
        row["flow_names"] = list(sf["names"])
    return arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--capture", default="1,2,3")
    p.add_argument("--max-records", type=int, default=40)
    p.add_argument("--record-after", type=int, default=1, help="record captures only after this step (skip spin-up)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == "cpu"
    capture_steps = {int(s) for s in a.capture.split(",") if s}
    case = a.out / "case"
    case.mkdir(exist_ok=True)
    for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01"):
        if not (case / name).exists():
            (case / name).symlink_to((a.case / name).resolve())
    report = {"scope": __doc__, "source": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
              "dirty": os.popen(f"git -C {ROOT} status --porcelain -- src").read().strip(),
              "inputs": {n: hashlib.sha256((case / n).read_bytes()).hexdigest() for n in
                         ("namelist.input", "wrfinput_d01", "wrfbdy_d01")},
              "fast_defaults": {k: v for k, v in FAST_DEFAULTS_STATUS.items() if k == "enabled"},
              "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GPUWRF_", "JAX_", "XLA_"))},
              "device": str(jax.devices()[0]), "steps": a.steps, "capture": sorted(capture_steps),
              "harness": "cpu_safe.install (Thomas scan for lax.linalg.tridiagonal_solve, E105)"}
    om._M9_RRTMG_COLUMN_TILE_COLS = int(os.environ.get("GPUWRF_RRTMG_COLUMN_TILE_COLS", "128"))
    install()
    cfg = NestedPipelineConfig(case, a.out / "output", a.out / "proof", hours=1, max_dom=1)
    t = time.monotonic()
    _, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
    n, c = bundles["d01"].namelist, carries["d01"]
    report["load_s"] = time.monotonic() - t
    om._resolve_operational_suite(n)
    m = n.metrics
    constants = {k: np.asarray(getattr(m, k)) for k in ("c1h", "c2h", "c1f", "c2f", "msftx", "msfty", "msfux", "msfuy",
                                                         "msfvx", "msfvy", "fnm", "fnp", "rdnw")}
    work = {k: getattr(n, k) for k in ("mp_physics", "dt_s", "diff_opt", "km_opt", "diff_6th_opt", "diff_6th_factor",
                                       "moist_adv_opt", "scalar_adv_opt", "bl_pbl_physics", "cu_physics", "rk_order",
                                       "radiation_cadence_steps") if hasattr(n, k)}
    work["spec_zone"] = int(n.boundary_config.spec_zone)
    work["dx"] = float(n.grid.projection.dx_m)
    work["dy"] = float(n.grid.projection.dy_m)
    report["work"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in work.items()}
    np.savez(a.out / "constants.npz", **constants)
    clock = om.build_clock_base(n)
    (a.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("LOADED", report["load_s"], report["work"], flush=True)
    records: dict[str, float] = {}
    n_records = 0
    summary = open(a.out / "summary.jsonl", "a")
    dt = float(n.dt_s)
    for it in range(1, a.steps + 1):
        CAP["buf"].clear()
        t = time.monotonic()
        c = om._advance_chunk(c, n, it, clock, n_steps=1, cadence=int(n.radiation_cadence_steps))  # production entry (unalias before donating jit)
        jax.block_until_ready(c)
        wall = time.monotonic() - t
        buf = sorted(CAP["buf"], key=lambda r: r[0])
        row, stats, by, final_upd, final_flow = analyse_step(buf, dt, it, wall)
        new_record = []
        for nme, st in stats.get("fin", {}).items():
            if st["count"] and st["min"] < records.get(nme, 0.0):
                records[nme] = st["min"]
                new_record.append(nme)
        want = it in capture_steps or (new_record and n_records < a.max_records and it > a.record_after)
        if want and by.get("pd") and final_upd:
            if it not in capture_steps:
                n_records += 1
            path = a.out / f"step{it:05d}.npz"
            arrays = build_arrays(by, final_upd, final_flow, row)
            np.savez_compressed(path, **arrays)
            row["captured"] = str(path)
            row["species"] = list(by["pd"][-1][0]["species"])
            row["new_record"] = new_record
        summary.write(json.dumps(row) + "\n")
        summary.flush()
        print("STEP", it, f"{wall:.1f}s", {k: {n_: v.get("min") for n_, v in d.items() if v["count"]} for k, d in stats.items()},
              "CAPTURED" if "captured" in row else "", flush=True)
        bad = sum(int((~np.isfinite(np.asarray(x))).sum()) for x in jax.tree.leaves(c.state)
                  if np.issubdtype(np.asarray(x).dtype, np.floating))
        if bad:
            raise SystemExit(f"nonfinite state at step {it}: {bad}")
    report["maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["status"] = "DONE"
    (a.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
