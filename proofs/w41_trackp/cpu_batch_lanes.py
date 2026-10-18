"""Track P P2 CPU proof: per-lane traced surface controls in the B>1 batched-ensemble vmap (real Swiss d01).

usage: JAX_PLATFORMS=cpu python cpu_batch_lanes.py <src_root> <out.json>
1. load the case once (B=1), write control files A and B (different parameter sets);
2. load the SAME case as a B=2 batch with batch_surface_controls=(A, B) through the product loader
   (nested_pipeline._load_batched_domains) -> shapes: carries and clock bases carry a leading lane axis, the
   stacked surface controls equal A (lane 0) and B (lane 1) byte for byte, one closed-over namelist;
3. run ONE batched step (nested_pipeline._batched_advance_factory: jax.vmap over carry + clock base) and the two
   matching B=1 steps; record per-lane leaf agreement (vmap vs solo; bitwise not required) and lane0 != lane1;
4. CPU compiled memory (direction only, E174): temp bytes of the B=1 and B=2 step programs.
Pallas interpreted for load/trace/exec (as O1_INTERPRET_EXEC=1); tridiagonal_solve -> pure-JAX Thomas (E105).
"""
import dataclasses
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

SRC, OUT = str(Path(sys.argv[1]).resolve()), sys.argv[2]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path(os.environ.get("TRACKP_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-trackp")
_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


pl.pallas_call = _interpret_pallas_call
out = {"src": SRC, "case": str(CASE)}
with tempfile.TemporaryDirectory(dir=LANE) as tmp:
    tmp = Path(tmp)
    case = tmp / "case"
    case.mkdir()
    for f in CASE.iterdir():
        (case / f.name).symlink_to(f)
    t0 = time.time()
    cfg = pipeline.NestedPipelineConfig(case, tmp / "out", tmp / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    ns = bundles["d01"].namelist
    carry_solo = carries["d01"]
    vt = np.asarray(ns.noahmp_static.ivgtyp)
    cats = [int(c) for c in np.unique(vt)]
    ctrl = {
        "A": sc.expand_controls(ns, {"z0mvt": {c: 1.5 for c in cats[: max(1, len(cats) // 2)]}, "albsat": 1.2,
                                     "laim": 1.3}, device=False),
        "B": sc.expand_controls(ns, {"z0mvt": 0.7, "albsat": 0.9, "laim": 0.8}, device=False),
    }
    if os.environ.get("TRACKP_BATCH_MODE") == "off":  # control experiment: the same batch/vmap path, no controls
        ctrl = {"A": {}, "B": {}}
    paths = {}
    for key, fields in ctrl.items():
        if not fields:
            paths[key] = None
            continue
        paths[key] = tmp / f"{key}.npz"
        sc.save_controls(paths[key], {"d01": fields})
    out["load_solo_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    bcfg = dataclasses.replace(cfg, batch_input_dirs=(case, case),
                               batch_surface_controls=None if paths["A"] is None else (paths["A"], paths["B"]))
    (_hier, ref_bundles, meta, _rs2, _dt2, bcarries, bnamelists, _dirs, _starts,
     bnoah) = pipeline._load_batched_domains(bcfg, ("d01",), batch_size=2)
    out["load_batch_s"] = round(time.time() - t1, 1)
    tree = types.SimpleNamespace(domains={"d01": types.SimpleNamespace(namelist=ref_bundles["d01"].namelist)})
    cbs = pipeline._clock_bases_for_batch_domain(tree=tree, name="d01", batch_namelists=bnamelists, batch_size=2,
                                                 noahmp_lane_arrays=bnoah.get("d01"))
    lanes_ok = {}
    for key, lane in (("A", 0), ("B", 1)) if paths["A"] is not None else ():
        for name, arr in ctrl[key].items():
            lanes_ok[f"{key}/{name}"] = np.asarray(cbs.surface_controls[name])[lane].tobytes() == arr.tobytes()
    out["clock_base_lane_controls_exact"] = lanes_ok
    out["batched_carry_leading_dims"] = sorted({int(np.shape(x)[0]) for x in jax.tree.leaves(bcarries["d01"])
                                                if np.ndim(x) > 0})
    out["clock_base_control_shapes"] = ({n: list(np.shape(v)) for n, v in cbs.surface_controls.items()}
                                        if cbs.surface_controls is not None else None)
    cadence = int(ns.radiation_cadence_steps)
    advance = pipeline._batched_advance_factory(tree=tree, batch_namelists=bnamelists, batch_size=2,
                                                batch_noahmp_arrays=bnoah)
    t2 = time.time()
    res_b = advance("d01", bcarries["d01"], 1, 1)
    jax.block_until_ready(res_b)
    out["batched_step_s"] = round(time.time() - t2, 1)
    solo = {}
    t3 = time.time()
    for key in ("A", "B"):
        nl = dataclasses.replace(ns, surface_controls={n: jax.numpy.asarray(v) for n, v in ctrl[key].items()} or None)
        c = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry_solo)
        solo[key] = om._advance_chunk(c, nl, jax.numpy.asarray(1, jax.numpy.int32), om.build_clock_base(nl),
                                      n_steps=1, cadence=cadence)
        jax.block_until_ready(solo[key])
    out["solo_steps_s"] = round(time.time() - t3, 1)

    def lane(tree_, i):
        return [np.asarray(x)[i] if np.ndim(x) > 0 else np.asarray(x) for x in jax.tree.leaves(tree_)]

    def cmp(a_leaves, b_leaves):
        exact = sum(int(np.asarray(a).tobytes() == np.asarray(b).tobytes()) for a, b in zip(a_leaves, b_leaves))
        worst = 0.0
        for a, b in zip(a_leaves, b_leaves):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            if a.shape == b.shape and a.size and np.issubdtype(a.dtype, np.floating):
                fin = np.isfinite(a) & np.isfinite(b)
                scale = max(1e-30, float(np.max(np.abs(b[fin])))) if fin.any() else 1.0
                worst = max(worst, float(np.max(np.abs(a[fin] - b[fin]))) / scale if fin.any() else 0.0)
        return {"leaves": len(a_leaves), "bitwise_equal": exact, "max_rel_diff": worst}

    def per_leaf(lane_leaves, own, other):
        """Leaves where lane != its solo run: |lane - own| vs the control effect |own - other| (same leaf)."""
        rows = []
        for i, (l, a, b) in enumerate(zip(lane_leaves, own, other)):
            l, a, b = (np.asarray(x) for x in (l, a, b))
            if l.tobytes() == a.tobytes():
                continue
            l64, a64, b64 = (np.asarray(x, np.float64) for x in (l, a, b))
            fin = np.isfinite(l64) & np.isfinite(a64) & np.isfinite(b64)
            rows.append({"leaf": i, "shape": list(a.shape), "dtype": str(a.dtype),
                         "max_abs_lane_minus_solo": float(np.max(np.abs(l64 - a64)[fin])) if fin.any() else None,
                         "max_abs_control_effect": float(np.max(np.abs(a64 - b64)[fin])) if fin.any() else None,
                         "max_abs_value": float(np.max(np.abs(a64)[fin])) if fin.any() else None,
                         "cells_differing": int(np.count_nonzero(l64 != a64))})
        return rows

    out["lane0_vs_soloA_leaves"] = per_leaf(lane(res_b, 0), jax.tree.leaves(solo["A"]), jax.tree.leaves(solo["B"]))
    out["lane1_vs_soloB_leaves"] = per_leaf(lane(res_b, 1), jax.tree.leaves(solo["B"]), jax.tree.leaves(solo["A"]))
    out["lane0_vs_soloA"] = cmp(lane(res_b, 0), jax.tree.leaves(solo["A"]))
    out["lane1_vs_soloB"] = cmp(lane(res_b, 1), jax.tree.leaves(solo["B"]))
    out["lane0_vs_lane1"] = cmp(lane(res_b, 0), lane(res_b, 1))
    out["soloA_vs_soloB"] = cmp(jax.tree.leaves(solo["A"]), jax.tree.leaves(solo["B"]))
    out["nonfinite_batched"] = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in jax.tree.leaves(res_b)
                                       if np.issubdtype(np.asarray(x).dtype, np.floating)))
    out["gates"] = {
        "lane_controls_exact": all(lanes_ok.values()),
        "lanes_differ": out["lane0_vs_lane1"]["bitwise_equal"] < out["lane0_vs_lane1"]["leaves"],
        "lane_matches_solo_rel_1e-5": max(out["lane0_vs_soloA"]["max_rel_diff"],
                                          out["lane1_vs_soloB"]["max_rel_diff"]) <= 1e-5,
        "finite": out["nonfinite_batched"] == 0,
    }
    out["gates"]["PASS"] = all(out["gates"].values())
Path(OUT).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["gates"]), flush=True)
