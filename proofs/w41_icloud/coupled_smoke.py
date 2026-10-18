"""CPU coupled smoke + real-call-path check for icloud=1/2/3 on the real Swiss case (RD11, 42x42x44).

usage: JAX_PLATFORMS=cpu python coupled_smoke.py SRC_ROOT OUT.json STEPS [ARM ...]   (ARM = 1 | 2 | 3)
Each arm edits &physics icloud in a private copy of RD11's namelist.input and loads it through the
real CLI loader (nested_pipeline._load_domains). Recorded: bound icloud, location-stripped jaxpr sha
of the production step loop, STEPS executed production steps (radiation fires at step 1, WRF
itimestep=1): non-finite count, census guards, held radiation CLDFRA statistics and fluxes, and for
icloud=2/3 the held CLDFRA versus the coupler's RRTMG cloud input rebuilt from the step-1 radiation
input state (initial carry) -- the end-to-end real-call-path check.
Pallas runs in interpret mode for load, trace and execution (release defaults on CPU, E156).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

SRC, OUT, STEPS, ARMS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.coupling import physics_couplers as pc  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-icloud")
_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


def canon(jaxpr) -> str:
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", str(jaxpr))
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


def stats(a):
    a = np.asarray(a, np.float64)
    return {"mean": float(a.mean()), "max": float(a.max()), "min": float(a.min()),
            "frac_pos": float((a > 0).mean()), "frac_partial": float(((a > 0) & (a < 1)).mean())}


pl.pallas_call = _interpret_pallas_call
out = {"src": SRC, "case": str(CASE), "steps": STEPS, "arms": {}}
for arm in ARMS:
    rec = {"icloud_requested": int(arm)}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(dir=LANE) as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            for f in CASE.iterdir():
                if f.name != "namelist.input":
                    (case / f.name).symlink_to(f)
            text = (CASE / "namelist.input").read_text()
            text = re.sub(r"^\s*icloud\s*=.*$", f" icloud = {int(arm)},", text, flags=re.M | re.I)
            (case / "namelist.input").write_text(text)
            cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
            _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
        ns, carry = bundles["d01"].namelist, carries["d01"]
        rec["icloud_bound"] = int(ns.icloud)
        rec["load_s"] = round(time.time() - t0, 1)
        om._resolve_operational_suite(ns)
        jax.clear_caches()
        cb = om.build_clock_base(ns)
        jx = jax.make_jaxpr(lambda c, nl=ns, cb=cb: om._advance_chunk_fori(
            c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
        rec["sha"] = hashlib.sha256(canon(jx).encode()).hexdigest()
        # The loaded carry shares some buffers between leaves; the production step donates its
        # carry, so execute on per-leaf copies (distinct buffers, identical values).
        carry = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, carry)
        c = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, carry)
        t2 = time.time()
        for i in range(STEPS):
            c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
            jax.block_until_ready(c)
            if i == 0:
                held = c.radiation_diagnostics
                rec["step1_cldfra"] = stats(held.cloud_fraction)
                rec["step1_fluxes"] = {k: float(np.asarray(getattr(held, k), np.float64).mean())
                                       for k in ("glw", "swdown", "lw_toa_up") if getattr(held, k, None) is not None}
                if int(ns.icloud) != 1:
                    sw, *_ = pc._rrtmg_column_inputs(
                        carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0, clock_base=om._rad_clock_base(cb)
                        if hasattr(om, "_rad_clock_base") else None, radiation_static=ns.radiation_static,
                        land_state=carry.noahmp_land, use_mp_re=int(om.mp_re_active(ns)),
                        mp_physics=int(ns.mp_physics), **om._rrtmg_pbl_cloud_kwargs(ns))
                    rebuilt = np.moveaxis(np.asarray(sw.cloud_fraction, np.float32), -1, 0)
                    heldf = np.asarray(held.cloud_fraction, np.float32)
                    rec["real_call_path"] = {"cells": int(heldf.size), "equal": int((rebuilt == heldf).sum()),
                                             "max_abs": float(np.abs(rebuilt.astype(np.float64) - heldf).max())}
        rec["exec_s"] = round(time.time() - t2, 1)
        leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
        rec["nonfinite_values"] = int(sum(int((~np.isfinite(x)).sum()) for x in leaves
                                          if np.issubdtype(x.dtype, np.floating)))
        census = getattr(c, "census", None)
        if census is not None:
            rec["guard_total"] = int(np.asarray(census.guards).sum())
        rec["final_cldfra"] = stats(c.radiation_diagnostics.cloud_fraction)
        rec["delta"] = {f: float(np.abs(np.asarray(getattr(c.state, f), np.float64)
                                        - np.asarray(getattr(carry.state, f), np.float64)).max())
                        for f in ("theta", "qv", "qc", "qi")}
    except Exception as exc:  # noqa: BLE001 - recorded result
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["traceback_tail"] = traceback.format_exc()[-2000:]
    rec["wall_s"] = round(time.time() - t0, 1)
    out["arms"][arm] = rec
    Path(OUT).write_text(json.dumps(out, indent=1, default=str))
    print(arm, {k: v for k, v in rec.items() if k != "traceback_tail"}, flush=True)
