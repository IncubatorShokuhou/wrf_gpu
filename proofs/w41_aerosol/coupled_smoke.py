"""AE06 CPU coupled smoke: aer_opt=0 vs 1 production steps on the real Swiss case (RD11, 42x42x44, single domain).

usage: JAX_PLATFORMS=cpu python coupled_smoke.py SRC_ROOT OUT.json STEPS [ARM ...]
ARM = 0 | 1 (mp8, aer_opt 0/1) | 28 | 3 (mp28 with aer_opt 0/3); default 0 1 28 3.
Each arm loads a private copy of RD11 whose namelist.input differs only by ``aer_opt`` (and mp_physics) through the real CLI
loader (nested_pipeline._load_domains) and run STEPS production steps (_advance_chunk_fori, radiation at step 1,
WRF itimestep=1).  The radiation clock is shifted by +12 h in BOTH arms (E185) so the 00Z January case has sun.
Recorded: bound aer_opt + static, non-finite values, census guards, step-1 held SW/LW, the step-1 held
SWDOWN of the aer_opt=1 arm versus the coupler's radiation diagnostics rebuilt from the initial carry with the
same Tegen static (real-call path), and the LW identity at step 1 (aerosol enters SW only).
Pallas runs in interpret mode (release defaults on CPU, E156).
"""
import json
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

SRC, OUT, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
ARMS = [int(a) for a in sys.argv[4:]] or [0, 1, 28, 3]
SETUP = {0: (8, 0), 1: (8, 1), 28: (28, 0), 3: (28, 3)}  # arm -> (mp_physics, aer_opt)
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
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-aerosol")
_real_pallas_call = pl.pallas_call
_real_clock = om.build_clock_base


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


def _noon_clock(*a, **k):
    cb = _real_clock(*a, **k)
    return cb._replace(rad_minute=cb.rad_minute + 720.0)


pl.pallas_call = _interpret_pallas_call
om.build_clock_base = _noon_clock
out = {"src": SRC, "case": str(CASE), "steps": STEPS, "clock_shift_min": 720, "arms": {}}
held1 = {}
for arm in ARMS:
    mp, aer = SETUP[arm]
    rec = {"mp_physics_requested": mp, "aer_opt_requested": aer}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(dir=LANE) as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            for f in CASE.iterdir():
                if f.name != "namelist.input":
                    (case / f.name).symlink_to(f)
            text = (CASE / "namelist.input").read_text()
            assert "aer_opt" not in text
            text = re.sub(r"^(\s*&physics\s*)$", r"\1\n aer_opt = %d," % aer, text, count=1, flags=re.M | re.I)
            text, n = re.subn(r"^(\s*mp_physics\s*=\s*)8,", r"\g<1>%d," % mp, text, flags=re.M)
            assert n == 1
            (case / "namelist.input").write_text(text)
            cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
            _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
        ns, carry = bundles["d01"].namelist, carries["d01"]
        rec["aer_opt_bound"] = int(ns.aer_opt)
        rec["mp_physics_bound"] = int(ns.mp_physics)
        rec["carry_taod5502d"] = carry.taod5502d is not None
        rec["static"] = None if ns.tegen_aerosol is None else list(np.asarray(ns.tegen_aerosol.aerodm).shape)
        rec["carry_aerod"] = carry.aerod is not None
        rec["load_s"] = round(time.time() - t0, 1)
        om._resolve_operational_suite(ns)
        jax.clear_caches()
        cb = om.build_clock_base(ns)
        carry = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, carry)
        c = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, carry)
        t2 = time.time()
        for i in range(STEPS):
            c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
            jax.block_until_ready(c)
            if i == 0:
                held = c.radiation_diagnostics
                held1[arm] = {k: np.asarray(getattr(held, k)) for k in ("swdown", "swup", "sw_toa_up", "glw",
                                                                        "lw_toa_up", "coszen")}
                if c.taod5502d is not None:
                    rec["step1_taod5502d"] = {"min": float(np.asarray(c.taod5502d).min()),
                                              "median": float(np.median(np.asarray(c.taod5502d))),
                                              "max": float(np.asarray(c.taod5502d).max())}
                rec["step1"] = {k: float(v.astype(np.float64).mean()) for k, v in held1[arm].items()}
                if aer in (1, 3):
                    rebuilt = pc.rrtmg_radiation_diagnostics(
                        carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                        clock_base=om._rad_clock_base(cb), radiation_static=ns.radiation_static,
                        land_state=carry.noahmp_land, use_mp_re=int(om.mp_re_active(ns)), mp_physics=int(ns.mp_physics),
                        **({"tegen_aerosol": ns.tegen_aerosol} if aer == 1 else {"thompson_aerosol": True}),
                        **om._rrtmg_pbl_cloud_kwargs(ns))
                    a = np.asarray(rebuilt.swdown, np.float64)
                    b = held1[arm]["swdown"].astype(np.float64)
                    rec["real_call_path_swdown"] = {"cells": int(a.size), "equal": int((a == b).sum()),
                                                    "max_abs": float(np.abs(a - b).max())}
        rec["exec_s"] = round(time.time() - t2, 1)
        leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
        rec["nonfinite_values"] = int(sum(int((~np.isfinite(x)).sum()) for x in leaves
                                          if np.issubdtype(x.dtype, np.floating)))
        census = getattr(c, "census", None)
        if census is not None:
            rec["guard_total"] = int(np.asarray(census.guards).sum())
        rec["delta"] = {f: float(np.abs(np.asarray(getattr(c.state, f), np.float64)
                                        - np.asarray(getattr(carry.state, f), np.float64)).max())
                        for f in ("theta", "qv", "u", "v")}
    except Exception as exc:  # noqa: BLE001 - recorded result
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["traceback_tail"] = traceback.format_exc()[-2500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    out["arms"][str(arm)] = rec
    Path(OUT).write_text(json.dumps(out, indent=1, default=str))
    print(arm, {k: v for k, v in rec.items() if k != "traceback_tail"}, flush=True)
for on, off in ((1, 0), (3, 28)):
    if on in held1 and off in held1:
        d = {k: float(np.abs(held1[on][k].astype(np.float64) - held1[off][k].astype(np.float64)).max()) for k in held1[off]}
        sun = held1[off]["coszen"] > 0.05
        out[f"arm{on}_minus_arm{off}_step1"] = {
            "max_abs": d, "sunlit_cells": int(sun.sum()),
            "swdown_mean_reduction_sunlit": float((held1[off]["swdown"] - held1[on]["swdown"])[sun].mean()) if sun.any() else None}
        Path(OUT).write_text(json.dumps(out, indent=1, default=str))
        print(on, off, out[f"arm{on}_minus_arm{off}_step1"], flush=True)
