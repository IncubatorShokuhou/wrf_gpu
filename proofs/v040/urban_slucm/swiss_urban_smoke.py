"""SLUCM coupled CPU smoke through the REAL CLI loader (nested_pipeline._load_domains).

usage: JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python swiss_urban_smoke.py <src_root> <case_dir> <out.json> <nsteps>
Case = RD11 Swiss 42x42 (release namelist) with a 4x4 land block relabelled ISURBAN (RD11 itself has no urban
cell, a SLUCM smoke on it would be vacuous).  Two arms in one process: sf_urban_physics=0 and =1.  Executes
<nsteps> production steps (operational_mode._advance_chunk_fori) and records finiteness, census guards and the
urban-cell surface fields / urban carry.  Release Triton kernels cannot run on CPU (E156): run with
GPUWRF_FAST_DEFAULTS=0 (legacy carry) or O1-style interpret mode.
"""
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

SRC, CASE, OUT, NSTEPS = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/o2-urban/probe")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
if os.environ.get("O1_PALLAS_INTERPRET") == "1":
    # Release fast defaults (REAL32 carry) select Triton Pallas kernels that cannot execute on CPU (E156): run every
    # pallas_call in interpret mode, and emulate the PTX "<op>.rn.f32" inline asm (exact f64 op rounded to REAL) --
    # the CPU shim of proofs/o2_km2/smoke.py. Not GPU binary qualification.
    import jax.experimental.pallas.triton as _pt
    import jax.numpy as _jnp
    from jax.experimental import pallas as _pl

    _real_pallas_call = _pl.pallas_call

    def _interpret_pallas_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pallas_call(*a, **k)

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op = asm.split()[0].split(".")[0]
        x, y = (_jnp.asarray(v, _jnp.float64) for v in args)
        exact = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if _jnp.dtype(out_dtype) == _jnp.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pl.pallas_call = _interpret_pallas_call
    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o2-urban/probe")


def run_arm(urban: int):
    with tempfile.TemporaryDirectory(dir=LANE) as tmp:
        case = Path(tmp) / "case"
        case.mkdir()
        for f in CASE.iterdir():
            if f.name != "namelist.input":
                (case / f.name).symlink_to(f.resolve())
        text = (CASE / "namelist.input").read_text()
        text = re.sub(r"^\s*sf_urban_physics\s*=.*$", f" sf_urban_physics = {urban},", text, flags=re.M | re.I)
        (case / "namelist.input").write_text(text)
        cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
        _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    ns, carry = bundles["d01"].namelist, carries["d01"]
    om._resolve_operational_suite(ns)
    cb = om.build_clock_base(ns)
    # Own copies: release defaults DONATE the carry (GPUWRF_CARRY_DONATE=1); keep the seed for the report.
    carry = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry)
    c = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry)
    for i in range(NSTEPS):
        # Production entry (unalias for donation + _advance_chunk_fori), one step per call.
        c = om._advance_chunk(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
        jax.block_until_ready(c)
    leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
    nonfinite = int(sum(int((~np.isfinite(x)).sum()) for x in leaves if np.issubdtype(x.dtype, np.floating)))
    census = getattr(c, "census", None)
    rec = {"urban": urban, "nonfinite": nonfinite, "theta_dtype": str(c.state.theta.dtype),
           "guards": None if census is None else int(np.asarray(census.guards).sum()),
           "sf_urban_physics_bound": int(ns.sf_urban_physics), "bundle": ns.urban_bundle is not None,
           "noahmp_ivgtyp_urban_block": np.asarray(ns.noahmp_static.ivgtyp)[18:22, 18:22].tolist()}
    s = c.state
    blk = (slice(18, 22), slice(18, 22))
    for f in ("t_skin", "hfx", "qfx", "ustar", "qsfc"):
        v = getattr(s, f, None)
        if v is not None:
            rec[f + "_urban_block_mean"] = float(np.asarray(v, np.float64)[blk].mean())
            rec[f + "_outside_mean"] = float(np.asarray(v, np.float64)[:10, :10].mean())
    rec["albedo_urban_block_mean"] = float(np.asarray(c.noahmp_land.albedo, np.float64)[blk].mean())
    if c.urban_slucm is not None:
        u0, u1 = carry.urban_slucm.state, c.urban_slucm.state
        rec["urban_carry"] = {k: {"init_block": float(np.asarray(getattr(u0, k), np.float64)[blk].mean()),
                                  "final_block": float(np.asarray(getattr(u1, k), np.float64)[blk].mean())}
                              for k in ("tr", "tb", "tg", "tc", "qc", "uc", "chr", "chc")}
        rec["frc_block"] = float(np.asarray(c.urban_slucm.frc_urb)[blk].mean())
        rec["utype_count"] = int((np.asarray(c.urban_slucm.utype) > 0).sum())
    return rec, c


out = {"src": SRC, "case": str(CASE), "nsteps": NSTEPS,
       "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"), "arms": {}}
finals = {}
for urban in (0, 1):
    t0 = time.time()
    rec, c = run_arm(urban)
    rec["wall_s"] = round(time.time() - t0, 1)
    out["arms"][str(urban)] = rec
    finals[urban] = c
    print(json.dumps(rec)[:1500], flush=True)
    Path(OUT).write_text(json.dumps(out, indent=1))
s0, s1 = finals[0].state, finals[1].state
mask = np.zeros((42, 42), bool)
mask[18:22, 18:22] = True
diff = np.abs(np.asarray(s1.t_skin, np.float64) - np.asarray(s0.t_skin, np.float64))
out["tsk_on_minus_off"] = {"max_in_block": float(diff[mask].max()), "max_outside_block_ring3": float(
    diff[:15, :15].max())}
Path(OUT).write_text(json.dumps(out, indent=1))
print(json.dumps(out["tsk_on_minus_off"]))
