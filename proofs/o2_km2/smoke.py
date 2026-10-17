"""CPU coupled km_opt=2 smoke on real Swiss d01 (42x42x44, dt18).

usage: smoke.py OUT.json [STEPS=3] [BL=5] [ARMS=km2]
Select GPUWRF_FAST_DEFAULTS=1 O1_PALLAS_INTERPRET=1 GPUWRF_CENSUS=1 for
release REAL32 carry. Pallas executes its real kernel bodies in interpret mode;
PTX REAL rounding is CPU-emulated, so this is not GPU binary qualification.
The carry is loaded with the release configuration; km2 uses its separate
zero-initialized LES TKE, the same cold-start seeding tested in the product.
"""
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o1_smag3d'))
import cpu_safe  # noqa: E402

cpu_safe.install()
import os  # noqa: E402

if os.environ.get("O1_PALLAS_INTERPRET") == "1":
    # Release fast defaults (GPUWRF_FAST_DEFAULTS=1, REAL32 carry) select Triton Pallas kernels that cannot
    # execute on CPU (E156): run EVERY pallas_call in interpret mode (same kernel bodies, CPU semantics).
    from jax.experimental import pallas as _pl

    _real_pallas_call = _pl.pallas_call

    def _interpret_pallas_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pallas_call(*a, **k)

    _pl.pallas_call = _interpret_pallas_call
    # The release REAL kernels also pin REAL4 rounding with PTX inline asm ("<op>.rn.f32"/"<op>.<mode>.f64"),
    # which has no CPU lowering. CPU-execution shim (NOT the product instruction): exact op in f64, then
    # round to the result precision (the same emulation as boundary_apply's own CPU prescreen).
    import jax.experimental.pallas.triton as _pt

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op, mode, ptx = asm.split()[0].split(".")[:3]
        a, b = (jnp.asarray(x, jnp.float64) for x in args)
        exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if jnp.dtype(out_dtype) == jnp.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
OUT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BL = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/o2-km2") as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns, carry0 = bundles["d01"].namelist, carries["d01"]
report = {"steps": STEPS, "bl_pbl_physics": BL, "dt": float(ns.dt_s),
          "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "pallas_interpret": os.environ.get("O1_PALLAS_INTERPRET"),
          "use_flux_advection": bool(ns.use_flux_advection), "theta_dtype": str(carry0.state.theta.dtype), "u_dtype": str(carry0.state.u.dtype)}
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["km2"]
arms = {k: v for k, v in {"km2": dataclasses.replace(ns, diff_opt=2, km_opt=2, bl_pbl_physics=BL),
                          "km4_release": ns}.items() if k in ARMS}
finals = {}
for name, nl in arms.items():
    cb = om.build_clock_base(nl)
    # Own copy per arm: release defaults DONATE the carry (GPUWRF_CARRY_DONATE=1).
    carry = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry0)
    if name == "km2":
        carry = carry.replace(state=carry.state.replace(tke=jnp.zeros_like(carry.state.theta, dtype=jnp.float32)))
    t0 = time.time()
    # Production entry _advance_chunk (unalias for donation + _advance_chunk_fori loop), one step per call.
    for i in range(STEPS):
        carry = om._advance_chunk(carry, nl, i + 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(carry)
    leaves = jax.tree.leaves(carry)
    nonfinite = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in leaves
                        if np.issubdtype(np.asarray(x).dtype, np.floating)))
    finals[name] = carry
    report[name] = {"wall_s": round(time.time() - t0, 1), "nonfinite_values": nonfinite, "leaves": len(leaves)}
    if getattr(carry, "census", None) is not None:
        from gpuwrf.diagnostics.census import EVENTS, GUARDS

        g = np.asarray(carry.census.guards)
        report[name]["guards_fired"] = {GUARDS[r]: dict(zip(EVENTS, map(int, g[r]))) for r in range(g.shape[0]) if g[r].any()}
        report[name]["guard_total"] = int(g.sum())
    print(name, report[name], flush=True)
    Path(OUT).write_text(json.dumps(report, indent=1))

assert report["km2"]["nonfinite_values"] == 0, report
assert report["km2"].get("guard_total", 0) == 0, report
tke = np.asarray(carry.state.tke)
report["km2"].update(tke_min=float(tke.min()), tke_max=float(tke.max()), tke_active=int(np.count_nonzero(tke)), tke_dtype=str(tke.dtype))
assert np.isfinite(tke).all() and tke.max() > 0
Path(OUT).write_text(json.dumps(report, indent=1))
