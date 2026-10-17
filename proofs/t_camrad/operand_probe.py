"""Measure changed CAM operand assembly on real c28 initial state, CPU only.

This is a same-state caller/resource probe, not a forecast or CUDA speed claim.
Both RRTMG entry flags are held on to use the production REAL operand builder;
no spectral solver is executed by the timed function.
"""
from pathlib import Path
import json
import os
import resource
import statistics
import subprocess
import sys
import time

root, case, output = (Path(s).resolve() for s in sys.argv[1:4])
sys.path.insert(0, str(root / "scripts/v025"))
import cpu_guard  # noqa: E402,F401
sys.path.insert(0, str(root / "src"))
import gpuwrf  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gpuwrf.coupling import cam_radiation as cam  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert os.environ["JAX_PLATFORMS"] == "cpu" and jax.devices()[0].platform == "cpu"
t0 = time.perf_counter()
cfg = NestedPipelineConfig(case, output.parent / "probe-output", output.parent / "probe-proof", hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
nml, state = bundles["d01"].namelist, carries["d01"].state
assert nml.bl_pbl_physics == 9 and nml.ra_lw_physics == 3 and nml.ra_sw_physics == 3
assert cam.pc._rrtmg_real_enabled()
receipt = dict(source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               platform="cpu", cache="NONE", case=str(case), shape=list(state.theta.shape),
               load_s=time.perf_counter()-t0, scope=__doc__, arms={})
for name, policy in (("old_mynn_default", {}), ("wrf_non_mynn", om._rrtmg_pbl_cloud_kwargs(nml))):
    fn = jax.jit(lambda s: cam.cam_operands(s, nml.grid, time_utc=nml.time_utc, lead_seconds=jnp.float32(3600.),
                                          radiation_static=nml.radiation_static, land_state=carries["d01"].noahmp_land,
                                          **policy))
    start = time.perf_counter()
    compiled = fn.lower(state).compile()
    compile_s = time.perf_counter()-start
    out = compiled(state)
    jax.block_until_ready(out)
    samples = []
    for _ in range(30):
        start = time.perf_counter()
        out = compiled(state)
        jax.block_until_ready(out)
        samples.append(time.perf_counter()-start)
    cf = np.asarray(out.cldfra)
    assert np.isfinite(cf).all()
    mem = compiled.memory_analysis()
    hlo = compiled.as_text()
    receipt["arms"][name] = dict(compile_s=compile_s, warm_median_s=statistics.median(samples),
        warm_min_s=min(samples), samples_s=samples, cloud_nonzero=int(np.count_nonzero(cf)),
        cloud_max=float(cf.max()), hlo_chars=len(hlo),
        memory_bytes={k:getattr(mem,k) for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")})
receipt["maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
receipt["elapsed_s"] = time.perf_counter()-t0
output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
print(json.dumps({k:{n:v for n,v in a.items() if n != "samples_s"} for k,a in receipt["arms"].items()}, indent=2))
