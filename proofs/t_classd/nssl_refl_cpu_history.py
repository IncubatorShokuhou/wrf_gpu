"""Recompute legacy NSSL one-moment REFL_10CM from original CPU-WRF history state; compare to CPU REFL_10CM.

Usage: JAX_PLATFORMS=cpu python proofs/t_classd/nssl_refl_cpu_history.py <case_dir> <variant> <out.json> [dry|moist]
<case_dir>/cpu/wrfout_d01_* are original WRF 4.7.1 outputs (READ-ONLY). WRF's alt is not written:
the dry density 1/alt is recovered by inverting the moist EOS (p, T, QVAPOR). ``dry`` = WRF
microphysics RHO (moist_physics_prep_em, module_big_step_utilities_em.F:5542); ``moist`` =
phy_prep (1+qv)/alt (:4856), the convention of the current port NSSL adapter.
"""
import glob, hashlib, json, sys
import numpy as np, netCDF4, jax
jax.config.update('jax_enable_x64', True)
from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
from gpuwrf.physics.nssl2mom.diagnostics import radardd02_one_moment
from gpuwrf.physics.nssl2mom.indices import FP32, LVH, NA

case_dir, variant, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rho_mode = sys.argv[4] if len(sys.argv) > 4 else 'dry'
assert rho_mode in ('dry', 'moist')
C = get_one_moment_constants('fp32', variant)
fn = jax.jit(lambda a, d: radardd02_one_moment(a, d, C, FP32))
RD, RV, CP, P0 = 287.0, 461.6, 7.0 * 287.0 / 2.0, 1.0e5
frames = []
for path in sorted(glob.glob(f'{case_dir}/cpu/wrfout_d01_*')):
    with netCDF4.Dataset(path) as ds:
        g = lambda n: np.asarray(ds.variables[n][0], np.float64)
        p = g('P') + g('PB'); qv = g('QVAPOR'); th = g('T') + 300.0
        tk = th * (p / P0) ** (RD / CP)
        rho = p / (RD * tk * (1.0 + qv * RV / RD)) * ((1.0 + qv) if rho_mode == 'moist' else 1.0)
        an = np.zeros((NA + 1,) + p.shape, np.float32)
        an[4], an[6], an[7] = g('QRAIN'), g('QSNOW'), g('QGRAUP')
        if variant == 19:
            an[8] = g('QHAIL'); an[LVH] = (g('QVGRAUPEL') * rho).astype(np.float32)
        cpu = np.asarray(ds.variables['REFL_10CM'][0], np.float32)
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    shape = p.shape
    got = np.asarray(fn(an.reshape(NA + 1, -1), rho.astype(np.float32).reshape(-1))).reshape(shape)
    inner = (slice(None), slice(1, -1), slice(1, -1))
    d = np.abs(got[inner].astype(np.float64) - cpu[inner])
    ring = cpu.copy(); ring[inner] = 0
    frames.append(dict(file=path.split('/')[-1], sha256=sha, cpu_nonzero=int(np.count_nonzero(cpu[inner])),
                       port_nonzero=int(np.count_nonzero(got[inner])), cpu_max=float(cpu.max()),
                       max_abs_dbz=float(d.max()), p999_abs_dbz=float(np.quantile(d, 0.999)),
                       support_mismatch=int(np.count_nonzero((got[inner] > 0) != (cpu[inner] > 0))),
                       support_mismatch_above_0p01=int(np.count_nonzero(((got[inner] > 0) != (cpu[inner] > 0))
                                                                         & (np.maximum(got[inner], cpu[inner]) > 0.01))),
                       ring_nonzero=int(np.count_nonzero(ring))))
summary = dict(case_dir=case_dir, variant=variant, rho=rho_mode, frames=len(frames),
               worst_max_abs_dbz=max(f['max_abs_dbz'] for f in frames),
               total_cpu_nonzero=sum(f['cpu_nonzero'] for f in frames),
               total_support_mismatch=sum(f['support_mismatch'] for f in frames),
               total_support_mismatch_above_0p01=sum(f['support_mismatch_above_0p01'] for f in frames),
               ring_nonzero=sum(f['ring_nonzero'] for f in frames))
json.dump(dict(summary=summary, frames=frames), open(out, 'w'), indent=1)
print(json.dumps(summary))
