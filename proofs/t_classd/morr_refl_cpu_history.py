"""Recompute Morrison REFL_10CM from an original CPU-WRF do_radar_ref=1 history and compare.

usage: JAX_PLATFORMS=cpu python proofs/t_classd/morr_refl_cpu_history.py <run_dir> <out.json>
WRF writes REFL_10CM = MAX(-35, refl10cm_hm(MP-exit qv,qr,nr,qs,ns,qg,ng,T1D,P)) on history-alarm
steps; history holds the same MP-exit species and P. T1D is reconstructed as (T+300)*(P/1e5)**(R/cp)
(history keeps TH = T1D/PII), so ~1 REAL ulp temperature differences are expected; the specified
zone (ring 0, spec_zone=1) is never computed by WRF microphysics and must stay at its initial value.
"""
import glob, hashlib, json, sys
import numpy as np, netCDF4, jax
jax.config.update('jax_enable_x64', True)
from gpuwrf.diagnostics.morrison_radar import morrison_refl_10cm

run, out = sys.argv[1], sys.argv[2]
fn = jax.jit(lambda *a: morrison_refl_10cm(*a))
frames = []
for path in sorted(glob.glob(f'{run}/wrfout_d01_*')):
    with netCDF4.Dataset(path) as ds:
        g = lambda n: np.asarray(ds[n][0], np.float32)
        p = g('P') + g('PB')
        t = ((g('T') + np.float32(300.)) * (p / np.float32(1e5)) ** np.float32(287. / 1004.5)).astype(np.float32)
        cols = [np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]) for a in
                (g('QVAPOR'), g('QRAIN'), g('QNRAIN'), g('QSNOW'), g('QNSNOW'), g('QGRAUP'), g('QNGRAUPEL'), t, p)]
        cpu = g('REFL_10CM')
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    nz, ny, nx = cpu.shape
    got = np.moveaxis(np.asarray(fn(*cols)).reshape(ny, nx, nz), -1, 0)
    inner = (slice(None), slice(1, -1), slice(1, -1))
    d = np.abs(got[inner].astype(np.float64) - cpu[inner])
    ring = cpu.copy(); ring[inner] = 0
    live = cpu[inner] > -35
    frames.append(dict(file=path.split('/')[-1], sha256=sha, cpu_live=int(live.sum()), cpu_max=float(cpu.max()),
                       max_abs_dbz=float(d.max()), p999_abs_dbz=float(np.quantile(d, .999)),
                       exact=int((got[inner] == cpu[inner]).sum()), n=int(d.size), ring_values=sorted({float(x) for x in np.unique(ring[ring != 0])})[:4]))
summary = dict(run=run, frames=len(frames), worst_max_abs_dbz=max(f['max_abs_dbz'] for f in frames[1:]),
               frame0_cpu_max=frames[0]['cpu_max'], total_live=sum(f['cpu_live'] for f in frames),
               exact_fraction=sum(f['exact'] for f in frames[1:]) / sum(f['n'] for f in frames[1:]))
json.dump(dict(summary=summary, frames=frames), open(out, 'w'), indent=1)
print(json.dumps(summary))
