"""Port WRF-real BL_GFS call (scan_adapters.gfs_bl_real_double) vs pristine oracle (lane t-cfix-sfc).

usage: JAX_PLATFORMS=cpu python compare_blgfs.py <lane_dir> [slack_ulp]
"""
from pathlib import Path
import json, sys
import numpy as np
import jax.numpy as jnp

LANE = Path(sys.argv[1]); SLACK = int(sys.argv[2]) if len(sys.argv) > 2 else 8
from gpuwrf.coupling.scan_adapters import gfs_bl_real_double

d = np.load(LANE / 'oracle' / 'oracle_blgfs.npz')
meta = json.loads((LANE / 'oracle' / 'oracle_blgfs.json').read_text())
PROF, SFC, TEND, DIAG = meta['prof'], meta['sfc'], meta['tend'], meta['diag']


def ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64); b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a); b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


def port(prof, sfc, has_qi):
    nz = prof.shape[1] - 1
    c = lambda k: jnp.asarray(prof[:, :nz, PROF.index(k)])
    s = lambda k: jnp.asarray(sfc[:, SFC.index(k)])
    args = (c('u'), c('v'), c('t'), c('qv'), c('qc'), c('p'), c('pi'), c('dz'), c('z'))
    kw = dict(psfc=s('psfc'), ust=s('ust'), hfx=s('hfx'), qfx=s('qfx'), tsk=s('tsk'), gz1oz0=s('gz1oz0'),
              psim=s('psim'), psih=s('psih'), wspd=s('wspd'), br=s('br'), qi=c('qi') if has_qi else None)
    return {k: np.asarray(v) for k, v in gfs_bl_real_double(args, kw, float(d['dt'])).items()}


def main():
    report = {}
    for q in (1, 0):
        out = port(d['prof'], d['sfc'], q)
        pairs = [(name, out[name], d[f'q{q}_zero_tend'][..., j], d[f'q{q}_tend_lower'][..., j], d[f'q{q}_tend_upper'][..., j])
                 for j, name in enumerate(TEND) if name in out and (q or name != 'rqiblten')]
        pairs += [(name, out[key], d[f'q{q}_zero_diag'][:, j], d[f'q{q}_diag_lower'][:, j], d[f'q{q}_diag_upper'][:, j])
                  for j, (name, key) in enumerate((('ust', 'ust'), ('pbl', 'pblh'), ('wspd', None), ('kpbl', 'kpbl')))
                  if key is not None]
        for name, val, ref, lo, hi in pairs:
            # E117: XLA:CPU flushes subnormal results; WRF/GPU keep them. Compare flushed values.
            ftz = lambda a: np.where(np.abs(a) < np.finfo(np.float32).tiny, np.float32(0), a).astype(np.float32)
            val, ref, lo, hi = ftz(np.asarray(val, np.float32)), ftz(ref), ftz(lo), ftz(hi)
            outside = np.maximum(ulp(np.minimum(val, lo), lo), ulp(np.maximum(val, hi), hi))
            w = np.unravel_index(int(np.argmax(outside)), outside.shape)
            report[f'q{q}/{name}'] = dict(max_ulp=int(ulp(val, ref).max()), outside=int((outside > SLACK).sum()),
                                          max_outside=int(outside.max()), at=[int(x) for x in w],
                                          port=float(val[w]), wrf=float(ref[w]), finite=bool(np.isfinite(val).all()),
                                          dtype=str(np.asarray(out[name if name in out else 'pblh']).dtype))
    for k, v in report.items():
        print(k, v)
    print('cells outside envelope+slack:', sum(v['outside'] for v in report.values()))
    (LANE / 'compare_blgfs.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
