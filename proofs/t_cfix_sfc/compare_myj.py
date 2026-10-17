"""Port MYJSFC kernel (physics.sf_myj.myjsfc_columns with WRF memory) vs pristine oracle.

usage: JAX_PLATFORMS=cpu python compare_myj.py <lane_dir> [--chain-port]
Per chained call on the oracle's own operands (default) or the port's own memory chain.
"""
from pathlib import Path
import json, sys
import numpy as np
import jax.numpy as jnp

LANE = Path(sys.argv[1]); CHAIN = '--chain-port' in sys.argv
from gpuwrf.physics.sf_myj import myjsfc_columns

d = np.load(LANE / 'oracle' / 'oracle_myj.npz'); env = np.load(LANE / 'oracle' / 'envelope_myj.npz')
meta = json.loads((LANE / 'oracle' / 'oracle_myj.json').read_text())
PROF, SFC, OUT = meta['prof'], meta['sfc'], meta['outputs']
PORT = {'ust': 'ustar', 'lh': 'flx_lh'}
MEMORY = ('ust', 'znt', 'qsfc', 'thz0', 'qz0', 'uz0', 'vz0', 'akhs', 'akms')
SLACK = 8  # disclosed: 5-iteration SFCDIF fixed point, XLA vs glibc libm


def ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64); b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a); b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


def port(prof, sfc, itimestep):
    nz = prof.shape[1] - 1
    col = lambda k: jnp.asarray(prof[:, :nz, PROF.index(k)])
    s = lambda k: jnp.asarray(sfc[:, SFC.index(k)])
    out = myjsfc_columns(col('u'), col('v'), col('t'), col('th'), col('qv'), col('qc'), col('pmid'), col('dz'),
                         col('q2'), tsk=s('tsk'), xland=s('xland'), z0base=s('z0base'),
                         psfc=jnp.asarray(prof[:, 0, PROF.index('pint')]), znt=s('znt'), ustar=s('ust'),
                         mavail=s('mavail'), qsfc=s('qsfc'), thz0=s('thz0'), qz0=s('qz0'), uz0=s('uz0'), vz0=s('vz0'),
                         itimestep=itimestep, akhs=s('akhs'), akms=s('akms'), wrf_real=True,
                          ht=s('ht'))
    return {k: np.asarray(out[PORT.get(k, k)], np.float32) for k in OUT if PORT.get(k, k) in out}


def main():
    report = {}
    sfc = d['zero_sfc1'].copy()
    for c in (1, 2, 3):
        operands = sfc if CHAIN else d[f'zero_sfc{c}']
        p = port(d['prof'], operands, c)
        ref = d[f'zero_out{c}']; lo = env[f'lower{c}']; hi = env[f'upper{c}']
        for k, val in p.items():
            j = OUT.index(k)
            u = ulp(val, ref[:, j])
            outside = np.maximum(ulp(np.minimum(val, lo[:, j]), lo[:, j]), ulp(np.maximum(val, hi[:, j]), hi[:, j]))
            w = int(np.argmax(outside))
            report[f'call{c}/{k}'] = dict(max_ulp=int(u.max()), outside=int((outside > SLACK).sum()),
                                          max_outside=int(outside.max()), row=w, port=float(val[w]), wrf=float(ref[w, j]),
                                          label=meta['labels'][w]['surface'])
        if CHAIN:
            sfc = operands.copy()
            for k in MEMORY:
                sfc[:, SFC.index(k)] = p[k]
    for k, v in report.items():
        if v['outside']:
            print(k, v)
    print('cells outside envelope+slack:', sum(v['outside'] for v in report.values()), 'fields', len(report))
    (LANE / ('compare_myj_chain.json' if CHAIN else 'compare_myj.json')).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
