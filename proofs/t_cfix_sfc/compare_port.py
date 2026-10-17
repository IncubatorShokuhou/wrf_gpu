"""Port selected-surface column_call vs pristine WRF oracle (lane t-cfix-sfc).

usage: JAX_PLATFORMS=cpu python compare_port.py <lane_dir> [--chain-port]
Runs gpuwrf.coupling.sfclay_selected.column_call on the oracle inputs (call 1) and
on call-2 inputs (default: the ORACLE's own call-1 memory, so each call is judged on
identical operands; --chain-port feeds the port's own call-1 memory instead).
Prints per scheme/field: max REAL4 ULP distance to the zero build, and the ULP
distance between the zero and fma pristine builds (WRF's own optimiser spread).
"""
from pathlib import Path
import json, sys
import numpy as np
import jax.numpy as jnp

LANE = Path(sys.argv[1])
CHAIN_PORT = '--chain-port' in sys.argv
from gpuwrf.coupling.sfclay_selected import column_call

d = np.load(LANE / 'oracle' / 'oracle.npz')
env = np.load(LANE / 'oracle' / 'envelope.npz')
SLACK = 2  # REAL4 output ULPs around WRF's own +-1 ULP input envelope
meta = json.loads((LANE / 'oracle' / 'oracle.json').read_text())
IN, OUT = meta['inputs'], meta['outputs']


def ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a); b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


def port(scheme, inp, itimestep):
    inputs = {k: jnp.asarray(inp[:, IN.index(k)]) for k in IN if k not in ('zol', 'regime')}
    return {k: np.asarray(v) for k, v in column_call(scheme, inputs, itimestep).items()}


def main():
    report = {}
    for scheme in (91, 7, 3):
        inp1 = d['inputs']
        p1 = port(scheme, inp1, 1)
        inp2 = d[f'sf{scheme}_zero_inp2'].copy()
        if CHAIN_PORT:
            for k in ('ust', 'znt', 'mol', 'qsfc', 'hfx', 'qfx'):
                if k in p1 and not (scheme == 3 and k not in ('ust', 'znt')):
                    inp2[:, IN.index(k)] = p1[k]
        p2 = port(scheme, inp2, 2)
        for call, p in ((1, p1), (2, p2)):
            zero = d[f'sf{scheme}_zero_out{call}']; fma = d[f'sf{scheme}_fma_out{call}']
            for k, val in p.items():
                ref = zero[:, OUT.index(k)]
                u = ulp(val, ref); spread = ulp(fma[:, OUT.index(k)], ref)
                worst = int(np.argmax(u))
                lo = env[f'sf{scheme}_lower{call}'][:, OUT.index(k)]; hi = env[f'sf{scheme}_upper{call}'][:, OUT.index(k)]
                below = ulp(np.minimum(val, lo), lo); above = ulp(np.maximum(val, hi), hi)
                outside = np.maximum(below, above)
                if k == 'zol' and scheme == 91:
                    # SFCLAY keeps the entry ZOL in regime 1; the port does not hold ZOL (no consumer).
                    outside = np.where(zero[:, OUT.index('regime')] == 1., 0, outside)
                report[f'sf{scheme}/call{call}/{k}'] = dict(max_ulp=int(u.max()), n_diff=int((u > 0).sum()),
                    outside_envelope=int((outside > SLACK).sum()), max_outside_ulp=int(outside.max()),
                    fma_max_ulp=int(spread.max()), worst_row=worst, port=float(val[worst]), wrf=float(ref[worst]),
                    finite=bool(np.isfinite(val).all()))
    bad = {k: v for k, v in report.items() if v['max_ulp'] > 0}
    for k, v in sorted(bad.items(), key=lambda kv: -kv[1]['max_ulp']):
        print(f"{k:28s} ulp {v['max_ulp']:>10d} n {v['n_diff']:3d} fma {v['fma_max_ulp']:>8d} "
              f"out {v['outside_envelope']:3d}/{v['max_outside_ulp']:>8d} "
              f"row {v['worst_row']:3d} port {v['port']:.8g} wrf {v['wrf']:.8g}")
    print('fields BYTE:', len(report) - len(bad), 'of', len(report), '; cells outside envelope+slack:',
          sum(v['outside_envelope'] for v in report.values()))
    (LANE / ('compare_chain.json' if CHAIN_PORT else 'compare.json')).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
