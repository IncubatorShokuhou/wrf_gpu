"""Check a port output set against WRF's REAL4 input-ULP envelope (lane t-shinhong).

usage: python check_envelope.py <lane_dir> <port_tag>
Rule (pre-registered in ulp_envelope.py): inside [lower, upper] widened by 2
REAL4 output ULPs. Every value must be finite. A cell is compared whenever the
reference OR the candidate is normal; only pairs where BOTH are subnormal/zero
are exempt (XLA:CPU flushes, E117) and must agree within FLT_MIN (review
rv-t-shinhong: the former AND mask skipped zeroed candidates).
"""
from pathlib import Path
import json, sys
import numpy as np

lane, tag = Path(sys.argv[1]), sys.argv[2]
d = np.load(lane / 'oracle.npz'); env = np.load(lane / 'ulp_envelope.npz')
port = np.load(lane / f'port_{tag}.npy', allow_pickle=True)
FIELDS = [('tke', 0), ('el_pbl', 1), ('exch_h', 2), ('u', 3), ('v', 4), ('theta', 5), ('qv', 6), ('qc', 7), ('qi', 8)]
tiny = np.finfo(np.float32).tiny
res = {name: dict(cells=0, outside=0, byte=0, worst_excess_ulp=0.0) for name, _ in FIELDS}
worst = []
for i in env['rows']:
    n = int(d['nz'][i])
    for name, j in FIELDS:
        lo = env['lower'][i][:n, j].astype(np.float32); hi = env['upper'][i][:n, j].astype(np.float32)
        act = np.asarray(port[i][name][:n], np.float32); ref = d['out_zero'][i][:n, j]
        lo2 = np.nextafter(np.nextafter(lo, np.float32(-np.inf)), np.float32(-np.inf))
        hi2 = np.nextafter(np.nextafter(hi, np.float32(np.inf)), np.float32(np.inf))
        for arr in (act, ref, lo, hi):
            assert np.isfinite(arr).all(), (int(i), name)
        normal = (np.abs(ref) >= tiny) | (np.abs(act) >= tiny)
        sub = ~normal
        out = normal & ((act < lo2) | (act > hi2))
        out |= sub & (np.abs(act.astype(np.float64) - ref) > tiny)
        r = res[name]
        r['cells'] += int(normal.sum()); r['outside'] += int(out.sum())
        r['byte'] += int(np.count_nonzero(normal & (act.view(np.uint32) == ref.view(np.uint32))))
        if out.any():
            k = int(np.argmax(np.where(out, np.maximum(lo2 - act, act - hi2), -np.inf)))
            excess = float(max(lo2[k] - act[k], act[k] - hi2[k]) / np.spacing(np.abs(ref[k])))
            r['worst_excess_ulp'] = max(r['worst_excess_ulp'], excess)
            worst.append(dict(row=int(i), field=name, k=k, act=float(act[k]), lo=float(lo[k]), hi=float(hi[k]),
                              ref=float(ref[k]), excess_ulp=excess))
summary = dict(tag=tag, envelope_sha256=json.loads((lane / 'ulp_envelope.json').read_text())['envelope_sha256'],
               fields=res, total_outside=sum(r['outside'] for r in res.values()),
               worst=sorted(worst, key=lambda w: -w['excess_ulp'])[:12])
(lane / f'envelope_{tag}.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(dict(total_outside=summary['total_outside'], fields=res), indent=1))
for w in summary['worst'][:8]:
    print(w)
