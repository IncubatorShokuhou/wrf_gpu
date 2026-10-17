"""Count NTU adaptive loop trips per column on real c88 states (CPU, NumPy literal path).

Counts advance_small calls (small-step loop), macro steps and the per-call flux
sub-steps, column by column, for a sample of cloudy columns. Diagnostic only.
usage: ntu_trip_counts.py OUT_JSON wrfout NCOLS
"""
import json
import os
import sys
import netCDF4
import numpy as np

assert os.environ['JAX_PLATFORMS'] == 'cpu'
from gpuwrf.physics.ntu import column, phases_generated as phases, math as ntumath  # noqa: E402
from gpuwrf.physics.ntu.registry import output_mapping  # noqa: E402

NAMES = {'qv': 'QVAPOR', 'qc': 'QCLOUD', 'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP', 'qh': 'QHAIL',
         'nc': 'QNCLOUD', 'nr': 'QNRAIN', 'ni': 'QNICE', 'ns': 'QNSNOW', 'ng': 'QNGRAUPEL', 'nh': 'QNHAIL'}
d = netCDF4.Dataset(sys.argv[2])
g = lambda k: np.asarray(d[k][0], np.float32)
col = lambda a: np.ascontiguousarray(np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]))
p = g('P') + g('PB'); ph = (g('PH') + g('PHB')) / np.float32(9.81)
f = dict(th=col(g('T') + 300), p=col(p), pii=col((p / 1e5) ** (2 / 7)).astype(np.float32), dz=col(ph[1:] - ph[:-1]),
         w=col(g('W')))
for n in output_mapping():
    f[n] = col(g(NAMES.get(n, n.upper())))
for n, k in (('rainnc', 'RAINNC'), ('snownc', 'SNOWNC'), ('grapnc', 'GRAUPELNC'), ('hailnc', 'HAILNC')):
    f[n] = np.asarray(d[k][0], np.float32).reshape(-1)
cloud = (f['qc'] + f['qi'] + f['qs'] + f['qr'] > 1e-8).sum(axis=1)
order = np.argsort(-cloud)
pick = np.unique(np.concatenate([order[:int(sys.argv[3]) // 2],
                                 order[np.linspace(0, len(order) - 1, int(sys.argv[3]) // 2).astype(int)]]))
counts = {'small': 0, 'flux_iters': 0}
orig_small, orig_while = phases.advance_small, column.while_loop


def small(*a, **k):
    counts['small'] += 1
    return orig_small(*a, **k)


phases.advance_small = small
orig_loop = ntumath._loop


def loop(start, stop, body, state, *, xp):
    name = body.__name__
    trips = [0]

    def counted(i, st):
        trips[0] += 1
        return body(i, st)
    out = orig_loop(start, stop, counted, state, xp=xp)
    counts[name] = counts.get(name, 0) + trips[0]
    counts[name + '_calls'] = counts.get(name + '_calls', 0) + 1
    return out


ntumath._loop = loop
orig_flux = column.flux


def flux(q, vt, rho, dz, dt, dtmn=None, *, xp=np):
    r = np.dtype(q.dtype).type
    nstep = int(np.max(np.maximum(1, np.asarray(vt * dt / dz + r(1), dtype=np.int32))))
    counts['flux_iters'] += nstep
    return orig_flux(q, vt, rho, dz, dt, dtmn, xp=xp)


column.flux = flux
rows = []
for c in pick:
    for k in list(counts):
        counts[k] = 0
    one = {k: (v[c:c + 1] if v.ndim == 2 else v[c:c + 1]) for k, v in f.items()}
    column.step(one, np.float32(18), ccnty=2, xp=np)
    rows.append(dict(column=int(c), cloudy_levels=int(cloud[c]), small_steps=counts['small'],
                     loops={k: v for k, v in counts.items() if k not in ('small',)}))
    print(json.dumps(rows[-1]), flush=True)
s = np.array([r['small_steps'] for r in rows])
out = dict(frame=os.path.basename(sys.argv[2]), n=len(rows), small_steps=dict(min=int(s.min()), median=float(np.median(s)),
           max=int(s.max()), mean=float(s.mean())), rows=rows)
open(sys.argv[1], 'w').write(json.dumps(out, indent=2) + '\n')
print(json.dumps(out['small_steps']))
