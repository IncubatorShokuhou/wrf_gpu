"""NTU column-order / latency attribution on REAL c88 columns (GPU, measurement only).

One program per column (grid=(ncol,)), so a column permutation cannot change any output bit;
it only changes which columns share SMs in which wave. Arms per frame, same executable:
natural order (production), heavy-first, light-first (proxy = levels with hydrometeor > 1e-8).
Permuted outputs are un-permuted and compared BITWISE with natural. Separately compiled
subsets (heaviest 1 and 170 columns) give the single-column latency floor and check column
independence (subset rows bitwise == the same rows of the full call).
usage: gpu_ntu_order.py OUT_JSON wrfout...
"""
import json
import os
import sys
import time
import netCDF4
import numpy as np

assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
import jax  # noqa: E402
assert jax.devices()[0].platform == 'gpu'
from gpuwrf.kernels.phys_ntu import full_column  # noqa: E402
from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES  # noqa: E402
from gpuwrf.physics.ntu.registry import output_mapping  # noqa: E402

NAMES = {'qv': 'QVAPOR', 'qc': 'QCLOUD', 'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP', 'qh': 'QHAIL',
         'nc': 'QNCLOUD', 'nr': 'QNRAIN', 'ni': 'QNICE', 'ns': 'QNSNOW', 'ng': 'QNGRAUPEL', 'nh': 'QNHAIL'}


def fields(path):
    d = netCDF4.Dataset(path)
    g = lambda k: np.asarray(d[k][0], np.float32)
    col = lambda a: np.ascontiguousarray(np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]))
    p = g('P') + g('PB'); ph = (g('PH') + g('PHB')) / np.float32(9.81)
    f = dict(th=col(g('T') + 300), p=col(p), pii=col((p / 1e5) ** (2 / 7)), dz=col(ph[1:] - ph[:-1]), w=col(g('W')))
    for n in output_mapping():
        f[n] = col(g(NAMES.get(n, n.upper())))
    for n, k in (('rainnc', 'RAINNC'), ('snownc', 'SNOWNC'), ('grapnc', 'GRAUPELNC'), ('hailnc', 'HAILNC')):
        f[n] = np.asarray(d[k][0], np.float32).reshape(-1)
    d.close()
    return f


def take(f, idx):
    return {k: np.ascontiguousarray(v[idx]) for k, v in f.items()}


def timed(compiled, f, n=4):
    times = []
    for _ in range(n):
        t = time.monotonic(); out = compiled(f, np.float32(18)); jax.block_until_ready(out); times.append(time.monotonic() - t)
    return {k: np.asarray(v) for k, v in out.items()}, times


fn = jax.jit(lambda f, d: full_column(f, d, output_names=OUTPUT_NAMES))
cache, rec = {}, dict(frames={}, scope='GPU NTU kernel timing by column order/subset, real c88 CPU-WRF columns; '
                      'permutation arms bitwise vs natural; measurement only')


def compiled_for(f):
    n = len(f['rainnc'])
    if n not in cache:
        t = time.monotonic(); cache[n] = fn.lower(f, np.float32(18)).compile(); rec[f'compile_s_{n}'] = time.monotonic() - t
    return cache[n]


for path in sys.argv[2:]:
    f = fields(path)
    hyd = sum(f[k] for k in ('qc', 'qr', 'qi', 'qs', 'qg', 'qh'))
    proxy = (hyd > 1e-8).sum(axis=1)
    heavy = np.argsort(-proxy, kind='stable'); light = heavy[::-1].copy()
    base, t_nat = timed(compiled_for(f), f)
    row = dict(columns=len(proxy), cloudy_columns=int((proxy > 0).sum()), proxy_max=int(proxy.max()),
               natural=dict(call_s=t_nat, median_s=float(np.median(t_nat[1:]))))
    for name, perm in (('heavy_first', heavy), ('light_first', light)):
        out, t = timed(compiled_for(f), take(f, perm))
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        diff = sorted(k for k in out if out[k][inv].tobytes() != base[k].tobytes())
        row[name] = dict(call_s=t, median_s=float(np.median(t[1:])), bitwise_vs_natural=not diff, differing=diff)
    for k in (1, 170):
        idx = heavy[:k]
        out, t = timed(compiled_for(take(f, idx)), take(f, idx))
        diff = sorted(n for n in out if out[n].tobytes() != base[n][idx].tobytes())
        row[f'heaviest_{k}'] = dict(call_s=t, median_s=float(np.median(t[1:])), rows_bitwise_vs_full=not diff, differing=diff)
    rec['frames'][os.path.basename(path)] = row
    print(json.dumps({os.path.basename(path): {k: (v['median_s'] if isinstance(v, dict) and 'median_s' in v else v)
                                                for k, v in row.items()}}), flush=True)
json.dump(rec, open(sys.argv[1], 'w'), indent=2)
