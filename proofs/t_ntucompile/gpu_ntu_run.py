"""Production NTU full_column on REAL c88 columns (GPU): save all outputs + s/call.

Two source trees are compared by running this once per tree and diffing the
saved NPZ bitwise on CPU (ntu_npz_compare.py). Inputs as gpu_ntu_ab.py.
usage: gpu_ntu_run.py OUT_PREFIX wrfout...
"""
import json
import os
import sys
import time
import numpy as np

assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
import jax  # noqa: E402
assert jax.devices()[0].platform == 'gpu'
import netCDF4  # noqa: E402
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


prefix = sys.argv[1]
fn = jax.jit(lambda f, d: full_column(f, d, output_names=OUTPUT_NAMES))
inputs = {os.path.basename(p): fields(p) for p in sys.argv[2:]}
t = time.monotonic()
compiled = fn.lower(next(iter(inputs.values())), np.float32(18)).compile()
rec = dict(compile_s=time.monotonic() - t, frames={})
arrays = {}
for name, f in inputs.items():
    times = []
    for _ in range(4):
        t = time.monotonic(); out = compiled(f, np.float32(18)); jax.block_until_ready(out); times.append(time.monotonic() - t)
    for k, v in out.items():
        arrays[f'{name}|{k}'] = np.asarray(v)
    rec['frames'][name] = dict(call_s=times, median_s=float(np.median(times[1:])),
                               finite=all(bool(np.isfinite(np.asarray(v)).all()) for v in out.values()))
np.savez(prefix + '.npz', **arrays)
json.dump(rec, open(prefix + '.json', 'w'), indent=2)
print(json.dumps(rec))
