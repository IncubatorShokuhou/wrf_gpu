"""NTU kernel runtime on REAL c88 columns (original CPU-WRF reference states), GPU.

Inputs: wrfout frames of the original CPU-WRF c88 run (dry theta from T+300 with
the WRF theta_m convention ignored: probe only), p=P+PB, dz from PH+PHB, all 28
prognostics. Times full_column per call; not a fidelity check.
usage: gpu_ntu_real.py OUT_JSON wrfout...
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
fn = jax.jit(lambda f, d: full_column(f, d, output_names=OUTPUT_NAMES))
rows = []
for path in sys.argv[2:]:
    d = netCDF4.Dataset(path)
    g = lambda k: np.asarray(d[k][0], np.float32)
    col = lambda a: np.ascontiguousarray(np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]))
    p = g('P') + g('PB'); ph = (g('PH') + g('PHB')) / np.float32(9.81)
    f = dict(th=col(g('T') + 300), p=col(p), pii=col((p / 1e5) ** (2 / 7)), dz=col(ph[1:] - ph[:-1]),
             w=col(g('W')))
    for n in output_mapping():
        f[n] = col(g(NAMES.get(n, n.upper())))
    for n, k in (('rainnc', 'RAINNC'), ('snownc', 'SNOWNC'), ('grapnc', 'GRAUPELNC'), ('hailnc', 'HAILNC')):
        f[n] = np.asarray(d[k][0], np.float32).reshape(-1)
    d.close()
    times = []
    for i in range(4):
        t = time.monotonic(); out = fn(f, np.float32(18)); jax.block_until_ready(out); times.append(time.monotonic() - t)
    finite = all(bool(np.isfinite(np.asarray(v)).all()) for v in out.values())
    rows.append(dict(frame=os.path.basename(path), ncol=f['th'].shape[0], nz=f['th'].shape[1], call_s=times, finite=finite,
                     cloudy_cols=int((f['qc'] + f['qi'] + f['qs'] + f['qr'] > 1e-8).any(axis=1).sum())))
    print(json.dumps(rows[-1]), flush=True)
open(sys.argv[1], 'w').write(json.dumps(rows, indent=2) + '\n')
