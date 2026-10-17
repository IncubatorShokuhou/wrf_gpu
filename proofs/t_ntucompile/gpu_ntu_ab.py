"""NTU kernel A/B on REAL c88 columns, GPU: warps variants vs production, bitwise + s/call.

Variant = num_warps override of the production full_column (trace-time proxy of the
`plt` module inside phys_ntu only); production is 4 warps. Inputs: original CPU-WRF
c88 frames (same construction as gpu_ntu_real.py). Every variant's 40 outputs are
compared BITWISE with production on the same inputs. Not a fidelity check by itself:
bitwise equality with production carries the production qualification over.
usage: gpu_ntu_ab.py OUT_JSON WARPS[,WARPS...] wrfout...
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
from jax.experimental.pallas import triton as real_plt  # noqa: E402
from gpuwrf.kernels import phys_ntu  # noqa: E402
from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES  # noqa: E402
from gpuwrf.physics.ntu.registry import output_mapping  # noqa: E402

NAMES = {'qv': 'QVAPOR', 'qc': 'QCLOUD', 'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP', 'qh': 'QHAIL',
         'nc': 'QNCLOUD', 'nr': 'QNRAIN', 'ni': 'QNICE', 'ns': 'QNSNOW', 'ng': 'QNGRAUPEL', 'nh': 'QNHAIL'}


class Plt:
    def __init__(self, warps):
        self.warps = warps

    def __getattr__(self, name):
        return getattr(real_plt, name)

    def CompilerParams(self, **kw):  # noqa: N802
        return real_plt.CompilerParams(**{**kw, 'num_warps': self.warps})


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


inputs = {os.path.basename(p): fields(p) for p in sys.argv[3:]}
variants = [4] + [int(w) for w in sys.argv[2].split(',') if int(w) != 4]
ref, rows = {}, []
for warps in variants:
    phys_ntu.plt = Plt(warps)
    fn = jax.jit(lambda f, d: phys_ntu.full_column(f, d, output_names=OUTPUT_NAMES))
    t = time.monotonic()
    compiled = fn.lower(next(iter(inputs.values())), np.float32(18)).compile()
    row = dict(num_warps=warps, compile_s=time.monotonic() - t, frames={})
    for name, f in inputs.items():
        times = []
        for _ in range(4):
            t = time.monotonic(); out = compiled(f, np.float32(18)); jax.block_until_ready(out)
            times.append(time.monotonic() - t)
        out = {k: np.asarray(v) for k, v in out.items()}
        if warps == 4:
            ref[name] = out
        diff = sorted(k for k in out if out[k].tobytes() != ref[name][k].tobytes())
        row['frames'][name] = dict(call_s=times, median_s=float(np.median(times[1:])), bitwise_vs_4w=not diff,
                                   differing=diff, finite=all(bool(np.isfinite(v).all()) for v in out.values()))
    rows.append(row)
    print(json.dumps(row), flush=True)
phys_ntu.plt = real_plt
open(sys.argv[1], 'w').write(json.dumps(dict(rows=rows, outputs=len(OUTPUT_NAMES),
     scope='GPU NTU kernel A/B, real c88 CPU-WRF columns; bitwise vs production 4 warps'), indent=2) + '\n')
