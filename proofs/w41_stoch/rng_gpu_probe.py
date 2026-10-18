"""GPU component probe (lane w41-stoch): parallel vs serial WRF/GNU Gaussian stream, bitwise + device timing.
Run under scripts/with_gpu_lock.sh from an immutable snapshot. usage: python rng_gpu_probe.py <out.json>"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
import gpuwrf  # noqa: F401  (x64/XLA configuration)
import jax, jax.numpy as jnp
import numpy as np
from gpuwrf.physics import stochastic_rng as rng
assert jax.devices()[0].platform == 'gpu', jax.devices()
out = {'device': str(jax.devices()[0]), 'platform': jax.devices()[0].platform, 'shapes': {}}
for shape in ((43, 43), (71, 121), (118, 268)):
    seed = jnp.asarray(rng.seed_array(811, 7, datetime(2023, 1, 15)))
    def chain(f):
        def run(s):   # one model step's worth: 7 chained calls on one stream (upper bound of st4)
            res = []
            for _ in range(7):
                a, b, s = f(s, shape); res += [a, b]
            return res, s
        return jax.jit(run)
    rec = {}
    outs = {}
    for name, f in (('serial', rng.gaussian_innovations_serial), ('parallel', rng.gaussian_innovations)):
        g = chain(f)
        t = time.perf_counter(); r = g(seed); jax.block_until_ready(r); compile_s = time.perf_counter() - t
        times = []
        for _ in range(20):
            t = time.perf_counter(); jax.block_until_ready(g(seed)); times.append(time.perf_counter() - t)
        outs[name] = jax.device_get(r)
        rec[name] = dict(first_call_s=round(compile_s, 3), median_ms_per_7_calls=round(1e3*float(np.median(times)), 3),
                         min_ms=round(1e3*min(times), 3))
    a, b = outs['serial'], outs['parallel']
    rec['bitwise_equal'] = bool(all(np.array_equal(np.asarray(x).view(np.uint32), np.asarray(y).view(np.uint32))
                                    for x, y in zip(a[0], b[0])) and np.array_equal(a[1], b[1]))
    rec['speedup'] = round(rec['serial']['median_ms_per_7_calls'] / rec['parallel']['median_ms_per_7_calls'], 2)
    out['shapes'][f'{shape[0]}x{shape[1]}'] = rec
    print(shape, rec, flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
sys.exit(0 if all(r['bitwise_equal'] for r in out['shapes'].values()) else 1)
