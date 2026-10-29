"""Locate first scalar differences between baked and traced default operands (CPU)."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels.dyn_acoustic_fp32 import _dampmag

assert jax.devices()[0].platform == 'cpu'
rows = []
for dt in (54., 18., 9., 4.5, 6., 3., 1.5):
    dynamic = jax.jit(lambda c: _dampmag(dt, c))
    static = jax.jit(lambda: jnp.float32(_dampmag(dt, .2)))
    before, after = np.asarray(static()), np.asarray(dynamic(jnp.float32(.2)))
    rows.append(dict(op='dampmag', dt=dt, off=float(before), on=float(after),
                     off_hex=before.tobytes().hex(), on_hex=after.tobytes().hex(), byte=before.tobytes()==after.tobytes()))
for dt in (54.,18.):
    def coeff(c):
        return jnp.asarray(c,jnp.float32)*jnp.float32(.015625)/(jnp.float32(2)*jnp.float32(dt))
    before=np.asarray(jax.jit(lambda: coeff(.12))())
    after=np.asarray(jax.jit(coeff)(jnp.float32(.12)))
    rows.append(dict(op='sixth_coef',dt=dt,off=float(before),on=float(after),off_hex=before.tobytes().hex(),on_hex=after.tobytes().hex(),byte=before.tobytes()==after.tobytes()))
Path(sys.argv[1]).write_text(json.dumps(dict(platform='cpu',gpu_executed=False,rows=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
