"""Count the actual staged JAX graph's buffers; this is not peak GPU VRAM."""
from pathlib import Path
import json
import math
import os
import sys

assert os.environ['JAX_PLATFORMS'] == 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling.glacier_surface_hook import glacier_kernel
from gpuwrf.io.noahmp_glacier_options import GlacierStatic

assert jax.devices()[0].platform == 'cpu'
root = (Path(sys.argv[2]) / 'proofs/s2_glacier' if len(sys.argv) > 2
        else Path(__file__).resolve().parent)
data = np.load(root / 'glacier_oracle.npz')
manifest = json.loads((root / 'oracle_manifest.json').read_text())
rows = []
for gla in (1, 2):
    index = np.flatnonzero(data['options'][:, -1] == gla)[0]
    packed = np.repeat(data['inputs'][index, :, None], 22, axis=1)
    inputs = {}
    for field in manifest['fields']:
        if field['intent'] == 'OUT' or field['name'] in ('iloc', 'jloc', 'nsnow', 'nsoil'):
            continue
        a = packed[field['offset']:field['offset'] + field['extent']]
        a = jnp.asarray(a, jnp.int32 if field['integer'] else jnp.float32)
        inputs[field['name']] = ([x.reshape(1, -1) for x in a]
                                 if field['extent'] > 1 else a[0].reshape(1, -1))
    config = GlacierStatic(tuple(range(22)), (), (1, 22), opt_gla=gla)
    graph = jax.make_jaxpr(lambda v: glacier_kernel(v, config, history=True))(inputs)
    calls = [e for e in graph.jaxpr.eqns if e.primitive.name == 'pallas_call']
    assert len(calls) == 3
    for stage, eqn in enumerate(calls):
        sizes = [math.prod(v.aval.shape) * np.dtype(v.aval.dtype).itemsize
                 for v in eqn.outvars]
        rows.append(dict(opt_gla=gla, stage=stage, output_buffers=len(sizes),
                         logical_output_bytes=sum(sizes)))
result = dict(platform='cpu', gpu_executed=False, glacier_columns=22,
              column_kernel_calls_per_step=3, rows=rows,
              scope='Actual Pallas graph output buffer sizes, before backend aliasing/DCE; '
                    'excludes natural-tile gather/scatter and is not measured GPU peak VRAM.')
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
