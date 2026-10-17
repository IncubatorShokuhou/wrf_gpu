"""CPU interpretation identity: lowered StableHLO of the interpreted Pallas kernel.

The typed column memory primitives use the stock Pallas discharge rules, so the
interpreted (CPU) program must be textually identical to the stock-load one.
Writes sha256 of lowered.as_text() (no debug locations) for one scheme/shape.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', choices=('ntu', 'jensen'), required=True)
    p.add_argument('--nz', type=int, default=44)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    import jax
    import jax.numpy as jnp
    assert jax.devices()[0].platform == 'cpu'
    jax.config.update('jax_enable_x64', False)  # ambient REAL; kernels scope their DOUBLE islands
    s = jax.ShapeDtypeStruct((2, a.nz), jnp.float32)
    col = jax.ShapeDtypeStruct((2,), jnp.float32)
    if a.scheme == 'ntu':
        from gpuwrf.kernels.phys_ntu import full_column
        from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES
        from gpuwrf.physics.ntu.column import PROGNOSTICS
        fields = {n: s for n in ('th', 'p', 'pii', 'dz', *PROGNOSTICS)}
        fields['w'] = jax.ShapeDtypeStruct((2, a.nz + 1), jnp.float32)
        fields.update({n: col for n in ('rainnc', 'snownc', 'grapnc', 'hailnc')})
        fn = lambda x, d: full_column(x, d, ccnty=2, interpret=True, output_names=OUTPUT_NAMES)
        args = (fields, jax.ShapeDtypeStruct((), jnp.float32))
    else:
        from gpuwrf.kernels.phys_jensen import full_column
        from gpuwrf.physics.jensen_ishmael.registry import output_mapping
        from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping
        mapping = output_mapping()
        names = [n for n in mapping if n not in ('effc', 'effi')] + ['th', 'p', 'dz']
        fields = {n: s for n in names}
        mapping.update(ice_diagnostic_mapping(55))
        outs = (*mapping, 'th')
        fn = lambda x, d, i: full_column(x, d, i, interpret=True, output_names=outs)
        args = (fields, jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((), jnp.int32))
    with jax.enable_x64(True):
        text = jax.jit(fn).lower(*args).as_text()
    result = {'scheme': a.scheme, 'nz': a.nz, 'bytes': len(text),
              'sha256': hashlib.sha256(text.encode()).hexdigest(),
              'custom_names': [w for w in ('column_global', 'qnse_global', 'p3_global') if w in text]}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
