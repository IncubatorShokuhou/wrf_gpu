"""Compile P3's Pallas jaxpr to CUDA TTIR without opening a GPU backend."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering

from gpuwrf.kernels.phys_p3 import full_column


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mp', type=int, default=50)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--operational-outputs', action='store_true')
    a = p.parse_args()
    assert jax.devices()[0].platform == 'cpu'
    with np.load(Path(__file__).with_name('fixtures')/f'columns_mp{a.mp}.npz') as fixture:
        fields = {name: jnp.asarray(value[:1, :, 0]) for name, value in fixture.items() if value.ndim == 3}
    if a.mp != 52:
        for name in ('qi2', 'ni2', 'qir2', 'bir2', 'vm2', 'di2', 'rhoi2'):
            fields.pop(name, None)
    if a.mp != 53:
        fields.pop('qzi', None)
    started = time.monotonic()
    outputs=None
    if a.operational_outputs:
        from gpuwrf.physics.p3.adapter import output_mapping
        outputs=(*output_mapping(a.mp),'th','rainncv','snowncv')
        inputs=('qv','qc','qr','qi','ni','nr','nc','qir','bir','th_old','qv_old','th','p','dz')
        inputs += ('qi2','ni2','qir2','bir2') if a.mp==52 else ('qzi',) if a.mp==53 else ()
        fields={name:fields[name] for name in inputs}
    with jax.enable_x64(True):
        program = jax.make_jaxpr(lambda f, dt, it: full_column(f, dt, it, mp=a.mp,output_names=outputs))(
            fields, jnp.float32(18), jnp.int32(1)).jaxpr
    params = next(e.params for e in program.eqns if e.primitive.name == 'pallas_call')
    def equations(value):
        if hasattr(value,'jaxpr'):
            yield from equations(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eqn in value.eqns:
                yield eqn
                for child in eqn.params.values():
                    yield from equations(child)
        elif isinstance(value,(tuple,list)):
            for child in value:yield from equations(child)
        elif isinstance(value,dict):
            for child in value.values():yield from equations(child)
    double_eqns=[e for e in equations(params['jaxpr'])
                 if any(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in e.outvars)]
    qzi_double=sum('p3/qzi.py' in str(e.source_info.traceback) for e in double_eqns)
    assert qzi_double==0,'CPU-only QZI wide emulator leaked into the native kernel'
    result = lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', 120)
    text = str(result.module)
    a.out.with_suffix('.ttir').write_text(text)
    report = {'backend': 'cpu', 'GPU_execution': False, 'target': 'cuda/sm120',
              'mp': a.mp, 'TTIR_characters': len(text), 'elapsed_s': time.monotonic()-started,
              'operational_outputs': outputs,
              'operational_inputs': tuple(fields) if outputs is not None else None,
              'output_bytes_per_column': None if outputs is None else 4*((len(outputs)-2)*fields['qv'].shape[-1]+2),
              'native_fp64_eqns':len(double_eqns),'qzi_native_fp64_eqns':qzi_double,
              'verdict': 'PASS'}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
