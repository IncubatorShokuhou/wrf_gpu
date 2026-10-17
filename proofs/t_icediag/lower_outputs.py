"""Measure changed native output projections on CPU, CUDA TTIR only (no GPU)."""
import argparse
import json
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering
from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mp', type=int, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    root = Path(__file__).resolve().parents[2]; mp = a.mp
    assert jax.devices()[0].platform == 'cpu'
    if mp == 55:
        from gpuwrf.kernels.phys_jensen import full_column
        from gpuwrf.physics.jensen_ishmael.registry import output_mapping
        d = json.loads((root / 'proofs/s2_mpmore/jensen55/fixtures/fp32_10.json').read_text())
        fields = {n: jnp.asarray(v, jnp.float32)[None, :] for n, v in d['input'].items()}
        before = (*output_mapping(), 'th')
        def call(names):
            return jax.make_jaxpr(lambda f, dt, it: full_column(f, dt, it, output_names=names))(
                fields, jnp.float32(d['dt']), jnp.int32(d['it']))
    else:
        from gpuwrf.kernels.phys_p3 import full_column
        from gpuwrf.physics.p3.adapter import output_mapping
        inputs = ('qv','qc','qr','qi','ni','nr','nc','qir','bir','th_old','qv_old','th','p','dz')
        inputs += ('qi2','ni2','qir2','bir2') if mp == 52 else ('qzi',) if mp == 53 else ()
        with np.load(root / f'proofs/s2_p3/fixtures/columns_mp{mp}.npz') as d:
            fields = {n: jnp.asarray(d[n][:1, :, 0]) for n in inputs}
        before = tuple(n for n in output_mapping(mp) if n not in ice_diagnostic_mapping(mp)) + ('th','rainncv','snowncv')
        def call(names):
            return jax.make_jaxpr(lambda f, dt, it: full_column(f, dt, it, mp=mp, output_names=names))(
                fields, jnp.float32(18), jnp.int32(1))
    after = ((*output_mapping(), *ice_diagnostic_mapping(mp), 'th') if mp==55 else
             (*output_mapping(mp), 'th', 'rainncv', 'snowncv'))
    report = {'mp': mp, 'platform': 'cpu', 'GPU_execution': False,
              'scope': 'isolated native output projection CUDA/sm120 TTIR; not cubin/E41/GPU runtime',
              'nz': fields['qv'].shape[-1], 'arms': {}}
    for tag, names in (('previous', before), ('retained', after)):
        start = time.monotonic()
        with jax.enable_x64(True):
            program = call(names)
        params = next(e.params for e in program.jaxpr.eqns if e.primitive.name == 'pallas_call')
        def equations(value):
            if hasattr(value, 'jaxpr'):
                yield from equations(value.jaxpr)
            elif hasattr(value, 'eqns'):
                for eq in value.eqns:
                    yield eq
                    for child in eq.params.values(): yield from equations(child)
            elif isinstance(value, (tuple, list)):
                for child in value: yield from equations(child)
            elif isinstance(value, dict):
                for child in value.values(): yield from equations(child)
        wide = [eq for eq in equations(params['jaxpr']) if any(
            getattr(getattr(v, 'aval', None), 'dtype', None) == np.dtype('float64') for v in eq.outvars)]
        outside = [eq for eq in wide if 'gammln' not in str(eq.source_info.traceback)]
        if mp == 55:
            assert not outside, 'FP64 outside original WRF gammln island'
        # P3's existing census includes weak f64 constants truncated to REAL;
        # E120/E193: classify actual TTIR ops rather than call them double math.
        module = lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', 120)
        ttir = str(module.module)
        double_ops = [line.strip() for line in ttir.splitlines() if 'f64' in line
                      and 'arith.constant' not in line and 'arith.truncf' not in line]
        report['arms'][tag] = dict(outputs=names, trace_lower_s=time.monotonic()-start,
            TTIR_bytes=len(ttir.encode()), output_bytes=sum(np.prod(v.aval.shape)*v.aval.dtype.itemsize
                                                           for v in program.jaxpr.outvars),
            pallas_calls=sum(e.primitive.name == 'pallas_call' for e in program.jaxpr.eqns),
            fp64_equations=len(wide), fp64_outside_WRF_gammln=len(outside),
            nonconstant_f64_TTIR_lines=double_ops)
        del module, ttir, program
        jax.clear_caches()
    report['extra_output_bytes_per_column'] = int(report['arms']['retained']['output_bytes'] - report['arms']['previous']['output_bytes'])
    report['maxrss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a.out.write_text(json.dumps(report, indent=2, default=int) + '\n')


if __name__ == '__main__':
    main()
