"""Lower the actual c14/c15 CLI-loaded release program to CUDA on a CPU client.

No GPU client, executable compilation or forecast claim. CPU interpretation is
used only for loader initialization; native Pallas is restored before tracing.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'proofs/o1_nlbind')]
# Match gpu_case.py: fast defaults (including Triton selection) precede JAX.
import gpuwrf
import cpu_safe
real_tridiagonal_solve = cpu_safe.jax.lax.linalg.tridiagonal_solve
cpu_safe.install()
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from gpuwrf.backends import pallas as facade
from gpuwrf.contracts import state as sc
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as op

assert jax.devices()[0].platform == 'cpu'
sc._gpu_device = lambda: jax.devices('cpu')[0]
result = dict(source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              platform='cpu', lowering_platform='cuda', gpu_executed=False, arms={},
              environment={k:os.environ.get(k) for k in ('GPUWRF_FAST_DEFAULTS','GPUWRF_CENSUS',
                  'GPUWRF_CENSUS_SLIM','GPUWRF_RRTMG_MP_RE','JAX_PALLAS_USE_MOSAIC_GPU')})
real_couple = op._coupled_physics_moist_tendf
real_backend = jax.default_backend
real_pallas_call = pl.pallas_call
def native_pallas_call(*a, **kw):
    kw['interpret'] = False
    return real_pallas_call(*a, **kw)
for case in ('c14','c15'):
    case_root = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/case
    seen = []
    def capture(*a, **kw):
        seen.append(sorted(a[1]))
        return real_couple(*a, **kw)
    op._coupled_physics_moist_tendf = capture
    start = time.monotonic()
    rec = dict(input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (case_root/'input').iterdir() if p.is_file()})
    try:
        jax.default_backend = real_backend
        pl.pallas_call = real_pallas_call
        cpu_safe.install()
        facade.cuda_native = lambda: False
        config = pipeline.NestedPipelineConfig(case_root/'input', OUT.parent/(case+'_out'),
                        OUT.parent/(case+'_proof'), hours=24, max_dom=1)
        _, bundles, _, _, _, carries = pipeline._load_domains(config, ('d01',))
        nl, carry = bundles['d01'].namelist, carries['d01']
        facade.cuda_native = lambda: True
        # E115/E156: explicit CUDA lowering alone does not change Python
        # branches that ask default_backend() while tracing the program.
        jax.default_backend = lambda: 'gpu'
        pl.pallas_call = native_pallas_call
        jax.lax.linalg.tridiagonal_solve = real_tridiagonal_solve
        op._resolve_operational_suite(nl)
        cb = op.build_clock_base(nl)
        # Same argument pytree as aot_precompile / _operational_advance_factory;
        # closing nl/clock/start in an outer lambda would bake dynamic inputs.
        traced = op._advance_chunk_fori.trace(carry,nl,jnp.asarray(1,jnp.int32),cb,
                                    n_steps=1,cadence=int(nl.radiation_cadence_steps))
        rec['physics_raw_keys'] = seen
        assert seen and all('Ni' not in keys for keys in seen), seen
        ir = traced.lower(lowering_platforms=('cuda',)).as_text()
        rec.update(status='PASS', stablehlo_sha256=hashlib.sha256(ir.encode()).hexdigest(),
                   stablehlo_bytes=len(ir.encode()), mp_physics=nl.mp_physics,cu_physics=nl.cu_physics,
                   phys_tend_rk_wrf=op._phys_tend_rk_wrf_enabled(),native_facade_at_trace=True,
                   python_backend_at_trace='gpu',pallas_interpret_at_trace=False,
                   direct_product_signature=True,start_step_dtype='int32',n_steps=1,
                   triton_custom_calls=ir.count('__gpu$xla.gpu.triton'))
        assert rec['phys_tend_rk_wrf']
        assert rec['triton_custom_calls'] > 0
        del ir, traced, carry, carries, bundles
    except Exception as exc:
        import traceback
        rec.update(status='FAIL', error=repr(exc),traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.monotonic()-start
    result['arms'][case] = rec
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(case,rec['status'],rec.get('error',''),rec['wall_s'],flush=True)
    jax.clear_caches()
assert all(r['status']=='PASS' for r in result['arms'].values()), {k:r.get('error') for k,r in result['arms'].items()}
