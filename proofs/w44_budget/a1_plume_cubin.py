"""A1 (critic advisory): CPU-only Triton sm_120 compile of the MYNN plume-sums kernel, production vs budget twin.
No GPU. Reports compile time, registers, STACK, local memory, SASS size, CALL count (E18/E137/E41 pre-screen)."""
import json, re, subprocess, sys, time, os
from pathlib import Path
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
import jax, jax.numpy as jnp
import triton
from jax._src.pallas.triton import lowering
from triton.backends.compiler import GPUTarget
from gpuwrf.kernels import phys_mynn_plume as P
from gpuwrf.kernels import phys_mynn_budget_plume as BP
OUT = Path(sys.argv[1]); assert jax.devices()[0].platform == 'cpu'
B, NZ, NUP = 112 * 94, 44, 8   # WN3 d03 columns, 44 levels, WRF NUP
z = lambda *s: jnp.zeros(s, jnp.float32)
def call(fn):
    lane = tuple(z(B, NUP) for _ in range(8))
    env = dict(thl=z(B, NZ), qt1=z(B, NZ), thv=z(B, NZ), p=z(B, NZ), dz=z(B, NZ), zw=z(B, NZ + 1), u=z(B, NZ), v=z(B, NZ),
               pblh=z(B), rhoz_dmp=z(B, NZ), active=None, interpret=True)
    return fn(*lane, **env)
rows = {}
for tag, fn in (('production', P.fused_plume_sums), ('budget_twin', BP.dmp_mf_columns_native.__globals__['fused_plume_sums'])):
    jx = jax.make_jaxpr(lambda: call(fn))().jaxpr
    eqs = [e for e in jx.eqns if e.primitive.name == 'pallas_call']
    assert len(eqs) == 1, len(eqs)
    eq = eqs[0]
    t0 = time.time()
    mod = lowering.lower_jaxpr_to_triton_module(eq.params['jaxpr'], eq.params['grid_mapping'], 'cuda', 120)
    ttir = OUT / f'{tag}.ttir'; ttir.write_text(mod.module.operation.get_asm(enable_debug_info=False))
    t1 = time.time()
    k = triton.compile(str(ttir), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4})
    t2 = time.time()
    cubin = OUT / f'{tag}.cubin'; cubin.write_bytes(k.asm['cubin'])
    ru = ' '.join(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(cubin)], text=True).split())
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(cubin)], text=True)
    instr = [l for l in sass.splitlines() if re.match(r'\s*/\*[0-9a-f]{4}\*/', l)]
    get = lambda key: (re.search(key + r':(\d+)', ru) or [None, None])[1]
    rows[tag] = dict(lower_s=round(t1 - t0, 2), triton_compile_s=round(t2 - t1, 2), ttir_lines=len(ttir.read_text().splitlines()),
                     REG=get('REG'), STACK=get('STACK'), LOCAL=get('LOCAL'), SHARED=get('SHARED'), sass_instr=len(instr),
                     calls=sum('CALL' in l for l in instr), resource_usage=ru)
    print(tag, {k2: v for k2, v in rows[tag].items() if k2 != 'resource_usage'}, flush=True)
v = dict(schema='wrf_gpu2.w44_budget.a1_plume_cubin.v1', shapes=dict(B=B, nz=NZ, nup=NUP, num_warps=4, target='sm_120 (GPUTarget cuda 120)'),
         rows=rows, e41_stack_ok=all(int(r['STACK'] or 0) <= 256 for r in rows.values()))
(OUT / 'result.json').write_text(json.dumps(v, indent=2) + '\n'); print('E41_STACK_OK', v['e41_stack_ok'])
