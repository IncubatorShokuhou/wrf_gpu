"""CPU-only Triton lowering + triton.compile (sm_120) of every RNG Pallas kernel (lane w41-stoch).
No GPU. usage: python rng_ttir_probe.py <out_dir> [ny nx]"""
import json, sys, time, traceback
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import jax, jax.numpy as jnp
from gpuwrf._x64_config import configure_jax_x64
configure_jax_x64()
from gpuwrf.physics import stochastic_rng as rng
from jax._src.pallas.triton import lowering
assert jax.default_backend() == 'cpu'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
shape = tuple(int(x) for x in sys.argv[2:4]) if len(sys.argv) > 3 else (43, 43)
seed = jnp.asarray(rng.seed_array(811, 7, datetime(2023, 1, 15)))
closed = jax.make_jaxpr(lambda s: rng.gaussian_innovations(s, shape))(seed)
calls = []
def walk(jaxpr):
    for e in jaxpr.eqns:
        if str(e.primitive) == 'pallas_call':
            calls.append(e.params)
        for v in e.params.values():
            for sub in (v if isinstance(v, (list, tuple)) else (v,)):
                if hasattr(sub, 'jaxpr') and hasattr(sub.jaxpr, 'eqns'):
                    walk(sub.jaxpr)
                elif hasattr(sub, 'eqns'):
                    walk(sub)
walk(closed.jaxpr)
records = []
for params in calls:
    name = params.get('name_and_src_info') or params.get('name')
    rec = dict(kernel=str(name).split(' ')[0])
    t = time.perf_counter()
    try:
        result = lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', 120)
        asm = result.module.operation.get_asm()
        (out/f"{rec['kernel']}.ttir").write_text(asm)
        rec.update(ttir_chars=len(asm), grid=list(result.grid), f64_tokens=asm.count('f64'))
        try:
            import triton
            from triton.backends.compiler import GPUTarget
            src = out/f"{rec['kernel']}.ttir"
            t1 = time.perf_counter()
            k = triton.compile(str(src), target=GPUTarget('cuda', 120, 32))
            md = k.metadata
            rec.update(triton_compile_s=round(time.perf_counter()-t1, 2), n_regs=getattr(k, 'n_regs', None),
                       n_spills=getattr(k, 'n_spills', None), shared=getattr(md, 'shared', None))
        except Exception as exc:
            rec['triton_compile'] = f'ERROR {type(exc).__name__}: {str(exc)[:300]}'
        rec['verdict'] = 'PASS' if 'triton_compile' not in rec else 'LOWER_ONLY'
    except Exception:
        rec['verdict'] = 'ERROR'; rec['error'] = traceback.format_exc()[-1500:]
    rec['elapsed_s'] = round(time.perf_counter()-t, 2)
    records.append(rec)
(out/'receipt.json').write_text(json.dumps(dict(shape=shape, kernels=records, scope='CPU-only; no GPU execution'), indent=1)+'\n')
print(json.dumps([{k: v for k, v in r.items() if k != 'error'} for r in records], indent=1))
for r in records:
    if 'error' in r: print(r['error'])
