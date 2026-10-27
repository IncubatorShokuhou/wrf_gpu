"""CPU proof: jit-level compiler_options + the AOT path's call-level lowered.compile(options) both land in the serialized GPU
executable's DebugOptions (CompileOnly sm_120 topology; toy program). Field 208 = xla_gpu_graph_min_graph_size, 435 = compiled thunks."""
import os, sys, importlib.util
from pathlib import Path
assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path('<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf'); sys.path.insert(0, str(ROOT / 'src'))
import jax, jax.numpy as jnp
from jax.experimental import topologies
from jax._src import compiler as jc
jc._get_cross_compile_backend = lambda _: None
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
F = lambda d: list(ax._proto_fields(d))
topo = topologies.get_topology_desc(platform='cuda', target_config=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/target_config.pbtxt').read_text(), topology='1x1x1')
sh = jax.sharding.SingleDeviceSharding(topo.devices[0])
def f(x):
    def body(c):
        i, v = c
        return i + 1, jnp.sin(v) * 1.0001 + v
    return jax.lax.while_loop(lambda c: c[0] < 3, body, (0, x))[1]
def fields(blob):
    gpu = g._gpu_payload(blob, ax, asm)
    if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT:
        _, info, pos = asm._chunks(gpu)[-1]; data, _ = asm._read_logical(gpu, pos, info['data_size'])
        recs = asm._decode_simple_chunk(data, info['num_records'])
    else:
        recs = [gpu]
    dbg = None
    for rec in recs:
        try:
            hwc = next(v for k, w, v in F(rec) if k == 1); cfg = next(v for k, w, v in F(hwc) if k == 2); dbg = next(v for k, w, v in F(cfg) if k == 14); break
        except Exception:
            continue
    if dbg is None: return 'no-debug-options'
    out = {}
    for k, w, v in F(dbg):
        if k in (208, 258, 435): out.setdefault(k, []).append(v.hex() if isinstance(v, bytes) else v)
    return out
arg = jax.ShapeDtypeStruct((1024,), jnp.float32, sharding=sh)
cases = {
 'plain jit, call {435}': (jax.jit(f), {'xla_gpu_experimental_aot_compiled_thunks': True}),
 'jit{208:1}, call {435}': (jax.jit(f, compiler_options={'xla_gpu_graph_min_graph_size': 1}), {'xla_gpu_experimental_aot_compiled_thunks': True}),
 'jit{258:-WHILE list}, call {435}': (jax.jit(f, compiler_options={'xla_gpu_enable_command_buffer': 'FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,CONDITIONAL'}), {'xla_gpu_experimental_aot_compiled_thunks': True}),
 'jit{208:1}, call None': (jax.jit(f, compiler_options={'xla_gpu_graph_min_graph_size': 1}), None),
}
for name, (fn, opts) in cases.items():
    low = fn.trace(arg).lower(lowering_platforms=('cuda',))
    comp = low.compile(opts) if opts is not None else low.compile()
    print(f'{name:36s}', fields(comp.runtime_executable().serialize()))
