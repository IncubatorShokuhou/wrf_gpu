"""Private NTU-only typed-f32 I/O probe; immutable model files are untouched.

Reuse the qualified P3 memory primitives for NTU float32 references. Original
DOUBLE/integer loads retain the original implementation. Physics arithmetic,
launch geometry, references and tolerances are unchanged. This CPU experiment
does not qualify a GPU forecast or claim new original-WRF fidelity.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source = Path(sys.argv[sys.argv.index('--source') + 1])
output = Path(sys.argv[sys.argv.index('--out') + 1])
typed_double = '--typed-double' in sys.argv
if typed_double:
    sys.argv.remove('--typed-double')
sys.path.insert(0, str(source / 'src'))
import gpuwrf
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from gpuwrf.kernels import phys_ntu, _p3_memory

assert all(device.platform == 'cpu' for device in jax.devices())
original = phys_ntu.plt
if typed_double:
    from jax._src.pallas.triton import lowering as tl
    from jaxlib.mlir import ir
    from jaxlib.triton import dialect as tt
    original_lower = _p3_memory._load_triton

    @tl.register_lowering(_p3_memory._load)
    def double_load(ctx, *args, args_tree):
        if not ctx.avals_out[0].shape or ctx.avals_out[0].dtype != jnp.float64:
            return original_lower(ctx, *args, args_tree=args_tree)
        ptr, indexers, mask, other = args_tree.unflatten(args)
        address, shape = _p3_memory._address(ctx, ptr, indexers)
        *_, mask_aval, other_aval = args_tree.unflatten(ctx.avals_in)
        f64 = ir.RankedTensorType.get(shape, ir.F64Type.get())
        i32 = ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(32))
        pred = (tl._full(i32, 1) if mask is None else
                tl._ir_cast(_p3_memory._operand(mask, mask_aval, shape), i32, signed=False))
        fallback = (tl._full(f64, 0.) if other is None else
                    tl._ir_cast(_p3_memory._operand(other, other_aval, shape), f64, signed=False))
        return tt.elementwise_inline_asm([f64],
            '{ .reg .pred p; setp.ne.u32 p, $2, 0; mov.b64 $0, $3; @p ld.global.b64 $0, [$1]; }',
            '=d,l,r,d', False, 1, [address, pred, fallback])


class TypedFloat32:
    def __getattr__(self, name):
        return getattr(original, name)

    @staticmethod
    def load(view, **kwargs):
        if view.dtype == jnp.float32 or (typed_double and view.dtype == jnp.float64):
            return _p3_memory.load(view, **kwargs)
        return original.load(view, **kwargs)

    @staticmethod
    def store(view, value, **kwargs):
        if view.dtype == jnp.float32:
            return _p3_memory.store(view, value, **kwargs)
        return original.store(view, value, **kwargs)


phys_ntu.plt = TypedFloat32()
jax.clear_caches()
# This checks changed memory wiring, not physics fidelity. Values come from
# the sealed original-WRF NTU fixture; arithmetic remains the qualified code.
fixture = source / 'proofs/s2_ntu/fixtures/columns_r4.npz'
values = np.load(fixture)['c08_s0_th'].astype(np.float32)
width = 1 << (len(values) - 1).bit_length()


def copy_kernel(src, dst):
    index = jnp.arange(width, dtype=jnp.int32)
    mask = index < len(values)
    result = phys_ntu.plt.load(src.at[index], mask=mask, other=0.)
    phys_ntu.plt.store(dst.at[index], result, mask=mask)


actual = pl.pallas_call(copy_kernel, grid=(), interpret=True,
    out_shape=jax.ShapeDtypeStruct(values.shape, jnp.float32))(values)
assert np.asarray(actual).tobytes() == values.tobytes()
variant = {'variant': 'NTU-only qualified P3 typed float32 I/O',
    'typed_DOUBLE_load_extension': typed_double,
    'model_files_changed': False, 'physics_arithmetic_changed': False,
    'memory_regression': 'PASS original fixture values BYTE; not physics fidelity',
    'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
    'helper_sha256': hashlib.sha256((source / 'src/gpuwrf/kernels/_p3_memory.py').read_bytes()).hexdigest(),
    'scope': 'CPU-only private compilation experiment; actual GPU/E41/oracle qualification remains separate'}
(output.parent / 'VARIANT.json').write_text(json.dumps(variant, indent=2) + '\n')
harness = source / 'proofs/t_lowersweep/lower_case.py'
sys.argv[0] = str(harness)
runpy.run_path(str(harness), run_name='__main__')
