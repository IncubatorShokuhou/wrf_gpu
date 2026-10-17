"""Pallas-Triton kernels compiled by STOCK Triton, launched by jaxlib.

JAX 0.10 lowers every Triton pallas_call to XLA's in-graph
``__gpu$xla.gpu.triton`` custom call (XLA CompileTritonToLLVM). For the native
CAM-MG column that route miscompiles control flow on sm_120 (activity loop never
taken: RAINNCV/SNOWNCV 0; independent of ptxas version/-O0, num_warps,
num_stages and of the column's loop/branch structure; lanes t-rainnc/t-cammgws,
2026-10-15). The defect sits in jaxlib/XLA's bundled Triton lowering: the
plugin compiler (``jax._src.lib.triton.compile``) drops the same QC/QI/CMEI
loads, while stock Python Triton compiles the identical TTIR correctly
(t-rainnc RN56/RN57/RN59). Kernels opting in with KernelCallCompilerParams are
lowered to the unchanged TTIR, compiled to PTX by the installed stock Triton
(all stages, make_ttir first, as triton.compile does for Python sources), and
launched by jaxlib ``triton_kernel_call`` (JAX's own non-default launch route in
jax/_src/pallas/triton/pallas_call_registration.py). All other pallas_calls keep
the default route.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
from pathlib import Path
import re
import tempfile
import zlib

import jax
from jax._src.interpreters import mlir
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.pallas import core as pallas_core
from jax._src.pallas.triton import core as triton_core
from jax._src.pallas.triton import lowering
from jax._src.pallas.triton.pallas_call_registration import avals_to_layouts, normalize_grid


VALIDATED_TRITON=(3,6)  # GPU-qualified stock Triton release family (t-cammgws CG25/CG26/CG28/CG33)
_INSTALL='pip install "triton==3.6.*"'


@dataclasses.dataclass(frozen=True)
class KernelCallCompilerParams(triton_core.CompilerParams):
    """Triton parameters selecting the stock-Triton + triton_kernel_call route."""
    enable_fp_fusion: bool = True


def _stock_triton():
    """The installed stock Triton, or an actionable error (key construction and lowering alike)."""
    try:
        import triton
    except ImportError as error:
        raise RuntimeError(f'native mp11 on CUDA needs stock Triton 3.6: {_INSTALL}') from error
    release=re.match(r'(\d+)\.(\d+)(?![0-9])',triton.__version__)
    if release is None or (int(release[1]),int(release[2]))!=VALIDATED_TRITON:
        raise RuntimeError(f'stock Triton {triton.__version__} is not validated for the mp11 CUDA route '
                           f'(validated 3.6.x): {_INSTALL}')
    return triton


def _stock_triton_root():
    return Path(_stock_triton().__file__).resolve().parent


def tree_digest(root):
    """Content (not names/sizes/mtimes) of every file of a package tree, path-ordered."""
    root=Path(root);digest=hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode()+b'\0')
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1<<22),b''):digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


@functools.lru_cache(maxsize=4)
def _cached_tree_digest(root):
    return tree_digest(root)


def stock_compiler_identity():
    """Content identity of the PTX generator (AOT/cheap-key input; built at key time, never per step).

    Covers the whole stock Triton package (compiled libtriton, Python CUDABackend and
    compiler stages, bundled libdevice) plus the route and kernel module sources.
    """
    root=_stock_triton_root()
    sources=tuple((Path(name).name,hashlib.sha256(Path(name).read_bytes()).hexdigest())
                  for name in (__file__,_kernel_module_file()))
    return ('stock-triton-content',_stock_triton().__version__,_cached_tree_digest(str(root)),sources)


def _kernel_module_file():
    from gpuwrf.kernels import phys_cammg
    return phys_cammg.__file__


@functools.lru_cache(maxsize=8)
def stock_compile(ttir: str, capability: int, num_warps: int, num_stages: int, enable_fp_fusion: bool = True,
                  compiler_identity: tuple = ()):
    """TTIR text -> (PTX, shared bytes, kernel name) with stock Triton, all stages.

    ``compiler_identity`` (stock_compiler_identity()) is part of the memo key only.
    """
    triton=_stock_triton()
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import Language
    target=GPUTarget('cuda',capability,32);backend=CUDABackend(target)
    options=backend.parse_options(dict(num_warps=num_warps,num_stages=num_stages,num_ctas=1,enable_fp_fusion=enable_fp_fusion))
    context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
    with tempfile.NamedTemporaryFile('w',suffix='.ttir') as handle:
        handle.write(ttir);handle.flush()
        module=ir.parse_mlir_module(handle.name,context)
    module.context=context
    metadata={'hash':hashlib.sha256(ttir.encode()).hexdigest(),'target':target,**options.__dict__,
              'triton_version':triton.__version__}
    stages={};backend.add_stages(stages,options,Language.TRITON)
    source=module
    for name in ('ttir','ttgir','llir','ptx'):source=stages[name](source,metadata)
    if metadata.get('global_scratch_size',0) or metadata.get('profile_scratch_size',0):
        raise NotImplementedError('stock Triton kernel needs global/profile scratch; triton_kernel_call passes none')
    return source,int(metadata['shared']),metadata['name']


def _lowering(ctx, *in_nodes, jaxpr, interpret, debug, input_output_aliases, grid_mapping, mesh,
              compiler_params, cost_estimate, out_avals, metadata, name):
    # Mirrors the non-default branch of JAX 0.10 pallas_call_lowering verbatim
    # except that a visible CUDA device is REQUIRED: the PTX is generated here
    # for that device's architecture (no silent sm_80 fallback).
    del interpret, out_avals, cost_estimate, name, metadata
    debug_info = jaxpr.debug_info
    if grid_mapping.num_dynamic_grid_bounds or grid_mapping.num_index_operands or mesh is not None:
        raise NotImplementedError('kernel-call route supports static grids without prefetch/mesh only')
    [platform] = ctx.platforms or ctx.module_context.platforms
    if platform != 'cuda':
        raise NotImplementedError(f'kernel-call route is CUDA only, got {platform!r}')
    try:
        gpu_device, *_ = jax.local_devices(backend='gpu')
    except RuntimeError as error:
        raise RuntimeError('kernel-call Triton route needs a visible CUDA device at lowering time') from error
    arch_name = str(gpu_device.compute_capability)
    compute_capability = int(arch_name.replace('.', ''))
    num_warps = 4 if compiler_params.num_warps is None else compiler_params.num_warps
    num_stages = 3 if compiler_params.num_stages is None else compiler_params.num_stages
    result = lowering.lower_jaxpr_to_triton_module(jaxpr, grid_mapping, platform, compute_capability)
    module_op = result.module.operation
    if debug:
        print(module_op.get_asm(enable_debug_info=True, pretty_debug_info=True))
    grid_x, grid_y, grid_z = normalize_grid(result.grid)
    ttir = str(result.module)
    ptx, shared, ptx_name = stock_compile(ttir, compute_capability, num_warps, num_stages,
                                         bool(getattr(compiler_params, 'enable_fp_fusion', True)),
                                         stock_compiler_identity())
    kernel = triton_kernel_call_lib.TritonKernel(ptx_name, num_warps, 1, shared, ptx, ttir, compute_capability)
    call = triton_kernel_call_lib.TritonKernelCall(
        kernel, grid_x, grid_y, grid_z,
        [triton_kernel_call_lib.create_array_parameter(0, 16)]*(len(ctx.avals_in)+len(ctx.avals_out)))
    return mlir.custom_call(
        call_target_name='triton_kernel_call',
        result_types=mlir.flatten_ir_types(map(mlir.aval_to_ir_type, ctx.avals_out)),
        operands=in_nodes,
        backend_config=zlib.compress(call.to_proto(debug_info.func_name, b'')),
        operand_layouts=avals_to_layouts(ctx.avals_in),
        result_layouts=avals_to_layouts(ctx.avals_out),
        operand_output_aliases=dict(input_output_aliases),
    ).results


pallas_core.register_lowering_rule(KernelCallCompilerParams, _lowering, 'gpu')
