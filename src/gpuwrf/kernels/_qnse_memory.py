"""Typed QNSE memory with stock Pallas reference effects and interpretation.

Triton 3.6 Coalesce walks the complete unrolled vertical dependency graph
from tensor memory operations. Scoped typed PTX keeps that analysis away
from QNSE without changing its equations, launch count or column layout.
The implementation follows the independently qualified MP52 workaround.
"""
import jax.numpy as jnp
from jax import tree_util
from jax.extend import core
from jax._src.state import discharge, primitives as sp
from jax._src.pallas import primitives as pp
from jax._src.pallas.triton import lowering as tl
from jaxlib.mlir import ir
from jaxlib.triton import dialect as tt


_load = core.Primitive('qnse_global_load')
_load.def_effectful_abstract_eval(pp._load_abstract_eval)
discharge.register_discharge_rule(_load)(pp._load_discharge_rule)

_store = core.Primitive('qnse_global_store')
_store.multiple_results = True


@_store.def_effectful_abstract_eval
def _store_abstract(*avals, args_tree):
    _, effects = pp._swap_abstract_eval(*avals, args_tree=args_tree)
    return [], effects


@discharge.register_discharge_rule(_store)
def _store_discharge(in_avals, out_avals, *args, args_tree):
    updated, _ = pp._swap_discharge_rule(in_avals, out_avals, *args, args_tree=args_tree)
    return updated, []


def load(view, *, mask=None, other=None):
    ref, transforms = sp.get_ref_and_transforms(view, None, 'load')
    args, tree = tree_util.tree_flatten((ref, transforms, mask, other))
    return _load.bind(*args, args_tree=tree)


def store(view, value, *, mask=None):
    ref, transforms = sp.get_ref_and_transforms(view, None, 'store')
    args, tree = tree_util.tree_flatten((ref, transforms, value, mask))
    _store.bind(*args, args_tree=tree)


def _address(ctx, ptr, indexers):
    if len(indexers) != 1:
        raise NotImplementedError('QNSE memory requires one explicit indexer')
    idx = indexers[0]
    shape = idx.get_indexer_shape_static()
    ptr = tl._compute_pointers_from_indices(ptr, ctx.block_infos[0], idx)
    i64 = ir.IntegerType.get_signless(64)
    address_type = ir.RankedTensorType.get(shape, i64) if shape else i64
    return tt.ptr_to_int(address_type, ptr), shape


def _operand(value, aval, shape):
    return tl._bcast_to(tl._ensure_ir_value(value, aval), shape)


@tl.register_lowering(_load)
def _load_triton(ctx, *args, args_tree):
    ptr, indexers, mask, other = args_tree.unflatten(args)
    if not ctx.avals_out[0].shape:
        return tl._masked_load_lowering_rule(
            ctx, *args, args_tree=args_tree, eviction_policy=None,
            cache_modifier=None, is_volatile=False)
    if ctx.avals_out[0].dtype != jnp.float32:
        raise TypeError('QNSE vector inputs are float32')
    address, shape = _address(ctx, ptr, indexers)
    *_, mask_aval, other_aval = args_tree.unflatten(ctx.avals_in)
    f32 = ir.RankedTensorType.get(shape, ir.F32Type.get())
    i32 = ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(32))
    pred = (tl._full(i32, 1) if mask is None else
            tl._ir_cast(_operand(mask, mask_aval, shape), i32, signed=False))
    fallback = (tl._full(f32, 0.) if other is None else
                tl._ir_cast(_operand(other, other_aval, shape), f32, signed=False))
    return tt.elementwise_inline_asm(
        [f32], '{ .reg .pred p; setp.ne.u32 p, $2, 0; '
        'mov.b32 $0, $3; @p ld.global.b32 $0, [$1]; }',
        '=f,l,r,f', False, 1, [address, pred, fallback])


@tl.register_lowering(_store)
def _store_triton(ctx, *args, args_tree):
    ptr, indexers, value, mask = args_tree.unflatten(args)
    *_, value_aval, mask_aval = args_tree.unflatten(ctx.avals_in)
    if value_aval.dtype not in (jnp.float32, jnp.int32):
        raise TypeError('QNSE outputs are float32 or int32 KPBL')
    address, shape = _address(ctx, ptr, indexers)
    i32 = ir.IntegerType.get_signless(32)
    if shape:
        i32 = ir.RankedTensorType.get(shape, i32)
    pred = (tl._full(i32, 1) if mask is None else
            tl._ir_cast(_operand(mask, mask_aval, shape), i32, signed=False))
    value = _operand(value, value_aval, shape)
    # Preserve bits for integer KPBL as well as REAL outputs. Replicated
    # lanes can arise from Triton's layout; exchange avoids racing writes.
    constraints = '=f,l,f,r' if value_aval.dtype == jnp.float32 else '=r,l,r,r'
    tt.elementwise_inline_asm(
        [value.type], '{ .reg .pred p; setp.ne.u32 p, $3, 0; '
        'mov.b32 $0, $2; @p atom.global.exch.b32 $0, [$1], $2; }',
        constraints, False, 1, [address, value, pred])
    return []
