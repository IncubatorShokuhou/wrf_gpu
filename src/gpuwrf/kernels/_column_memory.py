"""Typed global memory for the NTU mp56 / Jensen mp55 per-column kernels.

Triton 3.6 TritonGPUCoalesce walks the complete unrolled column DAG from every
tensor-of-pointer memory operation (FINDINGS E157; MP52/MP53/QNSE precedent).
Emitting these level accesses and table gathers as typed scalar PTX global
instructions keeps that analysis away from the kernels without changing their
arithmetic, buffers, launch geometry or the shared compiler pipeline.
Pallas reference effects and CPU discharge rules are the stock ones, so
interpretation is unchanged. Loads keep their masked fills; vector outputs use
atomic exchanges so replicated lanes of sliced layouts never race. Gather
indices follow the interpretation's wrap-then-clamp rule (see _in_bounds).
"""
import dataclasses

import jax.numpy as jnp
from jax import tree_util
from jax.extend import core
from jax._src.state import discharge, primitives as sp
from jax._src.pallas import primitives as pp
from jax._src.state import indexing
from jax._src.pallas.triton import lowering as tl
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.triton import dialect as tt


_load = core.Primitive('column_global_load')
_load.def_effectful_abstract_eval(pp._load_abstract_eval)
discharge.register_discharge_rule(_load)(pp._load_discharge_rule)

_store = core.Primitive('column_global_store')
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


def _in_bounds(index, dim):
    """Pallas interpretation's jnp gather index semantics, emitted for Triton.

    The interpreted kernel (the CPU reference) gathers with ``ref[indices]``:
    negative indices wrap once, then every index is clamped into [0, dim-1].
    Raw Triton addresses have no such rule, so lanes whose results the column
    code later discards (inactive levels/categories, padded levels) could read
    outside the buffer. Applying the identical rule keeps GPU == interpretation.
    """
    if isinstance(index, indexing.Slice) or not isinstance(index, ir.Value):
        return index
    zero, size, top = (tl._full(index.type, v) for v in (0, dim, dim - 1))
    negative = arith.cmpi(arith.CmpIPredicate.slt, index, zero)
    wrapped = arith.select(negative, arith.addi(index, size), index)
    return arith.minsi(arith.maxsi(wrapped, zero), top)


def _address(ctx, ptr, indexers):
    if len(indexers) != 1:
        raise NotImplementedError('column memory requires one explicit indexer')
    idx = indexers[0]
    if not all(isinstance(b, int) for b in ctx.block_infos[0].block_shape):
        raise NotImplementedError('column memory requires unblocked full-array refs')
    idx = dataclasses.replace(idx, indices=tuple(
        _in_bounds(i, d) for i, d in zip(idx.indices, idx.shape)))
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
    # Uniform scalars (column-invariant table entries, dt, surface sums, and
    # NTU's DOUBLE table reads) keep the stock load: Coalesce skips scalars.
    if not ctx.avals_out[0].shape:
        return tl._masked_load_lowering_rule(
            ctx, *args, args_tree=args_tree, eviction_policy=None,
            cache_modifier=None, is_volatile=False)
    if ctx.avals_out[0].dtype != jnp.float32:
        raise TypeError('column memory vector loads are float32')
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
    if value_aval.dtype != jnp.float32:
        raise TypeError('column memory outputs are float32')
    if not value_aval.shape:
        # Scalar surface sums keep Triton's single-writer store.
        idx = indexers[0]
        ptr = tl._compute_pointers_from_indices(ptr, ctx.block_infos[0], idx)
        pred = None if mask is None else tl._ensure_ir_value(mask, mask_aval)
        tl._store(ptr, tl._ensure_ir_value(value, value_aval), mask=pred)
        return []
    address, shape = _address(ctx, ptr, indexers)
    i32 = ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(32))
    pred = (tl._full(i32, 1) if mask is None else
            tl._ir_cast(_operand(mask, mask_aval, shape), i32, signed=False))
    value = _operand(value, value_aval, shape)
    # Sliced encodings can replicate lanes; exchange keeps identical writes
    # race-free. The old bits are never consumed.
    tt.elementwise_inline_asm(
        [value.type], '{ .reg .pred p; setp.ne.u32 p, $3, 0; '
        'mov.b32 $0, $2; @p atom.global.exch.b32 $0, [$1], $2; }',
        '=f,l,f,r', False, 1, [address, value, pred])
    return []
