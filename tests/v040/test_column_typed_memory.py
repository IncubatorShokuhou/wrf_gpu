"""Typed global memory of the NTU mp56 / Jensen mp55 column kernels (t-ntucompile).

Triton 3.6 Coalesce stalled >250 s on each kernel while any tensor-of-pointer
load/store remained (FINDINGS E157). The typed primitives must (1) interpret
exactly like stock Pallas loads/stores and (2) leave no tensor memory op in the
CUDA-target TTIR of either production kernel. CPU only; no CUDA context.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels import _column_memory as cm


def _copy_gather(load, store):
    def kernel(x, table, ox, og, osum):
        k = jnp.arange(8, dtype=jnp.int32)
        valid = k < 5
        a = load(x.at[k], mask=valid, other=np.float32(-7.))
        index = jnp.asarray(jnp.abs(a), jnp.int32) & 3
        g = load(table.at[index])                      # vector table gather
        store(ox.at[k], a * np.float32(2), mask=valid)
        store(og.at[k], g + a)
        store(osum.at[0], jnp.sum(g))                  # scalar store (stock path)
    return kernel


def _run(load, store):
    x = np.arange(8, dtype=np.float32) * np.float32(1.25) - np.float32(3)
    table = np.asarray([.1, .2, .3, .4], np.float32)
    shapes = (jax.ShapeDtypeStruct((8,), jnp.float32),) * 2 + (jax.ShapeDtypeStruct((1,), jnp.float32),)
    fn = lambda *a: pl.pallas_call(_copy_gather(load, store), out_shape=shapes, interpret=True)(*a)
    text = jax.jit(fn).lower(x, table).as_text()
    out = [np.asarray(v) for v in fn(x, table)]
    return text, out


def test_typed_memory_interprets_exactly_like_stock_pallas():
    stock_text, stock = _run(plt.load, plt.store)
    typed_text, typed = _run(cm.load, cm.store)
    assert typed_text == stock_text            # identical discharged CPU program
    for a, b in zip(stock, typed):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _production_ttir(scheme, monkeypatch):
    import gpuwrf.backends.pallas as facade
    from jax._src.pallas.triton import lowering
    monkeypatch.setattr(facade, 'cuda_native', lambda: True, raising=False)
    monkeypatch.setattr(facade, 'pallas_call', pl.pallas_call, raising=False)
    nz = 44
    s = jax.ShapeDtypeStruct((1, nz), jnp.float32)
    if scheme == 'ntu':
        from gpuwrf.kernels.phys_ntu import full_column
        from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES
        from gpuwrf.physics.ntu.column import PROGNOSTICS
        fields = {n: s for n in ('th', 'p', 'pii', 'dz', *PROGNOSTICS)}
        fields['w'] = jax.ShapeDtypeStruct((1, nz + 1), jnp.float32)
        fields.update({n: jax.ShapeDtypeStruct((1,), jnp.float32) for n in ('rainnc', 'snownc', 'grapnc', 'hailnc')})
        fn, args = (lambda x, d: full_column(x, d, output_names=OUTPUT_NAMES)), (fields, jax.ShapeDtypeStruct((), jnp.float32))
    else:
        from gpuwrf.kernels.phys_jensen import full_column
        from gpuwrf.physics.jensen_ishmael.registry import output_mapping
        from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping
        mapping = output_mapping()
        fields = {n: s for n in [n for n in mapping if n not in ('effc', 'effi')] + ['th', 'p', 'dz']}
        mapping.update(ice_diagnostic_mapping(55))
        outs = (*mapping, 'th')
        fn = lambda x, d, i: full_column(x, d, i, output_names=outs)
        args = (fields, jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((), jnp.int32))
    with jax.enable_x64(True):
        program = jax.make_jaxpr(fn)(*args).jaxpr
    params = next(e.params for e in program.eqns if e.primitive.name == 'pallas_call')
    return str(lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', 120).module)


@pytest.mark.parametrize('scheme', ['ntu', 'jensen'])
def test_production_kernels_have_no_tensor_memory_ops(scheme, monkeypatch):
    ttir = _production_ttir(scheme, monkeypatch)
    tensor_memory = re.findall(r'tt\.(?:load|store)[^\n]*tensor<[^\n]*!tt\.ptr', ttir)
    assert tensor_memory == []                  # Coalesce walks from each of these (E157)
    typed = ttir.count('ld.global')
    assert typed >= (50 if scheme == 'ntu' else 600)
    assert 'atom.global.exch.b32' in ttir


@pytest.mark.parametrize('scheme', ['ntu', 'jensen'])
def test_gather_indices_follow_interpretation_bounds(scheme, monkeypatch):
    # Interpretation gathers with jnp semantics (wrap negatives, clamp into the
    # table). The Triton lowering must apply the same rule to every typed
    # access; an unclamped gather read outside its table on the GPU (c87).
    ttir = _production_ttir(scheme, monkeypatch)
    typed = ttir.count('ld.global')
    assert ttir.count('arith.minsi') >= typed and ttir.count('arith.maxsi') >= typed
