"""QNSE memory effects, masked fill and typed native lowering controls."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import inspect
import json
from pathlib import Path
from jax.experimental import pallas as pl
from jax._src.pallas.triton import lowering

from gpuwrf.kernels._qnse_memory import load, store


def program(dtype, interpret):
    def kernel(x, y):
        index = jnp.arange(16)
        value = load(x.at[index], mask=index < 13, other=1.)
        store(y.at[index], value.astype(dtype), mask=index < 15)

    return pl.pallas_call(kernel, grid=(1,), interpret=interpret,
                          out_shape=jax.ShapeDtypeStruct((15,), dtype))


@pytest.mark.parametrize('dtype', (jnp.float32, jnp.int32))
def test_masked_interpreter_preserves_reference_bits(dtype):
    values = np.arange(-7, 6, dtype=np.float32)
    actual = np.asarray(jax.jit(program(dtype, True))(values))
    expected = np.concatenate((values, np.ones(2, np.float32))).astype(dtype)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('dtype', (jnp.float32, jnp.int32))
def test_native_memory_typed_under_x64(dtype):
    with jax.enable_x64(True):
        jp = jax.make_jaxpr(program(dtype, False))(jax.ShapeDtypeStruct((13,), jnp.float32))
        params = next(e.params for e in jp.jaxpr.eqns if e.primitive.name == 'pallas_call')
        module = lowering.lower_jaxpr_to_triton_module(
            params['jaxpr'], params['grid_mapping'], 'cuda', 120)
    raw = str(module.module)
    memory = [line for line in raw.splitlines() if 'elementwise_inline_asm' in line]
    assert any('ld.global.b32' in line for line in memory)
    assert any('atom.global.exch.b32' in line for line in memory)
    assert all('xf64>' not in line for line in memory)
    if dtype == jnp.int32:
        assert any('=r,l,r,r' in line for line in memory)


@pytest.mark.parametrize('scheme', ('sfc', 'pbl', 'mf'))
def test_actual_qnse_native_uses_typed_memory(scheme, monkeypatch):
    import gpuwrf.backends.pallas as facade
    monkeypatch.setattr(facade, 'cuda_native', lambda: True)
    monkeypatch.setattr(facade, 'pallas_call', pl.pallas_call)
    from gpuwrf.kernels.phys_qnse_columns import qnse_columns
    from gpuwrf.physics.pbl_qnse import qnsepbl_column
    from gpuwrf.physics.sfclay_qnse import qnsesfc_column
    from gpuwrf.physics.qnse_edmf import mfshconvpbl_column
    root = Path(__file__).resolve().parents[1] / 'proofs/s2_qnse'
    fixture = root / ('mf_oracle.json' if scheme == 'mf' else 'oracle.json')
    case = json.loads(fixture.read_text())['cases']['fp32'][0]
    section = {'sfc': 'INPUT_SFC', 'pbl': 'INPUT', 'mf': 'INPUT_MF'}[scheme]
    fn = {'sfc': qnsesfc_column, 'pbl': qnsepbl_column, 'mf': mfshconvpbl_column}[scheme]
    fields = {key: jax.ShapeDtypeStruct((*np.shape(value), 15), jnp.float32)
              for key, value in case[section].items() if key in inspect.signature(fn).parameters}
    jp = jax.make_jaxpr(lambda x: qnse_columns(
        scheme, x, interpret=False, block=16))(fields)
    params = next(e.params for e in jp.jaxpr.eqns if e.primitive.name == 'pallas_call')
    raw = str(lowering.lower_jaxpr_to_triton_module(
        params['jaxpr'], params['grid_mapping'], 'cuda', 120).module)
    assert 'ld.global.b32' in raw, 'QNSE production loads lost typed lowering'
    assert 'atom.global.exch.b32' in raw, 'QNSE production stores lost typed lowering'
    assert 'tt.load ' not in raw and 'tt.store ' not in raw
