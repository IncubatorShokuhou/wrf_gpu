"""KF ALIQ/GDRY: exact WRF REAL statements in the native release consumer."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_kf_column as K
from gpuwrf.physics import cumulus_kf as retained

FLAG = "GPUWRF_KF_REAL_CONSTANTS"
ROOT = Path(__file__).resolve().parents[3]
# Pristine module_cu_kfeta.F:720/730 with REAL operands from
# module_model_constants: SVP1=.6112, G=9.81, CP=7*287/2.
WRF_BITS = (1142475980, 3156214220)
PORT_BITS = (1142475981, 3156214219)


def _bits(values):
    a = np.asarray(values, np.float32)
    assert np.isfinite(a).all()
    return tuple(int(x) for x in a.view(np.uint32).reshape(-1))


def _fixture():
    d = json.loads((ROOT / "proofs/v060/savepoints/kf_case_1.json").read_text())
    return tuple(jnp.asarray(d["columns"][name], jnp.float32)
                 for name in ("T", "QV", "P", "DZ", "RHO", "W0AVG", "U", "V"))


def _literals(jaxpr):
    if isinstance(jaxpr, jax.extend.core.ClosedJaxpr):
        yield from _literals(jaxpr.jaxpr)
    elif isinstance(jaxpr, jax.extend.core.Jaxpr):
        for eqn in jaxpr.eqns:
            for arg in eqn.invars:
                if isinstance(arg, jax.extend.core.Literal):
                    a = np.asarray(arg.val)
                    # Pallas uses NaN literals to initialize output refs; inspect
                    # finite computation literals, not that internal sentinel.
                    if a.dtype == np.float32 and np.isfinite(a).all():
                        yield from _bits(a)
            for value in eqn.params.values():
                yield from _literals(value)
    elif isinstance(jaxpr, (tuple, list)):
        for value in jaxpr:
            yield from _literals(value)


@pytest.mark.parametrize("enabled,expected", [(False, PORT_BITS), (True, WRF_BITS)])
def test_runtime_and_compiled_constants_have_exact_source_bits(monkeypatch, enabled, expected):
    monkeypatch.setenv(FLAG, str(int(enabled)))
    assert _bits(K._runtime_constants()) == expected
    # Compiled evaluation cannot silently replace the REAL operand rounding
    # with the old fp64-expression-then-cast value.
    read = jax.jit(lambda: jnp.asarray(K._runtime_constants(), jnp.float32))
    assert _bits(read()) == expected


@pytest.mark.parametrize("enabled,expected", [(False, PORT_BITS), (True, WRF_BITS)])
def test_native_column_consumes_the_selected_literals(monkeypatch, enabled, expected):
    monkeypatch.setenv(FLAG, str(int(enabled)))
    closed = jax.make_jaxpr(lambda *a: K.kf_column(*a, 54., 9000., 40, interpret=True))(*_fixture())
    kernel = next(e.params["jaxpr"] for e in closed.jaxpr.eqns if str(e.primitive) == "pallas_call")
    consumed = set(_literals(kernel))
    assert set(expected).issubset(consumed)
    assert not set(WRF_BITS if not enabled else PORT_BITS).intersection(consumed)


def test_flag_is_a_trace_key_and_does_not_change_retained_constants(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
    before = tuple(np.float64(v).view(np.uint64) for v in (K.ALIQ, K.GDRY, retained.ALIQ, retained.GDRY))
    monkeypatch.setenv(FLAG, "0")
    off = global_trace_env_hash()
    monkeypatch.setenv(FLAG, "1")
    assert global_trace_env_hash() != off
    after = tuple(np.float64(v).view(np.uint64) for v in (K.ALIQ, K.GDRY, retained.ALIQ, retained.GDRY))
    assert before == after


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_exact_bit_gate_refuses_nonfinite(bad):
    with pytest.raises(AssertionError):
        _bits([bad, -0.009])
