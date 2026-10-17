"""Explicit f64 storage for legacy analytic gates registered before WRF REAL32.

Scope this fixture to a numerical test that requires it. Release/default-state
tests continue to exercise their real storage dtypes.
"""
import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.state import State, Tendencies


@pytest.fixture
def legacy_fp64_state(monkeypatch):
    def widen(value):
        return value.astype(jnp.float64) if jnp.issubdtype(value.dtype, jnp.floating) else value

    for contract in (State, Tendencies):
        original = contract.zeros.__func__

        def zeros(cls, *args, _original=original, **kwargs):
            return jax.tree.map(widen, _original(cls, *args, **kwargs))

        monkeypatch.setattr(contract, "zeros", classmethod(zeros))
    with jax.enable_x64():
        yield
