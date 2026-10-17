"""Actual-provider cache collision witnesses; no forecast-fidelity claim."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck, aot_executable as ax


def _nl(option):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.,
        mp_physics=option, ra_sw_physics=0, ra_lw_physics=0,
        sf_sfclay_physics=0, bl_pbl_physics=0, cu_physics=0)


def _snapshot(fn, operand, nl):
    identity = ck.fn_identity_hash(fn)
    key = ck.cheap_key(fn, (operand, nl), {}, nl)
    assert key is not None
    lowered = fn.lower(operand, nl)
    output = np.asarray(lowered.compile()(operand, nl))
    assert np.isfinite(output).all()
    return identity, key, ax.hlo_sha256_from_lowered(lowered), output


def _different(before, after):
    assert before[0] == after[0], "the witness must change provider data only"
    assert before[1] != after[1], "different actual provider values shared a full key"
    assert before[2] != after[2], "the provider mutation did not change HLO"
    assert not np.array_equal(before[3], after[3]), "the mutation did not change output"


@pytest.mark.parametrize("option,module_name,getter", [
    (5, "ferrier.native", "host_tables"),
    (15, "ferrier.native", "host_tables"),
    (38, "thompson_hail.native", "_cloud_coefficients"),
    (38, "thompson_hail.native", "_freezing_tables"),
    (38, "thompson_hail.context", "collision_tables"),
])
def test_cached_npz_provider_changes_full_key_hlo_output(monkeypatch, option, module_name, getter):
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    module = importlib.import_module("gpuwrf.physics." + module_name)
    original = getattr(module, getter)()
    field = sorted(original)[0]
    table = original[field].copy()
    table.flat[0] += table.dtype.type(1)
    changed = dict(original, **{field: table})
    nl, operand = _nl(option), jnp.asarray(0., table.dtype)

    @jax.jit
    def fn(x, config):
        # Read the actual host provider, then capture one scalar. No giant
        # executable/table copy is needed to demonstrate the cache collision.
        value = getattr(module, getter)()[field].reshape(-1)[0]
        return x + jnp.asarray(value, x.dtype)

    inactive = baked_table_hash(fn, replace(nl, mp_physics=0))
    before = _snapshot(fn, operand, nl)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, getter, lambda: changed)
            fn.clear_cache()
            after = _snapshot(fn, operand, nl)
            assert inactive == baked_table_hash(fn, replace(nl, mp_physics=0))
    finally:
        fn.clear_cache()
    _different(before, after)


@pytest.mark.parametrize("option,module_name,attribute", [
    (5, "ferrier.column", "MY600"),
    (38, "thompson_hail._coefficients", "AM_G"),
])
def test_static_physics_vectors_are_keyed(monkeypatch, option, module_name, attribute):
    module = importlib.import_module("gpuwrf.physics." + module_name)
    nl, operand = _nl(option), jnp.asarray(0., jnp.float32)
    @jax.jit
    def fn(x, config):
        return x + jnp.asarray(getattr(module, attribute)[0], x.dtype)
    before = _snapshot(fn, operand, nl)
    table = np.asarray(getattr(module, attribute)).copy()
    table[0] += np.float32(1)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, attribute, table)
            fn.clear_cache()
            after = _snapshot(fn, operand, nl)
    finally:
        fn.clear_cache()
    _different(before, after)


def test_goddard_actual_native_constant_table_is_keyed(monkeypatch):
    from gpuwrf.physics.goddard4ice import native
    nl, operand = _nl(7), jnp.asarray(0., jnp.float32)
    field = native.TABLES[0]
    @jax.jit
    def fn(x, config):
        return x + jnp.asarray(native.C[field][0], x.dtype)
    before = _snapshot(fn, operand, nl)
    table = native.C[field].copy()
    table[0] += np.float32(1)
    try:
        with monkeypatch.context() as patch:
            patch.setitem(native.C, field, table)
            fn.clear_cache()
            after = _snapshot(fn, operand, nl)
    finally:
        fn.clear_cache()
    _different(before, after)


def test_private_hail_function_globals_are_keyed(monkeypatch):
    from gpuwrf.physics.thompson_hail import native
    nl, operand = _nl(38), jnp.asarray(.001, jnp.float32)
    @jax.jit
    def fn(x, config):
        return native.TC._ice_distribution(x, jnp.float32(1), jnp.float32(1), wrf_hail=True)[1]
    before = _snapshot(fn, operand, nl)
    env = native.TC._ice_distribution.__globals__
    try:
        with monkeypatch.context() as patch:
            patch.setitem(env, "CIE2", np.float32(5))
            fn.clear_cache()
            after = _snapshot(fn, operand, nl)
    finally:
        fn.clear_cache()
    _different(before, after)
