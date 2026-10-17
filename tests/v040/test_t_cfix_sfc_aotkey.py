"""t-cfix-sfc baked values reach the AOT/cheap key (rv-t-cfix SFC2 series 2, SS09).

_WAVE_S1_COMMON AOT/CACHE KEY RULE + t-aotkey 76a72bdd1: host values bound into a traced
program (REAL tables built at import, kind_phys literal namespaces) must reach the key through
runtime/aot_baked_tables.baked_table_hash, so a different table can never be served the
executable compiled for another one. Source hashing alone does not key the bound values.
"""
from __future__ import annotations

from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
from gpuwrf.runtime.aot_executable import hlo_sha256_from_lowered

MYJ = dict(sf_sfclay_physics=2, bl_pbl_physics=2)
GFS = dict(sf_sfclay_physics=3, bl_pbl_physics=3)
BAKED = [
    (MYJ, 'sfclay_janjic', '_PSIM_REAL_NP'), (MYJ, 'sfclay_janjic', '_PSIH_REAL_NP'),
    (MYJ, 'sfclay_janjic', '_ZTMAX_REAL'), (MYJ, 'sfclay_janjic', '_DZETA_REAL'),
    (dict(bl_pbl_physics=2), 'myj_constants', 'SFC_ZTMIN1'),
    (dict(sf_sfclay_physics=91), 'sfclay_old_mm5', '_PSIMTB_REAL'),
    (dict(sf_sfclay_physics=91), 'sfclay_old_mm5', '_PSIHTB_REAL'),
    (GFS, 'bl_gfs', 'WRF_MONINP'), (GFS, 'bl_gfs', 'WRF_G'), (GFS, 'bl_gfs', 'WRF_RCP'),
]


def _namelist(**kwargs):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=0, ra_sw_physics=0,
                               ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0, **kwargs)


def _bump(value):
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(value[0] + 1.0, *value[1:])
    if isinstance(value, np.ndarray):
        changed = value.copy()
        changed.flat[0] += changed.dtype.type(1)
        return changed
    return type(value)(value + type(value)(1))


@pytest.mark.parametrize('selection,module,attr', BAKED, ids=[f'{m}.{a}' for _, m, a in BAKED])
def test_each_new_baked_value_changes_its_active_table_key_only(monkeypatch, selection, module, attr):
    target = importlib.import_module('gpuwrf.physics.' + module)
    nl = replace(_namelist(), **selection)

    def fn(x, nl):
        return x
    before, inactive = baked_table_hash(fn, nl), baked_table_hash(fn, _namelist())
    assert before != inactive
    with monkeypatch.context() as patch:
        patch.setattr(target, attr, _bump(getattr(target, attr)))
        assert before != baked_table_hash(fn, nl), (module, attr)
        assert inactive == baked_table_hash(fn, _namelist()), (module, attr)
    assert before == baked_table_hash(fn, nl)


def _consumer(family):
    """A tiny ACTUAL consumer of the bound value, and the attribute to change."""
    if family == 'myj':  # rv-t-cfix SS09: the MYJSFC REAL psi tables
        from gpuwrf.physics import sfclay_janjic as module
        return module, '_PSIM_REAL_NP', lambda x: x * module._psi_tables(True)[0][0]
    if family == 'sf91':
        from gpuwrf.physics import sfclay_old_mm5 as module
        return module, '_PSIMTB_REAL', lambda x: x * jnp.asarray(module._PSIMTB_REAL, jnp.float32)[0]
    from gpuwrf.physics import bl_gfs as module
    return module, 'WRF_MONINP', lambda x: x * module.WRF_MONINP.CON_G


@pytest.mark.parametrize('family,selection', [('myj', MYJ), ('sf91', dict(sf_sfclay_physics=91)), ('bl3', GFS)])
def test_changed_table_changes_full_key_hlo_and_output(monkeypatch, family, selection):
    """Different table => different full cheap key, lowered HLO and result (SS09 collision closed)."""
    module, attr, body = _consumer(family)
    nl = replace(_namelist(), **selection)
    x = jnp.ones((2,), jnp.float32)
    fn = jax.jit(lambda x: body(x))

    def snapshot():
        key = ck.cheap_key(fn, (x,), {}, nl)
        assert key is not None
        lowered = fn.lower(x)
        value = np.asarray(lowered.compile()(x))
        assert np.isfinite(value).all()
        return key, hlo_sha256_from_lowered(lowered), value

    before = snapshot()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, attr, _bump(getattr(module, attr)))
            fn.clear_cache()
            after = snapshot()
    finally:
        fn.clear_cache()
    assert before[1] != after[1], (family, 'lowered HLO')
    assert not np.array_equal(before[2], after[2]), (family, 'actual output')
    assert before[0] != after[0], (family, 'full cheap key')
