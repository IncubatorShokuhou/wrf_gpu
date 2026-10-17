"""Deletion-sensitive WRF nest-start inheritance/input precedence for extra fields."""
from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.nesting.scalar_initialization import initialize_extra_scalar_fields, _VARIABLES
from gpuwrf.nesting.scalar_species import extra_boundary_species
from gpuwrf.validation.moving_nest_testbed import build_flat_grid
from test_s1_nestspec import weather, assert_real

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proofs/s1_nestspec'))


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    from pristine_nest_start import build
    return build(tmp_path_factory.mktemp('pristine_nest_start'))


@pytest.fixture(scope='module')
def states():
    pg = build_flat_grid(nx=32, ny=28, nz=3, dx_m=9000.)
    cg = build_flat_grid(nx=22, ny=19, nz=3, dx_m=3000.)
    return SimpleNamespace(state=weather(pg, 3), grid=pg), weather(cg, 1), cg


def run_for(mp, *, fields=None, from_file=True, stream=0):
    fields = fields or {}
    def load(domain, var, lazy=False):
        return fields[(domain, var)]
    return SimpleNamespace(namelist={'physics': {'mp_physics': [mp, mp]},
                'time_control': {'input_from_file': [True, from_file], 'fine_input_stream': [0, stream]}},
        load_wrfinput=load, grid=lambda domain: SimpleNamespace(parent_grid_ratio=3, i_parent_start=6, j_parent_start=7))


@pytest.mark.parametrize('mp', [18, 24, 26, 28, 40])
def test_missing_input_inherits_pristine_sint(states, library, mp):
    from pristine_nest_start import interpolate
    parent, child, grid = states
    result = initialize_extra_scalar_fields(run_for(mp), 'd02', child, grid, parent_case=parent)
    for name in extra_boundary_species(mp):
        expected = interpolate(library, np.asarray(getattr(parent.state, name)), child_shape=(19, 22))
        assert_real(getattr(result, name), expected)
        assert getattr(result, name).dtype == jnp.float32
        assert not np.array_equal(expected, np.asarray(getattr(child, name)))


@pytest.mark.parametrize('mp', [18, 24, 26, 28, 40])
def test_present_child_input_overrides_interpolation(states, monkeypatch, mp):
    import gpuwrf.nesting.scalar_initialization as init
    parent, child, grid = states
    names = extra_boundary_species(mp)
    fields = {('d02', 'QNDROP' if mp == 18 and n == 'Nc' else _VARIABLES[n]):
              np.full(child.qv.shape, i+11, np.float32) for i, n in enumerate(names)}
    def no_interpolation(*args, **kwargs):
        raise AssertionError('overwritten parent interpolation was unnecessarily computed')
    monkeypatch.setattr(init, 'interp_sint_full', no_interpolation)
    result = initialize_extra_scalar_fields(run_for(mp, fields=fields), 'd02', child, grid, parent_case=parent)
    for i, name in enumerate(names):
        np.testing.assert_array_equal(getattr(result, name), np.full(child.qv.shape, i+11, np.float32))


@pytest.mark.parametrize('from_file,stream', [(False, 0), (True, 1)])
def test_child_input_ignored_when_registry_stream_not_selected(states, library, from_file, stream):
    from pristine_nest_start import interpolate
    parent, child, grid = states
    fields = {('d02', 'QHAIL'): np.full(child.qv.shape, 99., np.float32)}
    result = initialize_extra_scalar_fields(run_for(24, fields=fields, from_file=from_file, stream=stream),
                                            'd02', child, grid, parent_case=parent)
    expected = interpolate(library, np.asarray(parent.state.qh), child_shape=(19, 22))
    assert_real(result.qh, expected)


def test_default_release_initialization_is_identity(states):
    _, child, grid = states
    assert initialize_extra_scalar_fields(run_for(8), 'd01', child, grid) is child


def test_omitted_child_input_flag_keeps_registry_false(states, library):
    from pristine_nest_start import interpolate
    parent, child, grid = states
    run = run_for(24, fields={('d02', 'QHAIL'): np.full(child.qv.shape, 99., np.float32)})
    run.namelist['time_control']['input_from_file'] = [True]
    result = initialize_extra_scalar_fields(run, 'd02', child, grid, parent_case=parent)
    expected = interpolate(library, np.asarray(parent.state.qh), child_shape=(19, 22))
    assert_real(result.qh, expected)


def test_nssl_cloud_number_uses_registry_qndrop_not_qncloud(states):
    _, child, grid = states
    fields = {('d01', 'QNDROP'): np.full(child.qv.shape, 17., np.float32),
              ('d01', 'QNCLOUD'): np.full(child.qv.shape, 93., np.float32)}
    result = initialize_extra_scalar_fields(run_for(18, fields=fields), 'd01', child, grid)
    np.testing.assert_array_equal(result.Nc, np.full(child.qv.shape, 17., np.float32))


def test_root_extra_mixing_and_child_number_loop_cover_nssl(states):
    from gpuwrf.runtime.operational_mode import _mixing_scalar_species, _nested_number_scalar_species
    parent, _, _ = states
    nml = SimpleNamespace(mp_physics=18, moist_adv_opt=1, scalar_adv_opt=1)
    assert set(extra_boundary_species(18)) <= set(_mixing_scalar_species(parent.state, nml))
    assert _nested_number_scalar_species(nml) == ('Ni', 'Nr', 'Nc', 'Ns', 'Ng', 'Nh', 'Nn')
    assert _nested_number_scalar_species(SimpleNamespace(mp_physics=8)) == ('Ni', 'Nr')


@pytest.mark.parametrize('mp', [10, 14, 16, 26])
def test_number_lane_composes_without_second_wdm_transport(states, mp):
    from gpuwrf.runtime.operational_mode import (
        _mixing_scalar_species, _nested_number_scalar_species, _diffused_scalar_species,
    )
    parent, _, _ = states
    nml = SimpleNamespace(mp_physics=mp)
    assert _mixing_scalar_species(parent.state,nml) == _diffused_scalar_species(nml)
    assert _nested_number_scalar_species(nml) == (() if mp in (14,16,26) else ('Ni','Nr'))


def test_child_inherits_initialized_parent_aerosols(states, library):
    from pristine_nest_start import interpolate
    parent, child, grid = states
    # Distinct raw and initialized values make the ordering deletion-sensitive.
    initialized = parent.state.replace(nwfa=parent.state.nwfa*3, nifa=parent.state.nifa*5)
    result = initialize_extra_scalar_fields(run_for(28), 'd02', child, grid,
                                            parent_case=parent, parent_state=initialized)
    for name in ('nwfa', 'nifa'):
        expected = interpolate(library, np.asarray(getattr(initialized,name)), child_shape=(19,22))
        assert_real(getattr(result,name), expected)
        raw = interpolate(library, np.asarray(getattr(parent.state,name)), child_shape=(19,22))
        assert not np.array_equal(np.asarray(getattr(result,name)), raw)
