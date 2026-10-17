"""P3 Registry carries, root boundaries, and E78 lossless continuation."""
from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import P3_STATE_LEAVES
from gpuwrf.contracts.ice_diagnostics import ICE_DIAGNOSTIC_LEAVES
from gpuwrf.contracts.state import State, conditional_state_leaves_for_mp
from gpuwrf.coupling.boundary_apply import BoundaryConfig, root_scalar_rk1_split
from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.io.restart_schema import validate_held_state
from gpuwrf.physics.p3.registry import SCHEMES, scalar_species, moist_species, flow_only_species
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_runtime_state
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state


@pytest.mark.parametrize('mp,count', [(50,4), (51,4), (52,8), (53,5)])
def test_p3_allocates_only_selected_real_carries(mp, count):
    state = _state(make_ideal_grid(4,3,5))
    assert all(getattr(state, name) is None for name in P3_STATE_LEAVES)
    # P3 keeps main's slots after the 105-field prefix as later lanes append.
    assert State.__slots__[105:105+len(P3_STATE_LEAVES)] == P3_STATE_LEAVES
    selected = state.ensure_conditional_leaves(mp_physics=mp)
    names = conditional_state_leaves_for_mp(mp)
    # Registry.EM_COMMON:1599-1610 diagnostics were appended by 31321cce1a.
    # Keep the original particle-carry count and audit the extra history leaves.
    diagnostics = {"d_ice", "rho_ice", "v_ice"}
    if mp == 52:
        diagnostics |= {"d_ice2", "rho_ice2", "v_ice2"}
    assert set(names) - set(P3_STATE_LEAVES) == diagnostics
    assert len(names) == count + len(diagnostics)
    for name in (*P3_STATE_LEAVES, *ICE_DIAGNOSTIC_LEAVES):
        value = getattr(selected, name)
        if name in names:
            assert value.shape == state.qv.shape and value.dtype == jnp.float32
            assert np.count_nonzero(value) == 0
        else:
            assert value is None
    assert selected.qv is state.qv
    assert selected.re_cloud.dtype == selected.re_ice.dtype == jnp.float32
    assert selected.re_snow is None
    np.testing.assert_array_equal(np.asarray(selected.re_cloud),np.full(state.qv.shape,2.49e-6,np.float32))
    np.testing.assert_array_equal(np.asarray(selected.re_ice),np.full(state.qv.shape,4.99e-6,np.float32))


@pytest.mark.parametrize('mp', SCHEMES)
def test_p3_radiation_off_seeds_registry_zero_before_first_mp(mp):
    state=_state(make_ideal_grid(4,3,5)).ensure_conditional_leaves(mp_physics=mp,use_mp_re=0)
    assert np.count_nonzero(state.re_cloud)==np.count_nonzero(state.re_ice)==0
    assert state.re_snow is None


@pytest.mark.parametrize('mp', SCHEMES)
def test_transport_matches_original_registry_not_all_conditional_fields(mp):
    source = Path('data/wrf_pristine/WRF/Registry/Registry.EM_COMMON').read_text()
    package = next(line for line in source.splitlines() if re.search(rf'mp_physics=={mp}\s', line))
    moist = re.search(r'moist:([^;]+)', package)[1].split(',')
    scalar = re.search(r'scalar:([^;]+)', package)[1].split(',')
    translation = {'qni':'Ni', 'qnr':'Nr', 'qnc':'Nc', 'qni2':'Ni2'}
    scalar = [translation.get(name, name) for name in scalar]
    nl = SimpleNamespace(mp_physics=mp)
    assert set(op._advected_scalar_species(nl)) == set(moist) == set(moist_species(mp))
    assert set(op._nested_number_scalar_species(nl)) == set(scalar) == set(scalar_species(mp))
    assert set(op._diffused_scalar_species(nl)) == set(moist+scalar)
    assert 'mp_th_old' not in moist+scalar and 'mp_qv_old' not in moist+scalar
    assert len(set(moist+scalar)) == len(moist+scalar)


@pytest.mark.parametrize('mp', SCHEMES)
def test_root_flow_flags_keep_second_ice_mass_in_moist(mp):
    species = moist_species(mp) + scalar_species(mp)
    flags = BoundaryConfig(have_bcs_moist=False, have_bcs_scalar=False)
    relaxed, flow = root_scalar_rk1_split(flags, species, flow_only=flow_only_species(mp))
    assert relaxed == ('qv',)
    assert set(flow) == set(species)-{'qv'}
    with pytest.raises(NotImplementedError, match='have_bcs_scalar'):
        root_scalar_rk1_split(replace(flags, have_bcs_scalar=True), ('qir',), flow_only=('qir',))
    if mp == 52:
        with pytest.raises(NotImplementedError, match='have_bcs_moist'):
            root_scalar_rk1_split(replace(flags, have_bcs_moist=True), ('qi2',), flow_only=('qi2',))


@pytest.mark.parametrize('mp', SCHEMES)
@pytest.mark.parametrize('fmt', ['pickle','netcdf','checkpoint'])
def test_restart_all_selected_particle_and_history_bits(mp, fmt, tmp_path):
    grid = make_ideal_grid(4,3,5)
    state = _state(grid).ensure_conditional_leaves(mp_physics=mp)
    names = conditional_state_leaves_for_mp(mp) + ('re_cloud','re_ice')
    state = state.replace(**{name: jnp.arange(state.qv.size, dtype=jnp.float32).reshape(state.qv.shape)/2**20+(i+1)/2**12
                              for i,name in enumerate(names)})
    nl = op.OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, mp_physics=mp)
    radar=jnp.arange(state.qv.size,dtype=jnp.float32).reshape(state.qv.shape)/8-99
    carry = initial_operational_carry(state).replace(radar_reflectivity=radar)
    path = tmp_path/f'{mp}.{fmt}'
    if fmt == 'pickle':
        rp.write_restart(carry, nl, grid, 2, path)
        resumed = rp.read_restart(path)[0]
    elif fmt == 'netcdf':
        nc.write_wrfrst_carry(carry, grid, nl, path, valid_time='2026-06-03_00:00:36',
                              run_start='2026-06-03_00:00:00', step_index=2)
        resumed = nc.read_wrfrst_carry(path)[0]
    else:
        write_checkpoint(state, nl, grid, 2, path, runtime_state=carry)
        resumed = read_checkpoint_with_runtime_state(path)[4]
    assert np.asarray(resumed.radar_reflectivity).tobytes()==np.asarray(radar).tobytes()
    for name in names:
        before, after = np.asarray(getattr(state,name)), np.asarray(getattr(resumed.state,name))
        assert before.dtype == after.dtype == np.float32
        assert before.tobytes() == after.tobytes()
    with pytest.raises(ValueError, match='E78'):
        validate_held_state({name: getattr(state.replace(qib=None),name) for name in State.__slots__}, nl)
