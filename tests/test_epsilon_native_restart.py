"""EEPS/KEPS E78 through the native synchronized RestartStore public API."""
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.contracts.precision import EPSILON_PBL_LEAVES
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
from gpuwrf.nesting.boundary_construction import initialize_child_scalar_boundaries
from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars
from gpuwrf.runtime import restart_store as storage
from gpuwrf.runtime.operational_mode import _initial_carry_for_run, _physics_step_forcing
from test_temf_native_restart import exact, rewrite, save
from test_v040_temf import _grid_state, _namelist


def advance(carry, ns, seconds):
    # Physics State and auxiliary carry are separate forcing outputs; the
    # production step composes both before returning its next OperationalCarry.
    forcing = _physics_step_forcing(carry, ns, seconds, run_radiation=False)
    return forcing.carry.replace(state=forcing.state)


@pytest.fixture(scope="module", params=[16, 17])
def epsilon_case(request):
    grid, state = _grid_state()
    ns = _namelist(grid, bl_pbl_physics=request.param, sf_sfclay_physics=1)
    carry = _initial_carry_for_run(state, ns)
    for step in range(3):
        carry = advance(carry, ns, 18.*step)
    assert int(carry.state.epsilon_pbl_step) == 3
    assert np.max(np.asarray(carry.state.epsilon_exch_m)) > 0
    return ns, carry


def identity(tmp_path, ns):
    cfg = NestedPipelineConfig(tmp_path, tmp_path/'out', tmp_path/'proof', hours=1, max_dom=1)
    value = storage.nested_identity(cfg, {'d01': SimpleNamespace(namelist=ns)},
                                    datetime(2024, 6, 1, 12, tzinfo=timezone.utc))
    assert value['epsilon_pbl_config']['d01']['bl_pbl_physics'] == ns.bl_pbl_physics
    assert value['epsilon_pbl_config']['d01']['domain_shape'] == [ns.grid.nz, ns.grid.ny, ns.grid.nx]
    return value


def test_native_epsilon_exact_and_evolved_continuation(epsilon_case, tmp_path, monkeypatch):
    from gpuwrf.io import pbl_epsilon_schema
    validate = pbl_epsilon_schema.validate_epsilon_state
    admissions = []
    def checked(state, namelist, version):
        assert all(isinstance(state[k], np.ndarray) for k in EPSILON_PBL_LEAVES if state.get(k) is not None)
        admissions.append(True)
        return validate(state, namelist, version)
    monkeypatch.setattr(pbl_epsilon_schema, 'validate_epsilon_state', checked)
    ns, carry = epsilon_case
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path/'store', 10**7, 2, 0)
    generation = save(store, carry, config)
    snapshot, _ = store.read(generation, expected_identity=config)
    restored = snapshot['carries']['d01']
    exact(carry, restored)
    assert len(admissions) == 4  # save, temporary/published readback, explicit read
    monkeypatch.setattr(pbl_epsilon_schema, 'validate_epsilon_state', validate)
    uninterrupted = advance(carry, ns, 54.)
    resumed = advance(restored, ns, 54.)
    exact(uninterrupted, resumed)


def test_epsilon_no_lsm_history_uses_actual_surface_call(epsilon_case, monkeypatch):
    from gpuwrf.coupling import epsilon_pbl
    ns, carry = epsilon_case
    original = epsilon_pbl.epsilon_surface_step
    packets=[]
    def observed(*a, **kw):
        result=original(*a, **kw)
        packets.append(result[2])
        return result
    monkeypatch.setattr(epsilon_pbl,'epsilon_surface_step',observed)
    next_carry=advance(carry,ns,54.)
    assert len(packets)==1
    for name in ('t2','q2','u10','v10','lh','psfc'):
        np.testing.assert_array_equal(getattr(next_carry.history_diagnostics,name),getattr(packets[0],name))


def damage(payload, kind):
    state = payload['state_fields']
    if kind == 'missing_all':
        for k in EPSILON_PBL_LEAVES:
            if k in state: state[k] = None
    elif kind == 'missing_counter': state['epsilon_pbl_step'] = None
    elif kind == 'missing_marker': payload.pop('epsilon_pbl_schema_version')
    elif kind == 'wrong_shape': state['epsilon_pbl'] = state['epsilon_pbl'][:1]
    elif kind == 'fp64': state['epsilon_pbl'] = state['epsilon_pbl'].astype(np.float64)
    elif kind == 'wrong_integer': state['epsilon_pbl_step'] = np.asarray(3., np.float32)
    elif kind == 'nonfinite': state['epsilon_pbl'] = np.full_like(state['epsilon_pbl'], np.inf)
    elif kind == 'missing_history': payload['extra_fields']['history_diagnostics'] = None
    elif kind in ('history_fp64','history_shape'):
        history=payload['extra_fields']['history_diagnostics']
        t2=history.t2.astype(np.float64) if kind=='history_fp64' else history.t2[:1]
        payload['extra_fields']['history_diagnostics']=history._replace(t2=t2)
    else: raise AssertionError(kind)


@pytest.mark.parametrize('operation', ['save', 'read'])
@pytest.mark.parametrize('kind', ['missing_all', 'missing_counter', 'missing_marker',
                                  'wrong_shape', 'fp64', 'wrong_integer', 'nonfinite', 'missing_history',
                                  'history_fp64','history_shape'])
def test_native_epsilon_refuses_invalid_host_carry(epsilon_case, tmp_path, monkeypatch, operation, kind):
    ns, carry = epsilon_case
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path/'store', 10**7, 2, 0)
    generation = save(store, carry, config)
    if operation == 'save':
        original = storage._carry_to_payload
        def malformed(value, **kwargs):  # forwards mynn_config (w44-budget)
            payload = original(value, **kwargs)
            damage(payload, kind)
            return payload
        monkeypatch.setattr(storage, '_carry_to_payload', malformed)
        with pytest.raises(ValueError, match='E78|nonfinite.*restart'):
            save(store, carry, config)
        assert not list(store.root.glob('.partial-*'))
    else:
        rewrite(generation, lambda payload: damage(payload, kind))
        def forbidden(_): raise AssertionError('device restore preceded E78 admission')
        monkeypatch.setattr(storage, '_restore_carry', forbidden)
        for device in (False, True):
            with pytest.raises(ValueError, match='E78|nonfinite.*restart'):
                store.read(generation, expected_identity=config, device=device)


@pytest.mark.parametrize('nested', [False, True])
def test_native_epsilon_refuses_deleted_boundary_record(epsilon_case, tmp_path, monkeypatch, nested):
    ns, carry = epsilon_case
    boundary = replace(ns.boundary_config, nested_frozen_wrf_boundary_bundle=nested, have_bcs_scalar=not nested)
    ns = replace(ns, boundary_config=boundary)
    state = initialize_child_scalar_boundaries(carry.state, mp_physics=ns.mp_physics,
                                              bl_pbl_physics=ns.bl_pbl_physics)
    carry = carry.replace(state=state)
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path/'store', 10**7, 2, 0)
    generation = save(store, carry, config)
    rewrite(generation, lambda p: p['state_fields'].__setitem__(epsilon_pbl_scalars(ns.bl_pbl_physics)[0]+'_bdy', None))
    monkeypatch.setattr(storage, '_restore_carry', lambda _: pytest.fail('device allocation before E78'))
    with pytest.raises(ValueError, match='boundary records'):
        store.read(generation, expected_identity=config)


@pytest.mark.parametrize('kind', ['missing_binding', 'missing_domain', 'wrong_domain', 'wrong_pair', 'boundary_type'])
def test_native_epsilon_refuses_unbound_configuration(epsilon_case, tmp_path, kind):
    ns, carry = epsilon_case
    config = identity(tmp_path, ns)
    if kind == 'missing_binding': config.pop('epsilon_pbl_config')
    elif kind == 'missing_domain': config['epsilon_pbl_config'].clear()
    else:
        binding = config['epsilon_pbl_config']['d01']
        if kind == 'wrong_domain': binding['domain_shape'] = [ns.grid.nz, 1, 1]
        elif kind == 'wrong_pair': binding['sf_sfclay_physics'] = 5
        else: binding['boundary_config']['have_bcs_scalar'] = 1
    store = storage.RestartStore(tmp_path/'store', 10**7, 2, 0)
    with pytest.raises(ValueError, match='EEPS/KEPS.*E78'):
        save(store, carry, config)
