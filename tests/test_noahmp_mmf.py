"""MMF host admission/restart and pristine REAL4 equations.

Constructed restart packets test IO only; physical gates use original-WRF
savepoints from the real NCAR-geography Swiss case (proofs/s2_mmf).
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.noahmp_mmf import configuration, refusals, validate_restart
from gpuwrf.physics.noahmp.groundwater import GroundwaterStatic, GroundwaterCarry, MMFBundle


def packet(shape):
    xy = jnp.asarray(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1)
    static = GroundwaterStatic(xy, xy * 1000, xy, -xy, -xy, xy / 1000,
                               xy / 100, jnp.broadcast_to(xy / 100, (4, *shape)), xy)
    carry = GroundwaterCarry(*(-xy if name == 'zwt' else xy / 1000
                               for name in GroundwaterCarry._fields))
    return MMFBundle(static, carry, jnp.ones(shape, jnp.bool_), jnp.float32(.38),
                     jnp.float32(150), jnp.int32(3), 1)


@pytest.fixture
def case(monkeypatch):
    # Structural controls use established legacy IO precision and no radii.
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD', '0')
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, '_MP_RE', False)
    from gpuwrf.validation.tier2 import make_ideal_grid
    from test_v0110_wrfrst_netcdf import _state, _noahmp_land
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = make_ideal_grid(5, 4, 5)
    state = _state(grid)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
        use_noahmp=True, noahmp_mmf_enabled=True, noahmp_mmf_opt_inf=1, dt_s=50.)
    land=_noahmp_land(grid)
    land=land.replace(**{n:jnp.asarray(getattr(land,n),jnp.float32) for n in ('smois','sh2o','smcwtd')})
    return grid, nl, initial_operational_carry(state).replace(noahmp_mmf=packet(state.t_skin.shape),noahmp_land=land)


def exact(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(la, lb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert (x.shape, x.dtype, x.tobytes()) == (y.shape, y.dtype, y.tobytes())


def test_default_tree_and_active_option_roundtrip(case):
    _, nl, carry = case
    # Other lanes may append optional groups; inactive groups retain the
    # exact pre-MMF prefix, and an active MMF appends exactly its one group.
    prefix=tuple(carry.__dataclass_fields__).index('noahmp_mmf')
    assert len(carry.replace(noahmp_mmf=None).tree_flatten()[0]) == prefix
    assert len(carry.tree_flatten()[0]) == prefix+1
    from gpuwrf.coupling.flg_gfdl_radiation import initial_gfdl
    gfdl = initial_gfdl(carry.state)
    for mmf in (None, carry.noahmp_mmf):
        for legacy in (None, gfdl):
            value = carry.replace(noahmp_mmf=mmf, gfdl_radiation=legacy)
            leaves, tree = jax.tree.flatten(value)
            exact(value, jax.tree.unflatten(tree, leaves))
    for active in (False, True):
        value = replace(nl, noahmp_mmf_enabled=active)
        leaves, structure = jax.tree.flatten(value)
        assert jax.tree.unflatten(structure, leaves).noahmp_mmf_enabled == active


@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('change', [{}, {'opt_inf': 2}, {'opt_inf': 3}, {'opt_run': 4}])
def test_honour_selection_and_refusal(replay, change):
    from gpuwrf.io.namelist_binding import namelist_honour_report
    raw = {'physics': {'sf_surface_physics': 4, 'wtddt': 2.5},
           'noah_mp': {'opt_run': 5, 'opt_inf': 1, **change}}
    errors, _ = namelist_honour_report(raw, ('d01',), replay=replay)
    if change.get('opt_inf') == 3 or change.get('opt_run') == 4:
        assert errors
    else:
        assert not errors, errors


@pytest.mark.parametrize('key,value', [('sf_surface_physics', 3), ('sf_lake_physics', 1),
    ('sf_urban_physics', 1), ('num_soil_layers', 5), ('wtddt', float('nan')),
    ('wtddt', float('inf')), ('wtddt', 0), ('wtddt', 1.e40), ('wtddt', 1.e-50)])
def test_invalid_combinations_refused(key, value):
    raw = {'physics': {'sf_surface_physics': 4, key: value}, 'noah_mp': {'opt_run': 5}}
    assert refusals(raw, ('d01',))
    assert refusals({'physics': {'sf_surface_physics': 4}, 'noah_mp': {'opt_run': 5}}, ('d01', 'd02'))


@pytest.mark.parametrize('damage', ['missing', 'marker', 'precision', 'shape', 'nan', 'static', 'mask', 'cadence', 'wrong_cadence', 'inf', 'accumulator'])
def test_restart_schema_refuses_damage(case, damage):
    _, nl, carry = case
    value, version = carry.noahmp_mmf, 1
    if damage == 'missing': value = None
    elif damage == 'marker': version = None
    elif damage == 'precision': value = value._replace(carry=value.carry._replace(zwt=np.asarray(value.carry.zwt, np.float64)))
    elif damage == 'shape': value = value._replace(carry=value.carry._replace(rech=np.ones((1, 1), np.float32)))
    elif damage == 'nan': value = value._replace(static=value.static._replace(eqzwt=np.full(value.carry.zwt.shape, np.nan, np.float32)))
    elif damage == 'static': value = value._replace(static=value.static._replace(rechclim=None))
    elif damage == 'mask': value = value._replace(eligibility=np.ones(value.eligibility.shape, np.float32))
    elif damage == 'cadence': value = value._replace(stepwtd=np.int32(0))
    elif damage == 'wrong_cadence': value = value._replace(stepwtd=np.int32(2))
    elif damage == 'inf': value = value._replace(opt_inf=2)
    elif damage == 'accumulator': value = value._replace(acrech=1)
    with pytest.raises(ValueError, match='E78'):
        validate_restart(value, nl, version, carry.state.t_skin.shape)


@pytest.mark.parametrize('fmt', ['pickle', 'netcdf', 'checkpoint', 'native'])
def test_public_restart_roundtrips_complete_mmf(case, tmp_path, fmt):
    grid, nl, carry = case
    value, read = _save(case, tmp_path, fmt)
    exact(carry, read())


@pytest.mark.parametrize('fmt', ['pickle','netcdf','checkpoint','native'])
def test_active_mmf_requires_companion_soil_state(case,tmp_path,fmt):
    grid,nl,carry=case
    with pytest.raises(ValueError,match='E78'):
        _save((grid,nl,carry.replace(noahmp_land=None)),tmp_path,fmt)


def test_state_only_checkpoint_readers_refuse_discarding_mmf(case,tmp_path):
    from gpuwrf.runtime.checkpoint import read_checkpoint,read_checkpoint_with_land_state,read_checkpoint_with_runtime_state
    path,_=_save(case,tmp_path,'checkpoint')
    for read in (read_checkpoint,read_checkpoint_with_land_state):
        with pytest.raises(ValueError,match='MMF.*E78'):read(path)
    with pytest.raises(ValueError,match='selection.*E78'):
        read_checkpoint_with_runtime_state(path,namelist=replace(case[1],noahmp_mmf_enabled=False))


def _save(case, tmp_path, fmt):
    grid, nl, carry = case
    if fmt == 'pickle':
        from gpuwrf.io.restart import write_restart, read_restart
        path = tmp_path / 'carry.pkl'
        write_restart(carry, nl, grid, 2, path)
        return path, lambda: read_restart(path)[0]
    if fmt == 'netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
        path = tmp_path / 'carry.nc'
        write_wrfrst_carry(carry, grid, nl, path, valid_time='2026-10-12_00:01:40',
                          run_start='2026-10-12_00:00:00', step_index=2)
        return path, lambda: read_wrfrst_carry(path, namelist=nl)[0]
    if fmt == 'checkpoint':
        from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_runtime_state
        path = tmp_path / 'checkpoint.pkl'
        write_checkpoint(carry.state, nl, grid, 2, path, runtime_state=carry)
        return path, lambda: read_checkpoint_with_runtime_state(path)[4]
    from gpuwrf.runtime.restart_store import RestartStore
    store = RestartStore(tmp_path / 'store', 10**7, 2, 0)
    identity = {'noahmp_mmf_config': {'d01': configuration(nl)}}
    path, _ = store.save({'d01': carry}, {'d01': 2}, {'d01': 50.}, identity)
    return path, lambda: store.read(path, expected_identity=identity)[0]['carries']['d01']


@pytest.mark.parametrize('fmt', ['pickle', 'netcdf', 'checkpoint', 'native'])
def test_public_restart_missing_group_refused(case, tmp_path, fmt):
    path, read = _save(case, tmp_path, fmt)
    if fmt == 'netcdf':
        from gpuwrf.io.netcdf_lock import Dataset
        with Dataset(path, 'r+') as ds:
            kinds = json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND)
            kinds['noahmp_mmf'] = 'none'
            ds.GPUWRF_OPTIONAL_CARRY_KIND = json.dumps(kinds)
            order = json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER)
            order['noahmp_mmf'] = []
            ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER = json.dumps(order)
    else:
        native = fmt == 'native'
        payload_path = path / 'snapshot.pkl' if native else path
        payload = pickle.loads(payload_path.read_bytes())
        if native: payload['carries']['d01']['extra_fields']['noahmp_mmf'] = None
        elif fmt == 'checkpoint': payload['runtime_state'] = payload['runtime_state'].replace(noahmp_mmf=None)
        else: payload['carry']['extra_fields']['noahmp_mmf'] = None
        data = pickle.dumps(payload)
        payload_path.write_bytes(data)
        if native:
            manifest_path = path / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='E78'):
        read()


def test_actual_history_writer_emits_held_mmf_fields(case, tmp_path):
    from datetime import datetime
    from types import SimpleNamespace
    from gpuwrf.io.noahmp_mmf import history_fields
    from gpuwrf.io.wrfout_writer import MMF_HISTORY_FIELDS, write_wrfout_netcdf
    from gpuwrf.io.netcdf_lock import Dataset
    from test_m7_netcdf_writer import writer_authority
    grid,nl,carry=case
    fields=history_fields(carry.noahmp_mmf,SimpleNamespace(smcwtd=jnp.full(carry.state.t_skin.shape,.3,jnp.float32)))
    path=tmp_path/'mmf_history.nc'
    write_wrfout_netcdf(carry.state,grid,nl,path,domain='d01',domain_authority=writer_authority(grid,'d01'),
        valid_time=datetime(2026,10,12),run_start=datetime(2026,10,12),lead_hours=0,
        diagnostics=fields,variable_subset=MMF_HISTORY_FIELDS,include_mandatory_coords=True)
    with Dataset(path) as ds:
        for name in MMF_HISTORY_FIELDS:
            assert name in ds.variables
            a=np.asarray(ds[name][0]);b=np.asarray(fields[name])
            assert a.dtype==b.dtype and a.tobytes()==b.tobytes(),name


@pytest.mark.parametrize('damage', [None,'missing','zero','nan','inf','mapfactor','soil','seaice','glacier','metadata'])
def test_raw_input_admission_before_model_loading(tmp_path, damage):
    from gpuwrf.io.noahmp_mmf import preflight
    from gpuwrf.io.netcdf_lock import Dataset
    # IO controls only: these packets establish validation, never fidelity.
    fields={n:np.ones((4,5),np.float32) for n in ('FDEPTH','EQZWT','RECHCLIM','RIVERBED','HGT',
        'MAPFAC_MX','MAPFAC_MY','XLAND','ISLTYP','IVGTYP','SEAICE')}
    fields['EQZWT'] *= -1
    fields['SEAICE'][:]=0
    if damage=='missing': fields.pop('FDEPTH')
    elif damage=='zero': fields['FDEPTH'][:]=0
    elif damage=='nan': fields['HGT'][0,0]=np.nan
    elif damage=='inf': fields['EQZWT'][0,0]=np.inf
    elif damage=='mapfactor': fields['MAPFAC_MX'][0,0]=0
    elif damage=='soil': fields['ISLTYP'][0,0]=20
    elif damage=='seaice': fields['SEAICE'][0,0]=1
    elif damage=='glacier': fields['IVGTYP'][0,0]=15
    with Dataset(tmp_path/'wrfinput_d01','w') as ds:
        ds.createDimension('Time',1);ds.createDimension('south_north',4);ds.createDimension('west_east',5)
        if damage!='metadata':ds.ISICE=15
        for n,a in fields.items():ds.createVariable(n,'f4',('Time','south_north','west_east'))[0]=a
    raw={'physics':{'sf_surface_physics':4},'noah_mp':{'opt_run':5}}
    if damage is None:preflight(tmp_path,raw,('d01',))
    else:
        with pytest.raises(ValueError,match='MMF'):preflight(tmp_path,raw,('d01',))
