"""Independent terrain oracle and MYJSFC history producer/serialization checks."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.physics.microphysics_lin import lin_run
from gpuwrf.runtime.operational_state import HistoryDiagnostics

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / 'proofs/t_c03rain'


def test_lin_terrain_matches_original_wrf_all_six_regimes():
    path = PROOF / 'lin_terrain_original.npz'
    manifest = json.loads((PROOF / 'lin_terrain_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['fixture_sha256']
    with np.load(path) as d:
        args = [jnp.asarray(d[k]) for k in ('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN',
                                           'RHO','PII','P','Z','DZ8W')]
        out = lin_run(*args, float(d['DT'][0]), ht=d['HT'])
        # Existing Lin gates from proofs/v060/run_lin_parity.py, unchanged.
        for key, name in [('th','TH_OUT'),('qv','QV_OUT'),('qc','QC_OUT'),('qr','QR_OUT'),
                          ('qi','QI_OUT'),('qs','QS_OUT'),('qg','QG_OUT'),
                          ('rainncv','RAINNCV'),('snowncv','SNOWNCV'),('graupelncv','GRAUPELNCV')]:
            actual, ref = np.asarray(out[key]), d[name]
            assert np.isfinite(actual).all() and np.isfinite(ref).all(), key
            for row in range(6):
                if key == 'th': limit = 1e-2
                elif key.endswith('ncv'): limit = max(5e-4, .015*np.max(np.abs(ref[row])))
                else: limit = max(1e-7, .01*np.max(np.abs(ref[row])))
                assert np.max(np.abs(actual[row]-ref[row])) <= limit, (key,row)


def test_lin_adapter_supplies_actual_surface_height(monkeypatch):
    from test_v013_myj_janjic_operational import _grid, _state
    from gpuwrf.coupling import scan_adapters as adapters
    state = _state(_grid()).replace(ph=_state(_grid()).ph + 1500*9.81)
    received = {}
    def endpoint(*args, **kw):
        received.update(kw)
        return SimpleNamespace(validate_keys=lambda: None)
    monkeypatch.setattr(adapters, 'lin_physics_tendency', endpoint)
    monkeypatch.setattr(adapters, '_apply_mp_replacements', lambda state,*a,**kw:state)
    adapters.lin_adapter(state, 18)
    np.testing.assert_array_equal(received['ht'], np.asarray(state.ph[0]/adapters.GRAVITY_M_S2).ravel())


def test_lin_caller_matches_original_wrf_moist_prep_and_finish(monkeypatch):
    from test_non_thompson_precipitation import state_for
    from gpuwrf.coupling import scan_adapters as adapters
    from gpuwrf.contracts.physics_interfaces import PhysicsTendency
    data=json.loads((ROOT/'proofs/t_mprain/thermo_oracle.json').read_text())
    state=state_for(option=2,nz=len(data['theta_m_in']))
    col=lambda key:jnp.asarray(data[key],jnp.float32)[:,None,None].repeat(2,axis=2)
    state=state.replace(theta=col('theta_m_in'),qv=col('qv_in'),
        p_total=jnp.full_like(state.theta,90000.),
        ph_total=jnp.arange(state.theta.shape[0]+1,dtype=jnp.float32)[:,None,None].repeat(2,axis=2)*1000.)
    calls=[]
    def original_boundary(*args,**kw):
        calls.append(True)
        np.testing.assert_allclose(np.asarray(args[0])[0],data['dry_theta_in'],rtol=0,atol=4e-5)
        shape=(2,state.theta.shape[0])
        return PhysicsTendency(state_replacements={
            'theta':jnp.broadcast_to(jnp.asarray(data['dry_theta_out'],jnp.float32),shape),
            'qv':jnp.broadcast_to(jnp.asarray(data['qv_out'],jnp.float32),shape)})
    monkeypatch.setattr(adapters,'lin_physics_tendency',original_boundary)
    result=adapters.lin_adapter(state,18.)
    assert calls==[True]
    np.testing.assert_allclose(np.asarray(result.theta)[:,0,0],data['theta_m_out'],rtol=0,atol=6e-5)
    np.testing.assert_array_equal(result.qv,col('qv_out'))
    assert result.theta.dtype==jnp.float32


def test_myjsfc_adapter_retains_special_winds_without_changing_state(monkeypatch):
    from test_v013_myj_janjic_operational import _grid, _state
    from gpuwrf.physics.myj_adapters import janjic_sfclay_adapter
    state = _state(_grid())
    old = janjic_sfclay_adapter(state, 18)
    new, special = janjic_sfclay_adapter(state, 18, history=True)
    for a,b in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new)):
        np.testing.assert_array_equal(a,b)
    assert set(special) == {'u10e','v10e'}
    assert all(v.dtype == jnp.float32 and v.shape == state.t_skin.shape for v in special.values())
    assert np.max(np.abs(special['u10e'])) > 0


def test_history_write_preserves_special_winds_across_noah_overlay():
    zero = jnp.zeros((2,3),jnp.float32)
    history = HistoryDiagnostics(*(zero for _ in range(7)),u10e=zero+4,v10e=zero-2)
    out = history.write({k:zero+1 for k in history._fields[:7]})
    np.testing.assert_array_equal(out.u10e,zero+4)
    np.testing.assert_array_equal(out.v10e,zero-2)
    default = HistoryDiagnostics(*(zero for _ in range(7)))
    assert len(jax.tree_util.tree_leaves(default)) == 7
    assert default.write({k:zero for k in default._fields[:7]}).u10e is None


def test_myjsfc_seed_and_refresh_keep_loop_tree_static():
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    state=SimpleNamespace(t_skin=jnp.zeros((2,3),jnp.float32))
    history=seed_history_diagnostics(state,myjsfc=True)
    def body(i,h):
        return h._replace(u10e=h.u10e+1,v10e=h.v10e-1)
    refreshed=jax.lax.fori_loop(0,2,body,history)
    np.testing.assert_array_equal(refreshed.u10e,np.full((2,3),2,np.float32))
    assert jax.tree_util.tree_structure(refreshed) == jax.tree_util.tree_structure(history)


@pytest.mark.parametrize('scheme',[2,4])
@pytest.mark.parametrize('present',[False,True])
def test_lead_zero_special_winds_follow_original_physics_init(tmp_path,present,scheme):
    from test_initial_history_wrf_t0 import _writer,INPUT
    def no_resolve(*a,**kw): raise AssertionError('lead-zero surface solve')
    writer,state=_writer(tmp_path,no_resolve)
    writer.bundles['d01'].namelist=SimpleNamespace(sf_sfclay_physics=scheme)
    if present:
        with Dataset(tmp_path/'wrfinput_d01','r+') as ds:
            for name,value in [('U10E',4.5),('V10E',-2.5)]:
                ds.createVariable(name,'f4',('Time','south_north','west_east'))[0]=value
    fields=writer._initial_surface_fields('d01')
    # Original physics_init:1326-1327 overwrites these with initialized U10/V10,
    # even if an input happens to contain special-wind variables already.
    for name in ('U10E','V10E'):
        np.testing.assert_array_equal(fields[name],np.full(state.t_skin.shape,INPUT[name[:-1]],np.float32))
    writer.bundles['d01'].namelist=SimpleNamespace(sf_sfclay_physics=5)
    default=writer._initial_surface_fields('d01')
    assert 'U10E' not in default and 'V10E' not in default


def test_special_winds_restart_roundtrip(tmp_path):
    from test_v013_myj_janjic_operational import _grid, _state, _namelist
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
    grid=_grid();state=_state(grid);zero=jnp.zeros_like(state.t_skin,dtype=jnp.float32)
    history=seed_history_diagnostics(state)._replace(u10e=zero+4.5,v10e=zero-2.5)
    from gpuwrf.coupling.myj_surface import initial_myj_memory
    # MYJSFC INOUT memory is resident restart state with sf=2 (t-cfix-sfc F12, E78).
    carry=initial_operational_carry(state).replace(history_diagnostics=history,qnse=initial_myj_memory(state))
    nml=_namelist(grid,bl_pbl_physics=2,sf_sfclay_physics=2)
    write_restart(carry,nml,grid,7,tmp_path/'carry.pkl')
    write_wrfrst_carry(carry,grid,{},tmp_path/'carry.nc',valid_time='2026-07-25_18:02:06',
                       run_start='2026-07-25_18:00:00',step_index=7)
    for restored in (read_restart(tmp_path/'carry.pkl')[0],read_wrfrst_carry(tmp_path/'carry.nc')[0]):
        for name in ('u10e','v10e'):
            np.testing.assert_array_equal(getattr(restored.history_diagnostics,name),getattr(history,name))


@pytest.mark.parametrize('full,subset',[(False,None),(True,None),(True,('U10E','V10E'))])
def test_writer_emits_actual_special_winds_and_wrf_metadata(tmp_path,full,subset):
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    state,grid,nml = synthetic_case()
    data = {'U10E':np.full(state.t_skin.shape,4.5,np.float32),
            'V10E':np.full(state.t_skin.shape,-2.5,np.float32)}
    prepared=prepare_wrfout_payload(state,grid,nml,tmp_path/'special.nc',domain='d02',
        domain_authority=writer_authority(grid),valid_time=datetime(2026,5,25,19),
        run_start=datetime(2026,5,25,18),lead_hours=1,diagnostics=data,
        full_variable_set=full,variable_subset=subset)
    write_prepared_wrfout(prepared, expected_domain='d02', expected_domain_authority=writer_authority(grid))
    with Dataset(tmp_path/'special.nc') as ds:
        for name,expected in data.items():
            np.testing.assert_array_equal(ds[name][0],expected)
            assert ds[name].dimensions == ('Time','south_north','west_east')
            assert ds[name].units == 'm s-1'
            assert ds[name].description == f'Special {name[0]} at 10 M from MYJSFC'
