"""EEPS/KEPS ownership, E78 serialization, and Registry transport contracts."""
from dataclasses import replace
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES,EPSILON_PBL_LEAVES
from gpuwrf.coupling.epsilon_pbl import seed_epsilon_pbl_state
from gpuwrf.io.pbl_epsilon_schema import validate_epsilon_state
from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars,required_extra_boundary_records
from gpuwrf.nesting.boundary_construction import initialize_child_scalar_boundaries
from gpuwrf.runtime.operational_mode import OperationalNamelist,_nested_number_scalar_species,_diffused_scalar_species
from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state,read_wrfrst_state


def fixture(bl):
    grid=GridSpec.canary_3km_template()
    state=State(**{k:jnp.zeros(shape,dtype=DEFAULT_DTYPES.dtype_for(k))
                   for k,shape in _state_field_shapes(grid).items()})
    state=seed_epsilon_pbl_state(state,bl)
    from test_m7_restart_checkpoint_roundtrip import _synthetic_tendencies
    nl=OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=_synthetic_tendencies(grid),dt_s=18.,acoustic_substeps=4,
                          bl_pbl_physics=bl,sf_sfclay_physics=1,rad_rk_tendf=1,use_flux_advection=True)
    return state,grid,nl


@pytest.mark.parametrize('bl',[16,17])
def test_seed_and_registry_transport(bl):
    state,grid,nl=fixture(bl)
    validate_epsilon_state(state,nl,1)
    assert seed_epsilon_pbl_state(state,5) is state
    for name in epsilon_pbl_scalars(bl):
        assert name in _nested_number_scalar_species(nl)
        assert name in _diffused_scalar_species(nl)
        assert getattr(state,name).dtype==jnp.float32
    child=initialize_child_scalar_boundaries(state,mp_physics=8,bl_pbl_physics=bl)
    nested=replace(nl,boundary_config=replace(nl.boundary_config,nested_frozen_wrf_boundary_bundle=True))
    validate_epsilon_state(child,nested,1)
    assert set(required_extra_boundary_records(nested))=={n+'_bdy' for n in epsilon_pbl_scalars(bl)}
    for name in epsilon_pbl_scalars(bl):
        # Genuine deletion of a live scalar's boundary is refused before restart.
        with pytest.raises(ValueError,match='boundary'):
            validate_epsilon_state(child.replace(**{name+'_bdy':None}),nested,1)
    assert seed_epsilon_pbl_state(state.replace(epsilon_pbl_step=jnp.int32(7)),bl).epsilon_pbl_step==7


@pytest.mark.parametrize('bl',[16,17])
@pytest.mark.parametrize('format',['checkpoint','netcdf'])
def test_restart_exact_and_missing_schema_refused(tmp_path,bl,format):
    state,grid,nl=fixture(bl)
    state=state.replace(epsilon_pbl_step=jnp.int32(19),epsilon_pbl=jnp.full_like(state.qv,0.125))
    path=tmp_path/('restart.nc' if format=='netcdf' else 'restart.pkl')
    if format=='checkpoint':
        write_checkpoint(state,nl,grid,19,path)
        restored,*_=read_checkpoint(path)
    else:
        write_wrfrst_state(state,grid,nl,path,valid_time='2026-07-25_18:05:42',run_start='2026-07-25_18:00:00',step_index=19)
        restored,_=read_wrfrst_state(path)
    for name in EPSILON_PBL_LEAVES:
        before,after=getattr(state,name),getattr(restored,name)
        if before is None: assert after is None
        else:
            assert before.dtype==after.dtype
            np.testing.assert_array_equal(before,after)
    if format=='checkpoint':
        payload=pickle.loads(path.read_bytes());payload.pop('epsilon_pbl_schema_version');path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError,match='predates'):
            read_checkpoint(path)
    else:
        from gpuwrf.io.netcdf_lock import Dataset
        with Dataset(path,'a') as ds: ds.delncattr('GPUWRF_EPSILON_PBL_SCHEMA_VERSION')
        with pytest.raises(ValueError,match='predates'):
            read_wrfrst_state(path)


@pytest.mark.parametrize('bl',[16,17])
def test_missing_carry_and_nonfinite_refused(bl):
    state,_,nl=fixture(bl)
    with pytest.raises(ValueError,match='missing required'):
        validate_epsilon_state(state.replace(epsilon_pbl=None),nl,1)
    with pytest.raises(ValueError,match='nonfinite'):
        validate_epsilon_state(state.replace(epsilon_pbl=jnp.full_like(state.qv,jnp.nan)),nl,1)
    with pytest.raises(ValueError,match='requires bl'):
        validate_epsilon_state(state,{'bl_pbl_physics':5},1)


@pytest.mark.parametrize('bl',[16,17])
def test_full_carry_restart_and_held_history(tmp_path,bl):
    from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.io.wrfout_writer import _epsilon_history_fields
    state,grid,nl=fixture(bl)
    state=state.replace(epsilon_pbl_step=jnp.int32(23))
    carry=initial_operational_carry(state).replace(history_diagnostics=seed_history_diagnostics(state))
    path=tmp_path/'full.pkl';write_restart(carry,nl,grid,23,path)
    restored,*_=read_restart(path)
    path=tmp_path/'full.nc'
    write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-07-25_18:06:54',run_start='2026-07-25_18:00:00',step_index=23)
    restored_nc,_=read_wrfrst_carry(path,namelist=nl)
    for other in (restored,restored_nc):
        assert int(other.state.epsilon_pbl_step)==23
        for name in state.active_field_names():
            np.testing.assert_array_equal(getattr(other.state,name),getattr(state,name))
    fields=_epsilon_history_fields(state,bl,state.qv.shape)
    assert set(fields)==({'PEK_PBL','PEP_PBL'} if bl==16 else {'TKE_PBL','DISS_PBL','TPE_PBL','PR_PBL'})
    tke=fields['PEK_PBL' if bl==16 else 'TKE_PBL']
    np.testing.assert_array_equal(tke if bl==16 else tke[:-1],state.qke)
    if bl==17: np.testing.assert_array_equal(tke[-1],0.)
    with pytest.raises(ValueError,match='missing held'):
        _epsilon_history_fields(state.replace(epsilon_pbl=None),bl,state.qv.shape)


@pytest.mark.parametrize('bl',[16,17])
@pytest.mark.parametrize('format',['pickle','netcdf'])
def test_full_restart_refuses_deleted_surface_history_before_restoration(tmp_path,bl,format,monkeypatch):
    import json
    from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
    from gpuwrf.io import restart,wrfrst_netcdf
    from gpuwrf.io.netcdf_lock import Dataset
    state,grid,nl=fixture(bl)
    carry=initial_operational_carry(state).replace(history_diagnostics=seed_history_diagnostics(state))
    if format=='pickle':
        path=tmp_path/'full.pkl';restart.write_restart(carry,nl,grid,0,path)
        payload=pickle.loads(path.read_bytes())
        payload['carry']['extra_fields']['history_diagnostics']=None
        path.write_bytes(pickle.dumps(payload))
        def forbidden(_): pytest.fail('device restoration before EPS history admission')
        monkeypatch.setattr(restart,'_restore_carry',forbidden)
        with pytest.raises(ValueError,match='surface history.*E78'):
            restart.read_restart(path)
    else:
        path=tmp_path/'full.nc'
        wrfrst_netcdf.write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-07-25_18:00:00',
            run_start='2026-07-25_18:00:00',step_index=0)
        with Dataset(path,'a') as ds:
            kind=json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND);kind['history_diagnostics']='none'
            ds.GPUWRF_OPTIONAL_CARRY_KIND=json.dumps(kind)
            order=json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER);order['history_diagnostics']=[]
            ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER=json.dumps(order)
        monkeypatch.setattr(wrfrst_netcdf.jnp,'asarray',lambda *a,**k:pytest.fail('device allocation before EPS history admission'))
        with pytest.raises(ValueError,match='surface history.*E78'):
            wrfrst_netcdf.read_wrfrst_carry(path,namelist=nl)


@pytest.mark.parametrize('bl',[16,17])
@pytest.mark.parametrize('replay',[False,True])
def test_both_pipeline_binding_and_honour(bl,replay):
    from gpuwrf.io.namelist_binding import bind_namelist_options,namelist_honour_report
    from gpuwrf.coupling.physics_dispatch import resolve_physics_suite
    from gpuwrf.io.scheme_catalog import _IMPLEMENTED_NOTES
    from gpuwrf.io.namelist_check import validate_operational_namelist
    _,_,nl=fixture(bl)
    raw={'physics':{'bl_pbl_physics':[bl],'sf_sfclay_physics':[1]}}
    errors,_=namelist_honour_report(raw,('d01',),replay=replay)
    assert errors==[]
    validate_operational_namelist(raw)
    bound=bind_namelist_options(replace(nl,bl_pbl_physics=5),raw,'d01')
    assert bound.bl_pbl_physics==bl
    assert resolve_physics_suite(bound).pbl.option==bl
    note=_IMPLEMENTED_NOTES[('bl_pbl_physics',bl)]
    assert 'EXPERIMENTAL' in note and 'materially' in note
    # EEPS is nested-qualified (v0.4.2, proofs/w41_nestqual; tests/v042/test_nestqual_notes.py); KEPS stays pending
    if bl==16:
        assert 'Nested-qualified' in note and 'qualification pending' not in note
    else:
        assert 'GPU/coupled-forecast qualification pending' in note
    raw['physics']['sf_sfclay_physics']=[5]
    errors,_=namelist_honour_report(raw,('d01',),replay=replay)
    assert any('revised-MM5' in e for e in errors)
    raw['physics']['sf_sfclay_physics']=[1]
    raw['physics']['sf_surface_physics']=[2]
    errors,_=namelist_honour_report(raw,('d01',),replay=replay)
    assert any('no-LSM/Noah-MP' in e for e in errors)
    raw['physics']['sf_surface_physics']=[4]
    raw['physics']['bldt']=[10.]
    errors,_=namelist_honour_report(raw,('d01',),replay=replay)
    assert any('bldt=0' in e for e in errors)


@pytest.mark.parametrize('bl',[16,17])
def test_history_matches_original_registry_and_held_levels(tmp_path,bl):
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf,bind_wrfout_domain_authority
    from gpuwrf.io.netcdf_lock import Dataset
    from test_m7_netcdf_writer import authenticated_source_grid
    state,grid,nl=fixture(bl)
    state=state.replace(qke=jnp.arange(state.qke.size,dtype=jnp.float32).reshape(state.qke.shape)/1024,
                        epsilon_pbl=jnp.full_like(state.qke,0.03125))
    expected=({'PEK_PBL':state.qke,'PEP_PBL':state.epsilon_pbl} if bl==16 else
              {'TKE_PBL':state.qke,'DISS_PBL':state.epsilon_pbl,'TPE_PBL':state.tpe_pbl,'PR_PBL':state.pr_pbl})
    # Independent source schema, rather than the writer's variable-spec table.
    pristine=Path(__file__).resolve().parents[1]/'data/wrf_pristine/WRF'
    rows={tokens[2].upper():tokens for line in (pristine/'Registry/Registry.EM_COMMON').read_text().splitlines()
          if len(tokens:=line.split())>7 and tokens[0]=='state'}
    path=tmp_path/'history.nc'
    write_wrfout_netcdf(state,grid,nl,path,domain='d01',
        domain_authority=bind_wrfout_domain_authority('d01',authenticated_source_grid(grid,'d01'),grid),
        valid_time='2026-07-25_18:00:00',run_start='2026-07-25_18:00:00',lead_hours=0.,
        variable_subset=frozenset((*expected,'QKE')))
    with Dataset(path) as ds:
        assert 'QKE' not in ds.variables
        for name,value in expected.items():
            row=rows[name]
            assert row[3]=='ikj'
            stagger='Z' if row[6]=='Z' else ''
            vertical='bottom_top_stag' if stagger else 'bottom_top'
            assert ds[name].dimensions==('Time',vertical,'south_north','west_east')
            assert ds[name].stagger==stagger
            assert ds[name].dtype==np.dtype('float32')
            if name=='TKE_PBL':
                assert stagger=='Z'
                np.testing.assert_array_equal(ds[name][0,:grid.nz],value)
                # Original KEPS writes kts:kte; allocation init leaves top zero.
                assert 'initial_data_value = 0.' in (pristine/'tools/gen_allocs.c').read_text()
                np.testing.assert_array_equal(ds[name][0,grid.nz],np.zeros((grid.ny,grid.nx),np.float32))
            else:
                np.testing.assert_array_equal(ds[name][0],value)


@pytest.mark.parametrize('bl,requested',[(16,'T2'),(17,'T2'),(16,'PEP_PBL'),(17,'DISS_PBL'),(17,'TKE_PBL')])
def test_selected_epsilon_materialization_stays_inside_host_batch(tmp_path,monkeypatch,bl,requested):
    import jax
    from gpuwrf.io import wrfout_writer as writer
    from test_m7_netcdf_writer import authenticated_source_grid
    state,grid,nl=fixture(bl)
    sources={name:getattr(state,leaf) for name,leaf in
             (('PEK_PBL','qke'),('PEP_PBL','epsilon_pbl'))} if bl==16 else {
             name:getattr(state,leaf) for name,leaf in
             (('TKE_PBL','qke'),('DISS_PBL','epsilon_pbl'),('TPE_PBL','tpe_pbl'),('PR_PBL','pr_pbl'))}
    eps_ids={id(value) for value in sources.values()}
    batches=[];coerced=[]
    original_get=jax.device_get
    original_coerce=writer._coerce_array
    def device_get(values):
        batches.append({id(value) for value in jax.tree.leaves(values)} & eps_ids)
        return original_get(values)
    def coerce(name,value,shape,**kwargs):
        if name in sources:
            assert writer._WRFOUT_HOST_ARRAYS.get() is not None
            assert isinstance(value,np.ndarray)
            coerced.append(name)
        return original_coerce(name,value,shape,**kwargs)
    monkeypatch.setattr(jax,'device_get',device_get)
    monkeypatch.setattr(writer,'_coerce_array',coerce)
    payload=writer.prepare_wrfout_payload(state,grid,nl,tmp_path/'selected.nc',domain='d01',
        domain_authority=writer.bind_wrfout_domain_authority('d01',authenticated_source_grid(grid,'d01'),grid),
        valid_time='2026-07-25_18:00:00',run_start='2026-07-25_18:00:00',lead_hours=0.,
        variable_subset=frozenset((requested,)),include_mandatory_coords=False)
    selected={id(sources[requested])} if requested in sources else set()
    assert batches==[selected]
    assert coerced==([requested] if requested in sources else [])
    assert set(payload.fields)=={requested}


@pytest.mark.parametrize('bl',[16,17])
@pytest.mark.parametrize('records',[False,True])
@pytest.mark.parametrize('have',[False,True])
def test_root_boundary_group_flag_matches_solve_em(bl,records,have,monkeypatch):
    # WRF solve_em :2894/:2997 + Registry.EM_COMMON :2950: GROUP flag wins.
    from gpuwrf.runtime.operational_mode import _root_flow_only,_root_scalar_bdy_rk1_active
    from gpuwrf.coupling.boundary_apply import root_scalar_rk1_split
    state,_,nl=fixture(bl)
    if records: state=initialize_child_scalar_boundaries(state,mp_physics=8,bl_pbl_physics=bl)
    config=replace(nl.boundary_config,have_bcs_scalar=have,have_bcs_moist=False)
    nl=replace(nl,boundary_config=config,run_boundary=True,specified_bdy_cadence=True)
    monkeypatch.setenv('GPUWRF_ROOT_SCALAR_BDY_RK1','0')
    assert _root_scalar_bdy_rk1_active(nl)  # EPS always uses original WRF cadence.
    species=epsilon_pbl_scalars(bl)
    kwargs=_root_flow_only(nl,state)
    if have and not records:
        with pytest.raises(NotImplementedError,match='no lateral boundary'):
            root_scalar_rk1_split(config,species,**kwargs)
        with pytest.raises(ValueError,match='boundary'):
            validate_epsilon_state(state,nl,1)
    else:
        relaxed,flow=root_scalar_rk1_split(config,species,**kwargs)
        assert relaxed==(species if have else ())
        assert flow==(() if have else species)


@pytest.mark.parametrize('bl',[16,17])
def test_netcdf_state_resume_retains_required_root_boundary_flag(tmp_path,bl):
    import json
    from gpuwrf.io.netcdf_lock import Dataset
    state,grid,nl=fixture(bl)
    state=initialize_child_scalar_boundaries(state,mp_physics=8,bl_pbl_physics=bl)
    nl=replace(nl,boundary_config=replace(nl.boundary_config,have_bcs_scalar=True,have_bcs_moist=False))
    path=tmp_path/'root.nc'
    write_wrfrst_state(state,grid,nl,path,valid_time='2026-07-25_18:00:00',run_start='2026-07-25_18:00:00',step_index=0)
    read_wrfrst_state(path)
    with Dataset(path,'a') as ds:
        config=json.loads(ds.GPUWRF_EPSILON_PBL_CONFIG)
        assert config['boundary_config']['have_bcs_scalar'] is True
        order=json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
        order.remove(epsilon_pbl_scalars(bl)[0]+'_bdy')
        ds.GPUWRF_STATE_FIELD_ORDER=json.dumps(order)
    with pytest.raises(ValueError,match='boundary'):
        read_wrfrst_state(path)


@pytest.mark.parametrize('field',['pek_adv','pep_adv','tke_adv','diss_adv','tpe_adv'])
def test_registry_scalar_boundary_spelling_and_decoder(tmp_path,field):
    # Independent name reference: WRF Registry + generated scalar_dname_table
    # use lowercase names; wrf_bdyout appends the uppercase side suffix only.
    from gpuwrf.integration.d02_replay import _WRFBDY_SCALAR_BOUNDARY_VARS,_available_wrfbdy_variables
    from gpuwrf.io.boundary_replay import decode_wrfbdy,WRFBDY_BASE_SUFFIXES,WRFBDY_TENDENCY_SUFFIXES
    from gpuwrf.io.netcdf_lock import Dataset
    name=dict(_WRFBDY_SCALAR_BOUNDARY_VARS)[field]
    assert name==field
    path=tmp_path/'wrfbdy_d01'
    with Dataset(path,'w') as ds:
        for dim,size in [('Time',1),('DateStrLen',19),('bdy_width',1),('bottom_top',3),('side',2)]:
            ds.createDimension(dim,size)
        ds.createVariable('Times','S1',('Time','DateStrLen'))[:]=np.array(list('2026-07-25_18:00:00'),'S1')[None]
        for side in WRFBDY_BASE_SUFFIXES:
            for suffix in (WRFBDY_BASE_SUFFIXES[side],WRFBDY_TENDENCY_SUFFIXES[side]):
                ds.createVariable(field+'_'+suffix,'f4',('Time','bdy_width','bottom_top','side'))[:]=np.float32(0.125)
    available=_available_wrfbdy_variables(path,tuple(v for _,v in _WRFBDY_SCALAR_BOUNDARY_VARS))
    assert available==(field,)
    decoded=decode_wrfbdy(path,variables=available)['variables'][field]
    for side,values in decoded['sides'].items():
        np.testing.assert_array_equal(values['boundary'],np.full((1,3,2),0.125))
        assert values['boundary_variable']==field+'_'+WRFBDY_BASE_SUFFIXES[side]
