"""Independent original-WRF header/schema checks; no forecast or GPU."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
from gpuwrf.io.wrfout_writer import (
    WRFOUT_VARIABLE_SPECS, _dimension_sizes, prepare_wrfout_payload, write_prepared_wrfout,
)
from test_m7_netcdf_writer import synthetic_case, writer_authority

ORIGINAL = json.loads((Path(__file__).parents[1] / 'proofs/t_writer/cpu_schema.json').read_text())['variables']


@pytest.mark.parametrize('mp',[17,18,19,21,22])
def test_original_wrf_legacy_nssl_header_normalization(tmp_path,mp):
    source=tmp_path/'input.nc'
    with Dataset(source,'w') as ds:ds.MP_PHYSICS=np.int32(8)
    attrs=wrf_history_global_attributes(source,{'physics':{'mp_physics':mp}},'d01',18.)
    assert attrs['MP_PHYSICS']==18


@pytest.mark.parametrize('domain',['d01','d02'])
def test_original_wrf_shortwave_off_resolves_global_ghg(tmp_path,domain):
    source=tmp_path/'input.nc'
    with Dataset(source,'w') as ds:
        ds.GHG_INPUT=np.int32(1)
        ds.RA_SW_PHYSICS=np.int32(4)
    attrs=wrf_history_global_attributes(source,{'physics':{'ra_sw_physics':[4,0]}},domain,18.)
    assert attrs['GHG_INPUT']==0


@pytest.mark.parametrize('mp',[17,18,22])
def test_nssl_nested_writer_copies_producer_and_qualified_inactive_max(tmp_path,mp):
    from test_initial_history_wrf_t0 import _writer
    writer,state=_writer(tmp_path,lambda *a,**kw:{})
    nl=writer.bundles['d01'].namelist
    writer.bundles['d01'].namelist={**nl,'mp_physics':mp,'physics':{'mp_physics':mp}}
    # Original NSSL S6 values, not a port-vs-port numerical expectation.
    sp=json.loads((Path(__file__).parents[1]/'proofs/v034/f2_oracles/nssl_2mom/nssl_case_4.json').read_text())
    dbz=np.asarray(sp['columns']['DBZ_OUT'][:state.theta.shape[0]],np.float32)
    radar=np.broadcast_to(dbz[:,None,None],state.theta.shape).copy()
    assert radar.max()>0
    result=writer('d01',200,SimpleNamespace(state=state,radar_reflectivity=radar,nwp_diagnostics=None))
    with Dataset(result['wrfout']) as ds:
        np.testing.assert_array_equal(ds['REFL_10CM'][0],radar)
        # Authorized class B; every cell in all25 CPU frames is +REAL4zero.
        np.testing.assert_array_equal(ds['REFD_MAX'][0], np.zeros(state.t_skin.shape, np.float32))


def test_every_declared_cpu_variable_metadata_matches_original():
    checked = 0
    for name, original in ORIGINAL.items():
        spec = WRFOUT_VARIABLE_SPECS.get(name)
        if spec is None:
            continue  # Missing producers belong to the explicit matrix inventory gate.
        assert list(spec.dimensions) == original['dimensions'], name
        assert np.dtype(spec.dtype) == np.dtype(original['dtype']), name
        for attr, key in (('memory_order', 'MemoryOrder'), ('description', 'description'),
                          ('units', 'units'), ('stagger', 'stagger'), ('coordinates', 'coordinates')):
            assert getattr(spec, attr) == original['attrs'].get(key), (name, key)
        checked += 1
    assert checked >= 450


@pytest.mark.parametrize('lsm,n', [(0,5),(1,5),(2,4),(4,4),(5,10),(7,2),(8,3)])
def test_soil_dimension_follows_original_wrf_selector(lsm, n):
    # module_check_a_mundo.F:3566-3590, independent of supplied input's four layers.
    assert _dimension_sizes(nx=5, ny=4, nz=3,
        namelist={'sf_surface_physics':lsm,'soil_layers_stag':4})['soil_layers_stag'] == n


@pytest.mark.parametrize('full',[False,True])
def test_disabled_runtime_lsm_keeps_raw_wrf_five_layer_geometry(tmp_path,full):
    state,grid,nl=synthetic_case()
    nl['sf_surface_physics']=None  # actual native disabled-LSM representation
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'disabled.nc',domain='d01',
        domain_authority=authority,valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',
        lead_hours=0.,full_variable_set=full,source_global_attrs={'SF_SURFACE_PHYSICS':np.int32(0)})
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        assert len(ds.dimensions['soil_layers_stag'])==5
        if full:
            for name in ('DZS','ZS','SH2O','SMOIS','TSLB'):
                assert ds[name].shape[1]==5,name
        else:
            # Measured operational stream: no soil producer means no soil
            # arrays. Geometry still follows the authenticated WRF selector.
            assert not set(('DZS','ZS','SH2O','SMOIS','TSLB')) & set(ds.variables)


def test_goddard_initial_frame_copies_retained_cod_and_ctop(tmp_path):
    from test_initial_history_wrf_t0 import _writer
    writer,state=_writer(tmp_path,lambda *a,**kw:{})
    rad=SimpleNamespace(cod2d_out=np.full(state.t_skin.shape,3.25,np.float32),
                        ctop2d_out=np.full(state.t_skin.shape,650.,np.float32))
    result=writer('d01',0,SimpleNamespace(state=state,radiation_diagnostics=rad))
    with Dataset(result['wrfout']) as ds:
        np.testing.assert_array_equal(ds['COD2D_OUT'][0],rad.cod2d_out)
        np.testing.assert_array_equal(ds['CTOP2D_OUT'][0],rad.ctop2d_out)


@pytest.mark.parametrize('n', [6,9])
def test_ruc_soil_dimension_uses_actual_resolved_producer(n):
    static = SimpleNamespace(cfg=SimpleNamespace(nzs=n))
    assert _dimension_sizes(nx=5, ny=4, nz=3,
        namelist={'sf_surface_physics':3,'ruc_static':static})['soil_layers_stag'] == n


def test_changed_namelist_header_selectors_reach_every_output_frame(tmp_path):
    source = tmp_path/'initial.nc'
    keys = ('MP_PHYSICS','CU_PHYSICS','BL_PBL_PHYSICS','SF_SFCLAY_PHYSICS',
            'SF_SURFACE_PHYSICS','RA_LW_PHYSICS','RA_SW_PHYSICS','GHG_INPUT')
    with Dataset(source,'w') as ds:
        for key in keys:
            ds.setncattr(key,np.int32(8))  # deliberately stale wrfinput controls
    raw={'physics':{key.lower():[0,14] for key in keys}}
    attrs = wrf_history_global_attributes(source,raw,'d01',18.)
    state, grid, nl = synthetic_case()
    authority = writer_authority(grid,'d01')
    for h in (0,1,24):
        payload = prepare_wrfout_payload(state,grid,nl,tmp_path/f'frame{h}.nc',domain='d01',
            domain_authority=authority,valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',
            lead_hours=h,source_global_attrs=attrs)
        write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority)
        with Dataset(payload.target) as ds:
            for key in keys:
                assert ds.getncattr(key) == 0,key
    child = wrf_history_global_attributes(source,raw,'d02',6.)
    assert child['MP_PHYSICS'] == 14
    assert child['GHG_INPUT'] == 0  # scalar Registry entry uses first value


@pytest.mark.parametrize('mode',['full','operational','subset'])
def test_wdm5_number_fields_have_real_sources_and_original_metadata(tmp_path, mode):
    state,grid,nl=synthetic_case()
    nl['mp_physics']=14
    state.Nc=np.arange(60,dtype=np.float32).reshape(3,4,5)+7
    state.Nn=np.arange(60,dtype=np.float32).reshape(3,4,5)+103
    subset=('QNCLOUD','QNCCN') if mode=='subset' else None
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'wdm.nc',domain='d01',domain_authority=authority,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,
        full_variable_set=mode=='full',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    with Dataset(payload.target) as ds:
        for name,source in [('QNCLOUD',state.Nc),('QNCCN',state.Nn)]:
            np.testing.assert_array_equal(ds[name][0],source)
            for attr in ('units','description','MemoryOrder','stagger'):
                assert ds[name].getncattr(attr)==ORIGINAL[name]['attrs'][attr]


def test_wdm5_absent_number_source_is_refused(tmp_path):
    state,grid,nl=synthetic_case()
    nl['mp_physics']=14
    with pytest.raises(ValueError,match='QNCLOUD|QNCCN'):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'missing.nc',domain='d01',
            domain_authority=writer_authority(grid,'d01'),valid_time='2023-01-15T00:00:00',
            run_start='2023-01-15T00:00:00',lead_hours=0.,full_variable_set=True)


@pytest.mark.parametrize('bl,names', [(16,('PEK_PBL','PEP_PBL')),
    (17,('TKE_PBL','DISS_PBL','TPE_PBL','PR_PBL'))])
def test_actual_epsilon_serialization_keeps_original_registry_metadata(tmp_path,bl,names):
    state,grid,nl=synthetic_case()
    nl['bl_pbl_physics']=bl
    for leaf in ('qke','epsilon_pbl','tpe_pbl','pr_pbl'):
        setattr(state,leaf,np.ones(state.theta.shape,np.float32))
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'eps.nc',domain='d01',domain_authority=authority,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,full_variable_set=True)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        for name in names:
            for attr in ('description','units','stagger','MemoryOrder'):
                assert ds[name].getncattr(attr)==ORIGINAL[name]['attrs'][attr], (name,attr)


@pytest.mark.parametrize('mp,names', [(7,('QHAIL',)),(9,('QHAIL','QNCLOUD','QNSNOW','QNGRAUPEL','QNHAIL')),
    (10,('QNSNOW','QNGRAUPEL')), (50,('QIR','QIB')), (51,('QNCLOUD','QIR','QIB')),
    (52,('QNCLOUD','QIR','QIB','QICE2','QNICE2','QIR2','QIB2')),
    (53,('QNCLOUD','QIR','QIB','QZI')),
    (38,('QNCLOUD','QNGRAUPEL','QNWFA','QNIFA','QVGRAUPEL'))])
@pytest.mark.parametrize('full',[False,True])
def test_computed_milbrandt_and_morrison_species_are_emitted(tmp_path,mp,names,full):
    state,grid,nl=synthetic_case()
    nl['mp_physics']=mp
    mapping={'QHAIL':'qh','QNCLOUD':'Nc','QNSNOW':'Ns','QNGRAUPEL':'Ng','QNHAIL':'Nh',
        'QIR':'qir','QIB':'qib','QICE2':'qi2','QNICE2':'Ni2','QIR2':'qir2','QIB2':'qib2','QZI':'qzi',
        'QNWFA':'nwfa','QNIFA':'nifa','QVGRAUPEL':'qvolg'}
    for i,name in enumerate(names):
        setattr(state,mapping[name],np.full(state.theta.shape,i+.25,np.float32))
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'mp.nc',domain='d01',domain_authority=authority,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,full_variable_set=full)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        for name in names:
            assert name in ds.variables,name
            np.testing.assert_array_equal(ds[name][0],getattr(state,mapping[name]))


def test_restart_compatibility_keeps_previous_metadata_and_dimensions():
    from gpuwrf.io.wrfrst_netcdf import _from_wrfout
    old = _from_wrfout('THM')
    assert old.description == ('either 1) pert moist pot temp=(1+Rv/Rd Qv)*(theta+T0)-T0; '
                               'or 2) pert dry pot temp=theta; based on use_theta_m setting')
    nl={'sf_surface_physics':0,'soil_layers_stag':4,'stochastic_config':object()}
    history=_dimension_sizes(nx=5,ny=4,nz=3,namelist=nl)
    restart=_dimension_sizes(nx=5,ny=4,nz=3,namelist=nl,history_schema=False)
    assert history['soil_layers_stag']==5 and restart['soil_layers_stag']==4
    assert 'num_stoch_levels' in history and 'num_stoch_levels_stag' not in history
    assert 'num_stoch_levels_stag' in restart and 'num_stoch_levels' not in restart


def test_stochastic_spectral_history_copies_pristine_coefficients_in_host_batch(tmp_path,monkeypatch):
    import jax
    from gpuwrf.physics.stochastic_driver import StochasticCarry,PatternCarry
    from gpuwrf.io.stochastic_schema import history_fields,SPECTRAL_FIELDS
    with np.load(Path(__file__).parent/'fixtures/u2_stoch_real4.npz') as reference:
        expected={f'SPFORC{part}{suffix}':reference[f'{kind}_{step}'].copy()
            for suffix,step in ((2,0),(3,1),(4,2)) for part,kind in (('C','cosine'),('S','sine'))}
    assert all(np.count_nonzero(v) for v in expected.values())
    ny,nx=next(iter(expected.values())).shape
    state,grid,nl=synthetic_case(nx=nx-1,ny=ny-1,nz=3)
    nl['stochastic_config']=object()
    patterns=[None]*7
    for suffix,index in ((2,4),(3,5),(4,6)):
        sine=jax.device_put(expected[f'SPFORCS{suffix}'])
        cosine=jax.device_put(expected[f'SPFORCC{suffix}'])
        patterns[index]=PatternCarry(sine,cosine,np.zeros((ny,nx),np.float32),
            np.float32(0),np.float32(1),np.zeros((1,ny,nx),np.float32))
    carry=StochasticCarry(tuple(patterns),(None,)*6,None,np.zeros(7,np.int32),np.int32(1))
    fields=history_fields(carry,grid.nz,grid.ny,grid.nx)
    raw_ids={id(fields[k]) for k in SPECTRAL_FIELDS}
    calls=[]
    original_get=jax.device_get
    def capture(values):
        calls.append({id(v) for v in values})
        return original_get(values)
    monkeypatch.setattr(jax,'device_get',capture)
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'spectra.nc',domain='d01',domain_authority=authority,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,
        diagnostics=fields,full_variable_set=True)
    assert len(calls)==1 and raw_ids<=calls[0]
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        for name,want in expected.items():
            assert name in ds.variables
            np.testing.assert_array_equal(ds[name][0],want)
            assert list(ds[name].dimensions)==ORIGINAL[name]['dimensions']
            for attr in ('units','description','MemoryOrder','stagger','coordinates'):
                assert ds[name].getncattr(attr)==ORIGINAL[name]['attrs'][attr]


@pytest.mark.parametrize('mode',['full','operational','subset'])
def test_goddard_native_flux_names_copy_retained_endpoint_values(tmp_path,mode):
    # Original radiation_driver.F erbe_out(1:8) mediation, independently
    # enumerated here rather than obtaining expectations from the writer map.
    pairs=(('TLWDN','LWDNT'),('TLWUP','LWUPT'),('SLWDN','LWDNB'),('SLWUP','LWUPB'),
           ('TSWDN','SWDNT'),('TSWUP','SWUPT'),('SSWDN','SWDNB'),('SSWUP','SWUPB'))
    state,grid,nl=synthetic_case()
    nl.update(ra_lw_physics=5,ra_sw_physics=5)
    diagnostics={source:np.arange(20,dtype=np.float32).reshape(4,5)+i*100 for i,(_,source) in enumerate(pairs)}
    subset=tuple(name for name,_ in pairs) if mode=='subset' else None
    authority=writer_authority(grid,'d01')
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'flux.nc',domain='d01',domain_authority=authority,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,
        diagnostics=diagnostics,full_variable_set=mode=='full',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    with Dataset(payload.target) as ds:
        for name,source in pairs:
            assert name in ds.variables,name
            np.testing.assert_array_equal(ds[name][0],diagnostics[source])
            for attr in ('units','description','MemoryOrder','stagger','coordinates'):
                assert ds[name].getncattr(attr)==ORIGINAL[name]['attrs'][attr]


def test_goddard_native_flux_requires_real_source_and_rejects_nonfinite(tmp_path):
    state,grid,nl=synthetic_case()
    nl.update(ra_lw_physics=5,ra_sw_physics=5)
    kwargs=dict(domain='d01',domain_authority=writer_authority(grid,'d01'),
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=0.,full_variable_set=True)
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'absent.nc',**kwargs)
    assert 'SLWDN' not in payload.fields  # generic full defaults cannot invent it
    with pytest.raises(ValueError,match='nonfinite Goddard native flux SLWDN'):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'bad.nc',diagnostics={'LWDNB':np.full((4,5),np.inf)},**kwargs)
