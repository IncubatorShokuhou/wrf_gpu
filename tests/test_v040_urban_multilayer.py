"""Active BEP/BEM initialization, admission guards and restart state integrity."""
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import urban_multilayer_hook as hook

PROOF=Path(__file__).resolve().parents[1]/'proofs/v040/urban_bep_bem'
sys.path.insert(0,str(PROOF))
from bep_driver_parity import unpack as unpack_bep
from bem_driver_parity import unpack as unpack_bem


def compact(state,bundle):
    n=bundle.geometry.n
    if bundle.option==2:
        return state._replace(roof=state.roof[...,:n,:],wall=state.wall[...,:n,:],
                              wall_heat=state.wall_heat[...,:n],roof_heat=state.roof_heat[...,:n])
    nb=len(bundle.bem.levels);floors=max(bundle.bem.levels)
    return state._replace(roof=state.roof[...,:n,:],wall=state.wall[:,:nb,:,:n,:],
        wall_heat=state.wall_heat[:,:nb,:,:n],roof_heat=state.roof_heat[...,:n],
        temperature=state.temperature[:,:nb,:floors],humidity=state.humidity[:,:nb,:floors],
        window=state.window[:,:nb,:,:n],window_heat=state.window_heat[:,:nb,:,:n],
        floor=state.floor[:,:nb,:,:floors-1,:],under_ground=state.under_ground[:,:nb],
        roof_water=state.roof_water[...,:n],roof_latent=state.roof_latent[...,:n],
        pv_temperature=state.pv_temperature[...,:n],extra={})


@pytest.mark.parametrize('option',[2,3])
def test_compact_initialization_matches_pristine_registry_entries(option):
    from gpuwrf.config.paths import wrf_run_dir
    data=np.load(PROOF/f'savepoints/urban{option}_fp32_ac_on.npz')
    values=data['IN_VALUES'][:,0]
    bundle=hook.load_multilayer_bundle(wrf_run_dir(),option=option,ivgtyp=np.full((6,1),13),isurban=13)
    carry=hook.initial_multilayer_carry(bundle,tslb=values[:,38:42].T.reshape(4,6,1),
        frc_urb2d=values[:,4].reshape(6,1),nz=8,smois=jnp.full((4,6,1),.3,jnp.float32))
    full=(unpack_bep(data,'fp32') if option==2 else unpack_bem(data,'fp32'))[0]
    expected=compact(full,bundle)
    assert jax.tree.structure(carry.state)==jax.tree.structure(expected)
    for a,b in zip(jax.tree.leaves(carry.state),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)
        assert a.dtype==np.float32
    np.testing.assert_array_equal(carry.fraction,data['IN_FRACTION'][:,0])
    # Memory claim is active allocation, measured from actual leaves.
    fullbytes=sum(x.nbytes for x in jax.tree.leaves(full))
    compactbytes=sum(x.nbytes for x in jax.tree.leaves(carry.state))
    assert compactbytes<fullbytes/2


def test_bep_bem_pairing_rejections_name_wrf_reason():
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nl={'physics':{'sf_urban_physics':[2,3],'sf_surface_physics':[4,4],
                  'bl_pbl_physics':[1,2],'ra_sw_physics':[4,4],'ra_lw_physics':[4,4]}}
    errors,_=namelist_honour_report(nl,('d01','d02'))
    assert any('MYJURB' in x and 'd02' in x for x in errors),errors
    assert not any('BEP/BEM' in x and 'd01' in x for x in errors),errors
    nl['physics']['bl_pbl_physics']=[5,8]
    errors,_=namelist_honour_report(nl,('d01','d02'))
    assert any('WRF allows YSU1/MYJ2/BouLac8' in x for x in errors),errors
    errors,_=namelist_honour_report(nl,('d01','d02'),replay=True)
    assert sum('cannot evolve BEP/BEM' in x for x in errors)==2


@pytest.mark.parametrize('option',[2,3])
@pytest.mark.parametrize('pbl',[1,8])
def test_public_admission_and_honour_check(option,pbl):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nl={'physics':{'sf_urban_physics':option,'sf_surface_physics':4,
        'bl_pbl_physics':pbl,'sf_sfclay_physics':1,'ra_sw_physics':4,'ra_lw_physics':4}}
    validate_operational_namelist(nl)
    errors,_=namelist_honour_report(nl,('d01',))
    assert errors==[],errors


@pytest.mark.parametrize('option',[2,3])
@pytest.mark.parametrize('damage',[{'sf_sfclay_physics':5},{'ra_sw_physics':1},
    {'ra_lw_physics':1},{'urban_bundle':SimpleNamespace(option=0)}])
def test_direct_runtime_refuses_unqualified_urban_pairings(option,damage):
    from gpuwrf.runtime.operational_mode import _resolve_operational_suite
    from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection
    from test_v022_g3city_urban_lake import _minimal_operational_namelist
    ns=_minimal_operational_namelist(sf_urban_physics=option,sf_surface_physics=4,
        bl_pbl_physics=1,sf_sfclay_physics=1,use_noahmp=True,rad_rk_tendf=1,
        urban_bundle=SimpleNamespace(option=option))
    _resolve_operational_suite(ns)
    for name,value in damage.items():setattr(ns,name,value)
    reason='revised-MM5' if 'sf_sfclay_physics' in damage else 'BEP'
    with pytest.raises(UnsupportedSchemeSelection,match=reason):
        _resolve_operational_suite(ns)


def test_surface_driver_multilayer_history_is_lowest_model_level():
    data=np.load(PROOF/'savepoints/urban2_fp32_ac_on.npz')
    _,f,_,_=unpack_bep(data,'fp32')
    step=hook.UrbanMultilayerDiagnostics(jnp.arange(6),f.theta[:,0],
        jnp.asarray(data['IN_QV'][:,0,0]),f.u[:,0],f.v[:,0])
    zero=jnp.zeros(6,jnp.float32);psfc=jnp.asarray(data['IN_VALUES'][:,0,37])
    result=hook.multilayer_history_overwrite(zero,zero,zero,zero,step,psfc)
    for actual,key in zip(result,['T2','Q2','U10','V10']):
        ref=data[key][:,0]
        np.testing.assert_allclose(actual,ref,rtol=4*np.finfo(np.float32).eps,atol=0)


@pytest.mark.parametrize('option',[2,3])
def test_urban_restart_missing_or_wrong_carry_fails_closed(option):
    from gpuwrf.io.restart_schema import validate_urban_bep
    from gpuwrf.config.paths import wrf_run_dir
    bundle=hook.load_multilayer_bundle(wrf_run_dir(),option=option,ivgtyp=np.array([[13,7]]),isurban=13)
    carry=hook.initial_multilayer_carry(bundle,tslb=jnp.full((4,1,2),290,jnp.float32),frc_urb2d=jnp.zeros((1,2)),nz=8,
        smois=jnp.full((4,1,2),.3,jnp.float32))
    validate_urban_bep(carry,{'sf_urban_physics':option})
    with pytest.raises(ValueError,match='urban_bep'):
        validate_urban_bep(None,{'sf_urban_physics':option})
    with pytest.raises(ValueError,match='differs from selected'):
        validate_urban_bep(carry,{'sf_urban_physics':5-option})
    from gpuwrf.io.wrfrst_netcdf import PYTREE_CARRY_FIELDS
    assert 'urban_bep' in PYTREE_CARRY_FIELDS


def test_lcz_is_refused_before_column_initialization():
    from gpuwrf.config.paths import wrf_run_dir
    with pytest.raises(NotImplementedError,match='LCZ'):
        hook.load_multilayer_bundle(wrf_run_dir(),option=3,ivgtyp=np.array([[51]]),isurban=13)


@pytest.mark.parametrize('option',[2,3])
def test_lcz_flag_is_refused_even_without_lcz_land_categories(option,monkeypatch):
    # WRF urban_var_init selects UTYPE=5 for ISURBAN when this flag is active.
    from gpuwrf.io.namelist_binding import namelist_honour_report
    from gpuwrf.integration import nested_pipeline as pipeline
    nl={'physics':{'sf_urban_physics':option,'use_wudapt_lcz':1}}
    errors,_=namelist_honour_report(nl,('d01',))
    assert any('use_wudapt_lcz' in error and 'LCZ' in error for error in errors)
    monkeypatch.setattr(pipeline,'_domain_physics_int',lambda run,key,domain,default:1 if key=='use_wudapt_lcz' else default)
    with pytest.raises(NotImplementedError,match='use_wudapt_lcz'):
        pipeline._load_urban_multilayer(None,'d01',None,None,option=option,nz=8)


def test_complete_urban_pbl_gate_against_original(tmp_path):
    import json
    from pbl_parity import run
    result=tmp_path/'pbl.json'
    run(PROOF/'savepoints/pbl',result,budget=PROOF/'evidence/ROUND01.npz')
    data=json.loads(result.read_text())
    assert set(data)=={'ysu','boulac'}
    for scheme in data:
        for precision in ('fp64','fp32'):
            assert data[scheme][precision]['violations']==0,data[scheme]


def test_conditioned_budget_rejects_modified_packet(tmp_path):
    from conditioned_gate import load_budget
    original=PROOF/'evidence/ROUND01.npz'
    assert load_budget(original)
    changed=tmp_path/'budget.npz'
    changed.write_bytes(original.read_bytes()+b'changed')
    with pytest.raises(AssertionError):load_budget(changed)


@pytest.mark.parametrize('option',[2,3])
def test_full_canopy_driver_original_wrf_gate(tmp_path,option):
    """Full six-step state, source and diagnostic gate; unchanged REAL8/state."""
    import json
    result=tmp_path/'full.json'
    if option==2:
        from bep_driver_parity import run
        run(PROOF/'savepoints',result,budget=PROOF/'evidence/ROUND01.npz')
    else:
        from bem_driver_parity import run
        run(PROOF/'savepoints',result,('fp64','fp32'),6,budget=PROOF/'evidence/ROUND01.npz')
    for arm,record in json.loads(result.read_text()).items():
        assert record['violations']==0,(arm,record)


@pytest.mark.parametrize('case',range(1,7))
def test_selected_surface_layer_is_revised_mm5_in_real_parent(case):
    """New caller uses the existing SF1 frozen WRF savepoint gates, not MYNN."""
    import json
    from gpuwrf.coupling.urban_surface_layer import revised_surface_layer
    from test_v060_sfclay_revised_mm5 import FIELD_TOLERANCES
    data=json.loads((PROOF.parents[1]/'v060/savepoints'/f'sfclayrev1_case_{case}.json').read_text())
    c=data['columns'];arr=lambda key:jnp.asarray(c[key],jnp.float32).reshape(1,-1)
    state=SimpleNamespace(u=arr('U'),v=arr('V'),t_air=arr('T'),qv=arr('QV'),p=arr('P'),dz=arr('DZ'),
        psfc=arr('PSFC'),t_skin=arr('TSK'),xland=arr('XLAND'),lakemask=arr('LAKEMASK'),mavail=arr('MAVAIL'),
        roughness_m=arr('ZNT_IN'),ustar=arr('UST_IN'),mol=arr('MOL_IN'),hfx=arr('HFX_IN'),qfx=arr('QFX_IN'),
        qsfc=arr('QSFC_IN'),pblh=arr('PBLH'),dx_m=arr('DX'))
    jax.config.update('jax_enable_x64',True)
    # Pass the actual leaves dynamically; the parent remains wide for DOUBLE islands.
    names=tuple(vars(state))
    fn=jax.jit(lambda *args:revised_surface_layer(SimpleNamespace(**dict(zip(names,args)))))
    out=fn(*(getattr(state,name) for name in names))
    values={key:getattr(out,key.lower()) for key in ['HFX','LH','U10','V10','TH2','T2','Q2','QSFC','MOL','RMOL','ZOL','REGIME','PSIM','PSIH','BR','ZNT','CHS','CHS2','CQS2']}
    values.update(UST=out.fluxes.ustar,WSPD=out.fluxes.wspd,QFX=out.fluxes.qv_flux*out.fluxes.rhosfc)
    for name,actual in values.items():
        ref=np.asarray(c[name]);error=np.max(abs(np.asarray(actual).ravel()-ref))
        atol,rtol=FIELD_TOLERANCES[name]
        assert error<=atol or error<=rtol*max(np.max(abs(ref)),1.e-12),(case,name,error,atol,rtol)
        if name!='REGIME':assert actual.dtype==np.float32,(name,actual.dtype)


@pytest.mark.parametrize('option',[2,3])
@pytest.mark.parametrize('fmt',['pickle','netcdf'])
def test_complete_urban_restart_roundtrips_and_rejects_missing_state(tmp_path,monkeypatch,option,fmt):
    from dataclasses import replace
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.validation.tier2 import make_ideal_grid
    from test_v0110_wrfrst_netcdf import _state
    from test_restart_schema_held_physics import exact,save,read
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re,'_MP_RE',False)
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    grid=make_ideal_grid(nz=8,ny=3,nx=4);state=_state(grid)
    ivg=np.full((3,4),14);ivg[1,1]=13
    bundle=hook.load_multilayer_bundle(wrf_run_dir(),option=option,ivgtyp=ivg,isurban=13)
    urban=hook.initial_multilayer_carry(bundle,tslb=jnp.full((4,3,4),290,jnp.float32),
        frc_urb2d=jnp.zeros((3,4)),nz=8,smois=jnp.full((4,3,4),.3,jnp.float32))
    zero=jnp.zeros((3,4),jnp.float32)
    rad=RRTMGRadiationDiagnostics(**{name:zero for name in RRTMGRadiationDiagnostics._fields
                                     if name not in RRTMGRadiationDiagnostics._field_defaults},
                                urban_radiation=(zero,zero,zero,zero))
    carry=initial_operational_carry(state).replace(urban_bep=urban,radiation_diagnostics=rad)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,sf_urban_physics=option,
                          urban_bundle=bundle,bl_pbl_physics=1,sf_sfclay_physics=1)
    path=tmp_path/f'urban{option}_{fmt}'
    save(path,fmt,grid,nl,carry);exact(carry,read(path,fmt))
    with pytest.raises(ValueError,match='urban_bep'):
        save(path,fmt,grid,nl,carry.replace(urban_bep=None))
    with pytest.raises(ValueError,match='direct/diffuse'):
        save(path,fmt,grid,nl,carry.replace(radiation_diagnostics=rad._replace(urban_radiation=None)))
    with pytest.raises(ValueError,match='surface-layer diagnostics'):
        save(path,fmt,grid,nl,carry.replace(urban_bep=urban._replace(surface_layer=None)))
