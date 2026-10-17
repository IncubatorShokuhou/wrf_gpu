"""Actual glacier partition/caller seams; independent complete WRF fixtures."""
from pathlib import Path
import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts.noahmp_state import NoahMPLandState,NoahMPStatic,NoahMPFluxes
from gpuwrf.physics.noahmp.types import NoahMPForcing
from gpuwrf.io.noahmp_glacier_options import GlacierStatic
from gpuwrf.coupling import glacier_surface_hook as hook

ROOT=Path(__file__).resolve().parents[1]/'proofs/s2_glacier'
MANIFEST=json.loads((ROOT/'oracle_manifest.json').read_text())


def setup(gla=1):
    data=np.load(ROOT/'glacier_oracle.npz');idx=np.flatnonzero((data['names']=='accumulation')&np.all(data['options']==(2,1,2,1,gla),axis=1))[0]
    r={f['name']:data['inputs'][idx,f['offset']:f['offset']+f['extent']] for f in MANIFEST['fields']}
    plane=lambda a:jnp.broadcast_to(jnp.asarray(a,jnp.float32).reshape(-1,1,1),(np.size(a),1,2))
    values={k:jnp.zeros((1,2),jnp.int32 if k=='isnow' else jnp.float32) for k in NoahMPLandState.__slots__}
    for attr,name in {'tslb':'stc','smois':'smc','sh2o':'sh2o','snice':'snice','snliq':'snliq','zsnso':'zsnso'}.items():
        values[attr]=plane(r[name][3:] if attr=='tslb' else r[name])
    values['tsno']=plane(r['stc'][:3])
    for key in ('snowh','sneqv','sneqvo','albold','cm','ch','tg','tauss','qsfc'):
        values[key]=plane(r[key])[0]
    values['isnow']=plane(r['isnow'])[0].astype(jnp.int32);values['glacier_qsnow']=plane(r['qsnow'])[0]
    land=NoahMPLandState(**values)
    config=GlacierStatic((1,),(0,),(1,2),opt_gla=gla)
    static=NoahMPStatic(jnp.asarray([[10,15]],jnp.int32),jnp.ones((1,2),jnp.int32),
        jnp.ones((1,2)),jnp.ones((1,2)),jnp.zeros((1,2)),jnp.asarray([[10,15]],jnp.int32),
        plane(r['tbot'])[0],jnp.asarray([.1,.3,.6,1.],jnp.float32),jnp.asarray(r['zsoil']),
        jnp.zeros((1,2)),1000.,SimpleNamespace(isice=15),glacier=config)
    f={name:jnp.zeros((1,2),jnp.float32) for name in NoahMPForcing._fields[:18]}
    for attr,name in {'sfctmp':'sfctmp','sfcprs':'sfcprs','psfc':'sfcprs','uu':'uu','vv':'vv','qair':'q2',
                      'soldn':'soldn','lwdn':'lwdn','prcpnonc':'prcp','cosz':'cosz','zlvl':'zlvl'}.items():f[attr]=plane(r[name])[0]
    f['julian']=jnp.float32(15.);f['yearlen']=jnp.float32(365.)
    return land,NoahMPForcing(**f),static,r,data['reference'][idx]


@pytest.mark.parametrize('gla',(1,2))
def test_real_caller_partition_keeps_natural_points_and_previous_qsnow(gla,monkeypatch):
    land,forcing,static,r,reference=setup(gla)
    seen={}
    original=hook.glacier_kernel
    def kernel(inputs,config,**kw):
        seen['qsnow']=np.asarray(inputs['qsnow']);seen['gla']=config.opt_gla
        return original(inputs,config,**kw)
    monkeypatch.setattr(hook,'glacier_kernel',kernel)
    def natural(l,f,s,dt,**kw):
        assert l.tg.shape==(1,1) and np.asarray(s.ivgtyp).item()==10
        seen['natural']=True
        zero=jnp.zeros_like(l.tg)
        flux=NoahMPFluxes(zero,zero,zero,zero,l.t_skin+1.,l.qsfc,l.znt,l.emiss,l.albedo,zero,
                         t2=zero,t2mv=zero,t2mb=zero,q2=zero)
        return l.replace(tg=l.tg+1.),flux
    result,flux=hook.route_glacier_step(natural,land,forcing,static,18.,energy_params=None,rad_params=None)
    assert seen['natural'] and seen['gla']==gla
    np.testing.assert_array_equal(seen['qsnow'],np.asarray([[r['qsnow'][0]]],np.float32))
    np.testing.assert_array_equal(np.asarray(result.tg)[:,0],np.asarray(land.tg)[:,0]+1.)
    offsets={f['name']:f['offset'] for f in MANIFEST['fields']}
    for attr,name in [('tg','tg'),('snowh','snowh'),('sneqv','sneqv'),('glacier_qsnow','qsnow')]:
        np.testing.assert_allclose(np.asarray(getattr(result,attr))[0,1],reference[offsets[name]],rtol=2.e-6,atol=1.e-5)
    np.testing.assert_allclose(np.asarray(flux.hfx)[0,1],reference[offsets['fsh']],rtol=2.e-6,atol=1.e-4)
    assert np.asarray(result.canliq)[0,1]==np.asarray(result.canice)[0,1]==0
    assert np.asarray(result.tv)[0,1]==np.float32(-1.e36)


def test_ice_free_tree_keeps_released_leaves_and_aux():
    land,_,static,_,_=setup()
    inactive=land.replace(glacier_qsnow=None)
    children,aux=inactive.tree_flatten()
    assert len(children)==32 and aux is None
    children,aux=static.replace(glacier=None).tree_flatten()
    assert len(children)==14 and aux is None


def test_missing_held_glacier_rate_is_refused_before_kernel():
    land,f,s,_,_=setup()
    with pytest.raises(ValueError,match='QSNOWXY'):
        hook.route_glacier_step(None,land.replace(glacier_qsnow=None),f,s,18.,energy_params=None,rad_params=None)


def test_glacier_options_have_explicit_step_specs():
    from gpuwrf.contracts.physics_interfaces import scheme_step_spec
    for gla in (1,2):
        spec=scheme_step_spec('land_surface',4,f'opt_gla={gla}')
        assert spec.wrf_slot=='first_rk_surface_driver'
        assert 'glacier_qsnow' in spec.reads_carry and 'glacier_qsnow' in spec.writes_carry
        assert 'savepoint' in spec.oracle


def test_operational_adapter_routes_glacier_tile(monkeypatch):
    """Deletion-sensitive public adapter seam, separate from column physics."""
    from test_noahmp_coupler import _build
    from gpuwrf.physics import noahmp_coupler as cp
    state,land,static,rad,clock=_build()
    static=static.replace(glacier=GlacierStatic((0,),(1,2,3,4),(1,5)))
    seen=[]
    def natural(*args,**kwargs):
        raise AssertionError('natural path selected instead of glacier routing')
    def route(step,l,f,s,dt,**kwargs):
        seen.append(s.glacier)
        zero=jnp.zeros_like(l.tg)
        return l,NoahMPFluxes(zero,zero,zero,zero,l.t_skin,l.qsfc,l.znt,l.emiss,l.albedo,zero,
                             t2=l.t_skin,t2mv=l.t_skin,t2mb=l.t_skin,q2=zero)
    monkeypatch.setattr(cp,'noah_mp_step',natural)
    monkeypatch.setattr(hook,'route_glacier_with_lakes',route)
    cp.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=18.)
    assert seen==[static.glacier]


def test_lake_partition_composes_without_running_soil_on_either_special_tile():
    land,f,s,_,_=setup()
    lake=SimpleNamespace(indices=(0,),nonlake_indices=(1,),shape=(1,2))
    def forbidden(*args,**kwargs):
        raise AssertionError('lake/glacier point entered natural soil')
    result,flux=hook.route_glacier_with_lakes(forbidden,land,f,s,18.,energy_params=None,rad_params=None,
        lake_bundle=lake,frozen_land=land)
    for key in land.__slots__:
        a=np.asarray(getattr(land,key));b=np.asarray(getattr(result,key))
        np.testing.assert_array_equal(a[...,0],b[...,0],err_msg=key)
    assert np.asarray(result.tv)[0,1]==np.float32(-1.e36)
    assert np.all(np.asarray(result.smois)[:,0,1]==1.)
    assert np.isfinite(np.asarray(flux.hfx)).all()


@pytest.mark.parametrize('name',['opt_gla','opt_alb','opt_snf','opt_tbot','opt_stc'])
def test_glacier_option_boundaries_fail_closed(name):
    from gpuwrf.io.noahmp_glacier_options import glacier_options
    with pytest.raises(ValueError,match=name):glacier_options({'noah_mp':{name:99}})
    with pytest.raises(ValueError,match='integer'):glacier_options({'noah_mp':{name:True}})
