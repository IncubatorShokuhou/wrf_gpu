from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ocean_mixed_layer import initialize
from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
from gpuwrf.io.ocean_schema import validate
from gpuwrf.io.netcdf_lock import Dataset
from test_v013_operational_smoke import _grid,_base_state,_namelist
from test_m7_netcdf_writer import authenticated_source_grid


def scene():
    grid=_grid(nz=4,ny=2,nx=3)
    state=_base_state(grid).replace(t_skin=jnp.full((2,3),300.,jnp.float32),xland=jnp.full((2,3),2.,jnp.float32))
    ocean=initialize(state.t_skin)
    n=replace(_namelist(grid),sf_ocean_physics=1,isftcflx=2,ocean_initial=ocean)
    carry=initial_operational_carry(state).replace(ocean=ocean,hurricane_ustm=jnp.full((2,3),.35,jnp.float32),history_diagnostics=seed_history_diagnostics(state))
    return grid,state,n,carry


def test_ocean_controls_namelist_pytree_and_honour():
    from gpuwrf.io.namelist_binding import bind_namelist_options,namelist_honour_report
    from gpuwrf.io.namelist_check import validate_operational_namelist
    config={'physics':{'sf_ocean_physics':1,'sf_surface_physics':4,'sf_sfclay_physics':5,'isftcflx':2,
                       'oml_hml0':30.,'oml_gamma':.11,'oml_relaxation_time':7200.},'dynamics':{'diff_opt':1,'km_opt':4}}
    grid,_,n,_=scene()
    validate_operational_namelist(config)
    n=bind_namelist_options(n,config,'d01')
    assert (n.sf_ocean_physics,n.oml_hml0,n.oml_gamma,n.oml_relaxation_time,n.isftcflx)==(1,30.,.11,7200.,2)
    leaves,tree=jax.tree.flatten(n)
    restored=jax.tree.unflatten(tree,leaves)
    for a,b in zip(restored.ocean_initial,n.ocean_initial,strict=True):assert a is b
    other=replace(n,ocean_initial=n.ocean_initial._replace(tml=n.ocean_initial.tml+jnp.float32(1)))
    assert tree==jax.tree.structure(other)  # different case values reuse static program key
    assert tree==jax.tree.structure(n)
    assert restored.oml_gamma==.11 and restored.isftcflx==2
    assert namelist_honour_report(config,('d01',))[0]==[]
    for key,value,word in (('sf_ocean_physics',2,'PWP'),('oml_gamma',0.,'Gamma'),('isftcflx',3,'3/4'),('sst_update',1,'sst_update'),('sf_lake_physics',1,'LakeModel')):
        bad={**config,'physics':dict(config['physics'],**{key:value})}
        errors,_=namelist_honour_report(bad,('d01',))
        assert any(word in e for e in errors),errors


def test_pristine_caller_wind_and_flux_ownership(monkeypatch):
    import gpuwrf.coupling.ocean_surface_hook as hook
    import gpuwrf.physics.ocean_mixed_layer as kernel
    grid,state,n,carry=scene()
    entry=state.replace(u=jnp.full_like(state.u,65),v=jnp.full_like(state.v,-20))
    state=state.replace(u=jnp.full_like(state.u,-3),v=jnp.full_like(state.v,7),hfx=jnp.full((2,3),250.,jnp.float32),
                        qfx=jnp.full((2,3),.00028,jnp.float32),ustar=jnp.full((2,3),2.3,jnp.float32))
    rad=SimpleNamespace(swnorm=jnp.full((2,3),800.),swup_topographic=jnp.full((2,3),100.),glw=jnp.full((2,3),380.))
    land=SimpleNamespace(emiss=jnp.full((2,3),.98))
    # Only the callable is replaced: actual field selection/conversion is tested.
    calls=[]
    def record(c,**kw):
        calls.append(kw)
        return c
    monkeypatch.setattr(kernel,'step',record)
    class Land:
        emiss=jnp.full((2,3),.98)
        albedo=jnp.full((2,3),.1)
        t_skin=state.t_skin
        def replace(self,**kw):return self
    # An independent LH value deliberately differs from QFX*XLV, exposing an
    # incorrect caller reconstruction (the port's QFX passes rho conversions).
    held=carry.history_diagnostics._replace(lh=jnp.full((2,3),812.,jnp.float32))
    out,new=hook.ocean_surface_step(entry,state,carry.replace(radiation_diagnostics=rad,noahmp_land=Land(),history_diagnostics=held),n)
    assert len(calls)==1
    for name,expected in (('u',65.),('v',-20.),('hfx',250.),('lh',812.),('gsw',700.),('glw',380.)):
        np.testing.assert_allclose(calls[0][name],expected,rtol=1e-7,atol=1e-4,err_msg=name)
    np.testing.assert_array_equal(out.qv,state.qv)
    assert new.ocean is carry.ocean


def test_restart_pickle_and_checkpoint_missing_fail_closed(tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    grid,state,n,carry=scene()
    path=tmp_path/'ocean.pkl'
    write_restart(carry,n,grid,5,path)
    out,_,_,step=read_restart(path)
    assert step==5
    checked,_,_,_=read_restart(path,namelist=n)
    np.testing.assert_array_equal(checked.ocean.tml,carry.ocean.tml)
    with pytest.raises(ValueError,match='configuration'):
        read_restart(path,namelist=replace(n,oml_gamma=.2))
    for a,b in zip(jax.tree.leaves((carry.ocean,carry.hurricane_ustm)),jax.tree.leaves((out.ocean,out.hurricane_ustm)),strict=True):
        np.testing.assert_array_equal(a,b)
    with path.open('rb') as f:payload=pickle.load(f)
    payload['carry']['extra_fields']['ocean']=None
    with path.open('wb') as f:pickle.dump(payload,f)
    with pytest.raises(ValueError,match='E78'):read_restart(path)
    path=tmp_path/'checkpoint.pkl'
    write_checkpoint(state,n,grid,5,path,runtime_state=carry)
    restored=read_checkpoint_with_runtime_state(path)
    assert restored[-1].ocean is not None
    with path.open('rb') as f:payload=pickle.load(f)
    payload.pop('ocean_schema_version')
    with path.open('wb') as f:pickle.dump(payload,f)
    with pytest.raises(ValueError,match='E78'):read_checkpoint_with_runtime_state(path)


def test_netcdf_restart_history_and_schema(tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf,bind_wrfout_domain_authority
    grid,state,n,carry=scene()
    stamp=datetime(2026,7,25,tzinfo=timezone.utc)
    path=tmp_path/'ocean.nc'
    write_wrfrst_carry(carry,grid,n,path,step_index=5,run_start=stamp,valid_time=stamp,lead_hours=0.)
    out,_=read_wrfrst_carry(path,namelist=n)
    for a,b in zip(carry.ocean,out.ocean,strict=True):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(carry.hurricane_ustm,out.hurricane_ustm)
    with pytest.raises(ValueError,match='configuration'):
        read_wrfrst_carry(path,namelist=replace(n,oml_gamma=.2))
    from gpuwrf.coupling.ocean_surface_hook import history
    source=authenticated_source_grid(grid,'d01')
    wrfout=tmp_path/'history.nc'
    write_wrfout_netcdf(state,grid,n,wrfout,domain='d01',domain_authority=bind_wrfout_domain_authority('d01',source,grid),
        run_start=stamp,valid_time=stamp,lead_hours=0.,diagnostics=history(carry.ocean))
    with Dataset(wrfout) as ds:
        assert ds.SF_OCEAN_PHYSICS==1 and ds.OML_GAMMA==np.float32(n.oml_gamma) and ds.ISFTCFLX==2
        assert np.asarray(ds.SF_OCEAN_PHYSICS).dtype==np.int32 and np.asarray(ds.OML_GAMMA).dtype==np.float32
        for name,value in carry.ocean._asdict().items():np.testing.assert_array_equal(ds[name.upper()][0],value)
    with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_OCEAN_SCHEMA_VERSION')
    with pytest.raises(ValueError,match='E78'):read_wrfrst_carry(path,namelist=n)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_schema_rejects_nonfinite(bad):
    _,_,n,c=scene()
    with pytest.raises(ValueError,match='E78'):validate(c.ocean._replace(tml=c.ocean.tml.at[0,0].set(bad)),c.hurricane_ustm,n,1)
    with pytest.raises(ValueError,match='E78'):validate(c.ocean,c.hurricane_ustm.at[0,0].set(bad),n,1)


def test_restart_missing_surface_lh_fails_closed(tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart
    grid,state,n,c=scene()
    with pytest.raises(ValueError,match='LH'):
        write_restart(c.replace(history_diagnostics=None),n,grid,0,tmp_path/'bad.pkl')
    path=tmp_path/'valid.pkl';write_restart(c,n,grid,0,path)
    with path.open('rb') as f:payload=pickle.load(f)
    payload['carry']['extra_fields']['history_diagnostics']=None
    with path.open('wb') as f:pickle.dump(payload,f)
    with pytest.raises(ValueError,match='LH'):read_restart(path)


@pytest.mark.parametrize('hml0',[50.,0.,-1.])
def test_input_binding_real_netcdf(tmp_path,hml0):
    from gpuwrf.io.ocean_init import bind_ocean_model
    grid,state,n,_=scene()
    n=replace(n,oml_hml0=hml0)
    with Dataset(tmp_path/'wrfinput_d01','w') as ds:
        for name,length in (('Time',1),('south_north',2),('west_east',3)):ds.createDimension(name,length)
        for name,value in (('H0ML',22.),('HUML',3.),('HVML',4.),('TMOML',200.5)):
            ds.createVariable(name,'f4',('Time','south_north','west_east'))[0]=value
    out=bind_ocean_model(n,tmp_path,'d01',state=state)
    np.testing.assert_array_equal(out.ocean_initial.hml,50. if hml0>0 else 22.)
    np.testing.assert_array_equal(out.ocean_initial.tmoml,300. if hml0<0 else 295.)


def test_both_real_loaders_bind_and_initialize(monkeypatch,tmp_path):
    from test_namelist_binding import _load_nested,_build_daily_case
    import gpuwrf.io.ocean_init as init
    calls=[]
    initial=initialize(jnp.full((2,3),302.,jnp.float32))
    def seed(n,run_dir,domain,*,state):
        calls.append((domain,n.sf_ocean_physics,n.oml_hml0,n.oml_gamma,n.oml_relaxation_time,n.isftcflx))
        return replace(n,ocean_initial=initial)
    monkeypatch.setattr(init,'bind_ocean_model',seed)
    nested_dir=tmp_path/'nested';nested_dir.mkdir()
    extra=' sf_ocean_physics=1,\n oml_hml0=22.,\n oml_gamma=.11,\n oml_relaxation_time=7200.,\n isftcflx=2,\n'
    bundles=_load_nested(monkeypatch,nested_dir,{},extra_physics=extra)
    for domain in ('d01','d02','d03'):
        assert bundles[domain].namelist.ocean_initial is initial
        assert (domain,1,22.,.11,7200.,2) in calls
    daily_dir=tmp_path/'daily';daily_dir.mkdir()
    n=_build_daily_case(monkeypatch,daily_dir,extra+' sf_surface_physics=0,\n')
    assert n.ocean_initial is initial and (n.sf_ocean_physics,n.oml_gamma,n.isftcflx)==(1,.11,2)
    assert len(calls)==4


@pytest.mark.parametrize('sf',[1,5,91])
@pytest.mark.parametrize('opt',[1,2])
def test_hurricane_real_adapter_against_pristine_inputs(monkeypatch,sf,opt):
    """Actual State/column/diagnostic hook, with the independent WRF columns."""
    import json
    from pathlib import Path
    import gpuwrf.coupling.noahmp_surface_hook as noah
    from gpuwrf.coupling.hurricane_surface_hook import surface_step
    from test_hurricane_flux import INPUTS,OUTPUTS
    root=Path(__file__).resolve().parents[1]/'proofs/s2_oml/fixtures'
    data=json.loads((root/'flux.json').read_text())
    fma=json.loads((root/'flux_fma.json').read_text())
    key=f'{ {1:"revised",5:"mynn",91:"old"}[sf]}_{opt}'
    d=data[key]
    if sf==5:
        inp=dict(zip(INPUTS,np.asarray(d['inputs'],np.float32).T,strict=True))
        expected=dict(zip(OUTPUTS,np.asarray(d['output'],np.float32).T,strict=True))
        bounds=dict(zip(OUTPUTS,np.asarray(d['input_ulp_bound'],np.float32).T,strict=True))
        fma_values=dict(zip(OUTPUTS,np.asarray(fma[key]['output'],np.float32).T,strict=True))
    else:
        names=('U','V','T','QV','P','DZ','PSFC','TSK','XLAND','LAKEMASK','MAVAIL','ZNT_IN','UST_IN','MOL_IN','HFX_IN','QFX_IN','QSFC_IN','PBLH','DX')
        inp=dict(zip(('u','v','temperature','qv','pressure','dz','psfc','tsk','xland','lakemask','mavail','znt','ust','mol','hfx','qfx','qsfc','pblh','dx'),
                     (np.asarray(d[k],np.float32) for k in names),strict=True))
        inp['rho']=np.full(8,1.14,np.float32)
        expected={k.lower():np.asarray(v,np.float32) for k,v in d.items() if k!='input_ulp_bound'}
        bounds={k.lower():np.asarray(v,np.float32) for k,v in d['input_ulp_bound'].items()}
        fma_values={k.lower():np.asarray(v,np.float32) for k,v in fma[key].items() if k!='input_ulp_bound'}
    grid=_grid(nz=4,ny=2,nx=4);state=_base_state(grid)
    surface=('rho','mavail','pblh','xland','psfc','mol','qsfc','hfx','qfx')
    # Actual hook calls the shared grid-backed phy_prep view; this seam supplies
    # exact oracle columns, isolating the new caller from unrelated IC changes.
    view=SimpleNamespace(**{k:jnp.asarray(inp[k]).reshape(2,4) for k in surface})
    view.rho=view.rho[...,None]
    for dst,src in (('t_skin','tsk'),('roughness_m','znt'),('ustar','ust')):setattr(view,dst,jnp.asarray(inp[src]).reshape(2,4))
    for dst,src in (('u','u'),('v','v'),('theta','temperature'),('t_air','temperature'),('qv','qv'),('p','pressure'),('dz','dz')):
        setattr(view,dst,jnp.asarray(inp[src]).reshape(2,4,1))
    view.lakemask=jnp.asarray(inp.get('lakemask',np.zeros(8,np.float32))).reshape(2,4)
    view.dx_m=jnp.asarray(inp.get('dx',np.full(8,9000,np.float32))).reshape(2,4)
    monkeypatch.setattr(noah,'_build_column_view',lambda *_args:view)
    # The two MM5 drivers have differing per-column DX. Pass that WRF operand
    # exactly, via the hook's GridSpec projection (the production value is scalar).
    n=replace(_namelist(grid),sf_sfclay_physics=sf,isftcflx=opt,use_noahmp=False)
    n=replace(n,grid=SimpleNamespace(projection=SimpleNamespace(dx_m=view.dx_m)))
    c=initial_operational_carry(state).replace(hurricane_ustm=jnp.full((2,4),.35,jnp.float32))
    out,new,diag=surface_step(state,c,n)
    for name,value in (('hfx',diag.hfx),('lh',diag.lh),('t2',diag.t2),('q2',diag.q2),('ustm',new.hurricane_ustm)):
        difference=np.abs(np.asarray(value).ravel()-expected[name])
        bound=bounds[name]+np.abs(fma_values[name]-expected[name])
        assert np.all(difference<=bound),(sf,opt,name,difference,bound)
    np.testing.assert_array_equal(out.qv,state.qv)
    assert new.hurricane_ustm.dtype==jnp.float32
