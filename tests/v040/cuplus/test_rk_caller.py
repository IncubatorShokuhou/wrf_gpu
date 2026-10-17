"""Actual pristine CPS rates through the production RK source seam."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib
import numpy as np
import pytest
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state,_namelist

FIXTURE=Path(__file__).resolve().parents[3]/'proofs/v040/cuplus/full_callers'


def original(group,pbl=None,case=1):
    manifest=json.loads((FIXTURE/'manifest.json').read_text())
    file=FIXTURE/'full_callers.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==manifest['fixture_sha256']
    row=next(row for row in manifest['rows'] if row['group']==group and row['case']==case and row['step']==1
             and (pbl is None or row['pbl']==pbl))
    with np.load(file,allow_pickle=False) as data:
        values={key:np.array(data[row['key']+'_'+key]) for key in row['profiles']+row['scalars']}
    return values


@pytest.mark.parametrize('scheme,pbl',[(11,5),(10,5),(7,9),(7,2)])
@pytest.mark.parametrize('specified',[True,False])
def test_pristine_rates_enter_rk_without_euler_increment(monkeypatch,scheme,pbl,specified):
    from gpuwrf.coupling import cuplus,scan_adapters as adapters
    from gpuwrf.coupling.physics_couplers import MynnPBLSourceLeaves
    import gpuwrf.runtime.operational_mode as om
    raw=original({11:'mskf_vector',10:'cup',7:'zm_vector'}[scheme],
                 pbl if scheme==7 else None,case=1 if scheme==10 else 3)
    nz=len(raw['RTHCUTEN']);grid=_grid(nz=nz)
    if specified:grid=replace(grid,bc=replace(grid.bc,source='wrfbdy'))
    state=_state(grid)
    nml=_namelist(grid,cu_physics=scheme,bl_pbl_physics=pbl,sf_sfclay_physics=2 if pbl==2 else 5,
                  mp_physics=8,rad_rk_tendf=1,use_noahmp=False,run_boundary=specified)
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_CARRY_REAL_ALL','1')
    monkeypatch.setattr(om,'_microphysics_wrf_order_enabled',lambda:True)
    def surface(s,*a,**kw):return (s,jnp.full_like(s.t_skin,-.1,dtype=jnp.float32)) if kw.get('return_cumulus_inputs') else s
    monkeypatch.setattr(om,'surface_adapter',surface)
    # sf=2 runs ONE MYJSFC call with held memory (t-cfix-sfc F12): keep that real call and
    # tag its equivalent-roughness winds so their history retention stays observable.
    from gpuwrf.coupling import myj_surface as myjsfc
    real_myj=myjsfc.myj_surface_step
    def myj_surface(*a,**kw):
        s,held,noah,out=real_myj(*a,**kw)
        out=dict(out,u10e=jnp.full_like(s.t_skin,.25,dtype=jnp.float32),v10e=jnp.full_like(s.t_skin,-.5,dtype=jnp.float32))
        return s,held,noah,out
    monkeypatch.setattr(myjsfc,'myj_surface_step',myj_surface)
    zero=jnp.zeros_like(state.theta,dtype=jnp.float32)
    monkeypatch.setattr(om,'mynn_adapter_with_source_leaves',lambda s,*a,**kw:MynnPBLSourceLeaves(s,zero,zero,zero,zero))
    tp_kp=(jnp.full_like(state.t_skin,.3,dtype=jnp.float32),jnp.full(state.t_skin.shape,6,jnp.int32))
    pblrates={key:zero for key in ('rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten','rqniblten')}
    def cam(s,*a,camuw=None,**kw):
        assert kw['return_tendencies'] and kw['return_cumulus_inputs']
        return s,camuw,pblrates,tp_kp
    monkeypatch.setattr(adapters,'camuw_pbl_adapter',cam)
    monkeypatch.setattr(om,'myj_pbl_adapter',lambda s,*a,**kw:(s,pblrates,tp_kp))
    carry=om._initial_carry_for_run(state,nml)
    def driver(before,surface,grid,old,nml,step,**kw):
        if scheme==7:
            np.testing.assert_array_equal(np.asarray(kw['pbl_inputs'][1]),np.asarray(tp_kp[1]))
        if scheme==11:assert kw['zol'] is not None
        held=dict(old.held)
        aliases={'DUDT':'RUCUTEN','DVDT':'RVCUTEN','ISHALL':'SHALL','CLDFRA_DP_KF':'CLDFRA_DP','CLDFRA_SH_KF':'CLDFRA_SH'}
        for key,value in held.items():
            name=aliases.get(key,key)
            if name in raw:
                held[key]=jnp.broadcast_to(jnp.asarray(raw[name],value.dtype),value.shape)
        return cuplus.CuPlusCarry(old.w0avg,held)
    monkeypatch.setattr(cuplus,'driver',driver)
    forcing=om._physics_step_forcing(carry,nml,0.,run_radiation=False)
    if pbl==2:
        for name,value in (('u10e',.25),('v10e',-.5)):
            np.testing.assert_array_equal(np.asarray(getattr(forcing.carry.history_diagnostics,name)),
                                          np.full(state.t_skin.shape,value,np.float32))
    for name in ('theta','u','v','qv','qc','qi','Ni'):
        np.testing.assert_array_equal(np.asarray(getattr(forcing.state,name)),np.asarray(getattr(carry.state,name)))
    f=np.float32;metrics=nml.metrics
    mass=np.asarray(metrics.c1h,f)[:,None,None]*np.asarray(carry.state.mu_total,f)[None]+np.asarray(metrics.c2h,f)[:,None,None]
    expected={}
    for name,key in (('theta','RTHCUTEN'),('qv','RQVCUTEN'),('qc','RQCCUTEN'),('qi','RQICUTEN'),('qr','RQRCUTEN'),('qs','RQSCUTEN'),('Ni','RQINCUTEN')):
        if key in raw and not (scheme==7 and name in ('qr','qs')):
            a=mass*raw[key][:,None,None]
            expected[name]=np.zeros_like(a) if specified else a.copy()
            if specified:expected[name][:,1:-1,1:-1]=a[:,1:-1,1:-1]
    factor=f(1.)+f(f(461.6)/f(287.))*np.asarray(carry.state.qv,f)
    heat_term=factor*expected['theta']
    vapor_term=f(f(461.6)/f(287.))*np.asarray(carry.state.theta,f)/factor*expected['qv']
    heat=heat_term+vapor_term
    # Eight rounded source operations: use the two independent input-term
    # scales under cancellation, as the existing pristine CAM-UW seam does.
    # This is an input-ULP bound, not a relative error on the small difference.
    bound=8*np.finfo(f).eps*(np.abs(heat_term)+np.abs(vapor_term))
    assert np.all(np.abs(np.asarray(forcing.dry_tendencies.t_tendf)-heat)<=bound)
    for name,value in expected.items():
        if name!='theta':np.testing.assert_array_equal(np.asarray(forcing.dry_tendencies.moist_tendf[name]),value)
    if scheme in (7,11):
        for key,actual,axis in (('RUCUTEN',forcing.dry_tendencies.ru_tendf,2),('RVCUTEN',forcing.dry_tendencies.rv_tendf,1)):
            assert np.max(np.abs(raw[key]))>0.,key
            a=mass*raw[key][:,None,None]
            out=np.zeros_like(np.asarray(actual))
            if axis==2:
                if specified:out[:,1:-1,1:-1]=f(.5)*(a[:,1:-1,:-1]+a[:,1:-1,1:])
                else:
                    out[:,:,1:-1]=f(.5)*(a[:,:,:-1]+a[:,:,1:])
                    out[:,:,0]=f(.5)*(a[:,:,-1]+a[:,:,0]);out[:,:,-1]=out[:,:,0]
            else:
                if specified:out[:,1:-1,1:-1]=f(.5)*(a[:,:-1,1:-1]+a[:,1:,1:-1])
                else:
                    out[:,1:-1,:]=f(.5)*(a[:,:-1,:]+a[:,1:,:])
                    out[:,0,:]=f(.5)*(a[:,-1,:]+a[:,0,:]);out[:,-1,:]=out[:,0,:]
            np.testing.assert_array_equal(np.asarray(actual),out)
    if scheme==7:
        assert np.max(np.abs(raw['RQINCUTEN']))>0.  # genuine number source, not a zero-path check
        assert 'Nc' not in forcing.dry_tendencies.moist_tendf  # original P_QNC inactive for Thompson
