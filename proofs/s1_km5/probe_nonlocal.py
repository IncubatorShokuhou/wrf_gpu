"""Measure first divergence after two rejected precision hypotheses."""
from pathlib import Path
import sys,json
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'src'))
from gpuwrf.kernels.phys_sms3dtke import _call,_nonlocal_kernel
assert jax.devices()[0].platform=='cpu'
lane=Path('<USER_HOME>/wrf_gpu2_lanes/s1-km5');key='periodic_convective/'
with np.load(ROOT/'data/fixtures/sms3dtke-upstream-wrf-v1.npz') as z:
    get=lambda n:np.array(z[key+n][:,3:13,3:15] if z[key+n].ndim==3 else z[key+n][3:13,3:15])
    nz,nc=44,120
    rz=get('rdzw')[:nz];height=np.zeros((nz+1,10,12),np.float32)
    for k in range(nz):height[k+1]=np.float32(1)/rz[k]+height[k]
    prof=lambda n:get(n)[:nz].reshape(nz,nc)
    surf=lambda n:get(n).reshape(nc)
    args=tuple(jnp.asarray(x) for x in ([prof(n) for n in ('theta','qv','u','v','rho')]+[height.reshape(nz+1,nc)]+
        [surf(n) for n in ('ust','hfx_start','br','wspd','msftx','msfty','gu_start','gv_start')]))
    po=jax.ShapeDtypeStruct((nz+1,nc),jnp.float32);so=jax.ShapeDtypeStruct((nc,),jnp.float32)
    args=args+(jnp.asarray(1200.,jnp.float32),jnp.asarray(1200.,jnp.float32))
    kw=dict(nz=nz,isfflx=2,heat=.06)
    plain=_call(_nonlocal_kernel,args,(po,so,so,so),nc,'sms_nonlocal_plain_probe',**kw)
    debug=_call(_nonlocal_kernel,args,(po,so,so,so,jax.ShapeDtypeStruct((23,nc),jnp.float32)),nc,'sms_nonlocal_internal_probe',**kw)
    for a,b in zip(plain,debug[:4]):assert np.array_equal(a,b),'observer changed nonlocal outputs'
    expected=get('nlflux').reshape(nz+1,nc)
    a=np.asarray(plain[0]);print('nlflux diff max',abs(a-expected).max(),flush=True)
    refpath=lane/'K5-17-pristine-probe.bin'
    if not refpath.exists():refpath=lane/'oracle/nonlocal_probe.bin'
    ref=np.fromfile(refpath,np.float32).reshape(2,25)
    names=('hpbl','kpbl','govrth','sflux','wstar3','wscale','deltaoh_ratio','rigs','enlfrac2','wm2','dthv','hfxpbl','delxy','mlfrac','sfcfracn','sflux0','snlflux0','amf1','amf2','bmf2','amf3','bmf3','pthnl')
    table=np.asarray(debug[4]);report={}
    for record in ref:
        i,j=map(int,record[:2]);idx=(j-1)*12+i-1
        rows=[]
        for n,p,r in zip(names,table[:,idx],record[2:]):
            rows.append(dict(name=n,port=float(p),wrf=float(r),abs=float(abs(p-r)),ulps=float(abs(float(p)-float(r))/abs(float(np.spacing(r)))) if r else None))
        report[f'{i},{j}']=rows
    (lane/'K5-15-localization.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
