import glob, numpy as np, netCDF4, jax, json
jax.config.update('jax_enable_x64', True)
from gpuwrf.diagnostics.morrison_radar import morrison_refl_10cm
fn=jax.jit(lambda *a: morrison_refl_10cm(*a))
res=[]
for path in sorted(glob.glob('<USER_HOME>/wrf_gpu2_lanes/t-classd/c08r/run/wrfout_d01_*'))[1:]:
    with netCDF4.Dataset(path) as ds:
        g=lambda n: np.asarray(ds[n][0], np.float32)
        p=g('P')+g('PB'); t=((g('T')+np.float32(300.))*(p/np.float32(1e5))**np.float32(287./1004.5)).astype(np.float32)
        arrs=[g('QVAPOR'),g('QRAIN'),g('QNRAIN'),g('QSNOW'),g('QNSNOW'),g('QGRAUP'),g('QNGRAUPEL'),t,p]
        cpu=g('REFL_10CM')
    nz,ny,nx=cpu.shape
    cols=[np.moveaxis(a,0,-1).reshape(-1,nz) for a in arrs]
    got=np.moveaxis(np.asarray(fn(*cols)).reshape(ny,nx,nz),-1,0)
    d=np.abs(got-cpu); d[:,0,:]=0; d[:,-1,:]=0; d[:,:,0]=0; d[:,:,-1]=0
    for j,i in {(j,i) for _,j,i in np.argwhere(d>0.01)}:
        c=[a[:,j,i][None] for a in arrs]; best=None
        for delta in np.linspace(-0.3,0.3,61):
            cc=list(c); cc[7]=(c[7]+np.float32(delta)).astype(np.float32)
            v=np.asarray(fn(*cc))[0]; err=float(np.abs(v-cpu[:,j,i]).max())
            if best is None or err<best[1]: best=(float(delta),err)
        res.append(dict(frame=path[-19:],j=int(j),i=int(i),diff=float(d[:,j,i].max()),best_dT=best[0],best_err=best[1]))
errs=np.array([r['best_err'] for r in res]); dts=np.array([abs(r['best_dT']) for r in res])
print('columns',len(res),'explained(<0.01 dBZ with |dT|<=0.3K)',int((errs<0.01).sum()),'median|dT|',float(np.median(dts[errs<0.01])) if (errs<0.01).any() else None)
print('unexplained', [r for r in res if r['best_err']>=0.01][:5])
json.dump(res,open('<USER_HOME>/wrf_gpu2_lanes/t-classd/probe/p10_sensitivity.json','w'),indent=1)
