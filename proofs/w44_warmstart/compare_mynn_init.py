"""Compare the optional initializer against the saved WS11 original witness."""
from pathlib import Path
from contextlib import nullcontext
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, mynn_coldstart_init_columns

# Existing QKE median gate is inherited. New moment bounds frozen before the
# first candidate measurement; finite and background-QKE exact zero mandatory.
LIMITS={"QKE_relative_median":1e-4,"EL_atol":1e-3,"EL_rtol":2e-3,"QSQ_atol":2e-12,"QSQ_rtol":2e-3}


def main():
    oracle,out=map(Path,sys.argv[1:3])
    eager=len(sys.argv)>3 and sys.argv[3]=='eager'
    lines=(oracle/'input.txt').read_text().splitlines();nz,nb,_,_=map(float,lines[0].split());nz,nb=int(nz),int(nb)
    scalars=[];arrays=[];offset=1
    for i in range(nb):
        scalars.append(list(map(float,lines[offset].split())));offset+=1
        arrays.append([list(map(float,row.split()))for row in lines[offset:offset+nz]]);offset+=nz
    scalars,arrays=np.asarray(scalars,np.float32),np.asarray(arrays,np.float32)
    truth=np.load(oracle/'expected.npz')['values']
    candidate=np.empty_like(truth);pblh=np.empty(nb,np.float32)
    times=[]
    for dx in sorted(set(scalars[:,-1])):
        indices=np.flatnonzero(scalars[:,-1]==dx);a=arrays[indices];s=scalars[indices]
        value=lambda slot:jnp.asarray(a[:,:,slot])
        z=jnp.zeros_like(value(0))
        state=MynnPBLColumnState(value(0),value(1),value(2),value(3),value(4),value(5),value(6),value(7),value(8),z,z,z,
               qc=value(9),qi=value(10),qs=value(11),qsq=z,qc_bl=z,qi_bl=z,cldfra_bl=z,exner=value(13))
        mask=np.max(a[:,:,5]*np.float32(2),axis=-1)<np.float32(.0002)
        with jax.enable_x64(False), (jax.disable_jit() if eager else nullcontext()):
            f=lambda state,ust,land,mask:mynn_coldstart_init_columns(
                state,ust,float(dx),land,initialize_qke=mask,return_moments=True)
            if not eager:f=jax.jit(f)
            started=time.perf_counter()
            result=f(state,jnp.asarray(s[:,0]),jnp.asarray(s[:,4]),jnp.asarray(mask))
            jax.block_until_ready(result);times.append(time.perf_counter()-started)
        qke,h,el,qsq=map(np.asarray,result)
        candidate[indices,:,0]=qke;candidate[indices,:,1]=qsq;candidate[indices,:,2]=el;pblh[indices]=h
    assert np.isfinite(candidate).all()
    original=json.loads((oracle/'report.json').read_text())
    records=[]
    for mode in ('cold','background'):
        indices=np.array([i for i,r in enumerate(original['records']) if r['mode']==mode])
        for slot,name in enumerate(('QKE','QSQ','EL')):
            a,b=candidate[indices,:,slot],truth[indices,:,slot]
            error=np.abs(a.astype(float)-b)
            if name=='QKE':
                metric=float(np.median(error/np.maximum(np.abs(b),1e-8)))
                passed=metric<=LIMITS['QKE_relative_median'] if mode=='cold' else np.array_equal(a,b)
            else:
                metric=float(np.max(error/(LIMITS[name+'_atol']+LIMITS[name+'_rtol']*np.abs(b))))
                passed=metric<=1
            records.append({'mode':mode,'field':name,'passed':bool(passed),'metric':metric,'max_abs_error':float(error.max())})
    np.savez(out.with_suffix('.npz'),values=candidate,truth=truth)
    report={'passed':all(r['passed']for r in records),'records':records,'limits':LIMITS,'compile_execute_seconds':times,'eager_diagnostic':eager,
            'oracle':str(oracle),'backend':jax.default_backend(),'scope':'CPU original-init component; no GPU/forecast'}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed'],records


if __name__=='__main__':main()
