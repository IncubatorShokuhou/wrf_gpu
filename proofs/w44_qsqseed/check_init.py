"""Evaluate fresh REAL4 init against frozen original WRF and report cost.

Run XLA_FLAGS=--xla_cpu_max_isa=AVX to match qualified non-FMA WRF objects.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, mynn_coldstart_init_columns


def evaluate(packet):
    a,s=packet['arrays'],packet['scalars']; expected=packet['truth'][:,:,:3]
    actual=np.empty_like(expected); costs=[]
    for dx in sorted(set(s[:,-1])):
        ix=np.flatnonzero(s[:,-1]==dx); col=a[ix]
        v=lambda n:jnp.asarray(col[:,:,n]); z=jnp.zeros_like(v(0))
        state=MynnPBLColumnState(v(0),v(1),v(2),v(3),v(4),v(5),v(6),v(7),v(8),z,z,z,
             qc=v(9),qi=v(10),qs=v(11),qsq=z,qc_bl=z,qi_bl=z,cldfra_bl=z,exner=v(13))
        with jax.enable_x64(False):
            f=jax.jit(lambda state,ust,land:mynn_coldstart_init_columns(state,ust,float(dx),land,
                      return_moments=True))
            args=(state,jnp.asarray(s[ix,0]),jnp.asarray(s[ix,4]))
            started=time.perf_counter(); compiled=f.lower(*args).compile(); compile_s=time.perf_counter()-started
            result=compiled(*args);jax.block_until_ready(result)
            times=[]
            for _ in range(5):
                started=time.perf_counter();jax.block_until_ready(compiled(*args));times.append(time.perf_counter()-started)
        qke,_,el,qsq=map(np.asarray,result)
        actual[ix]=np.stack((qke,el,qsq),axis=-1)
        mem=compiled.memory_analysis()
        costs.append({'dx':float(dx),'ncol':len(ix),'compile_s':compile_s,'warm_median_s':float(np.median(times)),
                      'memory':{k:int(getattr(mem,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')}})
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    records=[]
    for slot,name in enumerate(('QKE','EL','QSQ')):
        x,y=actual[:,:,slot].astype(float),expected[:,:,slot].astype(float);err=abs(x-y)
        if name=='QKE':metric=float(np.median(err/np.maximum(abs(y),1e-8)));limit=1e-4
        else:
            atol=.001 if name=='EL' else 2e-12
            metric=float(np.max(err/(atol+.002*abs(y))));limit=1.
        records.append({'field':name,'metric':metric,'limit':limit,'passed':metric<=limit,'max_abs':float(err.max()),'truth_nonzero':int(np.count_nonzero(y))})
    return actual,{'passed':all(r['passed'] for r in records),'records':records,'costs':costs,'backend':jax.default_backend()}

if __name__=='__main__':
    packet=np.load(sys.argv[1]);actual,report=evaluate(packet)
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert report['passed'],report['records']
