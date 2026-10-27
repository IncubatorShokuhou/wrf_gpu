"""Explicit CUDA selection plus a CPU-only dispatch smoke for these launchers."""
import hashlib
import json
import os
from pathlib import Path
import time

CPU_PREFLIGHT='W44_WARMSTART_CPU_PREFLIGHT'


def cuda_environment(base, **values):
    env=dict(base,**values)
    # JAX's generic 'gpu' alias expands to CUDA AND ROCm. This port targets CUDA.
    env['JAX_PLATFORMS']='cuda'
    env['CUDA_VISIBLE_DEVICES']='0'
    env.pop('JAX_PLATFORM_NAME',None)
    env.pop(CPU_PREFLIGHT,None)
    return env


def asserted_device():
    dry=os.environ.get(CPU_PREFLIGHT)=='1'
    expected='cpu'if dry else'cuda'
    assert os.environ.get('JAX_PLATFORMS')==expected,(expected,os.environ.get('JAX_PLATFORMS'))
    if not dry:
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1','official GPU lock required'
    import jax
    device=jax.devices()[0]
    if dry:assert device.platform=='cpu',device
    else:assert device.platform=='gpu',device
    return device,dry


def cpu_mini(oracle,output,runner):
    """Four real WS11 columns, both DX and cycling branches; no fidelity claim."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.physics import mynn_pbl as P
    assert os.environ[CPU_PREFLIGHT]=='1'and jax.devices()[0].platform=='cpu'
    assert Path(P.__file__).resolve().is_relative_to(Path(os.environ['PYTHONPATH'].split(os.pathsep)[0]).resolve())
    lines=(Path(oracle)/'input.txt').read_text().splitlines()
    nz,nb,_,_=map(float,lines[0].split());nz,nb=int(nz),int(nb)
    scalars,arrays,offset=[],[],1
    for _ in range(nb):
        scalars.append(list(map(float,lines[offset].split())));offset+=1
        arrays.append([list(map(float,r.split()))for r in lines[offset:offset+nz]]);offset+=nz
    scalars,arrays=np.asarray(scalars,np.float32),np.asarray(arrays,np.float32)
    mode=np.max(arrays[:,:,5]*np.float32(2),axis=-1)<np.float32(.0002)
    records=[];start=time.monotonic()
    for dx in sorted(set(scalars[:,-1])):
        candidates=np.flatnonzero(scalars[:,-1]==dx)
        selected=np.array([next(i for i in candidates if mode[i]),next(i for i in candidates if not mode[i])])
        a,s=arrays[selected],scalars[selected]
        value=lambda slot:jnp.asarray(a[:,:,slot]);z=jnp.zeros_like(value(0))
        state=P.MynnPBLColumnState(value(0),value(1),value(2),value(3),value(4),value(5),value(6),value(7),value(8),z,z,z,
            qc=value(9),qi=value(10),qs=value(11),qsq=z,qc_bl=z,qi_bl=z,cldfra_bl=z,exner=value(13))
        function=jax.jit(lambda state,ust,land,mask:P.mynn_coldstart_init_columns(
            state,ust,float(dx),land,initialize_qke=mask,return_moments=True))
        result=function(state,jnp.asarray(s[:,0]),jnp.asarray(s[:,4]),jnp.asarray(mode[selected]))
        jax.block_until_ready(result);qke,height,el,qsq=map(np.asarray,result)
        assert qke.shape==el.shape==qsq.shape==(2,nz)and height.shape==(2,)
        assert all(np.isfinite(v).all()for v in (qke,height,el,qsq))
        assert (qke[1]==0).all()and(qsq[1]==0).all()
        records.append({'dx':float(dx),'witness_indices':selected.tolist(),'columns':2,'levels':nz,
                        'finite':True,'background_QKE_QSQ_exact_zero':True})
    report={'verdict':'CPU_ENVIRONMENT_SMOKE_PASS','runner':runner,'platform':jax.devices()[0].platform,
        'requested_platform':os.environ['JAX_PLATFORMS'],'model_body_reached':True,
        'model_function':'mynn_coldstart_init_columns','model_file':P.__file__,
        'model_file_sha256':hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        'records':records,'wall_s':time.monotonic()-start,
        'scope':'CPU environment/entry/model smoke only; NOT GPU or original-oracle fidelity qualification'}
    (Path(output)/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
