"""Build active WRF oracle columns and compare immutable candidate; no GPU."""
from pathlib import Path
import argparse
import json
import subprocess
import struct
import time
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.milbrandt2mom.column import columns

NAMES='t q qc qr qi qn qg qh nc nr ny nn ng nh'.split()


def inputs():
    # Adversarial active-process columns: thermodynamic envelope and species
    # constructed to force cold,
    # wet/dry hail, homogeneous freezing, saturation and size-limiter regimes.
    nc,nk=12,12
    a=np.zeros((nc,nk,16),np.float32)
    p=np.linspace(95000,25000,nk,dtype=np.float32)
    ps=np.full(nc,100000,np.float32)
    a[:,:,15]=p
    a[:,:,14]=np.asarray([0.,1.,5.,12.,-1.,3.,8.,.001,0.,25.,-3.,2.],np.float32)[:,None]
    for i,t0 in enumerate((300,282,267,248,235,278,260,285,270,255,290,240)):
        a[i,:,0]=t0-np.arange(nk)*np.float32(1.5)
        # independent saturation polynomial (Flatau liquid) solely to define operands.
        t=a[i,:,0]; tc=np.maximum(-80.,t-273.16)
        es=6.11239921+tc*(.443987641+tc*(.0142986287+tc*(.000264847430+tc*(.00000302950461+tc*(.0000000206739458+tc*(.0000000000640689451+tc*(-.0000000000000952447341-tc*.000000000000000976195544)))))))
        es*=100
        a[i,:,1]=.622*es/(p-es)*np.float32((.65,1.04,.98,1.02,1.01,.90,1.1,.8,1.05,1.0,1.02,.98)[i])
        if i==0: continue  # dry zero-process control
        a[i,:,2:8]=np.asarray((2e-4,3e-4,1e-5,1e-4,2e-4,4e-4),np.float32)
        a[i,:,8:14]=np.asarray((1e8,2e3,2e4,3e3,50.,.5),np.float32)
        if i in (1,7,10): a[i,:,4:8]=0; a[i,:,10:14]=0 # warm only
        if i in (3,4,11): a[i,:,7]=1e-3; a[i,:,13]=.02 # large hail
        if i==6: a[i,:,8:14]=0 # missing-number prescription
        if i==8: a[i,::2,2:8]=0; a[i,::2,8:14]=0 # clear levels between precip
    # WRF requires an empty upper level for count_columns + k+1 transport.
    a[:,-1,2:14]=0
    return a,ps,np.float32(18.)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('build'); parser.add_argument('out')
    parser.add_argument('--stage',choices=('full','local'),default='full')
    args=parser.parse_args(); root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    a,ps,dt=inputs(); nc,nk,_=a.shape
    inp=root/'input.bin'; out=root/'oracle.bin'
    inp.write_bytes(struct.pack('iif',nc,nk,dt)+a.tobytes(order='F')+ps.tobytes(order='F'))
    subprocess.run([str(Path(args.build)/'oracle'),str(inp),str(out),args.stage],check=True)
    ref=np.fromfile(out,np.float32)
    def take(count,shape):
        nonlocal ref
        result=ref[:count].reshape(shape,order='F'); ref=ref[count:]; return result
    expected=(take(nc*nk*14,(nc,nk,14)).transpose(0,2,1),take(nc*11,(nc,11)),
              take(nc*nk*6,(nc,nk,6)).transpose(0,2,1),take(nc*nk,(nc,nk)),take(nc,(nc,)))
    if args.stage=='local':
        from gpuwrf.physics.milbrandt2mom.local import local_processes
        from gpuwrf.physics.milbrandt2mom.constants import constants
        def local(fields,p,ps,w,dt):
            pres=ps*(p/ps)
            f,de,ide=local_processes(tuple(fields),pres,w,dt,constants(str(fields.dtype)),jnp.arange(fields.shape[1])<fields.shape[1]-1)
            inv=(jnp.asarray(constants(str(fields.dtype))['rgasd'],fields.dtype)*f[0])/pres
            return jnp.stack(f[:8]+tuple(x*inv for x in f[8:]))
        fn=jax.jit(jax.vmap(local,in_axes=(0,0,0,0,None)))
    else:
        fn=jax.jit(columns)
    arguments=(jnp.asarray(a[:,:,:14].transpose(0,2,1)),jnp.asarray(a[:,:,15]),jnp.asarray(ps),jnp.asarray(a[:,:,14]),jnp.asarray(dt))
    start=time.monotonic(); lowered=fn.lower(*arguments); lower_s=time.monotonic()-start
    start=time.monotonic(); exe=lowered.compile(); compile_s=time.monotonic()-start
    result=jax.device_get(exe(*arguments)); jax.block_until_ready(result)
    warm=[]
    for _ in range(5):
        start=time.monotonic(); jax.block_until_ready(exe(*arguments)); warm.append(time.monotonic()-start)
    checks={}; ok=True
    if args.stage=='local': result=(result,); expected=expected[:1]
    for category,actual,want in zip(('fields','rates','diam','zet','zec'),result,expected):
        if category=='fields':
            arrays=zip(NAMES,actual.transpose(1,0,2),want.transpose(1,0,2))
        else: arrays=[(category,actual,want)]
        for name,x,y in arrays:
            atol=(3e-5 if name=='t' else 3e-10 if name in NAMES[1:8] else 3e-3 if name in NAMES[8:]
                  else 3e-10 if name=='rates' else 3e-9 if name=='diam' else 3e-4)
            passed=bool(np.all(np.isfinite(x)) and np.allclose(x,y,rtol=3e-5,atol=atol))
            checks[name]={'pass':passed,'max_abs':float(np.max(np.abs(x-y))),
                          'worst_index':list(map(int,np.unravel_index(np.argmax(np.abs(x-y)),x.shape)))}
            # Manager 2026-10-13: Dm_* are discarded wrapper locals; reff
            # assignments are commented and has_req*=0. Disclose this diagnostic
            # without changing its tolerance or making it a driver-output gate.
            if name!='diam': ok &= passed
    receipt={'pass':ok,'checks':checks,'lower_s':lower_s,'compile_s':compile_s,'warm_median_s':float(np.median(warm)),
             'memory':exe.memory_analysis().__str__(),'backend':jax.devices()[0].platform,'work':'12 columns x12 levels dt18 all WRF driver flags',
             'cache':'NONE GPUWRF_JAX_CACHE=0','input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
             'census':{name:int(np.count_nonzero(a[:,:,j])) for j,name in enumerate(NAMES)}}
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    np.savez(root/'parity.npz',inputs=a,ps=ps,dt=dt,**{f'ref_{k}':v for k,v in zip(('fields','rates','diam','zet','zec'),expected)},
             **{f'jax_{k}':v for k,v in zip(('fields','rates','diam','zet','zec'),result)})
    print(json.dumps(receipt,indent=2))
    if not ok: raise SystemExit(1)

if __name__=='__main__': main()
