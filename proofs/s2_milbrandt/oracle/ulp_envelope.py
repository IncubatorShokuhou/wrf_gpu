"""Independent pristine-WRF input-ULP envelope for compiled REAL4 output conditioning.

Population fixed before running: each REAL input field (all points together),
surface pressure, and dt, one adjacent IEEE value in each direction. The bound
at each output point is its maximum departure from the unperturbed pristine
output. No candidate values participate in deriving it.
"""
from pathlib import Path
import argparse
import struct
import subprocess
import json
import hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('build');p.add_argument('fixture');p.add_argument('out');p.add_argument('--wrapper',action='store_true');a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    with np.load(a.fixture) as f:base=dict(f)
    original=base['inputs'];ps0=base['ps'];dt0=base['dt'];nc,nk,_=original.shape
    if a.wrapper:original=np.concatenate((original,base['pii'][:,:,None]),axis=2)
    output_names=('fields','rates','diam','zet','zec')
    shapes=((nc,nk,14),(nc,11),(nc,nk,6),(nc,nk),(nc,))
    if a.wrapper:
        output_names=('fields','rain','zet');shapes=((nc,nk,14),(nc,9),(nc,nk))
    envelope={name:np.zeros_like(base['ref_'+name]) for name in output_names}
    population=[]
    ninputs=original.shape[-1]
    for slot in range(ninputs+2):
        for sign in (-1,1):
            fields=original.copy();ps=ps0.copy();dt=dt0.copy()
            target=np.float32(-np.inf if sign<0 else np.inf)
            if slot<ninputs:fields[:,:,slot]=np.nextafter(fields[:,:,slot],target)
            elif slot==ninputs:ps=np.nextafter(ps,target)
            else:dt=np.nextafter(dt,target)
            inp=root/'ulp_input.bin';out=root/'ulp_output.bin'
            inp.write_bytes(struct.pack('iif',nc,nk,float(dt))+fields.tobytes(order='F')+ps.tobytes(order='F'))
            executable='wrapper' if a.wrapper else 'oracle'
            subprocess.run([str(Path(a.build)/executable),str(inp),str(out)],check=True,capture_output=True)
            raw=np.fromfile(out,np.float32);offset=0;values={}
            for name,shape in zip(output_names,shapes):
                count=int(np.prod(shape));val=raw[offset:offset+count].reshape(shape,order='F');offset+=count
                if name in ('fields','diam'):val=val.transpose(0,2,1)
                assert np.isfinite(val).all(),(slot,sign,name)
                values[name]=val
                envelope[name]=np.maximum(envelope[name],np.abs(val-base['ref_'+name]))
            population.append({'slot':slot,'sign':sign,'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest()})
    np.savez_compressed(root/'envelope.npz',**envelope)
    receipt={'reference_only':True,'candidate_used_to_derive_bound':False,'population':population,
             'definition':'max absolute pristine output change for one adjacent IEEE32 input value, per input field/ps/dt, both signs',
             'finite':True,'n_arms':len(population),'envelope_sha256':hashlib.sha256((root/'envelope.npz').read_bytes()).hexdigest()}
    (root/'envelope.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
