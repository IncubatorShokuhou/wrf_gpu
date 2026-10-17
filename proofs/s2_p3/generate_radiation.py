"""Original RRTMG P3 remap blocks, unchanged, on original wrapper outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--compiler',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.fixtures/'columns_mp52.npz') as f:
        qi,qi2=f['qi'][...,1].ravel(),f['qi2'][...,1].ravel()
        ri=np.linspace(0,125.e-6,qi.size,dtype=np.float32)
    x=np.stack((qi,qi2,ri));x.ravel(order='F').tofile(a.out/'input.bin')
    manifest={'precision':'REAL4','modifications':'external driver declarations only; remap bodies verbatim',
              'flags':['-O2','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false'],
              'sources':{},'samples':len(qi)}
    outputs={}
    for kind in ('sw','lw'):
        source=Path(f'<USER_HOME>/src/wrf_pristine/WRF/phys/module_ra_rrtmg_{kind}.F')
        text=source.read_text()
        match=re.search(r'(?im)^\s*IF\s*\(\s*has_reqs\s*\.eq\.\s*0\s*\.and\.\s*has_reqi\s*\.ne\.\s*0\s*\.and\.\s*has_reqc\s*\.ne\.\s*0\).*?^\s*END\s*IF\b',text,re.S)
        assert match
        body=match[0]
        manifest['sources'][kind]={'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                  'block_sha256':hashlib.sha256(body.encode()).hexdigest(),
                                  'line':text[:match.start()].count('\n')+1}
        program=f'''program oracle
implicit none
integer,parameter :: n={len(qi)}
integer :: u,i,j,k,kts,kte,ncol,has_reqc,has_reqi,has_reqs,inflg{kind},iceflg{kind},active
real :: x(3,n),y(4,n,2),qi3d(1,n,1),re_ice(1,n,1),qs1d(n),qi1d(n),resnow1d(1,n),reice1d(1,n)
open(newunit=u,file='input.bin',access='stream',form='unformatted');read(u)x;close(u)
do active=0,1
i=1;j=1;ncol=1;kts=1;kte=n
qi3d(1,:,1)=x(1,:)+x(2,:)
re_ice(1,:,1)=x(3,:)
qs1d=0.;qi1d=qi3d(1,:,1);resnow1d=10.;reice1d=5.
has_reqc=active;has_reqi=active;has_reqs=0
{body}
y(1,:,active+1)=qs1d;y(2,:,active+1)=qi1d;y(3,:,active+1)=resnow1d(1,:);y(4,:,active+1)=reice1d(1,:)
enddo
open(newunit=u,file='{kind}.bin',access='stream',form='unformatted',status='replace');write(u)y;close(u)
end program
'''
        (a.out/f'{kind}.f90').write_text(program)
        subprocess.run([a.compiler,*manifest['flags'],'-ffree-line-length-none',f'{kind}.f90','-o',kind],cwd=a.out,check=True,timeout=60)
        subprocess.run([str((a.out/kind).resolve())],cwd=a.out,check=True,timeout=60)
        outputs[kind]=np.fromfile(a.out/f'{kind}.bin',np.float32).reshape((4,len(qi),2),order='F')
    assert np.array_equal(outputs['sw'],outputs['lw'])
    np.savez_compressed(a.out/'radiation.npz',inputs=x,**outputs)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
