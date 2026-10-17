"""Original cloud DSD/Bigg rate on actual phase operands; isolate intrinsics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from compare_stages import records
from gpuwrf.physics.p3.process_support import _cloud, gamma, _r
from gpuwrf.physics.p3.constants import CONSTANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--compiler',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    reference=records(a.oracle/'stage_1.bin')
    inputs=np.stack([np.stack([v[name] for name in ('qc','nc','rho','t')],axis=0)
                     for (it,i),v in reference.items() if it==1],axis=1).reshape((4,-1))
    np.asfortranarray(inputs).ravel(order='F').tofile(a.out/'input.bin')
    source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_p3.F').read_text()
    staged,count=re.subn(r'(?im)^(\s*contains\s*)$',r' public :: get_cloud_dsd2,cons6,aimm\n\1',source,count=1)
    assert count==1
    (a.out/'module_mp_p3.F').write_text(staged)
    program=f'''program probe
use microphy_p3
implicit none
integer :: i,u,stat
real :: x(4,{inputs.shape[1]}),y(7,{inputs.shape[1]}),nc,mu,nu,lam,lo,hi,cd,cd1,dum,tmp1,rate
real :: dnu(16)
call p3_init('<USER_HOME>/src/wrf_pristine/WRF/run',1,.false.,model='WRF',stat=stat,abort_on_err=.true.)
if(stat/=STATUS_OK) stop 2
open(newunit=u,file='input.bin',access='stream',form='unformatted');read(u)x;close(u)
dnu=0.
do i=1,{inputs.shape[1]}
nc=x(2,i);mu=0.;nu=0.;lam=0.;lo=0.;hi=0.;cd=0.;cd1=0.
call get_cloud_dsd2(x(1,i),nc,mu,x(3,i),nu,dnu,lam,lo,hi,cd,cd1,1.)
rate=0.
if(x(1,i)>=1.e-14.and.x(4,i)<=269.15) then
dum=(1./lam)**3
tmp1=cd1*exp(aimm*(273.15-x(4,i)))
rate=cons6*gamma(7.+mu)*tmp1*dum**2
endif
y(:,i)=[nc,mu,lam,cd1,gamma(mu+1.),gamma(mu+7.),rate]
enddo
open(newunit=u,file='output.bin',access='stream',form='unformatted',status='replace');write(u)y;close(u)
end program
'''
    (a.out/'probe.f90').write_text(program)
    subprocess.run([a.compiler,'-cpp','-ffree-form','-ffree-line-length-none','-O2','-ffp-contract=off',
                    'module_mp_p3.F','probe.f90','-o','probe'],cwd=a.out,check=True,timeout=120)
    subprocess.run([str((a.out/'probe').resolve())],cwd=a.out,check=True,timeout=120)
    expected=np.fromfile(a.out/'output.bin',np.float32).reshape((7,-1),order='F')
    def calculate(q,n,rho,t):
        nn,mu,_,lam,_,_,_,cd1=_cloud(q,n,rho,_r(1))
        dum=(1/lam)**3
        tmp1=cd1*jnp.exp(_r(CONSTANTS['aimm'])*(_r(273.15)-t))
        rate=_r(CONSTANTS['cons6'])*gamma(mu+_r(7))*tmp1*dum**2
        return jnp.stack((nn,mu,lam,cd1,gamma(mu+1),gamma(mu+7),jnp.where((q>=_r(1e-14))&(t<=_r(269.15)),rate,_r(0))))
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'backend':'cpu','scores':{}}
    names=('nc','mu','lambda','cdist1','gamma_mu1','gamma_mu7','Bigg_rate')
    for mode in ('eager','jit'):
        with jax.disable_jit(mode=='eager'):
            actual=np.asarray(jax.jit(calculate)(*map(jnp.asarray,inputs)))
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        report['scores'][mode]={}
        for name,value,truth in zip(names,actual,expected):
            error=np.abs(value.astype(np.float64)-truth)
            index=int(error.argmax())
            report['scores'][mode][name]={'different':int((value!=truth).sum()),'max_abs':float(error.max()),
                'index':index,'candidate':float(value[index]),'reference':float(truth[index])}
    np.savez_compressed(a.out/'operands.npz',inputs=inputs,reference=expected,eager=np.asarray(calculate(*map(jnp.asarray,inputs))),jit=actual)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
