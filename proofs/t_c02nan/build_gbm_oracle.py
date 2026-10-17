"""Unchanged WRF 4.7.1 REAL4 GBMPBL with a file-fed single-column driver."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('--nz',type=int,default=44)
    p.add_argument('--control',choices=('zero','snan','optimized-no-fma','optimized-fma'),default='zero')
    a=p.parse_args();a.build.mkdir(parents=True,exist_ok=True)
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    wrf=Path(__file__).resolve().parents[2]/'data/wrf_pristine/WRF'
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-ffp-contract=off','-ffree-form','-ffree-line-length-none','-cpp',
        '-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffpe-summary=none']
    if a.control=='snan':
        flags=[s.replace('real=zero','real=snan').replace('integer=0','integer=-777').replace('logical=false','logical=true') for s in flags]
    if a.control.startswith('optimized'):
        flags[0]='-O2'
    if a.control=='optimized-fma':
        flags[1]='-ffp-contract=fast';flags.append('-mfma')
    files=['share/module_model_constants.F','phys/module_bl_gbmpbl.F'];hashes={}
    for name in files:
        source=wrf/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source,a.build/source.name)
        subprocess.run([fc,*flags,'-c',source.name],cwd=a.build,check=True,timeout=60)
    arrays=['u','v','th','t','qv','qc','qi','p','pii','dz','z','tke','rad','ru','rv','rth','rqv','rqc','rqi','kzm','kth','kethl','el']
    scalars=['psfc','znt','ust','hfx','qfx','tsk','gz','wspd','br','psim','psih','xland','zol','hol','pbl','regime']
    code=['program gbm_oracle','use module_bl_gbmpbl, only: gbmpbl','implicit none',
        f'integer,parameter::n={a.nz}', 'integer::k,kpbl(1,1)','real::dt',
        'real::'+','.join(f'{name}(1,n+1,1)' for name in arrays),
        'real::'+','.join(f'{name}(1,1)' for name in scalars)]
    code += [f'{name}=0.' for name in arrays+scalars]+['kpbl=0','open(42,file="column.in",status="old")']
    code += [f'read(42,*){name}(1,1:n,1)' for name in ['u','v','t','qv','qc','p','pii','dz','tke']]
    code += [f'read(42,*){name}(1,1)' for name in scalars[:12]]
    code += ['read(42,*)dt','close(42)','th(1,1:n,1)=t(1,1:n,1)/pii(1,1:n,1)','z(1,1,1)=.5*dz(1,1,1)',
        'do k=2,n','z(1,k,1)=z(1,k-1,1)+.5*(dz(1,k-1,1)+dz(1,k,1))','enddo']
    code += ['call gbmpbl(u,v,th,t,qv,qc,qi,p,pii,ru,rv,rth,rqv,rqc,rqi,kzm,kth,kethl,el, &',
        ' dz,z,psfc,tke,rad,znt,ust,zol,hol,pbl,kpbl,regime,psim,psih,xland,hfx,qfx,tsk,gz,wspd,br, &',
        ' dt,dt/60.,1,2,1,2,1,n+1,1,1,1,1,1,n+1,1,1,1,1,1,n)']
    code += ['do k=1,n']
    for label,name in [('u','ru'),('v','rv'),('theta','rth'),('qv','rqv'),('qc','rqc'),('qi','rqi'),('tke','tke'),('kzm','kzm'),('kth','kth'),('el_pbl','el')]:
        code.append(f'write(*,\'(A,I0,A,ES25.17)\') "{label}[",k,"]=",{name}(1,k,1)')
    code += ['enddo','do k=1,n+1','write(*,\'(A,I0,A,ES25.17)\') "kethl[",k,"]=",kethl(1,k,1)','enddo',
        'write(*,\'(A,ES25.17)\')"pblh=",pbl(1,1)','write(*,\'(A,I0)\')"kpbl=",kpbl(1,1)','end program']
    driver=a.build/'driver.f90';driver.write_text('\n'.join(code)+'\n')
    subprocess.run([fc,*flags,'-o','gbm_oracle',driver.name,'module_model_constants.o','module_bl_gbmpbl.o'],cwd=a.build,check=True,timeout=60)
    manifest=dict(source_sha256=hashes,flags=flags,compiler=fc,nz=a.nz,
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((a.build/'gbm_oracle').read_bytes()).hexdigest())
    (a.build/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
