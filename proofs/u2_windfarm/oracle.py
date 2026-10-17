"""Build/run untouched pristine Fitch computational routines on CPU.

Only module dependencies and the file/MPI initializer are excluded. The three
computational subroutines are copied verbatim and individually SHA256 sealed.
The wrapper supplies piconst, turbine globals and pbl_driver-style arguments.
Tolerances registered before comparison: float32 rtol=3e-6, atol=2e-7 for
DU/DV/QKE, atol=.25 W for POWER (libm/XLA transcendental rounding).
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

import numpy as np


def inputs():
    nz,ny,nx = 12,3,4
    dz = np.broadcast_to(np.array([20,20,20,20,20,20,30,40,50,60,70,80],np.float32)[:,None,None],(nz,ny,nx)).copy()
    z = np.concatenate([np.zeros((1,ny,nx),np.float32),np.cumsum(dz,axis=0)],axis=0)+np.float32(123.)
    qke = np.full_like(dz,.6)
    du,dv = np.full_like(dz,.001),np.full_like(dz,-.002)
    rows=((2.,.9,0.),(4.,.85,60.),(8.,.8,500.),(12.,.7,1600.),(20.,.4,2000.),(25.,.1,2000.))
    spec=(80.,80.,.12,2000.,rows)
    high=(220.,100.,.15,2000.,tuple(r for r in rows if r[0]!=8.))
    overpowered=(80.,80.,.12,20000.,tuple((w,ct,p*10) for w,ct,p in rows))
    # colocated, multiple columns, non-intersected layers and above-domain rotor.
    turbines=((1,1,spec),(1,1,spec),(2,1,high),(3,2,spec),(0,0,(900.,80.,.12,2000.,rows)),(0,2,overpowered))
    cases=[]
    for factor in (0.,.25,1.):
        for target in (0.,1.,2.,3.,4.,8.,10.,12.,20.,24.,25.,30.):
            profile=np.linspace(-.12,.16,nz,dtype=np.float32)
            # Shift profile to target hub speed at first turbine's hub (k=3,4).
            profile-=.5*(profile[3]+profile[4])
            u=np.broadcast_to((np.float32(target)+profile)[:,None,None],dz.shape).copy()
            v=np.full_like(u,-.2 if target else 0.)
            for shear in (True,False):
                cases.append(dict(u=u if shear else np.full_like(u,target),v=v if shear else np.zeros_like(v),
                                  z=z,dz=dz,qke=qke,du=du,dv=dv,dx=1000.,dt=18.,factor=factor,turbines=turbines))
    return cases


DRIVER = '''program oracle
use module_wind_fitch
implicit none
integer,parameter :: nx=4,ny=3,nz=12
real :: u(nx,nz+1,ny),v(nx,nz+1,ny),z(nx,nz+1,ny),dz(nx,nz+1,ny)
real :: qke(nx,nz+1,ny),du(nx,nz+1,ny),dv(nx,nz+1,ny),power(nx,ny)
real :: dx,dt
integer :: ncases,c,t,k,i,j,nval,kind
read(*,*) ncases,nt
allocate(nkind(nt),ival(nt,max_domains),jval(nt,max_domains))
allocate(hubheight(nt),diameter(nt),stc(nt),stc2(nt),cutin(nt),cutout(nt),npower(nt))
do c=1,ncases
read(*,*) correction_factor,dx,dt
turbws=0.;turbtc=0.;turbpw=0.
do t=1,nt
read(*,*) ival(t,1),jval(t,1),nkind(t),hubheight(t),diameter(t),stc(t),npower(t),nval
kind=nkind(t)
do k=1,nval
read(*,*) turbws(kind,k),turbtc(kind,k),turbpw(kind,k)
enddo
cutin(t)=turbws(kind,1);cutout(t)=turbws(kind,nval);stc2(t)=turbtc(kind,nval)
enddo
u=0.;v=0.;dz=0.;du=0.;dv=0.;qke=0.
read(*,*) u(:,1:nz,:),v(:,1:nz,:),z,dz(:,1:nz,:),qke(:,1:nz,:),du(:,1:nz,:),dv(:,1:nz,:)
call dragforce(1,z,u,v,dx,dz,dt,qke,du,dv,1,power, &
               1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz)
write(*,'(*(ES25.16E3,1x))') du(:,1:nz,:),dv(:,1:nz,:),qke(:,1:nz,:),power
enddo
end program oracle
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_wind_fitch.F'))
    p.add_argument('--init',choices=('zero','poison'),default='zero')
    p.add_argument('--optimization',choices=('literal','native'),default='literal')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    build=args.out/'build';build.mkdir(exist_ok=True)
    original=args.source.read_text()
    core=original[:original.index('  SUBROUTINE init_module_wind_fitch')]
    import re
    core=re.sub(r'^\s*USE module_.*\n','',core,flags=re.M|re.I)
    core=core.replace('  IMPLICIT NONE','  IMPLICIT NONE\n  INTEGER, PARAMETER :: max_domains=99\n  REAL, PARAMETER :: piconst=3.1415926535897932384626433')
    (build/'fitch.f90').write_text(core+'\nEND MODULE module_wind_fitch\n')
    (build/'driver.f90').write_text(DRIVER)
    compiler='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=(['-O0'] if args.optimization=='literal' else ['-O3','-march=native','-ffp-contract=fast'])
    init_flags=(['-finit-real=zero','-finit-integer=0','-finit-logical=false'] if args.init=='zero'
                else ['-finit-real=snan','-finit-integer=-77777','-finit-logical=true'])
    cmd=[compiler,*flags,*init_flags,'-ffree-line-length-none','-fcheck=all','fitch.f90','driver.f90','-o','oracle']
    subprocess.run(cmd,cwd=build,check=True,timeout=60)
    cases=inputs();lines=[f'{len(cases)} {len(cases[0]["turbines"])}']
    for case in cases:
        lines.append(f'{case["factor"]} {case["dx"]} {case["dt"]}')
        for t,(x,y,spec) in enumerate(case['turbines']):
            hub,diam,standing,rated,rows=spec
            lines.append(f'{x+1} {y+1} {t+1} {hub} {diam} {standing} {rated} {len(rows)}')
            lines.extend(' '.join(map(str,row)) for row in rows)
        lines.append(' '.join(str(float(x)) for key in ('u','v','z','dz','qke','du','dv')
                              for x in case[key].transpose(2,0,1).ravel(order='F')))
    run=subprocess.run([str(build/'oracle')],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=60)
    (args.out/'oracle.txt').write_text(run.stdout)
    values=np.fromstring(run.stdout,sep=' ').reshape(len(cases),-1)
    n=12*3*4
    outputs={key:values[:,idx*n:(idx+1)*n].reshape(len(cases),3,12,4).transpose(0,2,1,3).astype(np.float32)
             for idx,key in enumerate(('du','dv','qke'))}
    outputs['power']=values[:,3*n:].reshape(len(cases),3,4).astype(np.float32)
    np.savez(args.out/'oracle.npz',**outputs)
    seals={name:hashlib.sha256(re.search(r'  SUBROUTINE\s+'+name+r'\b.*?  END SUBROUTINE\s+'+name+r'\b',original,re.S|re.I)[0].encode()).hexdigest()
           for name in ('dragforce','turbine_area','dragcof')}
    manifest=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  routine_sha256=seals,compiler=compiler,command=cmd,cases=len(cases),
                  precision='WRF REAL32; compiler flags listed verbatim; FP traps off (production default)',
                  initialization=args.init,optimization=args.optimization,
                  regimes='cut-in/out, interior/equal knots, standing drag, shear, colocated turbines, no rotor found, TKE factors 0/.25/1',
                  tolerances={'rtol':3e-6,'source_atol':2e-7,'power_atol_W':.25},
                  oracle_sha256=hashlib.sha256((args.out/'oracle.npz').read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.rmtree(build)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
