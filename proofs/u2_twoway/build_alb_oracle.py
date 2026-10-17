"""Save verbatim pristine calc_p_rho_phi with independent supplied ALB.

Operands are real WN3 input crops. ALB is diagnosed from the child/base
geometry then passed through the independent Fortran feedback operator;
it is an explicit operand, not reconstructed by the pressure oracle.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import sys
import numpy as np
from build_oracle import build,call
from gpuwrf.io.netcdf_lock import Dataset

work,fixture,case=map(lambda p:Path(p).resolve(),sys.argv[1:])
work.mkdir(exist_ok=True,parents=True)
source=Path('<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F')
raw=source.read_text()
routine=raw[raw.index('SUBROUTINE calc_p_rho_phi ('):raw.index('END SUBROUTINE calc_p_rho_phi')+len('END SUBROUTINE calc_p_rho_phi')]
module='''module oracle_calc
use iso_c_binding
implicit none
real,parameter::r_d=287.,cpovcv=1004.5/717.5,cvpm=-717.5/1004.5,p1000mb=100000.,rvovrd=461.6/287.
integer,parameter::PARAM_FIRST_SCALAR=2,P_QV=1
contains
'''+routine+'''
subroutine gate(nx,ny,nz,opt,t,ph,phb,mu,mub,alb,pb,c1,c2,c3h,c4h,c3f,c4f,dnw,rdnw,ptop,p,al) bind(C)
integer(c_int),value::nx,ny,nz,opt
real(c_float)::t(nx,nz+1,ny),ph(nx,nz+1,ny),phb(nx,nz+1,ny),alb(nx,nz+1,ny),pb(nx,nz+1,ny)
real(c_float)::mu(nx,ny),mub(nx,ny),c1(nz+1),c2(nz+1),c3h(nz+1),c4h(nz+1),c3f(nz+1),c4f(nz+1)
real(c_float)::dnw(nz+1),rdnw(nz+1),ptop,p(nx,nz+1,ny),al(nx,nz+1,ny)
real::moist(nx,nz+1,ny,1),muts(nx,ny),dummy(nz+1)
moist=0.;dummy=0.;muts=mub+mu;p=0.;al=0.
call calc_p_rho_phi(moist,1,opt,al,alb,mu,muts,c1,c2,c3h,c4h,c3f,c4f,ph,phb,p,pb, &
 t,100000.,300.,ptop,dummy,dummy,dnw,rdnw,dummy,.true.,1, &
 1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz+1)
end subroutine
end module
subroutine wrf_error_fatal(message)
character(*)::message
stop 90
end subroutine
subroutine VPOW(y,x,p,n)
integer::n
real::y(n),x(n),p(n)
y=x**p
end subroutine
'''
(work/'calc.F90').write_text(module)
fc='<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran'
subprocess.run([fc,'-cpp','-shared','-fPIC','-O2','-march=nocona','-ffp-contract=off','-fcheck=bounds',
                '-ffree-line-length-none',str(work/'calc.F90'),'-o',str(work/'calc.so')],cwd=work,check=True)
lib=ctypes.CDLL(str(work/'calc.so'));lib.gate.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*18
feedback,feedback_meta=build(work/'feedback')
def read(domain,ny,nx):
    with Dataset(case/f'wrfinput_{domain}') as ds:
        vals={name:np.asarray(ds[name][0],dtype=np.float32)[..., :ny,:nx] for name in ('T','PH','PHB','MU','MUB','PB')}
        for name in ('C1H','C2H','C3H','C4H','C3F','C4F','DNW','RDNW'):
            vals[name.lower()]=np.asarray(ds[name][0],np.float32).ravel()
        vals['ptop']=np.asarray(ds['P_TOP'][0],np.float32).reshape(())
    return vals
p,c=read('d01',20,22),read('d02',30,36)
def alpha(v):
    col=v['MUB'][None]
    upper=v['c3f'][1:,None,None]*col+v['c4f'][1:,None,None]+v['ptop']
    lower=v['c3f'][:-1,None,None]*col+v['c4f'][:-1,None,None]+v['ptop']
    half=v['c3h'][:,None,None]*col+v['c4h'][:,None,None]+v['ptop']
    return (v['PHB'][1:]-v['PHB'][:-1])/half/np.log(lower/upper)
pa,ca=alpha(p),alpha(c)
data={};cases=[]
for option in (1,2):
 for smooth in (0,1,2):
    key=str(len(cases));vals={k:v.copy() for k,v in p.items()}
    for name in ('T','PH','PHB','MU','MUB','PB'):
        vals[name]=call(feedback,p[name],c[name],smooth=smooth)
    vals['alb']=call(feedback,pa,ca,smooth=smooth)
    nz,ny,nx=vals['T'].shape
    def field(a):
        if a.ndim==2:return np.asfortranarray(a.T)
        if a.shape[0]==nz:a=np.concatenate((a,np.zeros_like(a[:1])))
        return np.asfortranarray(a.transpose(2,0,1))
    ordered=[field(vals[n]) for n in ('T','PH','PHB','MU','MUB','alb','PB')]
    for name in ('c1h','c2h','c3h','c4h','c3f','c4f','dnw','rdnw'):
        v=vals[name];ordered.append(np.ascontiguousarray(np.pad(v,(0,nz+1-len(v)))))
    ordered.append(np.ascontiguousarray(vals['ptop']))
    outp=np.zeros((nx,nz+1,ny),np.float32,order='F');outa=np.zeros_like(outp,order='F')
    lib.gate(nx,ny,nz,option,*[ctypes.c_void_p(a.ctypes.data) for a in (*ordered,outp,outa)])
    for name,v in vals.items():data[f'{name}_{key}']=v
    data[f'p_{key}']=outp.transpose(1,2,0)[:nz].copy()
    data[f'al_{key}']=outa.transpose(1,2,0)[:nz].copy()
    cases.append(dict(hypsometric_opt=option,smooth=smooth,nz=nz,ny=ny,nx=nx))
np.savez_compressed(fixture,**data)
fixture.with_suffix('.json').write_text(json.dumps(dict(cases=cases,
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),
    feedback_sources=feedback_meta,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    provenance='WN3 native-input crops; separately supplied fedback ALB; verbatim original CPU-WRF calc_p_rho_phi'),indent=2)+'\n')
print(len(cases),'ALB/EOS cases',hashlib.sha256(fixture.read_bytes()).hexdigest())
