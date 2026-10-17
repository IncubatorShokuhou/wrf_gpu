"""Compile verbatim pristine NWP/helicity operators, without importing gpuwrf.

CPU-only generator; run in the lane's immutable source snapshot with QUIET
guard, timeout120 and the assigned nice19 CPU. The small JSON is the retained
proof object; build scratch is disposable. REAL=4, DOUBLE=8, contraction off.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
BUILD = Path(sys.argv[1]).resolve()
BUILD.mkdir(parents=True,exist_ok=True)
NWP = WRF / "phys/module_diag_nwp.F"
DIFF = WRF / "dyn_em/module_diffusion_em.F"
source = DIFF.read_text()
start = source.index("    SUBROUTINE cal_helicity (")
end = source.index("    END SUBROUTINE cal_helicity",start)+len("    END SUBROUTINE cal_helicity")
helicity = source[start:end]
names = ("KESSLERSCHEME LINSCHEME SBU_YLINSCHEME WSM3SCHEME WSM5SCHEME "
         "WSM6SCHEME ETAMPNEW THOMPSON THOMPSONAERO THOMPSONGH MORR_TWO_MOMENT "
         "GSFCGCESCHEME WDM5SCHEME WDM6SCHEME MILBRANDT2MOM CAMMGMPSCHEME "
         "FULL_KHAIN_LYNN MORR_TM_AERO NSSL_2MOM FAST_KHAIN_LYNN_SHPUND").split()
stub = """module module_configure
type grid_config_rec_type
logical::open_xs=.false.,open_xe=.false.,open_ys=.false.,open_ye=.false.
logical::specified=.true.,nested=.false.,periodic_x=.false.,periodic_y=.false.
integer:: nssl_2moment_on=0,nssl_hail_on=0
end type
end module
module module_state_description
""" + "\n".join(f"integer,parameter::{n}={i+1}" for i,n in enumerate(names)) + """
end module
module module_mp_thompson
integer,parameter::idx_bg1=5
end module
module extracted_helicity
use module_configure
real,parameter::g=9.81
contains
""" + helicity + """
end module
subroutine wrf_debug(level,text)
integer::level
character*(*)::text
end subroutine
"""
(BUILD / "stubs.f90").write_text(stub)
(BUILD / "module_diag_nwp.F90").write_bytes(NWP.read_bytes())
driver = """program oracle
use module_configure
use module_state_description
use module_diag_nwp
use extracted_helicity
implicit none
integer,parameter::nx=7,ny=6,nz=9,ns=7
integer::step,i,j,k
type(grid_config_rec_type)::cfg
real::u(nx+1,nz+1,ny+1),v(nx+1,nz+1,ny+1),w(nx+1,nz+1,ny+1)
real::p8w(nx+1,nz+1,ny+1),qg(nx+1,nz+1,ny+1),rho(nx+1,nz+1,ny+1)
real::ph(nx+1,nz+1,ny+1),phb(nx+1,nz+1,ny+1),ref(nx+1,nz+1,ny+1)
real::zx(nx+1,nz+1,ny+1),zy(nx+1,nz+1,ny+1),rdzw(nx+1,nz+1,ny+1)
real::zero3(nx+1,nz+1,ny+1),temp(nx+1,nz+1,ny+1)
real::u10(nx+1,ny+1),v10(nx+1,ny+1),ht(nx+1,ny+1)
real::msfux(nx+1,ny+1),msfvy(nx+1,ny+1)
real::fields(nx+1,ny+1,10),colmean(nx+1,ny+1),numpts(nx+1,ny+1),grplint(nx+1,ny+1)
real::znw(nz+1),fnm(nz+1),fnp(nz+1),dn(nz+1),dnw(nz+1),constants(5)
integer::istart(1),iend(1),jstart(1),jend(1)
open(10,file='input.bin',access='stream',form='unformatted',status='old')
read(10) fields,u,v,w,p8w,qg,rho,ph,phb,ref,zx,zy,rdzw,u10,v10,ht,msfux,msfvy,znw,fnm,fnp,dn,dnw,constants
close(10)
open(11,file='output.bin',access='stream',form='unformatted',status='replace')
istart=1;iend=nx;jstart=1;jend=ny
zero3=0.;temp=270.
do step=1,ns
call cal_helicity(cfg,u,v,w,fields(:,:,10),fields(:,:,4),ph,phb,msfux,msfux,msfvy,msfvy,ht, &
 constants(1),constants(2),dn,dnw,rdzw,rdzw,fnm,fnp,constants(3),constants(4),constants(5),zx,zy, &
 1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
write(11) fields(1:nx,1:ny,4),fields(1:nx,1:ny,10)
call diagnostic_output_nwp(cfg, &
 1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz, &
 istart,iend,jstart,jend,1,nz,1,u,v,temp,p8w,20.,real(step)/3.,1, &
 THOMPSON,0,0,0,0.,0.,0.,0.,0.,0.,real(step-1)*20.,1,.false.,1,step,u10,v10,w, &
 fields(:,:,1),fields(:,:,4),fields(:,:,2),fields(:,:,3),znw,colmean,numpts,fields(:,:,5), &
 fields(:,:,6),grplint,fields(:,:,7),ref,fields(:,:,8),fields(:,:,9),qg, &
 qr_curr=zero3,nr_curr=zero3,rho=rho,ph=ph,phb=phb,g=9.81,max_time_step=20,adaptive_ts=.false.)
do i=1,10
 write(11) fields(1:nx,1:ny,i)
enddo
enddo
close(11)
end program
"""
(BUILD / "driver.f90").write_text(driver)
compiler = os.environ.get("FC","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
command = [compiler,"-O0","-ffp-contract=off","-fno-fast-math","-cpp","-DNMM_CORE=0", "-ffree-line-length-none",
           "stubs.f90","module_diag_nwp.F90","driver.f90","-o","oracle"]
subprocess.run(command,cwd=BUILD,check=True,timeout=60)
nx,ny,nz = 7,6,9
y,x = np.mgrid[:ny,:nx].astype(np.float32)
k = np.arange(nz+1,dtype=np.float32)[:,None,None]
massk = k[:-1]
f = np.float32
terrain = f(150)+f(25)*x+f(11)*y
height = np.array([0,550,1300,2050,2800,3550,4300,5050,7500,12000],np.float32)
phb = (height[:,None,None]+terrain[None])*f(9.81)
ph = f(0.37)*k*x[None]-f(.12)*k*y[None]+f(.007)*x[None]*y[None]
z = (ph+phb)/f(9.81)
rdzw = f(1)/(z[1:]-z[:-1])
zx,zy = np.zeros_like(ph),np.zeros_like(ph)
rdx,rdy = f(1/3000),f(1/2700)
zx[:,:,1:] = rdx*(phb[:,:,1:]-phb[:,:,:-1])/f(9.81)+rdx*(ph[:,:,1:]-ph[:,:,:-1])/f(9.81)
zy[:,1:,:] = rdy*(phb[:,1:,:]-phb[:,:-1,:])/f(9.81)+rdy*(ph[:,1:,:]-ph[:,:-1,:])/f(9.81)
# Native C-grid velocities; all-positive and reversing columns exercise the gate.
yu,xu=np.mgrid[:ny,:nx+1].astype(np.float32)
yv,xv=np.mgrid[:ny+1,:nx].astype(np.float32)
u=f(6)-f(.4)*yu[None]+f(.13)*massk+f(.007)*xu[None]*massk
v=f(4)+f(.8)*xv[None]-f(.07)*massk+f(.009)*yv[None]*massk
w=np.broadcast_to(f(1.3)+f(.12)*k+f(.02)*x[None]-f(.01)*y[None],(nz+1,ny,nx)).copy()
w[4:6,2:4,4:6]=f(-.6)
rho=np.broadcast_to(f(1.1)-f(.07)*massk,(nz,ny,nx)).copy()
pattern=np.array([1e-12,1e-8,9.99999e-7,1e-6,1.000001e-6,3e-6,1e-4,1e-3,3e-2],np.float32)
qg=pattern[(np.arange(nz)[:,None,None]+np.arange(ny)[None,:,None]+np.arange(nx)[None,None,:])%len(pattern)]/rho
p8w=np.broadcast_to((f(98000)-f(8500)*k),(nz+1,ny,nx)).copy()
p8w[5,0,0]=f(40000);p8w[5,0,1]=np.nextafter(f(40000),f(np.inf))
refl=np.broadcast_to(f(3)*massk-f(5)+x[None]-f(2)*y[None],(nz,ny,nx)).copy()
u10=f(3)+x*f(.25);v10=f(-4)+y*f(.2)
msfux=f(1.01)+f(.001)*xu+f(.002)*yu
msfvy=f(.99)+f(.002)*xv-f(.001)*yv
znw=np.array([1,.9,.8,.7,.6,.5,.4,.3,.2,0],np.float32)
fnm=np.linspace(.35,.65,nz+1,dtype=np.float32);fnp=f(1)-fnm
dn=np.linspace(-.08,-.15,nz+1,dtype=np.float32);dnw=np.linspace(-.07,-.13,nz+1,dtype=np.float32)
constants=np.array([rdx,rdy,1.5,-.7,.2],np.float32)
initial=np.stack([np.full((ny,nx),f(.25)*(i+1),np.float32) for i in range(10)])
initial[9]=f(.013)*x+f(.019)*y
inputs=dict(u=u,v=v,w=w,p8w=p8w,qg=qg,rho=rho,ph=ph,phb=phb,refl_10cm=refl,zx=zx,zy=zy,rdzw=rdzw,
            u10=u10,v10=v10,terrain=terrain,msfux=msfux,msfvy=msfvy,znw=znw,fnm=fnm,fnp=fnp,dn=dn,dnw=dnw,constants=constants)

def pad3(a):
    # Source memory includes staggered eastern/northern endpoints and top.
    return np.pad(a,((0,nz+1-a.shape[0]),(0,ny+1-a.shape[1]),(0,nx+1-a.shape[2])),mode="edge")

def pad2(a):
    return np.pad(a,((0,ny+1-a.shape[0]),(0,nx+1-a.shape[1])),mode="edge")

with (BUILD / "input.bin").open("wb") as stream:
    stream.write(np.stack([pad2(a) for a in initial]).astype(np.float32).tobytes())
    for name in ("u","v","w","p8w","qg","rho","ph","phb","refl_10cm","zx","zy","rdzw"):
        stream.write(pad3(inputs[name]).transpose(1,0,2).astype(np.float32).tobytes())
    for name in ("u10","v10","terrain","msfux","msfvy"):
        stream.write(pad2(inputs[name]).astype(np.float32).tobytes())
    for name in ("znw","fnm","fnp","dn","dnw","constants"):
        stream.write(inputs[name].astype(np.float32).tobytes())
subprocess.run([str(BUILD / "oracle")],cwd=BUILD,check=True,timeout=30)
output=np.fromfile(BUILD / "output.bin",np.float32).reshape(7,12,ny,nx)
assert np.isfinite(output).all()
assert output[:,0].max()>0 and output[:,9].max()>0
fixture=dict(provenance="verbatim original WRF module + extracted cal_helicity; no gpuwrf import",precision="WRF REAL=4, DOUBLE=8",
             sources={str(p.relative_to(WRF)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (NWP,DIFF)},
             helicity_block_sha256=hashlib.sha256(helicity.encode()).hexdigest(),compiler=command,
             tolerance="NWP REAL 4 ULP; hail diameter 4 ULP; UH 5e-6 relative + 1e-6 absolute, frozen before execution",
             dimension_order="mass arrays (z,y,x), surface (y,x); native staggered u/v/w; specified BC ide=nx+1,jde=ny+1",
             units="SI; radar dBZ; hail metres; history minutes",itimestep=list(range(1,8)),dt=20.,history_interval=1,
             initial=initial.tolist(),inputs={n:a.tolist() for n,a in inputs.items()},expected=output.tolist())
target=ROOT / "proofs/u2_diag/nwp_oracle.json"
target.write_text(json.dumps(fixture,separators=(",",":"))+"\n")
print(json.dumps(dict(fixture=str(target),max_helicity=float(output[:,0].max()),max_hail=float(output[:,9:11].max()),steps=7)))
