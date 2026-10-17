"""Standalone verbatim WRF-Solar operator oracle; no JAX/model import.

Run with the lane CPU/QUIET/timeout rules and an explicit disposable build dir:
  python proofs/u2_diag/solar_generate_oracle.py <USER_HOME>/wrf_gpu2_lanes/u2-diag/solar_oracle_build
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/"data/wrf_pristine/WRF/phys/module_radiation_driver.F"
BUILD=Path(sys.argv[1]).resolve()
BUILD.mkdir(parents=True,exist_ok=True)
source=SOURCE.read_text()
start=source.index("   subroutine update_swinterp_parameters(")
end=source.index("   end subroutine interp_sw_radiation",start)+len("   end subroutine interp_sw_radiation")
block=source[start:end]
wrapper="module extracted_solar\ncontains\n"+block+"\nend module\n"
wrapper+="""program oracle
use extracted_solar
implicit none
integer,parameter::n=16,ns=11
integer::step,run_rad,first_step
real::fields(n,1,7),coszen(n,1),loc(n,1),direct(n,1),down(n,1),albedo(n,1),fluxes(n,1,5)
open(10,file='input.bin',access='stream',form='unformatted',status='old')
open(11,file='output.bin',access='stream',form='unformatted',status='replace')
read(10) fields
do step=1,ns
 read(10) run_rad,first_step,coszen,loc,direct,down,albedo
 if (run_rad.eq.1) then
  if (first_step.eq.1) then
   fields(:,:,2)=0.;fields(:,:,3)=0.;fields(:,:,5)=0.;fields(:,:,6)=0.
  endif
  call update_swinterp_parameters(1,n,1,1,1,n,1,1,coszen,loc,direct,down, &
      fields(:,:,1),fields(:,:,2),fields(:,:,3),fields(:,:,4),fields(:,:,5),fields(:,:,6),fields(:,:,7))
 endif
 call interp_sw_radiation(1,n,1,1,1,n,1,1,fields(:,:,7),loc,fields(:,:,1), &
      fields(:,:,2),fields(:,:,3),fields(:,:,4),fields(:,:,5),fields(:,:,6),albedo, &
      fluxes(:,:,1),fluxes(:,:,2),fluxes(:,:,3),fluxes(:,:,4),fluxes(:,:,5))
 write(11) fields,fluxes
enddo
close(10);close(11)
end program
"""
(BUILD/"oracle.f90").write_text(wrapper)
compiler=os.environ.get("FC","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
command=[compiler,"-O0","-ffp-contract=off","-fno-fast-math","-ffree-line-length-none","oracle.f90","-o","oracle"]
subprocess.run(command,cwd=BUILD,check=True,timeout=30)
F=np.float32
cos=np.array([.65,.2,1e-4,np.nextafter(F(1e-4),F(1)),.9,.6,.5,.5,.4,.4,.35,.35,.9,.9,.7,.7],np.float32)
loc=np.array([.6,.1,.8,.00010001,.95,.55,.5,.4,.3,.5,.3,.4,-.1,.0001,.6,.9],np.float32)
direct=np.array([600,50,400,.001,800,400,0,.5,200,100,1000,1,800,400,0,-5],np.float32)
down=np.array([800,100,600,.002,1000,700,1,.2,350,150,1200,2,1000,700,0,-2],np.float32)
albedo=np.array([.2,.9,.1,1,0,.3,.5,.4,.2,.4,.1,.2,.3,.4,.2,.1],np.float32)
radflags=[True,False,False,True,False,True,False,True,True,False,True]
rows=[]
for step in range(11):
    c,local,d,g,a=(x.copy() for x in (cos,loc,direct,down,albedo))
    local*=F([1,1.1,.8,.9,.7,.9,1,.8,1,.95,.5][step])
    a=np.minimum(F(.98),a+F(step)*F(.005))
    if step==3:
        c[13]=F(.4);local[13]=F(.3)
        d*=F(.35);g*=F(.55)
    if step==4:
        local[:2]=F(-.01)
    if step==5:
        c[:2]=F(-.2);local[:2]=F(-.1);d[:2]=F(0);g[:2]=F(0)
    if step==6:
        local[:2]=F(.1)  # sunrise without a new radiation solve stays zero
    if step==7:
        c[:2]=F(.2);local[:2]=F(.1);d[:2]=F(80);g[:2]=F(120)
    if step==8:
        c[:2]=F(.2);local[:2]=F(.15);d[:2]=F(800);g[:2]=F(950)
    if step==10:
        c*=F(.7);d*=F(1.5);g*=F(1.2)
    rows.append(np.stack((c,local,d,g,a)))
rows=np.asarray(rows,np.float32)
warm=np.stack((np.linspace(100,700,16,dtype=np.float32),np.full(16,.7,np.float32),np.full(16,800,np.float32),
               np.linspace(200,900,16,dtype=np.float32),np.full(16,1.2,np.float32),np.full(16,1000,np.float32),
               np.array([.7,.15,.8,.001,.95,.55,.5,.4,.3,.5,0,0,.95,.8,.8,.5],np.float32)))
warm[2,0]=F(0);warm[2,1]=F(-1);warm[5,8]=F(0);warm[5,9]=F(-1)
cases=[]
for label,initial in (("cold_start",warm), ("warm_restart_zero_refs",warm)):
    firstflags=[label=="cold_start"]+[False]*10
    with (BUILD/"input.bin").open("wb") as stream:
        stream.write(initial.tobytes())
        for run,first,row in zip(radflags,firstflags,rows,strict=True):
            stream.write(np.asarray((run,first),np.int32).tobytes());stream.write(row.tobytes())
    subprocess.run([str(BUILD/"oracle")],cwd=BUILD,check=True,timeout=10)
    expected=np.fromfile(BUILD/"output.bin",np.float32).reshape(11,12,16)
    assert np.isfinite(expected).all()
    cases.append(dict(label=label,initial=initial.tolist(),first_step=firstflags,
                      run_radiation=radflags,inputs=rows.tolist(),expected=expected.tolist()))
expected=np.asarray([x["expected"] for x in cases],np.float32)
census=dict(direct_lower_slope=int(np.sum(expected[:,:,1]==F(-.5))),direct_upper_slope=int(np.sum(expected[:,:,1]==F(2.5))),
            direct_zero_slope=int(np.sum(expected[:,:,1]==F(0))),
            direct_interior_slope=int(np.sum((expected[:,:,1]>F(-.5))&(expected[:,:,1]<F(2.5))&(expected[:,:,1]!=F(0)))),
            positive_dni=int(np.sum(expected[:,:,9]>0)),night_zero_dni=int(np.sum(expected[:,:,9]==0)),
            negative_diffuse=int(np.sum(expected[:,:,10]<0)))
assert all(census[name]>0 for name in ("direct_lower_slope","direct_upper_slope","direct_zero_slope","direct_interior_slope","positive_dni","night_zero_dni","negative_diffuse"))
fixture=dict(provenance="verbatim unmodified WRF update/interp routines; no gpuwrf/JAX import",
             source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",block_sha256=hashlib.sha256(block.encode()).hexdigest(),
             precision="WRF REAL=4; no DOUBLE",compiler=command,
             tolerance="frozen before candidate execution: rtol=2e-6,atol=2e-5 for all fields; no tolerance fitting",
             shape="16 independent surface cells",units="irradiance W m-2, cosines/exponents dimensionless",
             output_order=["swddir_ref","bb","Bx","swdown_ref","gg","Gx","coszen_ref","swdown","swddir","swddni","swddif","gsw"],
             sampling="update only Radiation_step; interpolation all ownsteps, including radiation; initial four parameters reset on step1",
             regimes=census,cases=cases,qualification="ORIGINAL SOURCE ORACLE ONLY; source draft has not been executed")
target=HERE/"solar_oracle.json"
target.write_text(json.dumps(fixture,separators=(",",":"))+"\n")
print(json.dumps(dict(fixture=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),regimes=census)))
