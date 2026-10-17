"""Pristine PHENOLOGY plus the verbatim SFLX FVEG selection for non-carbon modes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4])
    crop_mode='--crop' in sys.argv[4:]
    codes=tuple(int(v) for v in sys.argv[4:] if v!='--crop') or (1,3,4,7,8,9)
    build.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    routine=extract(text,"PHENOLOGY")
    caller=text[text.index("!input GVF should be consistent with LAI"):text.index("! Calling dynamic irrigation scheme")]
    module='''module phen_reference
implicit none
integer :: DVEG
type noahmp_parameters
real :: LAIM(12),SAIM(12),HVT,HVB,TMIN
integer :: ISWATER,ISBARREN,ISICE
logical :: urban_flag
end type
contains
'''+routine+"\nend module\n"
    (build/"reference.f90").write_text(module)
    driver='''program oracle
use phen_reference
implicit none
type(noahmp_parameters) :: parameters
integer :: n,i,vegtyp,yearlen,croptype,opt_crop,pgs
real :: snowh,tv,lat,julian,lai,sai,troot,elai,esai,igs,fb,shdfac,shdmax,fveg
croptype=0;opt_crop=0;pgs=0;troot=280.
parameters%ISWATER=17;parameters%ISBARREN=16;parameters%ISICE=15
read(*,*) n
do i=1,n
read(*,*) dveg,vegtyp,yearlen,snowh,tv,lat,julian,lai,sai,shdfac,shdmax
read(*,*) parameters%LAIM
read(*,*) parameters%SAIM
read(*,*) parameters%HVT,parameters%HVB,parameters%TMIN
parameters%urban_flag=vegtyp==13
call PHENOLOGY(parameters,vegtyp,croptype,snowh,tv,lat,yearlen,julian,lai,sai,troot,elai,esai,igs,pgs,fb)
'''+caller+'''write(*,'(6(ES25.16E3,1X))') lai,sai,elai,esai,fveg,igs
enddo
end program
subroutine wrf_error_fatal(message)
character(len=*),intent(in) :: message
print *,message
stop 1
end subroutine
'''
    if crop_mode:
        driver=driver.replace('croptype=0;opt_crop=0;pgs=0;troot=280.','croptype=1;opt_crop=1;pgs=0;troot=280.')
        driver=driver.replace('read(*,*) parameters%LAIM','read(*,*) croptype,pgs\nread(*,*) parameters%LAIM')
    (build/"driver.f90").write_text(driver)
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O2","-ffree-form","-ffree-line-length-none","-ffp-contract=off",
           "-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    # Actual MPTABLE data are used; only the regime operands are constructed.
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters,_parse_mptable_block
    wrf_root=next(parent for parent in source.parents if (parent/"run/SOILPARM.TBL").exists())
    table=wrf_root/"run/MPTABLE.TBL"
    p=load_noahmp_parameters(table.parent)
    block=_parse_mptable_block(table.read_text(),"noahmp_modis_parameters")
    cases=[]
    regimes=((2,0.,2.,.2,.8,.5,280.,15.),(5,5.,3.,.4,.9,-.5,265.,195.),
             (12,.1,.04,.01,.7,.5,273.,85.),(14,.3,2.,.3,.8,.5,280.,200.),
             (13,0.,2.,.5,.8,.5,280.,15.),(15,1.,2.,.2,.5,.5,260.,15.),
             (16,0.,2.,.3,.8,.5,280.,15.),(17,0.,2.,.3,.8,-.5,280.,15.))
    for dveg in codes:
        for regime,(veg,snowh,lai,shdfac,shdmax,lat,tv,julian) in enumerate(regimes):
            if crop_mode:veg=12
            cases.append(dict(dveg=dveg,vegtyp=veg,yearlen=365,snowh=snowh,tv=tv,lat=lat,julian=julian,
                lai=lai,sai=9.,shdfac=shdfac,shdmax=shdmax,
                laim=np.asarray(p.laim[veg]).tolist(),saim=np.asarray(p.saim[veg]).tolist(),
                hvt=float(p.hvt[veg]),hvb=float(p.hvb[veg]),tmin=float(block["TMIN"][veg-1])))
            if crop_mode:cases[-1].update(croptype=1,pgs=(0,1,2,3,4,6,7,8)[regime])
    def run(rows):
        lines=[str(len(rows))]
        for c in rows:
            lines += [" ".join(map(str,[c[key] for key in ("dveg","vegtyp","yearlen","snowh","tv","lat","julian","lai","sai","shdfac","shdmax")])),
                      " ".join(map(str,c["laim"]))," ".join(map(str,c["saim"])),f'{c["hvt"]} {c["hvb"]} {c["tmin"]}']
            if crop_mode:lines.insert(len(lines)-3,f'{c["croptype"]} {c["pgs"]}')
        result=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",
                              capture_output=True,text=True,check=True,timeout=30)
        return np.asarray([[float(value) for value in line.split()] for line in result.stdout.splitlines()],np.float32)
    expected=run(cases)
    assert expected.shape==(len(codes)*8,6) and np.isfinite(expected).all()
    variants=[];owner=[];nominals=[]
    fields=("snowh","tv","lat","julian","lai","sai","shdfac","shdmax")
    for i,case in enumerate(cases):
        nominals.append(len(variants));variants.append(case.copy());owner.append(i)
        for field in fields:
            for positive in (False,True):
                row=case.copy();row[field]=float(np.nextafter(np.float32(row[field]),np.float32(np.inf if positive else -np.inf)))
                variants.append(row);owner.append(i)
    results=run(variants)
    assert np.isfinite(results).all() and np.array_equal(results[nominals],expected)
    lo,hi=expected.copy(),expected.copy()
    for i,row in zip(owner,results):lo[i]=np.minimum(lo[i],row);hi[i]=np.maximum(hi[i],row)
    for i,case in enumerate(cases):case.update(expected=expected[i].tolist(),low=lo[i].tolist(),high=hi[i].tolist())
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
        table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),flags=flags,created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases,variables_out=["lai","sai","elai","esai","fveg","igs"],nominal_tolerance=dict(atol=1.e-6,rtol=3.e-6),
        census=dict(columns=len(cases),ulp_variants=len(variants),constructed=True,
                    regimes="NH/SH monthly clocks; prescribed/calculated/max vegetation; input LAI below cutoff; tall/short buried canopy; urban/ice/barren/water"),
        scope=f"MODIS vegetation phenology; CROPTYPE{1 if crop_mode else 0}; original PHENOLOGY + SFLX FVEG caller; old growth-stage/snow-burial coverage"),indent=1)+"\n")


if __name__=="__main__":main()
