"""Unmodified pristine HRT/HSTEP/ROSR12/FRH2O REAL4 references and input-ULP bounds."""
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    routines={name:extract(text,name) for name in ("HRT","HSTEP","ROSR12","FRH2O")}
    module='''module reference
implicit none
real,parameter :: GRAV=9.80616,HFUS=.3336e6,TFRZ=273.16
integer :: opt_tbot,opt_stc
type noahmp_parameters
real :: bexp(4),smcmax(4),psisat(4)
end type
contains
'''+"\n".join(routines.values())+"\nend module\n"
    (build/"reference.f90").write_text(module)
    driver='''program oracle
use reference
implicit none
type(noahmp_parameters) :: parameters
integer :: i,n,mode,isnow
real :: stc(-2:4),df(-2:4),hcpct(-2:4),zsnso(-2:4),ai(-2:4),bi(-2:4),ci(-2:4),rhs(-2:4),phi(-2:4),tbot,ssoil,zbot,dt,botflx,free,temperature,smc,sh2o
read(*,*) mode,n
do i=1,n
if(mode==1) then
read(*,*) opt_tbot,opt_stc,isnow,ssoil,tbot,zbot,dt
read(*,*) stc
read(*,*) df
read(*,*) hcpct
read(*,*) zsnso
ai=0.;bi=0.;ci=0.;rhs=0.;phi=0.
call HRT(parameters,3,4,isnow,zsnso,stc,tbot,zbot,dt,df,hcpct,ssoil,phi,ai,bi,ci,rhs,botflx)
write(*,'(28(ES25.16E3,1X))') ai,bi,ci,rhs
call HSTEP(parameters,3,4,isnow,dt,ai,bi,ci,rhs,stc)
write(*,'(7(ES25.16E3,1X))') stc
else
read(*,*) temperature,smc,sh2o,parameters%bexp(1),parameters%smcmax(1),parameters%psisat(1)
call FRH2O(parameters,1,free,temperature,smc,sh2o)
write(*,'(ES25.16E3)') free
endif
enddo
end program
subroutine wrf_message(message)
character(len=*),intent(in) :: message
end subroutine
'''
    (build/"driver.f90").write_text(driver)
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O2","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    def run(mode,rows):
        lines=[f"{mode} {len(rows)}"]
        for c in rows:
            if mode==1:
                lines.append(" ".join(map(str,[c[k] for k in ("opt_tbot","opt_stc","isnow","ssoil","tbot","zbot","dt")])))
                lines.extend(" ".join(map(str,c[k])) for k in ("stc","df","hcpct","zsnso"))
            else:lines.append(" ".join(map(str,[c[k] for k in ("temperature","smc","sh2o","bexp","smcmax","psisat")])))
        r=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",capture_output=True,text=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=" ",dtype=np.float32).reshape(len(rows),35 if mode==1 else 1)
        assert np.isfinite(out).all();return out
    thermal=[]
    for tbot in (1,2):
        for stc in (1,2,3):
            for snow in (0,1,2,3):
                for cooling in (False,True):
                    depth=[-.05,-.1,-.2] if snow==3 else ([0.,-.1,-.2] if snow==2 else ([0.,0.,-.2] if snow==1 else [0.,0.,0.]))
                    snowh=-depth[-1]
                    thermal.append(dict(opt_tbot=tbot,opt_stc=stc,isnow=-snow,ssoil=-50. if cooling else 100.,tbot=285.,zbot=-8.-snowh,dt=18.,
                        stc=[265.,268.,270.,272.,275.,279.,282.],df=[.12,.2,.3,.8,1.,1.2,1.5],hcpct=[3.e5,4.e5,6.e5,1.9e6,2.e6,2.1e6,2.2e6],
                        zsnso=depth+[-.1-snowh,-.4-snowh,-1.-snowh,-2.-snowh]))
    freezing=[dict(temperature=t,smc=smc,sh2o=min(smc,.12),bexp=b,smcmax=.45,psisat=.35)
              for t in (220.,250.,265.,272.,273.158,273.159,273.16,280.) for b in (4.5,7.) for smc in (.01,.2,.4)]
    groups={}
    for name,mode,cases in (("thermal",1,thermal),("freezing",2,freezing)):
        nominal=run(mode,cases);variants=[];owners=[];indices=[]
        for i,c in enumerate(cases):
            indices.append(len(variants));variants.append(c.copy());owners.append(i)
            fields=("ssoil","tbot","zbot","dt","stc","df","hcpct","zsnso") if mode==1 else tuple(c)
            for key in fields:
                operands=range(len(c[key])) if isinstance(c[key],list) else (None,)
                for index in operands:
                    for direction in (-np.inf,np.inf):
                        v=dict(c);value=c[key][index] if index is not None else c[key]
                        changed=float(np.nextafter(np.float32(value),np.float32(direction)))
                        if index is not None:v[key]=c[key].copy();v[key][index]=changed
                        else:v[key]=changed
                        variants.append(v);owners.append(i)
        results=run(mode,variants);assert np.array_equal(results[indices],nominal)
        low,high=nominal.copy(),nominal.copy()
        for i,row in zip(owners,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
        for i,c in enumerate(cases):c.update(expected=nominal[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
        groups[name]=dict(cases=cases,ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6))
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),routine_sha256={key:hashlib.sha256(value.encode()).hexdigest() for key,value in routines.items()},
        flags=flags,created_utc=datetime.now(timezone.utc).isoformat(),groups=groups,
        scope="Constructed 48 thermal columns with 0..3snow layers,6boundary choices;48FRH2O dry/wet,cold/threshold/warm columns; all REAL4 operands varied independently +/-1ULP"),indent=1)+"\n")


if __name__=="__main__":main()
