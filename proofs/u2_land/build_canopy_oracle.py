"""Original SFCDIF2 caller sweeps and CANRES/CALHUM, with frozen REAL4 ULP trajectories."""
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();routines={key:extract(text,key) for key in ("SFCDIF2","CANRES","CALHUM")}
    (build/"reference.f90").write_text('''module reference
implicit none
real,parameter :: GRAV=9.80616
type noahmp_parameters
real :: czil,rsmin,rsmax,rgl,hs,topt
end type
contains
'''+"\n".join(routines.values())+"\nend module\n")
    (build/"driver.f90").write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: parameters
integer :: mode,n,i,it,snow
real :: z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar,ustar,par,btran,eah,pressure,fveg,rc,psn
read(*,*) mode,n
do i=1,n
if(mode==1) then
read(*,*) z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar,ustar,parameters%czil,snow
do it=1,20
call SFCDIF2(parameters,it,z0,thz0,thlm,speed,zlm,1,1,akms,akhs,rlmo,wstar,ustar)
akms=akms/speed;akhs=akhs/speed
if(snow==1) then
akms=min(.01,akms);akhs=min(.01,akhs)
endif
write(*,'(5(ES25.16E3,1X))') akms,akhs,rlmo,wstar,ustar
enddo
else
read(*,*) par,thz0,btran,eah,pressure,fveg,parameters%rsmin,parameters%rsmax,parameters%rgl,parameters%hs,parameters%topt
call CANRES(parameters,par,thz0,btran,eah,pressure,fveg,rc,psn,1,1)
write(*,'(2(ES25.16E3,1X))') rc,psn
endif
enddo
end program
''')
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O0","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    sf_fields=("z0","thz0","thlm","speed","zlm","akms","akhs","rlmo","wstar","ustar","czil","snow")
    jarvis_fields=("par","temperature","btran","eah","pressure","fveg","rsmin","rsmax","rgl","hs","topt")
    sf_cases=[dict(zip(sf_fields,(z,280.,280.+delta,wind,20.,.01,.01,0.,0.,.1,.1,snow)))
              for z in (.002,.1,1.) for delta in (-15.,0.,15.) for wind in (.1,1.,10.) for snow in (0,1)]
    jarvis_cases=[dict(zip(jarvis_fields,(par,temp,btran,1000.,80000.,fveg,125.,5000.,30.,47.35,298.)))
                  for par in (0.,10.,200.) for temp in (260.,280.,300.,320.) for btran in (.01,.5,1.) for fveg in (.05,.8)]
    def run(mode,rows):
        fields=sf_fields if mode==1 else jarvis_fields
        lines=[f"{mode} {len(rows)}"]+[" ".join(str(c[k]) for k in fields) for c in rows]
        r=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=" ",dtype=np.float32).reshape(len(rows),100 if mode==1 else 2)
        assert np.isfinite(out).all();return out
    groups={}
    for name,mode,cases,fields in (("sfcdif2",1,sf_cases,sf_fields[:-1]),("jarvis",2,jarvis_cases,jarvis_fields)):
        expected=run(mode,cases);variants=[];owner=[];indices=[]
        for i,c in enumerate(cases):
            indices.append(len(variants));variants.append(c.copy());owner.append(i)
            for key in fields:
                for d in (-np.inf,np.inf):
                    v=c.copy();v[key]=float(np.nextafter(np.float32(c[key]),np.float32(d)))
                    variants.append(v);owner.append(i)
        result=run(mode,variants);assert np.array_equal(result[indices],expected)
        low,high=expected.copy(),expected.copy()
        for i,row in zip(owner,result):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
        for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
        groups[name]=dict(cases=cases,fields=list(fields),ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6))
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},
        created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,groups=groups,
        scope="54constructed stable/neutral/unstable xroughness/wind cases20sweeps; WRFcallerdivideswind every sweep and bare-onlysnowcap;72Jarvis PAR/T/stress/FVEG columns;allREAL4inputs varied independently"),indent=1)+"\n")


if __name__=="__main__":main()
