"""Pristine prognostic GROUNDWATER including its original DOUBLE S_NODE island."""
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();routine=extract(text,"GROUNDWATER")
    (build/"reference.f90").write_text("""module reference
implicit none
type noahmp_parameters
real :: BEXP(4),SMCMAX(4),PSISAT(4),DKSAT(4),TIMEAN
end type
contains
"""+routine+"\nend module\n")
    (build/"driver.f90").write_text("""program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: n,i,j,steps
real :: dt,sh2o(4),sice(4),wcnd(4),zsoil(4),stc(-2:4),zwt,wa,wt,fcrmax,qin,qdis
read(*,*)p%BEXP
read(*,*)p%SMCMAX
read(*,*)p%PSISAT
read(*,*)p%DKSAT
read(*,*)p%TIMEAN
read(*,*)zsoil
read(*,*)n,steps
stc=270.
do i=1,n
read(*,*)dt,zwt,wa,wt,fcrmax,sh2o,sice,wcnd
do j=1,steps
call GROUNDWATER(p,3,4,dt,sice,zsoil,stc,wcnd,fcrmax,1,1,sh2o,zwt,wa,wt,qin,qdis)
write(*,'(9(ES25.16E3,1X))')zwt,wa,wt,sh2o,qin,qdis
enddo
enddo
end program
""")
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O0","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    params=dict(bexp=[4.5,5.,4.,5.5],smcmax=[.45,.47,.45,.43],psisat=[.35,.4,.3,.45],dksat=[1.e-5,2.e-6,5.e-6,1.e-6],timean=10.5)
    zsoil=[-.1,-.4,-1.,-2.];steps=4
    cases=[]
    for zwt in (.15,.6,1.5,2.5,10.):
        for sh,ice,fcr,dt in ((.2,0.,0.,18.),(.44,0.,0.,1800.),(.1,.25,.8,18.),(.005,.2,.5,86400.)):
            for saturated in (False,True):
                cases.append(dict(dt=dt,state=[zwt,4900.,5100. if saturated else 4900.],fcrmax=fcr,
                                  sh2o=[min(sh,v) for v in params["smcmax"]],sice=[ice]*4,wcnd=[1.e-7,2.e-8,1.e-8,3.e-9]))
    def run(rows):
        lines=[" ".join(map(str,params[k])) for k in ("bexp","smcmax","psisat","dksat")]
        lines += [str(params["timean"])," ".join(map(str,zsoil)),f"{len(rows)} {steps}"]
        lines += [" ".join(map(str,[c["dt"],*c["state"],c["fcrmax"],*c["sh2o"],*c["sice"],*c["wcnd"]])) for c in rows]
        r=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=" ",dtype=np.float32).reshape(len(rows),steps,9);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ("dt","state","fcrmax","sh2o","sice","wcnd"):
            for index in (range(len(c[key])) if isinstance(c[key],list) else (None,)):
                for d in (-np.inf,np.inf):
                    v=c.copy();value=c[key][index] if index is not None else c[key]
                    changed=float(np.nextafter(np.float32(value),np.float32(d)))
                    if index is None:v[key]=changed
                    else:v[key]=c[key].copy();v[key][index]=changed
                    variants.append(v);owner.append(i)
    results=run(variants);assert np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),flags=flags,created_utc=datetime.now(timezone.utc).isoformat(),
        parameters=params,zsoil=zsoil,cases=cases,steps=steps,ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6),
        scope="40constructed deep/shallow, saturated/unsaturated/cold/frozen/dry/longstep originalGROUNDWATER trajectories4steps; originalREAL8S_NODEonly, allprognostic/forcingREAL4operands independently +/-1ULP"),indent=1)+"\n")


if __name__=="__main__":main()
