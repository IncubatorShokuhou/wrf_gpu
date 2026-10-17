"""Stage verbatim ENERGY soil-stress and raw-resistance branches as a REAL4 oracle."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def main():
    source,build,target=map(Path,sys.argv[1:4])
    build.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    stress=text[text.index("          IF(OPT_BTR == 1) then"):text.index("          BTRANI(IZ) = MAX(MPE")]
    resistance=text[text.index("       IF(OPT_RSF == 1 .OR. OPT_RSF == 4) THEN"):text.index("       IF(SH2O(1) < 0.01")]
    driver='''program oracle
implicit none
type parameters_type
real :: SMCWLT(4),SMCREF(4),SMCMAX(4),PSISAT(4),BEXP(4),RSURF_EXP,RSURF_SNOW
end type
type(parameters_type) :: parameters
integer :: n,k,iz,opt_btr,opt_rsf
real :: sh2o(4),zsoil(4),gx,psi,psiwlt,fsno,bevap,l_rsurf,d_rsurf,rsurf,result(4)
psiwlt=-150.
read(*,*) parameters%SMCWLT
read(*,*) parameters%SMCREF
read(*,*) parameters%SMCMAX
read(*,*) parameters%PSISAT
read(*,*) parameters%BEXP
read(*,*) parameters%RSURF_EXP,parameters%RSURF_SNOW
read(*,*) zsoil
read(*,*) n
do k=1,n
read(*,*) opt_btr,opt_rsf,sh2o,fsno
do iz=1,4
'''+stress+'''result(iz)=gx
enddo
bevap=max(0.,sh2o(1)/parameters%SMCMAX(1))
'''+resistance+'''write(*,'(5(ES25.16E3,1X))') result,rsurf
enddo
end program
'''
    (build/"driver.f90").write_text(driver)
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O2","-ffree-form","-ffree-line-length-none","-ffp-contract=off",
           "-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    params=dict(smcwlt=[.1,.12,.1,.08],smcref=[.3,.32,.3,.28],smcmax=[.45,.47,.45,.43],
                psisat=[.35,.4,.3,.45],bexp=[4.5,5.,4.,5.5],rsurf_exp=5.,rsurf_snow=50.)
    zsoil=[-.1,-.4,-1.,-2.]
    cases=[]
    for btr in (1,2,3):
        for rsf in (1,2,3,4):
            for sh,fsno in ((.005,0.),(.02,.9),(.1,0.),(.12,.5),(.2,0.),(.3,.5),(.4,1.),(.45,0.)):
                cases.append(dict(opt_btr=btr,opt_rsf=rsf,sh2o=[sh]*4,fsno=fsno))
    def run(rows):
        lines=[" ".join(map(str,params[key])) for key in ("smcwlt","smcref","smcmax","psisat","bexp")]
        lines += [f'{params["rsurf_exp"]} {params["rsurf_snow"]}'," ".join(map(str,zsoil)),str(len(rows))]
        lines += [" ".join(map(str,[c["opt_btr"],c["opt_rsf"],*c["sh2o"],c["fsno"]])) for c in rows]
        result=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",
                              capture_output=True,text=True,check=True,timeout=30)
        return np.asarray([[float(value) for value in line.split()] for line in result.stdout.splitlines()],np.float32)
    reference=run(cases)
    assert reference.shape==(96,5) and np.isfinite(reference).all()
    rows=[];owner=[];nominal=[]
    for i,case in enumerate(cases):
        nominal.append(len(rows));rows.append(case.copy());owner.append(i)
        for field in range(5):
            for positive in (False,True):
                variant=dict(case,sh2o=case["sh2o"].copy())
                direction=np.float32(np.inf if positive else -np.inf)
                if field<4:variant["sh2o"][field]=float(np.nextafter(np.float32(variant["sh2o"][field]),direction))
                else:variant["fsno"]=float(np.nextafter(np.float32(variant["fsno"]),direction))
                rows.append(variant);owner.append(i)
    variants=run(rows)
    assert np.isfinite(variants).all() and np.array_equal(variants[nominal],reference)
    lo,hi=reference.copy(),reference.copy()
    for index,row in zip(owner,variants):
        lo[index]=np.minimum(lo[index],row);hi[index]=np.maximum(hi[index],row)
    for i,case in enumerate(cases):case.update(expected=reference[i].tolist(),low=lo[i].tolist(),high=hi[i].tolist())
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stress_sha256=hashlib.sha256(stress.encode()).hexdigest(),resistance_sha256=hashlib.sha256(resistance.encode()).hexdigest(),
        flags=flags,created_utc=datetime.now(timezone.utc).isoformat(),parameters=params,zsoil=zsoil,cases=cases,
        variables_out=["gx1","gx2","gx3","gx4","rsurf"],nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6),
        census=dict(columns=96,ulp_variants=len(rows),constructed=True,
                    regimes="very dry/wilting/field-capacity/saturated/low-liquid frozen-soil proxy; snow fractions 0/.5/.9/1"),
        scope="ENERGY per-layer GX before root-depth weighting; raw RSURF before shared dry/urban overrides; caller uses IST=1"),indent=1)+"\n")


if __name__=="__main__":main()
