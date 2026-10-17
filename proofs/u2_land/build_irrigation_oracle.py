"""Verbatim WRF trigger/method/infiltration oracles with the driver's fractions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from build_snow_oracle import extract


def main():
    source, build, target = map(Path, sys.argv[1:4])
    build.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    names = ("TRIGGER_IRRIGATION", "SPRINKLER_IRRIGATION", "MICRO_IRRIGATION", "FLOOD_IRRIGATION", "IRR_PHILIP_INFIL", "WDFCND2")
    routines = {name: extract(text, name) for name in names}
    isnan = re.search(r"^  logical function isnan_lsm\b.*?^  end function isnan_lsm\s*$", text, re.M | re.S | re.I)[0]
    module = '''module irrigation_reference
implicit none
integer :: OPT_IRR, OPT_IRRM
type noahmp_parameters
integer :: NROOT
real :: IRR_FRAC,IRR_HAR,IRR_LAI,IRR_MAD,FILOSS,SPRIR_RATE,MICIR_RATE,FIRTFAC,IR_RAIN,PLTDAY,HSDAY
real :: SMCWLT(4),SMCREF(4),SMCMAX(4),BEXP(4),DKSAT(4),DWSAT(4)
end type
contains
'''
    module += "\n".join(routines.values()) + "\n" + isnan + "\nend module\n"
    (build / "reference.f90").write_text(module)
    # Copy the caller's method selection too, not a reimplementation in the oracle.
    caller = text[text.index("     SIFAC = SIFRA"):text.index("! trigger irrigation only at soil timestep")]
    driver = '''program oracle
use irrigation_reference
implicit none
type(noahmp_parameters), target :: p
type(noahmp_parameters), pointer :: parameters
integer :: n,i,cs,cm,cf
real :: zsoil(4),sh2o(4),smc(4),sice(4),fveg,julian,irrfra,lai,sifra,mifra,fifra,sifac,mifac,fifac
real :: t,u,v,ea,as,am,af,rs,rm,rf,ev,dt,fsur
parameters => p
read(*,*) p%IRR_FRAC,p%IRR_HAR,p%IRR_LAI,p%IRR_MAD,p%FILOSS,p%SPRIR_RATE,p%MICIR_RATE,p%FIRTFAC,p%IR_RAIN
read(*,*) p%SMCWLT
read(*,*) p%SMCREF
read(*,*) p%SMCMAX
read(*,*) p%BEXP
read(*,*) p%DKSAT
read(*,*) p%DWSAT
read(*,*) zsoil
read(*,*) n
do i=1,n
read(*,*) opt_irr,opt_irrm,p%NROOT,p%PLTDAY,p%HSDAY,sh2o,smc,fveg,julian,irrfra,lai,sifra,mifra,fifra,t,u,v,ea,as,am,af,cs,cm,cf,dt
'''
    driver += caller
    driver += '''call TRIGGER_IRRIGATION(p,4,zsoil,sh2o,fveg,julian,irrfra,lai,sifac,mifac,fifac,cs,cm,cf,as,am,af)
rs=0.;rm=0.;rf=0.;ev=0.
if(as>0.) call SPRINKLER_IRRIGATION(p,4,dt,sh2o,smc,t,u,v,ea,sifac,as,ev,rs)
sice=max(0.,smc-sh2o)
if(af>0.) call FLOOD_IRRIGATION(p,4,dt,sh2o,smc,sice,fifac,af,rf)
if(am>0.) call MICRO_IRRIGATION(p,4,dt,sh2o,smc,sice,mifac,am,rm)
call IRR_PHILIP_INFIL(p,smc,sh2o,sice,dt,4,fsur)
write(*,'(11(ES25.16E3,1X))') real(cs),real(cm),real(cf),as,am,af,rs,rm,rf,ev,fsur
enddo
end program
'''
    (build / "driver.f90").write_text(driver)
    fc = shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    subprocess.run([fc, "-O2", "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off",
                    "reference.f90", "driver.f90", "-o", "oracle.exe"], cwd=build, check=True, timeout=120)
    table = source.parent.parent / "run/MPTABLE.TBL"
    clean = "\n".join(line.split("!", 1)[0] for line in table.read_text().splitlines())
    global_keys = ("IRR_FRAC", "IRR_HAR", "IRR_LAI", "IRR_MAD", "FILOSS", "SPRIR_RATE", "MICIR_RATE", "FIRTFAC", "IR_RAIN")
    params = {key.lower(): float(re.search(rf"\b{key}\s*=\s*([-+\d.eEdD]+)", clean, re.I)[1].lower().replace("d", "e")) for key in global_keys}
    params.update(smcwlt=[.1,.12,.1,.08], smcref=[.3,.32,.3,.28], smcmax=[.45]*4,
                  bexp=[4.5]*4, dksat=[1.e-6]*4, dwsat=[1.e-5]*4)
    zsoil = [-.1,-.4,-1.,-2.]
    cases = []
    for irr in (1,2,3):
        for method in (0,1,2,3):
            for regime in range(8):
                sh = [.14,.16,.14,.12] if regime != 4 else [.29,.31,.29,.27]
                smc = [x + (.1 if regime == 5 else 0.) for x in sh]
                case = dict(opt_irr=irr,opt_irrm=method,nroot=1+regime%4,pltday=100.,hsday=300.,
                            sh2o=sh,smc=smc,fveg=.7,julian=50. if regime==2 else (290. if regime==3 else 150.),
                            irrfra=.5,lai=.05 if regime==1 else 2.,
                            fractions=[0.,0.,0.] if regime==0 else [.3,.4,.3],
                            t2=270. if regime==5 else 295.,u=3.,v=4.,eair=10000. if regime==6 else 1000.,
                            amounts=[1.e-6]*3 if regime==7 else [0.,0.,0.], counts=[1,2,3],dt=18.)
                cases.append(case)
    lines = [" ".join(str(params[key.lower()]) for key in global_keys)]
    lines += [" ".join(map(str,params[key])) for key in ("smcwlt","smcref","smcmax","bexp","dksat","dwsat")]
    lines += [" ".join(map(str,zsoil)), str(len(cases))]
    for c in cases:
        values = [c["opt_irr"],c["opt_irrm"],c["nroot"],c["pltday"],c["hsday"],*c["sh2o"],*c["smc"],
                  c["fveg"],c["julian"],c["irrfra"],c["lai"],*c["fractions"],c["t2"],c["u"],c["v"],c["eair"],
                  *c["amounts"],*c["counts"],c["dt"]]
        lines.append(" ".join(map(str,values)))
    result = subprocess.run([str((build / "oracle.exe").resolve())], input="\n".join(lines)+"\n",
                            text=True,capture_output=True,check=True,timeout=30)
    outputs = [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
    assert len(outputs)==len(cases) and np.isfinite(outputs).all()
    for case, values in zip(cases,outputs): case["expected"] = values
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        routine_sha256={name:hashlib.sha256(body.encode()).hexdigest() for name,body in routines.items()},
        caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        compiler=subprocess.check_output([fc,"--version"],text=True).splitlines()[0],
        flags="-O2 -ffp-contract=off; literal pristine REAL4 routines + method-fraction caller",
        created_utc=datetime.now(timezone.utc).isoformat(),parameters=params,zsoil=zsoil,cases=cases,
        variables_out=["irnumsi","irnummi","irnumfi","irwatsi","irwatmi","irwatfi","sprinkler","micro","flood","evaporated","fsur"],
        tolerance=dict(atol=1.e-10,rtol=3.e-6),
        census=dict(columns=len(cases),constructed=True,options=[1,2,3],methods=[0,1,2,3],
                    regimes="dry/wet/frozen soil; 1-4 roots; outside crop season; low LAI; saturated humidity fallback; amount exhaustion; unknown method")),indent=1)+"\n")


if __name__ == "__main__": main()
