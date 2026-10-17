"""Reproduce the unassigned GECROS module parameters without editing WRF.

This is a reference viability diagnostic, NOT a candidate parity fixture. The
zero arm uses the pristine Noah-MP caller's initialization statements and exact
GECROS bodies. Nonzero arms write module parameters only in the wrapper, to
measure sensitivity; they are explicitly NOT pristine WRF references.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


GLOBALS = ("ygo", "cfo", "lncmin", "clvi", "crti", "nlvi", "nrti", "hti", "rdi")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(text: str, name: str) -> str:
    match = re.search(
        rf"^\s*SUBROUTINE {name}\b.*?^\s*END SUBROUTINE {name}\s*$",
        text, re.I | re.M | re.S,
    )
    if match is None:
        raise ValueError(name)
    return match[0]


def finite_mask(bits: list[int]) -> list[bool]:
    return [(x & 0x7F800000) != 0x7F800000 for x in bits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    crop = source / "phys/module_sf_gecros.F"
    caller = source / "phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F"
    crop_text, caller_text = crop.read_text(), caller.read_text()
    caller_init = caller_text[caller_text.index("               hti    = 0.01"):
                              caller_text.index("               call gecros_init")]
    init_body = extract(caller_text, "gecros_init")
    driver = r"""program diagnostic
use module_sf_gecros
implicit none
integer :: mode, regime, n, j
real :: s(60), initial(60), outputs(6), params(9), temp, rsd
read(*,*) mode, regime
call caller_initialize(initial)
s=initial
! Dormancy flags are set to the active regime at the VEGE_FLUX call boundary.
! All forcing/state regimes are constructed, not captured forecast columns.
s(40)=1.;s(41)=1.;s(42)=-1.
if(regime==1) then
 s(1)=0.6;s(3)=30.;s(4)=80.;s(8)=30.;s(11)=.4;s(13)=3.;s(15)=3.
 s(24)=.6;s(25)=3.;s(32)=.7;s(33)=60.;s(49)=3.;s(50)=3.
endif
if(regime==2) then
 s(1)=1.5;s(3)=40.;s(4)=50.;s(6)=60.;s(7)=100.;s(8)=30.
 s(11)=.4;s(12)=.6;s(13)=2.;s(14)=2.;s(15)=3.;s(18)=5.;s(19)=2.
 s(20)=1.;s(21)=1.;s(24)=.6;s(25)=2.;s(32)=1.;s(33)=100.
 s(49)=2.;s(50)=3.
endif
if(mode==1) then
 ! Alternative initialization for sensitivity only, NOT WRF truth.
 ygo=1.;cfo=1.;lncmin=1.;clvi=1.;crti=1.;nlvi=1.;nrti=1.;hti=1.;rdi=1.
endif
if(mode==2) call diagnostic_parameters(initial)
params=[ygo,cfo,lncmin,clvi,crti,nlvi,nrti,hti,rdi]
temp=295.;rsd=600.
if(regime==3) then
 temp=270.;rsd=0.
endif
call gecros(150.5,18.,1,50.,100.,200.,0.,0.,3.,temp,1200.,rsd,300.,0., &
            100.,100.,.1,.04,s,outputs(1),outputs(2),outputs(3),outputs(4),outputs(5),outputs(6))
! Output bit patterns: preserve NaN/Inf evidence without nonstandard JSON numbers.
write(*,'(9(Z8.8,1X))') transfer(params,[(0,j=1,9)])
write(*,'(60(Z8.8,1X))') transfer(initial,[(0,j=1,60)])
write(*,'(60(Z8.8,1X))') transfer(s,[(0,j=1,60)])
write(*,'(6(Z8.8,1X))') transfer(outputs,[(0,j=1,6)])
end program

subroutine caller_initialize(state_gecros)
use module_sf_gecros, only: seednc,sla0,slnmin,ffat,flig,foac,fmin,npl,seedw,eg,fcrsh,lnci,cfv
implicit none
real :: hti,rdi,fpro,lncmin,fcar,cfo,clvi,crti,ygo,nlvi,laii,nrti,slnbi
real :: state_gecros(60)
! The WRF state array's previously untouched slots are compiler-defined zero.
state_gecros=0.
""" + caller_init + r"""
call gecros_init(28.,hti,rdi,clvi,crti,nlvi,laii,nrti,slnbi,state_gecros)
end subroutine

subroutine diagnostic_parameters(initial)
use module_sf_gecros
implicit none
real :: initial(60), fpro, fcar
! Deliberately assign the module parameters for a diagnostic sensitivity control.
! The production WRF caller does NOT do this; this arm is never an oracle.
fpro=6.25*seednc
fcar=1.-fpro-ffat-flig-foac-fmin
cfo=.444*fcar+.531*fpro+.774*ffat+.667*flig+.368*foac
ygo=cfo/(1.275*fcar+1.887*fpro+3.189*ffat+2.231*flig+.954*foac)*30./12.
lncmin=sla0*slnmin
clvi=initial(53);crti=initial(54);nrti=initial(55);nlvi=initial(56)
hti=initial(51);rdi=initial(52)
end subroutine
""" + init_body + "\n"
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    (build / "reference.f90").write_bytes(crop.read_bytes())
    assert sha((build / "reference.f90").read_bytes()) == sha(crop.read_bytes())
    (build / "driver.f90").write_text(driver)
    fc = shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    rows = []
    commands = []
    for optimization in ("-O0", "-O2"):
        flags = [optimization, "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off",
                 "-finit-real=zero", "-finit-integer=0", "-finit-logical=false", "-fcheck=all"]
        command = [fc, *flags, "reference.f90", "driver.f90", "-o", "probe.exe"]
        subprocess.run(command, cwd=build, check=True, timeout=120, capture_output=True, text=True)
        commands.append(command)
        for mode in range(3):
            for regime in range(4):
                run = subprocess.run([str(build / "probe.exe")], cwd=build, input=f"{mode} {regime}\n",
                                     check=True, capture_output=True, text=True, timeout=10)
                params, init, state, flux = [[int(x, 16) for x in line.split()]
                                            for line in run.stdout.splitlines()]
                assert len(params) == 9 and len(init) == len(state) == 60 and len(flux) == 6
                rows.append(dict(optimization=optimization, mode=mode, regime=regime,
                    module_parameters_bits=params, initial_state_bits=init, final_state_bits=state,
                    flux_bits=flux, nonfinite_state_slots=[i+1 for i,v in enumerate(finite_mask(state)) if not v],
                    nonfinite_flux_names=[n for n,v in zip(("ATRJC","ATRJS","FSR","FRSU","ARSWSU","ARSWSH"),finite_mask(flux)) if not v],
                    stderr=run.stderr.strip()))
    zero = [r for r in rows if r["mode"] == 0]
    intended = [r for r in rows if r["mode"] == 2]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="pristine-reference viability; constructed columns; no port and no forecast qualification",
        source_files={str(crop):sha(crop.read_bytes()),str(caller):sha(caller.read_bytes())},
        source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],text=True).strip(),
        extracted_initializer_sha256=sha(init_body.encode()),caller_initialization_sha256=sha(caller_init.encode()),
        driver_sha256=sha(driver.encode()),compiler=subprocess.check_output([fc,"--version"],text=True).splitlines()[0],
        compile_commands=commands, traps="off, matching production; fcheck=all",cache="NONE; no JAX/GPU",
        parameters=list(GLOBALS), modes={0:"pristine caller, compiler zero initialization",1:"wrapper-only all-one sensitivity",2:"wrapper-only locally-derived parameters, NOT pristine"},
        regimes={0:"initialized active seedling",1:"vegetative",2:"grain fill",3:"cold/night seedling"},
        all_pristine_globals_zero=all(not any(r["module_parameters_bits"]) for r in zero),
        pristine_nonfinite_columns=sum(bool(r["nonfinite_state_slots"] or r["nonfinite_flux_names"]) for r in zero),
        diagnostic_derived_nonfinite_columns=sum(bool(r["nonfinite_state_slots"] or r["nonfinite_flux_names"]) for r in intended),
        rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("all_pristine_globals_zero","pristine_nonfinite_columns","diagnostic_derived_nonfinite_columns")}))


if __name__ == "__main__":
    main()
