"""Extract the original Thompson mp38 entry/fall operators verbatim.

This small oracle deliberately does not replace the full mp38 caller oracle.
It lets density, number-balance and Brown--Lawler fall speeds be qualified
without recomputing the 9-plane collision quadrature at each experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def section(source: str, start: str, end: str) -> str:
    pos = source.index(start)
    return source[pos:source.index(end, pos) + len(end)]


def build(source: Path, out: Path, compiler: str, fma: bool = False) -> None:
    text = source.read_text()
    fragments = {
        "rho": section(text, "      REAL, DIMENSION(NRHG), PARAMETER, PRIVATE:: &", "rho_g = (/50., 100., 200., 300., 400., 500., 600., 700., 800./)"),
        "rho_not": section(text, "      REAL, PARAMETER, PRIVATE:: rho_not =", "rho_not = 101325.0/(287.05*298.0)"),
        "am": section(text, "      REAL, DIMENSION(NRHG), PARAMETER,PRIVATE:: am_g", "PI*rho_g(7)/6.0, PI*rho_g(8)/6.0, PI*rho_g(9)/6.0 /)"),
        "velocity_constants": section(text, "      REAL, DIMENSION(NRHG), PRIVATE:: & ! Computed from A.", "      REAL, PARAMETER, PRIVATE:: b_coeff = 0.54698726"),
        "init": section(text, "      cge(1,:) = bm_g + 1.", "      ogg3 = 1./cgg(3,1)"),
        "entry": section(text, "         if (qg1d(k) .gt. R1) then", "         if (.not. is_hail_aware) idx_bg(k) = idx_bg1"),
        "slope": section(text, "      do k = kte, kts, -1\n         lamg = (am_g(idx_bg(k))*cgg(3,1)*ogg2*ng(k)/rg(k))**obmg", "      enddo"),
        "fall": section(text, "       if (ANY(L_qg .eqv. .true.)) then\n       nstep = 0", "       if (nstep .gt. 0) onstep(4) = 1./REAL(nstep)\n       endif"),
        "sediment": section(text, "      if (ANY(L_qg .eqv. .true.)) then\n      nstep = NINT(1./onstep(4))", "      enddo\n      endif\n\n!+---+-----------------------------------------------------------------+\n!.. Instantly melt"),
        "gamma": section(text, "      REAL FUNCTION GAMMLN(XX)", "      END FUNCTION GAMMLN"),
        "wgamma": section(text, "      REAL FUNCTION WGAMMA(y)", "      END FUNCTION WGAMMA"),
    }
    # Original declarations are public only in this extracted oracle module;
    # every executable physics fragment below remains verbatim.
    declarations = "\n".join(fragments[n] for n in ("rho", "rho_not", "am", "velocity_constants")).replace(", PRIVATE", "").replace(",PRIVATE", "")
    program = f"""module original_hail_operators
 implicit none
 integer, parameter :: NRHG=9, idx_bg1=5, kts=1, kte=64
 real, parameter :: PI=3.1415926536, bm_g=3.,mu_g=0.,R1=1.E-12,R2=1.E-6,D0r=50.E-6
 real, parameter :: av_g_old=442.,bv_g_old=.89
 logical, parameter :: is_hail_aware=.true.
 {declarations}
 real :: cge(12,NRHG),cgg(12,NRHG),oamg(NRHG),ocmg(NRHG),obmg,oge1,ogg1,ogg2,ogg3
 contains
 {fragments['gamma']}
 {fragments['wgamma']}
 subroutine run
 implicit none
 real :: qg1d(kte),ng1d(kte),qb1d(kte),rho(kte),rg(kte),ng(kte),rb(kte),mvd_g(kte)
 real :: rhof(kte),visco(kte),dzq(kte),vtgk(kte+1),vtngk(kte+1),onstep(5)
 double precision :: lamg,ilamg(kte),N0_g(kte)
 real :: xrho_g,afall,bfall,vtg,delta_tp,DT,odzq,orho,pptgraul
 real :: sed_g(kte),sed_n(kte),sed_b(kte),qgten(kte),ngten(kte),qbten(kte)
 logical :: no_micro,L_qg(kte)
 integer :: n,m,k,idx_bg(kte),ksed1(5),nstep,field,sign
 character(32) :: arg
 {fragments['init']}
 DT=90.;no_micro=.true.;ksed1=0;onstep=1.;bfall=3*b_coeff-1
 do k=1,kte
   rho(k)=.3+real(k)*.02
   qg1d(k)=1.E-6*real(k)
   ng1d(k)=real(k)*100.
   qb1d(k)=qg1d(k)/(25.+real(mod(k-1,9))*100.)
   dzq(k)=50.+real(k)*10.
   rhof(k)=sqrt(RHO_NOT/rho(k))
   visco(k)=1.7E-5
 enddo
 qg1d(1)=0.;ng1d(1)=0.;qb1d(1)=0.
 qg1d(2)=1.E-13;ng1d(2)=0.;qb1d(2)=0.
 ng1d(3)=0.;ng1d(4)=1.E12
 qg1d(5)=.1;ng1d(5)=1.E-9
 qb1d(6)=-1.;qb1d(7)=1.
 field=0;sign=1
 if(command_argument_count()>0)then
 call get_command_argument(1,arg);read(arg,*)field
 call get_command_argument(2,arg);read(arg,*)sign
 endif
 select case(field)
 case(1)
 qg1d=nearest(qg1d,real(sign))
 case(2)
 ng1d=nearest(ng1d,real(sign))
 case(3)
 qb1d=nearest(qb1d,real(sign))
 case(4)
 rho=nearest(rho,real(sign))
 rhof=sqrt(RHO_NOT/rho)
 case(5)
 dzq=nearest(dzq,real(sign))
 case(6)
 visco=nearest(visco,real(sign))
 end select
 call dump('QG_INPUT',qg1d);call dump('NG_INPUT',ng1d);call dump('QB_INPUT',qb1d)
 do k=kts,kte
 {fragments['entry']}
 enddo
 {fragments['slope']}
 vtgk=0.;vtngk=0.
 {fragments['fall']}
 call dump('QG',qg1d);call dump('NG_IN',ng1d);call dump('QB',qb1d)
 call dump('RHO',rho);call dump('DZ',dzq);call dump('VISCO',visco)
 call dump('RG',rg);call dump('NG',ng);call dump('RB',rb)
 call dump('IDX',real(idx_bg));call dump('ILAMG',real(ilamg))
 call dump('VTG',vtgk(:kte));call dump('VTNG',vtngk(:kte))
 call dump('RHO_G',rho_g);call dump('AM_G',am_g)
 call dump('CGE',reshape(cge,[108]));call dump('CGG',reshape(cgg,[108]))
 call dump('NSTEP',[real(nstep)]);call dump('ONSTEP',[onstep(4)])
 qgten=0.;ngten=0.;qbten=0.;pptgraul=0.
 {fragments['sediment']}
 call dump('QGTEN',qgten);call dump('NGTEN',ngten);call dump('QBTEN',qbten)
 call dump('PRECIP',[pptgraul])
 end subroutine
 subroutine dump(name,values)
 character(*)::name
 real :: values(:)
 integer i
 write(*,'(A)',advance='no') trim(name)//'='
 do i=1,size(values)
 write(*,'(ES25.17,1X)',advance='no')values(i)
 enddo
 write(*,*)
 end subroutine
 end module
 program hail_oracle
 use original_hail_operators
 call run
 end program
"""
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "hail_operators.f90"
    generated.write_text(program)
    binary = out / "hail_operators"
    flags = (["-O2", "-march=native", "-ffp-contract=fast"] if fma else ["-O0", "-ffp-contract=off"]) + ["-ffree-line-length-none", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    subprocess.run([compiler, *flags, str(generated), "-o", str(binary)], check=True, cwd=out)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, cwd=out)
    values = {line.split("=", 1)[0].strip(): [float(v) for v in line.split("=", 1)[1].split()] for line in result.stdout.splitlines() if "=" in line}
    receipt = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fragments_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in fragments.items()}, "flags": flags, "scope": "Original entry/density/slope/Brown-Lawler fall operators; not full mp38 caller fidelity", "values": values}
    (out / "hail_operators.json").write_text(json.dumps(receipt, indent=2) + "\n")
    ensemble = {}
    for field in range(1, 7):
        for direction in (-1, 1):
            sample = subprocess.run([str(binary), str(field), str(direction)], check=True, capture_output=True, text=True, cwd=out)
            ensemble[f"{field}:{direction}"] = {line.split("=", 1)[0].strip(): [float(v) for v in line.split("=", 1)[1].split()] for line in sample.stdout.splitlines() if "=" in line}
    (out / "hail_operators_ulp.json").write_text(json.dumps({"input_fields": ["qg", "Ng", "qvolg", "rho", "dz", "visco"], "perturbation": "Original Fortran NEAREST independently +/-1ULP of each of six input vectors", "samples": ensemble, "base": receipt}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    parser.add_argument("--fma", action="store_true")
    args = parser.parse_args()
    build(args.source, args.out, args.compiler, args.fma)
