"""Rebuild the original Ferrier reference before testing the WRF caller clock.

Run in a detached checkout, with --output in the lane's artifact directory.
The pristine physics sources are never edited.  Only the project-authored
column driver is adapted, with every adaptation recorded in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120, check=True)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf", type=Path, required=True)
    parser.add_argument("--fc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument("--stress-only", action="store_true",
                        help="active autoconversion, large-rain and high-rime ice columns")
    parser.add_argument("--init-kind", choices=("zero", "nan", "none"), default="zero")
    parser.add_argument("--physics-fma", action="store_true",
                        help="original physics compiled with native FMA; input driver stays generic")
    parser.add_argument("--dump-constants", action="store_true",
                        help="append a read-only getter; original routine bodies remain verbatim")
    parser.add_argument("--ulp-ensemble", action="store_true",
                        help="independent original-WRF +/-1 input-ULP sensitivity ensemble")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = out / "build"
    build.mkdir()
    original = ROOT / "proofs/v018/mp_oracles/ferrier_hires/oracle"
    sources = [args.wrf / "phys/module_mp_etanew.F",
               args.wrf / "phys/module_mp_fer_hires.F",
               original / "ferrier_wrf_stubs.f90"]
    receipt = {"source_commit": run(["git", "rev-parse", "HEAD"], ROOT).strip(),
               "sources": {str(p): sha(p) for p in sources},
               "clock": "GSMDT is minutes; DT is seconds",
               "device": "cpu", "cache": "NONE", "arms": {}}
    try:
        for path in sources:
            shutil.copyfile(path, build / path.name)
        constants = "CP RD RV EPS EPS1 RCP RCPRV XLV XLF XLS XLV1 XLF1 XLS1 XLV2 XLS2 XLV3 XLS3 C1 ABFR CBFR CIACW CIACR CRACW ARAUT BRAUT BETA6 QAUT0 ARcw RFmax RFmx1 RH_NgC RH_NgT RQhail AVhail BVhail CN0r0 RQR_DRmin RQR_DRmax".split()
        if args.dump_constants:
            path = build / "module_mp_fer_hires.F"
            original_physics = path.read_text()
            getter = "\nsubroutine dump_ferrier_constants\ninteger :: u\nopen(newunit=u,file='constants.txt',status='replace')\n"
            for name in constants:
                getter += f"write(u,'(A,1X,ES24.16)') '{name}',{name}\n"
            getter += "close(u)\nend subroutine\n"
            marker = "      END MODULE module_mp_fer_hires"
            assert original_physics.count(marker) == 1
            path.write_text(original_physics.replace(marker, getter + marker))
            receipt["read_only_getter_append"] = {"physics_body_verbatim": True,
                                                  "extended_file_sha256": sha(path)}
        table = args.wrf / "run/ETAMPNEW_DATA.expanded_rain"
        shutil.copyfile(table, build / table.name)
        receipt["table_sha256"] = sha(table)
        flags = ["-O2", "-cpp", "-ffree-form", "-ffree-line-length-none",
                 "-ffpe-summary=none", "-fallow-argument-mismatch",
                 "-fconvert=big-endian", "-frecord-marker=4",
                 "-DIWORDSIZE=4", "-DRWORDSIZE=4"]
        if args.init_kind != "none":
            flags += [f"-finit-real={args.init_kind}", "-finit-integer=0", "-finit-logical=false"]
        for name in ["ferrier_wrf_stubs.f90", "module_mp_etanew.F", "module_mp_fer_hires.F"]:
            physics_flags = ["-march=native", "-ffp-contract=fast"] if args.physics_fma and name.startswith("module_mp_") else []
            run([args.fc, *flags, *physics_flags, "-c", name], build)
        receipt["physics_flags"] = flags + (["-march=native", "-ffp-contract=fast"] if args.physics_fma else [])
        # Public, unmodified WRF functions used by GPVS; no NumPy/libm
        # reconstruction of the single-precision saturation tables.
        (build / "tables.f90").write_text("""program tables
use module_mp_fer_hires, only: FPVSX, FPVSX0
implicit none
integer :: j
real :: t, xinc
xinc=(330.0-180.0)/(7501-1)
do j=1,7501
  t=180.0+(j-1)*xinc
  write(*,'(2(ES24.16,1X))') FPVSX(t),FPVSX0(t)
enddo
end program
""")
        run([args.fc, *flags, "-c", "tables.f90"], build)
        run([args.fc, "-o", "tables", "ferrier_wrf_stubs.o",
             "module_mp_etanew.o", "module_mp_fer_hires.o", "tables.o"], build)
        (out / "saturation_tables.txt").write_text(run(["./tables"], build))
        receipt["saturation_driver_sha256"] = sha(build / "tables.f90")
        receipt["saturation_tables_sha256"] = sha(out / "saturation_tables.txt")
        physics = (args.wrf / "phys/module_mp_fer_hires.F").read_text()
        constants_start = physics.index("      REAL, PRIVATE,PARAMETER ::", physics.index("TBPVS,TBPVS0"))
        constants_end = physics.index("!-- Debug 20120111", constants_start)
        helpers_start = physics.index("      REAL FUNCTION CONDENSE")
        helpers_end = physics.index("      INTEGER FUNCTION GET_INDEXR", helpers_start)
        # Verbatim internal bodies and PARAMETER declarations. The shim only
        # makes the original nested functions accessible to the test driver.
        shim = "module saturation_oracle\nuse module_mp_fer_hires, only: FPVS,FPVS0,GPVS\nimplicit none\n"
        shim += physics[constants_start:constants_end]
        shim += "logical :: warn2=.false.,warn3=.false.\ninteger,parameter :: MAX_ITERATIONS=10\nreal,parameter :: XMIN=180.0\ncontains\n"
        shim += physics[helpers_start:helpers_end] + "end module\n"
        (build / "saturation_oracle.f90").write_text(shim)
        (build / "saturation_driver.f90").write_text("""program test_saturation
use saturation_oracle
use module_mp_fer_hires, only: FPVS,FPVS0,GPVS
implicit none
real :: t,v,p,cloud,rh,ws,es,ct,dt,dv,dep,cond
real,parameter :: ts(8)=[285.,260.,230.,180.,330.,340.,170.,273.16]
real,parameter :: ratios(4)=[1.1,0.8,1.0,0.1]
integer :: i,j,k
call GPVS
do i=1,8
do j=1,4
do k=1,2
  t=ts(i)
  p=90000.
  if(i==3.or.i==4) p=50000.
  cloud=0.
  if(k==2) cloud=0.001
  rh=0.975
  es=min(1000.*FPVS0(t),0.99*p)
  ws=rh*(287.04/461.5)*es/(p-es)
  v=ratios(j)*ws
  cond=CONDENSE(p,cloud,t,v,rh,1,1,1)
  dt=t
  dv=v
  dep=DEPOSIT(p,dt,dv,rh,1,1,1)
  write(*,'(9(ES24.16,1X))') p,cloud,t,v,rh,cond,dep,dt,dv
enddo
enddo
enddo
end program
""")
        run([args.fc, *flags, "-c", "saturation_oracle.f90"], build)
        run([args.fc, *flags, "-c", "saturation_driver.f90"], build)
        run([args.fc, "-o", "saturation", "ferrier_wrf_stubs.o",
             "module_mp_etanew.o", "module_mp_fer_hires.o", "saturation_oracle.o", "saturation_driver.o"], build)
        (out / "saturation_reference.txt").write_text(run(["./saturation"], build))
        receipt["saturation_shim_sha256"] = sha(build / "saturation_oracle.f90")
        receipt["saturation_reference_sha256"] = sha(out / "saturation_reference.txt")
        driver = (original / "ferrier_oracle_driver.f90").read_text()
        arms = [("legacy_mp5", "dt", 60, 1, 1), ("caller_mp5", "0.0", 60, 1, 1),
                ("caller_mp15", "0.0", 60, 1, 1), ("rimed_mp15", "0.0", 60, 1, 1)]
        if args.holdout_only:
            arms = [("holdout_mp5", "0.0", 18, 2, 3),
                    ("holdout_mp15", "0.0", 54, 2, 2),
                    ("holdout_rimed_mp15", "0.0", 1, 1, 4)]
        if args.stress_only:
            arms = [("stress_mp5", "0.0", 18, 1, 3),
                    ("stress_mp15", "0.0", 54, 2, 2),
                    ("stress_rimed_mp15", "0.0", 1, 1, 4)]
        for arm, clock, dt_seconds, gid, steps in arms:
            arm_dir = out / arm
            arm_dir.mkdir()
            adapted = driver.replace("gsmdt = dt", f"gsmdt = {clock}")
            assert adapted.count(f"gsmdt = {clock}") == 1
            scheme = "mp5"
            if "mp15" in arm:
                scheme = "mp15"
                adapted = adapted.replace("fer_hires_init, FER_HIRES", "fer_hires_init, FER_HIRES, FER_HIRES_ADVECT")
                adapted = adapted.replace("trim(scheme) == 'mp5'", "trim(scheme) == 'mp15'")
                adapted = adapted.replace("scheme == 'mp5'", "scheme == 'mp15'")
                adapted = adapted.replace("real :: dt, dx, dy, gsmdt", "real :: dt, dx, dy, gsmdt\n  real :: qrimef(1,kx,1), qrimef_in(1,kx,1)")
                adapted = adapted.replace("th_in = th_phy", "qrimef = " + ("4.0" if "rimed" in arm else "1.0") + " * qi\n  qrimef_in = qrimef\n  th_in = th_phy")
                start = adapted.index("    call FER_HIRES(1,")
                end = adapted.index("\n  else", start)
                adapted = adapted[:start] + """    call FER_HIRES_ADVECT(1, dt, dx, dy, 1, rainnc, rainncv, &
      dz8w, rho_phy, p_phy, pi_phy, th_phy, qv, lowlyr, sr, qc, qr, qi, qrimef, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte)""" + adapted[end:]
                adapted = adapted.replace("call dump_field('DZ', dz)", "call dump_field('QRIMEF_IN', qrimef_in)\n    call dump_field('QRIMEF_OUT', qrimef)\n    call dump_field('DZ', dz)")
            if args.stress_only:
                adapted = adapted.replace("integer :: cid", "real :: frime_in(1,kx,1)\n  integer :: cid", 1)
                stress = """  qc(1,3:10,1)=0.003
  qr(1,3:10,1)=0.03
  qi(1,8:28,1)=0.005
  qt=qc+qr+qi
  f_rimef_phy=20.0
"""
                if scheme == "mp15":
                    stress += "  qrimef=" + ("20.0" if "rimed" in arm else "1.0") + "*qi\n  qrimef_in=qrimef\n"
                stress += "  frime_in=f_rimef_phy\n"
                adapted = adapted.replace("  th_in = th_phy", stress + "  th_in = th_phy", 1)
                adapted = adapted.replace("call dump_field('DZ', dz)", "call dump_field('F_RIMEF_PHY_IN', frime_in)\n    call dump_field('DZ', dz)")
            if args.holdout_only or args.stress_only:
                adapted = adapted.replace("dt = 60.0", f"dt = {dt_seconds}.0")
                adapted = adapted.replace("integer :: cid", "integer :: cid, step")
                adapted = adapted.replace("(1, dt, dx, dy, 1, rainnc", f"(step, dt, dx, dy, {gid}, rainnc")
                call_start = adapted.index("  if (trim(scheme)", adapted.index("  qs_in = qs"))
                call_end = adapted.index("\n  call dump_savepoint", call_start)
                adapted = adapted[:call_start] + f"  do step=1,{steps}\n" + adapted[call_start:call_end] + "\n  enddo\n" + adapted[call_end:]
                adapted = adapted.replace("call dump_field('DZ', dz)", "write(*,'(A,ES24.16)') 'SCALAR RAINNC ', rainnc(1,1)\n    call dump_field('DZ', dz)")
            if args.dump_constants:
                adapted = adapted.replace("fer_hires_init, FER_HIRES", "dump_ferrier_constants, fer_hires_init, FER_HIRES", 1)
                adapted = adapted.replace("  call dump_savepoint", "  call dump_ferrier_constants\n  call dump_savepoint", 1)
            perturb_fields = "th_phy qv qc qr qi p_phy pi_phy rho_phy dz8w".split()
            perturb_fields += ["qrimef" if scheme == "mp15" else "f_rimef_phy"]
            if args.ulp_ensemble:
                adapted = adapted.replace("integer :: cid", "character(len=32) :: ulp_name\n  real :: ulp_sign\n  integer :: cid", 1)
                block = "  call get_command_argument(3,ulp_name)\n  call get_command_argument(4,arg)\n  ulp_sign=1.0\n  if(len_trim(arg)>0) read(arg,*) ulp_sign\n  select case(trim(ulp_name))\n"
                for field in perturb_fields:
                    block += f"  case('{field}')\n    where({field}/=0.) {field}=nearest({field},ulp_sign)\n"
                block += "  end select\n"
                if scheme == "mp15":
                    block += "  qrimef_in=qrimef\n"
                adapted = adapted.replace("  th_in = th_phy", block + "  th_in = th_phy", 1)
            (build / "driver.f90").write_text(adapted)
            run([args.fc, *flags, "-c", "driver.f90"], build)
            run([args.fc, "-o", "oracle", "ferrier_wrf_stubs.o",
                 "module_mp_etanew.o", "module_mp_fer_hires.o", "driver.o"], build)
            receipt["arms"][arm] = {"driver_sha256": sha(build / "driver.f90"),
                                     "gsmdt_minutes": 60 if clock == "dt" else 0,
                                     "dt_seconds": dt_seconds, "gid": gid, "steps": steps, "cases": []}
            for case in range(1, 5):
                dump = arm_dir / f"case_{case}.dump"
                dump.write_text(run(["./oracle", scheme, str(case)], build))
                target = dump.with_suffix(".json")
                run(["python", str(original / "ferrier_dump_to_json.py"),
                     str(dump), str(target)], build)
                if args.dump_constants:
                    shutil.copyfile(build / "constants.txt", arm_dir / f"constants_{case}.txt")
                receipt["arms"][arm]["cases"].append({"path": str(target), "sha256": sha(target)})
            if args.ulp_ensemble:
                receipt["arms"][arm]["ulp_variants"] = []
                for field in perturb_fields:
                    for sign in (-1, 1):
                        variant = arm_dir / (field + ("_minus" if sign < 0 else "_plus"))
                        variant.mkdir()
                        item = {"field": field, "sign": sign, "cases": []}
                        for case in range(1, 5):
                            dump = variant / f"case_{case}.dump"
                            dump.write_text(run(["./oracle", scheme, str(case), field, str(sign)], build))
                            target = dump.with_suffix(".json")
                            run(["python", str(original / "ferrier_dump_to_json.py"), str(dump), str(target)], build)
                            item["cases"].append({"path": str(target), "sha256": sha(target)})
                        receipt["arms"][arm]["ulp_variants"].append(item)
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Exact owned path only; no model/reference input is deleted.
        shutil.rmtree(build)


if __name__ == "__main__":
    main()
