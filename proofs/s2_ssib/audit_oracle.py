#!/usr/bin/env python3
"""Rebuild the staged SSiB caller against byte-identical pristine WRF.

This is reference qualification, not a JAX parity test. Native REAL, a rebuilt
legacy fp64 control, and poisoned-local diagnostic arms share the same caller.
Run only on CPU, under a timeout and the lane's CPU/QUIET policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time


REPO = Path(__file__).resolve().parents[2]
STUBS = """subroutine wrf_message(msg)
character(len=*), intent(in) :: msg
write(*,'(A)') trim(msg)
end subroutine
subroutine wrf_error_fatal(msg)
character(len=*), intent(in) :: msg
write(0,'(A)') trim(msg)
stop 1
end subroutine
subroutine wrf_error_fatal3(fname, line, msg)
character(len=*), intent(in) :: fname, msg
integer, intent(in) :: line
write(0,'(A)') trim(msg)
stop 1
end subroutine
subroutine wrf_debug(level, msg)
integer, intent(in) :: level
character(len=*), intent(in) :: msg
end subroutine
"""
ROW = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(\S+)\s*$")
MISSING_E = re.compile(r"^([+-]?\d+\.\d+)([+-]\d{3})$")
SNOW_LEAVES = ("dzo", "wo", "tssn", "tssno", "bwo", "bto", "cto",
               "fio", "flo", "bio", "blo", "ho")


def snow_caller(original: str, steps: int, reverse: bool = False) -> str:
    """Extend only the caller; propagate EVERY WRF snow INOUT between steps."""
    code = original.replace("INTEGER :: case_id, i", "INTEGER :: case_id, i, step")
    anchor = "  !-- Per-column SSIB driver"
    before, after = code.split(anchor, 1)
    before += """  ! Cold snow: five positive snow-water equivalents, kg/m2.
  ! 1/10 kg/m2 stay single-layer; 20/40/100 cross the 0.07m threshold.
  SNOA_A = [1._WP, 10._WP, 20._WP, 40._WP, 100._WP]
  SNOA_0 = SNOA_A
  TM_A=268._WP; TC_A=268._WP; TGS_A=267._WP; TD_A=270._WP; TA_A=268._WP
  QM_A=0.002_WP; RLWDOWN_A=250._WP; SWDOWN1_A=100._WP; SUNANGLE_A=0.2_WP
  PPL_A=0._WP; PPC_A=0._WP
"""
    before += f"  DO step=1,{steps}\n  ITIME_A=step\n  WRITE(*,'(A,I0)') 'STEP=',step\n"
    # The old single-step oracle discarded all 48 snow output scalars. Retain
    # them in the caller, exactly as surface_driver's grid arguments do.
    store = "     ! capture updated 4-comp albedo carry"
    updates = "     TKAIR_A(i)=io_tkair\n"
    for layer in range(1, 5):
        for leaf in SNOW_LEAVES:
            updates += f"     {leaf}{layer}_A(i)={leaf}{layer}\n"
    after = after.replace(store, updates + store)
    dumps = ""
    for layer in range(1, 5):
        for leaf in SNOW_LEAVES:
            dumps += f"     CALL dump('{leaf.upper()}{layer}',i,{leaf}{layer}_A(i))\n"
    after = after.replace("     CALL dump('SNOWDEPTH',i, SNOWDEPTH_A(i))",
                          "     CALL dump('SNOWDEPTH',i, SNOWDEPTH_A(i))\n" + dumps)
    after = after.replace("\nCONTAINS\n", "\n  END DO ! trajectory steps\n\nCONTAINS\n")
    if reverse:
        after = after.replace("  DO i = 1, N", "  DO i = N, 1, -1", 1)
    return before + anchor + after


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(raw: str) -> dict[str, float]:
    result = {}
    for line in raw.splitlines():
        match = ROW.match(line)
        if match:
            key = f"{match[1]}[{match[2]}]"
            if key in result:
                raise ValueError(f"duplicate field {key}")
            token = MISSING_E.sub(r"\1E\2", match[3]).replace("D", "E")
            result[key] = float(token)
    if not result:
        raise ValueError("no oracle fields emitted")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrf", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    parser.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    parser.add_argument("--mode", choices=("fp32", "fp64", "poison", "zero", "one"), required=True)
    parser.add_argument("--snow", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--surface-caller", action="store_true")
    parser.add_argument("--fma", action="store_true")
    parser.add_argument("--snow-profile", choices=("cold","melt","snowfall","rainonsnow"),default="cold")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.output / "build"
    build.mkdir()
    source = args.wrf / "phys/module_sf_ssib.F"
    driver = REPO / "proofs/v017/oracle/ssib/ssib_oracle_driver.f90"
    shutil.copyfile(source, build / source.name)
    shutil.copyfile(driver, build / driver.name)
    if args.snow:
        code=snow_caller(driver.read_text(), args.steps, args.reverse)
        profiles={
            "melt": ("TM_A=278._WP; TC_A=273.2_WP; TGS_A=273.1_WP; TD_A=274._WP; TA_A=273.2_WP",
                     "QM_A=0.006_WP; RLWDOWN_A=350._WP; SWDOWN1_A=800._WP; SUNANGLE_A=0.8_WP", "PPL_A=0._WP; PPC_A=0._WP"),
            "snowfall": ("TM_A=263._WP; TC_A=264._WP; TGS_A=263._WP; TD_A=268._WP; TA_A=263._WP",
                         "QM_A=0.001_WP; RLWDOWN_A=240._WP; SWDOWN1_A=100._WP; SUNANGLE_A=0.2_WP", "PPL_A=0.5_WP; PPC_A=0.1_WP"),
            "rainonsnow": ("TM_A=278._WP; TC_A=273.1_WP; TGS_A=273._WP; TD_A=274._WP; TA_A=273.1_WP",
                           "QM_A=0.006_WP; RLWDOWN_A=350._WP; SWDOWN1_A=100._WP; SUNANGLE_A=0.2_WP", "PPL_A=1._WP; PPC_A=0.2_WP"),
        }
        if args.snow_profile in profiles:
            t,q,p=profiles[args.snow_profile]
            code=code.replace("TM_A=268._WP; TC_A=268._WP; TGS_A=267._WP; TD_A=270._WP; TA_A=268._WP",t)
            code=code.replace("QM_A=0.002_WP; RLWDOWN_A=250._WP; SWDOWN1_A=100._WP; SUNANGLE_A=0.2_WP",q)
            code=code.replace("PPL_A=0._WP; PPC_A=0._WP",p)
        if args.surface_caller:
            # WRF surface_driver overwrites the actual argument T2 (TA) with
            # TSK after SSIB, and restores that held value before the next call.
            # Keep the raw routine output separately for a column boundary gate.
            code=code.replace("  DO step=1,", "  TC_0=TC_A; TGS_0=TGS_A; TD_0=TD_A; TA_0=TA_A; TKAIR_A=0._WP\n  DO step=1,")
            code=code.replace("     TA_A(i)=io_ta ; BEDO_A(i)=io_bedo", "     CALL dump('TA_RAW',i,io_ta)\n     TA_A(i)=o_tgeff ; BEDO_A(i)=io_bedo")
            code=code.replace("     CALL dump('SNOWDEPTH',i, SNOWDEPTH_A(i))", "     CALL dump('SNOWDEPTH',i, SNOWDEPTH_A(i))\n     CALL dump('TKAIR',i,TKAIR_A(i))")
        (build / driver.name).write_text(code)
    (build / "wrfstubs.f90").write_text(STUBS)
    flags = ["-O2", "-ffree-form", "-ffree-line-length-none", "-cpp",
             "-DEM_CORE=1", "-ffpe-summary=none", "-fallow-argument-mismatch"]
    if args.mode == "fp64":
        flags += ["-fdefault-real-8", "-fdefault-double-8"]
    if args.mode in ("zero", "one"):
        flags += ["-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    if args.mode == "poison":
        flags += ["-g", "-fcheck=bounds", "-finit-real=snan",
                  "-ffpe-trap=invalid,zero,overflow"]
    flags += ["-march=native", "-ffp-contract=fast"] if args.fma else ["-ffp-contract=off"]
    receipt = {
        "kind": "pristine-reference-audit-NOT-port-fidelity",
        "source": str(source), "source_sha256": digest(source),
        "caller": str(driver), "caller_sha256": digest(driver),
        "mode": args.mode, "flags": flags,
        "cache": "NONE", "backend": "Fortran CPU, no JAX/GPU",
        "cases": [],
        "snow": args.snow, "steps": args.steps if args.snow else 1,
        "reverse_column_calls": args.reverse, "surface_caller":args.surface_caller, "snow_profile":args.snow_profile,
        "compiled_caller_sha256": digest(build / driver.name),
    }
    assert digest(build / source.name) == digest(source)
    if args.mode == "one":
        # Deliberately changed-source sensitivity control, NOT the oracle.
        # GCC accepts zero/nan/inf initialization, not arbitrary real values.
        raw = (build / source.name).read_text()
        start = raw.index("      SUBROUTINE UPDAT1(")
        end = raw.index("      END SUBROUTINE UPDAT1", start)
        section = raw[start:end]
        needle = "      SNOFAC = HLAT / ( HLAT + SNOMEL /1000. )"
        assert section.count(needle) == 1
        raw = raw[:start] + section.replace(needle, "      SNOWW(2) = 1.0 ! DIAGNOSTIC SENSITIVITY ONLY\n" + needle) + raw[end:]
        (build / source.name).write_text(raw)
        receipt["kind"] = "changed-source-SNOWW2-one-sensitivity-NOT-oracle"
    receipt["compiled_source_sha256"] = digest(build / source.name)
    command = [args.compiler, *flags, source.name, "wrfstubs.f90",
               driver.name, "-o", "ssib_oracle"]
    start = time.monotonic()
    compiled = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=180)
    receipt["compile_seconds"] = time.monotonic() - start
    receipt["compile_returncode"] = compiled.returncode
    (args.output / "compile.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode == 0:
        for case in range(1, 6):
            start = time.monotonic()
            run = subprocess.run([str(build / "ssib_oracle"), str(case), args.mode],
                                 cwd=build, capture_output=True, text=True, timeout=30)
            stem = args.output / f"case_{case}"
            stem.with_suffix(".stdout").write_text(run.stdout)
            stem.with_suffix(".stderr").write_text(run.stderr)
            entry = {"case": case, "returncode": run.returncode,
                     "seconds": time.monotonic() - start}
            if run.returncode == 0:
                blocks = run.stdout.split("STEP=")[1:] if args.snow else [run.stdout]
                all_values = [fields(block) for block in blocks]
                values = all_values[-1]
                nonfinite = [{"step": step, "field": key} for step, data in enumerate(all_values, 1)
                             for key, value in data.items() if not math.isfinite(value)]
                entry.update(fields=sum(map(len, all_values)), nonfinite=nonfinite,
                             snow_input_nonzero=sum(value != 0 for key, value in values.items()
                                                    if key.startswith("SNOA_IN[")))
                if args.snow:
                    entry["isnow_by_step"] = [[data[f"ISNOW[{i}]"] for i in range(1, 6)]
                                              for data in all_values]
                if args.mode == "fp64" and not args.snow:
                    frozen = json.loads((REPO / "proofs/v017/savepoints/ssib/fp64" /
                                         f"ssib_case_{case}.json").read_text())["columns"]
                    baseline = {f"{key}[{i}]": value for key, array in frozen.items()
                                for i, value in enumerate(array, 1)}
                    entry["legacy_same_inventory"] = values.keys() == baseline.keys()
                    entry["legacy_different_values"] = [key for key in values.keys() & baseline.keys()
                                                        if values[key] != baseline[key]]
            receipt["cases"].append(entry)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
