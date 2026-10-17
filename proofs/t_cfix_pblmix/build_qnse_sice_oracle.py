"""QNSE SICE (WRF XICE) pristine cases for F23 / audit B5.

Same UNMODIFIED WRF sources, stub and flags as proofs/s2_qnse/build_oracle.py; the driver
(qnse_sice_driver.f90) is proofs/s2_qnse/oracle.f90 plus cid 16 (sea-ice LAND point, XICE=1,
no snow) and cid 17 (identical, XICE=0).  Cases 1-15 must reproduce the committed
oracle.json byte-for-byte before 16/17 are written (E83).
Usage: python build_qnse_sice_oracle.py <wrf_root> <build_dir> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s2_qnse"))
from build_oracle import SOURCES, parse  # noqa: E402

FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
FLAGS = ["-O0", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DNMM_CORE=0", "-ffp-contract=off", "-fcheck=all",
         "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]


def main(wrf, build, out):
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("QUIET")
    wrf, build = Path(wrf), Path(build)
    build.mkdir(parents=True, exist_ok=True)
    sources = {}
    for rel in SOURCES:
        src = wrf / rel
        sources[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copyfile(src, build / src.name)
    shutil.copyfile(HERE / "qnse_sice_driver.f90", build / "oracle.f90")
    (build / "module_dm.f90").write_text(
        "module module_dm\ncontains\nsubroutine wrf_dm_maxval(a,b,c)\n"
        "integer :: a,b,c\nstop 98\nend subroutine\nend module\n")
    committed = json.loads((HERE.parent / "s2_qnse/oracle.json").read_text())
    assert committed["source_sha256"] == sources, "pristine sources differ from the committed QNSE oracle"
    cases = {}
    for mode in ("fp32", "fp64"):
        extra = [] if mode == "fp32" else ["-fdefault-real-8", "-fdefault-double-8"]
        subprocess.run([FC, *FLAGS, *extra, "module_model_constants.F", "module_dm.f90", "module_sf_qnsesfc.F",
                        "module_bl_qnsepbl.F", "oracle.f90", "-o", "oracle"], cwd=build, check=True, timeout=120)
        runs = [parse(subprocess.run([str(build / "oracle"), str(cid)], capture_output=True, text=True,
                                     check=True, timeout=20).stdout) for cid in range(1, 18)]
        assert runs[:15] == committed["cases"][mode], f"{mode}: driver does not reproduce committed cases 1-15"
        cases[mode] = {"sice1": runs[15], "sice0": runs[16]}
    manifest = {"source_sha256": sources,
                "driver_sha256": hashlib.sha256((HERE / "qnse_sice_driver.f90").read_bytes()).hexdigest(),
                "compiler": subprocess.check_output([FC, "--version"], text=True).splitlines()[0], "flags": FLAGS,
                "reproduces_committed_cases_1_15": True,
                "scope": "ARW single column QNSESFC then QNSEPBL at a sea-ice LAND point (XLAND=1, XICE=1, SNOW=0) and "
                         "the same column with XICE=0; qnsepbl.F:497 RLIVWV branch, qnsesfcinit Z0 branch",
                "cases": cases}
    Path(out).write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
