"""Build unmodified WRF RUC init/diagnostic modules and emit tracked savepoints.

Run with JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19
python proofs/o2_ruccli/build_oracle.py. No model run or GPU use.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
BUILD = Path("<USER_HOME>/wrf_gpu2_lanes/o2-ruccli/oracle_build")
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"


def guard():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("QUIET")


def run(args, cwd):
    guard()
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, timeout=180)


def main():
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    guard()
    BUILD.mkdir(parents=True, exist_ok=True)
    sources = [WRF / "share/module_model_constants.F", WRF / "frame/module_wrf_error.F",
               WRF / "phys/module_sf_ruclsm.F", WRF / "phys/module_sf_sfcdiags_ruclsm.F"]
    result = {"sources": {str(p.relative_to(WRF)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "compiler": run([FC, "--version"], BUILD).stdout.splitlines()[0],
              "driver_sha256": hashlib.sha256(Path(__file__).with_suffix(".f90").read_bytes()).hexdigest()
              if Path(__file__).with_suffix(".f90").exists() else
              hashlib.sha256((Path(__file__).parent / "ruc_init_diag_oracle.f90").read_bytes()).hexdigest()}
    for precision in ("fp32", "fp64"):
        scratch = BUILD / precision
        scratch.mkdir(exist_ok=True)
        flags = ["-O2", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DEM_CORE=1",
                 "-fallow-argument-mismatch", "-finit-integer=0"]
        if precision == "fp64":
            flags += ["-fdefault-real-8", "-fdefault-double-8"]
        for path in sources + [Path(__file__).parent / "ruc_init_diag_oracle.f90"]:
            # Copy only source files to private scratch; WRF originals stay read-only.
            shutil.copyfile(path, scratch / path.name)
        for name in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
            (scratch / name).symlink_to(WRF / "run" / name) if not (scratch / name).exists() else None
        (scratch / "stubs.f90").write_text('''logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_string(buf,n)
character(len=*)::buf
integer::n
end subroutine
subroutine wrf_dm_bcast_integer(buf,n)
integer::n,buf(*)
end subroutine
subroutine wrf_dm_bcast_real(buf,n)
integer::n
real::buf(*)
end subroutine
subroutine wrf_debug(level,msg)
integer::level
character(len=*)::msg
end subroutine
subroutine wrf_abort()
stop 1
end subroutine
''')
        files = [p.name for p in sources] + ["stubs.f90", "ruc_init_diag_oracle.f90"]
        for name in files:
            run([FC, *flags, "-c", name], scratch)
        run([FC, *flags, "-o", "oracle", *[str(Path(p).with_suffix(".o")) for p in files]], scratch)
        output = run(["./oracle"], scratch).stdout
        result[precision] = {"flags": flags,
                             "diagnostics": [[float(x) for x in line.split()[1:]] for line in output.splitlines() if line.startswith("DIAG")],
                             "init": [[float(x) for x in line.split()[1:]] for line in output.splitlines() if line.startswith("INIT")]}
    out = ROOT / "proofs/o2_ruccli/init_diag_savepoints.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    # Exact private paths; no shared/reference data is removed.
    guard()
    shutil.rmtree(BUILD / "fp32")
    shutil.rmtree(BUILD / "fp64")


if __name__ == "__main__":
    main()
