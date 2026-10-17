"""Extract unmodified pristine nudob/helpers into a lane-local REAL4/REAL8 oracle.

Usage: python build.py <build-dir>. No shared WRF build is changed or run.
module_domain and module_model_constants are unused USE dependencies in these
particular routines: empty modules resolve the names, supplying no constants,
types, procedures or arithmetic. DM_PARALLEL selects ERRF(5/9) initialization
exactly as the distributed WRF caller requires (serial nudob leaves kpbl_obs
uninitialized). No communication or WRF I/O is called by the extracted routines.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PRISTINE = Path(os.environ.get("PRISTINE", "<USER_HOME>/src/wrf_pristine/WRF"))
FC = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")
HERE = Path(__file__).resolve().parent


def extract(text, name, kind="SUBROUTINE", prefix=""):
    pattern = rf"(?im)^ *{prefix}{kind} +{name}\b.*?^ *END {kind} +{name}\b[^\n]*"
    match = re.search(pattern, text, re.S)
    if match is None:
        raise ValueError(f"missing pristine routine {name}")
    return match.group(0) + "\n"


def main(build):
    build.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ("module_fddaobs_rtfdda.F", "module_fddaobs_driver.F"):
        data = (PRISTINE / "phys" / name).read_bytes()
        (build / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    rt = (build / "module_fddaobs_rtfdda.F").read_text()
    drv = (build / "module_fddaobs_driver.F").read_text()
    input_data = (PRISTINE / "share/wrf_fddaobs_in.F").read_bytes()
    (build / "wrf_fddaobs_in.F").write_bytes(input_data)
    sources["wrf_fddaobs_in.F"] = hashlib.sha256(input_data).hexdigest()
    formats = []
    for label in (101, 102, 1021, 103, 104, 105):
        match = re.search(rf"(?im)^ *{label} +FORMAT[^\n]*", input_data.decode())
        if match is None:
            raise ValueError(f"missing pristine FORMAT {label}")
        formats.append(match.group(0))
    (build / "obs_formats.inc").write_text("\n".join(formats) + "\n")
    routines = (extract(rt, "nudob") + extract(rt, "calc_rcouple_scales")
                + extract(rt, "get_timewt", "FUNCTION", "REAL ")
                + extract(drv, "get_vif_hts_slab"))
    module = "module obs_oracle\ncontains\n" + routines + "end module obs_oracle\n"
    (build / "extracted.F90").write_text(module)
    (build / "unused_modules.f90").write_text(
        "module module_domain\nend module\nmodule module_model_constants\nend module\n"
        "subroutine wrf_error_fatal(msg)\ncharacter(*) msg\nprint *, msg\nstop 1\nend subroutine\n"
        "subroutine wrf_message(msg)\ncharacter(*) msg\nprint *, msg\nend subroutine\n")
    flags = ["-O2", "-ftree-vectorize", "-funroll-loops", "-ffree-form", "-ffree-line-length-none",
             "-cpp", "-DDM_PARALLEL", "-fcheck=bounds"]
    for kind in ("r4", "r8"):
        out = build / kind
        out.mkdir(exist_ok=True)
        (out / "tmp").mkdir(exist_ok=True)
        extra = [] if kind == "r4" else ["-fdefault-real-8", "-fdefault-double-8"]
        subprocess.run([FC, *flags, *extra, "-c", str(build / "unused_modules.f90"), str(build / "extracted.F90")],
                       cwd=out, env={**os.environ, "TMPDIR": str(out / "tmp")}, check=True, timeout=180)
        for driver, binary in (("driver.F90", "oracle"), ("helpers.F90", "helpers")):
            subprocess.run([FC, *flags, *extra, "unused_modules.o", "extracted.o", str(HERE / driver), "-o", binary],
                           cwd=out, env={**os.environ, "TMPDIR": str(out / "tmp")}, check=True, timeout=60)
        subprocess.run([FC, *flags, *extra, "-I" + str(build), str(HERE / "decode_driver.F90"), "-o", "decode"],
                       cwd=out, env={**os.environ, "TMPDIR": str(out / "tmp")}, check=True, timeout=60)
    manifest = {"sources": sources, "extract_sha256": hashlib.sha256(module.encode()).hexdigest(),
                "drivers": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                            for name in ("build.py", "driver.F90", "helpers.F90", "decode_driver.F90")},
                "compiler": subprocess.check_output([FC, "--version"], text=True).splitlines()[0],
                "flags": flags, "precision": {"r4": "WRF REAL", "r8": "-fdefault-real-8 -fdefault-double-8"},
                "boundary": "nudob single-level branches; ERRF and RKO supplied after errob; root full tile"}
    (build / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
