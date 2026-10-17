"""Build staged pristine WRF in a lane directory and freeze paired columns.

Run with CPU affinity/nice and JAX_PLATFORMS=cpu; never writes the WRF tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCES = ("share/module_model_constants.F", "phys/module_sf_qnsesfc.F", "phys/module_bl_qnsepbl.F")


def parse(text):
    sections = {"INPUT": {}, "OUTPUT": {}, "INPUT_SFC": {}, "OUTPUT_SFC": {}}
    section = None
    for line in text.splitlines():
        if line in sections:
            section = sections[line]
        elif section is not None and "=" in line:
            name, value = line.split("=", 1)
            if "[" in name:
                section.setdefault(name.split("[")[0], []).append(float(value))
            else:
                section[name] = float(value)
    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fc", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    args = ap.parse_args()
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("QUIET")
    args.build.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative in SOURCES:
        source = args.wrf / relative
        sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, args.build / source.name)
    shutil.copyfile(Path(__file__).with_name("oracle.f90"), args.build / "oracle.f90")
    # The ONLY imported DM procedure is inside NMM_CORE/DM_PARALLEL, excluded
    # for this serial ARW build. The stub exports a name for USE association;
    # it must never execute (otherwise this oracle aborts).
    (args.build / "module_dm.f90").write_text(
        "module module_dm\ncontains\nsubroutine wrf_dm_maxval(a,b,c)\n"
        "integer :: a,b,c\nstop 98\nend subroutine\nend module\n")
    cases = {}
    flags = ["-O0", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DNMM_CORE=0", "-ffp-contract=off", "-fcheck=all",
             "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    for mode in ("fp32", "fp64"):
        extra = [] if mode == "fp32" else ["-fdefault-real-8", "-fdefault-double-8"]
        subprocess.run([args.fc, *flags, *extra, "module_model_constants.F", "module_dm.f90",
                        "module_sf_qnsesfc.F", "module_bl_qnsepbl.F", "oracle.f90", "-o", "oracle"],
                       cwd=args.build, check=True, timeout=90)
        cases[mode] = []
        for cid in range(1, 16):
            result = subprocess.run([str(args.build / "oracle"), str(cid)], capture_output=True,
                                    text=True, check=True, timeout=20)
            (args.build.parent / f"{mode}_{cid}.txt").write_text(result.stdout)
            cases[mode].append(parse(result.stdout))
    manifest = {"source_sha256": sources, "driver_sha256": hashlib.sha256(
        Path(__file__).with_name("oracle.f90").read_bytes()).hexdigest(),
        "compiler": subprocess.check_output([args.fc, "--version"], text=True).splitlines()[0],
        "flags": flags, "fp64_flags": ["-fdefault-real-8", "-fdefault-double-8"],
        "scope": "ARW single-column QNSESFC then QNSEPBL, mfshconv=0; not a coupled forecast",
        "units": {"temperature": "K", "theta": "K", "p_mid": "Pa", "p_int": "Pa",
                  "u": "m s-1", "v": "m s-1", "tke": "m2 s-2", "EXCH_H": "m2 s-1",
                  "EXCH_M": "m2 s-1", "EL_MYJ": "m", "PBLH": "m", "qv": "kg kg-1 dry",
                  "qc": "kg kg-1", "RTHBLTEN": "K s-1", "RUBLTEN": "m s-2",
                  "RVBLTEN": "m s-2", "RQVBLTEN": "kg kg-1 s-1", "RQCBLTEN": "kg kg-1 s-1"},
        "tolerances": {"fp64": {"absolute": 5e-10, "relative": 5e-11},
                       "fp32": {"literal": "BYTE including signed zero",
                                "compiled": "independent input-ULP propagation enclosure; no absolute floors"}},
        "regimes": ["convective-land", "stable-night", "sea-smooth", "sea-transitional", "sea-rough",
                    "snow-land", "cloudy-convective", "weak-wind-stable", "negative-coriolis",
                    "first-step", "dry-surface", "CT-inversion-additive-tendencies",
                    "forced-flux", "absent-flux-optional", "first-step-water"], "cases": cases}
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: 15 paired columns x 2 precisions")


if __name__ == "__main__":
    main()
