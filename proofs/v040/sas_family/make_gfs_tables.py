"""Reproducible WRF lookup-table generation in the lane directory, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PHYS = ROOT / "data/wrf_pristine/WRF/phys"
VARS = "c1xtdp,c2xtdp,c1xthe,c2xthe,c1ythe,c2ythe,c1xtlcl,c2xtlcl,c1ytlcl,c2ytlcl,c1xma,c2xma,c1yma,c2yma,tbtdp,tbthe,tbtlcl,tbtma,tbqma"
SHAPES = {"tdp": (5001,), "the": (241,151), "tlcl": (151,61), "tma": (151,121), "qma": (151,121)}


def generate(work, dest):
    import numpy as np
    work.mkdir(parents=True, exist_ok=True)
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    files = ("module_gfs_machine.F", "module_gfs_physcons.F", "module_gfs_funcphys.F")
    manifest = {"origin": "pristine WRF 4.7.1 module_gfs_funcphys GFUNCPHYS", "source_sha256": {},
                "export_only": "Insert PUBLIC declaration before CONTAINS; all original source bytes preserved", "builds": {}}
    for name in files:
        manifest["source_sha256"][name] = hashlib.sha256((PHYS/name).read_bytes()).hexdigest()
    for mode in ("r4", "r8"):
        build = work / mode
        build.mkdir(exist_ok=True)
        for name in files:
            s = (PHYS/name).read_text()
            if "funcphys" in name:
                marker = "contains\n"
                assert s.count(marker) == 1
                s = s.replace(marker, "public :: " + VARS + "\n" + marker)
            (build/name).write_text(s)
        shutil.copyfile(Path(__file__).with_name("gfs_tables.F90"), build/"gfs_tables.F90")
        flags = ["-O2", "-w", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DEM_CORE=1", "-DNMM_CORE=0"]
        if mode == "r8":
            flags += ["-fdefault-real-8", "-fdefault-double-8"]
        subprocess.run([compiler, *flags, *files, "gfs_tables.F90", "-o", "dump_tables"], cwd=build, check=True, timeout=120)
        subprocess.run([str(build/"dump_tables")], cwd=build, check=True, timeout=120)
        data = np.fromfile(build/"gfs_tables.bin", dtype="=f8")
        arrays = {"coefficients": data[:14]}
        offset = 14
        for key, shape in SHAPES.items():
            count = int(np.prod(shape))
            arrays[key] = data[offset:offset+count].reshape(shape, order="F")
            offset += count
        assert offset == data.size and all(np.isfinite(a).all() for a in arrays.values())
        out = dest / f"gfs_sas_tables_{mode}.npz"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        manifest["builds"][mode] = {"flags": flags, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                   "bytes": out.stat().st_size, "float64_values": data.size}
    (dest/"gfs_sas_tables_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("destination", type=Path)
    opts = parser.parse_args()
    generate(opts.work.resolve(), opts.destination.resolve())
