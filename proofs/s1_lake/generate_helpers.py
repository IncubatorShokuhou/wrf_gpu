"""Build explicit DOUBLE helper fixtures in a disposable private directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wrf", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    p.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    build = args.out / "build"
    build.mkdir(exist_ok=False)
    source = args.wrf / "phys/module_sf_lake.F"
    (build / source.name).write_bytes(source.read_bytes())
    command = [args.compiler, "-O0", "-ffree-form", "-ffree-line-length-none", "-cpp",
               "-DEM_CORE=1", "-fcheck=bounds", "-ffp-contract=off", str(here / "wrf_stubs.F90"),
               source.name, str(here / "helper_driver.F90"), "-o", "helpers"]
    start = time.monotonic()
    try:
        compiled = subprocess.run(command, cwd=build, text=True, capture_output=True, timeout=120)
        (args.out / "build.log").write_text(compiled.stdout + compiled.stderr)
        compiled.check_returncode()
        ran = subprocess.run([str(build / "helpers")], cwd=build, text=True, capture_output=True, timeout=30)
        (args.out / "outputs.txt").write_text(ran.stdout)
        (args.out / "run.log").write_text(ran.stderr)
        ran.check_returncode()
        groups = {}
        for line in ran.stdout.splitlines():
            tag, *values = line.split()
            groups.setdefault(tag, []).append([float(x) for x in values])
        payload = {tag: np.array(rows, dtype=np.float64) for tag, rows in groups.items()}
        if not all(np.isfinite(x).all() for x in payload.values()):
            raise ValueError("Nonfinite original WRF helper output")
        for tag in groups:
            if tag.startswith("TRI_"):
                payload[tag] = payload[tag].reshape(25,3).T
        payload["TRI_TOP"] = np.array([0,5,10], np.int32)
        payload["TRI_FILTER"] = np.array([True, True, False])
        np.savez_compressed(args.out / "helpers.npz", **payload)
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        manifest = dict(source=str(source), source_sha256=sha(source),
                        driver_sha256=sha(here / "helper_driver.F90"),
                        stubs_sha256=sha(here / "wrf_stubs.F90"), build_command=command,
                        compiler=subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
                        fixture_sha256=sha(args.out / "helpers.npz"),
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        wall_s=time.monotonic()-start,
                        precision="All helper inputs/outputs WRF r8 DOUBLE, no kind override",
                        tolerance=dict(rtol=2e-14, atol=0.0),
                        shapes={tag: list(value.shape) for tag, value in payload.items()})
        (args.out / "helpers.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["shapes"]))
    finally:
        shutil.rmtree(build)


if __name__ == "__main__":
    main()
