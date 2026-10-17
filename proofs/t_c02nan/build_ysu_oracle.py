"""Stage unchanged WRF 4.7.1 YSU bodies with a file-fed column driver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--nz", type=int, default=44)
    parser.add_argument("--dt", type=float, default=18.0)
    args = parser.parse_args()
    assert not Path("/tmp/wrf_gpu2_quiet").exists()
    root = Path(__file__).resolve().parents[2]
    wrf = root / "data/wrf_pristine/WRF"
    args.build.mkdir(parents=True, exist_ok=True)
    files = ("phys/ccpp_kind_types.f90", "phys/physics_mmm/bl_ysu.F90", "phys/module_bl_ysu.F")
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags = ["-O0", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none", "-cpp",
        "-finit-real=zero", "-finit-integer=0", "-finit-logical=false", "-ffpe-summary=none"]
    manifest = dict(compiler=compiler, flags=flags, nz=args.nz, dt=args.dt,
        physics_bodies="unchanged pristine WRF 4.7.1", source_sha256={})
    for name in files:
        path = wrf / name
        manifest["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, args.build / path.name)
        subprocess.run([compiler, *flags, "-c", path.name], cwd=args.build, check=True, timeout=60)
    driver = (root / "proofs/v060/oracle/ysu_oracle_driver.f90").read_text()
    driver = driver.replace("KX = 32", f"KX = {args.nz}")
    driver = driver.replace("  LOGICAL :: flag_qc, flag_qi, flag_bep", "  LOGICAL :: flag_qc, flag_qi, flag_bep, from_file")
    lines = ["  INQUIRE(FILE='column.in', EXIST=from_file)", "  IF (from_file) THEN", "    OPEN(42,FILE='column.in',STATUS='OLD',ACTION='READ')"]
    for name in ("U", "V", "T", "QV", "P", "PDI", "PI", "DZ"):
        upper = "KX+1" if name == "PDI" else "KX"
        lines.append(f"    READ(42,*) {name}(1,1:{upper},1)")
    for name in ("PSFC", "ZNT", "UST", "HFX", "QFX", "WSPD", "BR", "PSIM", "PSIH", "XLAND", "U10", "V10"):
        lines.append(f"    READ(42,*) {name}(1,1)")
    lines += ["    CLOSE(42)", "  ENDIF", ""]
    driver = driver.replace("  RTHRATEN = 0.0", "\n".join(lines)+"\n  RTHRATEN = 0.0")
    driver = driver.replace("dt=60.0", f"dt={args.dt}").replace("'DT=', 60.0", f"'DT=', {args.dt}")
    path = args.build / "ysu_driver.f90"
    path.write_text(driver)
    subprocess.run([compiler, *flags, "-o", "ysu_oracle", path.name,
        "ccpp_kind_types.o", "bl_ysu.o", "module_bl_ysu.o"], cwd=args.build, check=True, timeout=60)
    manifest["driver_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["executable_sha256"] = hashlib.sha256((args.build / "ysu_oracle").read_bytes()).hexdigest()
    (args.build / "build.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
