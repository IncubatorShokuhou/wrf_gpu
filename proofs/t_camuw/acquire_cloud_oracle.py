"""Compile the verbatim original WRF resolved-cloud subroutine in a private build.

Produces base-case + one-input-ULP sensitivity oracle. Never edits pristine WRF.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

root, output, build = (Path(s).resolve() for s in sys.argv[1:4])
wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
original = (wrf / "phys/module_radiation_driver.F").read_bytes()
start = original.index(b"   SUBROUTINE cal_cldfra1(")
end = original.index(b"   END SUBROUTINE cal_cldfra1", start) + len(b"   END SUBROUTINE cal_cldfra1")
body = original[start:end] + b"\n"
build.mkdir(parents=True, exist_ok=False)
(build / "cal_cldfra1.f90").write_bytes(body)
compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran"
assert Path(compiler).exists(), compiler
flags = ["-O2", "-ffp-contract=off", "-ffree-line-length-none", "-finit-real=zero",
         "-finit-integer=0", "-finit-logical=false", "-I" + str(wrf / "frame")]
command = [compiler, *flags, str(build / "cal_cldfra1.f90"),
           str(root / "proofs/t_camuw/cloud_oracle_driver.f90"), "-o", str(build / "oracle")]
subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, cwd=build)

rows = []
# Clear, saturated, partially cloudy, ice/snow, weak cloud, cutoff and cold regimes.
for mp in (1, 3, 8, 50, 51):
    for t, p, qv, qc, qi, qs in (
        (285, 95000, .004, 0, 0, 0), (285, 95000, .012, 2e-5, 0, 0),
        (285, 95000, .006, 2e-5, 0, 0), (270, 70000, .0025, 3e-5, 2e-5, 1e-5),
        (250, 40000, .0005, 2e-6, 3e-5, 5e-5), (295, 100000, .009, 1e-7, 0, 0),
        (285, 95000, .006, 1e-12, 0, 0), (285, 95000, .006, 1e-13, 0, 0),
    ):
        rows.append([mp, *map(float, np.asarray([t, p, qv, qc, qi, qs], np.float32))])
expanded = []
for row in rows:
    expanded.append(row)
    for k in range(1, 7):
        for direction in (-np.inf, np.inf):
            perturbed = row.copy()
            perturbed[k] = float(np.nextafter(np.float32(row[k]), np.float32(direction)))
            expanded.append(perturbed)
text = "\n".join(" ".join(map(str, row)) for row in expanded) + "\n"
run = subprocess.run([str(build / "oracle")], input=text, check=True, capture_output=True,
                     text=True, timeout=30, cwd=build)
values = np.asarray([[float(s) for s in line.split()] for line in run.stdout.splitlines()])
assert values.shape == (len(rows) * 13, 2) and np.isfinite(values).all(), values.shape
cases = []
for k, row in enumerate(rows):
    result = values[k * 13:(k + 1) * 13]
    # Sum each input's one-ULP worst response; independent bound for contraction.
    sensitivity = sum(max(abs(result[2 * j + 1, 0] - result[0, 0]),
                          abs(result[2 * j + 2, 0] - result[0, 0])) for j in range(6))
    cases.append(dict(input=row, fraction=float(result[0, 0]), branch=int(result[0, 1]),
                      input_ulp_bound=float(sensitivity)))
receipt = dict(reference="verbatim original WRF cal_cldfra1; no-FMA REAL4",
               source=str(wrf / "phys/module_radiation_driver.F"),
               source_sha256=hashlib.sha256(original).hexdigest(),
               subroutine_sha256=hashlib.sha256(body).hexdigest(), compiler=compiler, flags=flags,
               compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
               rows=len(expanded), cases=cases, build_cleanup="removed after successful acquisition")
output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
shutil.rmtree(build)
print(json.dumps({"cases": len(cases), "rows": len(expanded), "branches": sorted(set(c["branch"] for c in cases))}))
