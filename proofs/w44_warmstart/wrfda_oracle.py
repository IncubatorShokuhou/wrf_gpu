"""Compile unedited WRFDA coupling and compare real WN3 input fields.

Usage: python wrfda_oracle.py PRISTINE CASE ARTIFACT_DIR
No model forecast and no JAX import; caller pins CPUs and guards QUIET.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from gpuwrf.cycling.boundary import coupled_field
from gpuwrf.io.netcdf_lock import Dataset


def main():
    pristine, case, artifacts = map(Path, sys.argv[1:])
    artifacts.mkdir(exist_ok=False, parents=True)
    compiler = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    routines = ("da_calc_mu_uv.inc", "da_couple.inc", "da_couple_uv.inc")
    sources = [pristine / "WRF/var/da/da_update_bc" / name for name in routines]
    binary = artifacts / "coupling.so"
    command = [str(compiler), "-shared", "-fPIC", "-O0", "-fcheck=all", "-ffp-contract=off",
               "-x", "f95", *map(str, sources), "-o", str(binary)]
    start = time.perf_counter()
    compile_result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (artifacts / "compile.log").write_text(compile_result.stdout + compile_result.stderr)
    compile_result.check_returncode()
    native = ct.CDLL(str(binary))
    pointer = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
    f3 = lambda a: np.asfortranarray(np.asarray(a).transpose(2, 1, 0), dtype=np.float32)
    f2 = lambda a: np.asfortranarray(np.asarray(a).T, dtype=np.float32)
    records = []
    for initial in sorted(case.glob("wrfinput_d0[123]")):
        with Dataset(initial) as ds:
            nx, ny, nz = (len(ds.dimensions[n]) for n in ("west_east", "south_north", "bottom_top"))
            bounds = [ct.c_int(n) for n in (1, nx, 1, ny, 1, nz)]
            u, v = f3(ds["U"][0]), f3(ds["V"][0])
            mu, mub = f2(ds["MU"][0]), f2(ds["MUB"][0])
            msfu, msfv = f2(ds["MAPFAC_U"][0]), f2(ds["MAPFAC_V"][0])
            c1, c2 = (np.ascontiguousarray(ds[n][0], np.float32) for n in ("C1H", "C2H"))
            native.da_couple_uv_(*[pointer(a) for a in (u, v, mu, mub, msfu, msfv, c1, c2)],
                                 *map(ct.byref, bounds))
            expected = {"U": u.transpose(2, 1, 0), "V": v.transpose(2, 1, 0)}
            mass = f2(np.asarray(ds["MU"][0]) + np.asarray(ds["MUB"][0]))
            ones = np.ones((nx, ny), np.float32, order="F")
            for name in ("W", "T", "PH", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN"):
                origin = "THM" if name == "T" and ds.USE_THETA_M else name
                value = f3(ds[origin][0])
                level = "F" if name in ("W", "PH") else "H"
                c1, c2 = (np.ascontiguousarray(ds[f"C{n}{level}"][0], np.float32) for n in (1, 2))
                dims = [ct.c_int(n) for n in (1, nx, 1, ny, 1, value.shape[-1])]
                msf = f2(ds["MAPFAC_M"][0]) if name == "W" else ones
                native.da_couple_(*[pointer(a) for a in (mass, value, msf, c1, c2)], *map(ct.byref, dims))
                expected[name] = value.transpose(2, 1, 0)
            for name, reference in expected.items():
                result = coupled_field(ds, name)
                equal = result.tobytes() == reference.tobytes()
                records.append({"file": str(initial), "field": name, "byte_equal": equal,
                                "nonzero_elements": int(np.count_nonzero(reference)), "elements": int(reference.size),
                                "max_abs_error": float(np.max(np.abs(result.astype(float) - reference)))})
    report = {"passed": all(r["byte_equal"] for r in records), "records": records,
              "command": command, "wall_seconds": time.perf_counter()-start,
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(case.glob("wrfinput_d0[123]"))},
              "scope": "original WRFDA coupling, real cold-input arrays; zero cloud fields disclosed, no cloudy-forecast claim",
              "jax_imported": "jax" in sys.modules}
    (artifacts / "report.json").write_text(json.dumps(report, indent=2)+'\n')
    binary.unlink()
    print({"passed": report["passed"], "byte_equal_fields": sum(r["byte_equal"] for r in records),
           "fields": len(records), "wall_seconds": report["wall_seconds"]})
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
