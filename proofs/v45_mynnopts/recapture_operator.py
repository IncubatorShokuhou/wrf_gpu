"""Recapture only mym_condensation from unedited pristine WRF objects.

Uses the frozen real WN3/Swiss operands. No case run or reference is modified.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    driver = Path(__file__).with_name("mym_condensation_driver.F90")
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    objects = [wrf / "phys/module_bl_mynnedmf.o", wrf / "phys/module_bl_mynnedmf_common.o",
               wrf / "phys/ccpp_kind_types.o", wrf / "share/module_model_constants.o"]
    executable = output / "oracle"
    command = [compiler, "-I" + str(wrf / "phys"), "-ffree-form", "-ffree-line-length-none",
               "-O2", "-ffp-contract=off", str(driver), *map(str, objects), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (output / "compile.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    sys.path.insert(0, str(root / "tests/v045"))
    from test_mynn_cloudpdf import _compiled_gate
    fixture = root / "tests/fixtures/o5_n333/mynn_cloudpdf_pristine_v1.npz"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    fields = ("dz", "zw", "p", "exner", "thl", "qw", "qv", "qc", "qi", "qs", "tsq", "qsq", "cov", "sh", "el", "th")
    with np.load(fixture) as data:
        ncol, nz = data["in_th"].shape
        assert (ncol, nz) == (384, 44)
        scalars = np.column_stack([data["in_" + name].reshape(ncol) for name in ("dx", "xland", "pblh", "hfx")])
        records = np.concatenate([scalars] + [data["in_" + name] for name in fields], axis=1).astype("<f4")
        assert np.isfinite(records).all()
        for option in (0, 1):
            inp, out = output / f"input_{option}.bin", output / f"original_{option}.bin"
            with inp.open("wb") as stream:
                np.asarray([ncol, nz, option], "<i4").tofile(stream)
                records.tofile(stream)
            subprocess.run([str(executable), str(inp), str(out)], check=True, timeout=30, capture_output=True)
            original = np.fromfile(out, dtype="<f4").reshape(ncol, 6, nz)
            checks = {}
            for index, name in enumerate(("qc_bl", "qi_bl", "cldfra_bl", "sgm")):
                checks[name] = np.ascontiguousarray(original[:, index]).tobytes() == data[f"cond{option}_{name}"].tobytes()
            assert all(checks.values()), checks
            # The independent reference is freshly recaptured above and is byte
            # identical to the fixture. Apply the unchanged full-compiled gate.
            _compiled_gate(data, option)
            rows.append(dict(option=option, original_sha256=sha(out), frozen_byte_checks=checks,
                             compiled_gate_passed=True))
    report = dict(passed=True, backend=jax.default_backend(), columns=ncol, levels=nz, rows=rows,
        fixture_sha256=sha(fixture), driver_sha256=sha(driver), executable_sha256=sha(executable),
        wrf_source_sha256=sha(wrf / "phys/module_bl_mynnedmf.F"),
        object_sha256={str(path): sha(path) for path in objects}, command=command,
        scope=__doc__)
    (output / "RECAPTURE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "backend", "columns", "levels", "rows")}))


if __name__ == "__main__":
    main()
