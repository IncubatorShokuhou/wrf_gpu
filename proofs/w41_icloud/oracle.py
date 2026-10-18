"""Build and run the verbatim WRF 4.7.1 cal_cldfra3 oracle (lane-dir build, E104).

Subroutine bytes are copied unchanged from the pristine sources; only the caller
(oracle_driver.f90) and the wrf_debug/wrf_error_fatal stubs are ours.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys")
COMPILER = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran"
FLAGS = ["-O2", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none", "-finit-real=zero",
         "-finit-integer=0", "-finit-logical=false"]
HERE = Path(__file__).resolve().parent


def _cut(text: bytes, start: bytes, end: bytes) -> bytes:
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return text[i:j] + b"\n"


def sources() -> dict:
    rad = (PRISTINE / "module_radiation_driver.F").read_bytes()
    mp = (PRISTINE / "module_mp_thompson.F").read_bytes()
    cf3 = _cut(rad, b"      SUBROUTINE cal_cldfra3(", b"      END SUBROUTINE adjust_cloudFinal")
    rslf = _cut(mp, b"      REAL FUNCTION RSLF(", b"      END FUNCTION RSLF")
    rsif = _cut(mp, b"      REAL FUNCTION RSIF(", b"      END FUNCTION RSIF")
    return {"module_radiation_driver.F": rad, "module_mp_thompson.F": mp,
            "cal_cldfra3..adjust_cloudFinal": cf3, "RSLF": rslf, "RSIF": rsif}


def build(build_dir: Path) -> dict:
    src = sources()
    build_dir.mkdir(parents=True, exist_ok=False)
    (build_dir / "thompson_sat.f90").write_bytes(
        b"module module_mp_thompson\nimplicit none\ncontains\n" + src["RSLF"] + src["RSIF"]
        + b"end module module_mp_thompson\n")
    (build_dir / "cldfra3.f90").write_bytes(
        b"module oracle_cldfra3\nimplicit none\ncontains\n" + src["cal_cldfra3..adjust_cloudFinal"]
        + b"end module oracle_cldfra3\n")
    files = [build_dir / "thompson_sat.f90", build_dir / "cldfra3.f90", HERE / "oracle_driver.f90"]
    subprocess.run([COMPILER, *FLAGS, "-J", str(build_dir), *map(str, files), "-o", str(build_dir / "oracle")],
                   check=True, capture_output=True, timeout=120)
    return {"compiler": COMPILER, "flags": FLAGS,
            "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in src.items()},
            "driver_sha256": hashlib.sha256((HERE / "oracle_driver.f90").read_bytes()).hexdigest()}


def run(build_dir: Path, columns: dict, gridkm: float, tag: str = "run") -> dict:
    """columns: qv qc qi qs dz p t as (ncol, nz) float32 and xland (ncol,)."""
    names = ("qv", "qc", "qi", "qs", "dz", "p", "t")
    arrays = {n: np.ascontiguousarray(columns[n], np.float32) for n in names}
    xland = np.ascontiguousarray(columns["xland"], np.float32)
    ncol, nz = arrays["qv"].shape
    inp, out = build_dir / f"{tag}.in", build_dir / f"{tag}.out"
    with open(inp, "wb") as fh:
        np.array([ncol, nz], np.int32).tofile(fh)
        np.array([gridkm], np.float32).tofile(fh)
        for c in range(ncol):
            for n in names:
                arrays[n][c].tofile(fh)
            xland[c:c + 1].tofile(fh)
    subprocess.run([str(build_dir / "oracle"), str(inp), str(out)], check=True, capture_output=True, timeout=600)
    raw = np.fromfile(out, np.float32).reshape(ncol, 4, nz)
    inp.unlink()
    out.unlink()
    return {"cldfra": raw[:, 0], "qc": raw[:, 1], "qi": raw[:, 2], "qv": raw[:, 3]}


def build_cldfra2(build_dir: Path) -> dict:
    """Verbatim cal_cldfra2 (radiation_driver.F:3801-3874) + a two-parameter state-description stub."""
    rad = (PRISTINE / "module_radiation_driver.F").read_bytes()
    body = _cut(rad, b"   SUBROUTINE cal_cldfra2(", b"   END SUBROUTINE cal_cldfra2")
    build_dir.mkdir(parents=True, exist_ok=False)
    (build_dir / "stub.f90").write_bytes(b"module module_state_description\ninteger, parameter :: KFCUPSCHEME=10, "
                                         b"KFETASCHEME=1\nend module module_state_description\n")
    (build_dir / "cldfra2.f90").write_bytes(b"module oracle_cldfra2\nimplicit none\ncontains\n" + body
                                            + b"end module oracle_cldfra2\n")
    files = [build_dir / "stub.f90", build_dir / "cldfra2.f90", HERE / "oracle2_driver.f90"]
    subprocess.run([COMPILER, *FLAGS, "-J", str(build_dir), *map(str, files), "-o", str(build_dir / "oracle2")],
                   check=True, capture_output=True, timeout=120)
    return {"sha256": hashlib.sha256(body).hexdigest()}


def run_cldfra2(build_dir: Path, qc, qi, f_qc: bool, f_qi: bool) -> np.ndarray:
    qc = np.ascontiguousarray(qc, np.float32).reshape(-1)
    qi = np.ascontiguousarray(qi, np.float32).reshape(-1)
    inp, out = build_dir / "c2.in", build_dir / "c2.out"
    with open(inp, "wb") as fh:
        np.array([qc.size, int(f_qc), int(f_qi)], np.int32).tofile(fh)
        qc.tofile(fh)
        qi.tofile(fh)
    subprocess.run([str(build_dir / "oracle2"), str(inp), str(out)], check=True, capture_output=True, timeout=120)
    cf = np.fromfile(out, np.float32)
    inp.unlink()
    out.unlink()
    return cf
