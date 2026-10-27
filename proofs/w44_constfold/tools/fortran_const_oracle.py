"""Pristine-gfortran CONSTANT oracle: WRF's own folded value of a Fortran constant expression.

Each entry is (label, use-statement(s), Fortran expression, kind) with kind 'r4' (REAL) or 'r8' (DOUBLE).
The tool writes one Fortran program that USEs the PRISTINE WRF 4.7.1 modules (the .mod files of the
build in <USER_HOME>/src/wrf_pristine/WRF, i.e. exactly the PARAMETER chain wrf.exe was compiled with),
compiles it with the pristine build compiler (conda gfortran 14.3) and prints the IEEE bit pattern of
each expression as gfortran folds it.  This is the bitwise reference for a port constant.

Expressions are written VERBATIM as in the WRF statement (same parentheses / left-to-right order).

usage (library): oracle(entries, workdir) -> {label: {"bits": int, "value": float, "kind": "r4"|"r8"}}
"""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
from pathlib import Path

WRF = Path(os.environ.get("WRF_PRISTINE", "<USER_HOME>/src/wrf_pristine/WRF"))
FC = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")
INCLUDES = [WRF / "frame", WRF / "phys", WRF / "share", WRF / "main"]


def oracle(entries, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    uses = []
    for _label, use, _expr, _kind in entries:
        for u in ([use] if isinstance(use, str) else use):
            if u and u not in uses:
                uses.append(u)
    body = ["program const_oracle"] + [f"  {u}" for u in uses] + ["  implicit none",
                                                                  "  integer(4) :: i4", "  integer(8) :: i8"]
    for idx, (label, _use, expr, kind) in enumerate(entries):
        if kind == "r4":
            body += [f"  i4 = transfer(real({expr}, 4), i4)", f"  write(*,'(I0,1X,I0)') {idx}, i4"]
        else:
            body += [f"  i8 = transfer(real({expr}, 8), i8)", f"  write(*,'(I0,1X,I0)') {idx}, i8"]
    body.append("end program const_oracle")
    src = "\n".join(body) + "\n"
    f90 = workdir / "const_oracle.f90"
    f90.write_text(src)
    exe = workdir / "const_oracle"
    cmd = [FC, "-ffree-line-length-none", "-O2"] + [f"-I{p}" for p in INCLUDES] + ["-o", str(exe), str(f90)]
    # module_model_constants etc. hold only PARAMETERs -> no object files needed
    subprocess.run(cmd, check=True, cwd=workdir)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split("\n")
    result = {}
    for line in out:
        if not line.strip():
            continue
        idx, bits = (int(x) for x in line.split())
        label, _use, expr, kind = entries[idx]
        if kind == "r4":
            value = struct.unpack("<f", struct.pack("<i", bits))[0]
            bits &= 0xFFFFFFFF
        else:
            value = struct.unpack("<d", struct.pack("<q", bits))[0]
            bits &= 0xFFFFFFFFFFFFFFFF
        result[label] = {"bits": bits, "value": value, "kind": kind, "expr": expr}
    result["_meta"] = {"compiler": FC, "wrf": str(WRF), "source_sha256": hashlib.sha256(src.encode()).hexdigest()}
    return result
