"""Compile the UNTOUCHED WRF RRTMG LW and SW 'MP option 3' staging blocks into a REAL4 fixture.

module_ra_rrtmg_lw.F:12069-12084 and module_ra_rrtmg_sw.F:10670-10685 (predicate from the
absent/false F_QI, warm_rain=.false. for WSM3 per physics_init.F:4463-4486), copied verbatim
and run on QC1D/QR1D/T1D after the driver's own zero initialisation of QI1D/QS1D.
Operands: the coldest-to-warm CPU-WRF c04 (mp3) history column at 2023-01-15_01 with the
largest qc+qr, plus threshold adversaries T = 273.15 exactly and one REAL ulp either side.
Usage: python build_wsm3_phase_oracle.py <cpu_wrfout> <out.json> <build_dir>
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np

FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
BASE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys")


def block(path):
    text = path.read_text()
    m = re.search(r"\n( +IF \( PRESENT \( F_QI \) \) THEN\n.*?! For MP option 3\n.*?\n +ENDDO\n +ENDIF\n)", text, re.S)
    assert m, path
    start = text[:m.start()].count("\n") + 2
    return m.group(1), start, start + m.group(1).count("\n") - 1


def main():
    wrfout, out, folder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    lw, lw_a, lw_b = block(BASE / "module_ra_rrtmg_lw.F")
    sw, sw_a, sw_b = block(BASE / "module_ra_rrtmg_sw.F")
    assert lw.split("! For MP option 3")[1] == sw.split("! For MP option 3")[1]
    d = nc.Dataset(wrfout)
    qc = np.asarray(d.variables["QCLOUD"][0], np.float32)
    qr = np.asarray(d.variables["QRAIN"][0], np.float32)
    thm = np.asarray(d.variables["THM"][0], np.float32) + np.float32(300)
    qv = np.asarray(d.variables["QVAPOR"][0], np.float32)
    p = np.asarray(d.variables["P"][0], np.float32) + np.asarray(d.variables["PB"][0], np.float32)
    d.close()
    t = thm / (np.float32(1) + np.float32(np.float32(461.6) / np.float32(287.0)) * qv) * (p / np.float32(1e5)) ** np.float32(2 / 7)
    j, i = np.unravel_index(np.argmax((qc + qr).sum(axis=0)), qc.shape[1:])
    nk = 20
    T, QC, QR = t[:nk, j, i].copy(), qc[:nk, j, i].copy(), qr[:nk, j, i].copy()
    edge = np.float32(273.15)
    T[[1, 2, 3]] = [edge, np.nextafter(edge, np.float32(0)), np.nextafter(edge, np.float32(400))]
    QC[[1, 2, 3]] = np.float32(2.5e-4)
    QR[[1, 2, 3]] = np.float32(1.5e-4)
    arrays = dict(T1D=T, QC1D=QC, QR1D=QR)

    def lit(a):
        return ",".join(f"transfer(int(z'{int(n):08X}',int32),0.0_real32)" for n in a.view(np.uint32))

    def program(body):
        return f"""program oracle
use iso_fortran_env, only: real32,int32
implicit none
integer :: K,kts,kte
logical :: predicate, warm_rain
real(real32) :: T1D({nk}),QC1D({nk}),QR1D({nk}),QI1D({nk}),QS1D({nk})
kts=1;kte={nk};warm_rain=.false.
T1D=[{lit(T)}]
QC1D=[{lit(QC)}]
QR1D=[{lit(QR)}]
QI1D=0.
QS1D=0.
call stage()
write(*,'({4 * nk}(Z8.8,1X))') (transfer(QC1D(K),0_int32),K=1,{nk}),(transfer(QR1D(K),0_int32),K=1,{nk}), &
  (transfer(QI1D(K),0_int32),K=1,{nk}),(transfer(QS1D(K),0_int32),K=1,{nk})
contains
subroutine stage(F_QI)
logical, optional :: F_QI
{body}
end subroutine
end program
"""
    folder.mkdir(parents=True, exist_ok=False)
    out_vals, receipts = {}, {}
    start = time.monotonic()
    for tag, body in (("lw", lw), ("sw", sw)):
        src = folder / f"{tag}.f90"
        src.write_text(program(body))
        exe = folder / tag
        cmd = [FC, "-O0", "-ffp-contract=off", "-ffree-line-length-none", str(src), "-o", str(exe)]
        subprocess.run(cmd, check=True, timeout=30)
        raw = subprocess.check_output([str(exe)], text=True, timeout=10)
        v = np.asarray([int(s, 16) for s in raw.split()], np.uint32).view(np.float32).reshape(4, nk)
        out_vals[tag] = {n: v[k].tolist() for k, n in enumerate(("QC1D", "QR1D", "QI1D", "QS1D"))}
        receipts[tag] = dict(program_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                             executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), command=cmd)
    assert out_vals["lw"] == out_vals["sw"]
    result = dict(
        reference="Untouched pristine RRTMG LW/SW MP-option-3 staging statements (WSM3); REAL-structured CPU-WRF "
                  "c04 column + 273.15 threshold adversaries; caller operator, not forecast truth",
        sources={n: hashlib.sha256((BASE / n).read_bytes()).hexdigest()
                 for n in ("module_ra_rrtmg_lw.F", "module_ra_rrtmg_sw.F")},
        lines={"lw": [lw_a, lw_b], "sw": [sw_a, sw_b]}, block_lw=lw, block_sw=sw,
        operand_file=str(wrfout), operand_column=dict(j=int(j), i=int(i), levels=nk),
        inputs={k: v.tolist() for k, v in arrays.items()}, outputs=out_vals["lw"], receipts=receipts,
        elapsed_s=time.monotonic() - start,
        compiler=subprocess.check_output([FC, "--version"], text=True).splitlines()[0])
    out.write_text(json.dumps(result, indent=1) + "\n")
    o = out_vals["lw"]
    print({k: (int(np.count_nonzero(o[k]))) for k in o}, result["lines"])


if __name__ == "__main__":
    main()
