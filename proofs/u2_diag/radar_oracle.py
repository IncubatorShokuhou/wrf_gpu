"""Build/run a lane-local pristine Thompson radar + final-ng oracle.

No JAX imports and no writes to the pristine checkout. The original operator,
radar library and final-ng source block are copied verbatim. Only initialization
of the moments actually read by these routines is staged. Numeric fixtures
contain original WRF history inputs plus independently constructed active
regimes; these are operator comparisons, not coupled-forecast claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def build(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    mp_path = source / "phys/module_mp_thompson.F"
    radar_path = source / "phys/module_mp_radar.F"
    mp = mp_path.read_text()
    header = mp[:mp.lower().index("\n      contains")]
    def routine(start, end):
        a = mp.lower().index(start.lower())
        b = mp.lower().index(end.lower(), a) + len(end)
        return mp[a:b]
    calc = routine("      subroutine calc_refl10cm ", "      end subroutine calc_refl10cm")
    gamma = routine("      REAL FUNCTION GAMMLN(XX)", "      END FUNCTION GAMMLN")
    wgamma = routine("      REAL FUNCTION WGAMMA(y)", "      END FUNCTION WGAMMA")
    begin = mp.index("         ng1d(k) = MAX(R2/rho(k), ng1d(k) + ngten(k)*DT)")
    end = mp.index("\n      enddo\n\n      end subroutine mp_thompson", begin)
    ngblock = mp[begin:end]
    ng = """
subroutine oracle_ng(qg1d,ng1d,rho,nz)
 integer nz,k
 real qg1d(nz),ng1d(nz),rho(nz),ngten(nz),qb1d(nz),qbten(nz),mvd_g(nz),DT
 integer idx_bg(nz)
 double precision lamg
 DT=0.
 ngten=0.
 qbten=0.
 qb1d=qg1d/400.
 do k=1,nz
""" + ngblock + "\n enddo\nend subroutine oracle_ng\n"
    init = """
      subroutine oracle_init
      integer n
      cre(2)=mu_r+1.
      cre(3)=bm_r+mu_r+1.
      cre(4)=bm_r*2.+mu_r+1.
      do n=2,4
       crg(n)=WGAMMA(cre(n))
      enddo
      org2=1./crg(2)
      org3=1./crg(3)
      obmr=1./bm_r
      cse(1)=bm_s+1.
      cse(3)=bm_s*2.
      oams=1./am_s
      obms=1./bm_s
      ocms=oams**obms
      cge(2,:)=mu_g+1.
      cge(3,:)=bm_g+mu_g+1.
      cge(4,:)=bm_g*2.+mu_g+1.
      do n=2,4
       cgg(n,:)=WGAMMA(cge(n,1))
      enddo
      ogg2=1./cgg(2,1)
      ogg3=1./cgg(3,1)
      obmg=1./bm_g
      call radar_init
      end subroutine
"""
    driver = """
program oracle
 use module_mp_thompson
 implicit none
 integer nz,ncol,i,k,v,kdiag,mode
 real,allocatable::a(:,:),z(:)
 read(*,*)mode,ncol,nz,kdiag
 allocate(a(nz,10),z(nz))
 call oracle_init
 do i=1,ncol
  do v=1,10
   read(*,*)a(:,v)
  enddo
  if(mode.eq.0)then
   call calc_refl10cm(a(:,1),a(:,2),a(:,3),a(:,4),a(:,5),a(:,6), &
                     a(:,7),a(:,8),a(:,9),a(:,10),z,1,nz,1,1,kdiag)
  else
   z=a(:,7)
   call oracle_ng(a(:,6),z,a(:,10),nz)
  endif
  do k=1,nz
   if(mode.eq.0) z(k)=max(-35.,z(k))
   write(*,'(ES24.16)')z(k)
  enddo
 enddo
end program
"""
    stub = """
module module_wrf_error
contains
 subroutine wrf_debug(n,s)
 integer n
 character(*)s
 end subroutine
end module
module module_model_constants
 real,parameter::RE_QC_BG=2.5e-6,RE_QI_BG=10e-6,RE_QS_BG=10e-6
end module
"""
    (output / "stub.f90").write_text(stub)
    (output / "radar.F90").write_text(radar_path.read_text())
    (output / "thompson.F90").write_text(header + "\nCONTAINS\n" + init + ng + calc + "\n" + gamma + "\n" + wgamma + "\nend module\n")
    (output / "driver.f90").write_text(driver)
    compiler = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    cmd = [compiler, "-cpp", "-ffree-form", "-ffree-line-length-none", "-O2", "-fno-fast-math", "-ffp-contract=off",
           "stub.f90", "radar.F90", "thompson.F90", "driver.f90", "-o", "oracle"]
    subprocess.run(cmd, cwd=output, check=True, timeout=90)
    return {"sources": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (mp_path, radar_path)},
            "calc_refl10cm_sha256": hashlib.sha256(calc.encode()).hexdigest(),
            "ng_block_sha256": hashlib.sha256(ngblock.encode()).hexdigest(),
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "command": cmd, "scope": "verbatim pristine calc_refl10cm + module_mp_radar + final ng1d block; not a coupled forecast"}


def execute(executable: Path, arrays, *, mode=0, ke_diag=None):
    ncol, _, nz = arrays.shape
    header = f"{mode} {ncol} {nz} {nz if ke_diag is None else ke_diag}\n"
    payload = header + "\n".join(" ".join(format(float(v), ".9e") for v in row) for col in arrays for row in col) + "\n"
    result = subprocess.run([str(executable)], input=payload, capture_output=True, text=True, check=True, timeout=60)
    return np.fromstring(result.stdout, sep=" ").reshape(ncol, nz).astype("f4")


def generate(source: Path, output: Path, reference: Path):
    manifest = build(source, output / "build")
    from netCDF4 import Dataset
    columns, provenance = [], []
    for path in sorted(reference.glob("wrfout_d01_*"))[1:]:
        with Dataset(path) as ds:
            p = np.asarray(ds["P"][:]) + np.asarray(ds["PB"][:])
            temp = (np.asarray(ds["T"][:]) + 300) * (p / 100000) ** np.float32(287 / 1004.5)
            fields = [np.asarray(ds[k][:]) for k in ("QVAPOR", "QCLOUD", "QRAIN", "QNRAIN", "QSNOW", "QGRAUP")]
            fields += [np.zeros_like(fields[0]), np.zeros_like(fields[0]), temp, p]
            candidates = np.argwhere((fields[4][0].max(axis=0) > 0) & (fields[5][0].max(axis=0) <= 1e-6))
            if not len(candidates):
                raise ValueError(f"no original snow columns in {path}")
            for y, x in candidates[np.linspace(0, len(candidates) - 1, 8, dtype=int)]:
                columns.append(np.asarray([f[0, :, y, x] for f in fields], dtype="f4"))
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    nz = columns[0].shape[-1] if columns else 44
    for regime in range(16):
        a = np.zeros((10, nz), dtype="f4")
        a[0] = .005
        a[8], a[9] = np.linspace(285, 240, nz), np.linspace(95000, 30000, nz)
        if regime & 1:
            a[2], a[3] = np.geomspace(1e-8, .003, nz), np.geomspace(1e2, 1e5, nz)
        if regime & 2:
            a[4] = np.geomspace(1e-6, .001, nz)
        if regime & 4:
            a[5], a[6] = np.geomspace(1e-6, .002, nz), np.geomspace(1e1, 1e4, nz)
        if regime & 8:
            a[8] = 255
        columns.append(a)
    a = np.asarray(columns, dtype="f4")
    dbz = execute(output / "build/oracle", a)
    bottom = execute(output / "build/oracle", a, ke_diag=1)
    # Preserve the rho lifetime separately from final T/qv for Ng final balance.
    ng_input = np.zeros((4, 10, nz), dtype="f4")
    ng_input[:, 5] = np.geomspace(1e-13, .03, nz)
    ng_input[:, 6] = np.geomspace(1e-9, 1e8, nz)
    ng_input[:, 9] = np.linspace(.3, 1.3, nz)
    ng_input[1, 6] = .1
    ng_input[2, 6] = 1e8
    ng_input[3, 6] = 0
    balanced_ng = execute(output / "build/oracle", ng_input, mode=1)
    fixture = output / "radar_oracle.npz"
    np.savez_compressed(fixture, inputs=a, dbz=dbz, bottom=bottom,
                        ng_inputs=ng_input[:, (5, 6, 9)], balanced_ng=balanced_ng)
    manifest.update(reference_files=provenance, real_history_field_columns=len(a) - 16, constructed_columns=16,
                    fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    tolerance={"max_absolute_dbz": 5e-5, "ng_max_ulp": 4},
                    note="History-field input columns are not call-site savepoints; active graupel number supplied by constructed operator cases.",
                    census={"rain_levels": int((a[:, 2] > 1e-12).sum()), "snow_levels": int((a[:, 4] > 1e-6).sum()),
                            "graupel_levels": int((a[:, 5] > 1e-6).sum()), "dbz_max": float(dbz.max())})
    (output / "radar_oracle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output, args.reference)
