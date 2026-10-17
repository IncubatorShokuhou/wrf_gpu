"""F24 pristine-statement oracle: WRF land-surface precipitation forcing for Noah-classic and PX.

Extracts the UNTOUCHED statements (assert present verbatim) and evaluates them in REAL4, no FMA:
  module_surface_driver.F:1568-1570  RAINBL = RAINBL + RAINCV + RAINNCV ; RAINBL = MAX(RAINBL, 0.0)
  module_sf_noahdrv.F:835            PRCP = RAINBL(i,j)/DT
  module_sf_noahdrv.F:859-866        IF(FRPCPN) FFROZP = SR(I,J)  (FRPCPN=.true.: SR present, surface_driver :2708)
  module_sf_pxlsm.F:482              PRECIP = MAX(1.0E-3*RAINBL(i,j)/DTBL, 0.0)
RAINBL starts at 0 each step (reset after the LSM, surface_driver :4539; bldt=0 -> one step).
usage: python build_f24_statement_oracle.py <out.json>
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF/phys")
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
STATEMENTS = {
    "module_surface_driver.F": ["RAINBL(i,j) = RAINBL(i,j) + RAINCV(i,j) + RAINNCV(i,j)",
                                "RAINBL(i,j) = MAX (RAINBL(i,j), 0.0)"],
    "module_sf_noahdrv.F": ["PRCP=RAINBL(i,j)/DT", "FFROZP=SR(I,J)"],
    "module_sf_pxlsm.F": ["PRECIP = MAX ( 1.0E-3*RAINBL(i,j)/DTBL,0.0)"],
}
RAINCV = [0., 1.e-6, 3.7e-4, .0123, .36, 2.5]
RAINNCV = [0., 2.e-7, 5.5e-4, .021, .91, 4.25]
SR = [0., .125, .5, 1.]
DT = [18., 54.]


def main():
    out = Path(sys.argv[1])
    digests = {}
    for name, lines in STATEMENTS.items():
        text = (WRF / name).read_text()
        for line in lines:
            assert line in text, (name, line)
        digests[name] = hashlib.sha256(text.encode()).hexdigest()
    rows = [(c, n, s, d) for c in RAINCV for n in RAINNCV for s in SR for d in DT]
    data = "\n".join(f"{c!r} {n!r} {s!r} {d!r}" for c, n, s, d in rows)
    program = f"""program probe
implicit none
real :: rainbl(1,1), raincv(1,1), rainncv(1,1), sr(1,1), dt, dtbl, prcp, ffrozp, precip
logical :: frpcpn
integer :: i, j, n
i = 1; j = 1; frpcpn = .true.
do n = 1, {len(rows)}
  read(*, *) raincv(1,1), rainncv(1,1), sr(1,1), dt
  dtbl = dt
  rainbl(1,1) = 0.
  {STATEMENTS['module_surface_driver.F'][0]}
  {STATEMENTS['module_surface_driver.F'][1]}
  {STATEMENTS['module_sf_noahdrv.F'][0]}
  IF (FRPCPN) THEN
    {STATEMENTS['module_sf_noahdrv.F'][1]}
  ENDIF
  {STATEMENTS['module_sf_pxlsm.F'][0]}
  write(*, '(3(Z8.8,1X))') prcp, ffrozp, precip
end do
end program probe
"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.f90"
        src.write_text(program)
        subprocess.run([FC, "-O0", "-ffp-contract=off", "-ffree-line-length-none", str(src), "-o",
                        str(Path(tmp) / "probe")], check=True)
        result = subprocess.run([str(Path(tmp) / "probe")], input=data + "\n", capture_output=True,
                                text=True, check=True).stdout.split()
    bits = np.array([int(x, 16) for x in result], np.uint32).reshape(len(rows), 3)
    values = bits.view(np.float32)
    payload = dict(reference="untouched pristine WRF 4.7.1 statements, REAL4, -O0 no FMA", sources=digests,
                   statements=STATEMENTS, program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                   inputs=[dict(raincv=c, rainncv=n, sr=s, dt=d) for c, n, s, d in rows],
                   prcp=values[:, 0].tolist(), ffrozp=values[:, 1].tolist(), precip=values[:, 2].tolist())
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(len(rows), "rows; max prcp", float(values[:, 0].max()))


if __name__ == "__main__":
    main()
