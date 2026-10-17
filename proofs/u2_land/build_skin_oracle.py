"""Compile the unmodified WRF SST-skin module in a private lane build directory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def main():
    source, build, target = map(Path, sys.argv[1:4])
    build.mkdir(parents=True, exist_ok=True)
    (build / "reference.f90").write_bytes(source.read_bytes())
    (build / "error.f90").write_text("module module_wrf_error\nend module\n")
    (build / "driver.f90").write_text('''program oracle
use module_sf_sstskin
implicit none
integer,parameter :: n=12, nsteps=24
integer :: i,k
real :: xl(n,1),glw(n,1),gsw(n,1),hfx(n,1),qfx(n,1),tsk(n,1),ust(n,1),emiss(n,1),sstsk(n,1),dtw(n,1),dt
do i=1,n
read(*,*) xl(i,1),glw(i,1),gsw(i,1),hfx(i,1),qfx(i,1),tsk(i,1),ust(i,1),emiss(i,1),sstsk(i,1),dtw(i,1),dt
enddo
do k=1,nsteps
call sst_skin_update(xl,glw,gsw,hfx,qfx,tsk,ust,emiss,dtw,sstsk,dt,5.67051e-8, &
 1,n,1,1,1,1,1,n,1,1,1,1,1,n,1,1,1,1)
do i=1,n
write(*,'(2(ES25.16E3,1X))') sstsk(i,1),dtw(i,1)
enddo
enddo
end program
''')
    fc = shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    subprocess.run([fc, "-O2", "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off",
                    "error.f90", "reference.f90", "driver.f90", "-o", "oracle.exe"], cwd=build,
                   check=True, timeout=120)
    cases = [
        [1., 350., 500., 20., .00002, 290., .2, .98, 0., 0., 18.],
        [2., 350., 500., 20., .00002, 290., .2, .98, 0., 0., 18.],
        [2., 350., 0., 20., .00002, 290., .2, .98, 290., 0., 18.],
        [2., 350., 800., 20., .00002, 290., .2, .98, 290., 0., 18.],
        [2., 250., 0., 100., .0001, 295., .001, .98, 295., 1., 18.],
        [2., 250., 0., 100., .0001, 295., .5, .98, 295., 1., 18.],
        [2., 450., 1200., -10., -.00001, 300., .01, .98, 300., .01, 18.],
        [2., 300., 20., 1., .000001, 270., .1, .98, 270., 0., 18.],
        [2., 380., 200., 0., 0., 285., 0., .95, 285., .1, 18.],
        [1., 300., 0., 20., .00002, 280., .2, .97, 279., .2, 18.],
        [1.5, 350., 500., 20., .00002, 290., .2, .98, 290., .1, 18.],
        [2., 350., 500., 20., .00002, 290., 1., .98, 290., 0., 18.],
    ]
    result = subprocess.run([str((build / "oracle.exe").resolve())],
                            input="\n".join(" ".join(map(str, row)) for row in cases) + "\n",
                            text=True, capture_output=True, check=True, timeout=30)
    outputs = np.asarray([[float(x) for x in line.split()] for line in result.stdout.splitlines()]).reshape(24, 12, 2)
    assert np.isfinite(outputs).all()
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
                   compiler=subprocess.check_output([fc, "--version"], text=True).splitlines()[0],
                   flags="-O2 -ffp-contract=off; unmodified WRF REAL4 module", created_utc=datetime.now(timezone.utc).isoformat(),
                   variables_in=["xland", "glw", "gsw", "hfx", "qfx", "tsk", "ust", "emiss", "sstsk", "dtw", "dt"],
                   variables_out=["sstsk", "dtw"], inputs=cases, output=outputs.tolist(),
                   tolerance=dict(sstsk_atol=3.1e-5, sstsk_rtol=0., dtw_atol=1.e-7, dtw_rtol=3.e-6),
                   census=dict(columns=12, steps=24, constructed=True,
                               regimes="day/night; warm-layer cooling; weak/strong stress; cold water; land; zero allocator seed"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
