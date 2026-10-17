"""Compile the verbatim pristine-WRF bucket block into a tiny REAL oracle.

Run on the assigned CPU core, with QUIET guard, in an immutable checkout:
  python proofs/u2_diag/generate_precipitation_oracle.py <lane-build-dir>
Creates the small tracked fixture here. Never writes to pristine WRF or case
directories. This is an operator oracle, not coupled-forecast qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/wrf_pristine/WRF/phys/module_diag_misc.F"
BUILD = Path(sys.argv[1]).resolve()
BUILD.mkdir(parents=True, exist_ok=True)
source = SOURCE.read_text()
start = source.index("! Compute precipitation accumulation in a given time window: prec_acc_dt")
end = source.index("   if (diag_print .eq. 0 ) return", start)
block = source[start:end]
wrapper = '''subroutine buckets(prec_acc_c,prec_acc_nc,snow_acc_nc,raincv,rainncv,snowncv,t2,curr_secs2,prec_acc_dt,n)
implicit none
integer,intent(in):: n
real,intent(inout):: prec_acc_c(n,1),prec_acc_nc(n,1),snow_acc_nc(n,1)
real,intent(in):: raincv(n,1),rainncv(n,1),snowncv(n,1),t2(n,1),curr_secs2,prec_acc_dt
integer:: ij,i,j,num_tiles,i_start(1),i_end(1),j_start(1),j_end(1)
num_tiles=1
i_start=1
i_end=n
j_start=1
j_end=1
''' + block + '''end subroutine
program oracle
implicit none
integer,parameter:: n=8
integer:: s,ns
real:: c(n,1),nc(n,1),sn(n,1),rc(n,1),rnc(n,1),snc(n,1),t2(n,1),clock,interval
open(10,file='input.bin',access='stream',form='unformatted',status='old')
open(11,file='output.bin',access='stream',form='unformatted',status='replace')
read(10) ns,interval,c,nc,sn
do s=1,ns
 read(10) clock,rc,rnc,snc,t2
 call buckets(c,nc,sn,rc,rnc,snc,t2,clock,interval,n)
 write(11) c,nc,sn
enddo
close(10)
close(11)
end program
'''
(BUILD / "oracle.f90").write_text(wrapper)
command = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-O0", "-ffp-contract=off", "-fno-fast-math", "-ffree-line-length-none", "oracle.f90", "-o", "oracle"]
subprocess.run(command, cwd=BUILD, check=True, timeout=30)
clocks = np.asarray([0,18,35.999996185302734,36,54,72,90,108,126,144,145,162,180], np.float32)
initial = np.arange(24, dtype=np.float32).reshape(3,8) + 100
rainc = np.array([.18, .1, -.5, 0, 2, 1e-8, .03, -.1], np.float32)
rainnc = np.array([.32, -.1, .1, 0, 1e-4, 7, -.03, 1], np.float32)
snow = np.array([.02, .1, -.9, -.3, 0, .125, .03, 1], np.float32)
t2 = np.array([270,278,270,278,273.15,273.14996,273.15003,260], np.float32)
inputs = np.stack([np.stack((rainc*(1+i%3),rainnc,snow,t2)) for i in range(len(clocks))])
cases = []
for interval in (.6, .7):
    with (BUILD / "input.bin").open("wb") as f:
        f.write(np.int32(len(clocks)).tobytes())
        f.write(np.float32(interval).tobytes())
        f.write(initial.tobytes())
        for clock, operands in zip(clocks,inputs,strict=True):
            f.write(clock.tobytes())
            f.write(operands.tobytes())
    subprocess.run([str(BUILD / "oracle")], cwd=BUILD, check=True, timeout=10)
    expected = np.fromfile(BUILD / "output.bin", np.float32).reshape(-1,3,8)
    assert np.isfinite(expected).all()
    cases.append(dict(interval=float(np.float32(interval)), initial=initial.tolist(), clocks=clocks.tolist(),
                      inputs=inputs.tolist(), expected=expected.tolist()))
fixture = dict(source="WRF/phys/module_diag_misc.F", source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               block_sha256=hashlib.sha256(block.encode()).hexdigest(), compiler=command,
               provenance="verbatim pristine block; eight active regimes, sequential resets; no port code imported",
               units="mm, K, seconds, minutes", precision="WRF REAL=4", tolerance="bit-exact eager; 2 ULP compiled",
               cases=cases)
target = ROOT / "proofs/u2_diag/precipitation_oracle.json"
target.write_text(json.dumps(fixture,indent=1)+"\n")
print(json.dumps(dict(fixture=str(target), source_sha256=fixture['source_sha256'], cases=len(cases),steps=len(clocks),columns=8)))
