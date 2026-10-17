"""Pristine mixed-precision lwrad oracle; original REAL4 output interface.

Build privately, without -fdefault-real-8. The prior driver supplies DOUBLE
column operands; its G/CP values are corrected to WRF model constants. This is
solver-only evidence; the goddardrad caller still needs its own full oracle.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

wrf,build,destination=map(Path,sys.argv[1:4])
build.mkdir(parents=True,exist_ok=True)
temp=build/'compiler_tmp';temp.mkdir(exist_ok=True)
compiler_env={**os.environ,'TMPDIR':str(temp)}
destination.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
src=wrf/'phys/module_ra_goddard.F'
text=src.read_text()
marker='  public :: init_goddardrad, goddardrad'
assert text.count(marker)==1
(build/'module_ra_goddard.F').write_text(text.replace(marker,'  public :: lwrad\n'+marker))
shutil.copy2(wrf/'phys/module_checkerror.F',build)
old=here.parent/'v013/oracle/radiation_lw'
shutil.copy2(old/'goddard_lw_stubs.f90',build/'stubs.f90')
driver=(old/'goddard_lw_oracle_driver.f90').read_text()
driver=driver.replace('G  = 9.80665_DP','G  = REAL(9.81,DP)').replace('CP = 1004.0_DP','CP = REAL(1004.5,DP)')
(build/'driver.f90').write_text(driver)
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DRWORDSIZE=4','-DNO_IEEE_MODULE',
       '-fcheck=bounds','-ffpe-summary=none','-fallow-argument-mismatch','-std=legacy']
cmd=[fc,*flags,'stubs.f90','module_checkerror.F','module_ra_goddard.F','driver.f90','-o','lw']
subprocess.run(cmd,cwd=build,env=compiler_env,check=True)
parse=runpy.run_path(str(old/'goddard_lw_dump_to_json.py'))['main']
for case in range(1,7):
    raw=build/f'lw_case_{case}.txt'
    with raw.open('w') as f:subprocess.run([str(build/'lw'),str(case)],cwd=build,stdout=f,check=True)
    parse(str(raw),str(destination/f'lw_case_{case}.json'))
manifest={'schema':'wrf-goddard-lwrad-native-v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'shim_sha256':hashlib.sha256((build/'module_ra_goddard.F').read_bytes()).hexdigest(),
          'driver_sha256':hashlib.sha256(driver.encode()).hexdigest(),'command':cmd,
          'precision':'original default REAL4; explicit fp_kind REAL8; output REAL4',
          'constants':'module_model_constants: g=REAL4(9.81), cp=7*287/2=1004.5',
          'affinity':sorted(os.sched_getaffinity(0)),'cache':'NONE','physics_changes':'visibility only',
          'tolerance':{'rtol':5e-7,'atol':3e-5},
          'savepoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('lw_case_*.json')}}
(destination/'lw_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
