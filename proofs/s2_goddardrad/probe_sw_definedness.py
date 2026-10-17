"""Measure whether original Goddard cloudy SW reads uninitialized locals.

No physics-source changes; visibility-only shim. Compare clear/cloudy with
compiler initialization diagnostics. This is a source-definedness probe, not
a numerical oracle or an authorized source correction.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('wrf',type=Path)
parser.add_argument('build',type=Path)
parser.add_argument('--arm',choices=('normal','zero','one','snan'))
args=parser.parse_args()
wrf,build=args.wrf,args.build
build.mkdir(parents=True,exist_ok=True)
temp=build/'compiler_tmp';temp.mkdir(exist_ok=True)
compiler_env={**os.environ,'TMPDIR':str(temp)}
here=Path(__file__).resolve().parent
src=wrf/'phys/module_ra_goddard.F';text=src.read_text()
marker='  public :: init_goddardrad, goddardrad'
assert text.count(marker)==1
shim=text.replace(marker,'  public :: swrad, fast_overcast\n'+marker)
shutil.copy2(wrf/'phys/module_checkerror.F',build)
shutil.copy2(here.parent/'v013/oracle/radiation_lw/goddard_lw_stubs.f90',build/'stubs.f90')
shutil.copy2(here/'sw_definedness_driver.f90',build/'driver.f90')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
base=['-O2','-g','-ffree-form','-ffree-line-length-none','-cpp','-DRWORDSIZE=4','-DNO_IEEE_MODULE',
      '-fcheck=bounds','-ffpe-summary=none','-fallow-argument-mismatch','-std=legacy']
results=[]
for name,extra in [('normal',[]),('zero',['-finit-real=zero']),('one',[]),('snan',['-finit-real=snan','-ffpe-trap=invalid'])]:
    if args.arm is not None and name!=args.arm:
        continue
    arm_source=shim
    if name=='one':
        # Sensitivity only: define the otherwise undefined local, without
        # changing any already-defined value. NEVER a pristine oracle arm.
        anchor='swh(ic,1)=0.'
        assert shim.count(anchor)==1
        arm_source=shim.replace(anchor,'taux(ic,:)=1._fp_kind ! diagnostic sensitivity only\n   '+anchor)
    (build/'module_ra_goddard.F').write_text(arm_source)
    cmd=[fc,*base,*extra,'stubs.f90','module_checkerror.F','module_ra_goddard.F','driver.f90','-o',name]
    subprocess.run(cmd,cwd=build,env=compiler_env,check=True)
    for cloudy in (0,1):
        p=subprocess.run([str(build/name),str(cloudy)],cwd=build,text=True,capture_output=True)
        stem=f'{name}_cloud{cloudy}'
        (build/f'{stem}.txt').write_text(p.stdout+p.stderr)
        results.append({'arm':stem,'returncode':p.returncode,'command':cmd,
                        'compiled_source_sha256':hashlib.sha256(arm_source.encode()).hexdigest(),
                        'source_change':'diagnostic taux=1 assignment' if name=='one' else 'visibility only',
                        'output_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),
                        'nonfinite_tokens':any(x in p.stdout.lower() for x in ('nan','inf'))})
manifest={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'driver_sha256':hashlib.sha256((here/'sw_definedness_driver.f90').read_bytes()).hexdigest(),
          'physics_changes':'one arm is an explicit taux=1 sensitivity; others visibility only',
          'affinity':sorted(os.sched_getaffinity(0)), 'cache':'NONE', 'arms':results}
(build/('manifest.json' if args.arm is None else f'manifest_{args.arm}.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
