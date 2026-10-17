"""Public-driver oracle, original source bytes and caller REAL4 argument map."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('wrf',type=Path);parser.add_argument('build',type=Path);parser.add_argument('destination',type=Path)
parser.add_argument('--fma',action='store_true',help='original source with native ISA/FMA contraction for compiler-spread diagnostic')
args=parser.parse_args();wrf,build,destination=args.wrf,args.build,args.destination
build.mkdir(parents=True,exist_ok=True);destination.mkdir(parents=True,exist_ok=True)
temp=build/'compiler_tmp';temp.mkdir(exist_ok=True)
here=Path(__file__).resolve().parent
src=wrf/'phys/module_ra_goddard.F'
shutil.copy2(src,build);shutil.copy2(wrf/'phys/module_checkerror.F',build)
shutil.copy2(wrf/'run/BROADBAND_CLOUD_GODDARD.bin',build)
shutil.copy2(here.parent/'v013/oracle/radiation_lw/goddard_lw_stubs.f90',build/'stubs.f90')
shutil.copy2(here/'wrapper_driver.f90',build/'driver.f90')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DRWORDSIZE=4','-DNO_IEEE_MODULE',
       '-fcheck=bounds','-ffpe-summary=none','-fallow-argument-mismatch','-std=legacy','-finit-real=zero',
       '-finit-integer=0','-finit-logical=false',
       '-fconvert=big-endian','-frecord-marker=4']
if args.fma:flags+=['-march=native','-ffp-contract=fast']
cmd=[fc,*flags,'stubs.f90','module_checkerror.F','module_ra_goddard.F','driver.f90','-o','wrapper']
subprocess.run(cmd,cwd=build,env={**os.environ,'TMPDIR':str(temp)},check=True)
arrays={}
for case in range(1,5):
    p=subprocess.run([str(build/'wrapper'),str(case)],cwd=build,text=True,capture_output=True,check=True)
    (build/f'wrapper_{case}.txt').write_text(p.stdout)
    items={key:[] for key in ('input','interface','scalar','lwflux','swflux','swextra','lw','sw')}
    for line in p.stdout.splitlines():
        words=line.split()
        if not words or words[0] not in items:continue
        name=words[0];start=2 if name in ('input','interface','lw','sw') else 1
        items[name].append([float(x) for x in words[start:]])
    for key,rows in items.items():
        value=np.asarray(rows,np.float32)
        if key in ('scalar','lwflux','swflux','swextra'):value=value[0]
        if key in ('lw','sw'):value=value[:,0]
        assert np.isfinite(value).all(),(case,key)
        arrays[f'{case}_{key}']=value
out=destination/'wrapper_real4.npz';np.savez_compressed(out,**arrays)
manifest={'schema':'wrf-goddard-public-wrapper-real4-v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'driver_sha256':hashlib.sha256((here/'wrapper_driver.f90').read_bytes()).hexdigest(),'command':cmd,
          'reference_policy':'manager-approved -finit-real=zero for upstream taux; otherwise original bytes',
          'source_changes':'NONE; exact public goddardrad caller; LUT initialized via public init_goddardrad',
          'cache':'NONE','affinity':sorted(os.sched_getaffinity(0)),
          'tolerance':{'rate_rtol':5e-6,'rate_atol':1e-10,'flux_rtol':1e-6,'flux_atol':5e-4},
          'savepoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(destination/'wrapper_real4.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
