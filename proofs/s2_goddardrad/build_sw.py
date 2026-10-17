"""Pristine WRF SW oracle under manager-approved compiler-defined zero init."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

wrf,build,destination=map(Path,sys.argv[1:4])
build.mkdir(parents=True,exist_ok=True);destination.mkdir(parents=True,exist_ok=True)
temp=build/'compiler_tmp';temp.mkdir(exist_ok=True)
here=Path(__file__).resolve().parent
src=wrf/'phys/module_ra_goddard.F';text=src.read_text()
marker='  public :: init_goddardrad, goddardrad'
assert text.count(marker)==1
shim=text.replace(marker,'  public :: swrad, fast_overcast\n'+marker)
(build/'module_ra_goddard.F').write_text(shim)
shutil.copy2(wrf/'phys/module_checkerror.F',build)
shutil.copy2(here.parent/'v013/oracle/radiation_lw/goddard_lw_stubs.f90',build/'stubs.f90')
shutil.copy2(here/'sw_driver.f90',build/'driver.f90')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DRWORDSIZE=4','-DNO_IEEE_MODULE',
       '-fcheck=bounds','-ffpe-summary=none','-fallow-argument-mismatch','-std=legacy','-finit-real=zero',
       '-finit-integer=0','-finit-logical=false']
cmd=[fc,*flags,'stubs.f90','module_checkerror.F','module_ra_goddard.F','driver.f90','-o','sw']
subprocess.run(cmd,cwd=build,env={**os.environ,'TMPDIR':str(temp)},check=True)
arrays={}
for case in range(1,9):
    p=subprocess.run([str(build/'sw'),str(case)],cwd=build,text=True,capture_output=True,check=True)
    (build/f'sw_case_{case}.txt').write_text(p.stdout)
    layers=[];interfaces=[];optics=[]
    for line in p.stdout.splitlines():
        words=line.split();name=words[0]
        if name=='scalar': scalar=np.asarray(words[1:],np.float64)
        elif name=='surface': surface=np.asarray(words[1:],np.float32)
        elif name=='layer':layers.append([float(x) for x in words[2:]])
        elif name=='interface':interfaces.append([float(x) for x in words[2:]])
        elif name=='optics':optics.append([float(x) for x in words[3:]])
    for key,value in {'scalar':scalar,'surface':surface,'layer':np.asarray(layers),'interface':np.asarray(interfaces),
                      'optics':np.asarray(optics).reshape(40,11,6)}.items():
        assert np.isfinite(value).all(),(case,key)
        arrays[f'{case}_{key}']=value
out=destination/'sw_native.npz';np.savez_compressed(out,**arrays)
manifest={'schema':'wrf-goddard-swrad-native-v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'shim_sha256':hashlib.sha256(shim.encode()).hexdigest(),
          'driver_sha256':hashlib.sha256((here/'sw_driver.f90').read_bytes()).hexdigest(),'command':cmd,
          'precision':'original default REAL4; explicit fp_kind REAL8; output REAL4',
          'reference_policy':'manager-approved -finit-real=zero; port zero only at undefined taux read original5315',
          'source_changes':'visibility only','cache':'NONE','affinity':sorted(os.sched_getaffinity(0)),
          'tolerance':{'rtol':5e-7,'atol':3e-7},'savepoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(destination/'sw_native.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
