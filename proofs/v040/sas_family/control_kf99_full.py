"""Original old-KF native-FMA control with the exact original generic driver object."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

reference=Path(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir(parents=True,exist_ok=True)
b=work/'build';b.mkdir(exist_ok=True);tmp=work/'tmp';tmp.mkdir(exist_ok=True)
env=dict(os.environ,TMPDIR=str(tmp));ref=reference/'build_r4_zero'
for name in ('module_cu_kf.F','module_wrf_error.mod','module_wrf_error.o','kf99_oracle.o'):
    shutil.copyfile(ref/name,b/name)
flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
       '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false',
       '-march=native','-ffp-contract=fast']
c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
for args in (flags+['-c','module_cu_kf.F'],['-O2','-o','oracle','module_wrf_error.o','module_cu_kf.o','kf99_oracle.o']):
    subprocess.run([c]+args,cwd=b,env=env,check=True)
report={'flags':flags,'generic_input_driver_object_sha256':hashlib.sha256((b/'kf99_oracle.o').read_bytes()).hexdigest(),
        'same_external_KFCPS_inputs':'identical generic driver object; only original physics module rebuilt native-FMA',
        'source_sha256':hashlib.sha256((b/'module_cu_kf.F').read_bytes()).hexdigest(),'outputs':{}}
for case in range(1,6):
    raw=subprocess.check_output([str(b/'oracle'),str(case)],cwd=b,env=env)
    path=work/f'case{case}_r4_fma.txt';path.write_bytes(raw)
    report['outputs'][str(case)]=hashlib.sha256(raw).hexdigest()
(work/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
