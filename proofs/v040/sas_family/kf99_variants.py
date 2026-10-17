"""Pristine old-KF phase flags and NCA boundary witnesses; original bodies unchanged."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from probe_ksas14 import parse

reference=Path(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir(parents=True,exist_ok=True)
tmp=work/'tmp';tmp.mkdir(exist_ok=True);env=dict(os.environ,TMPDIR=str(tmp))
source=(reference/'build_r8_zero/kf99_oracle.F90').read_text()
anchor='  NCA(1,1) = -100.0        ! allow convection this step'
assert anchor in source
overrides='''
  IF(COMMAND_ARGUMENT_COUNT()>=2) THEN
    CALL GET_COMMAND_ARGUMENT(2,arg)
    READ(arg,*) NCA(1,1)
  END IF
  IF(COMMAND_ARGUMENT_COUNT()>=3) THEN
    CALL GET_COMMAND_ARGUMENT(3,arg)
    SELECT CASE(TRIM(arg))
    CASE('warm')
      warm_rain=.TRUE.;F_QI=.FALSE.;F_QS=.FALSE.
    CASE('wsm3')
      warm_rain=.FALSE.;F_QI=.FALSE.;F_QS=.FALSE.
    CASE('nosnow')
      F_QS=.FALSE.
    CASE('norain')
      F_QR=.FALSE.
    END SELECT
  END IF
'''
source=source.replace(anchor,anchor+overrides)
manifest={'unmodified_physics':True,'arms':{}}
for mode in ('r4','r8'):
    ref=reference/f'build_{mode}_zero';b=work/mode;b.mkdir(exist_ok=True)
    for name in ('module_cu_kf.mod','module_cu_kf.o','module_wrf_error.mod','module_wrf_error.o'):
        shutil.copyfile(ref/name,b/name)
    (b/'variants.F90').write_text(source)
    flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
           '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    if mode=='r8':flags+=['-fdefault-real-8','-fdefault-double-8']
    c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    for args in (flags+['-c','variants.F90'],['-O2','-o','variants','module_wrf_error.o','module_cu_kf.o','variants.o']):
        subprocess.run([c]+args,cwd=b,env=env,check=True)
    for case in range(1,6):
        raw=subprocess.check_output([str(b/'variants'),str(case)],cwd=b,env=env)
        assert raw==(reference/f'case{case}_{mode}_zero.txt').read_bytes(),(mode,case,'baseline not BYTE')
    variants=[('equality',1,27.,''),('skip',1,28.,''),('expired',5,-24.,''),
              ('warm',1,-100.,'warm'),('wsm3',1,-100.,'wsm3'),('nosnow',1,-100.,'nosnow'),('norain',1,-100.,'norain')]
    manifest['arms'][mode]={'baseline_stdout_byte':5,'variants':{}}
    for label,case,nca,flag in variants:
        cmd=[str(b/'variants'),str(case),str(nca)]+([flag] if flag else [])
        raw=subprocess.check_output(cmd,cwd=b,env=env)
        path=work/f'{label}_{mode}.txt';path.write_bytes(raw)
        row=parse(path)
        manifest['arms'][mode]['variants'][label]={'case':case,'nca_input':nca,'phase_flag':flag,
            'output_sha256':hashlib.sha256(raw).hexdigest(),'rain':float(row['RAINCV']),'nca':float(row['NCA']),
            'finite':all(np.isfinite(x).all() for x in row.values())}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2));print(mode,manifest['arms'][mode],flush=True)
