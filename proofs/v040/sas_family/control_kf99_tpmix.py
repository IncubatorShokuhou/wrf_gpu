"""Original TPMIX on identical observed native inputs, generic vs native FMA.

The generic call must reproduce every staged primary output BYTE first. This
isolates compiler spread from changed upstream trajectories in a full KFPARA run.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np

reference=Path(sys.argv[1]);checkpoints=Path(sys.argv[2]);work=Path(sys.argv[3])
work.mkdir(parents=True,exist_ok=True);tmp=work/'tmp';tmp.mkdir(exist_ok=True)
env=dict(os.environ,TMPDIR=str(tmp))
records=json.loads((checkpoints/'helpers_r4.json').read_text())['TPMIX']
x=np.array([r['input'] for r in records],np.float32)
expected=np.array([r['output'] for r in records],np.float32)
program='''program control
use module_cu_kf, only: tpmix
implicit none
real,parameter :: rd=287.,rv=461.6,ep2=rd/rv,sv1=.6112,sv2=17.67,sv3=29.65,svt=273.15
real,parameter :: al=sv1*1000.,bl=sv2,cl=sv2*svt,dl=sv3,ai=613.2,bi=22.452,ci=6133.,di=.61
real,parameter :: lv0=3.15e6,lv1=2370.,ls0=2.905e6,ls1=259.532
real :: p,th,t,q,ql,qi,r,rl,qnl,qni
integer :: n,i
read(*,*) n
do i=1,n
read(*,*) p,th,t,q,ql,qi,r,rl
qnl=0.;qni=0.
call tpmix(p,th,t,q,ql,qi,qnl,qni,r,rl,lv0,lv1,ls0,ls1,ep2,al,bl,cl,dl,ai,bi,ci,di)
write(*,'(6ES27.17E3)') t,q,ql,qi,qnl,qni
end do
end program
'''
stdin=(str(len(x))+'\n'+'\n'.join(' '.join(format(float(v),'.17g') for v in row) for row in x)+'\n').encode()
(work/'input.txt').write_bytes(stdin);(work/'control.F90').write_text(program)
report={'input_sha256':hashlib.sha256(stdin).hexdigest(),'same_actual_inputs':True,'arms':{}}
for mode in ('generic','fma'):
    b=work/mode;b.mkdir(exist_ok=True)
    source=(reference/'build_r4_zero/module_cu_kf.F').read_bytes()
    (b/'module_cu_kf.F').write_bytes(source)
    for suffix in ('.mod','.o'):
        shutil.copyfile(reference/('build_r4_zero/module_wrf_error'+suffix),b/('module_wrf_error'+suffix))
    (b/'control.F90').write_text(program)
    flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
           '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    if mode=='fma':flags+=['-march=native','-ffp-contract=fast']
    c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    for args in (flags+['-c','module_cu_kf.F'],flags+['-c','control.F90'],
                 ['-O2','-o','control','module_wrf_error.o','module_cu_kf.o','control.o']):
        subprocess.run([c]+args,cwd=b,env=env,check=True)
    raw=subprocess.check_output([str(b/'control')],input=stdin,cwd=b,env=env)
    (work/f'{mode}.txt').write_bytes(raw)
    values=[]
    for line in raw.decode().splitlines():
        if not line.lstrip().startswith('*****'):values+=np.fromstring(line,sep=' ').tolist()
    actual=np.array(values,dtype=np.float32).reshape(expected.shape)
    np.savez_compressed(work/f'{mode}.npz',actual=actual,input=x)
    byte=actual.tobytes()==expected.tobytes()
    if mode=='generic':assert byte,'direct original call differs from staged original primary'
    report['arms'][mode]={'flags':flags,'primary_byte':byte,'source_sha256':hashlib.sha256(source).hexdigest(),
                         'spread_abs_by_output':np.max(np.abs(actual.astype(np.float64)-expected),axis=0).tolist()}
    (work/'manifest.json').write_text(json.dumps(report,indent=2));print(mode,report['arms'][mode],flush=True)
