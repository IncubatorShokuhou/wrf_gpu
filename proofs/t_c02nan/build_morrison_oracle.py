"""Pristine REAL4 and diagnostic REAL8 Morrison on the exact failing column."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
import numpy as np

root=Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN54_morrison_oracle')
tree=Path(__file__).resolve().parents[2]
wrf=Path('<USER_HOME>/src/wrf_pristine/WRF')
inp=Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN51_c08_final_window/first_bad_column_inputs.npz')
z=np.load(inp)
names=('th','qv','qc','qr','qi','qs','qg','ni','ns','nr','ng','pii','p','dz','w')
text='\n'.join([format(float(z['dt']),'.25e')]+[' '.join(format(float(x),'.25e') for x in z[k]) for k in names])+'\n'
(root/'column.txt').write_text(text)
driver=(tree/'proofs/v060/oracle/morrison_oracle_driver.f90').read_text()
driver=driver.replace('KX  = 40','KX  = 44').replace('INTEGER :: k, case_id, itimestep','INTEGER :: k, case_id, itimestep, input_unit')
old='''  CALL build_column(case_id, TH, QV, QC, QR, QI, QS, QG, NI, NS, NR, NG, &
                     RHO, PII, P, DZ, W)'''
load="  OPEN(NEWUNIT=input_unit, FILE='column.txt', STATUS='old', ACTION='read')\n  READ(input_unit,*) DT\n"
for key in names:load+='  READ(input_unit,*) '+key.upper()+'(1,:,1)\n'
load+='  CLOSE(input_unit)\n  RHO = P / (R_D * TH * PII)'
assert old in driver
driver=driver.replace(old,load).replace('itimestep = 1','itimestep = 36').replace('ES23.15','ES26.18')
sources={'module_model_constants.F':wrf/'share/module_model_constants.F',
         'module_mp_radar.F':wrf/'phys/module_mp_radar.F',
         'module_mp_morr_two_moment.F':wrf/'phys/module_mp_morr_two_moment.F',
         'morrison_stub_modules.f90':tree/'proofs/v060/oracle/morrison_stub_modules.f90'}
compiler='<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran'
receipt=dict(physics_untouched=True,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),sources={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()},driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),modes={})
start=time.monotonic()
for mode in ('fp32','fp64'):
    build=root/mode;build.mkdir(exist_ok=True)
    for name,p in sources.items():shutil.copyfile(p,build/name)
    (build/'morrison_oracle_driver.f90').write_text(driver)
    scalar_dtype=np.float32 if mode=='fp32' else np.float64
    exact_text='\n'.join([format(float(scalar_dtype(z['dt'])),'.25e')]+[' '.join(format(float(x),'.25e') for x in np.asarray(z[k],scalar_dtype)) for k in names])+'\n'
    (build/'column.txt').write_text(exact_text)
    flags=['-O2','-ffree-form','-ffree-line-length-none','-ffpe-summary=none','-cpp','-ffp-contract=off']
    if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
    log=[]
    for name in ('morrison_stub_modules.f90','module_model_constants.F','module_mp_radar.F','module_mp_morr_two_moment.F','morrison_oracle_driver.f90'):
        cmd=[compiler,*flags,'-c',name];v=subprocess.run(cmd,cwd=build,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.append(v.stdout);assert v.returncode==0,(cmd,v.stdout)
    cmd=[compiler,*flags,'-o','oracle',*[(Path(k).stem+'.o') for k in (*sources,'morrison_oracle_driver.f90')]]
    v=subprocess.run(cmd,cwd=build,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.append(v.stdout);assert v.returncode==0,(cmd,v.stdout)
    (root/(mode+'_build.log')).write_text(''.join(log))
    outputs=[subprocess.check_output([str(build/'oracle'),'1'],cwd=build,text=True) for _ in range(2)]
    assert outputs[0]==outputs[1],mode+' reference not repeatable'
    (root/(mode+'_original.txt')).write_text(outputs[0]);columns={};scalars={}
    for line in outputs[0].splitlines():
        if '=' not in line:continue
        key,value=line.split('=',1)
        if '[' in key:
            key,index=key.split('[',1);index=int(index[:-1])-1
            columns.setdefault(key,[None]*44)[index]=float(value)
        else:
            try:scalars[key]=float(value)
            except ValueError:pass
    dtype=np.float32 if mode=='fp32' else np.float64
    input_keys=('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','NI_IN','NS_IN','NR_IN','NG_IN','PII','P','DZ','W')
    inputs_byte={key:np.asarray(columns[key],dtype).tobytes()==np.asarray(z[name],dtype).tobytes() for key,name in zip(input_keys,names)}
    bad={key:int((~np.isfinite(np.asarray(values))).sum()) for key,values in columns.items() if not np.isfinite(values).all()}
    assert all(inputs_byte.values()), inputs_byte
    result=dict(columns=columns,scalars=scalars,inputs_BYTE=inputs_byte,nonfinite=bad,repeat_BYTE=True,
                original_executable_sha256=hashlib.sha256((build/'oracle').read_bytes()).hexdigest(),flags=flags)
    (root/(mode+'_original.json')).write_text(json.dumps(result,indent=2)+'\n')
    receipt['modes'][mode]={key:result[key] for key in ('inputs_BYTE','nonfinite','repeat_BYTE','flags','original_executable_sha256')}
    print(mode,json.dumps(receipt['modes'][mode]),flush=True)
receipt.update(seconds=time.monotonic()-start,peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,finished_utc=datetime.now(timezone.utc).isoformat())
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
