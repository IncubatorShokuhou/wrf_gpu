"""Manager causal acceptance: fixed/control agreement inside control/CPU spread.

Step1 is reported as a sensitivity witness BEFORE the injection; only step2
compares the causal intervention. Original CPU-WRF remains the truth. The port
injection arm is a causal control, not an independent fidelity reference.
"""
from pathlib import Path
import json
import sys
import numpy as np
import netCDF4

FIELDS=('QSQ','QC_BL','QI_BL','CLDFRA_BL','QKE')
root=Path(sys.argv[1]);result={}
for case,control,ref,stamp,times in (
 ('swiss','<USER_HOME>/wrf_gpu2_lanes/o5-n333/qsqdef/def_injqsq.npz','<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/swiss_1step_tkeF','2023-01-15_',('00:00:18','00:00:36')),
 ('prod',str(root/'prod_control.npz'),'<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/prod_2step_def','2026-07-26_',('00:00:54','00:01:48'))):
    fixed=np.load(root/f'{case}.npz');injected=np.load(control);records=[]
    for step,t in enumerate(times,1):
        with netCDF4.Dataset(Path(ref)/('wrfrst_d01_'+stamp+t)) as ds:
            for field in FIELDS:
                x,y=(data[f's{step}_{field.lower()}'][:,5:-5,5:-5] for data in (fixed,injected))
                z=np.asarray(ds.variables[field][0],float)[:x.shape[0],5:-5,5:-5]
                assert x.shape==y.shape==z.shape
                assert all(np.isfinite(v).all() for v in (x,y,z)),(case,field)
                norm=float(np.sqrt(np.mean(z*z)))
                def rms(a,b):
                    error=float(np.sqrt(np.mean((a-b)**2)))
                    if norm:return error/norm
                    assert error==0,(case,field,'zero-reference mismatch')
                    return 0.
                d,ce,fe=rms(x,y),rms(y,z),rms(x,z)
                records.append({'step':step,'field':field,'fixed_vs_control_rel_rms':d,
                    'control_vs_cpu_rel_rms':ce,'fixed_vs_cpu_rel_rms':fe,
                    'within_control_spread':d<=ce,'binding':step==2})
    before=next(r for r in records if r['step']==1 and r['field']=='QSQ')
    sensitive=before['control_vs_cpu_rel_rms']>.1
    result[case]={'records':records,'control':control,'control_seed_deletion_sensitive':sensitive,
         'passed':sensitive and all(r['within_control_spread'] for r in records if r['binding'])}
report={'passed':all(r['passed'] for r in result.values()),'cases':result,
        'rule':'step2 RMS(fixed-control)<=RMS(control-originalCPUWRF), eachfield; formal1%QCgate unchanged',
        'scope':'CPU interpreted-Pallas control, no GPU qualification'}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for case,data in result.items():
    print(case,data['passed'],data['control_seed_deletion_sensitive'])
    for r in data['records']:
        if r['binding']:print(r)
assert report['passed'],report
