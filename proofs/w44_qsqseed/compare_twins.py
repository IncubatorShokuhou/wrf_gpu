"""Frozen ring>=5 two-step checks against ORIGINAL CPU-WRF restart files."""
from pathlib import Path
import json
import sys
import netCDF4
import numpy as np

FIELDS=('QSQ','QC_BL','QI_BL','CLDFRA_BL','QKE','EL_PBL')

def compare(port,truths):
    p=np.load(port); records=[]
    for step,path in enumerate(truths,1):
        with netCDF4.Dataset(path) as ds:
            for field in FIELDS:
                a=p[f's{step}_{field.lower()}'].astype(float)
                b=np.asarray(ds.variables[field][0],float)[:a.shape[0]]
                assert a.shape==b.shape,(field,a.shape,b.shape)
                assert np.isfinite(a).all() and np.isfinite(b).all(),field
                a,b=a[:,5:-5,5:-5],b[:,5:-5,5:-5]
                rms=float(np.sqrt(np.mean(b*b)));error=float(np.sqrt(np.mean((a-b)**2)))
                rel=error/rms if rms else (0. if not error else float('inf'))
                # EL is a reported diagnostic; approved full-run gate names
                # QSQ/cloud leaves and QKE control (EL has its operator gate).
                limit=.002 if field=='QKE' else .01
                records.append({'step':step,'field':field,'rel_rms':rel,'rms_wrf':rms,
                    'rms_port':float(np.sqrt(np.mean(a*a))),'sum_wrf':float(b.sum()),'sum_port':float(a.sum()),
                    'max_abs':float(abs(a-b).max()),'limit':limit,'binding':field!='EL_PBL',
                    'passed':bool(rel<=limit)})
    return {'passed':all(r['passed'] for r in records if r['binding']),'records':records,
            'reference':'original CPU-WRF, default configuration','port':str(port),'truths':list(map(str,truths))}

if __name__=='__main__':
    root=Path(sys.argv[1]);reports={}
    for case,ref,stamp,times in (
      ('swiss','<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/swiss_1step_tkeF','2023-01-15_',('00:00:18','00:00:36')),
      ('prod','<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/prod_2step_def','2026-07-26_',('00:00:54','00:01:48'))):
        reports[case]=compare(root/f'{case}.npz',[Path(ref)/('wrfrst_d01_'+stamp+t) for t in times])
    report={'passed':all(r['passed'] for r in reports.values()),'cases':reports,'scope':'CPU native interpreted-Pallas two-step fidelity, no GPU claim'}
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for case,r in reports.items():
        print(case,r['passed'],[(x['step'],x['field'],round(x['rel_rms'],7)) for x in r['records']])
    assert report['passed'],report
