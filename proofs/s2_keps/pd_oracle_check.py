"""Existing pristine WRF PD and RK oracles on actual captured QC operands.

Use the previously registered C05 PD gate: ring>=1 max error bounded by
max(8*WRF FMA/no-FMA spread,1e-5*max|reference|), RMS by
max(8*spread RMS,1e-6*max|reference|). A QC event is source-justified here only
when unchanged WRF also returns negative PD tendency AND negative RK QC at
the same interior source point, and the candidate lies within WRF's own
pointwise FMA/no-FMA interval. This is not a forecast/D6 qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tree',type=Path)
    ap.add_argument('receipt',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--build',required=True,type=Path)
    a=ap.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    sys.path.insert(0,str(a.tree/'tests/v025/b_core'))
    import pd_scalar_oracle as pd
    import update_scalar_oracle as rk
    receipt=json.loads(a.receipt.read_text());arm=next(iter(receipt['arms'].values()))
    packet=arm['qc_packet'];header,values=packet['header'],packet['values']
    path=Path(arm['pd_arrays_path'])
    if not path.is_absolute(): path=a.tree/path
    assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['pd_arrays_sha256']
    arrays=dict(np.load(path,allow_pickle=False))
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in arrays.values())
    shape=arrays['q'].shape;point=np.unravel_index(header['flat_index'],shape)
    assert header['horizontal_ring0']==0 and header['own_step']==header['rk_stage']==3
    assert values['pd_preloaded_qc']==float(arrays['qold'][point])
    assert values['native_pd_tendency']==float(arrays['native_tendency'][point])
    assert values['before_flow_qc']==float(arrays['before_flow_qc'][point])
    result={'scope':__doc__,'source_tree':str(a.tree),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.tree,text=True).strip(),
            'point':list(map(int,point)),
            'array_sha256':arm['pd_arrays_sha256'],'observed_guards':arm['guards'],
            'candidate_pd_tendency':float(arrays['native_tendency'][point]),
            'candidate_qc':float(arrays['before_flow_qc'][point]),'references':{}}
    outputs={}
    for mode in ('off','fast'):
        pdlib=pd.build(a.build/('pd_'+mode),mode)
        rklib=rk.build(a.build/('rk_'+mode),mode)
        reference=pd.evaluate(pdlib,arrays,float(arrays['rdx']),float(arrays['rdy']),float(arrays['dt']))
        updated=rk.evaluate(rklib,dict(s1=arrays['qold'],tend=reference,sct=np.zeros_like(reference),
            mx=arrays['mx'],my=arrays['my'],c1=arrays['c1'],c2=arrays['c2'],
            muold=arrays['muold'],munew=arrays['update_mu'],mub=arrays['mub']),
            3,float(arrays['dt']),header['spec_zone'])
        assert np.isfinite(reference).all() and np.isfinite(updated).all()
        outputs[mode]=reference.astype(float)
        result['references'][mode]={'pd_tendency':float(reference[point]),'qc':float(updated[point]),
            'pd_manifest':json.loads((a.build/('pd_'+mode)/'manifest.json').read_text()),
            'rk_manifest':json.loads((a.build/('rk_'+mode)/'manifest.json').read_text())}
    region=(slice(None),slice(1,-1),slice(1,-1))
    error=(arrays['native_tendency'].astype(float)-outputs['off'])[region]
    spread=(outputs['fast']-outputs['off'])[region]
    scale=float(np.max(np.abs(outputs['off'][region])))
    maximum=float(np.max(np.abs(error)));rms=float(np.sqrt(np.mean(error**2)))
    maximum_bound=max(8*float(np.max(np.abs(spread))),1.e-5*scale)
    rms_bound=max(8*float(np.sqrt(np.mean(spread**2))),1.e-6*scale)
    result['frozen_pd_gate']={'max_error':maximum,'max_bound':maximum_bound,
        'rms_error':rms,'rms_bound':rms_bound,'passed':maximum<=maximum_bound and rms<=rms_bound}
    result['wrf_reproduces_negative']=any(r['pd_tendency']<0 and r['qc']<0 for r in result['references'].values())
    result['inside_wrf_compiler_interval']=all(
        min(r[name] for r in result['references'].values())<=result[candidate]<=
        max(r[name] for r in result['references'].values())
        for name,candidate in (('pd_tendency','candidate_pd_tendency'),('qc','candidate_qc')))
    result['status']='PASS' if (result['frozen_pd_gate']['passed'] and result['wrf_reproduces_negative']
                               and result['inside_wrf_compiler_interval']) else 'OPEN'
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['status']=='PASS','pristine PD/RK event justification remains open'


if __name__=='__main__':main()
