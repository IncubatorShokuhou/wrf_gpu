"""G03: source-origin intervention on G01 real operands, not a forecast rerun.

Two preregistered input interventions: erase entry graupel; then erase both
entry graupel and rain/number. All other real operands stay byte-identical.
With no snow/cloud-ice anywhere in these columns, surviving born graupel tests
the rain-freezing source; it cannot come from advected entry graupel.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G03')

def main():
    assert os.environ['JAX_PLATFORMS']=='cpu'
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    apply_fast_path_defaults()
    spec=importlib.util.spec_from_file_location('real_column_replay',ROOT/'proofs/v043/w43-conv/thompson/run_oracle.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    ref.DRV=Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G02/driver')
    ref.EXE=ref.DRV/'thompson_column_driver.exe'
    source=dict(np.load('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G01/columns.npz'))
    assert np.count_nonzero(source['in_qi'])==0 and np.count_nonzero(source['in_qs'])==0
    z={k:np.concatenate((v,v)) for k,v in source.items()}
    n=len(source['names'])
    z['in_qg'][:]=0
    z['in_qr'][n:]=0
    z['in_nr'][n:]=0
    idx=np.arange(2*n);OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    wrf=ref.run_wrf(z,idx,6.,'g03')
    port=ref.run_port(z,idx,6.)
    np.savez_compressed(OUT/'controls.npz',**{f'wrf_{k}':v for k,v in wrf.items()},
                       **{f'port_{k}':v for k,v in port.items()})
    rows=[]
    for i in range(2*n):
        a,b=wrf['qg'][i],port['qg'][i]
        rows.append(dict(name=str(z['names'][i]),control='no_qg' if i<n else 'no_qg_rain',
            wrf_positive=int(np.count_nonzero(a)),port_positive=int(np.count_nonzero(b)),
            zero_mismatches=int(np.count_nonzero((a>0)!=(b>0))),
            wrf_max=float(a.max()),port_max=float(b.max()),max_abs=float(np.max(np.abs(a-b)))))
    summary=dict(source_commit=os.environ['GRAUPEL_COMMIT'],backend='cpu',interpret=True,dt=6.,
        rows=rows,elapsed_s=time.monotonic()-start,original_qi_qs_cells=0)
    (OUT/'controls.json').write_text(json.dumps(summary,indent=2)+'\n')
    for kind in ('no_qg','no_qg_rain'):
        rr=[r for r in rows if r['control']==kind]
        print(kind,'positive WRF/port',sum(r['wrf_positive'] for r in rr),sum(r['port_positive'] for r in rr),
            'zero mismatches',sum(r['zero_mismatches'] for r in rr),'maxabs',max(r['max_abs'] for r in rr),flush=True)
    print('elapsed',summary['elapsed_s'],flush=True)

if __name__=='__main__':main()
