"""G02: identical reconstructed real operands through pristine WRF and native mp8.

CPU interpreter is an operator probe, not GPU or coupled-forecast validation.
Uses the established TH08 mp_gt_driver and MIX increment rule unchanged.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G02')
DRV = OUT/'driver'

def main():
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    defaults = apply_fast_path_defaults()
    spec = importlib.util.spec_from_file_location('real_column_replay', ROOT/'proofs/v043/w43-conv/thompson/run_oracle.py')
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    ref.DRV, ref.EXE = DRV, DRV/'thompson_column_driver.exe'
    z = dict(np.load('<USER_HOME>/wrf_gpu2_lanes/v45-graupel/G01/columns.npz'))
    idx = np.arange(len(z['names']))
    started = time.monotonic()
    wrf = ref.run_wrf(z, idx, 6., 'g02')
    print('pristine driver done',time.monotonic()-started,flush=True)
    port = ref.run_port(z, idx, 6.)
    np.savez_compressed(OUT/'replay.npz',**{f'wrf_{k}':v for k,v in wrf.items()},
                       **{f'port_{k}':v for k,v in port.items()})
    rows=[]
    for i,name in enumerate(z['names']):
        a,b = wrf['qg'][i],port['qg'][i]
        rows.append(dict(name=str(name),input_positive=int(np.count_nonzero(z['in_qg'][i])),
            wrf_positive=int(np.count_nonzero(a)),port_positive=int(np.count_nonzero(b)),
            zero_pattern_mismatches=int(np.count_nonzero((a>0)!=(b>0))),
            wrf_max=float(a.max()),port_max=float(b.max()),max_abs=float(np.max(np.abs(a-b)))))
    gate={}
    for f in ref.ORDER_OUT:
        a,b,base = wrf[f],port[f],z[f'in_{f}']
        dw = a.astype(np.float64)-base
        dp = b.astype(np.float64)-base
        err=np.max(np.abs(dp-dw),axis=1)
        ulp=2*np.spacing(np.max(np.abs(a),axis=1)).astype(np.float64)
        scale=np.max(np.abs(dw),axis=1)
        allowance=ulp+(1e-12 if f in ('qc','qr','qi','qs','qg') else 0)+1e-4*float(np.max(scale))
        tol=.05 if f=='ni' else .01
        failures=err>allowance+tol*scale
        gate[f]=dict(failures=int(np.count_nonzero(failures)),max_abs=float(err.max()),
            finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
            failed_names=z['names'][failures].tolist())
    summary=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        backend='cpu',interpret=True,dt=6.,columns=len(idx),rows=rows,gate=gate,
        elapsed_s=time.monotonic()-started,
        applied_thompson={k:v for k,v in defaults['applied'].items() if 'THOMPSON' in k})
    (OUT/'replay.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(gate=gate,zero_mismatches=sum(r['zero_pattern_mismatches'] for r in rows),
        columns_with_mismatch=[r for r in rows if r['zero_pattern_mismatches']],elapsed_s=summary['elapsed_s']),indent=2),flush=True)

if __name__=='__main__': main()
