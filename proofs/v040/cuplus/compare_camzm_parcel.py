"""Qualify the DOUBLE parcel at the actual original full CAM-ZM call boundary."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from compare_kfcup_parcel import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    import jax
    from gpuwrf.kernels.phys_camzm_column import camzm_buoyancy
    assert all(d.platform=='cpu' for d in jax.devices())
    record=dict(source=str(args.source),oracle=str(args.oracle),rows=[],gpu_executed=False,
                gate=json.loads((args.source/'proofs/v040/cuplus/CAMZM_PARCEL_GATE.json').read_text()))
    with jax.enable_x64(True):
        calls={}
        for row in json.loads(args.oracle.read_text())['records']:
            fields,scalars=parse(row['stderr'])
            msg=int(scalars['ZM_PARCEL_msg'])
            if msg not in calls:
                calls[msg]=jax.jit(lambda *a:camzm_buoyancy(*a[:5],pblt=a[5][0],tpert=a[5][1],msg=msg,interpret=True))
            inputs=tuple(np.asarray(fields['ZM_PARCEL_'+key],np.float64) for key in ('q','t','p','z','pf'))
            ctrl=np.asarray([scalars['ZM_PARCEL_pblt'],scalars['ZM_PARCEL_tpert']],np.float64)
            started=time.monotonic()
            actual={key:np.asarray(value) for key,value in jax.block_until_ready(calls[msg](*inputs,ctrl)).items()}
            result=dict(case=row['case'],pbl=row['pbl'],elapsed_s=time.monotonic()-started,fields={},failed=int(actual['failed']))
            for key,value in actual.items():
                if key=='failed':continue
                name='ZM_PARCEL_'+key
                truth=np.asarray(fields.get(name,scalars.get(name)),np.float64)
                assert np.isfinite(truth).all() and np.isfinite(value).all(),key
                error=float(np.max(np.abs(value-truth)))
                limit=float(128*np.finfo(np.float64).eps*np.max(np.abs(truth))) if key in ('tp','qstp','tl','cape') else 0.
                result['fields'][key]=dict(max_abs=error,limit=limit,pass_=error<=limit)
            record['rows'].append(result)
            args.output.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
