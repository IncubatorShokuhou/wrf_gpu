"""Independent whole-original CAM ZM_CONVR boundary comparison (CPU only)."""
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
    from gpuwrf.kernels.phys_camzm_column import camzm_convr
    assert all(d.platform=='cpu' for d in jax.devices())
    record=dict(source=str(args.source),oracle=str(args.oracle),rows=[],gpu_executed=False,
        gate='original DOUBLE: finitefirst; 128eps64*columnpeak; discrete EXACT; failed0. No bounds relaxed.')
    saved={}
    with jax.enable_x64(True):
        fn=jax.jit(lambda *a:camzm_convr(*a[:7],geos=a[7][0],pblh=a[7][1],tpert=a[7][2],
            landfrac=a[7][3],dt=a[7][4],dx=a[7][5],interpret=True))
        for row in json.loads(args.oracle.read_text())['records']:
            fields,scalars=parse(row['stderr'])
            inputs=tuple(np.asarray(fields['ZM_CORE_IN_'+key],np.float64) for key in ('t','q','pap','paph','dpp','zm','zi'))
            ctrl=np.asarray([scalars['ZM_CORE_CTRL_'+key] for key in ('geos','pblh','tpert','landfrac','dt')]+[9000.],np.float64)
            assert scalars['ZM_CORE_CTRL_tau']==600. and scalars['ZM_CORE_CTRL_limcnv']==2
            started=time.monotonic()
            actual={key:np.asarray(value) for key,value in jax.block_until_ready(fn(*inputs,ctrl)).items()}
            result=dict(case=row['case'],pbl=row['pbl'],elapsed_s=time.monotonic()-started,fields={},failed=int(actual['failed']))
            for key,value in actual.items():
                if key=='failed':continue
                name='ZM_CORE_OUT_'+key
                truth=np.asarray(fields.get(name,scalars.get(name)),np.float64)
                assert value.shape==truth.shape,(key,value.shape,truth.shape)
                assert np.isfinite(truth).all() and np.isfinite(value).all(),key
                error=float(np.max(np.abs(value-truth)))
                limit=float(128*np.finfo(np.float64).eps*np.max(np.abs(truth))) if key not in ('jt','maxg','lengath','jctop','jcbot') else 0.
                result['fields'][key]=dict(max_abs=error,limit=limit,pass_=error<=limit)
                saved[f"case{row['case']}_pbl{row['pbl']}_{key}"]=value
            record['rows'].append(result)
            args.output.write_text(json.dumps(record,indent=2)+'\n');np.savez(args.output.with_suffix('.npz'),**saved)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
