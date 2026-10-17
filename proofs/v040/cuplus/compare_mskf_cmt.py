"""Compare the MSKF DOUBLE island with captures of the whole pristine caller."""
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
    from gpuwrf.kernels.phys_mskf_cmt import mskf_cmt,_INPUTS
    assert all(d.platform=='cpu' for d in jax.devices())
    with jax.enable_x64(True):
        fn=jax.jit(lambda profiles,ctrl:mskf_cmt(profiles,dt=ctrl[0],gravity=ctrl[1],cp=ctrl[2],
            gas_constant=ctrl[3],dsubcld=ctrl[4],jb=ctrl[5],jd=ctrl[6],jt=ctrl[7],interpret=True))
        record=dict(source=str(args.source),oracle=str(args.oracle),rows=[],gpu_executed=False,
                    gate='32*eps64*columnpeak or absolute1e-14; finite first; original DOUBLE only')
        for row in json.loads(args.oracle.read_text())['records']:
            fields,scalars=parse(row['stderr'])
            if 'CMT_IN_DP' not in fields:continue
            inputs={name:np.asarray(fields['CMT_IN_'+name],np.float64) for name in _INPUTS}
            ctrl=np.asarray([scalars['CMT_CTRL_'+name] for name in ('DT','GRAV','CPRES','RGAS','DSUBCLD','JB','JD','JT')],np.float64)
            started=time.monotonic()
            actual={key:np.asarray(value) for key,value in jax.block_until_ready(fn(inputs,ctrl)).items()}
            result=dict(case=row['case'],wind=row['background_wind'],elapsed_s=time.monotonic()-started,fields={})
            for key,value in actual.items():
                name={'DPDX':'dpdx','DPDY':'dpdy'}.get(key,key)
                truth=np.asarray(fields['CMT_OUT_'+name],np.float64)
                assert np.isfinite(truth).all() and np.isfinite(value).all(),key
                limit=max(1e-14,float(np.max(np.abs(truth)))*32*np.finfo(np.float64).eps)
                error=float(np.max(np.abs(value-truth)))
                result['fields'][key]=dict(max_abs=error,limit=float(limit),pass_=error<=limit,
                    byte_values=int(np.count_nonzero(value.view(np.uint64)==truth.view(np.uint64))))
            record['rows'].append(result)
            args.output.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
