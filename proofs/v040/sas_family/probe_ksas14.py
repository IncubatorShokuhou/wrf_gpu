"""Pristine KSAS14 REAL4/REAL8 driver comparison. CPU-only."""
from __future__ import annotations
import argparse
import functools
import json
import re
from pathlib import Path
import subprocess
import time
import numpy as np
from evidence_gate import field_score, require_finite


def parse(path):
    out={}
    for line in path.read_text().splitlines():
        if '=' not in line:
            continue
        key,val=line.split('=',1)
        # Fortran ES without an explicit E3 width drops the E when the
        # exponent has three digits (actual old-KF W0AVG ~4e-101).
        val=val.strip().replace('D','E')
        if 'E' not in val.upper():
            val=re.sub(r'(?<=\d)([+-]\d{3})$',r'E\1',val)
        if '[' in key:
            key,idx=key.rstrip(']').split('[')
            out.setdefault(key,[]).append(float(val))
        else:
            out[key]=float(val)
    return {k:np.asarray(v) for k,v in out.items()}


def compare_fields(actual,inp,mode):
    fields={}
    for k,a in actual.items():
        b=inp[k].astype(a.dtype)
        operands=()
        if mode=='r4' and k not in ('HBOT','HTOP','RAINCV','PRATEC'):
            input_key={'RTHCUTEN':'T','RQVCUTEN':'QV','RQCCUTEN':'QC','RQICUTEN':'QI','RUCUTEN':'U','RVCUTEN':'V'}[k]
            operands=(inp['DT'],inp['STEPCU'],inp[input_key],inp['PI'])
        def bound():
            if k in ('HBOT','HTOP'):return np.zeros_like(b,dtype=np.float64)
            if mode=='r8':
                scale=np.max(np.abs(b),axis=1,keepdims=True) if b.ndim==2 else np.abs(b)
                return 1e-9*scale
            if k in ('RAINCV','PRATEC'):return 32*np.finfo(np.float32).eps*np.abs(b)
            factor=inp['DT'][:,None]*inp['STEPCU'][:,None]
            if k=='RTHCUTEN':factor=factor*inp['PI']
            original=inp[input_key].astype(np.float64)
            updated=original+b.astype(np.float64)*factor
            return 8*np.finfo(np.float32).eps*(np.abs(original)+np.abs(updated))/factor
        rec=field_score(a,b,bound,operands=operands)
        rec['active_oracle']=int(np.count_nonzero(b))
        rec['max_scaled']=(rec['max_abs']/(np.max(np.abs(b))+1e-300) if rec['max_abs'] is not None else None)
        fields[k]=rec
    return fields


def main():
    parser=argparse.ArgumentParser();parser.add_argument('work',type=Path)
    parser.add_argument('--mode',choices=('r4','r8'))
    parser.add_argument('--scheme',type=int,choices=(14,96),default=14);args=parser.parse_args()
    import jax
    from gpuwrf.physics.cumulus_ksas14 import cu_ksas14_columns
    from gpuwrf.physics.cumulus_nsas96 import cu_nsas96_columns
    jax.config.update('jax_enable_x64',True)
    assert jax.devices()[0].platform=='cpu'
    report={'backend':'cpu','scheme':args.scheme,'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arms':{}}
    keys=('T','QV','QC','QI','P','PI','RHO','DZ','U','V','W','XLAND','HPBL','HPBL_HOLD')
    for mode in ((args.mode,) if args.mode else ('r4','r8')):
        inputs=[parse(args.work/f'case{i}_{mode}.txt') for i in range(1,6)]
        inp={k:np.stack([d[k] for d in inputs]) for k in inputs[0]}
        require_finite(**inp)
        options=dict(dt=float(inp['DT'][0]),stepcu=int(inp['STEPCU'][0]),dx=float(inp['DX'][0]),mp_physics=int(inp['MP_PHYSICS'][0]),literals=mode)
        if args.scheme==14:
            fn=jax.jit(functools.partial(cu_ksas14_columns,**options))
            column_args=[inp[k] for k in keys]+[inp['ZNU']]
        else:
            fn=jax.jit(functools.partial(cu_nsas96_columns,row_width=1,**options))
            column_args=[inp[k] for k in ('T','QV','QC','QI','P','PI','RHO','DZ','U','V','W','P8W','XLAND','HPBL','HFX','QFX')]
        start=time.monotonic()
        actual={k:np.asarray(v) for k,v in fn(*column_args).items()}
        elapsed=time.monotonic()-start
        fields=compare_fields(actual,inp,mode)
        report['arms'][mode]={'fields':fields,'pass':all(f['pass'] for f in fields.values()),'elapsed_s':elapsed,
            'active_columns':int(np.count_nonzero(inp['RAINCV'])),'port_active':int(np.count_nonzero(actual['RAINCV']))}
        np.savez_compressed(args.work/f'candidate_{mode}.npz',**actual)
        (args.work/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.work/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
