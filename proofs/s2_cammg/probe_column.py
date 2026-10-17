"""Strict full public CAMMGMP gate, independently restored original call inputs."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',default='1,2,3,4,5,16,17,18')
    ap.add_argument('--fixture',default='public.npz');args=ap.parse_args()
    sys.path.insert(0,str(args.tree/'src'))
    from gpuwrf.physics.cammg.column import step
    from gpuwrf.physics.cammg.tables import get_globals
    from compiler_math import compiler_math as xp
    with np.load(args.tree/'proofs/s2_cammg/fixtures'/args.fixture) as z:bank={k:z[k] for k in z.files}
    receipt={'rows':[],'BYTE':True};start=time.monotonic()
    try:
        for case in map(int,args.cases.split(',')):
            for it in (1,2):
                prefix=f'c{case:02d}_s{it}'
                fields={k.split('_input_',1)[1]:v for k,v in bank.items() if k.startswith(prefix+'_input_')}
                mode=(1.e7,1.e6,1.e4) if case==16 else (3.e9,1.e9,5.e6) if case==17 else (0.,0.,0.) if case==18 else (1.e9,3.e8,2.e5)
                cu,shcu=(7,2) if case>=19 else (0,0)
                if case==20:shcu=0
                if case==21:cu=0
                if case in (22,24):cu,shcu=1,1
                with np.errstate(all='ignore'):
                    result,fatal=step(fields,18.,itimestep=it,cu=cu,shcu=shcu,accum_mode=mode[0],aitken_mode=mode[1],coarse_mode=mode[2],globals=get_globals(24),xp=xp)
                assert fatal==0,('originalfatal',case,it)
                assert set(result)==set(fields),'full public inventory mismatch'
                row={'case':case,'step':it,'BYTE':True,'fields':{}}
                for name,value in result.items():
                    ref=bank[prefix+'_output_'+name];got=np.asarray(value).reshape(-1)
                    assert got.shape==ref.shape and got.dtype==ref.dtype,(name,got.shape,ref.shape,got.dtype,ref.dtype)
                    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)),name
                    byte=got.tobytes()==ref.tobytes();row['BYTE']&=byte
                    if not byte:row['fields'][name]={'different':int(np.count_nonzero(got!=ref)),
                        'max_abs':float(np.max(np.abs(got-ref))),'first':np.argwhere(got!=ref)[:8].tolist()}
                receipt['rows'].append(row);receipt['BYTE']&=row['BYTE']
        receipt['seconds']=time.monotonic()-start
    finally:args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));assert receipt['BYTE'],'original full CAMMGMP BYTE gate failed'


if __name__=='__main__':main()
