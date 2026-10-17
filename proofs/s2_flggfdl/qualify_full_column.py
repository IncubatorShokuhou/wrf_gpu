"""Compare complete FLG heating/flux outputs to the unchanged native full call."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from literal import load


def main(root: Path,out: Path,columns: str,compiled: bool):
    from gpuwrf.physics import ra_flg
    with np.load(root/'tests/fixtures/s2_flg_column_v1.npz',allow_pickle=False) as a:
        data={k:a[k].copy() for k in a.files}
    with np.load(root/'tests/fixtures/s2_flg_full_v1.npz',allow_pickle=False) as a:
        native={k:a[k].copy() for k in a.files}
    indexes=list(range(40)) if columns=='all' else [int(x) for x in columns.split(',')]
    source=ra_flg if compiled else load(Path(ra_flg.__file__))
    run=source._column_from_preprocessed
    if compiled:
        import jax
        import jax.numpy as jnp
        assert jax.devices()[0].platform=='cpu'
        run=jax.jit(run)
    flux_fields={'glw':0,'gsw':1,'swdn':2,'sw_up_top':4,'lw_up_top':5,'direct':6,'diffuse':7,'dni':8}
    results=[];raw={key:[] for key in ('total_rate','lw_rate','sw_rate',*flux_fields)}
    out.parent.mkdir(parents=True,exist_ok=True)
    for i in indexes:
        pre={key[4:]:value[i] for key,value in data.items() if key.startswith('pre_')}
        pi=data['in_fields'][i,:,2]
        if compiled:
            pre={key:jnp.asarray(value) for key,value in pre.items()};pi=jnp.asarray(pi)
        t0=time.perf_counter();candidate=run(pre,pi)
        comparisons={}
        for key,value in candidate.items():
            value=np.asarray(value)
            reference=native['rates'][i,:,('total_rate','lw_rate','sw_rate').index(key)] if key.endswith('_rate') else native['flux'][i,flux_fields[key]]
            reference=np.asarray(reference)
            if value.dtype!=np.float32 or reference.dtype!=np.float32 or value.shape!=reference.shape:
                raise ValueError(f'Invalid dtype/geometry {key}')
            if not np.isfinite(value).all() or not np.isfinite(reference).all():
                raise ValueError(f'Non-finite full column {i} {key}')
            mismatch=np.count_nonzero(value.view(np.uint32)!=reference.view(np.uint32))
            error=np.abs(value.astype(np.float64)-reference.astype(np.float64))
            absolute=1.e-11 if key.endswith('_rate') else 3.e-6
            allowance=16.*np.spacing(np.abs(reference)).astype(np.float64)+absolute
            comparisons[key]={'BYTE':bool(mismatch==0),'mismatches':int(mismatch),'max_abs':float(error.max()),
                              'original_compiled_outside':int(np.count_nonzero(error>allowance))}
            raw[key].append(value)
        row={'column':i,'wall_s':time.perf_counter()-t0,'fields':comparisons}
        results.append(row)
        report={'mode':'compiled CPU' if compiled else 'literal REAL4/libm','columns_done':len(results),'columns_requested':len(indexes),
                'full_coverage':indexes==list(range(40)),'all_BYTE':all(x['BYTE'] for r in results for x in r['fields'].values()),'results':results,
                'source_sha256':hashlib.sha256(Path(ra_flg.__file__).read_bytes()).hexdigest()}
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(i,'BYTE' if all(v['BYTE'] for v in comparisons.values()) else 'DIFFERENT',row['wall_s'],flush=True)
    np.savez_compressed(out.with_suffix('.npz'),columns=np.asarray(indexes,np.int32),**{k:np.asarray(v,np.float32) for k,v in raw.items()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--columns',default='all');ap.add_argument('--compiled',action='store_true')
    a=ap.parse_args();main(a.root,a.out,a.columns,a.compiled)
