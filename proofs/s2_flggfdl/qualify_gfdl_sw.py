"""Native true-caller SWR93 parity, with literal and compiled evidence separate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import re

import jax
import jax.numpy as jnp
import numpy as np

from literal import load,_wrap
from capture_gfdl_sw import shape


def arguments(data,index,meta):
    nz=int(data['nlayers'][index]);args=[]
    bounds={k:1 for k in ('ids','jds','kds','ims','ime','jms','jme','kms','its','ite','jts','jte','kts')}
    bounds.update(ide=2,jde=2,kde=nz+1,kme=nz+1,kte=nz)
    bounds.update(imd=1,jmd=1,jndx=1)
    for key in meta['arguments']:
        if key in bounds:args.append(np.asarray(bounds[key],np.int32))
        elif key in data:args.append(data[key][index])
        elif 'out_'+key in data:args.append(np.zeros(shape(meta['shapes'][key],nz),data['out_'+key].dtype))
        else:raise ValueError(('missing native argument',key))
    return args


def main(source,fixture,output,selection,compiled):
    assert jax.devices()[0].platform=='cpu'
    meta=json.loads(fixture.with_suffix('.json').read_text())['source_capture']
    with np.load(fixture,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    indexes=list(range(len(data['nlayers']))) if selection=='all' else [int(x) for x in selection.split(',')]
    if compiled:
        import importlib.util
        spec=importlib.util.spec_from_file_location('swr93_compiled',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:module=load(source,source_control=True)
    levels=np.unique(data['nlayers'])
    assert len(levels)==1
    run=module.reference_factory(int(levels[0]))
    if compiled:
        reference_run=run
        def physical_outputs(*args):
            values=reference_run(*args)
            return {k:values[meta['arguments'].index(k)] for k in meta['outputs']}
        run=jax.jit(physical_outputs)
    artifact=None
    results=[];raw={k:[] for k in meta['outputs']}
    for i in indexes:
        args=arguments(data,i,meta)
        args=list(map(jnp.asarray,args)) if compiled else _wrap(args)
        started=time.perf_counter()
        if compiled and artifact is None:
            executable=run.lower(*args).compile()
            hlo=executable.as_text()
            memory=executable.memory_analysis()
            artifact=dict(compile_s=time.perf_counter()-started,
                f64_instructions=sum(bool(re.search(r'\bf64\[',line)) for line in hlo.splitlines()),
                hlo_lines=len(hlo.splitlines()),temporary_bytes=memory.temp_size_in_bytes,
                argument_bytes=memory.argument_size_in_bytes,output_bytes=memory.output_size_in_bytes)
            output.with_suffix('.hlo').write_text(hlo)
        values=executable(*args) if compiled else run(*args)
        fields={}
        for key in meta['outputs']:
            value=np.asarray(values[key] if compiled else values[meta['arguments'].index(key)])
            reference=data['out_'+key][i]
            if key in {'hswc','hswl'}:value=value[:,:int(levels[0])]
            assert value.shape==reference.shape and value.dtype==reference.dtype==np.float32,key
            assert np.isfinite(value).all() and np.isfinite(reference).all(),key
            err=np.abs(value.astype(np.float64)-reference.astype(np.float64))
            bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
            fields[key]=dict(BYTE=value.tobytes()==reference.tobytes(),max_abs=float(err.max()),
                outside_16ulp=int(np.count_nonzero(err>bound)))
            raw[key].append(value)
        results.append(dict(row=i,wall_s=time.perf_counter()-started,fields=fields))
        report=dict(mode='compiledCPU' if compiled else 'literalREAL4/libm',
            all_BYTE=all(v['BYTE'] for r in results for v in r['fields'].values()),
            completed=len(results),requested=len(indexes),results=results,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            compiled_artifact=artifact,
            scope='SWR93 native input/output parity; literal scalar statement predicates lazy as Fortran, active bounds strict; CGS flux erg/cm2/s, heating K/day; no GPU/operational qualification')
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(i,'BYTE' if all(v['BYTE'] for v in fields.values()) else 'DIFFERENT',results[-1]['wall_s'],flush=True)
    np.savez_compressed(output.with_suffix('.npz'),rows=np.asarray(indexes,np.int32),**{k:np.asarray(v) for k,v in raw.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','fixture','output'):p.add_argument(key,type=Path)
    p.add_argument('--rows',default='all');p.add_argument('--compiled',action='store_true')
    a=p.parse_args();main(a.source,a.fixture,a.output,a.rows,a.compiled)
