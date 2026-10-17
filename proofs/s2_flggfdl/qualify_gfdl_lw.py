"""Separate native LWR88 literal/compiled qualification on actual caller inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from literal import load,_wrap
from qualify_gfdl_sw import arguments


def main(source,fixture,output,selection,compiled):
    assert jax.devices()[0].platform=='cpu'
    meta=json.loads(fixture.with_suffix('.json').read_text())['source_capture']
    with np.load(fixture,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    indexes=list(range(len(data['nlayers']))) if selection=='all' else [int(x) for x in selection.split(',')]
    if compiled:
        spec=importlib.util.spec_from_file_location('lwr88_compiled',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:module=load(source,source_control=True)
    levels=np.unique(data['nlayers']);assert len(levels)==1
    provider={k[4:]:v for k,v in data.items() if k.startswith('co2_')}
    native_function=module.reference_factory(int(levels[0]),provider)
    if compiled:
        def physical_outputs(*args):
            values=native_function(*args)
            return {k:values[meta['arguments'].index(k)] for k in meta['outputs']}
        run=jax.jit(physical_outputs)
    else:run=native_function
    results=[];raw={k:[] for k in meta['outputs']};artifact=None
    for i in indexes:
        args=arguments(data,i,meta)
        args=list(map(jnp.asarray,args)) if compiled else _wrap(args)
        started=time.perf_counter()
        if compiled and artifact is None:
            executable=run.lower(*args).compile();hlo=executable.as_text();memory=executable.memory_analysis()
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
            assert value.shape==reference.shape and value.dtype==reference.dtype==np.float32,key
            assert np.isfinite(value).all() and np.isfinite(reference).all(),key
            error=np.abs(value.astype(np.float64)-reference.astype(np.float64))
            bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
            fields[key]=dict(BYTE=value.tobytes()==reference.tobytes(),max_abs=float(error.max()),
                             outside_16ulp=int(np.count_nonzero(error>bound)))
            raw[key].append(value)
        results.append(dict(row=i,wall_s=time.perf_counter()-started,fields=fields))
        report=dict(mode='compiledCPU' if compiled else 'literalREAL4/libm',
            completed=len(results),requested=len(indexes),all_BYTE=all(v['BYTE'] for r in results for v in r['fields'].values()),
            results=results,compiled_artifact=artifact,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            scope='native true-caller LWR88 outputs, CGSflux/Kday; native grid-specific CO2 oracle provider; no generic init/operational/GPU qualification')
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(i,'BYTE' if all(v['BYTE'] for v in fields.values()) else 'DIFFERENT',results[-1]['wall_s'],flush=True)
    np.savez_compressed(output.with_suffix('.npz'),rows=np.asarray(indexes,np.int32),**{k:np.asarray(v) for k,v in raw.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','fixture','output'):p.add_argument(key,type=Path)
    p.add_argument('--rows',default='all');p.add_argument('--compiled',action='store_true')
    a=p.parse_args();main(a.source,a.fixture,a.output,a.rows,a.compiled)
