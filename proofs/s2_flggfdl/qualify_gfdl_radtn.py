"""Native whole-call RADTN gate, including diagnostic accumulators and counts."""
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
from gpuwrf.physics import ra_gfdleta as optics
from gpuwrf.physics import ra_gfdleta_support as solar
from literal import load,_wrap
from qualify_gfdl_sw import arguments


def main(source,fixture,output,selection,compiled):
    assert jax.devices()[0].platform=='cpu' and jax.config.x64_enabled
    meta=json.loads(fixture.with_suffix('.json').read_text())['source_capture']
    with np.load(fixture,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    indexes=list(range(len(data['nlayers']))) if selection=='all' else [int(x) for x in selection.split(',')]
    if compiled:
        spec=importlib.util.spec_from_file_location('radtn_compiled',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        optical=optics;clock=solar
    else:
        module=load(source,source_control=True)
        optical=load(Path(optics.__file__));clock=load(Path(solar.__file__))
    levels=np.unique(data['nlayers']);assert len(levels)==1
    provider={k[4:]:v for k,v in data.items() if k.startswith('co2_')}
    domain={k[7:]:v for k,v in data.items() if k.startswith('domain_')}
    native=module.reference_factory(int(levels[0]),provider,domain,optical.cloud_overlap,
        optical.ferrier_saturation_pressure,optical.ozone_column,clock.zenith_dynamic)
    if compiled:
        def physical(environment,*args):
            values=native(environment,*args)
            return {k:values[meta['arguments'].index(k)] for k in meta['outputs']}
        run=jax.jit(physical)
    else:run=native
    results=[];raw={k:[] for k in meta['outputs']};artifact=None
    for i in indexes:
        args=arguments(data,i,meta);env={'r1':data['environment_r1'][i]}
        assert int(data['nphs'][i])>0,'Native DO step must be positive'
        if compiled:
            args=list(map(jnp.asarray,args));env={k:jnp.asarray(v) for k,v in env.items()}
        else:args=_wrap(args)
        started=time.perf_counter()
        if compiled and artifact is None:
            executable=run.lower(env,*args).compile();hlo=executable.as_text();memory=executable.memory_analysis()
            artifact=dict(compile_s=time.perf_counter()-started,
                f64_instructions=sum(bool(re.search(r'\bf64\[',line)) for line in hlo.splitlines()),
                native_DOUBLE_islands=['ddx','eex','prod'],hlo_lines=len(hlo.splitlines()),
                temporary_bytes=memory.temp_size_in_bytes,argument_bytes=memory.argument_size_in_bytes,
                output_bytes=memory.output_size_in_bytes)
            output.with_suffix('.hlo').write_text(hlo)
        values=executable(env,*args) if compiled else run(env,*args)
        fields={}
        for key in meta['outputs']:
            value=np.asarray(values[key] if compiled else values[meta['arguments'].index(key)])
            reference=data['out_'+key][i]
            assert value.shape==reference.shape and value.dtype==reference.dtype,key
            assert value.dtype in (np.float32,np.int32) and np.isfinite(value).all() and np.isfinite(reference).all(),key
            error=np.abs(value.astype(np.float64)-reference.astype(np.float64))
            bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12 if value.dtype==np.float32 else np.zeros(value.shape)
            fields[key]=dict(BYTE=value.tobytes()==reference.tobytes(),max_abs=float(error.max()),
                             outside_16ulp=int(np.count_nonzero(error>bound)))
            raw[key].append(value)
        results.append(dict(row=i,wall_s=time.perf_counter()-started,fields=fields))
        report=dict(mode='compiledCPU' if compiled else 'literalREAL4+nativeDOUBLE/libm',completed=len(results),requested=len(indexes),
            all_BYTE=all(v['BYTE'] for r in results for v in r['fields'].values()),results=results,compiled_artifact=artifact,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            scope='whole native RADTN physical+accumulator/count fields; only declaredDOUBLE preserved; genericCO2init/ETARA/operational/GPU pending')
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(i,'BYTE' if all(v['BYTE'] for v in fields.values()) else 'DIFFERENT',results[-1]['wall_s'],flush=True)
    np.savez_compressed(output.with_suffix('.npz'),rows=np.asarray(indexes,np.int32),**{k:np.asarray(v) for k,v in raw.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','fixture','output'):p.add_argument(key,type=Path)
    p.add_argument('--rows',default='all');p.add_argument('--compiled',action='store_true')
    a=p.parse_args();main(a.source,a.fixture,a.output,a.rows,a.compiled)
