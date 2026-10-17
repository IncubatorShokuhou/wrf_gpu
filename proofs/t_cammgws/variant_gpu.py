"""Officially locked CUDA screening of ONE structural CAM-MG variant (t-cammgws).

Derived from proofs/t_rainnc/cam_gpu.py (lane t-rainnc, same gates): pristine
public c03_s1 24-level column vs the literal reference, the 36 frozen original
controls (18 cases x 2 steps) at the UNCHANGED compiled-FMA/input-ULP enclosure,
and optionally the active 44-level history column. Exports executables for E41.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('tree',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--variant',required=True);p.add_argument('--history',type=Path)
    p.add_argument('--no-controls',action='store_true')
    p.add_argument('--compiler-option',action='append',default=[],help='KEY=VALUE XLA DebugOption for this executable (diagnostic)');a=p.parse_args()
    options={k:(v=='true' if v in ('true','false') else v) for k,v in (o.split('=',1) for o in a.compiler_option)}
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(a.tree/'src'),str(a.tree/'proofs/s2_cammg'),str(here)]
    import jax
    import jax.numpy as jnp
    from variants import install
    patch=install(a.variant)
    from gpuwrf.kernels.phys_cammg import full_column,SURFACE_FIELDS
    from gpuwrf.physics.cammg.column import step
    from gpuwrf.physics.cammg.tables import get_globals
    from compiler_math import compiler_math
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
    assert jax.devices()[0].platform=='gpu'
    a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.tree/'proofs/s2_cammg/fixtures/public.npz') as z:bank={k:z[k] for k in z.files}
    fields={k.split('_input_',1)[1]:v for k,v in bank.items() if k.startswith('c03_s1_input_')}
    original={k.split('_output_',1)[1]:v for k,v in bank.items() if k.startswith('c03_s1_output_')}
    samples=[('original24',fields,dict(pristine='c03_s1'),original)]
    if a.history:
        sys.path.insert(0,str(here))
        from history_column import history_fields
        real,meta=history_fields(a.history,a.tree);samples.append(('history44',real,meta,None))
    report=dict(platform='gpu',device=str(jax.devices()[0]),official_lock=True,variant=patch,compiler_options=options,
                cuda_root=os.environ.get('CUDA_ROOT'),jax_cuda_path=__import__('jax._src.lib',fromlist=['cuda_path']).cuda_path,
                scope='Native CUDA column same-input differential; no coupled trajectory claim',rows=[],
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.tree,text=True).strip(),
                src_tree=subprocess.check_output(['git','rev-parse','HEAD:src'],cwd=a.tree,text=True).strip())
    def save():(a.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for label,f,metadata,reference in samples:
        nz=f['th'].size-1
        with np.errstate(all='ignore'):
            literal,fatal=step(f,18.,itimestep=1,globals=get_globals(nz),xp=compiler_math)
        inputs={k:jnp.asarray(v).reshape((1,) if k in SURFACE_FIELDS else (1,nz+1)) for k,v in f.items()}
        start=time.monotonic()
        # Aerosol modes are executable ARGUMENTS: frozen controls 16-18 use non-default
        # modes (build_controls.configurations); closing over defaults ran them wrongly.
        defaults=tuple(map(jnp.float32,(1.e9,3.e8,2.e5)))
        lowered=jax.jit(lambda v,it,am,ai,co:full_column(v,18.,it,accum_mode=am,aitken_mode=ai,coarse_mode=co,
                        interpret=False)).lower(inputs,jnp.int32(1),*defaults)
        hlo=lowered.as_text();compiled=lowered.compile(compiler_options=options or None);compile_s=time.monotonic()-start
        blob=bytes(compiled._executable.xla_executable.serialize())
        folder=a.out/'cache'/'aot'/'components'/'d01';folder.mkdir(parents=True,exist_ok=True)
        (folder/(label+'.xlaexec')).write_bytes(blob)
        row=dict(label=label,metadata=metadata,compile_s=compile_s,fields={},
                 lowered_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                 num_stages_in_hlo=[s for s in ('num_stages = 1','num_stages = 3','"num_stages":1','"num_stages":3') if s in hlo],
                 executable_sha256=hashlib.sha256(blob).hexdigest(),executable_bytes=len(blob))
        native,nfatal=compiled(inputs,jnp.int32(1),*defaults);jax.block_until_ready((native,nfatal))
        row['fatal']=np.asarray(nfatal).tolist();row['literal_fatal']=int(fatal)
        for key,values in native.items():
            actual=np.asarray(values).reshape(-1);expected=np.asarray(literal[key]).reshape(-1)
            assert np.isfinite(actual).all() and np.isfinite(expected).all(),key
            row['fields'][key]=dict(max_abs=float(np.max(np.abs(actual-expected))),
                literal_max=float(expected.max()),gpu_max=float(actual.max()),
                literal_nonzero=int(np.count_nonzero(expected)),gpu_nonzero=int(np.count_nonzero(actual)),
                literal_original_BYTE=None if reference is None else expected.tobytes()==reference[key].tobytes())
        wall=[]
        for _ in range(10):
            t=time.perf_counter();jax.block_until_ready(compiled(inputs,jnp.int32(1),*defaults));wall.append(time.perf_counter()-t)
        row['warm_median_s']=float(np.median(wall))
        m=compiled.memory_analysis();row['memory']=dict(arguments=m.argument_size_in_bytes,
            outputs=m.output_size_in_bytes,temporary=m.temp_size_in_bytes)
        report['rows'].append(row);save()
        print(json.dumps(dict(variant=a.variant,label=label,compile_s=round(compile_s,3),
            rain=(row['fields']['rainncv']['literal_max'],row['fields']['rainncv']['gpu_max']),
            snow=(row['fields']['snowncv']['literal_max'],row['fields']['snowncv']['gpu_max']),
            worst={k:v['max_abs'] for k,v in sorted(row['fields'].items(),key=lambda kv:-kv[1]['max_abs'])[:3]},
            fatal=row['fatal'],warm_ms=round(1e3*row['warm_median_s'],3))),flush=True)
        if label=='original24' and not a.no_controls:
            with np.load(a.tree/'proofs/s2_cammg/fixtures/compiled_enclosure.npz') as z:bounds={k:z[k] for k in z.files}
            from build_controls import configurations
            controls=[];saved={}
            for case in range(1,19):
                for it in (1,2):
                    prefix=f'c{case:02d}_s{it}'
                    data={k.split('_input_',1)[1]:v for k,v in bank.items() if k.startswith(prefix+'_input_')}
                    inp={k:jnp.asarray(v).reshape((1,) if k in SURFACE_FIELDS else (1,25)) for k,v in data.items()}
                    cu,shcu,modes=configurations(case);assert (cu,shcu)==(0,0),(case,cu,shcu)
                    got,status=compiled(inp,jnp.int32(it),*map(jnp.float32,modes));jax.block_until_ready((got,status))
                    verdict=dict(case=case,step=it,finite=True,fatal=np.asarray(status).tolist(),outside=0,fields={})
                    for name,value in got.items():
                        actual=np.asarray(value).reshape(-1)
                        lo=bounds['lo_'+prefix+'_'+name];hi=bounds['hi_'+prefix+'_'+name]
                        assert actual.shape==lo.shape==hi.shape
                        assert all(np.isfinite(x).all() for x in (actual,lo,hi)),(prefix,name,'nonfinite')
                        assert np.all(lo<=hi)
                        count=int(np.count_nonzero((actual<lo)|(actual>hi)));verdict['outside']+=count
                        if count:verdict['fields'][name]=dict(outside=count,
                            max_excess=float(np.max(np.maximum(lo.astype(np.float64)-actual,actual.astype(np.float64)-hi))))
                    controls.append(verdict)
                    for name,value in got.items():saved[prefix+'_'+name]=np.asarray(value).reshape(-1)
            np.savez_compressed(a.out/'controls_gpu.npz',**saved)
            report['frozen_original_controls']=dict(calls=len(controls),outputs_per_call=27,
                outside=sum(v['outside'] for v in controls),rows=controls,
                PASS=all(v['outside']==0 and v['fatal']==[0] for v in controls),
                reference='preexisting pristine WRF compiler-FMA/input-ULP enclosure, unchanged')
            save()
            print(json.dumps(dict(variant=a.variant,**{k:v for k,v in report['frozen_original_controls'].items() if k!='rows'})),flush=True)
        del compiled;jax.clear_caches()


if __name__=='__main__':main()
