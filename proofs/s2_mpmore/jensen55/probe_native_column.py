"""Qualify one-column Pallas structure using CPU interpretation / TTIR only."""
from pathlib import Path
import json,os,subprocess,sys,time
import numpy as np
import resource

tree,out,action=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),sys.argv[3]
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(tree/'src'))
import jax,jax.numpy as jnp
from gpuwrf.kernels.phys_jensen import full_column
from gpuwrf.physics.jensen_ishmael.registry import output_mapping
operational=len(sys.argv)>4 and sys.argv[4]=='operational'
selection=(*output_mapping(),'th') if operational else None
call=lambda f,d,i,interpret=False:full_column(f,d,i,interpret=interpret,output_names=selection)
assert jax.devices()[0].platform=='cpu'
original=json.loads((tree/'proofs/s2_mpmore/jensen55/fixtures/fp32_4.json').read_text())
fields={n:jnp.asarray(a,jnp.float32) for n,a in original['input'].items()}
start=time.monotonic()
result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),CPU=True,GPU=False,action=action,
            output_projection='operational' if operational else 'complete')
if action=='lower':
    from jax._src.pallas.triton import lowering
    with jax.enable_x64(True):program=jax.make_jaxpr(call)(fields,jnp.float32(original['dt']),jnp.int32(original['it'])).jaxpr
    params=next(e.params for e in program.eqns if e.primitive.name=='pallas_call')
    def equations(value):
        if hasattr(value,'jaxpr'):yield from equations(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eqn in value.eqns:
                yield eqn
                for child in eqn.params.values():yield from equations(child)
        elif isinstance(value,(tuple,list)):
            for child in value:yield from equations(child)
        elif isinstance(value,dict):
            for child in value.values():yield from equations(child)
    wide=[e for e in equations(params['jaxpr']) if any(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in e.outvars)]
    outside=[e for e in wide if 'gammln' not in str(e.source_info.traceback)]
    result.update(fp64_equations=len(wide),fp64_equations_outside_original_gammln=len(outside))
    assert not outside,'FP64 outside original gammln DOUBLE island'
    native=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
    text=str(native.module);out.with_suffix('.ttir').write_text(text)
    result.update(TTIR_characters=len(text),target='cuda/sm120',elapsed_s=time.monotonic()-start)
elif action=='interpret':
    folder=tree/'proofs/s2_mpmore/jensen55'
    manifest=json.loads((folder/'column_controls/manifest.json').read_text())
    refs=[]
    for contraction in ('noFMA','FMA'):
        with np.load(folder/'column_controls'/('fp32_'+contraction+'.npz')) as data:refs.append(data['outputs'])
    output_names=tuple(manifest['output_names']) if selection is None else selection
    nz=len(original['input']['qv'])
    indices=np.concatenate([np.arange(manifest['output_names'].index(n)*nz,(manifest['output_names'].index(n)+1)*nz) for n in output_names]+
                           [np.arange(len(manifest['output_names'])*nz,len(manifest['output_names'])*nz+len(manifest['precipitation_names']))])
    low=np.minimum.reduce([a.min(axis=0) for a in refs])[:,indices];high=np.maximum.reduce([a.max(axis=0) for a in refs])[:,indices]
    baseline=refs[0][0][:,indices]
    fn=jax.jit(lambda f,d,i:call(f,d,i,interpret=True))
    compile_start=time.monotonic();executable=fn.lower(fields,jnp.float32(original['dt']),jnp.int32(original['it'])).compile()
    memory=executable.memory_analysis()
    result.update(trace_and_CPU_compile_s=time.monotonic()-compile_start,arguments_B=memory.argument_size_in_bytes,
                  outputs_B=memory.output_size_in_bytes,temps_B=memory.temp_size_in_bytes,
                  scope='CPU Pallas interpreter versus independent original WRF controls; known excesses disclosed, D6/GPU pending')
    rows=[];actual=[]
    for case in range(1,18):
        original=json.loads((folder/f'fixtures/fp32_{case}.json').read_text())
        fields={n:jnp.asarray(a,jnp.float32) for n,a in original['input'].items()}
        got,precip=jax.tree.map(np.asarray,executable(fields,jnp.float32(original['dt']),jnp.int32(original['it'])))
        values=np.concatenate([*(np.asarray(got[n]).ravel() for n in output_names),np.asarray([precip[n] for n in manifest['precipitation_names']],np.float32)])
        assert values.dtype==np.float32 and np.isfinite(values).all()
        actual.append(values);bad=(values<low[case-1])|(values>high[case-1])
        rows.append(dict(case=case,outside_original_enclosure=int(bad.sum()),indices=np.flatnonzero(bad).tolist()))
        print(rows[-1],flush=True)
    np.savez_compressed(out.with_suffix('.npz'),outputs=np.stack(actual))
    values=np.stack(actual);bad=(values<low)|(values>high)
    result.update(columns=rows,elapsed_s=time.monotonic()-start,output_names=output_names,
                  nonfinite_values=int((~np.isfinite(values)).sum()),outside_original_enclosure=int(bad.sum()),
                  peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  deviations=[dict(case=int(c+1),index=int(i),value=float(values[c,i]),low=float(low[c,i]),high=float(high[c,i]))
                              for c,i in zip(*np.where(bad))],fields={})
    for j,n in enumerate((*output_names,*manifest['precipitation_names'])):
        sl=slice(j*nz,(j+1)*nz) if j<len(output_names) else slice(len(output_names)*nz+j-len(output_names),len(output_names)*nz+j-len(output_names)+1)
        result['fields'][n]=dict(max_abs=float(np.max(np.abs(values[:,sl]-baseline[:,sl]))),outside_original_enclosure=int(bad[:,sl].sum()))
else:raise ValueError(action)
out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('deviations','columns')},flush=True)
