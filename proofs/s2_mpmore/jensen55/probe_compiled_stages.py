"""Measure compiled differences from actual original calls; this is not a gate."""
from pathlib import Path
import json,os,sys,time
import numpy as np

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(tree/'src'))
import jax,jax.numpy as jnp
from gpuwrf.physics.jensen_ishmael import shape,lookup
jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
folder=tree/'proofs/s2_mpmore/jensen55/stages'
manifest=json.loads((folder/'manifest.json').read_text())
result={'CPU':True,'scope':'Compiled error/resource census vs pristine observed calls; enclosure qualification pending','modes':{}}
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    with np.load(folder/(mode+'.npz')) as data:d={k:data[k].copy() for k in data.files}
    inputs={k:d['shape_inputs'][:,i] for i,k in enumerate(manifest['shape_input_names'])}
    start=time.monotonic()
    lowered=jax.jit(lambda fields,table:shape.var_check(**fields,gamma_tab=table)).lower(inputs,d['gamma_table'])
    if mode=='fp32':assert 'f64' not in str(lowered.compiler_ir('stablehlo'))
    executable=lowered.compile();compile_s=time.monotonic()-start
    got=jax.tree.map(np.asarray,executable(inputs,d['gamma_table']))
    fields={};bad_indices=set()
    integer=np.int32 if mode=='fp32' else np.int64
    for name,value in got.items():
        ref=d['shape_outputs'][:,manifest['shape_output_names'].index(name)]
        assert value.dtype==ref.dtype and np.isfinite(value).all()
        ulp=np.abs(value.view(integer).astype(np.int64)-ref.view(integer).astype(np.int64))
        bad=np.flatnonzero(ulp);bad_indices.update(map(int,bad))
        fields[name]=dict(not_BYTE=len(bad),max_ULP=int(ulp.max()),max_abs=float(np.max(np.abs(value-ref))),
                          max_rel=float(np.max(np.abs(value-ref)/np.abs(ref))))
    memory=executable.memory_analysis()
    m=dict(shape=dict(fields=fields,original_calls=len(d['shape_inputs']),not_BYTE_rows=len(bad_indices),
                      CPU_compile_s=compile_s,arguments_B=memory.argument_size_in_bytes,outputs_B=memory.output_size_in_bytes,
                      temps_B=memory.temp_size_in_bytes,fp64_ops_outside_original_DOUBLE=0 if mode=='fp32' else 'REAL8reference'))
    np.save(out.with_name(out.stem+'_'+mode+'_shape_indices.npy'),np.asarray(sorted(bad_indices),np.int32))
    v=d['lookup_values'];idx=d['lookup_indices']
    corners=tuple(v[:,5+i] for i in range(16))
    delta=(v[:,0]-idx[:,3].astype(R),v[:,1]-idx[:,4].astype(R),
           v[:,2]-idx[:,2].astype(R),v[:,3]-idx[:,1].astype(R))
    start=time.monotonic();executable=jax.jit(lookup.interpolate_corners).lower(corners,*delta).compile()
    value=np.asarray(executable(corners,*delta));ref=v[:,4]
    assert np.isfinite(value).all() and value.dtype==ref.dtype
    bad=value.view(integer)!=ref.view(integer)
    ulp=np.abs(value.view(integer).astype(np.int64)-ref.view(integer).astype(np.int64))
    memory=executable.memory_analysis()
    m['lookup']=dict(original_calls=len(v),not_BYTE=int(bad.sum()),max_ULP=int(ulp.max()),
                     max_abs=float(np.max(np.abs(value-ref))),CPU_compile_s=time.monotonic()-start,
                     arguments_B=memory.argument_size_in_bytes,outputs_B=memory.output_size_in_bytes,temps_B=memory.temp_size_in_bytes)
    np.save(out.with_name(out.stem+'_'+mode+'_lookup_indices.npy'),np.flatnonzero(bad).astype(np.int32))
    result['modes'][mode]=m
out.write_text(json.dumps(result,indent=2)+'\n')
print(result)
