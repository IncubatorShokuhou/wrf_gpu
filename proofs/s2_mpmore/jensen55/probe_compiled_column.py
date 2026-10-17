"""CPU compiled public-caller census on actual baked packaged tables, not a gate."""
from pathlib import Path
import json,os,resource,subprocess,sys,time
import numpy as np

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(tree/'src'))
import jax,jax.numpy as jnp
from gpuwrf.physics.jensen_ishmael.column import step
from gpuwrf.physics.jensen_ishmael import tables
assert jax.devices()[0].platform=='cpu'
folder=tree/'proofs/s2_mpmore/jensen55/fixtures'


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


result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),
            CPU=True,GPU=False,scope='Actual baked-table CPU compiled error/resource census;independent whole-column bounds pending',modes={})
start=time.monotonic();raw=tables._host_tables()
result['first_raw_host_table_load_s']=time.monotonic()-start
result['raw_host_array_bytes']=sum(a.nbytes for a in raw.values())
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    jax.config.update('jax_enable_x64',mode=='fp64')
    records=[json.loads((folder/f'{mode}_{case}.json').read_text()) for case in range(1,18)]
    def args(record):
        f={k:jnp.asarray(v,dtype=R) for k,v in record['input'].items()}
        return f,jnp.asarray(record['dt'],dtype=R),jnp.asarray(record['it'],dtype=jnp.int32)
    sample=args(records[3]);fn=jax.jit(step)
    start=time.monotonic();lowered=fn.lower(*sample);executable=lowered.compile();compile_s=time.monotonic()-start
    memory=executable.memory_analysis()
    report=dict(trace_and_CPU_compile_s=compile_s,arguments_B=memory.argument_size_in_bytes,
                outputs_B=memory.output_size_in_bytes,temps_B=memory.temp_size_in_bytes,
                constants_B=getattr(memory,'constant_size_in_bytes',None),columns=[])
    if mode=='fp32':
        program=jax.make_jaxpr(fn)(*sample)
        wide=[e for e in equations(program) if any(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in e.outvars)]
        outside=[e for e in wide if 'gammln' not in str(e.source_info.traceback)]
        report.update(fp64_equations=len(wide),fp64_equations_not_attributed_to_original_gammln=len(outside),
                      first_unattributed_sources=[str(e.source_info.traceback)[-2000:] for e in outside[:3]],
                      x64_after=bool(jax.config.jax_enable_x64))
    for case,original in enumerate(records,1):
        got,precip=jax.tree.map(np.asarray,executable(*args(original)))
        differences={}
        for group,actual in [('output',got),('precipitation',precip)]:
            for name,value in actual.items():
                expected=np.asarray(original[group][name],R);assert value.dtype==expected.dtype
                bad=value!=expected;finite=np.isfinite(value)
                same=(np.signbit(value)==np.signbit(expected))&finite
                I=np.int32 if mode=='fp32' else np.int64
                bits=value[same].view(I).astype(np.int64);reference=expected[same].view(I).astype(np.int64)
                ulp=int(np.max(np.abs(bits-reference))) if len(bits) else 0
                nonzero=expected!=0
                relative=float(np.max(np.abs(value[nonzero]-expected[nonzero])/np.abs(expected[nonzero]))) if np.any(nonzero) else 0.
                if bad.any() or not finite.all():
                    differences[name]=dict(not_BYTE=int(bad.sum()),nonfinite=int((~finite).sum()),max_abs=float(np.max(np.abs(value-expected))),
                                           max_ULP_same_sign=ulp,max_relative_nonzero=relative,changed_zero=int(np.count_nonzero(bad&~nonzero)))
        report['columns'].append(dict(case=case,differences=differences))
    elapsed=[]
    for _ in range(7):
        start=time.monotonic();jax.tree.map(lambda a:a.block_until_ready(),executable(*sample));elapsed.append(time.monotonic()-start)
    report['warm_CPU_invocation_median_ms']=1000*float(np.median(elapsed))
    report['peak_process_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['modes'][mode]=report
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(mode,{k:v for k,v in report.items() if k not in ('columns','first_unattributed_sources')},flush=True)
    print('BYTE_columns',sum(not r['differences'] for r in report['columns']),'nonfinite',sum(d['nonfinite'] for r in report['columns'] for d in r['differences'].values()),flush=True)
    fn.clear_cache()
