"""CPU lowering / exact-launch smoke / locked GPU fidelity and gate-cost probe.

The benchmark tiles the real-structured fixture to 31,239 columns; it measures
this kernel's gate cost only, never forecast speed or a release gate.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import textwrap
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('lower','smoke','gpu'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='gpu':assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
    else:assert os.environ['JAX_PLATFORMS']=='cpu'
    import jax
    import jax.numpy as jnp
    from gpuwrf.kernels import phys_thompson_full as full
    from gpuwrf.physics import thompson_column as tc
    platform=jax.devices()[0].platform
    assert platform==('gpu' if args.mode=='gpu' else 'cpu'),jax.devices()
    fixture=ROOT/'tests/v025/b_thompson/fixtures/graupel_entry_sedimentation.npz'
    z=dict(np.load(fixture))
    spec=importlib.util.spec_from_file_location('graupel_gate',ROOT/'tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    os.environ.update(helper.C24,GPUWRF_THOMPSON_WRF_CONSTANTS='1')
    f=lambda name:jnp.asarray(z['in_'+name],jnp.float32)
    p,qv,T=f('p'),f('qv'),f('th')*f('pii');zero=jnp.zeros_like(qv)
    state=tc.ThompsonColumnState(qv=qv,qc=f('qc'),qr=f('qr'),qi=f('qi'),qs=f('qs'),qg=f('qg'),
        Ni=f('ni'),Nr=f('nr'),Ns=zero,Ng=zero,T=T,p=p,
        rho=tc.density_from_pressure_temperature(p,T,qv),dz=f('dz8w'),w=f('w'))
    receipt=dict(mode=args.mode,backend=platform,gpu_executed=args.mode=='gpu',fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('src/gpuwrf/physics/thompson_column.py','src/gpuwrf/kernels/phys_thompson_full.py','src/gpuwrf/kernels/phys_thompson_prep.py')},
        work=dict(scheme='Thompson mp8',dt_s=6.,levels=44,fixture_columns=len(z['names']),cadence='one physics call',
                  forecast_claim=False,benchmark_columns=31239 if args.mode=='gpu' else None))
    if args.mode=='lower':
        from jax._src.pallas.triton import lowering
        from gpuwrf.kernels.phys_thompson_prep import sediment_and_finish
        one=jax.tree.map(lambda x:x[:1],state)
        sizes={}
        for name,call in [('full',lambda s:full.full_column(s,6.)),
                          ('fused',lambda s:sediment_and_finish(s,6.,graupel_sed_on=jnp.array([[False]])))]:
            jp=jax.make_jaxpr(call)(one).jaxpr
            params=next(e.params for e in jp.eqns if e.primitive.name=='pallas_call')
            lowered=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
            text=str(lowered.module);(args.out/(name+'.ttir')).write_text(text);sizes[name]=len(text)
        receipt.update(status='PASS',target='cuda/sm120',ttir_bytes=sizes,GPU_validation='NOT-APPLICABLE')
    else:
        interpret=args.mode=='smoke'
        original=tc._thompson_source_sink_body
        code=textwrap.dedent(inspect.getsource(original))
        needle='graupel_entry = state.qg > R1 if _mixed_phase_wrf_enabled() else None'
        assert code.count(needle)==1
        namespace=dict(vars(tc));exec(compile(code.replace(needle,'graupel_entry = None'),tc.__file__,'exec'),namespace)
        variants={};fidelity={};timings={}
        for name,fn in [('candidate',original),('deletion',namespace['_thompson_source_sink_body'])]:
            try:
                tc._thompson_source_sink_body=fn
                jax.clear_caches()
                if args.mode=='gpu':
                    jax.config.update('jax_compilation_cache_dir',str(args.out/'cache'/name/'jax'))
                    jax.config.update('jax_persistent_cache_min_compile_time_secs',0.)
                def call(s):return full.full_column(s,6.,interpret=interpret)
                call.__name__='graupel_'+name
                jit=jax.jit(call)
                started=time.monotonic();compiled=jit.lower(state).compile();out,ppt=compiled(state)
                jax.block_until_ready((out,ppt))
                port={k:np.asarray(getattr(out,{'ni':'Ni','nr':'Nr','th':'T'}.get(k,k)),np.float32) for k in helper.FIELDS}
                port['th']=port['th']/z['in_pii'];port['precip']=sum(np.asarray(ppt[k]) for k in ('rain','snow','graupel','ice'))
                errors=helper._excess(port,z,np.arange(len(z['names'])),helper._scales(z))
                fidelity[name]=dict(zero_mismatches=int(np.count_nonzero((port['qg']>0)!=(z['wrf_qg']>0))),
                    field_gate_failures=[(list(k),float(v)) for k,v in errors.items() if v>helper.TOL[k[1]]],
                    finite=all(bool(np.isfinite(v).all()) for v in port.values()),
                    compile_and_first_s=time.monotonic()-started)
                if args.mode=='gpu':
                    large=jax.tree.map(lambda x:jnp.tile(x,(int(np.ceil(31239/x.shape[0])),1))[:31239],state)
                    variants[name]=(jit.lower(large).compile(),large)
            finally:tc._thompson_source_sink_body=original
        assert fidelity['candidate']['zero_mismatches']==0,fidelity
        assert not fidelity['candidate']['field_gate_failures'],fidelity
        assert fidelity['deletion']['zero_mismatches']>0,fidelity
        assert all(r['finite'] for r in fidelity.values())
        if args.mode=='gpu':
            for compiled,s in variants.values():jax.block_until_ready(compiled(s))
            for name in ('candidate','deletion','deletion','candidate'):
                compiled,s=variants[name];samples=[]
                for _ in range(20):
                    t=time.perf_counter();jax.block_until_ready(compiled(s));samples.append(time.perf_counter()-t)
                timings.setdefault(name,[]).extend(samples)
            medians={k:statistics.median(v) for k,v in timings.items()}
            receipt.update(timing_samples_s=timings,median_s=medians,gate_cost_ratio=medians['candidate']/medians['deletion'],
                           performance_gate='PASS' if medians['candidate']<=1.10*medians['deletion'] else 'HOLD',
                           device_memory_stats=jax.devices()[0].memory_stats(),
                           memory_analysis={k:str(v[0].memory_analysis()) for k,v in variants.items()})
        receipt.update(status='HOLD' if receipt.get('performance_gate')=='HOLD' else 'PASS',fidelity=fidelity,
                       GPU_validation='PENDING E41/coupled N72' if args.mode=='gpu' else 'NOT-APPLICABLE')
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('timing_samples_s','source_sha256')},indent=2),flush=True)
    if receipt['status']=='HOLD':raise SystemExit(3)

if __name__=='__main__':main()
