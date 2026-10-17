"""Manager-window GPU qualification: real Swiss four-step tslist+short timing.

Launch only through gpu_arm.sh / the official GPU lock. No GPU subprocesses.
The parent shell runs the cubin gate after this process releases CUDA.
"""
from dataclasses import replace
import hashlib,json,os
from pathlib import Path
import sys,time,statistics

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'src'))
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
import jax
import jax.numpy as jnp
import numpy as np
assert jax.devices()[0].platform=='gpu',jax.devices()
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.timeseries.runtime import prepare_time_series

source=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
case=out/'case'
case.mkdir()
for name in ('wrfinput_d01','wrfbdy_d01'):
    (case/name).symlink_to(source/name)
text=(source/'namelist.input').read_text().replace('&domains\n','&domains\n max_ts_level=5,\n ts_buf_size=3,\n')
(case/'namelist.input').write_text(text)
(case/'tslist').write_bytes((repo/'proofs/u2_tslist/oracle/tslist').read_bytes())
config=NestedPipelineConfig(case,out/'station_files',out/'proof',1,1)
started=time.perf_counter()
_,bundles,_,run_start,dt,carries=_load_domains(config,('d01',))
nl=bundles['d01'].namelist
clock=om.build_clock_base(nl)
base=jax.tree.map(jnp.copy,carries['d01'])
runtime,seeded=prepare_time_series(config,('d01',),bundles,carries,dt,run_start,Gen2Run(case).namelist)
on=seeded['d01']
assert runtime is not None
assert all(a.committed for a in jax.tree.leaves(on.timeseries))

def advance(carry,start,count):
    return om._advance_chunk(carry,nl,start,clock,n_steps=count,cadence=int(nl.radiation_cadence_steps))

record={'backend':jax.devices()[0].platform,'device':str(jax.devices()[0]),'source':str(repo),
        'source_commit':os.environ.get('TSLIST_SOURCE_COMMIT','UNKNOWN'),
        'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in repo.glob('src/gpuwrf/timeseries/*.py')},
        'work':'real Swiss42x42x44 dt18 4steps level5 tsbuf3 2inside+1outside station; release fast defaults',
        'initialization_s':time.perf_counter()-started}
t0=time.perf_counter()
base=advance(base,1,4)
jax.block_until_ready(base)
record['off_first4_s']=time.perf_counter()-t0
t0=time.perf_counter()
on=advance(on,1,3)
jax.block_until_ready(on)
cache_before=om._advance_chunk_fori._cache_size()
on=runtime.flush({'d01':on})['d01']
on=advance(on,4,1)
jax.block_until_ready(on)
cache_after=om._advance_chunk_fori._cache_size()
record['on_first4_including_flush_s']=time.perf_counter()-t0
record['compiled_variants_after_first_on']=cache_before
record['compiled_variants_after_flush']=cache_after
assert cache_after==cache_before,'buffer flush changed placement and recompiled whole model'
on=runtime.flush({'d01':on})['d01']
record['files']=runtime.report()
base_fields=jax.device_get(base.state)
on_fields=jax.device_get(on.state)
record['observer_regression']={}
for key in ('theta','u','v','w','qv','p_total','ph_total'):
    a,b=np.asarray(getattr(base_fields,key)),np.asarray(getattr(on_fields,key))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta=a.astype(np.float64)-b
    record['observer_regression'][key]={'max_abs':float(np.max(np.abs(delta))),
                                         'rmse':float(np.sqrt(np.mean(delta**2))),
                                         'byte_equal':a.tobytes()==b.tobytes()}
# The timing interval is a compiled multi-step call with no observer DtoH.
# Expand the station buffer once to bound 16 warm rows, outside the timed call.
buf=on.timeseries
buf=replace(buf,hours=jnp.zeros((16,),jnp.float32),surface=jnp.zeros((16,2,14),jnp.float32),
            profiles=jnp.zeros((16,2,7,5),jnp.float32))
on=on.replace(timeseries=jax.device_put(buf,device=on.state.theta.device))
on=advance(on,5,1)
jax.block_until_ready(on)
on=on.replace(timeseries=replace(on.timeseries,count=jax.device_put(jnp.int32(0),device=on.state.theta.device)))
base=advance(base,5,1)
jax.block_until_ready(base)
# Compare both already-compiled variants before enabling profiler collection.
# Reset the observer outside each timed interval; both arms start from step 5.
record['resolved_work']={'dt_s':dt['d01'],'radiation_cadence_steps':int(nl.radiation_cadence_steps),
                         'warm_start_step':6,'warm_steps':16,'warm_buffer_capacity':16,
                         'mp':int(nl.mp_physics),'land':int(nl.sf_surface_physics)}
matched={'off':[],'on':[]}
for repeat in range(9):
    for label in (('off','on') if repeat%2==0 else ('on','off')):
        carry=jax.tree.map(jnp.copy,base if label=='off' else on)
        jax.block_until_ready(carry)  # release presets donate inputs; copying is outside timing
        t0=time.perf_counter()
        result=advance(carry,6,16)
        jax.block_until_ready(result)
        matched[label].append(time.perf_counter()-t0)
record['matched_unprofiled_warm16_s']=matched
record['matched_unprofiled_median_s']={k:statistics.median(v) for k,v in matched.items()}
record['matched_on_over_off']=record['matched_unprofiled_median_s']['on']/record['matched_unprofiled_median_s']['off']
record['compiled_memory']={}
for label,carry in [('off',base),('on',on)]:
    executable=om._advance_chunk_fori.lower(carry,nl,6,clock,n_steps=16,
                                           cadence=int(nl.radiation_cadence_steps)).compile()
    memory=executable.memory_analysis()
    record['compiled_memory'][label]={key:int(getattr(memory,key)) for key in
        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
record['device_memory_stats_cumulative_before_trace']=jax.devices()[0].memory_stats()
record['matched_scope']='9 interleaved 16-step OFF/ON pairs, both outside CUDA profiler capture; same step-5 inputs and step6-21 work; compiled memory is full evolving forecast, cumulative device peak includes all variants'
for label,carry in [('off',base),('on',on)]:
    if label=='on' and os.environ.get('TSLIST_GPU_PROFILE','1')=='1':
        import ctypes
        cudart=ctypes.CDLL('/usr/local/cuda-13.1/lib64/libcudart.so')
        assert cudart.cudaProfilerStart()==0
    t0=time.perf_counter()
    carry=advance(carry,6,16)
    jax.block_until_ready(carry)
    record[label+'_warm16_s']=time.perf_counter()-t0
    if label=='on' and os.environ.get('TSLIST_GPU_PROFILE','1')=='1':
        assert cudart.cudaProfilerStop()==0
    record[label+'_warm_s_per_own_step']=record[label+'_warm16_s']/16
    audited=(carry.state,carry.timeseries)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(jax.device_get(audited))
               if np.asarray(x).dtype.kind in 'fc')
record['timing_scope']='single short warm 16-step component sample; no S1 speedup claim or noise attribution'
record['profile_scope']='cudaProfiler capture only around ON warm16; Nsight may inflate that component wall'
if os.environ.get('TSLIST_GPU_PROFILE','1')!='1':
    record['profile_scope']='No profiler attached; matched pairs and final single chunks are unprofiled'
record['scope']='CPU-WRF file/diagnostic parity is covered by tracked oracles; observer diff is GPU-vs-GPU regression only'
(out/'bench.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
