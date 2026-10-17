"""Official-lock Swiss stream qualification; no GPU child process.

Nine interleaved unprofiled full-forecast OFF/ON pairs, followed by ONE profiled
ON16 chunk. E41 is scanned by the parent shell after CUDA has exited.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'src'))
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
import jax
import jax.numpy as jnp
import numpy as np
assert jax.devices()[0].platform == 'gpu',jax.devices()
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.streams.runtime import prepare_streams
from gpuwrf.io.netcdf_lock import Dataset

assert __import__('datetime').datetime.now(__import__('datetime').timezone.utc).hour < 19, 'defer late arm'
source = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
case = out/'case'
case.mkdir()
for name in ('wrfinput_d01','wrfbdy_d01'):
    (case/name).symlink_to(source/name)
(case/'namelist.input').write_bytes((source/'namelist.input').read_bytes())
(case/'gpuwrf.streams.json').write_text(json.dumps({'version':1,'streams':[{
    'name':'airport','domain':'d01','interval_steps':2,'buffer_frames':16,
    'region':{'stations':[{'id':'ctr','i':20,'j':20},{'id':'west','i':9,'j':12}]},
    'variables':{'T2':['instant','min','max','mean'],'RAINNC':['instant','accumulation']}}]}))
config = NestedPipelineConfig(case,out/'output',out/'proof',1,1)
started = time.perf_counter()
_,bundles,_,run_start,dt,carries = _load_domains(config,('d01',))
nl = bundles['d01'].namelist
clock = om.build_clock_base(nl)
base = jax.tree.map(jnp.copy,carries['d01'])
runtime,seeded = prepare_streams(config,('d01',),carries,dt,run_start)
on = seeded['d01']
assert all(a.committed for a in jax.tree.leaves(on.regional_streams))

def advance(carry,start,count):
    return om._advance_chunk(carry,nl,start,clock,n_steps=count,cadence=int(nl.radiation_cadence_steps))

record = {'backend':jax.devices()[0].platform,'device':str(jax.devices()[0]),'source':str(repo),
          'source_commit':os.environ.get('STREAM_SOURCE_COMMIT','UNKNOWN'),
          'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in repo.glob('src/gpuwrf/streams/*.py')},
          'work':'real Swiss42x42x44 dt18 release defaults; 2masscenters 6channels interval2 cap16',
          'initialization_s':time.perf_counter()-started}
base = advance(base,1,4)
jax.block_until_ready(base)
on = advance(on,1,3)
jax.block_until_ready(on)
before = om._advance_chunk_fori._cache_size()
on = runtime.flush({'d01':on})['d01']
on = advance(on,4,1)
jax.block_until_ready(on)
after = om._advance_chunk_fori._cache_size()
record['compiled_variants_after_first_on'] = before
record['compiled_variants_after_flush'] = after
assert before == after, 'stream flush recompiled whole forecast'
on = runtime.flush({'d01':on})['d01']
runtime.writer.drain()
record['first4_stream_report'] = runtime.report()
record['first4_frames'] = []
for path in sorted((out/'output/streams').glob('*.nc')):
    with Dataset(path) as ds:
        record['first4_frames'].append({'filename':path.name,'times':ds['time'][:].tolist(),
                                       'T2':ds['T2'][:].tolist(),'points':len(ds.dimensions['point'])})
assert [row['times'] for row in record['first4_frames']] == [[36.],[72.]]
record['observer_regression'] = {}
left,right = jax.device_get((base.state,on.state))
for key in left.active_field_names():
    a,b = np.asarray(getattr(left,key)),np.asarray(getattr(right,key))
    assert np.isfinite(a).all() and np.isfinite(b).all(),key
    record['observer_regression'][key] = {'byte_equal':a.tobytes()==b.tobytes(),
        'max_abs':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
record['regression_all_state_byte_equal'] = all(v['byte_equal'] for v in record['observer_regression'].values())
record['census_first4'] = {}
for label,c in [('off',base),('on',on)]:
    assert c.census is not None
    record['census_first4'][label] = {'guards':np.asarray(c.census.guards).tolist(),
                                     'f2_counts':np.asarray(c.census.f2_counts).tolist()}
assert record['regression_all_state_byte_equal'], 'observer changes the physical program; inspect before claiming regression'
base = advance(base,5,1)
on = advance(on,5,1)
jax.block_until_ready((base,on))
record['resolved_work'] = {'dt_s':dt['d01'],'radiation_cadence_steps':int(nl.radiation_cadence_steps),
    'warm_start_step':6,'warm_steps':16,'mp':int(nl.mp_physics),'land':int(nl.sf_surface_physics),
    'pbl':int(nl.bl_pbl_physics),'cu':int(nl.cu_physics),'interval_steps':2,'buffer_frames':16}
matched = {'off':[],'on':[]}
for repeat in range(9):
    for label in (('off','on') if repeat%2==0 else ('on','off')):
        carry = jax.tree.map(jnp.copy,base if label=='off' else on)
        jax.block_until_ready(carry)  # Donation-proof copies outside timing.
        t0 = time.perf_counter()
        result = advance(carry,6,16)
        jax.block_until_ready(result)
        matched[label].append(time.perf_counter()-t0)
record['matched_unprofiled_warm16_s'] = matched
record['matched_unprofiled_median_s'] = {k:statistics.median(v) for k,v in matched.items()}
record['matched_on_over_off'] = record['matched_unprofiled_median_s']['on']/record['matched_unprofiled_median_s']['off']
record['compiled_memory'] = {}
for label,carry in [('off',base),('on',on)]:
    executable = om._advance_chunk_fori.lower(carry,nl,6,clock,n_steps=16,
                                             cadence=int(nl.radiation_cadence_steps)).compile()
    memory = executable.memory_analysis()
    record['compiled_memory'][label] = {key:int(getattr(memory,key)) for key in
        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
record['cumulative_allocator_stats'] = jax.devices()[0].memory_stats()
# Capture only compiled step work. Output-boundary transfers/writes are outside
# this range; import, copy and compilation are excluded from paired timing.
if os.environ.get('STREAM_PROFILE','1') == '1':
    import ctypes
    cudart = ctypes.CDLL('/usr/local/cuda-13.1/lib64/libcudart.so')
    assert cudart.cudaProfilerStart() == 0
on = advance(on,6,16)
jax.block_until_ready(on)
if os.environ.get('STREAM_PROFILE','1') == '1':
    assert cudart.cudaProfilerStop() == 0
assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(jax.device_get(on))
           if np.asarray(a).dtype.kind in 'fc')
record['warm_frames_before_flush'] = int(on.regional_streams[0].count)
record['census_on21'] = {'guards':np.asarray(on.census.guards).tolist(),
                          'f2_counts':np.asarray(on.census.f2_counts).tolist()}
t0 = time.perf_counter()
on = runtime.flush({'d01':on})['d01']
runtime.close()
record['last_flush_and_join_s'] = time.perf_counter()-t0
record['stream_report'] = runtime.report()
record['scope'] = ('9interleaved unprofiled16step pairs, full evolving forecast; physical bytes=GPU regression only. '
    'CPU-WRF sampler/window parity has separate independent receipts. No S1/R1/S3/longforecast claim. '
    'Cumulative allocator peak includes both arms, compilation and proof copies, not isolated ON VRAM.')
(out/'bench.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('backend','regression_all_state_byte_equal','matched_unprofiled_median_s','matched_on_over_off','compiled_memory')}),flush=True)
