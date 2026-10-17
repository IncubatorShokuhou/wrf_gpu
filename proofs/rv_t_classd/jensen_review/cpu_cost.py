"""Projection A/B on real c87 columns. CPU XLA costs, never GPU timings."""
import sys,json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'snapshot'
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.jensen_ishmael.column import step
from gpuwrf.physics.jensen_ishmael.registry import output_mapping
from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping

assert jax.devices()[0].platform=='cpu'
folder=ROOT/'proofs/t_classd/jensen_refl_oracle/fixtures'
m=json.loads((folder/'manifest.json').read_text());d=np.load(folder/'oracle.npz')
nz=m['nz'];ncol=32
fields={n:jnp.asarray(d['inputs'][:ncol,i*nz:(i+1)*nz]) for i,n in enumerate(m['input_names'])}
assert all(a.dtype==jnp.float32 for a in fields.values())
names=(*output_mapping(),*ice_diagnostic_mapping(55),'th')
assert len(set(names))==len(names)
executables={};results={};values={}
for radar in (False,True):
    selected=(*names,*(('dbz',) if radar else ()))
    def f(x):
        out,precip=step(x,jnp.float32(m['dt']),m['it'])
        return tuple(out[n] for n in selected),precip
    start=time.monotonic();exe=jax.jit(f).lower(fields).compile();compile_s=time.monotonic()-start
    out=jax.block_until_ready(exe(fields));values[radar]=jax.tree.map(np.asarray,out)
    memory=exe.memory_analysis()
    results[str(radar)]=dict(output_names=selected,compile_s=compile_s,memory_analysis=str(memory),
        arguments_B=memory.argument_size_in_bytes,outputs_B=memory.output_size_in_bytes,temps_B=memory.temp_size_in_bytes,
        cost_analysis=exe.cost_analysis(),optimized_hlo_instructions=sum(' = ' in line for line in exe.as_text().splitlines()))
    executables[radar]=exe
for a,b in zip(jax.tree.leaves(values[False]),jax.tree.leaves((values[True][0][:-1],values[True][1])),strict=True):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
assert all(np.isfinite(a).all() for a in jax.tree.leaves(values))
dbz=values[True][0][-1];i=m['output_names'].index('dbz');ref=d['outputs'][:ncol,i*nz:(i+1)*nz]
assert np.array_equal(dbz>-35,ref>-35)
assert np.max(abs(dbz.astype(np.float64)-ref))<=1e-3
samples={False:[],True:[]}
for rep in range(20):
    for radar in ((False,True) if rep%2==0 else (True,False)):
        start=time.monotonic();jax.block_until_ready(executables[radar](fields));samples[radar].append(time.monotonic()-start)
for radar in (False,True):
    results[str(radar)]['samples_s']=samples[radar]
    results[str(radar)]['median_s']=statistics.median(samples[radar])
result=dict(platform='cpu',GPU=False,scope='portable XLA column projection, NOT native Triton or whole-step performance',
    real_c87_columns=ncol,nz=nz,actual_input_dtype='float32',prognostic_precip_outputs_byte=True,
    dbz_max_abs_error=float(np.max(abs(dbz.astype(np.float64)-ref))),results=results,
    incremental_result_array_bytes=ncol*nz*4,default_per_domain_held_radar_bytes='4*nz*ny*nx',
    c87_full_grid_42x42x44_one_radar_array_bytes=4*42*42*44,
    warning='New native output prevents possible DBZ dead-code elimination; GPU runtime/register/spill/traffic effects remain unmeasured.')
(ROOT.parent/'RV35_CPU_COST.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
