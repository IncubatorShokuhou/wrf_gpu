"""Station-component CPU compile/memory/timing, explicitly not GPU/S1 numbers."""
import importlib.util,json
from pathlib import Path
import sys,time
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'src'))
spec=importlib.util.spec_from_file_location('tslist_test_inputs',repo/'tests/test_tslist.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.timeseries.sampling import sample_fields,seed_buffer
assert jax.devices()[0].platform=='cpu'
state,hist,rad,soil,metrics=module.patch_fields(0)
fn=jax.jit(lambda x,y:sample_fields(state,hist,rad,soil,metrics,x,y,5))
x,y=jnp.array([0,2],jnp.int32),jnp.array([0,0],jnp.int32)
t0=time.perf_counter()
compiled=fn.lower(x,y).compile()
compile_s=time.perf_counter()-t0
jax.block_until_ready(compiled(x,y))
samples=[]
for _ in range(25):
    t0=time.perf_counter()
    jax.block_until_ready(compiled(x,y))
    samples.append(time.perf_counter()-t0)
memory=compiled.memory_analysis()
locs=module.locations()
default_locs=tuple(locs[i%2] for i in range(5))
buf=seed_buffer(default_locs,15,200,land=[True]*5)
receipt={'backend':'cpu','scope':'2station44level sampler only; no forecast/GPU performance claim',
         'compile_s':compile_s,'warm_call_median_s':float(np.median(samples)),
         'warm_call_samples_s':samples,'compiled_memory':{name:getattr(memory,name) for name in
           ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
         'default_buffer_5stations_15levels_200rows_bytes':sum(int(x.nbytes) for x in jax.tree.leaves(buf)),
         'default_OFF_model_jaxpr_sha':'2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a',
         'default_OFF_observer_device_leaves':0,
         'GPU_launch_count_peak_and_readiness':'UNMEASURED; prepared manager arm required'}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
