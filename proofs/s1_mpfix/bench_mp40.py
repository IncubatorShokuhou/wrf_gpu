"""Source-bound CPU-only changed-artifact timing; no GPU speed inference."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,sys.argv[1]+'/src')
import jax
import jax.numpy as jnp
from gpuwrf.physics.microphysics_morrison_aero import morrison_aero_run
jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform == 'cpu'
fixtures=Path(sys.argv[2])
inputs=['TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','NI_IN','NS_IN','NR_IN','NG_IN','NC_IN','PII','P','DZ','W','KZH_IN','AEROCU_DUST1','AEROCU_DUST2','AEROCU_DUST3','AEROCU_DUST4','AEROCU_SEASALT','AEROCU_SULFATE','AEROCU_BCPHOB','AEROCU_BCPHIL','AEROCU_OCPHOB','AEROCU_OCPHIL']
data=[json.loads((fixtures/f'dry_fp32_{case}.json').read_text()) for case in range(8,13)]
args=[jnp.asarray([d['columns'][name] for d in data],jnp.float32) for name in inputs]+[jnp.float32(60.)]
t0=time.perf_counter()
compiled=morrison_aero_run.lower(*args,aercu_opt=0).compile()
compile_s=time.perf_counter()-t0
jax.block_until_ready(compiled(*args))
times=[]
for _ in range(9):
    t0=time.perf_counter(); jax.block_until_ready(compiled(*args)); times.append(time.perf_counter()-t0)
source=Path(sys.argv[1])
print(json.dumps(dict(backend='cpu',shape=[5,40],work='mp40 aercu0 dt60 REAL4 dynamic full-output call',cache='NONE/GPUWRF_JAX_CACHE=0',
 source={name:hashlib.sha256((source/'src/gpuwrf/physics'/name).read_bytes()).hexdigest() for name in ['microphysics_morrison_aero.py','_morrison_aero_cold.py']},
 compile_s=compile_s,median_s=statistics.median(times),samples_s=times,memory=str(compiled.memory_analysis())),indent=2))
