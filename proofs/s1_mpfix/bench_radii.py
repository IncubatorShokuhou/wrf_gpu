"""Changed CPU NSSL full-column artifact cost for radius-only production."""
import json
from pathlib import Path
import statistics
import sys
import time
import hashlib
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
import jax
import jax.numpy as jnp
from gpuwrf.physics.nssl2mom.driver import nssl2mom_column
from gpuwrf.physics.nssl2mom.constants import get_constants
from gpuwrf.physics.nssl2mom.indices import FP32
jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
sp=json.loads((root/'proofs/v034/f2_oracles/nssl_2mom/nssl_case_4.json').read_text())
names={'th':'TH','qv':'QV','qc':'QC','qr':'QR','qi':'QI','qs':'QS','qg':'QH','qh':'QHL','qndrop':'CCW','qnr':'CRW','qni':'CCI','qns':'CSW','qng':'CHW','qnh':'CHL','qnn':'CN','qvolg':'VHW','qvolh':'VHL','pii':'PII','p':'P','w':'W','dz':'DZ','rho':'DN'}
fields={key:jnp.asarray(sp['columns'][name if key in ('pii','p','w','dz','rho') else name+'_IN'],jnp.float32) for key,name in names.items()}
C=get_constants('fp32')
result={}
for active in (False,True):
    fn=jax.jit(lambda f:nssl2mom_column(f,sp['scalars']['DT'],C,FP32,first_step=True,radii=active))
    t0=time.perf_counter();compiled=fn.lower(fields).compile();compile_s=time.perf_counter()-t0
    jax.block_until_ready(compiled(fields));samples=[]
    for _ in range(11):
        t0=time.perf_counter();jax.block_until_ready(compiled(fields));samples.append(time.perf_counter()-t0)
    result[str(active)]=dict(compile_s=compile_s,median_s=statistics.median(samples),samples_s=samples,memory=str(compiled.memory_analysis()))
print(json.dumps(dict(backend='cpu',work='NSSL18 40levels active pristinecase4 REAL4 dt60 first_step1 diag0;radius-only selector changes',cache='NONE/GPUWRF_JAX_CACHE=0',source={n:hashlib.sha256((root/'src/gpuwrf/physics/nssl2mom'/n).read_bytes()).hexdigest() for n in ('driver.py','diagnostics.py')},arms=result),indent=2))
