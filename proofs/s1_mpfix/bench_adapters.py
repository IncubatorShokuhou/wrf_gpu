"""Dynamic, full-State CPU cost of the corrected WRF adapter boundaries."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.coupling import scan_adapters as A
from gpuwrf.coupling.physics_couplers import R_D_OVER_CP, GRAVITY_M_S2

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
mp=int(sys.argv[2])
if mp==40:
    data=[json.loads((root/f'proofs/s1_mpfix/fixtures/dry_fp32_{i}.json').read_text()) for i in range(8,13)]
else:
    assert mp==24
    data=[json.loads((root/'proofs/v013/savepoints_wsm7_fp64/wsm7_case_4.json').read_text())]
nz=40; grid=SimpleNamespace(nz=nz,ny=1,nx=len(data))
state=State(**{n:jnp.zeros(shape,DEFAULT_DTYPES.dtype_for(n))
               for n,shape in _state_field_shapes(grid,mp_physics=mp).items()})
state=state.ensure_conditional_leaves(mp_physics=mp,use_mp_re=int(mp==24))
col=lambda n:jnp.asarray([d['columns'][n] for d in data],jnp.float32).T[:,None,:]
p=col('P'); qv=col('QV_IN'); exner=(p/1e5)**R_D_OVER_CP
temperature=col('TH_IN')*col('PII') if mp==40 else col('T_IN')
dry=temperature/exner
theta=dry*(1.+(jnp.float32(461.6)/jnp.float32(287.))*qv)
dz=col('DZ' if mp==40 else 'DELZ')
ph=jnp.concatenate((jnp.zeros((1,1,len(data)),jnp.float32),
                    jnp.cumsum(dz*jnp.float32(GRAVITY_M_S2),axis=0)),axis=0)
updates=dict(theta=theta,p_total=p,ph_total=ph,qv=qv)
fields=[('qc','QC'),('qr','QR'),('qi','QI'),('qs','QS'),('qg','QG')]
fields+=([('Ni','NI'),('Ns','NS'),('Nr','NR'),('Ng','NG'),('Nc','NC')] if mp==40 else [('qh','QH')])
updates.update({name:col(key+'_IN') for name,key in fields})
state=state.replace(**updates)
call=A.morrison_aero_adapter if mp==40 else A.wsm7_adapter
dt=float(data[0]['scalars']['DT'])
start=time.perf_counter()
compiled=jax.jit(lambda s:call(s,dt,grid)).lower(state).compile()
compile_s=time.perf_counter()-start
result=compiled(state); jax.block_until_ready(result)
assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
samples=[]
for _ in range(11):
    start=time.perf_counter(); jax.block_until_ready(compiled(state)); samples.append(time.perf_counter()-start)
source=root/'src/gpuwrf/coupling/scan_adapters.py'
print(json.dumps(dict(backend='cpu',mp=mp,shape=[len(data),40],dtype='REAL4 carry',
    work=f'dynamic full-State adapter dt{dt} radius_active={mp==24}',cache='NONE',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),compile_s=compile_s,
    median_s=statistics.median(samples),samples_s=samples,memory=str(compiled.memory_analysis())),indent=2))
