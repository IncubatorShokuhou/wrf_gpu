"""CPU changed-artifact timing/memory on repeated original WRF wrapper columns.

This is a component measurement, not GPU throughput or a forecast benchmark.
The four original columns also check the mixed day/night tile's oracle outputs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

source,out=map(Path,sys.argv[1:3])
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(source/'src'))
import jax
import numpy as np
from gpuwrf.coupling.goddard_radiation import solve_goddard_columns

assert jax.devices()[0].platform=='cpu'
save=source/'proofs/s2_goddardrad/savepoints'
with np.load(save/'wrapper_real4.npz') as f: data={k:f[k] for k in f.files}


def column(case):
    c=data[f'{case}_input'];i=data[f'{case}_interface'];s=data[f'{case}_scalar']
    radii=np.stack((c[:,12],c[:,13],c[:,14],c[:,15],c[:,16],np.zeros(12,np.float32)),axis=-1)
    condensate=np.concatenate((c[:,3:8],np.zeros((12,1),np.float32)),axis=-1)
    return dict(T=c[:,0],p=c[:,1],p8w=i[:,0],t8w=i[:,1],qv=c[:,2],condensate=condensate,
                radius=radii,rho=c[:,8],dz=c[:,9],pi=c[:,10],cloud=c[:,11],latitude=s[0],
                cosz=s[1],albedo=s[3],emiss=s[4],tsk=s[5])


records=[]
for name,cases in [('mixed',[1,2,3,4]*32),('night',[3]*128)]:
    cols=[column(c) for c in cases]
    arrays={k:np.stack([c[k] for c in cols]) for k in cols[0]}
    # All wrapper cases use the same solar constant and Julian day.
    solcon=data['1_scalar'][2]
    args=(arrays,np.int32(100),solcon)
    start=time.perf_counter();compiled=solve_goddard_columns.lower(*args).compile()
    compile_s=time.perf_counter()-start
    values=compiled(*args);jax.block_until_ready(values)
    if name=='mixed':
        tol=json.loads((save/'wrapper_real4.json').read_text())['tolerance']
        for row,case in enumerate(cases[:4]):
            for actual,key in zip(values[:2],('lw','sw')):
                np.testing.assert_allclose(actual[row],data[f'{case}_{key}'],rtol=tol['rate_rtol'],atol=tol['rate_atol'])
            np.testing.assert_allclose(np.asarray([x[row] for x in values[11:]]),data[f'{case}_swextra'],rtol=tol['flux_rtol'],atol=tol['flux_atol'])
    assert all(np.isfinite(x).all() for x in values)
    assert not np.any(values[10])
    samples=[]
    for _ in range(9):
        start=time.perf_counter();jax.block_until_ready(compiled(*args));samples.append(time.perf_counter()-start)
    mem=compiled.memory_analysis()
    records.append({'regime':name,'columns':128,'levels':12,'compile_s':compile_s,
                    'warm_median_s':float(np.median(samples)),'warm_samples_s':samples,
                    'memory':{k:getattr(mem,k) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}})
result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        'platform':'cpu','gpu_executed':False,'cache':'NONE','affinity':sorted(os.sched_getaffinity(0)),
        'work':'128 columns repeated pristine public cases,12levels; solver tile128;9warm samples',
        'oracle_mixed_tile':'PASS','records':records}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
