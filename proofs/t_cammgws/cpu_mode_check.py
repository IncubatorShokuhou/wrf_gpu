"""CPU-only: does the native column (interpret mode) with DEFAULT aerosol modes
reproduce the GPU frozen-control violations of cases 16-18 exactly, and do the
CORRECT per-case modes give zero? Same public bank/enclosure as the GPU runner.
usage: cpu_mode_check.py TREE OUT.json [GPU_RESULT.json]"""
import json,sys,time
from pathlib import Path
import numpy as np
tree=Path(sys.argv[1]);out=Path(sys.argv[2]);gpu=Path(sys.argv[3]) if len(sys.argv)>3 else None
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/s2_cammg')]
import jax,jax.numpy as jnp
assert jax.devices()[0].platform=='cpu'
from build_controls import configurations
from gpuwrf.kernels.phys_cammg import full_column,SURFACE_FIELDS
with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as z:bank={k:z[k] for k in z.files}
with np.load(tree/'proofs/s2_cammg/fixtures/compiled_enclosure.npz') as z:bounds={k:z[k] for k in z.files}
fn=jax.jit(lambda v,it,am,ai,co:full_column(v,18.,it,accum_mode=am,aitken_mode=ai,coarse_mode=co,interpret=True))
default=(1.e9,3.e8,2.e5);rows=[]
for case in range(16,19):
    for it in (1,2):
        prefix=f'c{case:02d}_s{it}'
        data={k.split('_input_',1)[1]:v for k,v in bank.items() if k.startswith(prefix+'_input_')}
        inp={k:jnp.asarray(v).reshape((1,) if k in SURFACE_FIELDS else (1,25)) for k,v in data.items()}
        for label,modes in (('default_modes',default),('case_modes',configurations(case)[2])):
            t=time.monotonic();got,status=fn(inp,jnp.int32(it),*map(jnp.float32,modes));jax.block_until_ready(got)
            fields={};total=0
            for name,value in got.items():
                a=np.asarray(value).reshape(-1);lo=bounds['lo_'+prefix+'_'+name];hi=bounds['hi_'+prefix+'_'+name]
                n=int(np.count_nonzero((a<lo)|(a>hi)));total+=n
                if n:fields[name]=n
            rows.append(dict(case=case,step=it,modes=label,mode_values=list(map(float,modes)),outside=total,fields=fields,
                             fatal=np.asarray(status).tolist(),seconds=time.monotonic()-t))
            print(json.dumps(rows[-1]),flush=True)
report=dict(backend='cpu',GPU_execution=False,rows=rows)
if gpu:
    g=json.loads(gpu.read_text())['frozen_original_controls']['rows']
    gmap={(r['case'],r['step']):(r['outside'],{k:v['outside'] for k,v in r['fields'].items()}) for r in g}
    report['gpu_default_mode_rows_equal_cpu_default_mode_rows']=all(
        gmap[(r['case'],r['step'])]==(r['outside'],r['fields']) for r in rows if r['modes']=='default_modes')
    report['gpu_source']=str(gpu)
report['case_modes_all_zero']=all(r['outside']==0 for r in rows if r['modes']=='case_modes')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
