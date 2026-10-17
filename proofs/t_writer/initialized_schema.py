"""Exercise the real loader and writer at initialization, without advancing WRF.

The established CPU harness interprets Pallas and replaces the LAPACK solver
with Thomas to avoid the known CPU deadlock. Two writer phases use the SAME
initialized carry; numerical values are not forecast proof.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import jax
from jax.experimental import pallas as pl

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'proofs/o1_nlbind'))
import cpu_safe
cpu_safe.install()
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.netcdf_lock import Dataset

assert jax.devices()[0].platform=='cpu'
facade.cuda_native=lambda:False
original=pl.pallas_call
def interpreted(*args,**kwargs):
    kwargs['interpret']=True
    kwargs.pop('compiler_params',None)
    return original(*args,**kwargs)
pl.pallas_call=interpreted

case,output=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
output.mkdir(parents=True,exist_ok=True)
inputs=json.loads((case/'inputs.json').read_text())
domains=tuple(inputs['domains'])
result=dict(case=case.name,platform=jax.devices()[0].platform,
            scope='CPU initialized carry / real host writer; no model steps or forecast',phases={})
start=time.monotonic()
try:
    cfg=pipeline.NestedPipelineConfig(case/'input',output/'frames',output/'proof',hours=inputs['hours'],max_dom=len(domains))
    _,bundles,_,run_start,dt,carries=pipeline._load_domains(cfg,domains)
    result['init_s']=time.monotonic()-start
    writer=pipeline._PerDomainWrfoutWriter(output_dir=output/'frames',input_dir=case/'input',
        run_start=run_start,bundles=bundles,output_cadence_steps={d:1 for d in domains},dt_by_domain=dt,
        initial_lands={d:c.noahmp_land for d,c in carries.items()})
    for phase in (0,1):
        result['phases'][str(phase)]={}
        for domain in domains:
            writer(domain,phase,carries[domain])
            path=Path(writer.written[domain][-1])
            with Dataset(path) as ds:
                result['phases'][str(phase)][domain]=dict(
                    serialized_bytes=path.stat().st_size,
                    dimensions={k:len(v) for k,v in ds.dimensions.items()},
                    variables={k:dict(dimensions=list(v.dimensions),dtype=str(v.dtype),attrs={
                        a:(v.getncattr(a).item() if hasattr(v.getncattr(a),'item') else v.getncattr(a))
                        for a in v.ncattrs()}) for k,v in ds.variables.items()},
                    globals={k:(ds.getncattr(k).item() if hasattr(ds.getncattr(k),'item') else ds.getncattr(k)) for k in ds.ncattrs()})
            path.unlink()  # own schema-only serialization, never scientific reference
    result['status']='COMPLETE'
except Exception as exc:
    result.update(status='REFUSED',error=str(exc),traceback=traceback.format_exc())
finally:
    result['wall_s']=time.monotonic()-start
    (output/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],case.name,result.get('error'),result['wall_s'],flush=True)
