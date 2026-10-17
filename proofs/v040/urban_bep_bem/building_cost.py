"""BEM floor batching: CPU executable cost plus regression and WRF step-1 gates.

The immutable baseline function is read from its Git blob. Byte comparison is
regression only; numerical fields are independently compared with pristine WRF.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import jax
import numpy as np

from bem_driver_parity import unpack,packed
from gpuwrf.physics import urban_bem_building as building,urban_bem_model as model
from gpuwrf.physics.urban_bep_static import read_bep_classes
from gpuwrf.config.paths import wrf_run_dir  # GPUWRF_WRF_ROOT or <repo>/data/wrf_pristine/WRF
from gpuwrf.physics.urban_bem_static import bem_class
from gpuwrf.physics.urban_bep_blend import noahmp_bep_blend


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ['JAX_PLATFORMS']=='cpu';assert jax.devices()[0].platform=='cpu'
    jax.config.update('jax_enable_x64',True)
    source=subprocess.check_output(['git','show',a.baseline+':src/gpuwrf/physics/urban_bem_building.py'],text=True)
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='building_step')
    old_source=ast.get_source_segment(source,node);namespace=dict(vars(building));exec(compile(old_source,'<immutable_scalar_floor_baseline>','exec'),namespace)
    old_building=namespace['building_step'];new_building=building.building_step
    wrapper=inspect.getsource(model.bem_column_step)
    variants={}
    for name,fn in [('scalar_floor',old_building),('batched_floor',new_building)]:
        ns=dict(vars(model));ns['building_step']=fn;exec(compile(wrapper,'<'+name+'_canopy>','exec'),ns)
        variants[name]=ns['bem_column_step']
    data=np.load(a.save/'urban3_fp32_ac_on.npz');data64=np.load(a.save/'urban3_fp64_ac_on.npz')
    _,classes=read_bep_classes(str(wrf_run_dir()/'URBPARM.TBL'),np.float32)
    geo=bem_class(classes[1],1,np.float32);args=unpack(data,'fp32');results={};values={}
    for name,canopy in variants.items():
        def step(state,forcing,rural,frc,other):
            latitude,direct,diffuse,rain,qv=other
            r=canopy(state,forcing,geo,latitude=latitude,sw_direct=direct,sw_diffuse=diffuse,rain=rain,qv=qv,fraction=frc)
            surface,pbl=noahmp_bep_blend(r,forcing,rural,frc,raw_moisture=r.moisture_source)
            return packed(r.state,surface,pbl,r.diagnostics)
        started=time.perf_counter();lowered=jax.jit(step).lower(*args);lower_s=time.perf_counter()-started
        started=time.perf_counter();exe=lowered.compile();compile_s=time.perf_counter()-started
        out=exe(*args);jax.block_until_ready(out);timings=[]
        for _ in range(7):
            started=time.perf_counter();out=exe(*args);jax.block_until_ready(out);timings.append(time.perf_counter()-started)
        values[name]={k:np.asarray(v) for k,v in out.items()};rows=[]
        for key,actual in values[name].items():
            rs=1 if data[key].shape[1]==7 else 0;ref=data[key][:,rs];ref64=data64[key][:,rs]
            if actual.ndim==2 and ref.shape[-1]==9 and actual.shape[-1]==8:ref=ref[:,:8];ref64=ref64[:,:8]
            scale=1. if key in ('TSK','TS_URB','TRB_URB4D','TW1_URB4D','TW2_URB4D','TGB_URB4D') else 1.e-3
            limit=2*abs(ref-ref64)+8*np.finfo(np.float32).eps*np.maximum(abs(ref),scale)
            assert actual.shape==ref.shape and np.isfinite(actual).all(),key
            rows.append({'field':key,'max_abs':float(abs(actual-ref).max()),'violations':int(np.sum(abs(actual-ref)>limit))})
        memory=exe.memory_analysis();keys=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')
        results[name]={'lower_s':lower_s,'compile_s':compile_s,'warm_median_s':statistics.median(timings),
            'optimized_hlo_chars':len(exe.as_text()),'memory':{key:getattr(memory,key,None) for key in keys},
            'WRF_frozen_R4_fields':rows,'WRF_frozen_R4_violations':sum(x['violations'] for x in rows)}
        a.out.write_text(json.dumps(results,indent=2)+'\n');print(name,compile_s,results[name]['warm_median_s'],flush=True)
    different={key:{'unequal':int(np.sum(values['scalar_floor'][key]!=values['batched_floor'][key])),
        'max_abs':float(abs(values['scalar_floor'][key]-values['batched_floor'][key]).max())}
        for key in values['scalar_floor'] if not np.array_equal(values['scalar_floor'][key],values['batched_floor'][key])}
    results['regression_only']={'different_fields':different,'baseline_git':a.baseline,
        'baseline_function_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
        'candidate_function_sha256':hashlib.sha256(inspect.getsource(new_building).encode()).hexdigest()}
    a.out.write_text(json.dumps(results,indent=2)+'\n');print('regression differing fields',different,flush=True)


if __name__=='__main__':main()
