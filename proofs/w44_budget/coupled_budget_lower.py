"""CPU type/lowering screen of the public budget program on real WN3 d01.

No model execution, GPU compilation, physical BYTE or WRF fidelity claim.
The production GPU path remains unmodified; CPU-only Pallas interpretation is
used here solely to expose scan carry shapes and the diagnostic dataflow.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only-on',action='store_true')
    a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    root=a.source.resolve();sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from gpuwrf.backends import pallas as facade
    from gpuwrf.contracts import state as S
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as O
    from gpuwrf.diagnostics.mynn_budget import seed,configuration
    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    import cpu_safe
    assert jax.devices()[0].platform=='cpu'
    assert Path(O.__file__).resolve().is_relative_to(root)
    original=pl.pallas_call
    def interpreted(*args,**kwargs):
        kwargs['interpret']=True;kwargs.pop('compiler_params',None)
        return original(*args,**kwargs)
    pl.pallas_call=interpreted;facade.cuda_native=lambda:False
    S._gpu_device=lambda:jax.devices('cpu')[0];cpu_safe.install()
    a.out.mkdir(exist_ok=False)
    config=pipeline.NestedPipelineConfig(a.input_dir,a.out/'unused_output',a.out/'init_proof',hours=3,max_dom=1)
    start=time.perf_counter()
    _h,bundles,_x,_rs,_dt,carries=pipeline._load_domains(config,('d01',))
    initial=carries['d01'];base=bundles['d01'].namelist
    record=dict(scope=__doc__,platform='cpu',source_commit=subprocess.check_output(
        ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),init_s=time.perf_counter()-start,
        shape=list(initial.state.theta.shape),steps=2,arms={})
    path=a.out/'result.json'
    for name,on in (('off',False),('on',True)):
        if a.only_on and not on:continue
        nl=replace(base,tke_budget=int(on),bl_mynn_output=int(on))
        O._resolve_operational_suite(nl)
        carry=jax.tree.map(lambda x:jnp.array(x,copy=True),initial)
        if on:carry=carry.replace(mynn_budget=seed(carry.state.theta.shape,configuration(nl)))
        carry=O._protect_initial_carry_seeds(carry,nl)
        jax.clear_caches();start=time.perf_counter()
        lower=O._advance_chunk_fori.lower(carry,nl,1,O.build_clock_base(nl),
            n_steps=2,cadence=int(nl.radiation_cadence_steps))
        hlo=lower.compiler_ir('hlo').as_hlo_text()
        debug_ir=lower.as_text(debug_info=True)
        record['arms'][name]=dict(lower_s=time.perf_counter()-start,static_config_hash=static_config_hash(nl),
            hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),hlo_chars=len(hlo),
            twin_traced='mynn_budget_twin.py' in debug_ir,
            debug_ir_sha256=hashlib.sha256(debug_ir.encode()).hexdigest(),debug_ir_chars=len(debug_ir),
            history_leaves=len(jax.tree.leaves(carry.mynn_budget)),
            dt=float(nl.dt_s),cadence=int(nl.radiation_cadence_steps))
        path.write_text(json.dumps(record,indent=2)+'\n');print(name,json.dumps(record['arms'][name]),flush=True)
    if not a.only_on:assert record['arms']['off']['history_leaves']==0
    assert record['arms']['on']['history_leaves']==21
    if not a.only_on:assert record['arms']['off']['static_config_hash']!=record['arms']['on']['static_config_hash']
    assert record['arms']['on']['twin_traced'],'budget dataflow disappeared before compilation'
    record['verdict']='PUBLIC_BUDGET_2STEP_CPU_LOWER'
    path.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
