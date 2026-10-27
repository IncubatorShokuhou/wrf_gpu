"""Compiled two-step caller regression on a frozen real WN3 cloud State.

This is an adapter/State regression screen. It is not a coupled GPU proof,
an independent WRF oracle, or a replacement for the day/night term gates.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(a.source/'src'))
    import jax
    import numpy as np
    from gpuwrf.coupling import physics_couplers as P
    from gpuwrf.coupling.mynn_budget_adapter import mynn_sources_with_history,_build
    from gpuwrf.diagnostics.mynn_budget import configuration,history_fields
    spec=importlib.util.spec_from_file_location('frozen_cloud',a.source/'tests/v025/fid_cloud/test_sgs_carry.py')
    F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
    state=F._cloud_state()
    assert jax.devices()[0].platform=='cpu'
    a.out.mkdir(exist_ok=False)
    config=configuration(dict(tke_budget=1,bl_mynn_output=1))
    result=dict(scope=__doc__,platform='cpu',shape=list(state.theta.shape),steps=2,arms={})
    results={}
    for label in ('off','on'):
        def run(initial):
            carry=initial;physical=[];history=[]
            for _ in range(2):
                if label=='off':source=P.mynn_adapter_with_source_leaves(carry,18.,restart=True)
                else:
                    source,values=mynn_sources_with_history(carry,18.,budget_config=config,restart=True)
                    history.append(values)
                physical.append(source);carry=source.state
            return tuple(physical),tuple(history)
        jax.clear_caches();start=time.perf_counter()
        program=jax.jit(run).lower(state).compile()
        results[label]=jax.block_until_ready(program(state))
        memory=program.memory_analysis()
        result['arms'][label]=dict(compile_first_s=time.perf_counter()-start,
            memory={k:int(getattr(memory,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')})
    fields={}
    left,lt=jax.tree.flatten_with_path(results['off'][0]);right,rt=jax.tree.flatten_with_path(results['on'][0])
    assert lt==rt
    for (path,x),(other,y) in zip(left,right,strict=True):
        assert path==other
        x,y=np.asarray(x),np.asarray(y)
        assert x.dtype==y.dtype and x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        fields[str(path)]=dict(byte_equal=x.tobytes()==y.tobytes(),max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    result.update(fields=fields,total_fields=len(fields),byte_fields=sum(v['byte_equal'] for v in fields.values()))
    result['verdict']='ADAPTER_2STEP_BYTE' if result['total_fields']==result['byte_fields'] else 'STOP_NONBYTE_ADAPTER'
    if result['verdict']=='ADAPTER_2STEP_BYTE':
        for history in results['on'][1]:
            values=history_fields(history);assert len(values)==21
            assert all(np.isfinite(np.asarray(v)).all() for v in values.values())
        result['production_adapter_sha256']=_build(True,True)[1]
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fields'}))
    if result['verdict']!='ADAPTER_2STEP_BYTE':raise SystemExit(2)


if __name__=='__main__':main()
