"""Bind paid v013 singleton results across unrelated main composition.

Compare this receipt's actual input bytes, active output ABI, complete canonical
CPU jaxpr and every nested closed constant. Trace only, no numerical rerun.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.tree.resolve();sys.path.insert(0,str(root/'src'))
    import jax
    import numpy as np
    spec=importlib.util.spec_from_file_location('standard_smoke',root/'tests/test_v013_operational_smoke.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert jax.devices()[0].platform=='cpu'
    def fingerprint(value):
        x=np.asarray(value)
        return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.tobytes()).hexdigest())
    def constants(value,path='root'):
        rows={}
        if hasattr(value,'jaxpr') and hasattr(value,'consts'):
            rows.update({f'{path}/const{i}':fingerprint(x) for i,x in enumerate(value.consts)})
            rows.update(constants(value.jaxpr,path+'/jaxpr'))
        elif hasattr(value,'eqns'):
            for i,e in enumerate(value.eqns):
                for key,x in sorted(e.params.items()):rows.update(constants(x,f'{path}/eq{i}/{key}'))
        elif isinstance(value,(tuple,list)):
            for i,x in enumerate(value):rows.update(constants(x,f'{path}/{i}'))
        elif isinstance(value,dict):
            for key,x in sorted(value.items()):rows.update(constants(x,f'{path}/{key}'))
        return rows
    report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        scope='CPU trace identity of exact standard v013 forcing + post-RK MP; regression reuse only',rows={})
    for mp in (50,51,52,53,55):
        grid=module._grid();state=module._base_state(grid).ensure_conditional_leaves(mp_physics=mp)
        nml=module._namelist(grid,mp_physics=mp,bl_pbl_physics=0,sf_sfclay_physics=0,cu_physics=0)
        module._resolve_operational_suite(nml);carry=module.initial_operational_carry(state)
        def call(c):
            after=module._physics_step_forcing(c,nml,0.0,run_radiation=False).state
            return module._apply_post_rk_microphysics(after,nml) if module._microphysics_wrf_order_enabled() else after
        program,shape=jax.make_jaxpr(call,return_shape=True)(carry)
        # Exact canonicalization used by the established o1-nlbind identity.
        text=re.sub(r'name_and_src_info=[^\]\n]*','',str(program))
        text=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
        row=dict(program_sha256=hashlib.sha256(text.encode()).hexdigest(),
            inputs=[fingerprint(x) for x in jax.tree.leaves(carry)],closed_constants=constants(program),
            active_outputs={n:dict(shape=list(getattr(shape,n).shape),dtype=str(getattr(shape,n).dtype))
                            for n in shape.__slots__ if getattr(shape,n) is not None},
            writer_profiles_inactive=all(getattr(shape,n,None) is None and getattr(state,n,None) is None
                                         for n in ('rimi','exch_tke')))
        assert row['writer_profiles_inactive']
        report['rows'][str(mp)]=row;a.out.write_text(json.dumps(report,indent=2)+'\n')
        print('PROGRAM',mp,row['program_sha256'],len(row['closed_constants']),flush=True)
        del program,shape,state,carry
        jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)


if __name__=='__main__':main()
