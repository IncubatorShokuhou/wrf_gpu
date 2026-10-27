"""Actual private ENT initializer deletion, checked against original WRF."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path[:0]=[str(args.source/'src'),str(args.source/'tests/v025/t_cfix_mynn')]
    import jax
    import numpy as np
    import f20_port as F
    from gpuwrf.physics import mynn_budget_twin as T
    from gpuwrf.kernels import phys_mynn_budget_plume as B
    from gpuwrf.kernels import phys_mynn_plume as production
    fixture=args.source/'tests/v025/t_cfix_mynn/fixtures/f20_columns.npz'
    meta=json.loads(fixture.with_suffix('.json').read_text())['wn3_d03_day']
    with np.load(fixture) as f:cols={k.split('__',1)[1]:f[k].copy() for k in f.files if k.startswith('wn3_d03_day__')}
    cols['qsfc']=cols['qv'][:,0].copy();cols['znt']=np.full_like(cols['tsk'],np.float32(.1))
    state,flux,_=F.column_inputs(cols)
    with np.load('<USER_HOME>/wrf_gpu2_lanes/w44-budget/P14_pristine_reference/wn3_d03_day.npz') as f:ref=f['edmf_ent'].copy()
    assert np.count_nonzero(ref[:,0])>0
    original=B.dmp_mf_columns_native;production_function=production.dmp_mf_columns_native
    values={}
    try:
        for label,value in (('healthy',.001),('deleted',0.)):
            B.dmp_mf_columns_native=B._build(_surface_ent=value)[0]
            jax.clear_caches()
            program=jax.jit(lambda s,f:T.step(s,float(meta['dt']),surface=f,edmf=True,dx=float(meta['dx']))[9]).lower(state,flux).compile()
            result=np.asarray(jax.block_until_ready(program(state,flux)))
            assert np.isfinite(result).all() and np.isfinite(ref).all()
            bound=.05*np.max(np.abs(ref),axis=-1)
            passed=bool(np.all(np.abs(result-ref)<=bound[:,None]))
            values[label]=dict(passed=passed,max_abs=float(np.max(np.abs(result-ref))),
                               columns_failed=int(np.count_nonzero(np.any(np.abs(result-ref)>bound[:,None],axis=-1))))
        assert values['healthy']['passed'],'healthy ENT source must pass first'
        try:
            assert values['deleted']['passed'],'ENT surface initializer deletion numeric mismatch'
        except AssertionError as error:
            assert str(error)=='ENT surface initializer deletion numeric mismatch'
            values['genuine_assertion_kills']=1;values['setup_errors']=0
        else:
            raise AssertionError('ENT deletion survived')
    finally:
        B.dmp_mf_columns_native=original
    assert production.dmp_mf_columns_native is production_function,'production function was patched'
    args.out.write_text(json.dumps(values,indent=2)+'\n');print(json.dumps(values))


if __name__=='__main__':main()
