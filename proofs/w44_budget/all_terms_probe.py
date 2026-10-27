"""Compiled CPU native-MYNN history vs unmodified original public driver.

Pre-register before comparing: all old outputs BYTE (diagnostic admission);
six raw mean tendencies retain the established 1e-3 absolute bound; nonzero
TKE/plume profiles use a 5% per-column profile-max bound, no fitted absolute
floor; original exact-zero sub/det/top conventions are exact. Closure compares
the original residual using the sum of the five registered term bounds. These
component gates do not replace the final compiled GPU/coupled gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--edmf',type=int,choices=(0,1),required=True)
    p.add_argument('--case',default='wn3_d03_day')
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    root=args.source.resolve()
    sys.path[:0]=[str(root/'src'),str(root/'tests/v025/t_cfix_mynn')]
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.kernels import phys_mynn_columns as N
    from gpuwrf.physics import mynn_budget as B
    from gpuwrf.physics import mynn_pbl as P
    import f20_port as F
    assert jax.devices()[0].platform=='cpu'
    fixture=root/'tests/v025/t_cfix_mynn/fixtures/f20_columns.npz'
    meta=json.loads(fixture.with_suffix('.json').read_text())[args.case]
    with np.load(fixture) as f:
        cols={k.split('__',1)[1]:f[k].copy() for k in f.files if k.startswith(args.case+'__')}
    cols['qsfc']=cols['qv'][:,0].copy();cols['znt']=np.full_like(cols['tsk'],.1)
    state,flux,_=F.column_inputs(cols)
    with np.load(args.reference/(args.case+'.npz')) as f:
        expected={k:f[k].copy() for k in f.files}
    assert all(np.isfinite(v).all() for v in expected.values())
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='wrf_gpu2.w44_budget.all_terms_component.v1',platform='cpu',edmf=args.edmf,
        case=args.case,source=str(root),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((args.reference/(args.case+'.npz')).read_bytes()).hexdigest(),
        scenario_census={key:int(np.count_nonzero(cols[key])) for key in ('qc','qi','qs','qni')},
        tolerance=dict(mean_absolute=1.e-3,history_profile_relative=.05,zero_conventions='exact'),
        env={k:v for k,v in os.environ.items() if k.startswith('GPUWRF_MYNN') or k=='XLA_FLAGS'},arms={})
    output=args.out/'result.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    outputs={}
    for arm in ('off','on'):
        initial=state if arm=='off' else state.replace(tke_budget=tuple(jnp.zeros_like(state.theta) for _ in B.TKE_NAMES),
            edmf_output=tuple(jnp.zeros_like(state.theta) for _ in B.EDMF_NAMES))
        jax.clear_caches()
        start=time.perf_counter()
        program=jax.jit(lambda s,f:N.step_mynn_columns_native(s,float(meta['dt']),surface=f,
            edmf=bool(args.edmf),dx=float(meta['dx']))).lower(initial,flux).compile()
        outputs[arm]=jax.block_until_ready(program(initial,flux))
        memory=program.memory_analysis()
        result['arms'][arm]=dict(compile_first_s=time.perf_counter()-start,
            memory={key:int(getattr(memory,key)) for key in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
        output.write_text(json.dumps(result,indent=2)+'\n')
    off,on=outputs['off'][0],outputs['on'][0]
    fields={}
    for key in state.__slots__:
        if key in ('tke_budget','edmf_output') or getattr(off,key) is None:
            continue
        a,b=np.asarray(getattr(off,key)),np.asarray(getattr(on,key))
        assert np.isfinite(a).all() and np.isfinite(b).all(),key
        d=a.astype(np.float64)-b.astype(np.float64)
        fields[key]=dict(byte_equal=a.tobytes()==b.tobytes(),max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))))
    a,b=np.asarray(outputs['off'][1]),np.asarray(outputs['on'][1]);assert np.isfinite(a).all() and np.isfinite(b).all()
    fields['pblh']=dict(byte_equal=a.tobytes()==b.tobytes(),max_abs=float(np.max(np.abs(a-b))))
    result['old_fields']=fields
    result['byte_fields']=sum(v['byte_equal'] for v in fields.values());result['total_fields']=len(fields)
    if result['byte_fields']!=len(fields):
        result['verdict']='STOP_NONBYTE_COMPONENT';output.write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(2)
    actual=dict(zip(B.TKE_NAMES,on.tke_budget));actual.update(zip(B.EDMF_NAMES,on.edmf_output))
    actual.update(zip(B.PBL_NAMES,B.mean_tendencies(state,on,float(meta['dt']))))
    report={};bounds={}
    for name,value in actual.items():
        value=np.asarray(value);ref=expected['dqke' if name=='Dtke' else name.lower()]
        assert value.shape==ref.shape and np.isfinite(value).all() and np.isfinite(ref).all(),name
        delta=np.abs(value.astype(np.float64)-ref.astype(np.float64))
        if name in B.PBL_NAMES:
            bound=np.full_like(ref[:,0],1.e-3,dtype=np.float64)
        else:
            bound=.05*np.max(np.abs(ref.astype(np.float64)),axis=-1)
        if name in B.EDMF_NAMES[6:] or (not args.edmf and name in B.EDMF_NAMES):
            passed=not np.count_nonzero(value) and not np.count_nonzero(ref)
        else:
            passed=bool(np.all(delta<=bound[:,None]))
        if name in B.TKE_NAMES:
            passed=passed and not np.count_nonzero(value[:,-1])
        report[name]=dict(max_abs=float(delta.max()),rms=float(np.sqrt(np.mean(delta*delta))),
            max_reference=float(np.max(np.abs(ref))),columns_exceeding=int(np.count_nonzero(np.any(delta>bound[:,None],axis=-1))),passed=bool(passed))
        bounds[name]=bound
    residual=lambda v:v['Dtke']-(v['qSHEAR']+v['qBUOY']+v['qWT']-v['qDISS'])
    ref={n:expected['dqke' if n=='Dtke' else n.lower()].astype(np.float64) for n in B.TKE_NAMES}
    candidate={n:np.asarray(actual[n],np.float64) for n in B.TKE_NAMES}
    diff=np.abs(residual(candidate)-residual(ref));bound=sum(bounds[n] for n in B.TKE_NAMES)
    result['terms']=report;result['closure']=dict(max_abs=float(diff.max()),passed=bool(np.all(diff<=bound[:,None])),
        original_max_abs=float(np.max(np.abs(residual(ref)))),note='original residual comparison, not zero-closure enforcement')
    result['verdict']='COMPONENT_ORACLE_PASS' if all(v['passed'] for v in report.values()) and result['closure']['passed'] else 'COMPONENT_ORACLE_FAIL'
    np.savez(args.out/'accepted_outputs.npz',**{k:np.asarray(v) for k,v in actual.items()})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],byte_fields=result['byte_fields'],total_fields=len(fields),
        failed=[k for k,v in report.items() if not v['passed']])),flush=True)
    if result['verdict']!='COMPONENT_ORACLE_PASS':raise SystemExit(2)


if __name__=='__main__':main()
