"""Two-step compiled MYNN component gate on real WN3 d03 columns.

Every physical/carried result comes ONLY from the production native call.
The diagnostic twin receives the exact same state/surface operands and returns
only history. This selected-column component screen does not replace coupled GPU
validation. An old-field byte failure stops before history values are saved.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',choices=('wn3_d03_day','wn3_d03_night'),required=True)
    p.add_argument('--barrier',choices=('none','output','input'),default='output')
    p.add_argument('--edmf',type=int,choices=(0,1),default=1)
    p.add_argument('--platform',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--export-cache',type=Path)
    p.add_argument('--columns',type=int,default=32)
    args=p.parse_args()
    if args.platform=='cpu':
        assert os.environ.get('JAX_PLATFORMS')=='cpu'
        assert not Path('/tmp/wrf_gpu2_quiet').exists()
        assert args.export_cache is None or os.environ.get('BUDGET_CPU_ALL_PHASES')=='1', 'CPU exports require a private all-phase smoke cache'
    else:
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1','official GPU lock required'
        assert os.environ.get('JAX_PLATFORMS')!='cpu'
        assert args.export_cache is not None
        assert args.export_cache.resolve()==Path(os.environ['GPUWRF_JAX_CACHE_DIR']).resolve()
    root=args.source.resolve()
    sys.path[:0]=[str(root/'src'),str(root/'tests/v025/t_cfix_mynn')]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import f20_port as F
    from gpuwrf.kernels import phys_mynn_columns as N
    from gpuwrf.physics import mynn_budget_twin as T
    from gpuwrf.physics.surface_layer import mynn_driver_surface_fluxes
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    assert jax.devices()[0].platform==args.platform
    fixture=root/'tests/v025/t_cfix_mynn/fixtures/f20_columns.npz'
    meta=json.loads(fixture.with_suffix('.json').read_text())[args.case]
    with np.load(fixture) as f:
        cols={k.split('__',1)[1]:f[k].copy() for k in f.files if k.startswith(args.case+'__')}
    assert 1<=args.columns<=len(cols['th'])
    cols={k:v[:args.columns].copy() for k,v in cols.items()}
    cols['qsfc']=cols['qv'][:,0].copy();cols['znt']=np.full_like(cols['tsk'],.1)
    state,_flux,_=F.column_inputs(cols)
    constants={k:jnp.asarray(cols[k],jnp.float32) for k in ('hfx','qfx','tsk','ust','wspd','psfc','xland')}
    def surface(s):
        c=constants
        drv=mynn_driver_surface_fluxes(c['hfx'],c['qfx'],s.rho[:,0],s.qv[:,0],
            c['tsk'],s.p[:,0],exner=s.exner[:,0],pbl_limits=True)
        return SurfaceFluxes(ustar=c['ust'],theta_flux=drv.flt,qv_flux=drv.flqv,
            tau_u=-(c['ust']*c['ust'])*s.u[:,0]/c['wspd'],
            tau_v=-(c['ust']*c['ust'])*s.v[:,0]/c['wspd'],rhosfc=s.rho[:,0],
            fltv=drv.fltv,xland=c['xland'],t_skin=c['tsk'],wspd=c['wspd'],psfc=c['psfc'])
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='wrf_gpu2.w44_budget.twin_component.v1',source=str(root),case=args.case,
        platform=args.platform,scope=__doc__,barrier=args.barrier,edmf=args.edmf,steps=2,
        columns=int(state.theta.shape[0]),levels=int(state.theta.shape[1]),
        production_source_sha256=T.PRODUCTION_SOURCE_SHA256,
        column_slots=list(state.__slots__),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        scenario_census={k:int(np.count_nonzero(cols[k])) for k in ('qc','qi','qs','qni')},
        cpu_xla_flags=os.environ.get('XLA_FLAGS'),arms={})
    record=args.out/'result.json'
    record.write_text(json.dumps(report,indent=2)+'\n')
    results,programs={},{}
    for label,enabled in (('off',False),('on',True)):
        def run(initial):
            s=initial;physical=[];history=[]
            for _ in range(2):
                f=surface(s);before=s
                # This is always the unchanged production function/return.
                s,pblh=N.step_mynn_columns_native(before,float(meta['dt']),surface=f,
                    edmf=bool(args.edmf),dx=float(meta['dx']))
                physical.append((s,pblh))
                if enabled:
                    history.append(T.step(before,float(meta['dt']),surface=f,edmf=bool(args.edmf),
                        dx=float(meta['dx']),barrier=args.barrier))
            return tuple(physical),tuple(history)
        jax.clear_caches()
        begin=time.perf_counter()
        lowered=jax.jit(run).lower(state)
        program=lowered.compile()
        programs[label]=program
        results[label]=jax.block_until_ready(program(state))
        m=program.memory_analysis()
        report['arms'][label]=dict(compile_first_s=time.perf_counter()-begin,memory={k:int(getattr(m,k))
            for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
        if args.export_cache is not None:
            from gpuwrf.runtime.aot_precompile import _serialize_domain_blob
            name=f'budget_component_{args.case}_edmf{args.edmf}_{label}'
            exported=_serialize_domain_blob(name,program,args.export_cache,lowered=lowered)
            report['arms'][label]['program_name']=name
            report['arms'][label]['aot_export']=exported
            record.write_text(json.dumps(report,indent=2)+'\n')
            assert exported['aot_written'],exported
        record.write_text(json.dumps(report,indent=2)+'\n')
    fields={}
    for i in range(2):
        left,lt=jax.tree.flatten_with_path(results['off'][0][i])
        right,rt=jax.tree.flatten_with_path(results['on'][0][i])
        assert lt==rt and len(left)==len(right)
        for (path,a),(other,b) in zip(left,right,strict=True):
            assert path==other
            a,b=np.asarray(a),np.asarray(b)
            assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
            d=a.astype(np.float64)-b.astype(np.float64)
            fields[f'step{i+1}/{path}']=dict(byte_equal=a.tobytes()==b.tobytes(),
                max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))))
    report.update(fields=fields,byte_fields=sum(v['byte_equal'] for v in fields.values()),total_fields=len(fields))
    if report['byte_fields']!=len(fields):
        report['verdict']='STOP_NONBYTE_TWIN_COMPONENT'
        record.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(verdict=report['verdict'],byte_fields=report['byte_fields'],total_fields=len(fields))),flush=True)
        raise SystemExit(2)
    arrays={}
    from gpuwrf.physics.mynn_budget import TKE_NAMES,EDMF_NAMES,PBL_NAMES
    names=TKE_NAMES+EDMF_NAMES+PBL_NAMES
    for i,values in enumerate(results['on'][1]):
        assert len(values)==len(names)==21
        for name,value in zip(names,values,strict=True):
            value=np.asarray(value);assert np.isfinite(value).all(),name
            arrays[f'step{i+1}__{name}']=value
    # Save the accepted production inputs for a same-input original driver at
    # step 2; they are NOT original-WRF trajectory truth.
    for i,(s,_pblh) in enumerate(((state,None),results['off'][0][0])):
        for key in s.__slots__:
            value=getattr(s,key)
            if value is not None:
                arrays[f'input{i+1}__{key}']=np.asarray(value)
    np.savez(args.out/'accepted_history_and_inputs.npz',**arrays)
    times={label:[] for label in programs}
    for label in ('off','on','on','off')*2:
        begin=time.perf_counter();jax.block_until_ready(programs[label](state));times[label].append((time.perf_counter()-begin)*1000)
    report[args.platform+'_wall_ms']={k:dict(samples=v,median=statistics.median(v)) for k,v in times.items()}
    if args.platform=='gpu':report['device_memory_stats']=jax.devices()[0].memory_stats()
    report['verdict']='TWIN_COMPONENT_2STEP_BYTE'
    record.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(verdict=report['verdict'],byte_fields=report['byte_fields'],total_fields=len(fields))),flush=True)


if __name__=='__main__':main()
