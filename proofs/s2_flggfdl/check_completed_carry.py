"""Restart/history checks on a retained completed CPU arm; no model re-solve."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import pickle
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path);parser.add_argument('completed',type=Path);parser.add_argument('out',type=Path)
parser.add_argument('--history-only',action='store_true',help='reuse completed FG68 restart BYTE checks; rerun only changed history assertion')
args=parser.parse_args();sys.path.insert(0,str(args.source/'src'))
import jax
import numpy as np
from gpuwrf.io.restart import write_restart,read_restart
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
from gpuwrf.io.flg_gfdl_schema import marker,validate
from gpuwrf.integration.nested_pipeline import _noahmp_surface_diagnostics_for_output
from gpuwrf.integration.daily_pipeline import _surface_diagnostics_for_output
assert jax.devices()[0].platform=='cpu'
with args.completed.open('rb') as stream:carry,namelist,start=pickle.load(stream)
args.out.mkdir(exist_ok=True,parents=True)
code=int(namelist.ra_lw_physics);assert code in (7,99)
validate(carry.radiation_diagnostics,carry.gfdl_radiation,namelist,
         marker(carry.radiation_diagnostics,carry.gfdl_radiation),state_shape=carry.state.qv.shape)


def exact(left,right):
    a,ta=jax.tree.flatten(left);b,tb=jax.tree.flatten(right)
    assert ta==tb and len(a)==len(b)
    for index,(x,y) in enumerate(zip(a,b,strict=True)):
        x,y=np.asarray(x),np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes(),index
    return len(a)


report=dict(scheme=code,model_rerun=False,checks={})
if args.history_only:report['restart_reuse']='FG68 all three full-carry BYTE checks completed before harness list-equality assertion at line62; source/IO unchanged'
for kind in (() if args.history_only else ('pickle_restart','netcdf_restart','checkpoint')):
    path=args.out/('roundtrip.nc' if kind=='netcdf_restart' else kind+'.pkl')
    started=time.perf_counter()
    if kind=='pickle_restart':
        write_restart(carry,namelist,namelist.grid,3,path)
        restored,_nl,_grid,step=read_restart(path,namelist=namelist);assert step==3
    elif kind=='netcdf_restart':
        write_wrfrst_carry(carry,namelist.grid,namelist,path,valid_time=start+timedelta(seconds=3*namelist.dt_s),
                          run_start=start,step_index=3)
        restored,metadata=read_wrfrst_carry(path,namelist=namelist)
    else:
        write_checkpoint(carry.state,namelist,namelist.grid,3,path,runtime_state=carry,land_state=carry.noahmp_land)
        _state,_nl,_grid,step,restored=read_checkpoint_with_runtime_state(path,namelist=namelist);assert step==3
    report['checks'][kind]=dict(BYTE=True,leaves=exact(carry,restored),wall_s=time.perf_counter()-started)
    path.unlink()  # Each disposable serialized arm is cleaned immediately.
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')

subset=('GLW','SWDOWN','OLR','LWUPT','SWUPT','SWDNB','SWDDIR','SWDDIF','SWDDNI')
work=[]
nested=_noahmp_surface_diagnostics_for_output(carry.state,namelist,start,lead_seconds=3*namelist.dt_s,
    noahmp_land=carry.noahmp_land,noahmp_rad=carry.noahmp_rad,radiation_diagnostics=carry.radiation_diagnostics,
    variable_subset=subset,output_radiation_work=work,history_diagnostics=carry.history_diagnostics)
daily=_surface_diagnostics_for_output(carry.state,namelist,start,lead_seconds=3*namelist.dt_s,
    variable_subset=subset,output_radiation_work=work,operational_carry=carry)
assert work and not any(work),f'output recomputed radiation: {work}'
for key in subset:
    assert key in nested and key in daily,key
    assert np.asarray(nested[key]).tobytes()==np.asarray(daily[key]).tobytes(),key
for key in ('OLR','LWUPT','SWUPT','SWDNB'):assert np.count_nonzero(nested[key])==0,key
for key,field in (('GLW','glw'),('SWDOWN','swdown'),('SWDDIR','swddir'),('SWDDIF','swddif'),('SWDDNI','swddni')):
    assert np.asarray(nested[key]).tobytes()==np.asarray(getattr(carry.radiation_diagnostics,field)).tobytes(),key
report['checks']['both_pipeline_history']=dict(BYTE=True,fields=list(subset),radiation_output_calls=0,
    raw_call_counts=work,
    internal_rlwtoa_max=float(np.max(carry.radiation_diagnostics.rlwtoa)),history_OLR_max=float(np.max(nested['OLR'])))
(args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
