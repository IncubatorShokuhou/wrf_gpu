"""Complete ordinary-roof BEP+BEM caller versus pristine WRF savepoints.

The oracle caller sets SWDDIR=.8*SWDOWN and SWDDIF=.2*SWDOWN; reproduce
those typed fixture operations only here. Production takes actual radiation.
"""
import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from bep_driver_parity import unpack as bep_unpack, packed as bep_packed
from gpuwrf.physics.urban_bep_static import read_bep_classes
from gpuwrf.config.paths import wrf_run_dir  # GPUWRF_WRF_ROOT or <repo>/data/wrf_pristine/WRF
from gpuwrf.physics.urban_bem_static import bem_class
from gpuwrf.physics.urban_bem_model import BemState,bem_column_step
from gpuwrf.physics.urban_bep_blend import noahmp_bep_blend
from conditioned_gate import load_budget,column_budget,finite_reference,comparison


def unpack(data,precision):
    # BEP and BEM have different packed wall sizes; share only forcing parsing.
    reduced=dict(data)
    for key in ['TW1_URB4D','TW2_URB4D']:reduced[key]=data[key][...,:360]
    for key in ['SFW1_URB3D','SFW2_URB3D']:reduced[key]=data[key][...,:36]
    base,forcing,rural,frc=bep_unpack(reduced,precision)
    dtype=np.float64 if precision=='fp64' else np.float32
    field=lambda name:jnp.asarray(data[name][:,0],dtype)
    def pair(k1,k2,material=False):
        if material:
            a=field(k1).reshape(-1,15,18,10,2).transpose(0,1,4,2,3)
            b=field(k2).reshape(-1,15,18,10,2).transpose(0,1,4,2,3)
            return jnp.stack((a,b),axis=3).reshape(-1,15,4,18,10)
        a=field(k1).reshape(-1,15,18,2).transpose(0,1,3,2)
        b=field(k2).reshape(-1,15,18,2).transpose(0,1,3,2)
        return jnp.stack((a,b),axis=3).reshape(-1,15,4,18)
    level=lambda key:field(key).reshape(-1,18,2).transpose(0,2,1)
    extra={key:field(key) for key in ['DRAIN_URB4D','SFRV_URB3D','LFRV_URB3D','TRV_URB4D','QR_URB4D']}
    state=BemState(base.roof,pair('TW1_URB4D','TW2_URB4D',True),base.ground,
        pair('SFW1_URB3D','SFW2_URB3D'),base.roof_heat,base.ground_heat,
        field('TLEV_URB3D').reshape(-1,15,18),field('QLEV_URB3D').reshape(-1,15,18),
        pair('TW1LEV_URB3D','TW2LEV_URB3D'),pair('SFWIN1_URB3D','SFWIN2_URB3D'),
        field('TFLEV_URB3D').reshape(-1,15,10,17,2).transpose(0,1,4,3,2),
        field('TGLEV_URB3D').reshape(-1,15,10,2).transpose(0,1,3,2),
        level('DGR_URB3D'),field('DG_URB3D'),level('LFR_URB3D'),field('LFG_URB3D'),level('T_PV_URB3D'),extra)
    values=np.asarray(data['IN_VALUES'][:,0],dtype)
    other=(jnp.asarray(values[:,5]),jnp.asarray(dtype(.8)*values[:,13]),jnp.asarray(dtype(.2)*values[:,13]),
        jnp.asarray(values[:,17]),field('IN_QV')[:,:8])
    return state,forcing,rural,frc,other


def packed(state,surface,pbl,diagnostics):
    # Common surfaces and mesoscale fields; replace building-sized wall fields.
    from gpuwrf.physics.urban_bep_model import BepState
    base=BepState(state.roof,state.wall[:,0],state.ground,state.wall_heat[:,0],state.roof_heat,state.ground_heat)
    data=bep_packed(base,surface,pbl)
    for side,key in [(0,'TW1_URB4D'),(1,'TW2_URB4D')]:
        data[key]=state.wall[:,:,side::2].transpose(0,1,3,4,2).reshape(-1,5400)
    for value,k1,k2 in [(state.wall_heat,'SFW1_URB3D','SFW2_URB3D'),
            (state.window,'TW1LEV_URB3D','TW2LEV_URB3D'),(state.window_heat,'SFWIN1_URB3D','SFWIN2_URB3D')]:
        for side,key in [(0,k1),(1,k2)]:data[key]=value[:,:,side::2].transpose(0,1,3,2).reshape(-1,540)
    data.update(TLEV_URB3D=state.temperature.reshape(-1,270),QLEV_URB3D=state.humidity.reshape(-1,270),
        TFLEV_URB3D=state.floor.transpose(0,1,4,3,2).reshape(-1,5100),
        TGLEV_URB3D=state.under_ground.transpose(0,1,3,2).reshape(-1,300),
        DG_URB3D=state.ground_water,LFG_URB3D=state.ground_latent)
    for value,key in [(state.roof_water,'DGR_URB3D'),(state.roof_latent,'LFR_URB3D'),(state.pv_temperature,'T_PV_URB3D')]:
        data[key]=value.transpose(0,2,1).reshape(-1,36)
    data.update(state.extra)
    for value,key in zip(diagnostics,['SF_AC_URB3D','LF_AC_URB3D','CM_AC_URB3D','SFVENT_URB3D','LFVENT_URB3D']):data[key]=value
    return data


def run(save,out,precisions,steps,budget=None,actual_out=None):
    assert os.environ['JAX_PLATFORMS']=='cpu';assert jax.devices()[0].platform=='cpu'
    jax.config.update('jax_enable_x64',True);results={};budgets=load_budget(budget);saved={}
    for precision in precisions:
      for ac in ['ac_on','ac_off']:
        dtype=np.float64 if precision=='fp64' else np.float32
        data=dict(np.load(save/f'urban3_{precision}_{ac}.npz'))
        ref64=dict(np.load(save/f'urban3_fp64_{ac}.npz'))
        finite_reference(data)
        finite_reference(ref64,'REAL8 reference')
        _,classes=read_bep_classes(str(wrf_run_dir()/'URBPARM.TBL'),dtype)
        geo=bem_class(classes[1],1,dtype)
        if ac=='ac_off':geo.params['ac_switch']=0
        state,forcing,rural,frc,other=unpack(data,precision)
        def model(state,forcing,rural,frc,other):
            latitude,direct,diffuse,rain,qv=other
            result=bem_column_step(state,forcing,geo,latitude=latitude,sw_direct=direct,sw_diffuse=diffuse,rain=rain,qv=qv,fraction=frc)
            surface,pbl=noahmp_bep_blend(result,forcing,rural,frc,raw_moisture=result.moisture_source)
            return result.state,packed(result.state,surface,pbl,result.diagnostics)
        step=jax.jit(model);rows=[];start=time.perf_counter()
        for s in range(1,steps+1):
            state,values=step(state,forcing,rural,frc,other)
            for name,value in values.items():
                actual=np.asarray(value);rs=s if data[name].shape[1]==7 else s-1;ref=data[name][:,rs]
                if actual.ndim==2 and ref.shape[-1]==9 and actual.shape[-1]==8:ref=ref[:,:8]
                assert actual.shape==ref.shape,(name,actual.shape,ref.shape)
                assert np.isfinite(actual).all(),name
                if precision=='fp64':limit=2e-11+2e-12*np.abs(ref)
                else:
                    r64=ref64[name][:,rs]
                    if r64.shape!=ref.shape:r64=r64[:,:8]
                    scale=1. if name in ['TSK','TS_URB','TRB_URB4D','TW1_URB4D','TW2_URB4D','TGB_URB4D'] else 1.e-3
                    limit=2*np.abs(ref-r64)+8*np.finfo(np.float32).eps*np.maximum(np.abs(ref),scale)
                extra=column_budget(budgets,'urban3_'+ac,name,s,actual.shape,data[name].shape[1]) if precision=='fp32' else 0.
                rows.append(dict(step=s,field=name,**comparison(actual,ref,limit,extra)))
                if actual_out is not None:saved.setdefault(precision+'_'+ac+'__'+name,[]).append(actual)
        key=f'{precision}_{ac}';results[key]=dict(wall_s=time.perf_counter()-start,fields=rows,violations=sum(r['violations'] for r in rows))
        out.write_text(json.dumps(results,indent=2)+'\n');print(key,results[key]['violations'],results[key]['wall_s'],flush=True)
    if actual_out is not None:np.savez_compressed(actual_out,**{key:np.stack(value,axis=1) for key,value in saved.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precisions',nargs='+',default=['fp64','fp32']);p.add_argument('--steps',type=int,default=6)
    p.add_argument('--budget',type=Path);p.add_argument('--actual-out',type=Path)
    a=p.parse_args();run(a.save,a.out,a.precisions,a.steps,a.budget,a.actual_out)
