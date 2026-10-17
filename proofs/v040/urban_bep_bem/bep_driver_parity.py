"""Full BEP + literal Noah-MP blend vs the staged pristine caller (both REALs)."""
from functools import partial
import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.urban_bep_static import read_bep_classes
from gpuwrf.config.paths import wrf_run_dir  # GPUWRF_WRF_ROOT or <repo>/data/wrf_pristine/WRF
from gpuwrf.physics.urban_bep_model import BepState,BepForcing,bep_column_step
from gpuwrf.physics.urban_bep_blend import RuralSurface,noahmp_bep_blend
from conditioned_gate import load_budget,column_budget,finite_reference,comparison


def unpack(data,precision):
    dtype=np.float64 if precision=='fp64' else np.float32
    field=lambda name: jnp.asarray(data[name][:,0],dtype=dtype)
    roof=field('TRB_URB4D').reshape(-1,18,10,2).transpose(0,3,1,2)
    w1=field('TW1_URB4D').reshape(-1,18,10,2).transpose(0,3,1,2)
    w2=field('TW2_URB4D').reshape(-1,18,10,2).transpose(0,3,1,2)
    wall=jnp.stack((w1,w2),axis=2).reshape(-1,4,18,10)
    ground=field('TGB_URB4D').reshape(-1,10,2).transpose(0,2,1)
    wh1=field('SFW1_URB3D').reshape(-1,18,2).transpose(0,2,1)
    wh2=field('SFW2_URB3D').reshape(-1,18,2).transpose(0,2,1)
    wh=jnp.stack((wh1,wh2),axis=2).reshape(-1,4,18)
    rh=field('SFR_URB3D').reshape(-1,18,2).transpose(0,2,1)
    state=BepState(roof,wall,ground,wh,rh,field('SFG_URB3D'))
    values=field('IN_VALUES');dz=field('IN_DZ')[:,:8]
    # Literal WRF cumulative interface reconstruction.
    heights=[jnp.zeros((len(dz),),dtype=dz.dtype)]
    for k in range(8):heights.append(heights[-1]+dz[:,k])
    z=jnp.stack(heights,axis=-1)
    forcing=BepForcing(z,field('IN_U')[:,:8],field('IN_V')[:,:8],field('IN_TH')[:,:8],
        field('IN_RHO')[:,:8],field('IN_P')[:,:8],values[:,13],values[:,14],values[:,6],values[:,8],values[:,7],values[:,0])
    rural=RuralSurface(values[:,20],values[:,21],values[:,22],values[:,24],values[:,25],values[:,26],values[:,31])
    return state,forcing,rural,field('IN_FRACTION')


def packed(state,surface,pbl):
    roof=state.roof.transpose(0,2,3,1).reshape(-1,360)
    w1=state.wall[:,::2].transpose(0,2,3,1).reshape(-1,360)
    w2=state.wall[:,1::2].transpose(0,2,3,1).reshape(-1,360)
    data={'TRB_URB4D':roof,'TW1_URB4D':w1,'TW2_URB4D':w2,
          'TGB_URB4D':state.ground.transpose(0,2,1).reshape(-1,20),
          'SFW1_URB3D':state.wall_heat[:,::2].transpose(0,2,1).reshape(-1,36),
          'SFW2_URB3D':state.wall_heat[:,1::2].transpose(0,2,1).reshape(-1,36),
          'SFR_URB3D':state.roof_heat.transpose(0,2,1).reshape(-1,36),'SFG_URB3D':state.ground_heat}
    for name in ['a_u','a_v','a_t','a_q','a_e','b_u','b_v','b_t','b_q','b_e']:
        data[name.upper()+'_BEP']=getattr(pbl,name)
    data.update(DLG_BEP=pbl.ground_length,DL_U_BEP=pbl.building_length,SF_BEP=pbl.surface,VL_BEP=pbl.volume)
    for name,value in [('TSK',surface.skin),('HFX',surface.heat),('QFX',surface.moisture),('LH',surface.latent),
        ('GRDFLX',surface.ground),('ALBEDO',surface.albedo),('EMISS',surface.emissivity),('UST',surface.ust),
        ('TS_URB',surface.urban_skin),('SH_URB',surface.urban_heat),('LH_URB',surface.urban_latent),
        ('G_URB',surface.urban_ground),('RN_URB',surface.urban_net_radiation)]:data[name]=value
    return data


def run(save,out,budget=None):
    assert os.environ['JAX_PLATFORMS']=='cpu';assert jax.devices()[0].platform=='cpu'
    jax.config.update('jax_enable_x64',True)
    results={};budgets=load_budget(budget)
    for precision in ['fp64','fp32']:
        data=dict(np.load(save/f'urban2_{precision}_ac_on.npz'))
        finite_reference(data)
        data64=dict(np.load(save/'urban2_fp64_ac_on.npz')) if precision=='fp32' else None
        if data64 is not None:finite_reference(data64,'REAL8 reference')
        dtype=np.float64 if precision=='fp64' else np.float32
        _,classes=read_bep_classes(str(wrf_run_dir()/'URBPARM.TBL'),dtype)
        geo=classes[1];state,forcing,rural,frc=unpack(data,precision)
        def model(state,forcing,rural,frc):
            result=bep_column_step(state,forcing,geo)
            surface,pbl=noahmp_bep_blend(result,forcing,rural,frc)
            return result.state,packed(result.state,surface,pbl)
        step=jax.jit(model);rows=[];start=time.perf_counter()
        for s in range(1,7):
            state,values=step(state,forcing,rural,frc)
            for name,value in values.items():
                actual=np.asarray(value);reference_step=s if data[name].shape[1]==7 else s-1
                ref=data[name][:,reference_step]
                if actual.ndim==2 and ref.shape[-1]==9 and actual.shape[-1]==8:ref=ref[:,:8]
                assert actual.shape==ref.shape,(name,actual.shape,ref.shape)
                assert np.isfinite(actual).all(),name
                # Fixed full-driver gate before first candidate measurement.
                if precision=='fp64':limit=2e-11+2e-12*np.abs(ref)
                else:
                    ref64=data64[name][:,reference_step]
                    if ref64.shape!=ref.shape:ref64=ref64[:,:8]
                    scale=1. if name in ['TSK','TS_URB','TRB_URB4D','TW1_URB4D','TW2_URB4D','TGB_URB4D'] else 1.e-3
                    limit=2.*np.abs(ref-ref64)+8.*np.finfo(np.float32).eps*np.maximum(np.abs(ref),scale)
                extra=column_budget(budgets,'urban2_ac_on',name,s,actual.shape,data[name].shape[1]) if precision=='fp32' else 0.
                rows.append(dict(step=s,field=name,**comparison(actual,ref,limit,extra)))
        results[precision]={'wall_s':time.perf_counter()-start,'fields':rows,'violations':sum(r['violations'] for r in rows)}
    out.write_text(json.dumps(results,indent=2)+'\n')
    print({p:{'violations':r['violations'],'wall_s':r['wall_s']} for p,r in results.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--budget',type=Path)
    args=parser.parse_args();run(args.save,args.out,args.budget)
