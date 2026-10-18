"""PV / green-roof BEP+BEM caller versus pristine WRF (lane oracle fixtures).

Same gate as proofs/v040/urban_bep_bem/bem_driver_parity.py: REAL8 2e-11+2e-12|ref|;
REAL4 2|ref4-ref8|+8eps*max(|ref|,scale) (WRF's own REAL4-vs-REAL8 spread), optional
ROUND01 budget columns of the matched ordinary-roof arm reported separately, never mixed in.
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v040/urban_bep_bem'));sys.path.insert(0,str(HERE))
import jax, jax.numpy as jnp, numpy as np  # noqa: E402
from bem_driver_parity import unpack, packed  # noqa: E402
from conditioned_gate import comparison, finite_reference  # noqa: E402
from gpuwrf.physics.urban_bep_static import read_bep_classes  # noqa: E402
from gpuwrf.physics.urban_bem_static import bem_class  # noqa: E402
from gpuwrf.physics import urban_bem_model as M  # noqa: E402
from gpuwrf.physics.urban_bep_blend import noahmp_bep_blend  # noqa: E402

LEVEL=lambda value:value.transpose(0,2,1).reshape(value.shape[0],-1)
LAYER=lambda value:value.transpose(0,2,3,1).reshape(value.shape[0],-1)
HISTORY={'ep_pv':'EP_PV_URB3D','qgr':'QGR_URB3D','tgr':'TGR_URB3D','draingr':'DRAINGR_URB3D'}
GREEN={'gr_temperature':('TRV_URB4D',LAYER),'gr_moisture':('QR_URB4D',LAYER),'gr_sensible':('SFRV_URB3D',LEVEL),
       'gr_latent':('LFRV_URB3D',LEVEL),'gr_drain':('DRAIN_URB4D',LEVEL)}


def green_extra(data,dtype):
    """Registry-packed (ind_grd/ind_zdf: level slowest, direction fastest) -> BemState.extra."""
    f=lambda name:jnp.asarray(data[name][:,0],dtype)
    layer=lambda name:f(name).reshape(-1,18,10,2).transpose(0,3,1,2)
    level=lambda name:f(name).reshape(-1,18,2).transpose(0,2,1)
    q=layer('QR_URB4D')
    return dict(gr_temperature=layer('TRV_URB4D'),gr_moisture=q,gr_moisture0=q[:,0,-1,0],gr_sensible=level('SFRV_URB3D'),
                gr_latent=level('LFRV_URB3D'),gr_drain=level('DRAIN_URB4D'),qgr=f('QGR_URB3D'),tgr=f('TGR_URB3D'),
                draingr=f('DRAINGR_URB3D'))


def _table(save,arm,workdir):
    """The arm's URBPARM.TBL, regenerated from run_oracle.ARMS and checked against the fixture manifest."""
    import hashlib, json as _json
    from run_oracle import ARMS, table
    text=table(ARMS[arm]);path=Path(workdir)/f'URBPARM_{arm}.TBL';path.write_text(text)
    manifest=_json.loads((save/'manifest.json').read_text())
    for precision in ('fp64','fp32'):
        assert manifest[f'{arm}_{precision}.npz']['table_sha256']==hashlib.sha256(path.read_bytes()).hexdigest(),arm
    return path


def run(save,workdir,out,arms,steps,precisions=('fp64','fp32')):
    assert os.environ['JAX_PLATFORMS']=='cpu';assert jax.devices()[0].platform=='cpu'
    jax.config.update('jax_enable_x64',True);results={}
    for arm in arms:
        tbl=_table(save,arm,workdir)
        for precision in precisions:
            dtype=np.float64 if precision=='fp64' else np.float32
            data=dict(np.load(save/f'{arm}_{precision}.npz'));ref64=dict(np.load(save/f'{arm}_fp64.npz'))
            finite_reference(data);finite_reference(ref64,'REAL8 reference')
            _,classes=read_bep_classes(str(tbl),dtype)
            geo=bem_class(classes[1],1,dtype)
            state,forcing,rural,frc,other=unpack(data,precision)
            extra=M.initial_extra(geo,state.roof.shape[0],dtype,smois_top=jnp.zeros(state.roof.shape[0],dtype),levels=18)
            if M.green_config(geo) is not None:
                extra.update(green_extra(data,dtype))
            state=state._replace(extra=extra)
            def model(state,forcing,rural,frc,other):
                latitude,direct,diffuse,rain,qv=other
                result=M.bem_column_step(state,forcing,geo,latitude=latitude,sw_direct=direct,sw_diffuse=diffuse,rain=rain,qv=qv,fraction=frc)
                surface,pbl=noahmp_bep_blend(result,forcing,rural,frc,raw_moisture=result.moisture_source)
                values=packed(result.state._replace(extra={}),surface,pbl,result.diagnostics)
                values.update({HISTORY[k]:v for k,v in result.state.extra.items() if k in HISTORY})
                values.update({GREEN[k][0]:GREEN[k][1](v) for k,v in result.state.extra.items() if k in GREEN})
                return result.state,values
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
                        scale=1. if name in ['TSK','TS_URB','TRB_URB4D','TW1_URB4D','TW2_URB4D','TGB_URB4D','T_PV_URB3D','TRV_URB4D','TGR_URB3D'] else 1.e-3
                        limit=2*np.abs(ref-r64)+8*np.finfo(np.float32).eps*np.maximum(np.abs(ref),scale)
                    row=dict(step=s,field=name,**comparison(actual,ref,limit,0.))
                    row['byte']=bool(np.array_equal(actual.astype(dtype),ref.astype(dtype)))
                    rows.append(row)
            key=f'{arm}_{precision}'
            results[key]=dict(wall_s=time.perf_counter()-start,violations=sum(r['violations'] for r in rows),
                              byte_fields=sum(r['byte'] for r in rows),field_steps=len(rows),fields=rows)
            if out is not None:out.write_text(json.dumps(results,indent=2)+'\n')
            worst={}
            for r in rows:
                if r['violations']:worst[r['field']]=max(worst.get(r['field'],0),r.get('max_ratio',r['violations']))
            results[key]['worst']=worst
            print(key,'violations',results[key]['violations'],'byte',results[key]['byte_fields'],'/',len(rows),
                  'bad',{k:round(float(v),2) for k,v in sorted(worst.items())},flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--steps',type=int,default=6)
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);run(a.save,a.work,a.out,a.arms,a.steps)
