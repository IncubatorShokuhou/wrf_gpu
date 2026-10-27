"""Original-WRF QKE seed on snow columns, plus the THLV-only intervention.

Three source-unchanged calls: old default helper; existing return_moments=True
helper selected by the fix; old helper with snow zeroed only for its input.
WRF init THLV excludes snow (driver :675-684); the ordinary SGS-aware THLV adds
it to ice. No forecast or model-source mutation is performed.
"""
from pathlib import Path
import json
import sys
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, mynn_coldstart_init_columns

packet=np.load(sys.argv[1]);a,s,truth=packet['arrays'],packet['scalars'],packet['truth'][:,:,0]
values={name:np.empty_like(truth) for name in ('old','fixed','old_without_snow')}
for dx in sorted(set(s[:,-1])):
    ix=np.flatnonzero(s[:,-1]==dx);col=a[ix];v=lambda n:jnp.asarray(col[:,:,n]);z=jnp.zeros_like(v(0))
    state=MynnPBLColumnState(v(0),v(1),v(2),v(3),v(4),v(5),v(6),v(7),v(8),z,z,z,
       qc=v(9),qi=v(10),qs=v(11),qsq=z,qc_bl=z,qi_bl=z,cldfra_bl=z,exner=v(13))
    def calls(state,ust,land):
        old=mynn_coldstart_init_columns(state,ust,float(dx),land)[0]
        fixed=mynn_coldstart_init_columns(state,ust,float(dx),land,return_moments=True)[0]
        cleared=mynn_coldstart_init_columns(state.replace(qs=jnp.zeros_like(state.qs)),ust,float(dx),land)[0]
        return old,fixed,cleared
    with jax.enable_x64(False):
        result=jax.jit(calls)(state,jnp.asarray(s[ix,0]),jnp.asarray(s[ix,4]));jax.block_until_ready(result)
    for name,value in zip(values,result):values[name][ix]=np.asarray(value)
snow=(a[:,:,11]>1e-6).any(axis=1);assert snow.any() and a[:,:,11].max()>1e-4
records={}
for mask,label in ((np.ones(len(a),bool),'all'),(snow,'snow')):
    t=truth[mask].astype(float);norm=float(np.sqrt(np.mean(t*t)))
    records[label]={}
    for name,v in values.items():
        x=v[mask].astype(float);assert np.isfinite(x).all() and np.isfinite(t).all()
        records[label][name]={'relative_rms_to_wrf':float(np.sqrt(np.mean((x-t)**2))/norm),
            'max_abs_to_wrf':float(abs(x-t).max()),'median_relative_to_wrf':float(np.median(abs(x-t)/np.maximum(abs(t),1e-8)))}
    delta=values['fixed'][mask].astype(float)-values['old_without_snow'][mask]
    records[label]['thlv_intervention']={'max_abs_fixed_vs_old_no_snow':float(abs(delta).max()),
        'relative_rms':float(np.sqrt(np.mean(delta*delta))/norm),
        'byte_equal':values['fixed'][mask].tobytes()==values['old_without_snow'][mask].tobytes()}
report={'backend':jax.default_backend(),'snow_columns':int(snow.sum()),'columns':len(a),'max_input_snow':float(a[:,:,11].max()),
    'records':records,'scope':'REAL4 original operator and THLV causal intervention; no GPU/forecast'}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert records['snow']['fixed']['median_relative_to_wrf']<=1e-4
assert records['snow']['thlv_intervention']['relative_rms']<=1e-3  # existing Tier2 conditioning class
