"""Bisect expressions on saved actual PROD pre-canopy operands (CPU diagnosis).

New versus scalar_folds: all 8400 real parent cells, thermoprop_full as well as
surface resistance/humidity, current coefficient routing and optimized HLO.
Separately compiled expressions are diagnostic, never forecast/WRF fidelity.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(a.source/'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.noahmp import energy as e
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.real_constfold import real_constant
assert all(d.platform=='cpu' for d in jax.devices())
a.out.mkdir(parents=True,exist_ok=False)
z=np.load(a.inputs/'off.npz')
params={k.split("['vege']['p'].")[1]:jnp.asarray(z[k]) for k in z.files if k.startswith("['vege']['p'].")}
land=NoahMPLandState(**{name:jnp.asarray(z[f"['vege']['land_state'][<flat index {i}>]"])
    for i,name in enumerate(NoahMPLandState.__slots__[:-1])})
cap,exponent,psi=(params[k] for k in ('smcmax','bexp','psisat'))
fsno=jnp.asarray(z["['vege']['fsno']"])
# This real case has no snow; the saved ZSNSO soil interfaces are ZSOIL.
assert not np.asarray(land.snowh).any()
zsoil=land.zsnso[3:]
wilt=params['smcwlt'][0]
inv_d=jnp.asarray(np.float32(1.)/(np.float32(2.2e-5)*np.asarray(cap[0])**2*
    (np.float32(1.)-np.asarray(wilt)/np.asarray(cap[0]))**
    (np.float32(2.)+np.float32(3.)/np.asarray(exponent[0]))))
psi_grav=jnp.asarray(-np.asarray(psi[0])*np.float32(e.GRAV))
rec={'scope':__doc__,'gpu_executed':False,'input_receipt_sha256':hashlib.sha256((a.inputs/'receipt.json').read_bytes()).hexdigest(),'rows':{}}
def compare(x,y):
 x,y=np.asarray(x),np.asarray(y)
 same=(x==y)|(np.isnan(x)&np.isnan(y));err=np.where(same,0,np.abs(x.astype(np.float64)-y.astype(np.float64)))
 index=np.unravel_index(np.nanargmax(err),err.shape)
 return dict(byte=x.tobytes()==y.tobytes(),cells=int(np.count_nonzero(~same)),max_abs=float(np.nanmax(err)),
             worst_index=[int(i) for i in index],off=float(x[index]),on=float(y[index]))
def run(name,fn,*args):
 lowered=jax.jit(fn).lower(*args);exe=lowered.compile()
 (a.out/(name+'.hlo')).write_text(exe.as_text())
 return jax.block_until_ready(exe(*args))
def thermo(state,c):
 return e.thermoprop_full(state,SimpleNamespace(**{**params,'smcmax':c}))
off=run('thermo_off',lambda state:thermo(state,cap),land)
on=run('thermo_on',thermo,land,cap)
rec['rows']['thermo']={key:compare(x,y) for key,x,y in zip(('df_full','hcpct_full','df_top','stc_top','dz_top'),off,on)}
rec['rows']['thermo_production']={'off':compare(z["['vege']['df_top']"],off[2]),
    'on':compare(np.load(a.inputs/'on0.npz')["['vege']['df_top']"],on[2])}
def components(smc,sh2o,c):
 thks=(7.7**params['quartz'])*(2.0**(1.0-params['quartz']))
 xunfroz=jnp.where(smc>0.,sh2o/jnp.maximum(smc,e.MPE),1.)
 xu=xunfroz*c
 satratio=smc/c
 gammd=(1.-c)*2700.
 thkdry=(.135*gammd+64.7)/(2700.-.947*gammd)
 thksat=thks**(1.-c)*e.TKICE**(c-xu)*.57**xu
 ake_unfroz=jnp.where(satratio>.1,jnp.log10(jnp.maximum(satratio,e.MPE))+1.,0.)
 ake=jnp.where((sh2o+.0005)<smc,satratio,ake_unfroz)
 return dict(thks=thks,cap_inverse=1./c,one_cap=1.-c,thks_pow=thks**(1.-c),gammd=gammd,
     thkdry=thkdry,xunfroz=xunfroz,xu=xu,satratio=satratio,thksat=thksat,ake=ake,
     soil_df=ake*(thksat-thkdry)+thkdry)
off_c=run('components_off',lambda smc,sh:components(smc,sh,cap),land.smois,land.sh2o)
on_c=run('components_on',components,land.smois,land.sh2o,cap)
rec['rows']['thermo_components']={key:compare(off_c[key],on_c[key]) for key in off_c}
def evaporation(sh,tg,fsno,c,b,ps,pg=None,dr=None,ci=None):
 c,b,ps=c[0],b[0],ps[0]
 ratio=jnp.maximum(.01,sh)/c if ci is None else jnp.maximum(.01,sh)*ci[0]
 power=ratio**(-b)
 potential=-ps*power
 product=potential*e.GRAV
 humidity=fsno+(1.-fsno)*jnp.exp(product/(e.RW*tg)) if pg is None else fsno+(1.-fsno)*jnp.exp(pg*power/(e.RW*tg))
 wet=sh/c if ci is None else sh*ci[0]
 length=(-zsoil[0])*(jnp.exp((1.-jnp.minimum(1.,wet))**params['rsurf_exp'])-1.)/real_constant(e._E_M1_REAL,2.71828-1.,jnp.result_type(c))
 diffusivity=2.2e-5*c*c*(1.-wilt/c)**(2.+3./b)
 resistance=length/diffusivity if dr is None else length*dr
 resistance=jnp.where((sh<.01)&(land.snowh==0.),1.e6,resistance)
 return dict(rhsur=humidity,rsurf=resistance)
state_args=(land.sh2o[0],land.tg,fsno)
ev_off=run('evap_off',lambda sh,tg,fs:evaporation(sh,tg,fs,cap,exponent,psi),*state_args)
ev_cap=run('evap_cap',lambda sh,tg,fs,c,b,dr:evaporation(sh,tg,fs,c,b,psi,dr=dr),*state_args,cap,exponent,inv_d)
ev_psi=run('evap_psi',lambda sh,tg,fs,ps,pg:evaporation(sh,tg,fs,cap,exponent,ps,pg=pg),*state_args,psi,psi_grav)
rec['rows']['evap_cap']={key:compare(ev_off[key],ev_cap[key]) for key in ev_off}
rec['rows']['evap_psi']={key:compare(ev_off[key],ev_psi[key]) for key in ev_off}
for label,res in (('off',ev_off),('on0',ev_cap),('on1',ev_psi)):
 raw=np.load(a.inputs/(label+'.npz'))
 rec['rows']['evap_production_'+label]={key:compare(raw[f"['vege']['{key}']"],v) for key,v in res.items()}
# Causal interventions target the two measured first-different operations.
cap_host=np.asarray(cap)
cap_inv=jnp.asarray(np.float32(1.)/cap_host)
gamma=(np.float32(1.)-cap_host)*np.float32(2700.)
dry_host=(np.float32(.135)*gamma+np.float32(64.7))/(np.float32(2700.)-np.float32(.947)*gamma)
dry=jnp.asarray(dry_host)
rec['rows']['host_coefficients']={'inverse':compare(off_c['cap_inverse'],cap_inv),'dry':compare(off_c['thkdry'],dry)}
original_soil=e._tdfcnd_soil
source=inspect.getsource(original_soil)
assert source.count('smc / p.smcmax')==1 and source.count('thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)')==1
source=source.replace('def _tdfcnd_soil(smc, sh2o, p: EnergyParams):','def _bound_soil(smc, sh2o, p, cap_inv, dry):')
source=source.replace('smc / p.smcmax','smc * cap_inv')
source=source.replace('thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)','thkdry = dry')
namespace=dict(vars(e));exec(compile(source,'<measured_soil_intervention>','exec'),namespace)
bound_soil=namespace['_bound_soil']
def bound_thermo(state,c,ci,td):
 e._tdfcnd_soil=lambda smc,sh2o,p:bound_soil(smc,sh2o,p,ci,td)
 try:return thermo(state,c)
 finally:e._tdfcnd_soil=original_soil
fixed=run('thermo_bound',bound_thermo,land,cap,cap_inv,dry)
rec['rows']['thermo_bound']={key:compare(x,y) for key,x,y in zip(('df_full','hcpct_full','df_top','stc_top','dz_top'),off,fixed)}
ev_psi_original=run('evap_psi_original',lambda sh,tg,fs,ps:evaporation(sh,tg,fs,cap,exponent,ps),*state_args,psi)
rec['rows']['evap_psi_original']={key:compare(ev_off[key],ev_psi_original[key]) for key in ev_off}
ev_all=run('evap_all_bound',lambda sh,tg,fs,c,b,ps,dr,ci:evaporation(sh,tg,fs,c,b,ps,dr=dr,ci=ci),*state_args,cap,exponent,psi,inv_d,cap_inv)
rec['rows']['evap_all_bound']={key:compare(ev_off[key],ev_all[key]) for key in ev_off}
(a.out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2),flush=True)
