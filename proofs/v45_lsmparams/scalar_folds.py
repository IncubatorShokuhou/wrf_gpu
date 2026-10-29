"""Compare parameter-only Noah-MP folds before any state-dependent arithmetic."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
sys.path[:0] = [str(a.source/'src'), str(a.source/'proofs/w44_lsmparams')]
import jax
import jax.numpy as jnp
import numpy as np
from columns import build
from gpuwrf.physics.noahmp.energy import GRAV, RW

land,_,_,ep,_,_,_ = build()
values = tuple(jnp.asarray(v,jnp.float32) for v in (ep.smcmax[0],ep.smcwlt[0],ep.bexp[0],ep.psisat[0]))

def expressions(smcmax,smcwlt,bexp,psisat):
    diffusivity=2.2e-5*smcmax*smcmax*(1.-smcwlt/smcmax)**(2.+3./bexp)
    return {'d_rsurf':diffusivity,'inv_d_rsurf':1./diffusivity,
            'psisat_grav':-psisat*GRAV,
            'bexp_inv':-1./bexp,
            'wdf_exponent':bexp+2.,'wcnd_exponent':2.*bexp+3.}

off = jax.jit(lambda:expressions(*values))()
on = jax.jit(expressions)(*values)
smcmax,smcwlt,bexp,psisat = (np.asarray(v) for v in values)
host = {'d_rsurf':np.float32(2.2e-5)*smcmax*smcmax*(np.float32(1.)-smcwlt/smcmax)**(np.float32(2.)+np.float32(3.)/bexp),
        'psisat_grav':-psisat*np.float32(GRAV),'bexp_inv':-np.float32(1.)/bexp,
        'wdf_exponent':bexp+np.float32(2.),'wcnd_exponent':np.float32(2.)*bexp+np.float32(3.)}
host['inv_d_rsurf']=np.float32(1.)/host['d_rsurf']
record = dict(platform='cpu',gpu_executed=False,scope=__doc__,rows={})
for key in off:
    x,y,z = (np.asarray(d[key]) for d in (off,on,host))
    record['rows'][key] = dict(off_hex=x.tobytes().hex(),on_hex=y.tobytes().hex(),host_hex=z.tobytes().hex(),
                               off=x.tolist(),on=y.tolist(),host=z.tolist(),
                               dynamic_byte=x.tobytes()==y.tobytes(),host_byte=x.tobytes()==z.tobytes())

def state_expressions(moisture,temperature,cap,wilt,exponent,psi):
    ratio = jnp.maximum(.01,moisture)/cap
    power = ratio**(-exponent)
    potential = -psi*power
    return dict(ratio=ratio,power=power,potential=potential,potential_grav=potential*GRAV,
                relative_humidity=jnp.exp(potential*GRAV/(RW*temperature)),
                resistance_length=jnp.exp((1.-jnp.minimum(1.,moisture/cap))**ep.rsurf_exp)-1.)

state_values=(land.sh2o[0],land.tg)
reference=jax.jit(lambda *s:state_expressions(*s,*values))(*state_values)
record['state_rows']={}
for index,name in ((0,'smcmax'),(2,'bexp'),(3,'psisat')):
    def fn(moisture,temperature,value):
        args=list(values);args[index]=value
        return state_expressions(moisture,temperature,*args)
    candidate=jax.jit(fn)(*state_values,values[index])
    record['state_rows'][name]={}
    for key in reference:
        x,y=np.asarray(reference[key]),np.asarray(candidate[key])
        record['state_rows'][name][key]=dict(byte=x.tobytes()==y.tobytes(),
            off_hex=x.tobytes().hex(),on_hex=y.tobytes().hex(),max_abs=float(np.abs(x-y).max()))
a.out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
