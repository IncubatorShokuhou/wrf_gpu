"""Diagnostic hooks around unchanged shared Thompson code objects."""
from pathlib import Path
import argparse
import json
import sys
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_hail import column

sys.path.insert(0, str(Path(__file__).parent))
from check_columns import parse


def main(reference, out, case):
    ref = parse(reference / f"case{case}.txt")
    arrays = {name: jnp.asarray(values, jnp.float32)[None,:] for name,values in ref["columns"].items()}
    fields = {"qv":"QV", "qc":"QC", "qr":"QR", "qi":"QI", "qs":"QS", "qg":"QG", "Ni":"NI", "Nr":"NR", "Nc":"NC", "Ng":"NG", "qvolg":"QB", "nwfa":"NWFA", "nifa":"NIFA", "T":"T"}
    values = {name:arrays[original+"_IN"] for name,original in fields.items()}
    p,temp,qv=arrays["P"],values["T"],values["qv"]
    rho=np.float32(.622)*p/(np.float32(287.04)*temp*(qv+np.float32(.622)))
    state=column.ThompsonHailColumnState(**values,p=p,rho=rho,dz=arrays["DELZ"],w=jnp.zeros_like(qv))
    captured={}
    names=("qv","qc","qr","qi","qs","qg","Nr","Nc","Ng","qvolg","nwfa","nifa","T","rho")
    def save(prefix,state):
        for name in names:captured[prefix+"_"+name]=getattr(state,name)
    def stage(fn,label):
        def hook(state,*args,**kwargs):
            save(label+"_pre",state)
            result=fn(state,*args,**kwargs)
            save(label+"_post",result[0] if isinstance(result,tuple) else result)
            if label=="evap":
                captured["evap_skip"]=jnp.asarray(kwargs["skip_evaporation"])
                captured["evap_graupel_melt"]=jnp.asarray(kwargs["graupel_melt"])
            return result
        return hook
    shared=SimpleNamespace(**vars(column.tc));aerosol=SimpleNamespace(**vars(column.aero))
    shared._wrf_mixed_phase_sources=stage(column.tc._wrf_mixed_phase_sources,"sources")
    aerosol._rain_evaporation_aero=stage(column.aero._rain_evaporation_aero,"evap")
    env=dict(column._body.__globals__)
    env.update(tc=shared,aero=aerosol,_sedimentation=stage(column._sedimentation,"sed"),
               _saturation_adjustment=stage(column._saturation_adjustment,"sat"))
    body=FunctionType(column._body.__code__,env,column._body.__name__,column._body.__defaults__,column._body.__closure__)
    def traced(state):
        result=body(state,90.)
        return result,dict(captured)
    result,stages=jax.jit(traced)(state)
    stages={name:np.asarray(value).tolist() for name,value in stages.items()}
    out.write_text(json.dumps({"case":case,"platform":jax.devices()[0].platform,"stages":stages},indent=2)+"\n")
    for phase in ("sources","sat","evap","sed"):
        for name in ("qv","qr","T"):
            a=np.asarray(stages[phase+"_pre_"+name]);b=np.asarray(stages[phase+"_post_"+name])
            print(phase,name,"maxchange",float(np.max(np.abs(b-a))))
    print("evap skip",stages["evap_skip"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("reference",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--case",type=int,default=5)
    args=parser.parse_args()
    main(args.reference,args.out,args.case)
