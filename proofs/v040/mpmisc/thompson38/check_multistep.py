"""Compare evolving production code objects with three original own-steps."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from check_columns import parse
from literal_reference import executor, LiteralState
from gpuwrf.physics.thompson_hail.column import ThompsonHailColumnState, step_thompson_hail_column_with_precip


FIELDS = dict(qv="QV", qc="QC", qr="QR", qi="QI", qs="QS", qg="QG", Ni="NI", Nr="NR", Nc="NC", Ng="NG", qvolg="QB", nwfa="NWFA", nifa="NIFA", T="T")


def main(reference, output, compiled=False):
    if (reference/"step1.txt").exists():
        refs = [parse(reference/f"step{step}.txt") for step in (1,2,3)]
    else:
        refs=[]
        with np.load(reference/"original_multistep.npz",allow_pickle=False) as packed:
            for step in (1,2,3):
                values={name.split("__",1)[1]:packed[name] for name in packed.files if name.startswith(f"step{step}__")}
                refs.append(dict(columns={name:value.tolist() for name,value in values.items() if value.ndim},scalars={name:float(value) for name,value in values.items() if not value.ndim}))
    first = refs[0]
    f = np.float32
    values = {field:np.asarray(first["columns"][code+"_IN"],np.float32)[None,:] for field,code in FIELDS.items()}
    p = np.asarray(first["columns"]["P"],np.float32)[None,:]
    pii = np.asarray(first["columns"]["PII"],np.float32)[None,:]
    rho = f(.622)*p/(f(287.04)*values["T"]*(values["qv"]+f(.622)))
    values.update(p=p, rho=rho, dz=np.asarray(first["columns"]["DELZ"],np.float32)[None,:], w=np.asarray(first["columns"]["W_IN"],np.float32)[None,:])
    state_type = ThompsonHailColumnState if compiled else LiteralState
    state = state_type(**values)
    accum = {name:f(first["scalars"][name+"_IN"]) for name in ("RAINNC","SNOWNC","GRAUPELNC")}
    result = dict(scope="Original unmodified WRF evolving three-own-step caller, positive held aerosol emission and nonzero REAL accumulation",
                  compiled=compiled, steps=[], byte_equal=True)
    started=time.monotonic()
    with nullcontext(step_thompson_hail_column_with_precip) if compiled else executor() as body:
        for step,ref in enumerate(refs,1):
            state, precip = body(state,18.)
            values={name:np.asarray(getattr(state,name),np.float32) for name in state.__slots__}
            # Public mp_gt_driver :1358 and :1320-1321, before next own-step.
            theta=values["T"]/pii
            values["T"]=theta*pii
            values["nwfa"]=values["nwfa"].copy()
            values["nifa"]=values["nifa"].copy()
            values["nwfa"][...,0]+=f(ref["scalars"]["NWFA2D"])*f(18.)
            values["nifa"][...,0]+=f(ref["scalars"]["NIFA2D"])*f(18.)
            state=state_type(**values)
            r,s,g,i=(f(np.asarray(precip[name]).item()) for name in ("rain","snow","graupel","ice"))
            accum["RAINNC"]=(((accum["RAINNC"]+r)+s)+g)+i
            accum["SNOWNC"]=(accum["SNOWNC"]+s)+i
            accum["GRAUPELNC"]+=g
            surface=dict(accum,RAINNCV=((r+s)+g)+i,SNOWNCV=s+i,GRAUPELNCV=g,SR=((s+g)+i)/((((r+s)+g)+i)+f(1.e-12)))
            checks={}
            for field,code in FIELDS.items():
                a=values[field][0];b=np.asarray(ref["columns"][code+"_OUT"],np.float32)
                checks[field]=dict(byte_equal=bool(np.array_equal(a,b)),matching=int(np.sum(a==b)),total=40,max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
            surfaces={name:dict(byte_equal=bool(value==f(ref["scalars"][name])),actual=float(value),reference=ref["scalars"][name]) for name,value in surface.items()}
            theta_check=dict(actual=theta[0].tolist(),reference_available="TH_OUT" in ref["columns"])
            if theta_check["reference_available"]:
                expected=np.asarray(ref["columns"]["TH_OUT"],np.float32)
                theta_check.update(byte_equal=bool(np.array_equal(theta[0],expected)),matching=int(np.sum(theta[0]==expected)),total=40,max_abs=float(np.max(np.abs(theta[0].astype(float)-expected.astype(float)))))
            exact=all(x["byte_equal"] for x in checks.values()) and all(x["byte_equal"] for x in surfaces.values())
            if theta_check["reference_available"]:exact &= theta_check["byte_equal"]
            result["steps"].append(dict(step=step,fields=checks,surfaces=surfaces,public_theta=theta_check,byte_equal=exact))
            result["byte_equal"] &= exact
    result["wall_s"]=time.monotonic()-started
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("reference",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--compiled",action="store_true")
    args=parser.parse_args();main(args.reference,args.output,args.compiled)
