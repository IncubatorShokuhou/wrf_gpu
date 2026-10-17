"""Distinct NSAS96 deep core vs WRITE-only pristine driver savepoints."""
from __future__ import annotations
import argparse
import functools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from evidence_gate import require_finite
from probe_ksas14 import parse,compare_fields


def main():
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("stage",type=Path)
    p.add_argument("output",type=Path);p.add_argument("--mode",choices=("r4","r8"))
    p.add_argument("--phases",type=Path);p.add_argument("--baseline",type=Path);args=p.parse_args()
    if bool(args.phases)!=bool(args.baseline):p.error("--phases and --baseline are required together")
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.cumulus_nsas96 import nsas96_deep_column
    jax.config.update("jax_enable_x64",True)
    assert jax.devices()[0].platform=="cpu"
    args.output.mkdir(parents=True,exist_ok=True)
    report={"source":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "scope":"deep only; full CU_NSAS shallow not yet ported","backend":"cpu","arms":{}}
    for mode in ((args.mode,) if args.mode else ("r8","r4")):
        dtype=np.float32 if mode=="r4" else np.float64
        inp=[parse(args.reference/f"case{i}_{mode}.txt") for i in range(1,6)]
        inp={k:np.stack([d[k] for d in inp]).astype(dtype) for k in inp[0]}
        with np.load(args.stage/f"deep_{mode}.npz") as saved:
            stage={k:v.astype(dtype) for k,v in saved.items()}
        require_finite(**inp)
        require_finite(**stage)
        def pad(a):return np.concatenate((np.zeros((len(a),1),dtype),a),axis=1)
        keys=("T","QV","QI","QC","U","V")
        columns=[pad(inp[k]) for k in keys]+[pad(stage[k]) for k in ("DEL","PRSL","Z","DOT")]
        columns+=[inp["XLAND"]]
        limits=[stage["LIMITS"][:,i].astype(np.int32) for i in range(3)]
        km=inp["T"].shape[1]
        delt=dtype(inp["DT"][0]*inp["STEPCU"][0])
        col=functools.partial(nsas96_deep_column,km=km,mode=mode,ncloud=2,pgcon=.55,dx_factor_nsas=0,debug=bool(args.phases))
        fn=jax.jit(jax.vmap(lambda *a:col(*a[:11],jnp.asarray(delt),jnp.asarray(inp["DX"][0]),*a[11:])))
        start=time.monotonic()
        raw=fn(*columns,*limits)
        if args.phases:
            result,phase_values=raw
            phase_values={k:np.asarray(v) for k,v in phase_values.items()}
        else:result=raw
        result=tuple(np.asarray(x) for x in result)
        if args.phases:
            normal_col=functools.partial(nsas96_deep_column,km=km,mode=mode,ncloud=2,pgcon=.55,dx_factor_nsas=0,debug=False)
            normal_fn=jax.jit(jax.vmap(lambda *a:normal_col(*a[:11],jnp.asarray(delt),jnp.asarray(inp["DX"][0]),*a[11:])))
            normal_result=tuple(np.asarray(x) for x in normal_fn(*columns,*limits))
            observer_raw_exact=all(np.array_equal(a,b) for a,b in zip(result,normal_result))
        elapsed=time.monotonic()-start
        actual_states=dict(zip(keys,result[:6]))
        expected=dict(inp)
        actual={}
        rdelt=dtype(1.)/delt
        for input_key,output_key in zip(keys,("RTHCUTEN","RQVCUTEN","RQICUTEN","RQCCUTEN","RUCUTEN","RVCUTEN")):
            a=(actual_states[input_key][:,1:]-inp[input_key])
            b=(stage[input_key]-inp[input_key])
            if input_key=="T":a,b=a/inp["PI"],b/inp["PI"]
            actual[output_key]=a*rdelt
            expected[output_key]=b*rdelt
        for idx,key in ((6,"RAINCV"),(7,"HBOT"),(8,"HTOP")):
            a=result[idx];b=stage[{"RAINCV":"RAIN","HBOT":"KBOT","HTOP":"KTOP"}[key]][:,0]
            if key=="RAINCV":a,b=a*dtype(1000.)/inp["STEPCU"][0],b*dtype(1000.)/inp["STEPCU"][0]
            actual[key],expected[key]=a,b
        actual["PRATEC"]=result[6]*dtype(1000.)/delt
        expected["PRATEC"]=stage["RAIN"][:,0]*dtype(1000.)/delt
        fields=compare_fields(actual,expected,mode)
        icps_exact=bool(np.array_equal(result[9],stage["ICPS"][:,0]))
        report["arms"][mode]={"fields":fields,"elapsed_s":elapsed,"icps_exact":icps_exact,
            "active_original":int(np.count_nonzero(stage["ICPS"])),"active_port":int(np.count_nonzero(result[9])),
            "pass":icps_exact and all(v["pass"] for v in fields.values())}
        if args.phases:
            with np.load(args.baseline/f"actual_{mode}.npz") as baseline:
                observer_exact=all(np.array_equal(actual[k],baseline[k]) for k in actual)
            source_phases=json.loads(args.phases.read_text())
            phase_comparison={}
            for i in range(5):
                original=source_phases[f"case{i+1}_{mode}"]
                observed={}
                for key,values in original.items():
                    b=np.asarray(values,dtype=dtype)
                    a=phase_values[key]
                    a=a[i:i+1] if a.ndim==1 else a[i,1:1+len(b)]
                    delta=np.abs(a.astype(np.float64)-b)
                    ulp=np.abs(np.spacing(np.maximum(np.abs(a),np.abs(b))))
                    observed[key]={"same":bool(np.array_equal(a,b)),"finite":bool(np.isfinite(a).all() and np.isfinite(b).all()),
                        "max_abs":float(delta.max()),"max_ulp":float(np.max(delta/ulp)),
                        "first_differences":np.flatnonzero(a!=b)[:8].tolist()}
                phase_comparison[f"case{i+1}"]=observed
            report["arms"][mode]["phase_comparison"]=phase_comparison
            report["arms"][mode]["observer_final_byte_exact"]=observer_exact
            report["arms"][mode]["observer_raw_byte_exact"]=observer_raw_exact
            report["arms"][mode]["valid_for_phase_diagnosis"]=observer_exact and observer_raw_exact and icps_exact
            np.savez_compressed(args.output/f"debug_phases_{mode}.npz",**phase_values)
        np.savez_compressed(args.output/f"actual_{mode}.npz",**actual)
        (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
