"""Compare two real nested parent steps with original CPU-WRF feedback=1.

CPU uses Pallas interpretation, RN32 PTX emulation and the existing Thomas
harness shim. This is CPU coupled evidence, never a GPU qualification claim.
The per-frame observer is outside each timestep executable.
"""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

assert os.environ["JAX_PLATFORMS"] == "cpu"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"proofs/o1_smag3d"))
import cpu_safe
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as pt
cpu_safe.install()
assert jax.devices()[0].platform=="cpu" and jax.config.x64_enabled
real_pallas_call = pl.pallas_call


def interpret_call(*args, **kwargs):
    kwargs["interpret"] = True
    kwargs.pop("compiler_params",None)
    return real_pallas_call(*args,**kwargs)


def cpu_inline_asm(asm, *, args, result_shape_dtypes, **kwargs):
    op = asm.split()[0].split(".")[0]
    with jax.enable_x64(True):
        a,b = (jnp.asarray(x,jnp.float64) for x in args)
        exact = {"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"div":lambda:a/b}[op]()
        dtype = result_shape_dtypes[0].dtype
        if jnp.dtype(dtype)==jnp.float32:
            exact = jax.lax.reduce_precision(exact,exponent_bits=8,mantissa_bits=23)
        return [exact.astype(dtype)]


pl.pallas_call=interpret_call
pt.elementwise_inline_asm=cpu_inline_asm
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.nesting_options import resolve_nesting_options
from gpuwrf.io.wrfout_writer import _theta_dry_for_output
from gpuwrf.runtime.domain_tree import DomainTree,run_operational_domain_tree
state_contract._gpu_device=lambda:jax.devices("cpu")[0]
case,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
steps=int(sys.argv[3]) if len(sys.argv)>3 else 2
manifest_file=ROOT/"proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
manifest=json.loads(manifest_file.read_text())["fields"]
report={"reference_plan":json.loads((case/"PLAN.json").read_text()),"status":"LOADING","frames":[],
        "backend":"cpu", "cache":"NONE", "steps":steps,
        "scope":"Frozen D6 and predeclared scalar gates; unbounded auxiliary fields are reported only",
        "field_disposition_receipt":"proofs/u2_twoway/QUALIFICATION.json",
        "manifest_sha256":hashlib.sha256(manifest_file.read_bytes()).hexdigest()}
start=time.monotonic()


def save():
    report["wall_s"]=time.monotonic()-start
    out.write_text(json.dumps(report,indent=2)+"\n")


def observe(name,step,carry):
    jax.block_until_ready(carry)
    s=carry.state
    valid=run_start+timedelta(seconds=step*dts[name])
    ref=case/f"wrfout_{name}_{valid:%Y-%m-%d_%H:%M:%S}"
    arrays={key:np.asarray(getattr(s,attr)) for key,attr in {
        "U":"u","V":"v","W":"w","P":"p_perturbation","PH":"ph_perturbation","MU":"mu_perturbation",
        "QVAPOR":"qv","QCLOUD":"qc","QRAIN":"qr","QICE":"qi","QSNOW":"qs","QGRAUP":"qg",
        "QNICE":"Ni","QNRAIN":"Nr","QKE":"qke","RAINC":"rainc_acc","RAINNC":"rain_acc",
        "SNOWNC":"snow_acc","GRAUPELNC":"graupel_acc"}.items()}
    arrays["T"]=_theta_dry_for_output(s,np.asarray(s.theta),np.asarray(s.qv))-300.
    arrays["HGT"]=np.asarray(bundles[name].grid.terrain_height)
    arrays['TSK']=np.asarray(s.t_skin)
    if step==0:
        overrides=bundles[name].writer_static_overrides or {}
        with Dataset(input_view/f'wrfinput_{name}') as initial:
            for key in ('T2','Q2','U10','V10','PSFC'):
                arrays[key]=np.asarray(overrides[key]) if key in overrides else np.asarray(initial[key][0])
    arrays.update({key:np.asarray(getattr(bundles[name].metrics,attr)) for key,attr in {
        "MAPFAC_MX":"msftx","MAPFAC_MY":"msfty","MAPFAC_UX":"msfux","MAPFAC_UY":"msfuy",
        "MAPFAC_VX":"msfvx","MAPFAC_VY":"msfvy","F":"f","E":"e",
        "SINALPHA":"sina","COSALPHA":"cosa"}.items()})
    if carry.noahmp_land is not None and step==0:
        from gpuwrf.io.land_history import load_land_history_inputs
        from gpuwrf.config.paths import wrf_run_dir
        arrays.update(load_land_history_inputs(case/f"wrfinput_{name}",table_dir=wrf_run_dir(),
            parameters=bundles[name].namelist.noahmp_static.parameters,
            field_overrides=bundles[name].writer_static_overrides))
        arrays={key:val for key,val in arrays.items() if not key.startswith("_") and not key.endswith("_INIT")}
    if step and carry.history_diagnostics is not None:
        arrays.update({key:np.asarray(getattr(carry.history_diagnostics,attr))
            for key,attr in {"T2":"t2","U10":"u10","V10":"v10","PSFC":"psfc"}.items()})
    if carry.noahmp_land is not None:
        from gpuwrf.io.land_history import land_history_diagnostics
        values,_=land_history_diagnostics(carry.noahmp_land,initial_lands[name],land_inputs[name],
            own_step=step,history=carry.land_history)
        arrays.update(values)
    if getattr(carry,'feedback_history',None) is not None:
        arrays.update({key:np.asarray(value) for key,value in carry.feedback_history.items()})
    if carry.energy_accumulators is not None:
        arrays.update({key:np.asarray(value) for key,value in carry.energy_accumulators.items()})
    if carry.base_state is not None:
        arrays.update({key:np.asarray(getattr(carry.base_state,attr)) for key,attr in {"PB":"pb","PHB":"phb","MUB":"mub"}.items()})
        if os.environ.get('GPUWRF_DYN_CARRY_FP32')=='1':
            assert all(np.asarray(x).dtype==np.float32 for x in jax.tree.leaves(carry.base_state)), 'native feedback base must remain WRF REAL'
    if carry.h_diabatic is not None:
        arrays["H_DIABATIC"]=np.asarray(carry.h_diabatic)
    frame={"domain":name,"step":step,"reference":str(ref),"fields":{},"nonfinite":sum(
        int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(carry)
        if np.issubdtype(np.asarray(x).dtype,np.floating))}
    with Dataset(ref) as ds:
        for key,actual in arrays.items():
            if key not in ds.variables:
                frame["fields"][key]={"gate":"MISSING_REFERENCE"}
                continue
            expected=np.asarray(ds[key][0])
            assert actual.shape==expected.shape,(key,actual.shape,expected.shape)
            finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            delta=actual.astype(np.float64)-expected.astype(np.float64)
            stats={"finite":finite,"max_abs":float(np.max(np.abs(delta))),"rmse":float(np.sqrt(np.mean(delta*delta)))}
            bounds=manifest.get(key,{})
            tests=[stats[k]<=bounds[k] for k in ("rmse","max_abs") if k in bounds]
            extra=None
            if not tests:
                if key in ('QCLOUD','QRAIN','QICE','QSNOW','QGRAUP'):
                    extra=report['reference_plan']['gate']['extra_mass']
                elif key in ('QNICE','QNRAIN'):
                    extra=report['reference_plan']['gate']['extra_numbers']
            if extra is not None:
                violations=np.abs(delta)>extra['atol']+extra['rtol']*np.abs(expected.astype(np.float64))
                stats['pointwise_violations']=int(np.count_nonzero(violations))
                tests=[not violations.any()]
            stats["gate"]="PASS" if finite and tests and all(tests) else ("FAIL" if not finite or tests else "REPORT_ONLY")
            frame["fields"][key]=stats
    if carry.census is not None:
        frame["guards"]=np.asarray(carry.census.guards).tolist()
    report["frames"].append(frame)
    report["status"]="RUNNING"
    save()
    print(name,step,"nonfinite",frame["nonfinite"],"FAIL",[key for key,val in frame["fields"].items() if val["gate"]=="FAIL"],flush=True)


observe.wants_carry=True
try:
    # The model sees native inputs only. Gen2 static readers may prefer a
    # history file if one is discoverable, so never expose oracle outputs.
    input_view=out.parent/'cpu-input-view'
    input_view.mkdir(exist_ok=True)
    for path in input_view.iterdir():
        if path.is_symlink():path.unlink()
    for path in case.iterdir():
        if path.name.startswith(('wrfinput','wrfbdy')) or path.name=='namelist.input':
            (input_view/path.name).symlink_to(path)
    cfg=pipeline.NestedPipelineConfig(input_view,out.parent/"history",out.parent/"proof",hours=1,max_dom=2)
    report['model_input_view']=str(input_view)
    report['oracle_history_visible_to_model']=False
    report['cpu_affinity']=sorted(os.sched_getaffinity(0))
    report['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    seed=out.parent/"cpu_initial_seed.pkl"
    seal=seed.with_suffix(".json")
    signature={"src_tree":subprocess.check_output(["git","rev-parse","HEAD:src"],cwd=ROOT,text=True).strip(),
               "case_plan_sha256":hashlib.sha256((case/"PLAN.json").read_bytes()).hexdigest(),
               "env":{k:v for k,v in os.environ.items() if k.startswith(("GPUWRF_","JAX_","XLA_"))}}
    loaded=False
    if seed.is_file() and seal.is_file():
        saved=json.loads(seal.read_text())
        if saved["signature"]==signature and hashlib.sha256(seed.read_bytes()).hexdigest()==saved["sha256"]:
            with seed.open("rb") as file:
                hierarchy,bundles,meta,run_start,dts,carries=pickle.load(file)
            loaded=True
    if not loaded:
        hierarchy,bundles,meta,run_start,dts,carries=pipeline._load_domains(cfg,("d01","d02"))
        try:
            with seed.open("wb") as file:
                pickle.dump((hierarchy,bundles,meta,run_start,dts,carries),file,protocol=5)
            seal.write_text(json.dumps({"signature":signature,"sha256":hashlib.sha256(seed.read_bytes()).hexdigest()},indent=2)+"\n")
        except Exception as exc:
            report["seed_cache_error"]=f"{type(exc).__name__}: {exc}"
    report["initial_seed_reused"]=loaded
    from gpuwrf.io.land_history import load_land_history_inputs
    from gpuwrf.config.paths import wrf_run_dir
    initial_lands={name:jax.device_get(carry.noahmp_land) for name,carry in carries.items()
                   if carry.noahmp_land is not None}
    land_inputs={name:load_land_history_inputs(input_view/f'wrfinput_{name}',table_dir=wrf_run_dir(),
        parameters=bundles[name].namelist.noahmp_static.parameters,
        field_overrides=bundles[name].writer_static_overrides) for name in initial_lands}
    nesting=meta["nesting"]
    assert nesting==dict(feedback=True,smooth_option=2,spec_zone=1),nesting
    report["load_s"]=time.monotonic()-start
    for name,carry in carries.items():
        observe(name,0,carry)
    if any(f['nonfinite'] or any(v['gate']=='FAIL' for v in f['fields'].values()) for f in report['frames']):
        report.update(status='FAIL_INITIAL',verdict='FAIL')
        save()
        raise SystemExit(2)  # Do not pay for step compilation after a hard initial failure.
    tree=DomainTree.from_domains(hierarchy,bundles,feedback_enabled=True,
        feedback_smooth_option=nesting["smooth_option"],feedback_spec_zone=nesting["spec_zone"])
    if steps:
        result=run_operational_domain_tree(tree,root_steps=steps,carries=carries,output=observe,
            output_cadence_steps={"d01":1,"d02":1},block_between=True,event_aware_fusion_k=0)
        report["own_steps"]=result.own_steps
    report["status"]="COMPLETE"
    report["core_verdict"]="PASS" if all(f["nonfinite"]==0 and all(v["gate"]!="FAIL" for v in f["fields"].values()) for f in report["frames"]) else "FAIL"
    report["verdict"]="CPU_COUPLED_D6_PASS_AUX_REPORTED" if report["core_verdict"]=="PASS" else "FAIL"
    save()
except Exception as exc:
    report.update(status="ERROR",error=f"{type(exc).__name__}: {exc}")
    save()
    raise
