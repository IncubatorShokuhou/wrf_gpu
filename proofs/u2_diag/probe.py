"""CPU-only Swiss loader/trace/coupled-smoke probe and standalone bucket costs.

Run with the wave's affinity, QUIET guard and timeout. O1's CPU harness owns
the Thomas tridiagonal fallback; smoke uses FAST_DEFAULTS=0, identity uses 1.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
LANE=Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag")
MODE,OUTPUT=sys.argv[1:3]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"proofs/o1_nlbind"))
sys.path.insert(0,str(ROOT/"tests"))
import cpu_safe
cpu_safe.install()
import jax
import jax.numpy as jnp
import numpy as np
assert jax.devices()[0].platform == "cpu"
from gpuwrf.diagnostics.precipitation import PrecipitationBuckets, update_buckets


def canon(value):
    text=re.sub(r"name_and_src_info=[^\]\n]*","",str(value))
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?","",text)


result=dict(mode=MODE,source=str(ROOT),backend="cpu",cache="disabled",affinity=sorted(os.sched_getaffinity(0)))
if MODE == "cost":
    shape=(117,267)
    zero=jnp.zeros(shape,jnp.float32)
    args=(PrecipitationBuckets(zero,zero,zero),zero+.01,zero+.02,zero+.03,zero+270.,jnp.float32(18),jnp.float32(60))
    start=time.perf_counter()
    compiled=jax.jit(update_buckets).lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    result["compile_first_s"]=time.perf_counter()-start
    samples=[]
    for _ in range(30):
        start=time.perf_counter(); out=compiled(*args); jax.block_until_ready(out)
        samples.append(time.perf_counter()-start)
    stats=compiled.memory_analysis()
    result.update(shape=shape,warm_median_s=float(np.median(samples)),warm_samples_s=samples,
                  memory={key:getattr(stats,key) for key in ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")},
                  hlo_f64_tokens=len(re.findall(r"\bf64\b",compiled.as_text())),
                  held_bytes_on=3*np.prod(shape).item()*4,held_bytes_off=0,
                  finite=all(np.isfinite(np.asarray(a)).all() for a in out),
                  gpu_launches_time_peak="UNKNOWN; no GPU run")
elif MODE == "mutant":
    suite=importlib.import_module("test_online_precipitation")
    original=suite.update_buckets
    killed=[]
    for name in ("reset_deleted","convective_snow_deleted"):
        def mutant(buckets,raincv,rainncv,snowncv,t2,clock,interval):
            return original(buckets,raincv,rainncv,snowncv,
                            jnp.full_like(t2,280.) if name == "convective_snow_deleted" else t2,
                            jnp.float32(1) if name == "reset_deleted" else clock,interval)
        suite.update_buckets=mutant
        try:
            suite.test_pristine_sequential_bucket_operator(False)
        except AssertionError as exc:
            killed.append(dict(mutant=name,exception=type(exc).__name__,message=str(exc)[:800]))
        else:
            raise AssertionError(f"mutant survived: {name}")
        finally:
            suite.update_buckets=original
    result.update(killed=killed,setup_errors=0)
else:
    from gpuwrf.contracts import state as sc
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as om
    from jax.experimental import pallas as pl
    sc._gpu_device=lambda:jax.devices("cpu")[0]
    original=pl.pallas_call
    def interpret(*a,**kw):
        kw["interpret"]=True;kw.pop("compiler_params",None)
        return original(*a,**kw)
    source=Path(os.environ.get("U2_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
    result["case"]=str(source)
    with tempfile.TemporaryDirectory(dir=LANE,prefix="probe_case_") as tmp:
        case=Path(tmp)/"case";case.mkdir()
        for p in source.iterdir():
            if p.name != "namelist.input": (case/p.name).symlink_to(p)
        text=(source/"namelist.input").read_text()
        if MODE in ("smoke", "solar-smoke"):
            text=text.replace("&physics","&physics\n prec_acc_dt=.6,\n do_radar_ref=1,",1)
            text=text.replace("&time_control","&time_control\n nwp_diagnostics=1,",1)
            if MODE == "solar-smoke":
                text=text.replace("&physics", "&physics\n swint_opt=1,", 1)
        (case/"namelist.input").write_text(text)
        config=pipeline.NestedPipelineConfig(case,Path(tmp)/"out",Path(tmp)/"proof",hours=1,max_dom=1)
        pl.pallas_call=interpret
        start=time.perf_counter()
        try:
            _,bundles,_,_,_,carries=pipeline._load_domains(config,("d01",))
        finally:
            pl.pallas_call=original
        ns,carry=bundles["d01"].namelist,carries["d01"]
        result["load_s"]=time.perf_counter()-start
        result["prec_acc_dt"]=ns.prec_acc_dt
        result["nwp_diagnostics"]=ns.nwp_diagnostics
        result["do_radar_ref"]=ns.do_radar_ref
        result["swint_opt"]=ns.swint_opt
        cb=om.build_clock_base(ns)
        if MODE == "identity":
            # Main's portability facade interprets Pallas during real CPU
            # initialization. Trace its original CUDA callable afterwards,
            # exactly as proofs/s1_port/release_identity.py does. CPU Thomas
            # and all actual device/backend guards keep their CPU policy.
            from gpuwrf.backends import pallas as facade
            facade.cuda_native=lambda:True
            result["scope"]="CPU trace of native default program; portable facade native only while tracing; no GPU execution"
            jax.clear_caches();start=time.perf_counter()
            program=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,ns,1,cb,n_steps=1,cadence=int(ns.radiation_cadence_steps)))(carry)
            result.update(sha256=hashlib.sha256(canon(program).encode()).hexdigest(),trace_s=time.perf_counter()-start)
            expected=os.environ.get("U2_EXPECTED_SHA", "2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a")
            assert result["sha256"].startswith(expected), (result["sha256"], expected)
        elif MODE in ("smoke", "solar-smoke"):
            result["steps"]=[]
            for i in (1,2,3):
                start=time.perf_counter()
                carry=om._advance_chunk_fori(carry,ns,i,cb,n_steps=1,cadence=int(ns.radiation_cadence_steps))
                jax.block_until_ready(carry)
                leaves=jax.tree.leaves(carry)
                bad=sum(int((~np.isfinite(np.asarray(x))).sum()) for x in leaves if np.issubdtype(x.dtype,np.floating))
                record=dict(step=i,wall_s=time.perf_counter()-start,nonfinite=bad,
                            bucket_max={name:float(np.max(np.asarray(x))) for name,x in zip(carry.precipitation_buckets._fields,carry.precipitation_buckets)},
                            guards=None if carry.census is None else int(np.asarray(carry.census.guards).sum()))
                record["nwp_max"]={name:float(np.max(np.abs(np.asarray(x)))) for name,x in
                                   zip(carry.nwp_diagnostics._fields,carry.nwp_diagnostics)}
                record["reflectivity_range"]=[float(np.min(np.asarray(carry.radar_reflectivity))),
                                               float(np.max(np.asarray(carry.radar_reflectivity)))]
                if MODE == "solar-smoke":
                    from gpuwrf.diagnostics.solar import history_fields
                    record["solar_flux_range"]={name:[float(np.min(np.asarray(x))),float(np.max(np.asarray(x)))]
                                                for name,x in history_fields(carry.solar_interpolation).items()}
                    record["solar_carry_bytes"]=sum(np.asarray(x).nbytes for x in jax.tree.leaves(carry.solar_interpolation))
                    result["scope"]="Real midnight Swiss coupled CPU smoke; daylight fidelity checked separately against original WRF"
                result["steps"].append(record)
                Path(OUTPUT).write_text(json.dumps(result,indent=2))
                assert bad == 0 and record["guards"] in (0,None)
        else:
            raise ValueError(MODE)
Path(OUTPUT).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
