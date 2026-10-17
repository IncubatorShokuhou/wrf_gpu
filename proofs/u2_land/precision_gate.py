"""Source REAL4 BYTE and compiled-runtime arithmetic vs independent WRF ULP envelopes.

All numerical inputs are runtime JIT arguments. The independent pristine
envelope is frozen before inspecting candidate errors; nominal tolerances stay.
Usage: python precision_gate.py SOURCE_ROOT ULP_JSON OUTPUT_JSON
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

TREE,BOUND,OUT=map(Path,sys.argv[1:4])
sys.path[:0]=[str(TREE/"src"),str(Path(__file__).parent)]
os.environ["GPUWRF_NOAHMP_NATIVE_REAL"]="1"
import jax
import jax.numpy as jnp
import numpy as np
from literal_real32 import NumpyReal,source_backend
from gpuwrf.physics.noahmp.precip_heat import _atm_partition
from gpuwrf.physics.noahmp.energy_radiation import TwoStreamParams,snow_age,snowalb_bats,snowalb_class
from gpuwrf.physics.noahmp.irrigation import (IrrigationParameters,IrrigationStatic,seed_irrigation,
    method_fractions,trigger_irrigation,philip_infiltration,sprinkler_irrigation,micro_irrigation,flood_irrigation)
from gpuwrf.coupling.sst_skin_hook import SSTSkinState,sst_skin_update
from gpuwrf.physics.noahmp.surface_options import soil_moisture_stress,evaporation_resistance
from gpuwrf.physics.noahmp.radiation_options import canopy_gaps

assert jax.devices()[0].platform=="cpu"
envelope=json.loads(BOUND.read_text())["bounds"]
surface_fixture=TREE/"proofs/u2_land/surface_oracle.json"
if surface_fixture.exists():
    cases=json.loads(surface_fixture.read_text())["cases"]
    envelope["surface"]=dict(nominal=[c["expected"] for c in cases],low=[c["low"] for c in cases],high=[c["high"] for c in cases])
phenology_fixture=TREE/"proofs/u2_land/phenology_oracle.json"
if phenology_fixture.exists():
    cases=json.loads(phenology_fixture.read_text())["cases"]
    envelope["phenology"]=dict(nominal=[c["expected"] for c in cases],low=[c["low"] for c in cases],high=[c["high"] for c in cases])
gap_fixture=TREE/"proofs/u2_land/gap_oracle.json"
if gap_fixture.exists():
    cases=json.loads(gap_fixture.read_text())["cases"]
    envelope["gaps"]=dict(nominal=[c["expected"] for c in cases],low=[c["low"] for c in cases],high=[c["high"] for c in cases])
dynamic_fixture=TREE/"proofs/u2_land/phenology_dynamic_oracle.json"
if dynamic_fixture.exists():
    cases=json.loads(dynamic_fixture.read_text())["cases"]
    envelope["phenology_dynamic"]=dict(nominal=[c["expected"] for c in cases],low=[c["low"] for c in cases],high=[c["high"] for c in cases])
literal_only="--literal-only" in sys.argv[4:]
selected={value for value in sys.argv[4:] if not value.startswith("--")} or {"snow","irrigation","skin"}
report=dict(source_tree=str(TREE),source_commit=subprocess.check_output(["git","-C",str(TREE),"rev-parse","HEAD"],text=True).strip(),
            selected_cases=sorted(selected),platform="cpu",gpu_executed=False,bounds_sha256=hashlib.sha256(BOUND.read_bytes()).hexdigest(),
            method="unchanged-body host REAL4/libm BYTE; JAX jit with real runtime inputs vs pristine +/-1inputULP",results={})
report["literal_only"]=literal_only
if surface_fixture.exists():report["surface_fixture_sha256"]=hashlib.sha256(surface_fixture.read_bytes()).hexdigest()
report["model_source_sha256"]={fn.__module__:hashlib.sha256(Path(sys.modules[fn.__module__].__file__).read_bytes()).hexdigest()
    for fn in (_atm_partition,snow_age,trigger_irrigation,philip_infiltration,sst_skin_update,soil_moisture_stress,evaporation_resistance)}


def record(name,literal,compiled):
    bounds=envelope[name]
    expected=np.asarray(bounds["nominal"],np.float32)
    literal=np.asarray(literal,np.float32)
    assert literal.shape==expected.shape and np.isfinite(literal).all() and np.isfinite(expected).all()
    same=literal.view(np.uint32)==expected.view(np.uint32)
    if compiled is None:
        report["results"][name]=dict(values=int(expected.size),literal_byte=int(same.sum()),literal_differing=int((~same).sum()),
                                    literal_differences=np.argwhere(~same).tolist())
        OUT.write_text(json.dumps(report,indent=2)+"\n")
        return
    compiled=np.asarray(compiled,np.float32)
    assert compiled.shape==expected.shape and np.isfinite(compiled).all()
    lo,hi=np.asarray(bounds["low"],np.float32),np.asarray(bounds["high"],np.float32)
    excess=np.maximum(np.maximum(lo.astype(float)-compiled,compiled.astype(float)-hi),0.)
    def ordered(value):
        bits=value.view(np.uint32).astype(np.int64)
        return np.where(bits&0x80000000,0x80000000-(bits&0x7fffffff),0x80000000+bits)
    ulp=np.abs(ordered(compiled)-ordered(expected))
    report["results"][name]=dict(values=int(expected.size),literal_byte=int(same.sum()),literal_differing=int((~same).sum()),
        compiled_within_ulp_envelope=int((excess==0.).sum()),compiled_outside=int((excess>0.).sum()),
        compiled_max_ulp=int(ulp.max()),compiled_max_abs=float(np.abs(compiled.astype(float)-expected).max()),
        max_excess=float(excess.max()),literal_differences=np.argwhere(~same).tolist(),
        compiled_excess_indices=np.argwhere(excess>0.).tolist())
    OUT.write_text(json.dumps(report,indent=2)+"\n")


def snow_matrix(engine):
    fx=json.loads((TREE/"proofs/u2_land/snow_oracle.json").read_text())
    param=fx["parameters"]
    bats=tuple(param[key] for key in ("BATS_COSZ","BATS_VIS_NEW","BATS_NIR_NEW","BATS_VIS_AGE","BATS_NIR_AGE","BATS_VIS_DIR","BATS_NIR_DIR"))
    results=[]
    for code in (1,2,3,4,5):
        data=np.asarray([case["inputs"][1:] for case in fx["cases"] if case["inputs"][0]==code],np.float32)
        def compute(values):
            values=values.T[:,None,:]
            force=SimpleNamespace(**dict(zip(fx["variables_in"][1:],values)))
            p=TwoStreamParams(*([None]*7),**{key:engine.asarray(param[key.upper()],engine.float32)
                for key in ("tau0","grain_growth","extra_growth","dirt_soot","swemx")})
            precip=_atm_partition(force,code)
            age,fage=snow_age(force.dt,force.tg,force.sneqvo,force.sneqv,force.tauss,p)
            direct,diffuse=snowalb_bats(force.cosz,fage,bats)
            alb,ad,ai=snowalb_class(precip[2],force.dt,force.albold,p)
            return engine.stack((*precip,age,fage,*direct,*diffuse,alb,*ad,*ai),-1)[0]
        result=jax.jit(compute)(jnp.asarray(data)) if engine is jnp else compute(engine.asarray(data,engine.float32))
        results.append(np.asarray(result))
    return np.concatenate(results)


def irrigation_matrix(engine):
    fx=json.loads((TREE/"proofs/u2_land/irrigation_oracle.json").read_text())
    outputs=[]
    for irr in (1,2,3):
        for method in (0,1,2,3):
            cases=[case for case in fx["cases"] if case["opt_irr"]==irr and case["opt_irrm"]==method]
            data=np.asarray([[c["nroot"],c["pltday"],c["hsday"],*c["sh2o"],*c["smc"],c["fveg"],c["julian"],c["irrfra"],c["lai"],
                *c["fractions"],c["t2"],c["u"],c["v"],c["eair"],*c["amounts"],*c["counts"],c["dt"]] for c in cases],np.float32)
            def compute(values):
                x=values.T[:,None,:]
                params={key:engine.asarray(value,engine.float32).reshape(4,1,1) if isinstance(value,list)
                        else engine.asarray(value,engine.float32) for key,value in fx["parameters"].items()}
                p=IrrigationParameters(**params,nroot=x[0].astype(engine.int32),pltday=x[1],hsday=x[2])
                sh,smc=x[3:7],x[7:11]
                bundle=IrrigationStatic(p,x[13],x[15],x[16],x[17],engine.ones_like(x[13],dtype=bool))
                state=seed_irrigation((1,len(cases)))._replace(irwatsi=x[22],irwatmi=x[23],irwatfi=x[24],
                    irnumsi=x[25].astype(engine.int32),irnummi=x[26].astype(engine.int32),irnumfi=x[27].astype(engine.int32))
                fractions=method_fractions(bundle,method)
                state=trigger_irrigation(p,fx["zsoil"],sh,x[11],x[12],x[13],x[14],fractions,state,opt_irr=irr,opt_irrm=method)
                amount_s,rate_s,evap,_=sprinkler_irrigation(p,x[28],sh,smc,x[18],x[19],x[20],x[21],fractions[0],state.irwatsi)
                live=state.irwatsi>0.
                amount_s,rate_s,evap=(engine.where(live,value,0.) for value in (amount_s,rate_s,evap))
                sice=engine.maximum(0.,smc-sh)
                amount_f,rate_f=flood_irrigation(p,x[28],sh,smc,sice,fractions[2],state.irwatfi)
                amount_m,rate_m=micro_irrigation(p,x[28],sh,smc,sice,fractions[1],state.irwatmi)
                fsur=philip_infiltration(p,smc,sh,sice,x[28])
                return engine.stack((state.irnumsi.astype(engine.float32),state.irnummi.astype(engine.float32),state.irnumfi.astype(engine.float32),
                    amount_s,amount_m,amount_f,rate_s,rate_m,rate_f,evap,fsur),-1)[0]
            result=jax.jit(compute)(jnp.asarray(data)) if engine is jnp else compute(engine.asarray(data,engine.float32))
            outputs.append(np.asarray(result))
    return np.concatenate(outputs)


def skin_matrix(engine):
    fx=json.loads((TREE/"proofs/u2_land/skin_oracle.json").read_text())
    data=np.asarray(fx["inputs"],np.float32)
    def compute(values,skin):
        x=values.T
        return sst_skin_update(*x[:8],skin,x[10,0])
    fn=jax.jit(compute) if engine is jnp else compute
    values=engine.asarray(data,engine.float32)
    skin=SSTSkinState(values[:,5],values[:,8],values[:,9])
    outputs=[]
    for _ in range(24):
        skin=fn(values,skin)
        outputs.append(np.stack((np.asarray(skin.sstsk),np.asarray(skin.dtw)),-1))
    return np.stack(outputs)


def surface_matrix(engine):
    fx=json.loads(surface_fixture.read_text())
    outputs=[]
    p=fx["parameters"]
    for btr in (1,2,3):
        for rsf in (1,2,3,4):
            cases=[c for c in fx["cases"] if c["opt_btr"]==btr and c["opt_rsf"]==rsf]
            values=np.asarray([[*c["sh2o"],c["fsno"]] for c in cases],np.float32)
            def compute(values):
                sh2o=values[:,:4].T
                fsno=values[:,4]
                gx=soil_moisture_stress(sh2o,*[engine.asarray(p[key],engine.float32)[:,None]
                    for key in ("smcwlt","smcref","smcmax","psisat","bexp")],opt_btr=btr)
                rsurf=evaporation_resistance(sh2o[0],p["smcmax"][0],p["smcwlt"][0],p["bexp"][0],fx["zsoil"][0],p["rsurf_exp"],fsno,
                                             opt_rsf=rsf,rsurf_snow=p["rsurf_snow"])
                return engine.concatenate((gx,rsurf[None,:]),axis=0).T
            result=jax.jit(compute)(jnp.asarray(values)) if engine is jnp else compute(engine.asarray(values,engine.float32))
            outputs.append(np.asarray(result))
    return np.concatenate(outputs)


def phenology_matrix(engine):
    import importlib.util
    spec=importlib.util.spec_from_file_location("land_phenology_gate",TREE/"tests/test_u2_phenology_options.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return np.concatenate([mod.matrix(dveg,engine)[0] for dveg in (1,3,4,7,8,9)])


def gap_matrix(engine):
    cases=json.loads(gap_fixture.read_text())["cases"]
    outputs=[]
    for opt in (1,2,3):
        values=engine.asarray([[c[k] for k in ("cosz","vai","fveg","rc","hvt","hvb")] for c in cases if c["opt_rad"]==opt],engine.float32)
        def compute(values):return engine.stack(canopy_gaps(*values.T,opt_rad=opt),axis=-1)
        outputs.append(np.asarray(jax.jit(compute)(values) if engine is jnp else compute(values)))
    return np.concatenate(outputs)


def dynamic_phenology_matrix(engine):
    import importlib.util
    spec=importlib.util.spec_from_file_location("land_dynamic_gate",TREE/"tests/test_u2_phenology_options.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return np.concatenate([mod.matrix(dveg,engine)[0] for dveg in (2,5,6)])


for name,fn in (("snow",snow_matrix),("irrigation",irrigation_matrix),("skin",skin_matrix),("surface",surface_matrix),("phenology",phenology_matrix),("gaps",gap_matrix),("phenology_dynamic",dynamic_phenology_matrix)):
    if name not in selected:continue
    with source_backend():literal=fn(NumpyReal())
    compiled=None if literal_only else fn(jnp)
    record(name,literal,compiled)
    print(name,report["results"][name],flush=True)
report["literal_status"]="PASS" if all(row["literal_differing"]==0 for row in report["results"].values()) else "FAIL"
report["compiled_envelope_status"]="NOT_RUN" if literal_only else (
    "PASS" if all(row["compiled_outside"]==0 for row in report["results"].values()) else "DEVIATION")
OUT.write_text(json.dumps(report,indent=2)+"\n")
