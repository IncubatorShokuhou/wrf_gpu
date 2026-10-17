"""Independent pristine WRF oracle for every Noah-MP irrigation trigger and method."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.noahmp.irrigation import (IrrigationParameters, IrrigationStatic, seed_irrigation,
    method_fractions, trigger_irrigation, philip_infiltration, sprinkler_irrigation, micro_irrigation, flood_irrigation)

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("opt_irr", [1,2,3])
@pytest.mark.parametrize("opt_irrm", [0,1,2,3])
@pytest.mark.parametrize("compiled", [False,True])
def test_pristine_wrf_irrigation_all_regimes(opt_irr,opt_irrm,compiled):
    fx=json.loads((ROOT/"proofs/u2_land/irrigation_oracle.json").read_text())
    cases=[c for c in fx["cases"] if c["opt_irr"]==opt_irr and c["opt_irrm"]==opt_irrm]
    r=lambda key:jnp.asarray([c[key] for c in cases],jnp.float32).reshape(1,-1)
    p=IrrigationParameters(**{**{key:jnp.asarray(value,jnp.float32).reshape((4,1,1)) if isinstance(value,list)
                                       else jnp.asarray(value,jnp.float32) for key,value in fx["parameters"].items()},
                              "pltday":r("pltday"),"hsday":r("hsday"),"nroot":r("nroot").astype(jnp.int32)})
    fractions=jnp.asarray([c["fractions"] for c in cases],jnp.float32).T[:,None,:]
    sh2o=jnp.asarray([c["sh2o"] for c in cases],jnp.float32).T[:,None,:]
    smc=jnp.asarray([c["smc"] for c in cases],jnp.float32).T[:,None,:]
    amounts=jnp.asarray([c["amounts"] for c in cases],jnp.float32).T[:,None,:]
    counts=jnp.asarray([c["counts"] for c in cases],jnp.int32).T[:,None,:]
    fields={key:r(key) for key in ("irrfra","fveg","julian","lai","t2","u","v","eair")}
    def evaluate(sh2o,smc,amounts,counts,fields,p,fractions):
        r=lambda key:fields[key]
        bundle=IrrigationStatic(p,r("irrfra"),*fractions,jnp.ones_like(r("irrfra"),dtype=bool))
        state=seed_irrigation((1,len(cases)))._replace(irnumsi=counts[0],irnummi=counts[1],irnumfi=counts[2],
                                                    irwatsi=amounts[0],irwatmi=amounts[1],irwatfi=amounts[2])
        fractions=method_fractions(bundle,opt_irrm)
        next=trigger_irrigation(p,fx["zsoil"],sh2o,r("fveg"),r("julian"),r("irrfra"),r("lai"),fractions,state,
                                opt_irr=opt_irr,opt_irrm=opt_irrm)
        si,mi,fi=fractions
        amount_s,rate_s,evap,bad=sprinkler_irrigation(p,18.,sh2o,smc,r("t2"),r("u"),r("v"),r("eair"),si,next.irwatsi)
        active_s=next.irwatsi>0.
        amount_s,rate_s,evap=(jnp.where(active_s,x,0.) for x in (amount_s,rate_s,evap))
        sice=jnp.maximum(0.,smc-sh2o)
        amount_f,rate_f=flood_irrigation(p,18.,sh2o,smc,sice,fi,next.irwatfi)
        amount_m,rate_m=micro_irrigation(p,18.,sh2o,smc,sice,mi,next.irwatmi)
        fsur=philip_infiltration(p,smc,sh2o,sice,18.)
        return jnp.stack((next.irnumsi,next.irnummi,next.irnumfi,amount_s,amount_m,amount_f,rate_s,rate_m,rate_f,evap,fsur),-1)
    operands=(sh2o,smc,amounts,counts,fields,p,fractions)
    result=np.asarray(jax.jit(evaluate)(*operands) if compiled else evaluate(*operands))[0]
    expected=np.asarray([c["expected"] for c in cases])
    assert result.dtype==np.float32 and np.isfinite(result).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(result[:,:3],expected[:,:3])
    np.testing.assert_allclose(result[:,3:],expected[:,3:],**fx["tolerance"])


@pytest.mark.parametrize("problem", ["absent","version","dtype","shape","nan","selection"])
def test_irrigation_restart_fail_closed(problem):
    from gpuwrf.io.noahmp_irrigation import validate_restart
    state=seed_irrigation((2,3))
    config,version=(1,0),1
    if problem=="absent":state=None
    if problem=="version":version=None
    if problem=="dtype":state=state._replace(irnumsi=np.zeros((2,3),np.float32))
    if problem=="shape":state=state._replace(irwatsi=np.zeros((2,4),np.float32))
    if problem=="nan":state=state._replace(irwatsi=np.full((2,3),np.nan,np.float32))
    if problem=="selection":config=(0,0)
    with pytest.raises(ValueError,match="irrigation restart"):
        validate_restart(state,config,version,(2,3))


@pytest.mark.parametrize("format", ["pickle","netcdf"])
def test_irrigation_restart_roundtrip_and_selector_guard(tmp_path,format):
    from dataclasses import replace
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_noahmp_checkpoint_v2 import _namelist
    from test_v0110_wrfrst_netcdf import _state
    from gpuwrf.io.noahmp_options import NoahMPOptions
    grid=GridSpec.canary_3km_template()
    state=_state(grid)
    irrigation=seed_irrigation(state.t_skin.shape)
    irrigation=irrigation._replace(irnumsi=irrigation.irnumsi+3,irwatsi=irrigation.irwatsi+.01,
                                  irsivol=irrigation.irsivol+.2,ireloss=irrigation.ireloss+.1)
    carry=initial_operational_carry(state).replace(noahmp_irrigation=irrigation)
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(opt_irr=1,opt_irrm=0))
    requested=replace(nl,noahmp_options=NoahMPOptions(opt_irr=1,opt_irrm=1))
    path=tmp_path/"irrigation"
    if format=="pickle":
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,2,path)
        restored,*_=read_restart(path)
        with pytest.raises(ValueError,match="restart .*differ"):
            read_restart(path,namelist=requested)
    else:
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:00:36",run_start="2026-06-03_00:00:00",step_index=2)
        restored,_=read_wrfrst_carry(path,namelist=nl)
        with Dataset(path,"r") as ds:
            for name,value in irrigation._asdict().items():
                if name!="loss_fallback":np.testing.assert_array_equal(ds[name.upper()][0],value)
        with pytest.raises(ValueError,match="restart .*differ"):
            read_wrfrst_carry(path,namelist=requested)
    for expected,actual in zip(irrigation,restored.noahmp_irrigation):
        np.testing.assert_array_equal(expected,actual)


@pytest.mark.parametrize('native_columns',(False,True))
def test_real_column_reaches_soil_irrigation_through_full_noah_driver(monkeypatch,native_columns):
    import importlib.util
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    monkeypatch.setenv('GPUWRF_NOAHMP_COLUMN_KERNELS','1' if native_columns else '0')
    monkeypatch.setenv('GPUWRF_NOAHMP_LAYER_SELECT','1' if native_columns else '0')
    monkeypatch.setenv('GPUWRF_NOAHMP_LAYER_LISTS','1' if native_columns else '0')
    jax.clear_caches()
    path=ROOT/"tests/v025/b_noahmp/test_snow_wrf_order_oracle.py"
    spec=importlib.util.spec_from_file_location("u2_snow_columns",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    columns=json.loads((path.parent/"fixtures/swiss_snow_wrf_order.json").read_text())["columns"][:1]
    c=module._build(columns)
    land=c["land"].replace(sh2o=jnp.full((4,1,1),.12,jnp.float32),smois=jnp.full((4,1,1),.12,jnp.float32))
    forcing=c["forcing"]._replace(prcpconv=jnp.zeros((1,1)),prcpnonc=jnp.zeros((1,1)))
    fx=json.loads((ROOT/"proofs/u2_land/irrigation_oracle.json").read_text())
    params={key:jnp.asarray(value,jnp.float32).reshape(4,1,1) if isinstance(value,list) else jnp.float32(value)
            for key,value in fx["parameters"].items()}
    p=IrrigationParameters(**params,pltday=jnp.zeros((1,1)),hsday=jnp.full((1,1),365.),nroot=jnp.full((1,1),3,jnp.int32))
    one=jnp.ones((1,1),jnp.float32)
    bundle=IrrigationStatic(p,one,one,one*0,one*0,jnp.ones((1,1),bool))
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.physics.noahmp import surface_options
    from gpuwrf.physics.noahmp import noahmp_driver
    selections=[]
    old_phenology,old_water=noahmp_driver.noahmp_phenology_table,noahmp_driver.noahmp_water_hydro
    def phenology(*args,**kw):
        selections.append(("dveg",kw["dveg"]))
        return old_phenology(*args,**kw)
    def water(*args,**kw):
        assert "fveg_override" in kw
        selections.append(("water_fveg",True))
        return old_water(*args,**kw)
    monkeypatch.setattr(noahmp_driver,"noahmp_phenology_table",phenology)
    monkeypatch.setattr(noahmp_driver,"noahmp_water_hydro",water)
    old_stress,old_resistance=surface_options.soil_moisture_stress,surface_options.evaporation_resistance
    def stress(*args,**kw):
        selections.append(("btr",kw["opt_btr"]))
        return old_stress(*args,**kw)
    def resistance(*args,**kw):
        selections.append(("rsf",kw["opt_rsf"]))
        return old_resistance(*args,**kw)
    monkeypatch.setattr(surface_options,"soil_moisture_stress",stress)
    monkeypatch.setattr(surface_options,"evaporation_resistance",resistance)
    try:
        result=noah_mp_step(land,forcing,c["static"],c["dt"],energy_params=c["ep"],rad_params=c["rp"],
                            options=NoahMPOptions(opt_irr=1,opt_irrm=2,opt_btr=3,opt_rsf=4,rsurf_snow=50.,dveg=8,
                                                  opt_tbot=1,opt_stc=3,opt_frz=2,opt_rad=1,opt_crs=2,opt_sfc=2),
                            irrigation=(bundle,seed_irrigation((1,1))))
    except TypeError as error:
        assert False,f"Full Noah driver cannot dispatch irrigation: {error}"
    assert len(result)==3
    assert int(np.asarray(result[2].irnummi).sum())==1
    assert float(np.asarray(result[2].irmivol).sum())>0.
    assert ("btr",3) in selections and ("rsf",4) in selections
    assert ("dveg",8) in selections and ("water_fveg",True) in selections
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(result))
    jax.clear_caches()
