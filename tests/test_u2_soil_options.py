"""Original pedotransfer and expanded per-layer energy/water parameter ABI."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp.soil_options import pedotransfer_sr2006
    fx=json.loads((ROOT/'proofs/u2_land/soil_oracle.json').read_text());cases=fx['cases']
    inputs=tuple(engine.asarray([c[k] for c in cases],engine.float32).T for k in ('sand','clay','orgm'))
    coeff={k:engine.asarray(v,engine.float32) for k,v in fx['coefficients'].items()}
    def compute(sand,clay,orgm,coeff):
        values=pedotransfer_sr2006(sand,clay,orgm,coeff)
        return engine.stack([values[key] for key in fx['outputs']],axis=0).transpose(2,0,1)
    return np.asarray(jax.jit(compute)(*inputs,coeff) if engine is jnp else compute(*inputs,coeff)),cases,fx


def test_original_pedotransfer_runtime_inputs():
    actual,cases,fx=matrix()
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c['expected'] for c in cases],**fx['nominal_tolerance'])


@pytest.mark.parametrize('mode',[2,3])
def test_actual_soil_binding_reaches_energy_water_and_keeps_cold_init(tmp_path,monkeypatch,mode):
    from dataclasses import replace
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.io.noahmp_soil import bind_soil,SelectedSoilParameters
    from gpuwrf.physics.noahmp.water_hydro import _soil_param
    from gpuwrf.contracts.grid import GridSpec
    from test_noahmp_checkpoint_v2 import _namelist
    import importlib.util
    from types import SimpleNamespace
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1')
    path=ROOT/'tests/v025/b_noahmp/test_snow_wrf_order_oracle.py'
    spec=importlib.util.spec_from_file_location('soil_columns',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    c=mod._build(json.loads((path.parent/'fixtures/swiss_snow_wrf_order.json').read_text())['columns'][:2])
    nl=replace(_namelist(GridSpec.canary_3km_template()),noahmp_static=c['static'],noahmp_land=c['land'],noahmp_options=NoahMPOptions(opt_soil=mode))
    target=tmp_path/'wrfinput_d01'
    with Dataset(target,'w') as ds:
        for key,n in (('Time',1),('south_north',1),('west_east',2),('soil_comp',8)):ds.createDimension(key,n)
        for k in range(1,5):ds.createVariable('SOILCL'+str(k),'f4',('Time','south_north','west_east'))[:]=k+1.5
        ds.createVariable('SOILCOMP','f4',('Time','soil_comp','south_north','west_east'))[:]=np.array([40.,50.,60.,70.,30.,20.,10.,5.],np.float32)[None,:,None,None]
    out=bind_soil(nl,tmp_path,'d01',state=SimpleNamespace(t_skin=c['land'].t_skin))
    assert isinstance(out.noahmp_static.parameters,SelectedSoilParameters)
    for key in fx_outputs():
        value=getattr(out.noahmp_static.parameters,key)
        assert value.shape==(4,1,2) and value.dtype==jnp.float32
        np.testing.assert_array_equal(_soil_param(out.noahmp_static.parameters,key,out.noahmp_static,c['land'].smois,0.),value)
        if key in out.noahmp_energy_params._fields:np.testing.assert_array_equal(getattr(out.noahmp_energy_params,key),value)
    np.testing.assert_array_equal(out.noahmp_land.sh2o,nl.noahmp_land.sh2o)
    if mode==2:
        table=np.asarray(c['static'].parameters.bexp,np.float32)
        np.testing.assert_array_equal(out.noahmp_energy_params.bexp[:,0,0],table[[3,4,5,6]])
    else:
        assert not np.array_equal(out.noahmp_energy_params.bexp,c['ep'].bexp)


def fx_outputs():
    from gpuwrf.physics.noahmp.soil_options import OUTPUTS
    return OUTPUTS
