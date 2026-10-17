"""Liu crop geography, old growth-stage phenology and actual Noah-MP consumer."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_crop_fraction_strict_selection_and_initial_pools():
    from gpuwrf.io.noahmp_crop import crop_initialization
    field=np.zeros((5,1,5),np.float32);field[4]=1.
    field[0,0,:]=[.7,.2,.4,.7,.7];field[1,0,:]=[.2,.7,.4,.2,.2]
    field[4,0,3]=.49
    inputs={key:np.full((1,5),value,np.float32) for key,value in (('LAI_INIT',2.),('XSAI_INIT',.2),('LFMASS',60.),('RTMASS',500.),('STMASS',66.),('WOOD',500.),('STBLCP',1000.),('FASTCP',1000.))}
    cat,pools=crop_initialization(inputs,SimpleNamespace(ivgtyp=np.array([[14,12,10,14,17]])),{'DEFAULT_CROP':[0]},field)
    np.testing.assert_array_equal(cat,[[1,2,0,0,0]])
    np.testing.assert_array_equal(pools[1],[[0.,0.,0.,500.,500.]])
    np.testing.assert_array_equal(pools[0][0,:3],np.float32(2.)/np.array([.015,.030,.035],np.float32))


def test_crop_reaches_actual_noah_driver_and_old_pgs(monkeypatch):
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1');monkeypatch.setenv('GPUWRF_NOAHMP_COLUMN_KERNELS','0');jax.clear_caches()
    path=ROOT/'tests/v025/b_noahmp/test_snow_wrf_order_oracle.py'
    spec=importlib.util.spec_from_file_location('crop_columns',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    column=json.loads((path.parent/'fixtures/swiss_snow_wrf_order.json').read_text())['columns'][0].copy();column['vegtyp']=12
    c=mod._build([column])
    from gpuwrf.physics.noahmp.crop import CropCarbonState,CropParameters,CropStaticParameters
    from gpuwrf.physics.noahmp.carbon import CarbonParameters
    from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    fx=json.loads((ROOT/'proofs/u2_land/crop_oracle.json').read_text());case=fx['cases'][2]
    crop=CropParameters(*(jnp.full((1,1),case['parameters'][k],jnp.float32) for k in fx['scalar_keys']),
        *(jnp.asarray(case['parameters'][k],jnp.float32)[:,None,None] for k in fx['array_keys']))
    natural=CarbonParameters(*(jnp.ones((1,1),jnp.float32) for _ in CarbonParameters._fields))
    p=CropStaticParameters(natural,crop,jnp.ones((1,1),jnp.int32))
    pools=CropCarbonState(*(jnp.full((1,1),v,jnp.float32) for v in case['state']),jnp.full((1,1),4,jnp.int32))
    forcing=c['forcing']._replace(julian=jnp.float32(case['julian']))
    land=c['land'].replace(lai=jnp.full((1,1),2.,jnp.float32),sai=jnp.full((1,1),.2,jnp.float32))
    phen=noahmp_phenology_table(land,forcing,c['static'],crop=(p.croptype,pools.pgs))
    np.testing.assert_array_equal(phen.lai,land.lai);np.testing.assert_array_equal(phen.igs,[[1.]])
    phen_before=noahmp_phenology_table(land,forcing,c['static'],crop=(p.croptype,jnp.zeros((1,1),jnp.int32)))
    np.testing.assert_array_equal(phen_before.igs,[[0.]])
    result=noah_mp_step(land,forcing,c['static'],c['dt'],energy_params=c['ep'],rad_params=c['rp'],
        options=NoahMPOptions(opt_crop=1),carbon=(p,pools),history=True)
    assert len(result)==3 and isinstance(result[2],CropCarbonState)
    assert result[2].pgs.dtype==jnp.int32
    # Cold Swiss ENERGY T2 can be below GDDTBASE; GDD then holds while the
    # original stage selection still replaces the old PGS=4 with stage 3.
    np.testing.assert_array_equal(result[2].pgs,[[3]])
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    jax.clear_caches()


@pytest.mark.parametrize('problem',['missing_grain','pgs_float','pgs_range','gdd_nan'])
def test_crop_restart_extra_state_fails_closed(problem):
    from gpuwrf.io.noahmp_carbon import validate_restart
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.carbon import CarbonState
    from gpuwrf.physics.noahmp.crop import CropCarbonState
    values=[np.zeros((1,2),np.float32) for _ in range(8)]+[np.zeros((1,2),np.int32)]
    group=CropCarbonState(*values)
    if problem=='missing_grain':group=CarbonState(*values[:6])
    if problem=='pgs_float':group=group._replace(pgs=values[0])
    if problem=='pgs_range':group=group._replace(pgs=np.full((1,2),9,np.int32))
    if problem=='gdd_nan':group=group._replace(gdd=np.full((1,2),np.nan,np.float32))
    with pytest.raises(ValueError,match='carbon restart'):validate_restart(group,SimpleNamespace(noahmp_options=NoahMPOptions(opt_crop=1)),1,(1,2))
