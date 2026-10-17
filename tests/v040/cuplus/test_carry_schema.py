"""Exact restart retention and malformed-payload refusal for cuplus."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state,_namelist
from gpuwrf.coupling.cuplus import initial_carry,CuPlusCarry
from gpuwrf.io.cuplus_schema import validate


@pytest.mark.parametrize('scheme',[7,10,11])
@pytest.mark.parametrize('bad',['missing','extra','profile_shape','scalar_shape','real_dtype','integer_dtype','nonfinite','version'])
def test_rejects_malformed_carry(scheme,bad):
    grid=_grid();state=_state(grid);nml=_namelist(grid,cu_physics=scheme)
    carry=initial_carry(state,scheme);held=dict(carry.held);version=1
    if bad=='missing':held.pop('RTHCUTEN')
    if bad=='extra':held['unexpected']=jnp.float32(0.)
    if bad=='profile_shape':held['RTHCUTEN']=held['RTHCUTEN'][:,:-1]
    if bad=='scalar_shape':held['RAINCV']=held['RAINCV'][:,None]
    if bad=='real_dtype':held['RTHCUTEN']=np.asarray(held['RTHCUTEN'],np.float64)
    if bad=='integer_dtype':
        key='ENTROPY_FAILED' if scheme==7 else 'ISHALL';held[key]=np.asarray(held[key],np.int64)
    if bad=='nonfinite':held['RTHCUTEN']=held['RTHCUTEN'].at[0,0].set(jnp.nan)
    if bad=='version':version=0
    with pytest.raises(ValueError,match='cuplus restart'):
        validate(CuPlusCarry(carry.w0avg,held),nml,state.theta.shape,version)


@pytest.mark.parametrize('scheme',[7,10,11])
@pytest.mark.parametrize('format',['checkpoint','netcdf'])
def test_both_restart_formats_preserve_every_driver_leaf(tmp_path,scheme,format):
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.coupling.scan_adapters import initial_camuw_carry
    grid=_grid();state=_state(grid)
    nml=_namelist(grid,cu_physics=scheme,bl_pbl_physics=9 if scheme==7 else 5)
    cu=initial_carry(state,scheme)
    # Distinct nonzero timer/rates/diagnostics expose silent omission or resets.
    held={k:(v+np.int32(0) if v.dtype==jnp.int32 else v+jnp.float32(.125)) for k,v in cu.held.items()}
    carry=initial_operational_carry(state).replace(cuplus=CuPlusCarry(cu.w0avg+jnp.float32(.25),held))
    if scheme==7:carry=carry.replace(camuw_pbl=initial_camuw_carry(state))
    if format=='checkpoint':
        from gpuwrf.io.restart import write_restart,read_restart
        path=tmp_path/'checkpoint.pkl';write_restart(carry,nml,grid,7,path)
        actual,_,_,step=read_restart(path,namelist=nml);assert step==7
    else:
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        path=tmp_path/'wrfrst.nc';write_wrfrst_carry(carry,grid,nml,path,
            valid_time='2023-01-15_13:00:00',run_start='2023-01-15_12:00:00',step_index=7)
        actual,_=read_wrfrst_carry(path,namelist=nml)
    left=jax.tree.leaves(carry.cuplus);right=jax.tree.leaves(actual.cuplus)
    assert len(left)==len(right)
    for x,y in zip(left,right,strict=True):
        assert x.dtype==y.dtype
        np.testing.assert_array_equal(np.asarray(x).view(np.uint32),np.asarray(y).view(np.uint32))


def test_missing_active_carry_and_failed_original_cam_entropy_are_refused():
    grid=_grid();state=_state(grid);nml=_namelist(grid,cu_physics=7)
    with pytest.raises(ValueError,match='missing'):validate(None,nml,state.theta.shape)
    cu=initial_carry(state,7);held=dict(cu.held,ENTROPY_FAILED=jnp.ones_like(cu.held['ENTROPY_FAILED']))
    with pytest.raises(ValueError,match='entropy'):
        validate(CuPlusCarry(cu.w0avg,held),nml,state.theta.shape,1)


def test_original_cup_defaults_explicit_bindings_and_static_key_roundtrip():
    from gpuwrf.io.namelist_binding import bind_namelist_options
    grid=_grid();nml=_namelist(grid)
    assert nml.cup_num_bins==1 and nml.cup_r_bin_size==1.
    bound=bind_namelist_options(nml,{'physics':{'cu_physics':[10], 'numbins':[5],
        'thbinsize':[.1],'rbinsize':[.0001],'mindeepfreq':[.1],'minshallowfreq':[.001]}},'d01')
    assert (bound.cup_num_bins,bound.cup_th_bin_size,bound.cup_r_bin_size)==(5,.1,.0001)
    assert bound.cu_diag==1
    leaves,tree=jax.tree.flatten(bound);restored=jax.tree.unflatten(tree,leaves)
    assert restored.cup_min_shallow_freq==.001 and restored.cup_min_deep_freq==.1
    mskf=replace(nml,cu_physics=11,cu_rad_feedback=False,icloud=0)
    assert mskf.cu_rad_feedback and mskf.icloud==1
