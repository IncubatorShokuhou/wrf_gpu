"""Independent F2 two-step caller audit: actual CU adapters, sentinel column kernels."""
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
from gpuwrf.runtime import operational_mode as op

ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/F24/candidate')
spec=importlib.util.spec_from_file_location('rf_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
f=importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


@pytest.mark.parametrize('mp',[6,0])
@pytest.mark.parametrize('cu',[2,3,5,6,16,93])
def test_real_cu_adapter_two_step_noah_forcing(monkeypatch,cu,mp):
    grid=f._grid(ny=3,nx=4,nz=8)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=mp)
    # A huge rounded accumulated total would erase these amounts if differenced.
    state=state.replace(rainc_acc=jnp.full(state.t_skin.shape,1.e8,jnp.float32))
    dt=18.
    nl=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),dt_s=dt,
        mp_physics=mp,cu_physics=cu,bl_pbl_physics=0,sf_sfclay_physics=5,use_noahmp=True,
        run_physics=True,run_boundary=False,disable_guards=True,rad_rk_tendf=1,use_flux_advection=True)
    op._resolve_operational_suite(nl)  # The regression must be an admitted configuration.
    monkeypatch.setenv('GPUWRF_FULL_WRFOUT_VARIABLES','0')
    monkeypatch.setenv('GPUWRF_MICROPHYSICS_WRF_ORDER','1')
    calls=[]
    amount=[.36]
    def column(*args,**kw):
        calls.append(cu)
        zero=jnp.zeros_like(args[0])
        rates={k:zero for k in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RUCUTEN','RVCUTEN')}
        rates['RAINCV']=jnp.asarray(amount[0],args[0].dtype)
        return rates
    if cu in (6,16):
        monkeypatch.setattr(sa,'tiedtke_column_jax' if cu==6 else 'ntiedtke_column_jax',column)
    elif cu==2:
        def bmj(*args,**kw):
            calls.append(cu)
            zero=jnp.zeros_like(args[0])
            tendency=SimpleNamespace(state_tendencies={'theta':zero,'qv':zero},
                accumulator_increments={'rainc_acc':jnp.asarray(amount[0],zero.dtype)})
            return SimpleNamespace(tendency=tendency,carry=SimpleNamespace(cumulus={'cldefi':jnp.asarray(.8,zero.dtype)}))
        monkeypatch.setattr(sa,'step_bmj_column',bmj)
    elif cu==3:
        def gf(*args,**kw):
            calls.append(cu)
            zero=jnp.zeros_like(args[0])
            return {**{k:zero for k in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')},
                    'RAINCV':jnp.full((12,),amount[0],zero.dtype)}
        monkeypatch.setattr(sa,'gfdrv_batched',gf)
    else:
        import gpuwrf.physics._grell_cup_jax as grell
        def tile(**kw):
            calls.append(cu)
            zero=jnp.zeros_like(kw['t'])
            return {**{k:zero for k in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')},
                    'PRATEC':jnp.full(state.t_skin.shape,amount[0]/dt,zero.dtype)}
        monkeypatch.setattr(grell,'g3drv_tile' if cu==5 else 'grelldrv_tile',tile)
    monkeypatch.setitem(op.SFCLAY_SCAN_ADAPTERS,5,lambda st,*a,**kw:st)
    monkeypatch.setattr(op,'_rk_scan_step',lambda c,*a,**kw:c)
    # MP solver is not this audit's subject; retain actual post-RK dispatch.
    z=jnp.zeros_like(state.t_skin)
    water=dict(rain=z,snow=z,graupel=z,ice=z,total=z)
    monkeypatch.setitem(op.MP_SCAN_ADAPTERS,6,lambda st,*a,**kw:(st,water) if kw.get('return_precipitation') else st)
    observed=[]
    def surface(st,land,static,dt,**kw):
        view=_build_column_view(st,grid)
        if kw.get('precipitation') is not None:view=view.replace(**{k:v for k,v in kw['precipitation']._asdict().items() if k!='sr'})
        observed.append(assemble_noahmp_forcing(view,static,kw['radiation'],kw['clock'],dt))
        return st,land
    monkeypatch.setattr(op,'noahmp_surface_step',surface)
    carry=op._initial_carry_for_run(state,nl).replace(noahmp_rad=(z,z,z))
    for step in (1,2):
        amount[0]=.36*step
        carry=op._physics_boundary_step(carry,nl,jnp.int32(step),run_radiation=False)
    assert len(calls)==2
    np.testing.assert_array_equal(observed[0].prcpconv,0.)
    np.testing.assert_allclose(observed[1].prcpconv,np.float32(.36)/np.float32(dt),rtol=1.e-6,atol=0.)
    np.testing.assert_allclose(carry.noahmp_precipitation.prcpconv,np.float32(.72)/np.float32(dt),rtol=1.e-6,atol=0.)
    assert all(x.dtype==jnp.float32 for x in carry.noahmp_precipitation[:5])
