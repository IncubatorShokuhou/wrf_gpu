"""Independent two-step operational precipitation wiring; mocked solvers, not forecast fidelity."""
import dataclasses
import importlib.util
import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
from gpuwrf.runtime import operational_mode as op

TREE = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/candidate')
spec = importlib.util.spec_from_file_location('f1producer', TREE/'tests/v040/test_t_calleraudit_land_precip.py')
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


@pytest.mark.parametrize('code', sorted(producer.KERNEL))
def test_two_step_real_dispatch_to_surface_forcing(monkeypatch, code):
    _two_step(monkeypatch, code)


@pytest.mark.parametrize('code', [5,7,8,9,11,15,17,18,19,21,22,28,38,50,51,52,53,55,56])
def test_two_step_explicit_dispatch_to_surface_forcing(monkeypatch, code):
    _two_step(monkeypatch, code)


def _two_step(monkeypatch, code):
    f = producer._fixture()
    grid = f._grid(nz=8, ny=3, nx=4)
    state = producer._real(f._state(grid).ensure_conditional_leaves(mp_physics=code))
    nl = dataclasses.replace(f._namelist(grid), dt_s=18., mp_physics=code, bl_pbl_physics=0,
        cu_physics=0, sf_sfclay_physics=5, use_noahmp=True, run_physics=True,
        ra_sw_physics=0, ra_lw_physics=0, run_boundary=False, disable_guards=True, rad_rk_tendf=0)
    monkeypatch.setenv('GPUWRF_FULL_WRFOUT_VARIABLES','0')
    monkeypatch.setenv('GPUWRF_MICROPHYSICS_WRF_ORDER','1')
    calls = []
    def kernel(*args, **kwargs):
        calls.append(code)
        amounts = {'rain_acc':7., 'snow_acc':3., 'graupel_acc':2., 'hail_acc':1.}
        increments = {k:jnp.full((12,), v) for k,v in amounts.items() if getattr(state,k,None) is not None}
        diagnostics = {'ri3d':jnp.zeros((12,8))}
        if code == 1:
            diagnostics['rainncv'] = jnp.full((12,),5.)
        return PhysicsTendency(accumulator_increments=increments, diagnostics=diagnostics)
    if code in producer.KERNEL:
        monkeypatch.setattr(sa, producer.KERNEL[code], kernel)
    else:
        # Established explicit adapter contract is a slot sentinel here; this
        # verifies dispatch/carry/cadence, not those adapters' kernel numerics.
        slots = producer._pristine_driver_slots()[18 if code in (17,19,21,22) else code]
        surface = lambda value:jnp.full(state.t_skin.shape,value,jnp.float32)
        snow=3. if 'snowncv' in slots else 0.
        graupel=2. if 'graupelncv' in slots else 0.
        hail=1. if 'hailncv' in slots else 0.
        water=dict(rain=surface(7.-snow-graupel-hail),snow=surface(snow),graupel=surface(graupel),
                   ice=surface(0.),hail=surface(hail),total=surface(7.),rainncv=surface(7.),hailncv=surface(hail))
        if code in (17,18,19,21,22):
            water=dict(rain_liquid=water['rain'],rainncv=surface(7.),snowncv=surface(snow),
                       grplncv=surface(graupel),hailncv=surface(hail))
        def explicit(st,*args,**kw):
            calls.append(code)
            radar=kw.get('return_radar') or kw.get('return_radar_context') or 'radar_depth' in kw
            if radar:return st,water,None
            return (st,water) if kw.get('return_precipitation') else st
        routes={5:('gpuwrf.physics.ferrier.adapter','ferrier_adapter'),
                15:('gpuwrf.physics.ferrier.adapter','ferrier_adapter'),
                7:('gpuwrf.coupling.goddard4ice_adapter','goddard4ice_adapter'),
                38:('gpuwrf.coupling.thompson_hail_adapter','thompson_hail_adapter'),
                9:('gpuwrf.physics.milbrandt2mom.adapter','milbrandt_adapter'),
                11:('gpuwrf.physics.cammg.adapter','cammg_adapter'),
                55:('gpuwrf.physics.jensen_ishmael.adapter','jensen_adapter'),
                56:('gpuwrf.physics.ntu.adapter','ntu_adapter')}
        if code in (8,28):
            monkeypatch.setattr(op,'thompson_adapter' if code==8 else 'thompson_aero_adapter',explicit)
        else:
            module,name=(('gpuwrf.physics.nssl2mom.adapter','nssl2mom_adapter') if code in (17,18,19,21,22) else
                         ('gpuwrf.physics.p3.adapter','p3_adapter') if code in (50,51,52,53) else routes[code])
            monkeypatch.setattr(importlib.import_module(module),name,explicit)
        import gpuwrf.diagnostics.online as online
        monkeypatch.setattr(online,'radar_from_context',lambda held,*a,**kw:held)
    monkeypatch.setitem(op.SFCLAY_SCAN_ADAPTERS, 5, lambda st,*a,**kw:st)
    monkeypatch.setattr(op,'_rk_scan_step',lambda c,*a,**kw:c)
    observed = []
    def surface(st, land, static, dt, **kw):
        # Real operational State -> column view -> real Noah forcing assembler.
        view = _build_column_view(st, grid)
        if kw.get('precipitation') is not None:
            view = view.replace(**kw['precipitation']._asdict())
        observed.append(assemble_noahmp_forcing(view, static, kw['radiation'], kw['clock'], dt))
        return st, land
    monkeypatch.setattr(op,'noahmp_surface_step',surface)
    carry = op._initial_carry_for_run(state,nl)
    z = jnp.zeros_like(state.t_skin)
    carry = carry.replace(noahmp_rad=(z,z,z))
    for step in (1,2):
        carry,_ = op._physics_boundary_step_with_limiter_diagnostics(carry,nl,jnp.int32(step),run_radiation=False)
    assert calls == [code,code]
    slots = producer._pristine_driver_slots()[18 if code in (17,19,21,22) else code]
    expected = dict(prcpnonc=(5. if code == 1 else 7.)/18.,
        prcpsnow=3./18. if 'snowncv' in slots else 0.,
        prcpgrpl=2./18. if 'graupelncv' in slots else 0.,
        prcphail=1./18. if 'hailncv' in slots else 0.)
    for name,value in expected.items():
        np.testing.assert_array_equal(getattr(observed[0],name),0.)
        np.testing.assert_allclose(getattr(observed[1],name),value,rtol=1e-6,atol=0.,err_msg=f'{code}:{name}')
        assert np.all(np.isfinite(getattr(observed[1],name)))
    assert all(x.dtype == jnp.float32 for x in carry.noahmp_precipitation)
