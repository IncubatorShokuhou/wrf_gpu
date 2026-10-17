"""Check the real surface hooks reach the actual solver input with active amounts."""
import importlib.util
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pytest
import gpuwrf.runtime.noahmp_precipitation as precip
import gpuwrf.coupling.noahclassic_surface_hook as noah
import gpuwrf.coupling.pleim_xiu_surface_hook as px

ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/F24/candidate')
spec=importlib.util.spec_from_file_location('surface_smoke',ROOT/'tests/test_v013_operational_smoke.py')
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)

class Captured(Exception):pass

@pytest.mark.parametrize('lsm',[2,7])
def test_active_carry_reaches_real_lsm_solver(monkeypatch,lsm):
    original=precip.seed_precipitation
    def seed(state,**kw):
        held=original(state,**kw)
        updates=dict(prcpconv=jnp.full(state.t_skin.shape,.002,jnp.float32),
                     prcpnonc=jnp.full(state.t_skin.shape,.003,jnp.float32))
        if lsm==2:updates['sr']=jnp.full(state.t_skin.shape,.4375,jnp.float32)
        return held._replace(**updates)
    monkeypatch.setattr(precip,'seed_precipitation',seed)
    observed=[]
    if lsm==2:
        def kernel(forcing,*args,**kw):
            observed.append((np.asarray(forcing.prcp),np.asarray(forcing.ffrozp)))
            raise Captured()
        monkeypatch.setattr(noah,'sflx_step',kernel)
        function=smoke.test_sf_surface_noahclassic_operational_runs_and_advances_land
    else:
        def kernel(*args,**kw):
            observed.append(np.asarray(args[11]))
            raise Captured()
        monkeypatch.setattr(px,'pxlsm_columns',kernel)
        function=smoke.test_sf_surface_pleim_xiu_operational_runs_and_advances_land
    with pytest.raises(Captured):function()
    assert len(observed)==1
    if lsm==2:
        np.testing.assert_allclose(observed[0][0],.005,rtol=1e-6,atol=0.)
        np.testing.assert_array_equal(observed[0][1],.4375)
    else:
        np.testing.assert_allclose(observed[0],5.e-6,rtol=1e-6,atol=0.)
