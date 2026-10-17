"""NSSL radius-only production uses final scaled state, without running radar."""
import jax
import jax.numpy as jnp
import numpy as np

import oracle_io as o
from gpuwrf.physics.nssl2mom import diagnostics, driver
from gpuwrf.physics.nssl2mom.constants import get_constants
from gpuwrf.physics.nssl2mom.indices import FP64
from test_driver_io import fields_in
from test_wiring import _state
from gpuwrf.physics.nssl2mom.adapter import nssl2mom_adapter
# Production order (operational_mode imports it at module level): the adapter's lazy import of
# scan_adapters must not first run inside the jit below, where sfclay_revised_mm5's import-time
# jnp tables would be traced (TracerArrayConversionError when this file runs alone).
import gpuwrf.coupling.scan_adapters  # noqa: F401,E402
from gpuwrf.physics import rrtmg_mp_re as R


def test_radius_only_full_driver_matches_pristine(monkeypatch):
    jax.config.update('jax_enable_x64',True)
    def radar_forbidden(*a,**kw):
        raise AssertionError('Radii must not require radar diagnostics')
    monkeypatch.setattr(diagnostics,'radardd02',radar_forbidden)
    sp=o.load_case('fp64',4)
    C=get_constants('fp64')
    fields={k:jnp.asarray(v,jnp.float64) for k,v in fields_in(sp).items()}
    result=jax.jit(lambda f:driver.nssl2mom_column(f,sp['scalars']['DT'],C,FP64,first_step=True,radii=True))(fields)
    assert set(result[2])=={'re_cloud','re_ice','re_snow'}
    for n,on in zip(('re_cloud','re_ice','re_snow'),('RE_CLOUD_OUT','RE_ICE_OUT','RE_SNOW_OUT')):
        np.testing.assert_allclose(result[2][n],sp['columns'][on],rtol=1e-12,atol=0.)


def test_actual_adapter_holds_pristine_radii_and_disable_stays_absent(monkeypatch):
    jax.config.update('jax_enable_x64',True)
    monkeypatch.setattr(R,'_MP_RE',True)
    sp=o.load_case('fp64',4)
    state=_state(sp).ensure_conditional_leaves(mp_physics=18)
    result=jax.jit(lambda s:nssl2mom_adapter(s,sp['scalars']['DT'],first_step=True))(state)
    for n,on in zip(('re_cloud','re_ice','re_snow'),('RE_CLOUD_OUT','RE_ICE_OUT','RE_SNOW_OUT')):
        assert getattr(result,n).dtype==jnp.float32
        np.testing.assert_allclose(np.asarray(getattr(result,n))[:,0,0],sp['columns'][on],rtol=1e-3,atol=0.)
    # No new radius pytree tail if the operational caller disabled use_mp_re.
    original=state.replace(re_cloud=None,re_ice=None,re_snow=None)
    def column_stub(fields,*a,**kw):
        assert kw['radii'] is False
        out={n:fields[n] for n in driver.WRF_TO_AN}
        out['th']=fields['th']
        zero=jnp.zeros(fields['th'].shape[:-1])
        return out,dict(rain_liquid=zero,snowncv=zero,grplncv=zero,hailncv=zero),{}
    import gpuwrf.physics.nssl2mom.adapter as module
    monkeypatch.setattr(module,'nssl2mom_column',column_stub)
    after=module.nssl2mom_adapter(original,60.)
    assert after.re_cloud is after.re_ice is after.re_snow is None
