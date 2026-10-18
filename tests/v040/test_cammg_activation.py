"""CAM-MG Registry selection at real scan entry and radiation handoff."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.cammg_fields import STATE_FIELDS
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes, conditional_state_leaves_for_mp
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.runtime.operational_mode import _operational_scan_state


def _state(mp=8):
    grid = SimpleNamespace(nz=4, ny=2, nx=3)
    state = State(**{name: jnp.zeros(shape, DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid, mp_physics=mp).items()})
    return state.replace(theta=jnp.full_like(state.theta, 290.),
                         p=jnp.full_like(state.p, 80000.),
                         qv=jnp.full_like(state.qv, .01),
                         qc=jnp.full_like(state.qc, .001),
                         qi=jnp.full_like(state.qi, 2.e-5),
                         qs=jnp.full_like(state.qs, 3.e-5),
                         t_skin=jnp.full_like(state.t_skin, 285.),
                         xland=jnp.ones_like(state.xland))


def _scan(state, mp):
    return _operational_scan_state(state, SimpleNamespace(
        mp_physics=mp, use_mp_re=1, ra_lw_physics=4, ra_sw_physics=4))


def _byte_equal(left, right):
    a, ta = jax.tree.flatten(left)
    b, tb = jax.tree.flatten(right)
    assert ta == tb
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert (x.shape, x.dtype, x.tobytes()) == (y.shape, y.dtype, y.tobytes())


@pytest.mark.parametrize('mp', (0, 6, 8, 9, 18, 24, 28, 40, 50, 55, 56))
def test_scan_does_not_allocate_inactive_cammg(mp):
    state = _scan(_state(mp), mp)
    assert all(getattr(state, name) is None for name in STATE_FIELDS)
    assert not set(STATE_FIELDS) & set(conditional_state_leaves_for_mp(mp, include_all_conditional=True))
    assert not set(STATE_FIELDS) & _state_field_shapes(
        SimpleNamespace(nz=4, ny=2, nx=3), mp_physics=mp, include_all_conditional=True).keys()
    # Existing concrete scan-tail layout stays allocated (the historical perf contract).
    assert state.qh is not None and state.nwfa is not None and state.qir is not None


def test_active_cammg_scan_preserves_every_held_leaf():
    state = _state(11)
    state = state.replace(**{name: jnp.full_like(getattr(state, name), i)
                             for i, name in enumerate(STATE_FIELDS)})
    result = _scan(state, 11)
    for name in STATE_FIELDS:
        _byte_equal(getattr(result, name), getattr(state, name))
    assert set(STATE_FIELDS) <= set(conditional_state_leaves_for_mp(11, include_all_conditional=True))


@pytest.mark.parametrize('mp', (6, 8, 24, 28, 40, 50, 55))
def test_radiation_ignores_other_scheme_cammg_fields(monkeypatch, mp):
    monkeypatch.setattr(R, '_MP_RE', True)
    state = _state(mp).ensure_conditional_leaves(mp_physics=mp)
    contaminated = state.replace(lradius=jnp.full_like(state.qc, 37.),
                                 iradius=jnp.full_like(state.qc, 83.),
                                 cldfra_mp_all=jnp.full_like(state.qc, .73))
    kwargs = dict(time_utc='2026-02-28_12:00:00', lead_seconds=18.,
                  mp_physics=mp, cloud_mp_physics=mp, use_mynn_sgs_cloud=False)
    clean = C._rrtmg_column_inputs(state, None, **kwargs)
    got = C._rrtmg_column_inputs(contaminated, None, **kwargs)
    _byte_equal(got, clean)
    # WRF module_physics_init.F:985-1025: use_mp_re=1 + RRTMG sets has_reqc/i/s for Thompson, WSM3/5/6/7,
    # WDM5/6/7, P3 and Jensen (t-wsmdz b8ad6b92a4 wired WSM5/WSM6), never for Morrison-aero (40).
    if mp in (6, 8, 24, 28, 50, 55):
        # A present unrelated radius must not displace this scheme's held producer.
        np.testing.assert_array_equal(got[0].re_cloud, C._to_columns(state.re_cloud))
    else:
        assert got[0].re_cloud is None and got[1].re_cloud is None


def test_cammg_radiation_uses_registry_cloud_and_micron_radii():
    state = _scan(_state(11), 11).replace(
        lradius=jnp.full((4, 2, 3), 17., jnp.float32),
        iradius=jnp.full((4, 2, 3), 83., jnp.float32),
        cldfra_mp_all=jnp.full((4, 2, 3), .73, jnp.float32))
    sw, lw, *_ = C._rrtmg_column_inputs(state, None,
        time_utc='2026-02-28_12:00:00', lead_seconds=18., mp_physics=11,
        use_mp_re=0, use_mynn_sgs_cloud=False)
    for column in (sw, lw):
        np.testing.assert_array_equal(column.cloud_fraction, np.full((2, 3, 4), .73, np.float32))
        np.testing.assert_array_equal(column.re_cloud, np.full((2, 3, 4), 17., np.float32))
        np.testing.assert_array_equal(column.re_ice, np.full((2, 3, 4), 83., np.float32))
        np.testing.assert_array_equal(column.qi, C._to_columns(state.qi) + C._to_columns(state.qs))
        np.testing.assert_array_equal(column.qs, np.zeros((2, 3, 4), np.float32))
        assert column.cammg_radii


@pytest.mark.parametrize('missing', ('lradius', 'iradius'))
def test_active_cammg_radiation_refuses_missing_producer(missing):
    state = _state(11).replace(**{missing: None})
    with pytest.raises(ValueError, match='CAM-MG radiation requires held'):
        C._rrtmg_column_inputs(state, None, mp_physics=11)
