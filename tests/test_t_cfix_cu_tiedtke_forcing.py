"""t-cfix-cu F14: Tiedtke (cu=6) / New-Tiedtke (cu=16) closure forcing as WRF's caller builds it.

cumulus_driver (module_cumulus_driver.F:866-883, :1389-1442): Tiedtke QVFTEN = the held
RQVFTEN (previous step's last-RK-stage qv advection, decoupled, phy_prep_part2) and
QVPBLTEN = RQVBLTEN; New-Tiedtke THFTEN = (RTHFTEN + RTHRATEN + RTHBLTEN)*pi and
QVFTEN = RQVFTEN + RQVBLTEN.  Both schemes zero their forcing at itimestep 1
(module_cu_tiedtke.F:364-370, module_cu_ntiedtke.F:451-458).  The port recomputed an
RK1 unlimited qv flux on the current state and used theta_m(after)-theta_m(before)/dt.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _case(cu):
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("rk_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=6, nx=7, nz=8)
    state = f._state(grid)
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, mp_physics=8, cu_physics=cu, bl_pbl_physics=0,
                              sf_sfclay_physics=0, rad_rk_tendf=1, use_flux_advection=True)
    th = jnp.full_like(state.theta, 3.0e-5)
    qv = jnp.full_like(state.theta, 2.0e-8)
    ra = jnp.full_like(state.theta, -1.2e-5)
    carry = initial_operational_carry(state).replace(cu_forcing=(th, qv), rthraten=ra)
    return state, carry, nml, (th, qv, ra)


def _capture(monkeypatch, name):
    from gpuwrf.runtime import operational_mode as om

    seen = {}

    class Stop(Exception):
        pass

    def spy(*a, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(om, name, spy)
    return seen, Stop


@pytest.mark.parametrize("step", (1, 2))
def test_tiedtke_gets_held_rqvften_and_step1_zero(step, monkeypatch):
    from gpuwrf.runtime import operational_mode as om

    state, carry, nml, (_th, qv, _ra) = _case(6)
    seen, Stop = _capture(monkeypatch, "tiedtke_adapter")
    with pytest.raises(Stop):
        om._physics_step_forcing(carry, nml, jnp.asarray((step - 1) * float(nml.dt_s)), run_radiation=False)
    expect = np.zeros(state.theta.shape) if step == 1 else np.asarray(qv, np.float64)
    np.testing.assert_allclose(np.asarray(seen["qvften"]), expect, rtol=1e-7, atol=0)
    if step == 1:
        assert not np.any(np.asarray(seen["qvpblten"]))


@pytest.mark.parametrize("step", (1, 2))
def test_ntiedtke_gets_wrf_thften_and_qvften(step, monkeypatch):
    from gpuwrf.coupling.scan_adapters import _exner_columns
    from gpuwrf.runtime import operational_mode as om

    state, carry, nml, (th, qv, ra) = _case(16)
    seen, Stop = _capture(monkeypatch, "ntiedtke_adapter")
    with pytest.raises(Stop):
        om._physics_step_forcing(carry, nml, jnp.asarray((step - 1) * float(nml.dt_s)), run_radiation=False)
    if step == 1:
        assert not np.any(np.asarray(seen["thften"])) and not np.any(np.asarray(seen["qvften"]))
        return
    pi = np.asarray(_exner_columns(state.p), np.float64)
    # no PBL here: RTHBLTEN = RQVBLTEN = 0
    np.testing.assert_allclose(np.asarray(seen["thften"]),
                               (np.asarray(th, np.float64) + np.asarray(ra, np.float64)) * pi, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(seen["qvften"]), np.asarray(qv, np.float64), rtol=1e-7)


@pytest.mark.parametrize("cu", (2, 3, 6, 16))
def test_cudt_cadence_fails_closed(cu):
    from gpuwrf.runtime.operational_mode import UnsupportedSchemeSelection, _resolve_operational_suite

    _state, _carry, nml, _f = _case(cu)
    with pytest.raises(UnsupportedSchemeSelection, match="cudt>0"):
        _resolve_operational_suite(dataclasses.replace(nml, cumulus_cadence_steps=5, cudt_minutes=5.0))


def test_capture_multiplies_by_msfty():
    from gpuwrf.coupling.cu_forcing import capture_coupled

    th = jnp.full((3, 2, 4), 2.0, jnp.float32)
    qv = jnp.full((3, 2, 4), 3.0, jnp.float32)
    msf = jnp.asarray(np.linspace(0.98, 1.03, 8).reshape(2, 4))
    a, b = capture_coupled(th, qv, msf, jnp.float32)
    m = np.broadcast_to(np.asarray(msf, np.float32)[None], (3, 2, 4))
    np.testing.assert_allclose(np.asarray(a), np.float32(2.0) * m, rtol=0)
    np.testing.assert_allclose(np.asarray(b), np.float32(3.0) * m, rtol=0)


def test_tiedtke_gets_wrf_znu_half_levels(monkeypatch):
    """cumulus_driver.F:1519 ZNU = grid%znu = (znw(k)+znw(k+1))/2, not a pressure proxy."""
    from gpuwrf.coupling import scan_adapters as A

    state, _carry, nml, _f = _case(6)
    seen = []
    real = A.tiedtke_column_jax

    def spy(*args, **kw):
        jax.debug.callback(lambda z: seen.append(np.asarray(z)), args[16])  # traced under vmap
        return real(*args, **kw)

    monkeypatch.setattr(A, "tiedtke_column_jax", spy)
    jax.clear_caches()
    A.tiedtke_adapter(state, float(nml.dt_s), nml.grid)
    eta = np.asarray(nml.grid.eta_levels, np.float64)
    znu = 0.5 * (eta[:-1] + eta[1:])
    assert seen
    got = np.asarray(seen[0]).reshape(-1, znu.size)
    np.testing.assert_array_equal(got, np.broadcast_to(znu, got.shape))


@pytest.mark.parametrize("cu", (6, 16))
def test_rthften_capture_only_for_schemes_that_read_it(cu, monkeypatch):
    """module_em.F:654-665 sets RTHFTEN (theta advection) only for GD/GF/G3/NTiedtke; Tiedtke (6)
    reads RQVFTEN only (:1365-1376), so its actual RK step must not trace the theta capture."""
    from gpuwrf.runtime import operational_mode as om

    _state, carry, nml, _forcing = _case(cu)
    nml = dataclasses.replace(nml, run_boundary=False)
    calls = []
    real = om._cu_forcing_theta_advection

    def spy(*a, **kw):
        calls.append(True)
        return real(*a, **kw)

    monkeypatch.setattr(om, "_cu_forcing_theta_advection", spy)
    jax.eval_shape(lambda c: om._physics_boundary_step(c, nml, jnp.int32(2), run_radiation=False), carry)
    assert (len(calls) > 0) == (cu == 16), (cu, len(calls))


def test_tiedtke_finish_decouples_rqvften_only():
    """phy_prep_part2 (module_big_step_utilities_em.F:5291-5331): Tiedtke decouples RQVFTEN; RTHFTEN
    is neither decoupled nor dry-converted (the dry conversion would make it -Rv/Rd*th_phy^2/t*RQVFTEN)."""
    from types import SimpleNamespace

    from gpuwrf.coupling.cu_forcing import finish_cu_forcing

    nz, ny, nx = 3, 2, 2
    metrics = SimpleNamespace(c1h=jnp.ones((nz,)), c2h=jnp.zeros((nz,)))
    held = jnp.zeros((nz, ny, nx), jnp.float32)
    coupled_qv = jnp.full((nz, ny, nx), 4.0e-3, jnp.float32)
    theta_m, qv, muts = (jnp.full((nz, ny, nx), 300.0, jnp.float32), jnp.full((nz, ny, nx), 1.0e-2, jnp.float32),
                         jnp.full((ny, nx), 8.0e4, jnp.float32))
    rth, rqv = finish_cu_forcing((held, coupled_qv), theta_m, qv, muts, metrics, rthften=False)
    assert rth is held
    np.testing.assert_allclose(np.asarray(rqv), 4.0e-3 / 8.0e4, rtol=1e-6)
    rth16, _ = finish_cu_forcing((held, coupled_qv), theta_m, qv, muts, metrics)
    assert float(np.max(np.abs(np.asarray(rth16)))) > 0.0  # the G3/NTiedtke conversion does move RTHFTEN


def test_tiedtke_step_keeps_held_rthften_through_phy_prep_part2(monkeypatch):
    """Caller wiring: a cu=6 step's phy_prep_part2 leaves the RTHFTEN leaf as it was (WRF never sets,
    decouples or dry-converts it for Tiedtke) and decouples the captured RQVFTEN by the final mass."""
    from gpuwrf.runtime import operational_mode as om

    _state, carry, nml, _forcing = _case(6)
    sentinel = jnp.full_like(carry.state.theta, 7.0e-5)
    c_qv = jnp.full_like(carry.state.theta, 1.0e-3)
    final_mu = carry.state.mu_total + 150.0

    def rk(c, nml_, **kw):
        return c.replace(state=c.state.replace(mu_total=final_mu), cu_forcing=(sentinel, c_qv))

    monkeypatch.setattr(om, "_rk_scan_step", rk)
    got = om._physics_boundary_step(carry, nml, jnp.asarray(2), run_radiation=False)
    np.testing.assert_array_equal(np.asarray(got.cu_forcing[0]), np.asarray(sentinel))
    mass = np.asarray(nml.metrics.c1h)[:, None, None] * np.asarray(final_mu)[None] + np.asarray(nml.metrics.c2h)[:, None, None]
    np.testing.assert_allclose(np.asarray(got.cu_forcing[1]), 1.0e-3 / mass, rtol=1e-6)
