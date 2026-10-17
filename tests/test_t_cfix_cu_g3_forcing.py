"""t-cfix-cu F7: G3/GD closure forcing = WRF's held RTHFTEN/RQVFTEN + RTHRATEN + PBL rates.

WRF (module_em.F:654-665, :1365-1376) stores the last RK stage's theta / qv
advection times msfty in RTHFTEN/RQVFTEN; phy_prep_part2 (solve_em.F:3669,
module_big_step_utilities_em.F:5293-5331) decouples both by the final mass and
makes RTHFTEN dry; the next step's cumulus_driver (:866-883) feeds
``(RTHFTEN+RTHRATEN+RTHBLTEN)*pi`` and ``RQVFTEN+RQVBLTEN`` to G3/GD.  The port
fed ``(state - physics_entry)/dt`` (~0 in source-leaf mode) and RTHRATEN = 0.

Gates (CPU): (1) REAL4 literal of the WRF phy_prep_part2 statements (source text
guarded against the pristine checkout) vs ``finish_cu_forcing``; (2) the RK stage
captures the final stage's theta/qv advection x msfty exactly once; (3) the
operational slot hands the held pair + held RTHRATEN to the WRF kernel;
(4) a missing carry is refused (E78).  The end-to-end pristine comparison on real
c05 lives in proofs/t_cfix_cu/f7_c05_oracle.json (pristine serial WRF history).
"""
from __future__ import annotations

import ctypes
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"

LITERAL = """subroutine finish(n,nz,c1,c2,muts,t2,qv,rth,rqv) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: n,nz
real(c_float) :: c1(nz),c2(nz),muts(n),t2(n,nz),qv(n,nz),rth(n,nz),rqv(n,nz)
real(c_float), parameter :: r_d=287., r_v=461.6, t0=300.
real(c_float) :: th_phy
integer :: i,k
do k=1,nz
do i=1,n
 rqv(i,k)=rqv(i,k)/(c1(k)*muts(i)+c2(k))
enddo
enddo
do k=1,nz
do i=1,n
 rth(i,k)=rth(i,k)/(c1(k)*muts(i)+c2(k))
enddo
enddo
do k=1,nz
do i=1,n
 th_phy = (t2(i,k) + t0) / (1. + (R_v/R_d) * qv(i,k))
 rth(i,k) = th_phy/(t2(i,k)+t0) * (rth(i,k) - (R_v/R_d) * th_phy * rqv(i,k))
enddo
enddo
end subroutine finish
"""


@pytest.fixture(scope="module")
def literal(tmp_path_factory):
    bsu = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text()
    for stmt in ("RQVFTEN(I,K,J)=RQVFTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))",
                 "RTHFTEN(I,K,J)=RTHFTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))",
                 "th_phy(i,k,j) = (t_new(i,k,j) + t0) / (1. + (R_v/R_d) * qv(i,k,j))",
                 "rthften(i,k,j) = th_phy(i,k,j)/(t_new(i,k,j)+t0) *",
                 "(rthften(i,k,j) - (R_v/R_d) * th_phy(i,k,j) * rqvften(i,k,j))"):
        assert stmt in bsu, stmt  # the literal transcribes exactly these pristine statements
    if not Path(FC).exists():
        pytest.skip("gfortran for the REAL4 literal is unavailable")
    d = tmp_path_factory.mktemp("cu_forcing_literal")
    (d / "f.f90").write_text(LITERAL)
    subprocess.run([FC, "-O2", "-shared", "-fPIC", str(d / "f.f90"), "-o", str(d / "f.so")],
                   check=True, capture_output=True, timeout=120)
    fn = ctypes.CDLL(str(d / "f.so")).finish
    fn.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7
    return fn


def _real_case(seed=7, nz=6, ny=5, nx=4):
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    c1 = np.linspace(1.0, 0.1, nz).astype(np.float32)
    c2 = np.linspace(0.0, 4000.0, nz).astype(np.float32)
    muts = (88000.0 + 500.0 * f(ny, nx)).astype(np.float32)
    theta = (290.0 + 15.0 * np.abs(f(nz, ny, nx))).astype(np.float32)
    qv = (0.012 * np.abs(f(nz, ny, nx))).astype(np.float32)
    mass = c1[:, None, None] * muts[None] + c2[:, None, None]
    rth_c = (mass * 3e-4 * f(nz, ny, nx)).astype(np.float32)
    rqv_c = (mass * 4e-7 * f(nz, ny, nx)).astype(np.float32)
    return c1, c2, muts, theta, qv, rth_c, rqv_c


def test_finish_matches_wrf_phy_prep_part2_literal(literal):
    from gpuwrf.coupling.cu_forcing import finish_cu_forcing

    c1, c2, muts, theta, qv, rth_c, rqv_c = _real_case()
    nz, ny, nx = theta.shape
    t2 = theta - np.float32(300.0)  # exact (Sterbenz): WRF holds the perturbation t_2
    assert np.array_equal(t2 + np.float32(300.0), theta)
    cols = lambda a: np.array(a.reshape(nz, ny * nx).T, np.float32, order="F", copy=True)  # noqa: E731
    rth, rqv = cols(rth_c), cols(rqv_c)
    keep = (np.ascontiguousarray(c1), np.ascontiguousarray(c2), np.ascontiguousarray(muts.reshape(-1)),
            cols(t2), cols(qv), rth, rqv)  # alive for the whole foreign call
    literal(ny * nx, nz, *[a.ctypes.data for a in keep])
    back = lambda a: a.T.reshape(nz, ny, nx)  # noqa: E731
    metrics = dataclasses.make_dataclass("M", ["c1h", "c2h"])(c1, c2)
    got = finish_cu_forcing((jnp.asarray(rth_c), jnp.asarray(rqv_c)), jnp.asarray(theta), jnp.asarray(qv),
                            jnp.asarray(muts), metrics)
    for g, w in zip(got, (back(rth), back(rqv))):
        g = np.asarray(g)
        assert g.dtype == np.float32
        ulp = np.abs(g.view(np.int32).astype(np.int64) - w.view(np.int32).astype(np.int64))
        assert ulp.max() <= 1, ulp.max()  # REAL4 literal: byte-exact up to one rounding
    # Deletion sensitivity: skipping the dry conversion is far outside one ulp.
    moist = np.asarray(got[0]) * theta / (theta / (1 + np.float32(461.6 / 287.0) * qv)) \
        + np.float32(461.6 / 287.0) * (theta / (1 + np.float32(461.6 / 287.0) * qv)) * np.asarray(got[1])
    assert np.max(np.abs(moist - back(rth))) > 1e3 * np.max(np.abs(np.spacing(back(rth))))


def test_capture_is_final_stage_theta_and_qv_advection_times_msfty(monkeypatch):
    """The RK capture stores set_tend's value: last-stage advection x msfty, once per step."""
    from gpuwrf.coupling import cu_forcing as CF
    from gpuwrf.runtime import operational_mode as om

    calls = []
    real_theta = om._cu_forcing_theta_advection
    real_capture = CF.capture_coupled

    def theta_spy(haloed, vel, namelist):
        out = real_theta(haloed, vel, namelist)
        calls.append(("theta", out))
        return out

    def capture_spy(theta_adv, qv_adv, msfty, dtype, **kw):
        calls.append(("capture", theta_adv, qv_adv, msfty))
        return real_capture(theta_adv, qv_adv, msfty, dtype, **kw)

    monkeypatch.setattr(om, "_cu_forcing_theta_advection", theta_spy)
    monkeypatch.setattr(CF, "capture_coupled", capture_spy)
    moist = []
    real_moist = om._scalar_transport_coupled_tendencies

    def moist_spy(*a, **kw):
        out = real_moist(*a, **kw)
        if "qv" in kw.get("species", ()):
            moist.append((kw.get("rk_step"), out[tuple(kw["species"]).index("qv")]))
        return out

    monkeypatch.setattr(om, "_scalar_transport_coupled_tendencies", moist_spy)
    carry, nml = _wired_case()
    jax.clear_caches()
    out = jax.eval_shape(lambda c: om._physics_boundary_step(c, nml, jnp.asarray(1, jnp.int32),
                                                             run_radiation=False), carry)
    assert [c[0] for c in calls] == ["theta", "capture"]
    _theta, theta_adv, qv_adv, msfty = calls[1]
    assert theta_adv is calls[0][1]
    final = [o for step, o in moist if step == int(nml.rk_order)]
    assert len(final) == 1 and qv_adv is final[0]
    assert msfty.shape == nml.metrics.msfty.shape  # the step traces the namelist (firewall)
    assert out.cu_forcing is not None and out.cu_forcing[0].shape == carry.state.theta.shape


def _wired_case():
    """Small real-structured operational case (rrtm_lw wiring fixture) with cu=5."""
    import importlib.util

    from gpuwrf.coupling.cu_forcing import initial_cu_forcing
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("rk_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=10, nx=10, nz=8)
    state = f._state(grid)
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, mp_physics=0, cu_physics=5, bl_pbl_physics=0,
                              sf_sfclay_physics=0, rad_rk_tendf=1, use_flux_advection=True)
    carry = initial_operational_carry(state).replace(cu_forcing=initial_cu_forcing(state.theta))
    return carry, nml


def test_operational_slot_feeds_held_forcing_and_radiation(monkeypatch):
    from gpuwrf.physics import _grell_cup_jax as G
    from gpuwrf.runtime import operational_mode as om

    carry, nml = _wired_case()
    th = jnp.full_like(carry.state.theta, 2.5e-5)
    qv = jnp.full_like(carry.state.theta, -3.0e-8)
    ra = jnp.full_like(carry.state.theta, -1.5e-5)
    carry = carry.replace(cu_forcing=(th, qv), rthraten=ra)
    seen = {}

    class Stop(Exception):
        pass

    def spy(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(G, "g3drv_tile", spy)
    with pytest.raises(Stop):
        om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    np.testing.assert_array_equal(seen["rthften"], np.asarray(th, seen["rthften"].dtype))
    np.testing.assert_array_equal(seen["rqvften"], np.asarray(qv, seen["rqvften"].dtype))
    np.testing.assert_array_equal(seen["rthraten"], np.asarray(ra, seen["rthraten"].dtype))


def test_missing_held_forcing_is_refused():
    from gpuwrf.runtime import operational_mode as om

    carry, nml = _wired_case()
    with pytest.raises(ValueError, match="RTHFTEN/RQVFTEN"):
        om._physics_step_forcing(carry.replace(cu_forcing=None), nml, jnp.asarray(0.0), run_radiation=False)


def test_initial_carry_seeds_zero_forcing_only_for_g3_gd():
    from gpuwrf.coupling.cu_forcing import CU_FORCING_SCHEMES, cu_forcing_required

    assert set(CU_FORCING_SCHEMES) >= {5, 93}
    assert not cu_forcing_required(1) and not cu_forcing_required(0)


def test_euler_pbl_increment_reaches_g3_as_rate_not_as_state(monkeypatch):
    """A PBL still applied as an Euler increment: WRF G3 reads the time-n state and RQVBLTEN."""
    from gpuwrf.physics import _grell_cup_jax as G
    from gpuwrf.runtime import operational_mode as om

    carry, nml = _wired_case()
    nml = dataclasses.replace(nml, bl_pbl_physics=3)
    entry_qv = np.asarray(carry.state.qv)

    def euler_pbl(st, dt, grid, **kw):
        return st.replace(qv=st.qv + 1.0e-3, theta=st.theta + 0.5)

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 3, euler_pbl)
    seen = {}

    class Stop(Exception):
        pass

    def spy(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(G, "g3drv_tile", spy)
    with pytest.raises(Stop):
        om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    np.testing.assert_allclose(np.asarray(seen["q"]), entry_qv, rtol=0, atol=0)  # time-n vapour
    np.testing.assert_allclose(np.asarray(seen["rqvblten"]), 1.0e-3 / float(nml.dt_s), rtol=1e-6)
    assert float(np.min(np.asarray(seen["rthblten"]))) > 0.0  # the PBL heating arrives as RTHBLTEN


def test_step_finishes_captured_forcing_at_phy_prep_part2(monkeypatch):
    """After the RK the captured pair is decoupled by the FINAL mass and made dry, and held."""
    from gpuwrf.coupling import cu_forcing as CF
    from gpuwrf.runtime import operational_mode as om

    carry, nml = _wired_case()
    c_th = jnp.full_like(carry.state.theta, 2.0)
    c_qv = jnp.full_like(carry.state.theta, 1.0e-3)
    final_mu = carry.state.mu_total + 150.0

    def rk(c, nml_, **kw):
        return c.replace(state=c.state.replace(mu_total=final_mu), cu_forcing=(c_th, c_qv))

    calls = []
    real = CF.finish_cu_forcing

    def spy(coupled, theta_m, qv, muts, metrics, **kw):
        assert kw == {"rthften": True}  # G3 (cu=5): RTHFTEN decoupled + dry-converted
        out = real(coupled, theta_m, qv, muts, metrics, **kw)
        calls.append((coupled, muts, out))
        return out

    monkeypatch.setattr(om, "_rk_scan_step", rk)
    monkeypatch.setattr(CF, "finish_cu_forcing", spy)
    got = om._physics_boundary_step(carry, nml, jnp.asarray(1), run_radiation=False)
    assert len(calls) == 1
    coupled, muts, out = calls[0]
    assert coupled[0] is c_th and coupled[1] is c_qv
    np.testing.assert_array_equal(np.asarray(muts), np.asarray(final_mu))
    for g, w in zip(got.cu_forcing, out):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("cu", [5, 3, 16])
def test_legacy_radiation_reaches_cumulus_this_step_from_entry_state(monkeypatch, cu):
    """WRF first_rk_step_part1 runs radiation_driver (:264) before cumulus_driver, whose
    G3/GD/GF/NTiedtke closures read RTHRATEN (:866-883): on a radiation step the closure
    sees THIS step's rate (legacy SW/LW solved once, on the step-entry state) — not the
    previous radt's held rate (the late radiation slot)."""
    from gpuwrf.coupling import scan_adapters
    from gpuwrf.coupling.scan_adapters import _exner_columns
    from gpuwrf.physics import _grell_cup_jax as G
    from gpuwrf.runtime import operational_mode as om

    carry, nml = _wired_case()
    nml = dataclasses.replace(nml, cu_physics=cu, ra_sw_physics=1, ra_lw_physics=1)
    carry = carry.replace(rthraten=jnp.full_like(carry.state.theta, -2.0e-5),
                          state=carry.state.replace(pblh=jnp.full_like(carry.state.pblh, 900.0)))
    entry = carry.state
    assert carry.radiation_diagnostics is None
    log = {"calls": [], "rad_inputs": []}

    def radiation(st, *args, **kwargs):
        log["calls"].append("radiation")
        log["rad_inputs"].append(st)
        return jnp.full_like(st.theta, -1.0e-4)

    monkeypatch.setattr(om, "rrtm_lw_theta_tendency", radiation)
    monkeypatch.setattr(om, "dudhia_sw_theta_tendency", radiation)

    class Stop(Exception):
        pass

    def stop(value, st=None):
        log["calls"].append("cumulus")
        log["rthraten"] = np.asarray(value, np.float64)
        log["cu_p"] = None if st is None else np.asarray(st.p, np.float64)
        raise Stop

    if cu == 5:
        monkeypatch.setattr(G, "g3drv_tile", lambda **kw: stop(kw["rthraten"]))
    elif cu == 3:
        monkeypatch.setattr(scan_adapters, "gfdrv_batched", lambda *a, **kw: stop(np.asarray(kw["rthraten1"])[:, 1:]))
    else:
        monkeypatch.setattr(om, "ntiedtke_adapter", lambda st, dt, grid, **kw: stop(
            np.asarray(kw["thften"]) / np.asarray(_exner_columns(st.p)), st))
    with pytest.raises(Stop):  # lead 3 dt: a radiation step that is not itimestep 1 (NTiedtke zeroing)
        om._physics_step_forcing(carry, nml, jnp.asarray(3.0 * float(nml.dt_s)), run_radiation=True)
    assert log["calls"] == ["radiation", "radiation", "cumulus"], log["calls"]
    np.testing.assert_allclose(log["rthraten"], -2.0e-4, rtol=1e-6, atol=0)  # SW + LW of this step
    for st in log["rad_inputs"]:
        for name in ("theta", "qv", "p", "t_skin"):
            np.testing.assert_array_equal(np.asarray(getattr(st, name)), np.asarray(getattr(entry, name)),
                                          err_msg=name)
