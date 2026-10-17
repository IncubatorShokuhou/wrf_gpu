"""t-cfix-cu F9: cumulus R*CUTEN are RK sources (WRF), not a step-entry Euler increment.

WRF never adds cumulus tendencies to the state before the dycore:
calculate_phy_tend (module_em.F:2249-2300) couples every R*CUTEN with the time-n
mass, phy_cu_ten (module_physics_addtendc.F:1125-1618) adds the scheme's field set to
t/moist/ru/rv_tendf (add_a2a / add_a2c), the RK stages integrate them, advance_ppt
accumulates RAINC (+ the KFSCHEME NCA clear) and phy_prep_part2 decouples the held
rates by the final mass.  The port Euler-added BMJ/GF/G3/GD/Tiedtke/NTiedtke and the
held SAS (4/14/94/95/96) and old-KF (99) rates before the dycore.

Gates (CPU): REAL4 literal of the WRF couple / clear / decouple statements (pristine
source text guarded) vs coupling.cu_rk; per scheme, the operational slot in
source-leaf mode leaves the physics state untouched, hands phy_cu_ten's exact field
set to the RK seam as add_a2a/add_a2c of mass x rate, and (held schemes) decouples
by the final mass after the RK step.
"""
from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"

LITERAL = """subroutine lifetime(n,nz,dt,c1,c2,mut,muts,r,nca,coupled,held,nout) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: n,nz
real(c_float),value :: dt
real(c_float) :: c1(nz),c2(nz),mut(n),muts(n),r(n,nz),nca(n),coupled(n,nz),held(n,nz),nout(n)
integer :: i,k
do k=1,nz
do i=1,n
 coupled(i,k)=(c1(k)*mut(i)+c2(k))*r(i,k)
enddo
enddo
held=coupled
do i=1,n
 nout(i)=nca(i)
 IF ( NCA(I) .GT. 0 ) THEN
  IF ( NINT(NCA(I) / DT) .le. 0 ) held(i,:)=0.
  nout(i)=NCA(I)-DT
 ENDIF
enddo
do k=1,nz
do i=1,n
 held(i,k)=held(i,k)/(c1(k)*muts(i)+c2(k))
enddo
enddo
end subroutine lifetime
"""


@pytest.fixture(scope="module")
def literal(tmp_path_factory):
    em = (WRF / "dyn_em/module_em.F").read_text()
    add = (WRF / "phys/module_physics_addtendc.F").read_text()
    bsu = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text()
    assert "RTHCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHCUTEN(I,K,J)" in em
    assert "RQCCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCCUTEN(I,K,J)" in em
    assert "RUCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUCUTEN(I,K,J)" in em
    assert "IF ( NINT(NCA(I,J) / DT) .le. 0 ) THEN" in add and "NCA(I,J)=NCA(I,J)-DT" in add
    assert "RAINC(I,J)  = RAINC(I,J) + PRATEC(I,J)*DT" in add
    assert "RTHCUTEN(I,K,J)=RTHCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))" in bsu
    if not Path(FC).exists():
        pytest.skip("gfortran for the REAL4 literal is unavailable")
    d = tmp_path_factory.mktemp("cu_rk_literal")
    (d / "f.f90").write_text(LITERAL)
    subprocess.run([FC, "-O2", "-shared", "-fPIC", str(d / "f.f90"), "-o", str(d / "f.so")],
                   check=True, capture_output=True, timeout=120)
    fn = ctypes.CDLL(str(d / "f.so")).lifetime
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 9
    return fn


def _ulp(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return int(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max())


def test_held_couple_clear_decouple_match_wrf_literal(literal):
    from gpuwrf.coupling.cu_rk import couple_held, decouple_held

    rng = np.random.default_rng(3)
    nz, ny, nx, dt = 5, 4, 3, np.float32(54.0)
    c1 = np.linspace(1.0, 0.2, nz).astype(np.float32)
    c2 = np.linspace(0.0, 3000.0, nz).astype(np.float32)
    mut = (90000.0 + 300 * rng.standard_normal((ny, nx))).astype(np.float32)
    muts = (mut + 40 * rng.standard_normal((ny, nx))).astype(np.float32)
    nca = np.where(rng.random((ny, nx)) < 0.5, np.float32(20.0), np.float32(162.0)).astype(np.float32)
    rates = tuple((1e-4 * rng.standard_normal((nz, ny, nx))).astype(np.float32) for _ in range(6)) + (
        np.abs(rng.standard_normal((ny, nx))).astype(np.float32) * np.float32(1e-3),)
    metrics = dataclasses.make_dataclass("M", ["c1h", "c2h"])(c1, c2)
    coupled = couple_held(tuple(jnp.asarray(r) for r in rates), jnp.asarray(mut), metrics)
    clear = (nca > 0) & (np.floor(nca / dt + np.float32(0.5)) <= 0)
    held = tuple(c if i == 6 else jnp.where(jnp.asarray(clear)[None], 0.0, c) for i, c in enumerate(coupled))
    back = decouple_held(held, jnp.asarray(muts), metrics)
    cols = lambda a: np.array(a.reshape(nz, ny * nx).T, np.float32, order="F", copy=True)  # noqa: E731
    for i in range(6):
        r = cols(rates[i])
        out_c, out_h = np.zeros_like(r, order="F"), np.zeros_like(r, order="F")
        nout = np.zeros(ny * nx, np.float32)
        keep = (c1, c2, mut.reshape(-1).copy(), muts.reshape(-1).copy(), r, nca.reshape(-1).copy(), out_c, out_h, nout)
        literal(ny * nx, nz, float(dt), *[a.ctypes.data for a in keep])
        assert _ulp(np.asarray(coupled[i]), out_c.T.reshape(nz, ny, nx)) == 0
        assert _ulp(np.asarray(back[i]), out_h.T.reshape(nz, ny, nx)) <= 1
    assert np.asarray(back[6]) is not None and np.array_equal(np.asarray(back[6]), rates[6])  # PRATEC never coupled
    from gpuwrf.coupling.kf99_adapter import kf99_advance_ppt
    got_clear, got_nca = kf99_advance_ppt(jnp.asarray(nca), float(dt))
    assert np.array_equal(np.asarray(got_clear), clear) and clear.any() and (~clear).any()
    np.testing.assert_array_equal(np.asarray(got_nca), nout.reshape(ny, nx))


def _fixture(cu, **over):
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("rk_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=6, nx=7, nz=8)
    state = f._state(grid)
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, mp_physics=8, cu_physics=cu, bl_pbl_physics=0,
                              sf_sfclay_physics=0, rad_rk_tendf=1, use_flux_advection=True)
    nml = dataclasses.replace(nml, **over)
    return grid, state, initial_operational_carry(state), nml


def _rates(state, names, scale=1.0):
    shape = state.theta.shape
    rng = np.random.default_rng(11)
    out = {n: jnp.asarray((scale * 1e-4 * rng.standard_normal(shape)).astype(np.float32)) for n in names}
    return out


STATELESS = {  # cu -> (phy_cu_ten field set, how the operational slot finds the adapter)
    2: (("theta", "qv"), "bmj_adapter"),
    3: (("theta", "qv", "qc", "qi"), 3),
    5: (("theta", "qv", "qc", "qi"), 5),
    93: (("theta", "qv", "qc", "qi"), 93),
    6: (("theta", "qv", "qc", "qi", "u", "v"), "tiedtke_adapter"),
    16: (("theta", "qv", "qc", "qi", "u", "v"), "ntiedtke_adapter"),
}


@pytest.mark.parametrize("cu", sorted(STATELESS))
def test_stateless_cumulus_feeds_rk_sources_not_state(cu, monkeypatch):
    from gpuwrf.coupling import cu_forcing as CF
    from gpuwrf.coupling.cuplus import momentum_sources
    from gpuwrf.coupling.kf_rk import kf_add_a2a
    from gpuwrf.runtime import operational_mode as om

    names, where = STATELESS[cu]
    grid, state, carry, nml = _fixture(cu)
    if CF.cu_forcing_required(cu):
        carry = carry.replace(cu_forcing=CF.initial_cu_forcing(state.theta))
    rates = _rates(state, names)
    rain = jnp.full(state.t_skin.shape, 0.25, state.rainc_acc.dtype)
    seen = {}

    def fake(st, *a, return_rates=False, **kw):
        seen["return_rates"] = return_rates
        assert return_rates, "source-leaf mode must request the rates, not an Euler-applied state"
        payload = (dict(rates), rain)
        return (payload, jnp.zeros(st.theta.shape[1:])) if cu == 2 else payload

    if isinstance(where, str):
        monkeypatch.setattr(om, where, fake)
    else:
        monkeypatch.setitem(om.CU_STATELESS_SCAN_ADAPTERS, where, fake)
    captured = {}
    real = om._source_leaf_dry_tendencies

    def spy(*a, **kw):
        captured.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(om, "_source_leaf_dry_tendencies", spy)
    out = om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    for name in ("theta", "qv", "qc", "qi", "u", "v"):  # no Euler add before the dycore
        np.testing.assert_array_equal(getattr(out.state, name), getattr(state, name), err_msg=name)
    np.testing.assert_allclose(out.state.rainc_acc, np.asarray(state.rainc_acc) + 0.25, rtol=1e-6)
    mass = (grid.metrics.c1h[:, None, None] * state.mu_total[None] + grid.metrics.c2h[:, None, None])
    px, spec, nest = om._acoustic_lateral_bc_flags(nml)
    tendf = captured["kf_tendf"]
    for slot, name in enumerate(("theta", "qv", "qc", "qr", "qi", "qs")):
        if name in names:
            expect = kf_add_a2a(jnp.asarray(mass, jnp.float32) * rates[name], specified_or_nested=spec or nest,
                                periodic_x=px)
            np.testing.assert_allclose(np.asarray(tendf[slot]), np.asarray(expect), rtol=2e-7, atol=0)
        else:
            assert tendf[slot] is None, (cu, name)  # not in this scheme's phy_cu_ten case
    if "u" in names:
        ru, rv = momentum_sources(jnp.asarray(mass, jnp.float32) * rates["u"],
                                  jnp.asarray(mass, jnp.float32) * rates["v"],
                                  specified_or_nested=spec or nest, periodic_x=px, periodic_y=px)
        np.testing.assert_allclose(np.asarray(captured["cuplus_momentum"][0]), np.asarray(ru), rtol=2e-7)
        np.testing.assert_allclose(np.asarray(captured["cuplus_momentum"][1]), np.asarray(rv), rtol=2e-7)
    else:
        assert captured.get("cuplus_momentum") is None
    moist = out.dry_tendencies.moist_tendf
    for name in ("qc", "qi"):
        assert (name in moist) == (name in names), (cu, name)


HELD = {  # cu -> (module, cadence function, layout length)
    4: ("gpuwrf.coupling.scalesas_adapter", "scalesas_cadence_step", 7),
    94: ("gpuwrf.coupling.sas94_adapter", "sas94_cadence_step", 7),
    95: ("gpuwrf.coupling.osas95_adapter", "osas95_cadence_step", 7),
    14: ("gpuwrf.coupling.native_sas_adapter", "native_sas_cadence_step", 9),
    96: ("gpuwrf.coupling.native_sas_adapter", "native_sas_cadence_step", 9),
}


@pytest.mark.parametrize("cu", sorted(HELD))
def test_held_sas_rates_couple_feed_rk_then_decouple(cu, monkeypatch):
    import importlib

    from gpuwrf.coupling.cu_rk import HELD_APPLIED
    from gpuwrf.runtime import operational_mode as om

    module, fn, n = HELD[cu]
    grid, state, carry, nml = _fixture(cu)
    rng = np.random.default_rng(5)
    rates = tuple(jnp.asarray((1e-4 * rng.standard_normal(state.theta.shape)).astype(np.float32))
                  for _ in range(6)) + (jnp.full(state.t_skin.shape, 2e-3, jnp.float32),)
    rates = rates + tuple(jnp.asarray((1e-4 * rng.standard_normal(state.theta.shape)).astype(np.float32))
                          for _ in range(n - 7))
    carry = carry.replace(cumulus_tendencies=tuple(jnp.zeros_like(r) for r in rates),
                          osas_rng_key=jnp.zeros((2,), jnp.uint32))
    monkeypatch.setattr(importlib.import_module(module), fn, lambda entry, st, held, *a, **kw: (st, rates))
    captured = {}
    real = om._source_leaf_dry_tendencies

    def spy(*a, **kw):
        captured.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(om, "_source_leaf_dry_tendencies", spy)
    out = om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    for name in ("theta", "qv", "qc", "qi", "u", "v"):
        np.testing.assert_array_equal(getattr(out.state, name), getattr(state, name), err_msg=name)
    mass = np.asarray(grid.metrics.c1h[:, None, None] * state.mu_total[None] + grid.metrics.c2h[:, None, None],
                      np.float32)
    held = out.carry.cumulus_tendencies
    for i, r in enumerate(rates):  # calculate_phy_tend coupled the whole held set in the carry
        expect = np.asarray(r) if i == 6 else mass * np.asarray(r)
        np.testing.assert_allclose(np.asarray(held[i]), expect, rtol=2e-7)
    tendf = captured["kf_tendf"]
    applied = HELD_APPLIED[cu]
    for slot, name in enumerate(("theta", "qv", "qc", "qr", "qi", "qs")):
        assert (tendf[slot] is not None) == (name in applied), (cu, name)
    assert (captured.get("cuplus_momentum") is not None) == ("u" in applied)
    np.testing.assert_allclose(out.state.rainc_acc, np.asarray(state.rainc_acc) + float(nml.dt_s) * 2e-3, rtol=1e-6)


def test_held_rates_decoupled_by_final_mass_after_the_rk_step(monkeypatch):
    """phy_prep_part2 after the RK step: the coupled held set returns to the carry / final mass."""
    import gpuwrf.coupling.scalesas_adapter as S
    from gpuwrf.runtime import operational_mode as om

    grid, state, carry, nml = _fixture(4)
    rates = tuple(jnp.full(state.theta.shape, 1e-5 * (i + 1), jnp.float32) for i in range(6)) + (
        jnp.full(state.t_skin.shape, 2e-3, jnp.float32),)
    carry = carry.replace(cumulus_tendencies=tuple(jnp.zeros_like(r) for r in rates))
    monkeypatch.setattr(S, "scalesas_cadence_step", lambda entry, st, held, *a, **kw: (st, rates))
    final_mu = state.mu_total + 120.0
    monkeypatch.setattr(om, "_rk_scan_step", lambda c, nml, **kw: c.replace(state=c.state.replace(mu_total=final_mu)))
    got = om._physics_boundary_step(carry, nml, jnp.asarray(1), run_radiation=False)
    m0 = grid.metrics.c1h[:, None, None] * state.mu_total[None] + grid.metrics.c2h[:, None, None]
    mf = grid.metrics.c1h[:, None, None] * final_mu[None] + grid.metrics.c2h[:, None, None]
    for i, r in enumerate(rates[:6]):
        np.testing.assert_allclose(np.asarray(got.cumulus_tendencies[i]),
                                   np.asarray(m0, np.float32) * np.asarray(r) / np.asarray(mf, np.float32), rtol=3e-7)
    np.testing.assert_array_equal(np.asarray(got.cumulus_tendencies[6]), np.asarray(rates[6]))


def test_kf99_rk_route_clears_expired_rates_after_consumption(monkeypatch):
    import gpuwrf.coupling.kf99_adapter as K
    from gpuwrf.runtime import operational_mode as om

    grid, state, carry, nml = _fixture(99)
    rates = tuple(jnp.full(state.theta.shape, 1e-5 * (i + 1), jnp.float32) for i in range(6)) + (
        jnp.full(state.t_skin.shape, 2e-3, jnp.float32),)
    dt = float(nml.dt_s)
    nca = jnp.where(jnp.arange(state.t_skin.size).reshape(state.t_skin.shape) % 2 == 0,
                    0.4 * dt, 10.0 * dt).astype(jnp.float32)  # NINT(0.4)=0 -> expired; NINT(10) -> held
    carry = carry.replace(cumulus_tendencies=rates, cumulus_carry=(jnp.zeros_like(state.theta), nca))
    monkeypatch.setattr(K, "kf99_refresh", lambda *a, **kw: (rates, jnp.zeros_like(state.theta), nca))
    captured = {}
    real = om._source_leaf_dry_tendencies

    def spy(*a, **kw):
        captured.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(om, "_source_leaf_dry_tendencies", spy)
    out = om._physics_step_forcing(carry, nml, jnp.asarray(54.0), run_radiation=False)
    assert all(captured["kf_tendf"][s] is not None for s in range(6))  # KFSCHEME adds all six
    clear = (np.asarray(nca) > 0) & (np.floor(np.asarray(nca) / dt + 0.5) <= 0)
    assert clear.any() and (~clear).any()
    held = np.asarray(out.carry.cumulus_tendencies[0])
    assert np.all(held[:, clear] == 0.0) and np.all(held[:, ~clear] != 0.0)
    assert np.all(np.asarray(captured["kf_tendf"][0])[:, 1:-1, 1:-1] != 0.0)  # consumed before the clear
    np.testing.assert_allclose(np.asarray(out.carry.cumulus_carry[1]),
                               np.where(np.asarray(nca) > 0, np.asarray(nca) - dt, np.asarray(nca)), rtol=1e-6)


@pytest.mark.parametrize("bl", (2, 3))
def test_euler_pbls_publish_rk_sources_in_source_leaf_mode(bl, monkeypatch):
    """F9 (PBL part): MYJ and non-sf1 PBLs (GFS ...) hand RU/RV/RTH/RQVBLTEN to the RK seam."""
    from gpuwrf.runtime import operational_mode as om

    grid, state, carry, nml = _fixture(0, bl_pbl_physics=bl)
    shape = state.theta.shape
    nz, ny, nx = shape
    rng = np.random.default_rng(9)
    rates = {k: jnp.asarray((1e-4 * rng.standard_normal(shape)).astype(np.float32)) for k in ("u", "v", "theta", "qv")}

    def cols(a):
        return jnp.moveaxis(a, 0, -1).reshape(ny * nx, nz)

    if bl == 2:
        def fake_myj(st, dt, grid_, *, return_tendencies=False, return_cumulus_inputs=False, **kw):
            assert return_tendencies, "source-leaf MYJ must publish rates"
            zero = jnp.zeros_like(rates["qv"])
            out = {"rublten": rates["u"], "rvblten": rates["v"], "rthblten": rates["theta"], "rqvblten": rates["qv"],
                   "rqcblten": zero, "rqiblten": zero}  # F6 (pblmix) MYJ condensate rates
            return st, out, (jnp.zeros((ny, nx)), jnp.zeros((ny, nx), jnp.int32))
        monkeypatch.setattr(om, "myj_pbl_adapter", fake_myj)
    else:
        def fake_pbl(st, dt, grid_, *, _with_sources=False, **kw):
            assert _with_sources, "source-leaf PBL must publish rates"
            euler = st.replace(theta=st.theta + 1.0, qv=st.qv + 1e-3)  # must be discarded
            return euler, {k: cols(v) for k, v in rates.items()}
        monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 3, fake_pbl)
    captured = {}
    real = om._source_leaf_dry_tendencies

    def spy(*a, **kw):
        captured["args"] = a
        return real(*a, **kw)

    monkeypatch.setattr(om, "_source_leaf_dry_tendencies", spy)
    monkeypatch.setenv("GPUWRF_PHYS_TEND_RK_WRF", "1")
    out = om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    for name in ("theta", "qv", "u", "v"):
        np.testing.assert_array_equal(getattr(out.state, name), getattr(state, name), err_msg=name)
    a = captured["args"]  # (mu, rthraten, rthblten, rqvblten, rublten, rvblten, ...)
    for got, key in zip(a[2:6], ("theta", "qv", "u", "v")):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(rates[key]), err_msg=key)
    assert "qv" in out.dry_tendencies.moist_tendf
