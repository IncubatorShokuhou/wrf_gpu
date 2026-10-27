"""CPU gates for the exact WRF spec-zone walk of a specified ROOT (o5-coast, GPUWRF_ROOT_SPEC_STAGE_CLOCK).

Oracle = scripts/o5_root_spec_walk_oracle.py (pristine WRF formulas, no gpuwrf imports): real.exe couples every
wrfbdy record with its own mass, spec_bdy_dry hands (rec[l+1]-rec[l])/interval to spec_bdyupdate, which adds
dts_rk*btend to the ring after every acoustic substep from time level n (small_step_prep restarts each stage).
The released root pinned the absolute value at lead_seconds + dt_rk (lead_seconds = END of the step) from the
first substep on; these gates fail for that clock and for any decoupled (leaf-rate) tendency.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import boundary_apply as ba
from gpuwrf.coupling import root_spec_walk as rsw
from gpuwrf.dynamics.core import acoustic
from gpuwrf.kernels import dyn_acoustic_fp32
from gpuwrf.runtime import operational_mode as om
from scripts import o5_root_spec_walk_oracle as oracle

NZ, NY, NX, BW = 3, 7, 9, 5
INTERVAL, DT, NS = 3600.0, 54.0, 4
CFG = ba.BoundaryConfig(spec_bdy_width=BW, spec_zone=1, relax_zone=4, update_cadence_s=INTERVAL)


def _sides(field2d_or_3d):
    """d02_replay._field_sides_3d convention on (z, y, x): {W,E,S,N: (bw, z, tan)}, b=0 outermost."""
    a = np.asarray(field2d_or_3d, np.float64)
    if a.ndim == 2:
        a = a[None]
    return {"W": np.moveaxis(a[:, :, :BW], 2, 0), "E": np.moveaxis(np.flip(a[:, :, -BW:], 2), 2, 0),
            "S": np.moveaxis(a[:, :BW, :], 1, 0), "N": np.moveaxis(np.flip(a[:, -BW:, :], 1), 1, 0)}


def _leaf(per_record_sides, length):
    """Stack {W,E,S,N: (bw, z, tan)} per record into the (time, side, bw, z, len) leaf (zero padded)."""
    out = []
    for sides in per_record_sides:
        rec = []
        for s in "WESN":
            a = sides[s]
            rec.append(np.pad(a, ((0, 0), (0, 0), (0, length - a.shape[-1]))))
        out.append(np.stack(rec))
    return np.stack(out)


def _case(seed=7):
    """Two coupled wrfbdy-like records and the DECOUPLED leaves the production loader would store."""
    rng = np.random.default_rng(seed)
    c1h, c2h = np.linspace(0.95, 0.2, NZ), np.linspace(10.0, 2000.0, NZ)
    c1f, c2f = np.linspace(1.0, 0.1, NZ + 1), np.linspace(0.0, 3000.0, NZ + 1)
    mub = 9.0e4 + 500.0 * rng.random((NY, NX))
    msfuy = 1.0 + 0.02 * rng.random((NY, NX + 1))
    msfvx = 1.0 + 0.02 * rng.random((NY + 1, NX))
    length = max(NX + 1, NY + 1)
    recs = []
    for k in range(2):
        mu = 300.0 * rng.standard_normal((NY, NX))
        total = mub + mu
        mt_h = c1h[:, None, None] * total + c2h[:, None, None]
        mt_f = c1f[:, None, None] * total + c2f[:, None, None]
        mtu = np.stack([oracle.mu_staggered_u(row) for row in total])                    # (ny, nx+1)
        mtv = np.stack([oracle.mu_staggered_u(col) for col in total.T]).T                 # (ny+1, nx)
        t_c = (2.0 * rng.standard_normal((NZ, NY, NX))) * mt_h                            # coupled T_B
        ph_c = (50.0 * rng.standard_normal((NZ + 1, NY, NX))) * mt_f                      # coupled PH_B
        u_c = (5.0 * rng.standard_normal((NZ, NY, NX + 1))) * (c1h[:, None, None] * mtu + c2h[:, None, None]) / msfuy
        v_c = (5.0 * rng.standard_normal((NZ, NY + 1, NX))) * (c1h[:, None, None] * mtv + c2h[:, None, None]) / msfvx
        recs.append(dict(mu=mu, total=total, t=t_c, ph=ph_c, u=u_c, v=v_c, mt_h=mt_h, mt_f=mt_f,
                         mt_u=(c1h[:, None, None] * mtu + c2h[:, None, None]) / msfuy,
                         mt_v=(c1h[:, None, None] * mtv + c2h[:, None, None]) / msfvx))
    leaves = {
        "mu_bdy": _leaf([_sides(r["mu"]) for r in recs], length),
        "mub_bdy": _leaf([_sides(mub) for _ in recs], length),
        "theta_bdy": _leaf([{s: v + 300.0 for s, v in _sides(r["t"] / r["mt_h"]).items()} for r in recs], length),
        "ph_bdy": _leaf([_sides(r["ph"] / r["mt_f"]) for r in recs], length),
        "u_bdy": _leaf([_sides(r["u"] / r["mt_u"]) for r in recs], length),
        "v_bdy": _leaf([_sides(r["v"] / r["mt_v"]) for r in recs], length),
    }
    state = SimpleNamespace(**{k: jnp.asarray(v) for k, v in leaves.items()})
    metrics = SimpleNamespace(c1h=jnp.asarray(c1h), c2h=jnp.asarray(c2h), c1f=jnp.asarray(c1f),
                              c2f=jnp.asarray(c2f), msfuy=jnp.asarray(msfuy), msfvx=jnp.asarray(msfvx))
    return state, metrics, recs


def _ring(rate, ring_field):
    """Compare only WRF spec_bdyupdate cells: S/N rows full, W/E columns rows 1..-2 (module_bc.F)."""
    a = np.asarray(rate, np.float64)
    out = np.zeros_like(a)
    out[..., 0, :], out[..., -1, :] = ring_field[..., 0, :], ring_field[..., -1, :]
    out[..., 1:-1, 0], out[..., 1:-1, -1] = ring_field[..., 1:-1, 0], ring_field[..., 1:-1, -1]
    return a, out


def test_operational_mode_uses_the_moved_root_walk_rates() -> None:
    """Seam: the step reaches the rates through coupling.root_spec_walk (moved out of boundary_apply so no code
    sits above its Pallas kernel, E109); a dropped or stale import would only fail when the walk is traced."""

    assert om.root_walk_rates is rsw.root_walk_rates
    assert not hasattr(ba, "root_walk_rates") and not hasattr(ba, "RootWalkRates")


@pytest.mark.parametrize("lead", [DT, 2 * DT, INTERVAL - DT, INTERVAL])
def test_root_walk_rates_restore_wrf_coupled_record_tendencies(lead) -> None:
    state, metrics, recs = _case()
    r = rsw.root_walk_rates(state, lead, metrics, CFG, theta_offset=300.0, nz=NZ, ny=NY, nx=NX, dtype=jnp.float64)
    for name, got in (("mu", np.asarray(r.mu)[None]), ("t", r.theta), ("ph", r.ph), ("u", r.u), ("v", r.v)):
        want = oracle.btend(recs[0][name] if name != "mu" else recs[0]["mu"][None],
                            recs[1][name] if name != "mu" else recs[1]["mu"][None], INTERVAL)
        got, want = _ring(got, want)
        mask = np.zeros(got.shape, bool)
        mask[..., 0, :] = mask[..., -1, :] = True
        mask[..., 1:-1, 0] = mask[..., 1:-1, -1] = True
        np.testing.assert_allclose(got[mask], want[mask], rtol=1e-11, atol=1e-9, err_msg=name)
        assert np.all(got[~mask] == 0.0), name


def test_decoupled_leaf_rate_is_not_the_wrf_tendency() -> None:
    """Deletion check: dropping the per-record re-coupling (leaf rate) must miss WRF's btend."""
    state, metrics, recs = _case()
    leaf_rate = ba.specified_boundary_tendency(state.theta_bdy, DT, INTERVAL, z_len=NZ, y_len=NY, x_len=NX,
                                               dtype=jnp.float64, config=CFG)
    want = oracle.btend(recs[0]["t"], recs[1]["t"], INTERVAL)
    assert np.abs(np.asarray(leaf_rate)[:, 0, :] - want[:, 0, :]).max() > 1e-3 * np.abs(want[:, 0, :]).max()


def _walk_ring_times(mode_clock: str):
    """Ring valid time after every substep of every RK stage for the port's root clock (one root step)."""
    step_end = 7 * DT
    t_n = step_end - DT
    times = []
    for dt_rk, nsub, dts in oracle.wrf_rk_substeps(DT, NS):
        if mode_clock == "walk":
            times.append([t_n + m * dts for m in range(1, nsub + 1)])
        else:
            pin = ba.root_spec_stage_lead(step_end, DT, dt_rk, step_end + dt_rk)
            times.append([float(pin)] * nsub)
    return times


def test_walk_ring_times_per_rk_stage_match_wrf_and_released_clock_does_not(monkeypatch) -> None:
    wrf = oracle.wrf_ring_times(7 * DT, DT, NS)
    # Walk: start at bdy(t_n), + dts*btend per substep on a linear trajectory == WRF ring times exactly.
    state, metrics, recs = _case()
    r = rsw.root_walk_rates(state, 7 * DT, metrics, CFG, theta_offset=300.0, nz=NZ, ny=NY, nx=NX, dtype=jnp.float64)
    rate = np.asarray(r.mu)
    t_n = 6 * DT
    start = recs[0]["mu"] + t_n * (recs[1]["mu"] - recs[0]["mu"]) / INTERVAL
    for (dt_rk, nsub, dts), times in zip(oracle.wrf_rk_substeps(DT, NS), wrf):
        walked = oracle.walk(start, rate, dts, nsub)
        for value, t in zip(walked, times):
            exact = recs[0]["mu"] + t * (recs[1]["mu"] - recs[0]["mu"]) / INTERVAL
            np.testing.assert_allclose(value[0, :], exact[0, :], rtol=1e-12, atol=1e-9)
    assert _walk_ring_times("walk") == wrf
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "0")
    released = _walk_ring_times("released")
    lead = [r[-1] - w[-1] for r, w in zip(released, wrf)]
    assert lead == [DT, DT, DT], "released root clock must be one step ahead at every stage end"


def test_mode_selection_default_off_and_env_walk(monkeypatch) -> None:
    """Merged default OFF (manager 2026-10-24 18:51Z): unset = released clock until the GPU WALK arm passes;
    the release candidate flips FAST_PATH_DEFAULTS to walk in one line (separate commit)."""
    from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS

    assert "GPUWRF_ROOT_SPEC_STAGE_CLOCK" not in FAST_PATH_DEFAULTS
    monkeypatch.delenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", raising=False)
    assert ba.root_spec_stage_clock_mode() == 0
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "walk")
    assert ba.root_spec_stage_clock_mode() == ba.ROOT_SPEC_WALK
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "0")
    assert ba.root_spec_stage_clock_mode() == 0
    assert ba.root_spec_stage_lead(10.0, 3.0, 1.0, 11.0) == 11.0
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "bogus")
    with pytest.raises(ValueError):
        ba.root_spec_stage_clock_mode()


def test_env_walk_activates_production_gate_only_for_a_specified_root(monkeypatch) -> None:
    def namelist(force_geopotential=True, cadence=True, nested_bundle=False, run_boundary=True):
        return SimpleNamespace(specified_bdy_cadence=cadence, run_boundary=run_boundary,
                               grid=SimpleNamespace(bc=SimpleNamespace(source="wrfbdy")),
                               boundary_config=SimpleNamespace(force_geopotential=force_geopotential,
                                                               nested_frozen_wrf_boundary_bundle=nested_bundle))
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "walk")
    assert om._root_spec_walk_active(namelist())
    assert not om._root_spec_walk_active(namelist(cadence=False))
    assert not om._root_spec_walk_active(namelist(force_geopotential=False, nested_bundle=True))  # live nest
    assert not om._root_spec_walk_active(namelist(run_boundary=False))
    monkeypatch.setenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", "0")
    assert not om._root_spec_walk_active(namelist())
    monkeypatch.delenv("GPUWRF_ROOT_SPEC_STAGE_CLOCK", raising=False)
    assert not om._root_spec_walk_active(namelist())


def test_production_wiring_uses_walk_rates_and_wrf_ring_kernels() -> None:
    src = inspect.getsource(om._acoustic_core_state_from_prep)
    assert "_root_spec_walk_active(namelist)" in src and "root_walk_rates(" in src
    assert "elif walk_rates is not None:" in src, "root walk branch missing"
    walk = src.split("elif walk_rates is not None:", 1)[1].split("else:", 1)[0]
    for line in ("mu_spec_target = walk_rates.mu", "muts_spec_target = walk_rates.mu",
                 "theta_spec_target = walk_rates.theta", "ph_bdy_target_full = walk_rates.ph"):
        assert line in walk
    assert "muave_spec_target" not in walk and "u_spec_tan_target" not in walk
    assert "u_work_bdy, v_work_bdy = walk_rates.u, walk_rates.v" in src
    stage = inspect.getsource(om._rk_stage_acoustic) if hasattr(om, "_rk_stage_acoustic") else inspect.getsource(om)
    assert "root_spec_walk=bool(lead_seconds is not None and _root_spec_walk_active(namelist))" in stage
    assert "or _root_spec_walk_active(namelist))" in stage  # native mu_work rate
    core = inspect.getsource(acoustic.acoustic_substep_core)
    assert core.count("bool(cfg.nested_frozen_wrf_boundary_bundle) or bool(cfg.root_spec_walk)") >= 4
    native = inspect.getsource(dyn_acoustic_fp32)
    assert native.count("cfg.nested_frozen_wrf_boundary_bundle or cfg.root_spec_walk") == 3


# ---------------------------------------------------------------------------------------------------------------
# Production-path numerics (critic cr-o5-coast 2026-10-24): call the NATIVE acoustic boundary adapters that the release
# path executes, compiled, with fp32 operands; expected values from WRF's spec_bdyupdate / spec_bdyupdate_ph formulas.
class _Fields(SimpleNamespace):
    def replace(self, **updates):
        return _Fields(**(vars(self) | updates))


def _wrf_ring(array, target):
    """WRF spec_bdyupdate ownership: S/N rows full, W/E columns rows 1..-2 (module_bc.F:1955-2064)."""
    out = np.array(array, copy=True)
    t = np.asarray(target)
    out[..., 0, :], out[..., -1, :] = t[..., 0, :], t[..., -1, :]
    out[..., 1:-1, 0], out[..., 1:-1, -1] = t[..., 1:-1, 0], t[..., 1:-1, -1]
    return out


def _walk_cfg(dt, walk=True):
    return SimpleNamespace(dt=dt, root_spec_walk=walk, nested_frozen_wrf_boundary_bundle=False, spec_zone=1,
                           spec_w_zero_grad=True, dt_full=dt, normal_bdy_relax_strength=1.0,
                           specified_relax_tendency=True)


@pytest.mark.parametrize("dts,nsub", [(18.0, 1), (13.5, 2), (13.5, 4)])
def test_native_mass_walk_accumulates_dts_btend_per_substep(dts, nsub) -> None:
    import jax

    rng = np.random.default_rng(3)
    f2 = lambda: jnp.asarray(rng.normal(size=(NY, NX)), jnp.float32)  # noqa: E731
    f3 = lambda: jnp.asarray(rng.normal(size=(NZ, NY, NX)), jnp.float32)  # noqa: E731
    rate_mu, rate_t = f2() * 0.01, f3() * 0.1
    before = _Fields(mu=f2() + 12, mu_work=f2() - 3, muts=f2() + 9.0e4, muave=f2(), theta_coupled_work=f3() + 1234,
                     mu_spec_target=rate_mu, muts_spec_target=rate_mu, theta_spec_target=rate_t,
                     mu_work_spec_target=rate_mu, muave_spec_target=None)
    cfg = _walk_cfg(dts)
    names = ("mu", "mu_work", "muts", "theta_coupled_work")
    expected = {n: np.asarray(getattr(before, n)) for n in names}
    for _ in range(nsub):
        after = before.replace(mu=f2() + 700, mu_work=f2() + 701, muts=f2() + 702, theta_coupled_work=f3() + 703,
                               muave=f2() + 704)

        def call(mu, mu_work, muts, theta, after=after):
            got = dyn_acoustic_fp32._mass_boundaries(
                before.replace(mu=mu, mu_work=mu_work, muts=muts, theta_coupled_work=theta), after, cfg)
            return got.mu, got.mu_work, got.muts, got.theta_coupled_work, got.muave

        out = jax.jit(call)(before.mu, before.mu_work, before.muts, before.theta_coupled_work)
        for n, value in zip(names, out[:4]):
            rate = np.asarray(rate_t if n == "theta_coupled_work" else rate_mu)
            expected[n] = _wrf_ring(np.asarray(getattr(after, n)), expected[n] + np.float32(dts) * rate)
            np.testing.assert_array_equal(np.asarray(value), expected[n], err_msg=n)
        np.testing.assert_array_equal(np.asarray(out[4]), np.asarray(after.muave), err_msg="muave must not be pinned")
        before = before.replace(**dict(zip(names, out[:4])))


def test_native_mass_walk_differs_from_released_absolute_pin() -> None:
    f2 = jnp.full((NY, NX), 5.0, jnp.float32)
    before = _Fields(mu=f2, mu_work=f2, muts=f2, muave=f2, theta_coupled_work=jnp.full((NZ, NY, NX), 5.0, jnp.float32),
                     mu_spec_target=f2 * 0 + 0.5, muts_spec_target=f2 * 0 + 0.5,
                     theta_spec_target=jnp.full((NZ, NY, NX), 0.5, jnp.float32), mu_work_spec_target=f2 * 0 + 0.5,
                     muave_spec_target=f2 * 0 + 0.5)
    after = before.replace(mu=f2 + 1, mu_work=f2 + 1, muts=f2 + 1, muave=f2 + 1)
    walk = dyn_acoustic_fp32._mass_boundaries(before, after, _walk_cfg(10.0, True))
    pinned = dyn_acoustic_fp32._mass_boundaries(before, after, _walk_cfg(10.0, False))
    assert float(walk.mu[0, 3]) == 10.0          # 5 + 10 * 0.5 (walk from the pre-substep ring)
    assert float(pinned.mu[0, 3]) == 0.5         # released absolute pin = target value
    assert float(walk.muave[0, 3]) == 6.0 and float(pinned.muave[0, 3]) == 0.5


def test_native_uv_walk_covers_all_sides_not_absolute_pin() -> None:
    import jax

    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.normal(size=(NZ, NY, NX + 1)), jnp.float32)
    v = jnp.asarray(rng.normal(size=(NZ, NY + 1, NX)), jnp.float32)
    ru, rv = (jnp.asarray(rng.normal(size=a.shape) * 0.01, jnp.float32) for a in (u, v))
    before = _Fields(u=u, v=v, u_work_bdy=ru, v_work_bdy=rv, u_spec_tan_target=None, v_spec_tan_target=None,
                     ru_m=u * 0, rv_m=v * 0)
    after = before.replace(u=u + 100, v=v + 100)

    def call(u, v):
        got = dyn_acoustic_fp32._uv_boundaries(before.replace(u=u, v=v), after, _walk_cfg(13.5))
        return got.u, got.v

    gu, gv = jax.jit(call)(u, v)
    for got, adv, start, rate in ((gu, after.u, u, ru), (gv, after.v, v, rv)):
        want = _wrf_ring(np.asarray(adv), np.asarray(start) + np.float32(13.5) * np.asarray(rate))
        np.testing.assert_array_equal(np.asarray(got), want)


def test_native_ph_walk_matches_wrf_spec_bdyupdate_ph_and_keeps_w_zero_grad() -> None:
    from scripts import v0234_1500_scientific_rca_source_oracle as nested_oracle

    rng = np.random.default_rng(5)
    nzw = NZ + 1
    g = lambda shape, s=1.0, o=0.0: jnp.asarray(o + s * rng.normal(size=shape), jnp.float32)  # noqa: E731
    c1f, c2f = g((nzw,), 0.1, 0.8), g((nzw,), 50.0, 500.0)  # 1-D like the native state
    before = _Fields(ph=g((nzw, NY, NX), 10.0), w=g((nzw, NY, NX)), w_save=g((nzw, NY, NX)), mut=g((NY, NX), 10, 9e4),
                     ph_bdy_target=g((nzw, NY, NX), 0.5), ph_save_for_spec=g((nzw, NY, NX), 20.0, 1e3),
                     mu_spec_target=g((NY, NX), 0.01), w_spec_target=None, c1f=c1f, c2f=c2f,
                     msfty=g((NY, NX), 0.01, 1.0))
    after = before.replace(ph=before.ph + 3.0, w=before.w + 1.0, muts=g((NY, NX), 10, 9e4))
    walk = dyn_acoustic_fp32._w_boundaries(before, after, _walk_cfg(13.5, True))
    released = dyn_acoustic_fp32._w_boundaries(before, after, _walk_cfg(13.5, False))
    ring = nested_oracle.wrf_spec_ph_update(np.asarray(before.ph), np.asarray(before.ph_save_for_spec),
                                            np.asarray(before.ph_bdy_target), np.asarray(before.mu_spec_target)[None],
                                            np.asarray(after.muts)[None], np.asarray(c1f)[:, None, None],
                                            np.asarray(c2f)[:, None, None], 13.5)
    want = _wrf_ring(np.asarray(after.ph, np.float64), ring)
    np.testing.assert_allclose(np.asarray(walk.ph, np.float64), want, rtol=2e-6, atol=2e-3)
    np.testing.assert_array_equal(np.asarray(walk.ph)[:, 1:-1, 1:-1], np.asarray(after.ph)[:, 1:-1, 1:-1])
    assert np.abs(np.asarray(walk.ph) - np.asarray(released.ph)).max() > 1.0, "walk must not be the absolute ph pin"
    np.testing.assert_array_equal(np.asarray(walk.w), np.asarray(released.w))  # specified root keeps zero_grad_bdy


def test_native_mass_noop_mutant_is_killed(monkeypatch) -> None:
    import jax

    monkeypatch.setattr(dyn_acoustic_fp32, "_mass_boundaries", lambda before, after, cfg: after)
    jax.clear_caches()
    with pytest.raises(AssertionError):
        test_native_mass_walk_accumulates_dts_btend_per_substep(18.0, 1)
