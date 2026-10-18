"""Morrison mp=10/40 + Noah-MP + do_radar_ref=1: one microphysics call feeds BOTH side channels.

WRF microphysics_driver (MP_MORR_TWO_MOMENT[_AERO]) fills the public RAINNCV/SNOWNCV/GRAUPELNCV
slots that force the NEXT land-surface call (module_surface_driver.F:1568/:3283) and, under
``diagflag .and. do_radar_ref==1``, REFL_10CM from refl10cm_hm on the same MP-exit columns.
Composition of t-calleraudit F1 (public precipitation carry) with t-classd (Morrison radar).

Compiled two-step seam (E85 pattern): the real operational dispatcher, carry seeds, MP adapter,
precipitation_from_step and radar_from_context run under CPU JIT; only RK and the Noah-MP solver
are substituted (the solver stub records the precipitation operand it is handed). Structural
evidence, not forecast fidelity: the Morrison operator and refl10cm_hm are proven against
pristine WRF elsewhere (tests/v040/test_classd_morrison_radar.py, F1 slot oracle).
"""
import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
DT = 30.0
ADAPTERS = {mp: sa.MP_SCAN_ADAPTERS[mp] for mp in (10, 40)}  # unpatched, captured at import


def _fixture():
    spec = importlib.util.spec_from_file_location("classd_compose_fixture",
                                                  ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(mp, monkeypatch):
    from gpuwrf.contracts import state as sc
    monkeypatch.setattr(sc, "_gpu_device", lambda: jax.devices("cpu")[0])
    f = _fixture()
    grid = f._grid(nz=8, ny=5, nx=6)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=mp)
    # Warm half: rain in the lowest levels; cold half: snow + graupel in the lowest levels,
    # so the public RAINNCV and SNOWNCV/GRAUPELNCV slots are active within one 30 s step.
    k = jnp.arange(grid.nz)[:, None, None]
    cold = jnp.arange(grid.nx)[None, None, :] >= grid.nx // 2
    low = k < 4
    shape = state.qv.shape
    put = lambda mask, value, leaf: jnp.broadcast_to(jnp.where(mask, value, 0.), shape).astype(leaf.dtype)  # noqa: E731
    state = state.replace(
        theta=jnp.broadcast_to(jnp.where(cold, 262.0, 300.0), shape).astype(state.theta.dtype),
        qr=put(~cold & low, 2.e-3, state.qr), Nr=put(~cold & low, 2.e4, state.Nr),
        qs=put(cold & low, 1.e-3, state.qs), Ns=put(cold & low, 1.e5, state.Ns),
        qg=put(cold & low, 5.e-4, state.qg), Ng=put(cold & low, 1.e3, state.Ng))
    return f, grid, state


def _case(mp, radar, monkeypatch):
    f, grid, state = _inputs(mp, monkeypatch)
    nl = dataclasses.replace(
        f._namelist(grid), dt_s=DT, mp_physics=mp, do_radar_ref=int(radar), diagnostic_history_interval=1.0,
        use_noahmp=True, run_physics=True, run_boundary=False, disable_guards=True,
        # sf_sfclay_physics=5 (MYNN-SL, solved inside the stubbed Noah-MP call): WRF surface_driver.F:1490
        # returns before EVERY land call when sf_sfclay_physics=0 (t-cfix-sfc F21), so 0 cannot feed Noah-MP.
        ra_sw_physics=0, ra_lw_physics=0, bl_pbl_physics=0, cu_physics=0, sf_sfclay_physics=5)
    carry = op._initial_carry_for_run(state, nl)
    # The daily/nested pipelines seed the land carry after this call; the solver is stubbed,
    # so its history products (land_history/energy sums/surface history) are not exercised.
    zero = jnp.zeros_like(state.t_skin)
    carry = carry.replace(noahmp_land=state.t_skin, noahmp_rad=(zero, zero, zero),
                          land_history=None, energy_accumulators=None, history_diagnostics=None)
    return grid, nl, carry


def _two_steps(mp, nl, carry, monkeypatch):
    calls, handed = [], []
    real = ADAPTERS[mp]

    def spy(state, dt, grid=None, **kwargs):
        out = real(state, dt, grid, **kwargs)
        calls.append((dict(kwargs), out))
        return out

    def noahmp(state, land, *args, precipitation=None, **kwargs):
        handed.append(precipitation)
        return state, land

    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    monkeypatch.setitem(op.MP_SCAN_ADAPTERS, mp, spy)
    monkeypatch.setattr(op, "_rk_scan_step", lambda c, *a, **kw: c)
    monkeypatch.setattr(op, "noahmp_surface_step", noahmp)
    monkeypatch.setattr(op, "_noahmp_params", lambda nml: (None, None))
    monkeypatch.setattr(op, "_lane_noahmp_static", lambda nml, clock_base=None: None)

    def run(c):
        first = op._physics_boundary_step(c, nl, jnp.int32(1), run_radiation=False)
        second = op._physics_boundary_step(first, nl, jnp.int32(2), run_radiation=False)
        outs = tuple(out for _, out in calls)
        # Expected land carry from each call's own public slots, in the same compiled program
        # (XLA rewrites /dt into a reciprocal multiply under jit; E123/E130).
        expect = tuple(precipitation_from_step(out[1], jnp.zeros_like(out[0].t_skin), DT,
                                               spec_zone=op._microphysics_spec_zone(nl)) for out in outs)
        return first, second, outs, tuple(handed), expect

    first, second, outs, handed_out, expect = jax.jit(run)(carry)
    return first, second, [kw for kw, _ in calls], outs, handed_out, expect


@pytest.mark.parametrize("mp", [10, 40])
def test_one_morrison_call_feeds_land_precipitation_and_radar(mp, monkeypatch):
    grid, nl, carry = _case(mp, True, monkeypatch)
    assert carry.noahmp_precipitation is not None and carry.radar_reflectivity is not None
    # REAL carry: mp=10 runs through _fp64_island (its tuple pass-through carries both channels).
    assert carry.state.theta.dtype == jnp.float32 and carry.state.qr.dtype == jnp.float32
    first, second, kwargs, outs, handed, expected = _two_steps(mp, nl, carry, monkeypatch)
    # One adapter call per step, with both side channels requested (no second MP call).
    assert kwargs == [dict(return_precipitation=True, return_radar=True)] * 2
    assert len(handed) == 2
    for step, (carry_after, out, expect) in enumerate(zip((first, second), outs, expected), start=1):
        updated, precip, context = out
        assert set(precip) >= {"rain", "snow", "graupel", "total"} and len(context) == 9
        for a, b in zip(jax.tree.leaves(expect), jax.tree.leaves(carry_after.noahmp_precipitation), strict=True):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), step
    # WRF: the land call of step 2 is forced by step 1's public slots (not zeros).
    for a, b in zip(jax.tree.leaves(first.noahmp_precipitation), jax.tree.leaves(handed[1]), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    total = np.asarray(outs[0][1]["total"])
    cold = np.arange(total.shape[-1]) >= total.shape[-1] // 2
    assert np.all(total > 0) and np.all(np.asarray(outs[0][1]["snow"])[:, cold] > 0)
    assert np.all(np.asarray(first.noahmp_precipitation.prcpnonc) > 0)
    # Radar: retained at step 1 (no history alarm), refl10cm_hm of step 2's MP exit at the alarm.
    np.testing.assert_array_equal(np.asarray(first.radar_reflectivity), np.asarray(carry.radar_reflectivity))
    from gpuwrf.diagnostics.morrison_radar import morrison_refl_10cm
    nz, ny, nx = carry.radar_reflectivity.shape
    exit_state = outs[1][0]
    # refl10cm_hm consumes REAL columns (morrison_reflectivity casts to f32): under _fp64_island
    # (mp=10) the context is the f64 kernel output and the State leaf its f32 narrowing.
    for value, leaf in zip(outs[1][2][:7], ("qv", "qr", "Nr", "qs", "Ns", "qg", "Ng")):
        np.testing.assert_array_equal(np.asarray(value, np.float32),
                                      np.asarray(sa._mp_in(getattr(exit_state, leaf), ny, nx, nz), np.float32))
    dbz = np.moveaxis(np.asarray(morrison_refl_10cm(*outs[1][2])).reshape(ny, nx, nz), -1, 0).astype(np.float32)
    zone = op._microphysics_spec_zone(nl)
    if zone:  # WRF microphysics never computes the specified zone: radar keeps its old value there
        inside = np.zeros((ny, nx), bool); inside[zone:ny - zone, zone:nx - zone] = True
        dbz = np.where(inside[None], dbz, np.asarray(first.radar_reflectivity))
    np.testing.assert_array_equal(np.asarray(second.radar_reflectivity), dbz)
    assert np.asarray(second.radar_reflectivity).max() > 0


@pytest.mark.parametrize("mp", [10, 40])
def test_radar_selection_does_not_change_state_or_land_precipitation(mp, monkeypatch):
    _, nl_on, carry_on = _case(mp, True, monkeypatch)
    on = _two_steps(mp, nl_on, carry_on, monkeypatch)
    _, nl_off, carry_off = _case(mp, False, monkeypatch)
    assert carry_off.radar_reflectivity is None and carry_off.noahmp_precipitation is not None
    off = _two_steps(mp, nl_off, carry_off, monkeypatch)
    assert off[2] == [dict(return_precipitation=True)] * 2  # F1 generic branch, radar off
    for c_on, c_off in ((on[0], off[0]), (on[1], off[1])):
        for name in ("state", "noahmp_precipitation"):
            a, b = jax.tree.leaves(getattr(c_on, name)), jax.tree.leaves(getattr(c_off, name))
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), name


@pytest.mark.parametrize("mp", [10, 40])
@pytest.mark.parametrize("real", [False, True])
def test_adapter_returns_precipitation_then_radar(mp, real, monkeypatch):
    """Direct seam, legacy and REAL carry: mp=10 runs through _fp64_island (tuple pass-through)."""
    _, grid, state = _inputs(mp, monkeypatch)
    if real:
        state = state.replace(_cast=False, **{n: getattr(state, n).astype(jnp.float32) for n in state.__slots__
                                 if getattr(getattr(state, n, None), "dtype", None) == jnp.float64})
    assert state.theta.dtype == (jnp.float32 if real else jnp.float64)
    adapter = ADAPTERS[mp]
    assert (getattr(adapter, "__wrapped__", None) is not None) == (mp == 10)
    plain = adapter(state, DT, grid)
    only_precip, precip = adapter(state, DT, grid, return_precipitation=True)
    only_radar, radar = adapter(state, DT, grid, return_radar=True)
    both, precip2, radar2 = adapter(state, DT, grid, return_precipitation=True, return_radar=True)
    for other in (only_precip, only_radar, both):
        for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(other), strict=True):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for a, b in zip(jax.tree.leaves(precip), jax.tree.leaves(precip2), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for a, b in zip(radar, radar2, strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert both.theta.dtype == state.theta.dtype
    assert np.all(np.asarray(precip["total"]) > 0)
