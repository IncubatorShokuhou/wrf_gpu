"""t-calleraudit F1: every admitted microphysics scheme feeds WRF's public precipitation
slots (RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV) to the land-surface caller.

WRF: module_surface_driver.F:1568 (RAINBL) and :3283 (MP_RAINNC=RAINNCV, MP_SNOW=SNOWNCV,
MP_GRAUP=GRAUPELNCV, MP_HAIL=HAILNCV) -> module_sf_noahmpdrv.F:779-783 (PRCP* = slot/DT).
Before this lane the port seeded the land precipitation carry only for mp 8/5/7/15/38, so
Noah-MP/RUC/SSiB received ZERO rain and snow under every other scheme.
"""
import dataclasses
import importlib.util
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
SLOTS = ("rainncv", "snowncv", "graupelncv", "hailncv")
KERNEL = {1: "kessler_physics_tendency", 2: "lin_physics_tendency", 3: "wsm3_physics_tendency",
          4: "wsm5_physics_tendency", 6: "wsm6_physics_tendency", 10: "morrison_tendency",
          13: "sbu_ylin_physics_tendency", 14: "wdm5_physics_tendency", 16: "wdm6_physics_tendency",
          24: "wsm7_physics_tendency", 26: "wdm7_physics_tendency", 40: "morrison_aero_tendency",
          97: "goddard_physics_tendency"}


def _pristine_driver_slots():
    """mp_physics code -> public slots its CASE block passes as keywords (pristine WRF 4.7.1)."""
    registry = (WRF / "Registry/Registry.EM_COMMON").read_text()
    package = {int(m[2]): m[1].upper() for m in re.finditer(r"(?m)^package\s+(\w+)\s+mp_physics==(\d+)\s", registry)}
    driver = (WRF / "phys/module_microphysics_driver.F").read_text().split("\n")
    cases = [(i, m[1].upper()) for i, line in enumerate(driver)
             if (m := re.match(r"\s*CASE\s*\(\s*(\w+)\s*\)", line))]
    by_case = {}
    for n, (start, name) in enumerate(cases):
        end = cases[n + 1][0] if n + 1 < len(cases) else len(driver)
        body = "".join(line.split("!")[0] for line in driver[start:end]).upper().replace(" ", "").replace("&", "")
        # The ACTUAL argument (grid variable), whatever the dummy is called (Milbrandt GRPLNCV=GRAUPELNCV).
        by_case[name] = {slot for slot in SLOTS if re.search(rf"={slot.upper()}[,)]", body)}
    return {code: by_case[name] for code, name in package.items() if name in by_case}


def test_port_public_slots_are_the_pristine_driver_slots():
    wrf = _pristine_driver_slots()
    # Independent reference: parsed pristine Registry + microphysics_driver, not port code (E98).
    assert wrf[1] == {"rainncv"} and wrf[13] == {"rainncv"} and wrf[24] >= {"hailncv"}
    for code, slots in sa.MP_PUBLIC_PRECIP_SLOTS.items():
        assert set(slots) == wrf[code], code
    for code in (9, 18):  # explicit-branch producers that must carry HAILNCV
        assert wrf[code] == set(SLOTS), code
    assert wrf[28] == {"rainncv", "snowncv", "graupelncv"}


def _fixture():
    spec = importlib.util.spec_from_file_location("calleraudit_rrtm_fixture",
                                                  ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real(state):
    return state.replace(**{n: getattr(state, n).astype(jnp.float32) for n in state.__slots__
                            if getattr(getattr(state, n, None), "dtype", None) == jnp.float64})


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("code", sorted(sa.MP_PUBLIC_PRECIP_SLOTS))
def test_generic_mp_caller_feeds_only_wrf_public_slots(monkeypatch, code, real):
    f = _fixture()
    grid = f._grid(nz=8, ny=3, nx=4)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=code)
    state = _real(state) if real else state
    ncol, nz = grid.ny * grid.nx, grid.nz
    amounts = {"rain_acc": 7.0, "snow_acc": 3.0, "graupel_acc": 2.0, "hail_acc": 1.0}

    def kernel(*args, **kwargs):
        diagnostics = {"ri3d": jnp.zeros((ncol, nz))}
        if code == 1:  # Kessler public RAINNCV = last sedimentation sub-step only
            diagnostics["rainncv"] = jnp.full((ncol,), 5.0)
        present = {k: v for k, v in amounts.items() if getattr(state, k, None) is not None}
        return PhysicsTendency(accumulator_increments={k: jnp.full((ncol,), v) for k, v in present.items()},
                               diagnostics=diagnostics)

    monkeypatch.setattr(sa, KERNEL[code], kernel)
    nl = dataclasses.replace(f._namelist(grid), mp_physics=code, run_physics=True)
    updated, water = op._apply_post_rk_microphysics(state, nl, return_precipitation=True)
    slots = sa.MP_PUBLIC_PRECIP_SLOTS[code]
    held = precipitation_from_step(water, jnp.zeros(state.t_skin.shape), 18.0)
    has = lambda acc: getattr(state, acc, None) is not None  # noqa: E731
    expect = {"prcpnonc": 5.0 if code == 1 else 7.0, "prcpsnow": 3.0 * ("snowncv" in slots),
              "prcpgrpl": 2.0 * ("graupelncv" in slots), "prcphail": 1.0 * ("hailncv" in slots and has("hail_acc"))}
    assert has("snow_acc") and has("graupel_acc") and has("hail_acc") == (code in (24, 26))
    for name, amount in expect.items():
        np.testing.assert_allclose(np.asarray(getattr(held, name)), amount / 18.0, rtol=1e-6, err_msg=name)
    # The model's own disjoint accumulators are untouched by the public-slot selection.
    np.testing.assert_allclose(np.asarray(updated.snow_acc - state.snow_acc), 3.0)


def test_nssl_and_milbrandt_hail_and_all_phase_total_reach_the_land(monkeypatch):
    nssl = {k: jnp.full((2, 3), v) for k, v in
            dict(rainncv=10., rain_liquid=4., snowncv=3., grplncv=2., hailncv=1., sr=.6).items()}
    held = precipitation_from_step(sa.nssl_public_precipitation(nssl), jnp.zeros((2, 3)), 10.)
    for name, amount in (("prcpnonc", 10.), ("prcpsnow", 3.), ("prcpgrpl", 2.), ("prcphail", 1.)):
        np.testing.assert_allclose(np.asarray(getattr(held, name)), amount / 10.)
    import gpuwrf.physics.milbrandt2mom.adapter as milbrandt
    from types import SimpleNamespace
    amounts = {k: jnp.full((2, 3), v) for k, v in dict(rain=5., snow=3., graupel=2., ice=0., rainncv=10.,
                                                        hailncv=1.).items()}
    monkeypatch.setattr(milbrandt, "milbrandt_adapter", lambda s, *a, **k: (s, amounts, None))
    monkeypatch.setattr(op, "_microphysics_interior_only", lambda before, after, nml: after)
    state = SimpleNamespace(t_skin=jnp.zeros((2, 3)))
    _, water = op._apply_post_rk_microphysics(
        state, SimpleNamespace(run_physics=True, mp_physics=9, dt_s=10., grid=None), return_precipitation=True)
    held = precipitation_from_step(water, jnp.zeros((2, 3)), 10.)
    np.testing.assert_allclose(np.asarray(held.prcpnonc), 1.0)
    np.testing.assert_allclose(np.asarray(held.prcphail), .1)


@pytest.mark.parametrize("code", [17, 28])
def test_explicit_and_alias_producers_return_precipitation(monkeypatch, code):
    from types import SimpleNamespace
    calls = []
    amounts = {"rain": jnp.ones((2, 3))}

    def adapter(state, *args, **kwargs):
        calls.append(kwargs.get("return_precipitation"))
        return (state, amounts) if kwargs.get("return_precipitation") else state
    if code == 28:
        monkeypatch.setattr(op, "thompson_aero_adapter", adapter)
    else:
        monkeypatch.setitem(op.MP_SCAN_ADAPTERS, 17, adapter)
    monkeypatch.setattr(op, "_microphysics_interior_only", lambda before, after, nml: after)
    state = SimpleNamespace(t_skin=jnp.zeros((2, 3)))
    actual, water = op._apply_post_rk_microphysics(
        state, SimpleNamespace(run_physics=True, mp_physics=code, dt_s=10., grid=None), return_precipitation=True)
    assert actual is state and water is amounts and calls == [True]


@pytest.mark.parametrize("code,seeded", [(0, True), (1, True), (6, True), (13, True), (28, True), (97, True)])
def test_operational_init_seeds_land_precipitation_for_every_precipitating_scheme(code, seeded):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _initial_carry_for_run
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18,
        acoustic_substeps=4, use_noahmp=True, run_physics=True, mp_physics=code,
        ra_sw_physics=0, ra_lw_physics=0, cu_physics=0, bl_pbl_physics=0)
    carry = _initial_carry_for_run(state, nml)
    assert (carry.noahmp_precipitation is not None) == seeded


def test_real_wsm6_kernel_amounts_reach_the_land_carry():
    f = _fixture()
    grid = f._grid(nz=8, ny=3, nx=4)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=6)
    qr = jnp.zeros_like(state.qr).at[0:4].set(2.e-3)
    qs = jnp.zeros_like(state.qs).at[4:6].set(1.e-3)
    state = state.replace(qr=qr, qs=qs, qv=jnp.full_like(state.qv, 5.e-3))
    nl = dataclasses.replace(f._namelist(grid), mp_physics=6, run_physics=True)
    updated, water = op._apply_post_rk_microphysics(state, nl, return_precipitation=True)
    total = np.asarray(water["total"])
    assert np.all(total > 0)
    # Independent channel: the model's disjoint accumulators sum to the all-phase RAINNCV.
    acc = sum(np.asarray(getattr(updated, n) - getattr(state, n)) for n in ("rain_acc", "snow_acc", "graupel_acc"))
    np.testing.assert_allclose(acc, total, rtol=1e-5)
    held = precipitation_from_step(water, jnp.zeros(state.t_skin.shape), float(nl.dt_s))
    np.testing.assert_allclose(np.asarray(held.prcpnonc), total / float(nl.dt_s), rtol=1e-6)


# --- F2: convective RAINCV of the stateless cumulus schemes reaches the land surface -----------
# WRF module_surface_driver.F:1568 RAINBL += RAINCV and :3283 MP_RAINC = RAINCV ->
# module_sf_noahmpdrv.F:779 PRCPCONV = MP_RAINC/DT for EVERY cumulus scheme. Before this lane
# cu 2/3/5/6/16/93 (no held-rate tuple) fed PRCPCONV = 0.

@pytest.mark.parametrize("rk", [1, 0])  # 1: t-cfix-cu F9 RK-coupled rates; 0: Euler state increment
@pytest.mark.parametrize("mp", [6, 0])  # mp=0: convection-only (rv-t-calleraudit F2 FIX)
@pytest.mark.parametrize("cu", [2, 3, 5, 6, 16, 93])
def test_stateless_cumulus_raincv_reaches_land_through_the_actual_step(monkeypatch, cu, mp, rk):
    from gpuwrf.coupling.cu_forcing import cu_forcing_required, initial_cu_forcing
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
    f = _fixture()
    grid = f._grid(ny=3, nx=4, nz=8)
    state = f._state(grid)
    dt = 54.0
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, mp_physics=mp, cu_physics=cu, bl_pbl_physics=0,
                              sf_sfclay_physics=0, rad_rk_tendf=rk, run_boundary=False, dt_s=dt)
    # The production seed decides the carry (convection-only included: WRF RAINBL += RAINCV).
    assert op._land_precipitation_seed(dataclasses.replace(nml, use_noahmp=True)) == "plain"
    carry = initial_operational_carry(state).replace(noahmp_precipitation=seed_precipitation(state))
    if cu_forcing_required(cu):  # t-cfix-cu F7/F14: WRF's held RTHFTEN/RQVFTEN (zero at itimestep 1)
        carry = carry.replace(cu_forcing=initial_cu_forcing(state.theta))
    raincv = jnp.full(state.t_skin.shape, .36)
    seen = []

    def cumulus(st, *args, **kwargs):
        # F9 RK path: (uncoupled rates, RAINCV); Euler path: the State (+ RAINCV on request).
        seen.append(bool(kwargs.get("return_rates") or kwargs.get("return_raincv")))
        rates = {"theta": jnp.zeros_like(st.theta), "qv": jnp.zeros_like(st.theta)}
        if cu == 2:
            cldefi = jnp.ones_like(st.t_skin)
            if kwargs.get("return_rates"):
                return (rates, raincv), cldefi
            return (st, cldefi, raincv) if kwargs.get("return_raincv") else (st, cldefi)
        if kwargs.get("return_rates"):
            return rates, raincv
        return (st, raincv) if kwargs.get("return_raincv") else st
    if cu == 6:
        monkeypatch.setattr(op, "tiedtke_adapter", cumulus)
    elif cu == 16:
        monkeypatch.setattr(op, "ntiedtke_adapter", cumulus)
    elif cu == 2:
        monkeypatch.setattr(op, "bmj_adapter", cumulus)
    else:
        monkeypatch.setitem(op.CU_STATELESS_SCAN_ADAPTERS, cu, cumulus)
    zero = jnp.zeros_like(state.t_skin)
    monkeypatch.setitem(op.MP_SCAN_ADAPTERS, 6, lambda st, *a, **k: (
        (st, dict(rain=zero, snow=zero, graupel=zero, ice=zero, hail=zero, total=zero))
        if k.get("return_precipitation") else st))
    monkeypatch.setattr(op, "_rk_scan_step", lambda c, nml, **kw: c)
    out = op._physics_boundary_step(carry, nml, jnp.asarray(2), run_radiation=False)
    assert seen == [True]
    np.testing.assert_allclose(np.asarray(out.noahmp_precipitation.prcpconv),
                               np.float32(.36) / np.float32(dt), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(out.noahmp_precipitation.prcpnonc), 0)


def test_tiedtke_adapter_returns_the_raincv_it_accumulates(monkeypatch):
    f = _fixture()
    grid = f._grid(ny=1, nx=2, nz=8)
    state = f._state(grid)

    def column(*args, **kw):
        zero = jnp.zeros_like(args[0])
        rates = {n: zero for n in ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN",
                                   "RUCUTEN", "RVCUTEN")}
        rates["RAINCV"] = jnp.asarray(.25, args[0].dtype)
        rates["ICUM1"] = jnp.bool_(True)  # t-cfix-cu c06 ROOT2 ABI: convective tile row
        return rates
    monkeypatch.setattr(sa, "tiedtke_column_jax", column)
    updated, raincv = sa.tiedtke_adapter(state, 18., grid, return_raincv=True)
    np.testing.assert_allclose(np.asarray(raincv), .25, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(updated.rainc_acc - state.rainc_acc), np.asarray(raincv), rtol=1e-6)
    assert not isinstance(sa.tiedtke_adapter(state, 18., grid), tuple)
