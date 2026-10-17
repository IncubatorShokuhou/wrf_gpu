"""t-calleraudit F24: Noah-classic (sf_surface 2) and Pleim-Xiu (7) receive WRF's land precipitation.

WRF: module_surface_driver.F:1568 RAINBL = RAINCV + RAINNCV (reset per step :4539, bldt=0);
module_sf_noahdrv.F:835 PRCP = RAINBL/DT and :859-866 FFROZP = SR (FRPCPN is .true. whenever SR is
present, surface_driver :2708); module_sf_pxlsm.F:482 PRECIP = MAX(1.0E-3*RAINBL/DTBL, 0). The port
hard-wired all three to 0. The SR channel is optional in the held precipitation carry (None for the
Noah-MP/RUC/SSiB release path, E78 refusal for Noah-classic without it).
"""
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime.noahmp_precipitation import NoahMPPrecipitation, precipitation_from_step, seed_precipitation

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads((ROOT / "proofs/t_calleraudit/f24_statement_oracle.json").read_text())


def _ulp_diff(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def _carry_from_oracle_inputs():
    """The port's held carry for every oracle row, built through the real producer seam."""
    rows = ORACLE["inputs"]
    f32 = lambda key: np.asarray([r[key] for r in rows], np.float32)  # noqa: E731
    raincv, rainncv, sr, dt = f32("raincv"), f32("rainncv"), f32("sr"), f32("dt")
    held = []
    for seconds in (18., 54.):
        sel = dt == np.float32(seconds)
        precip = {"rain": jnp.asarray(rainncv[sel]), "snow": jnp.zeros(sel.sum(), jnp.float32),
                  "graupel": jnp.zeros(sel.sum(), jnp.float32), "ice": jnp.zeros(sel.sum(), jnp.float32),
                  "total": jnp.asarray(rainncv[sel]), "sr": jnp.asarray(sr[sel])}
        conv = jnp.asarray(raincv[sel]) / jnp.float32(seconds)
        held.append((sel, precipitation_from_step(precip, conv, seconds, with_sr=True)))
    return held


def test_noahclassic_and_px_forcing_match_pristine_statements():
    from gpuwrf.coupling.noahclassic_surface_hook import noahclassic_precipitation
    from gpuwrf.coupling.pleim_xiu_surface_hook import pleim_xiu_precipitation
    prcp_ref = np.asarray(ORACLE["prcp"], np.float32)
    ffrozp_ref = np.asarray(ORACLE["ffrozp"], np.float32)
    precip_ref = np.asarray(ORACLE["precip"], np.float32)
    worst = {"prcp": 0, "precip": 0}
    for sel, held in _carry_from_oracle_inputs():
        prcp, ffrozp = noahclassic_precipitation(held)
        precip = pleim_xiu_precipitation(held)
        np.testing.assert_array_equal(np.asarray(ffrozp, np.float32), ffrozp_ref[sel])  # SR passthrough exact
        worst["prcp"] = max(worst["prcp"], int(_ulp_diff(prcp, prcp_ref[sel]).max()))
        worst["precip"] = max(worst["precip"], int(_ulp_diff(precip, precip_ref[sel]).max()))
        assert np.all(np.asarray(prcp)[prcp_ref[sel] > 0] > 0)
    # The carry holds RAINCV/DT and RAINNCV/DT separately (Noah-MP's own interface); summing the two
    # rounded quotients differs from WRF's (RAINCV+RAINNCV)/DT by at most a few ULP (disclosed bound).
    assert worst["prcp"] <= 1 and worst["precip"] <= 2, worst  # measured 1 / 2 ULP (224, 180 of 288 exact)


def test_noahclassic_assembler_consumes_the_held_carry():
    from gpuwrf.coupling import noahclassic_surface_hook as hook
    shape = (2, 3)
    held = seed_precipitation(SimpleNamespace(t_skin=jnp.zeros(shape)), with_sr=True)._replace(
        prcpconv=jnp.full(shape, 2.e-4, jnp.float32), prcpnonc=jnp.full(shape, 3.e-4, jnp.float32),
        sr=jnp.full(shape, .25, jnp.float32))
    prcp, ffrozp = hook.noahclassic_precipitation(held)
    np.testing.assert_allclose(np.asarray(prcp), 5.e-4, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ffrozp), .25)
    with pytest.raises(ValueError, match="E78"):
        hook.noahclassic_precipitation(held._replace(sr=None))
    with pytest.raises(ValueError, match="E78"):
        hook.noahclassic_precipitation(None)


def test_sr_channel_is_optional_and_off_for_the_release_carry():
    state = SimpleNamespace(t_skin=jnp.zeros((2, 3)))
    assert seed_precipitation(state).sr is None
    assert seed_precipitation(state, with_sr=True).sr is not None
    # A pre-F24 (5-field) held tuple restores with sr=None, i.e. refused by Noah-classic (E78).
    old = NoahMPPrecipitation(*seed_precipitation(state)[:5])
    assert old.sr is None
    amounts = {k: jnp.ones((2, 3)) for k in ("rain", "snow", "graupel", "ice")}
    assert precipitation_from_step(amounts, jnp.zeros((2, 3)), 18.).sr is None
    # Thompson's own SR (mp_gt_driver :1308): (snow + graupel + ice)/(RAINNCV + 1e-12).
    sr = precipitation_from_step(amounts, jnp.zeros((2, 3)), 18., with_sr=True, thompson_sr=True).sr
    np.testing.assert_allclose(np.asarray(sr), np.float32(3.) / (np.float32(4.) + np.float32(1e-12)))
    with pytest.raises(ValueError, match="SR"):
        precipitation_from_step(amounts, jnp.zeros((2, 3)), 18., with_sr=True)


@pytest.mark.parametrize("lsm,noahmp,run,want", [(2, False, True, "sr"), (7, False, True, "plain"),
                                                (4, True, True, "plain"), (3, False, True, "plain"),
                                                (8, False, True, "plain"), (0, False, True, None),
                                                (2, False, False, None)])
def test_land_precipitation_seed_selects_the_consumer(lsm, noahmp, run, want):
    from gpuwrf.runtime.operational_mode import _land_precipitation_seed
    nml = SimpleNamespace(run_physics=run, use_noahmp=noahmp, sf_surface_physics=lsm)
    assert _land_precipitation_seed(nml) == want


@pytest.mark.parametrize("code", [1, 13, 2, 6, 10, 40, 24, 97])
def test_generic_mp_public_precipitation_carries_the_scheme_sr(code):
    from gpuwrf.contracts.physics_interfaces import PhysicsTendency
    from gpuwrf.coupling.scan_adapters import _mp_public_precipitation
    inc = {k: jnp.full((6,), v) for k, v in dict(rain_acc=4., snow_acc=1., graupel_acc=.5, hail_acc=.25).items()}
    diag = {} if code in (1, 13) else {"sr": jnp.full((6,), .375)}
    out = _mp_public_precipitation(PhysicsTendency(accumulator_increments=inc, diagnostics=diag), 2, 3, code)
    # WRF: Kessler and SBU-YLin never set SR (Registry initial 0); others pass their own SR.
    np.testing.assert_array_equal(np.asarray(out["sr"]), 0. if code in (1, 13) else .375)


@pytest.mark.parametrize("lsm", [2, 7])
def test_actual_physics_step_hands_the_held_carry_to_the_lsm(monkeypatch, lsm):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    spec = importlib.util.spec_from_file_location("f24_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=3, nx=4, nz=8)
    state = f._state(grid)
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, sf_surface_physics=lsm, mp_physics=0, cu_physics=0,
                              bl_pbl_physics=0, sf_sfclay_physics=1, rad_rk_tendf=1, run_boundary=False)
    held = seed_precipitation(state, with_sr=lsm == 2)._replace(prcpnonc=jnp.full(state.t_skin.shape, 7e-4, jnp.float32))
    zero = jnp.zeros(state.t_skin.shape)
    rad = SimpleNamespace(gsw=zero, glw=zero)
    carry = initial_operational_carry(state).replace(
        noahmp_precipitation=held, noahclassic_land=object() if lsm == 2 else None,
        noahclassic_rad=(zero, zero, zero) if lsm == 2 else None, px_land=object() if lsm == 7 else None,
        px_rad=rad if lsm == 7 else None)
    monkeypatch.setattr(op, "_refresh_noahmp_rad", lambda *a, **k: (zero, zero, zero))
    seen = []

    def hook(st, land, static, dt, **kw):
        seen.append(kw.get("precipitation"))
        return st, land
    monkeypatch.setattr(op, "noahclassic_surface_step" if lsm == 2 else "pleim_xiu_surface_step", hook)
    op._physics_step_forcing(carry, nml, 18., run_radiation=False)
    assert len(seen) == 1 and seen[0] is held


def test_sr_channel_restart_roundtrip_and_pre_v6_noahclassic_refusal(tmp_path):
    import pickle
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    nml = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18, acoustic_substeps=4)
    shape = state.t_skin.shape
    land = NoahClassicLandState(**{name: jnp.zeros((*shape, 4) if name in ("stc", "smc", "sh2o", "smcrel") else shape)
                                   for name in NoahClassicLandState._fields})
    held = seed_precipitation(state, with_sr=True)._replace(
        prcpnonc=jnp.full(shape, .003, jnp.float32), sr=jnp.full(shape, .4375, jnp.float32))
    carry = initial_operational_carry(state).replace(noahclassic_land=land, noahmp_precipitation=held)
    pkl, nc = tmp_path / "sr.pkl", tmp_path / "sr.nc"
    write_restart(carry, nml, grid, 3, pkl)
    write_wrfrst_carry(carry, grid, {}, nc, valid_time="2026-07-25_18:00:54", run_start="2026-07-25_18:00:00",
                       step_index=3)
    for restored in (read_restart(pkl)[0], read_wrfrst_carry(nc)[0]):
        assert np.asarray(restored.noahmp_precipitation.sr).tobytes() == np.asarray(held.sr).tobytes()
    with pkl.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["format_version"] == 6
    payload["format_version"] = 5
    with pkl.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="SR channel.*E78|E78.*Noah-classic"):
        read_restart(pkl)
