"""w42-swdfid: history COSZEN is WRF's HELD radiation-call value, not the output instant (CPU, seconds).

WRF radiation_driver computes ``coszen`` once per radiation call with ``calc_coszen(..., xtime + radt*0.5, ...)`` and the
history writes that held field (module_radiation_driver.F; a call at the step starting at xtime, every
STEPRA = nint(radt*60/dt) steps from itimestep 1).  The port's radiation drivers hold the same value in
``carry.radiation_diagnostics.coszen``; the writer used to recompute COSZEN at the output instant instead (up to .055 off
at low sun).  Gates (oracle = original CPU-WRF 4.7.1 history, #146 WN3 0420 C arm, 72 h, fixtures/coszen_held_0420_v1.npz,
d01 dt 54 s with non-aligned outputs, d02 18 s, d03 6 s; builder proofs/w42_swdfid/build_coszen_fixture.py):
  O  the port's held coszen (release REAL solar geometry at the WRF call schedule, solar time = call + radt/2) ==
     CPU-WRF history COSZEN on all 72 forecast frames (lead 1-72 h) of each domain (every 4th cell; <= 1e-6); the
     output-instant value misses by > 1e-2;
  S  the native RRTMG driver hands the radiation solve exactly that solar time (spy on the real _refresh_rrtmg_driver);
  W  the real nested writer emits the carry's held value (written file == CPU-WRF at real d03 cells), keeps 0 at lead
     zero and the output-instant fallback for carries without held radiation diagnostics.
E200: every candidate, control and reference value is checked finite before it enters a max or a comparison
(``finite_max_abs``); NaN/+Inf/-Inf regressions poison the held geometry, the output-instant control and the CPU-WRF
reference (cr-w42-swdfid VERDICT 4fb6bacd: a held-NaN mutant passed the Python-max oracle).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.coupling import physics_couplers as pc

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "coszen_held_0420_v1.npz"
FIXTURE_SHA = "07ab06eeb5784228362abff3f9cb3ecdf43e795959c8f1145ad4a5058dd7c3a1"
RUN_START = datetime(2026, 4, 21, tzinfo=timezone.utc)  # 20260420_18z case: model start 2026-04-21_00 (JULDAY 111)


@pytest.fixture(scope="module")
def fx():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    with np.load(FIXTURE) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def wrf_held_call(lead_s: float, dt: float, radt_min: float) -> float:
    """Start time [s] of the last radiation call at or before the history write at ``lead_s`` (end of own step N)."""

    step = int(round(lead_s / dt))
    cadence = int(round(radt_min * 60.0 / dt))  # WRF STEPRA = nint(radt*60/dt)
    return ((step - 1) // cadence) * cadence * dt  # last itimestep s with mod(s-1, STEPRA) == 0, call at (s-1)*dt


def held_coszen(lat, lon, call_s: float, radt_min: float, monkeypatch):
    monkeypatch.setattr(pc, "_SOLAR_JULIAN_WRF", True)  # release solar path (GPUWRF_FAST_DEFAULTS)
    geo = pc._compute_solar_geometry(jnp.asarray(lat, jnp.float32), jnp.asarray(lon, jnp.float32), RUN_START, call_s,
                                     solar_lead_seconds=call_s + 30.0 * radt_min)
    return np.asarray(geo.coszen, np.float32)


def assert_finite(values, label: str) -> None:
    assert np.all(np.isfinite(np.asarray(values))), f"non-finite {label}"  # E200: before any max or comparison


def finite_max_abs(candidate, reference, label: str) -> float:
    """max |candidate - reference| after rejecting non-finite values on either side (E200)."""

    assert_finite(candidate, f"{label} candidate")
    assert_finite(reference, f"{label} reference")
    return float(np.max(np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64))))


def domain_worst(fx, dom: str, monkeypatch) -> tuple[float, float]:
    """(worst held, worst output-instant) |COSZEN - CPU-WRF| over the 72 forecast frames of one domain."""

    lat, lon, ref = fx[f"{dom}_xlat"], fx[f"{dom}_xlong"], fx[f"{dom}_coszen"]
    assert_finite(ref, f"{dom} CPU-WRF reference")
    dt, radt = float(fx[f"{dom}_dt_s"]), float(fx[f"{dom}_radt_min"])
    worst_held = worst_instant = 0.0
    for frame in range(1, ref.shape[0]):
        lead = float(fx[f"{dom}_xtime_min"][frame]) * 60.0
        held = held_coszen(lat, lon, wrf_held_call(lead, dt, radt), radt, monkeypatch)
        worst_held = max(worst_held, finite_max_abs(held, ref[frame], f"{dom} held"))
        instant = np.asarray(pc._compute_coszen(lat, lon, RUN_START, lead_seconds=lead), np.float32)
        worst_instant = max(worst_instant, finite_max_abs(instant, ref[frame], f"{dom} output-instant control"))
    return worst_held, worst_instant


@pytest.mark.parametrize("dom", ["d01", "d02", "d03"])
def test_port_held_coszen_is_cpu_wrf_history(fx, dom, monkeypatch):
    assert np.all(fx[f"{dom}_coszen"][0] == 0.0)  # WRF lead-zero history precedes the first radiation call
    worst_held, worst_instant = domain_worst(fx, dom, monkeypatch)
    assert worst_held <= 1.0e-6, worst_held
    assert worst_instant > 1.0e-2, worst_instant  # the old output-instant history


BAD = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf}


@pytest.mark.parametrize("side", ["candidate", "reference"])
@pytest.mark.parametrize("bad", list(BAD))
def test_finite_max_abs_rejects_nonfinite(bad, side):
    good = np.full((3, 4), 0.5, np.float32)
    poisoned = good.copy()
    poisoned[1, 2] = BAD[bad]
    args = (poisoned, good) if side == "candidate" else (good, poisoned)
    with pytest.raises(AssertionError, match=f"non-finite probe {side}"):
        finite_max_abs(*args, "probe")


@pytest.mark.parametrize("bad", list(BAD))
def test_oracle_rejects_nonfinite_held_geometry(fx, monkeypatch, bad):
    """The critic's held-NaN mutant (one poisoned cell here) and its +-Inf siblings fail the domain oracle."""

    original = pc._compute_solar_geometry

    def poisoned(*args, **kwargs):
        geo = original(*args, **kwargs)
        return geo._replace(coszen=jnp.asarray(geo.coszen).at[0, 0].set(BAD[bad]))

    monkeypatch.setattr(pc, "_compute_solar_geometry", poisoned)
    with pytest.raises(AssertionError, match="non-finite d03 held candidate"):
        domain_worst(fx, "d03", monkeypatch)


@pytest.mark.parametrize("bad", list(BAD))
def test_oracle_rejects_nonfinite_instant_control(fx, monkeypatch, bad):
    original = pc._compute_coszen

    def poisoned(*args, **kwargs):
        out = np.array(original(*args, **kwargs), np.float32)
        out[0, 0] = BAD[bad]
        return out

    monkeypatch.setattr(pc, "_compute_coszen", poisoned)
    with pytest.raises(AssertionError, match="non-finite d03 output-instant control candidate"):
        domain_worst(fx, "d03", monkeypatch)


@pytest.mark.parametrize("bad", list(BAD))
def test_oracle_rejects_nonfinite_reference(fx, monkeypatch, bad):
    ref = fx["d03_coszen"].copy()
    ref[40, 3, 4] = BAD[bad]
    with pytest.raises(AssertionError, match="non-finite d03 CPU-WRF reference"):
        domain_worst(dict(fx, d03_coszen=ref), "d03", monkeypatch)


def test_driver_solar_time_is_call_plus_half_radt(monkeypatch):
    """S: the real native driver passes lead_seconds = call time and solar_lead_seconds = call + radt/2."""

    from gpuwrf.runtime import operational_mode as om

    seen = {}

    class Captured(Exception):
        pass

    def spy(state, grid, **kw):
        seen.update(kw)
        raise Captured

    monkeypatch.setattr(om, "rrtmg_theta_tendency", spy)
    nl = SimpleNamespace(ra_sw_physics=4, ra_lw_physics=4, radiation_interval_s=1800.0, dt_s=54.0,
                         radiation_cadence_steps=33, aer_opt=0, boundary_config=SimpleNamespace(force_geopotential=True),
                         mp_physics=8, cu_physics=1, sf_urban_physics=0, sf_surface_physics=4, topo_shading=1, slope_rad=1,
                         topo_shadow_length_m=25000.0, time_utc=RUN_START, radiation_static=None, grid=None, icloud=1,
                         bl_pbl_physics=5, use_mp_re=1)
    carry = SimpleNamespace(state=SimpleNamespace(), noahmp_land=None, solar_interpolation=None, census=None,
                            rthraten=None, radiation_diagnostics=SimpleNamespace(
                                shadow_height_m=jnp.zeros((1, 1), jnp.float32)))
    with pytest.raises(Captured):
        om._refresh_rrtmg_driver(carry, nl, 3564.0, True, None)
    assert float(seen["lead_seconds"]) == 3564.0
    assert float(seen["solar_lead_seconds"]) == 3564.0 + 900.0


def _writer(tmp_path, monkeypatch, lat, lon, dt, cadence):
    from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
    from gpuwrf.io import gen2_accessor, wrf_history_metadata
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    try:
        from test_m7_netcdf_writer import authenticated_source_grid, synthetic_case
        from test_initial_history_wrf_t0 import _wrfinput
    finally:
        sys.path.pop(0)
    state, grid, _ = synthetic_case()
    assert lat.shape == (grid.ny, grid.nx)
    source = authenticated_source_grid(grid, "d01")
    monkeypatch.setattr(gen2_accessor, "Gen2Run", lambda _: SimpleNamespace(namelist={}, grid=lambda _: source))
    monkeypatch.setattr(daily, "_load_static_latlon_writer_diagnostics",
                        lambda *a, **kw: ({"XLAT": lat, "XLONG": lon}, {}))
    monkeypatch.setattr(wrf_history_metadata, "wrf_history_global_attributes", lambda *a, **kw: {})
    writer = nested._PerDomainWrfoutWriter(
        output_dir=tmp_path / "out", input_dir=tmp_path, run_start=RUN_START,
        bundles={"d01": SimpleNamespace(grid=grid, namelist=SimpleNamespace(use_noahmp=False, sf_sfclay_physics=5))},
        output_cadence_steps={"d01": cadence}, dt_by_domain={"d01": dt})
    writer._surface_diagnostics_for_output = lambda *a, **kw: {}
    _wrfinput(tmp_path / "wrfinput_d01", grid.ny, grid.nx)
    return writer, state


@pytest.mark.parametrize("frame", [8, 36, 65], ids=["0421_08z", "0422_12z", "0423_17z"])
def test_nested_writer_emits_held_coszen(fx, tmp_path, monkeypatch, frame):
    """W: written COSZEN == CPU-WRF history at real d03 cells (held carry value), not the output instant."""

    lat, lon = fx["d03_xlat"][10:14, 5:10], fx["d03_xlong"][10:14, 5:10]  # 4 x 5 real d03 land/sea cells
    ref = fx["d03_coszen"][frame, 10:14, 5:10]
    dt, radt = float(fx["d03_dt_s"]), float(fx["d03_radt_min"])
    lead = float(fx["d03_xtime_min"][frame]) * 60.0
    writer, state = _writer(tmp_path, monkeypatch, lat, lon, dt, 600)
    held = held_coszen(lat, lon, wrf_held_call(lead, dt, radt), radt, monkeypatch)
    carry = SimpleNamespace(state=state, radiation_diagnostics=SimpleNamespace(coszen=jnp.asarray(held)))
    result = writer("d01", int(round(lead / dt)), carry)
    with Dataset(result["wrfout"]) as nc:
        written = np.asarray(nc["COSZEN"][0])
    assert_finite(written, "written COSZEN")
    assert_finite(held, "held carry COSZEN")
    np.testing.assert_array_equal(written, held)  # the held carry value, bit for bit
    assert finite_max_abs(written, ref, "written vs CPU-WRF") <= 1.0e-6
    instant = np.asarray(pc._compute_coszen(lat, lon, RUN_START, lead_seconds=lead), np.float32)
    assert finite_max_abs(instant, ref, "output-instant control") > 1.0e-3  # the frame discriminates the old behaviour


def test_nested_writer_fallbacks_unchanged(fx, tmp_path, monkeypatch):
    """Lead zero writes 0 (even with a held value); a carry without held radiation keeps the output-instant value."""

    lat, lon = fx["d03_xlat"][10:14, 5:10], fx["d03_xlong"][10:14, 5:10]
    writer, state = _writer(tmp_path, monkeypatch, lat, lon, 6.0, 600)
    carry = SimpleNamespace(state=state, radiation_diagnostics=SimpleNamespace(coszen=jnp.full(lat.shape, 0.5, jnp.float32)))
    with Dataset(writer("d01", 0, carry)["wrfout"]) as nc:
        assert np.all(np.asarray(nc["COSZEN"][0]) == 0.0)
    with Dataset(writer("d01", 6000, state)["wrfout"]) as nc:
        written = np.asarray(nc["COSZEN"][0])
    instant = np.asarray(pc._compute_coszen(lat, lon, RUN_START, lead_seconds=36000.0), np.float32)
    assert finite_max_abs(written, instant, "fallback output-instant") <= 1e-7  # assert_allclose would accept NaN==NaN
