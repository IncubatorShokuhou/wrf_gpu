"""t-cfix-pblmix F17 (class B): GBM (bl_pbl_physics=12) consumes the held RTHRATEN.

Oracle: UNMODIFIED WRF 4.7.1 module_bl_gbmpbl.F GBMPBL (REAL4, -O0 no contraction,
compiler-defined zero init as proofs/t_c02nan/build_gbm_oracle.py) on 16 REAL c08
CPU-WRF columns with real TKE_PBL and a CONSTRUCTED cloud-top RTHRATEN profile
(history has no RTHRATEN; proofs/t_cfix_pblmix/gbm_columns.py).  WRF uses it in
the cloud-top TKE buoyancy source (gbmpbl.F:756-761) and the multi-layer PBL-height
search radnnll (:969-985); the port now carries WRF's multi-layer pblhgt/my.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FX = json.loads((ROOT / "proofs/t_cfix_pblmix/gbm_rthraten_fixture.json").read_text())
C, S = FX["inputs"]["columns"], FX["inputs"]["scalars"]
FIELDS = (("u", "u"), ("v", "v"), ("theta", "theta"), ("qv", "qv"), ("qc", "qc"), ("tke", "tke"),
          ("kth", "kth"), ("kzm", "kzm"), ("el_pbl", "el_pbl"))


def _port(rad=True, double=False):
    from gpuwrf.physics.bl_gbm import gbm_columns
    dtype = np.float64 if double else np.float32
    col = lambda n: jnp.asarray(np.asarray(C[n], dtype))  # noqa: E731
    one = lambda n: jnp.asarray(np.asarray(S[n], dtype))  # noqa: E731
    with jax.enable_x64(double):
        out = jax.jit(lambda: gbm_columns(
            col("U"), col("V"), col("T"), col("QV"), col("QC"), col("P"), col("PI"), col("DZ"), col("TKE"),
            psfc=one("PSFC"), znt=one("ZNT"), ust=one("UST"), hfx=one("HFX"), qfx=one("QFX"), tsk=one("TSK"),
            gz1oz0=one("GZ1OZ0"), wspd=one("WSPD"), br=one("BR"), psim=one("PSIM"), psih=one("PSIH"), dt=18.0,
            xland=one("XLAND"), rthraten=col("RAD") if rad else None))()
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def _err(out, ref):
    """Field -> max column-normalized error (non-finite candidate/reference rejected)."""
    bad = {}
    for port, wrf in FIELDS:
        got, want = out[port], np.asarray(ref[wrf], np.float64)
        assert np.isfinite(got).all() and np.isfinite(want).all(), port
        scale = np.maximum(np.abs(want).max(axis=1, keepdims=True), 1e-30)
        bad[wrf] = np.abs(got - want) / scale
    return bad


def test_fixture_exercises_rthraten_and_layers():
    rad, norad = FX["rad_r8"], FX["norad"]
    assert np.asarray(C["RAD"]).min() < -1e-4 and len(FX["inputs"]["j"]) >= 16
    assert np.abs(np.asarray(FX["rad"]["tke"]) - np.asarray(norad["tke"])).max() > 1e-3
    assert np.abs(np.asarray(FX["rad"]["kth"]) - np.asarray(norad["kth"])).max() > 1e-2
    assert len(set(np.asarray(rad["kpbl"]).tolist())) >= 4   # several PBL-top layer indices


def test_double_port_matches_pristine_gbmpbl_transcription():
    ref = FX["rad_r8"]
    out = _port(rad=True, double=True)
    worst = {k: float(v.max()) for k, v in _err(out, ref).items()}
    assert max(worst.values()) < 1e-9, worst
    np.testing.assert_allclose(out["pblh"], ref["pblh"], rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(out["kpbl"], ref["kpbl"])


@pytest.mark.parametrize("rad", [True, False], ids=["rthraten", "zero"])
def test_real_port_inside_wrf_real_double_spread(rad):
    # Operational REAL path vs the REAL4 pristine build, bounded by 4x the WRF build's own
    # REAL-vs-DOUBLE spread (independent of the port) + a 1e-6 column-relative floor.
    # Measured worst ratio 2.1x (TKE, the semi-implicit solve near its 1e-3 floor);
    # every other field <= 2x. The DOUBLE gate above proves the transcription.
    ref = FX["rad" if rad else "norad"]
    spread = _err({p: np.asarray(FX["rad"][w], np.float64) for p, w in FIELDS}, FX["rad_r8"])
    err = _err(_port(rad), ref)
    for name, e in err.items():
        assert e.max() <= 4.0 * spread[name].max() + 1e-6, (name, float(e.max()), float(spread[name].max()))
    np.testing.assert_array_equal(_port(rad)["kpbl"], ref["kpbl"])
    np.testing.assert_allclose(_port(rad)["pblh"], ref["pblh"], rtol=1e-5, atol=1e-2)


def test_dropping_rthraten_misses_pristine():
    worst = {k: float(v.max()) for k, v in _err(_port(rad=False, double=True), FX["rad_r8"]).items()}
    assert worst["tke"] > 1e-5 or worst["kth"] > 1e-5, worst


def test_operational_gbm_receives_held_rthraten(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    ns = _namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=12, sf_sfclay_physics=1,
                   sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=1)
    carry = initial_operational_carry(state)
    carry = carry.replace(rthraten=jnp.full_like(carry.rthraten, -3.0e-5))
    seen = {}

    def pbl(st, dt, grid, **kw):
        seen.update(kw)
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        return st, {key: zero for key in ("u", "v", "theta", "qv", "qc")}

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 12, pbl)
    om._physics_step_forcing(carry, ns, 0., run_radiation=False, first_timestep=True)
    assert seen.get("rthraten") is not None
    np.testing.assert_array_equal(np.asarray(seen["rthraten"]), np.asarray(carry.rthraten))


def _gbm_radiation_case(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    ns = _namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=12, sf_sfclay_physics=1,
                   sf_surface_physics=0, use_noahmp=False, ra_lw_physics=1, ra_sw_physics=1, rad_rk_tendf=1)
    om._resolve_operational_suite(ns)
    entry = _base_state(grid)
    carry = initial_operational_carry(entry).replace(rthraten=jnp.full_like(entry.theta, -3.0e-5))
    assert carry.radiation_diagnostics is None   # legacy RTHRATEN refresh path
    log = {"calls": [], "rad_inputs": []}

    def radiation(st, *a, **kw):
        log["calls"].append("radiation")
        log["rad_inputs"].append(st)
        return jnp.full_like(st.theta, -3.5e-5)

    def pbl(st, dt, grid, **kw):
        log["calls"].append("pbl")
        log["pbl_rthraten"] = np.asarray(kw["rthraten"])
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        return st, {key: zero for key in ("u", "v", "theta", "qv", "qc")}

    monkeypatch.setattr(om, "rrtm_lw_theta_tendency", radiation)
    monkeypatch.setattr(om, "dudhia_sw_theta_tendency", radiation)
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 12, pbl)
    return om, ns, entry, carry, log


def test_radiation_step_gbm_reads_this_steps_rthraten_from_entry_state(monkeypatch):
    """WRF first_rk_step_part1: radiation_driver (:264) before surface/PBL (:1113), so GBMPBL
    (RTHRATEN=grid%rthraten) sees THIS step's refresh; one SW + one LW solve on the step-entry
    state, reused as the held rate."""
    om, ns, entry, carry, log = _gbm_radiation_case(monkeypatch)
    forcing = om._physics_step_forcing(carry, ns, 0., run_radiation=True, first_timestep=True)
    assert log["calls"] == ["radiation", "radiation", "pbl"], log["calls"]
    current = np.asarray(forcing.carry.rthraten)
    np.testing.assert_allclose(current, -7.0e-5, rtol=1e-6, atol=0.)   # SW + LW sentinels, not the held -3e-5
    np.testing.assert_array_equal(log["pbl_rthraten"], current)
    for st in log["rad_inputs"]:   # entry state, not the post-surface-layer state
        for name in ("theta", "qv", "ustar", "hfx", "t_skin"):
            np.testing.assert_array_equal(np.asarray(getattr(st, name)), np.asarray(getattr(entry, name)), err_msg=name)


def test_held_step_gbm_reads_the_held_rthraten(monkeypatch):
    om, ns, entry, carry, log = _gbm_radiation_case(monkeypatch)
    forcing = om._physics_step_forcing(carry, ns, 18., run_radiation=False)
    assert log["calls"] == ["pbl"], log["calls"]
    np.testing.assert_array_equal(log["pbl_rthraten"], np.asarray(carry.rthraten))
    np.testing.assert_array_equal(np.asarray(forcing.carry.rthraten), np.asarray(carry.rthraten))
