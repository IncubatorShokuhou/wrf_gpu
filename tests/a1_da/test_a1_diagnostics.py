"""Tests for gpuwrf.assimilation.letkf.diagnostics (lane a1-obs). CPU only."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import diagnostics as dg
from gpuwrf.assimilation.letkf import operators as ops

# ALISIOS read conditions (2026-10-17 01:14Z): obs are read ONLY from the guard_call-admitted copies in guarded_v1
GUARDED = Path("<DATA_ROOT>/wrf_gpu2/scratch/a1-da/obs/guarded_v1")
IGRA = GUARDED / "igra2_SPM00060018_admitted.txt"            # 12 admitted launches (02-27/28, 05-02..05)
DERIVED = GUARDED / "derived_twi_admitted.parquet"          # shared-truth rows of exactly those launches


def _same(a, b):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if isinstance(a, (bool, np.bool_)):
        return bool(a) == bool(b)
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))


@pytest.mark.skipif(not (IGRA.exists() and DERIVED.exists()), reason="guarded sonde copies absent")
def test_soundings_match_shared_derived_parquet_admitted_launches():
    """Independent oracle: the shared ALISIOS parquet (derive_60018.py) for every guard-admitted launch.

    (A full-archive run before the ALISIOS read conditions gave 16,797/16,797 identical launches; it touched
    guard-DENY days and is disclosed in proofs/a1_da/obs/GUARDED_READS.json - never re-run it unguarded.)
    """
    pd = pytest.importorskip("pandas")
    ref = pd.read_parquet(DERIVED).set_index("time")
    n, found, mismatches = 0, 0, []
    for head, levels in dg.iter_igra2_soundings(IGRA, "SPM00060018"):
        got = dg._twi(dg.sounding_profile(levels))
        row = ref.loc[head["time"]]
        for key in dg.TWI_FIELDS:
            exp = row[key] if key in row.index else np.nan
            exp = None if (not isinstance(exp, (bool, np.bool_)) and pd.isna(exp)) else exp
            if not _same(got.get(key), exp):
                mismatches.append((str(head["time"]), key, got.get(key), exp))
        n += 1
        found += bool(got.get("twi_found"))
    assert n == 12 and found >= 8
    assert not mismatches, mismatches[:10]


@pytest.mark.skipif(not (IGRA.exists() and DERIVED.exists()), reason="guarded sonde copies absent")
def test_probe_launch_2026_05_04_12z():
    pd = pytest.importorskip("pandas")
    got = dg.sounding_inversion(IGRA, "SPM00060018", "2026-05-04T12:00Z")
    ref = pd.read_parquet(DERIVED)
    row = ref[ref.time == pd.Timestamp("2026-05-04 12:00")].iloc[0]
    assert got["time"] == str(row.time) and got["twi_found"]
    for key in ("twi_base_m", "twi_top_m", "twi_strength_k", "moist_base_m", "moist_top_m", "rh_base_pct"):
        assert got[key] == pytest.approx(row[key], abs=1e-9)
    assert got["twi_base_m"] == pytest.approx(558.478755, abs=1e-5) and got["twi_strength_k"] == pytest.approx(1.8)
    # the generic column entry point gives the same answer on the sonde's own profile
    for head, levels in dg.iter_igra2_soundings(IGRA, "SPM00060018", "2026-05-04 12:00", "2026-05-04 12:00"):
        prof = dg.sounding_profile(levels)
        again = dg.detect_inversion(prof.z_m.values, prof.t_c.values, prof.rh_pct.values)
        assert {k: again.get(k) for k in dg.TWI_FIELDS} == {k: got.get(k) for k in dg.TWI_FIELDS}


def test_rule_edges_on_synthetic_columns():
    z = np.array([100.0, 250.0, 400.0, 500.0, 550.0, 700.0, 900.0, 1200.0, 2000.0, 3000.0])
    t = np.array([20.0, 19.0, 18.0, 18.0, 19.0, 21.0, 20.0, 18.0, 12.0, 6.0])
    rh = np.array([80.0, 92.0, 95.0, 96.0, 70.0, 40.0, 30.0, 30.0, 30.0, 30.0])
    r = dg.detect_inversion(z, t, rh)
    # warming steps 400->500 (isothermal), 500->550, 550->700: one layer base 400, top 700, strength 3 K
    assert r["twi_found"] and r["twi_base_m"] == 400.0 and r["twi_top_m"] == 700.0 and r["twi_strength_k"] == 3.0
    assert r["moist_base_m"] == 250.0 and r["moist_top_m"] == 400.0 and r["max_rh_below_twi"] == 95.0
    # base below 300 m does not count: shift everything down by 200 m -> layer base 200 m rejected
    assert not dg.detect_inversion(z - 200.0, t, rh)["twi_found"]
    assert not dg.detect_inversion(z - 120.0, t, rh)["twi_found"]      # base 280 m < 300 m
    assert dg.detect_inversion(z - 100.0, t, rh)["twi_base_m"] == 300.0  # base exactly 300 m is admitted
    # a cooling gap < 100 m merges two warming layers; >= 100 m splits
    z2 = np.array([100.0, 500.0, 540.0, 600.0, 650.0, 700.0, 2000.0])
    t2 = np.array([20.0, 16.0, 16.5, 16.2, 17.0, 17.5, 10.0])
    merged = dg.detect_inversion(z2, t2, np.full(7, 50.0))
    assert merged["twi_base_m"] == 500.0 and merged["twi_top_m"] == 700.0
    z3 = np.array([100.0, 500.0, 560.0, 700.0, 760.0, 800.0, 2000.0])
    t3 = np.array([20.0, 16.0, 16.5, 16.2, 17.0, 17.5, 10.0])
    split = dg.detect_inversion(z3, t3, np.full(7, 50.0))
    assert split["twi_base_m"] == 500.0 and split["twi_top_m"] == 560.0
    # thin layer (< 50 m) skipped, next qualifying layer chosen
    z4 = np.array([100.0, 500.0, 530.0, 700.0, 900.0, 1000.0, 2000.0])
    t4 = np.array([20.0, 16.0, 16.5, 15.0, 13.0, 14.0, 8.0])
    assert dg.detect_inversion(z4, t4, np.full(7, 50.0))["twi_base_m"] == 900.0
    # duplicate heights: the FIRST level at a height is kept (derive_60018 line 50 drop_duplicates)
    z5 = np.array([100.0, 400.0, 500.0, 500.0, 560.0, 1500.0, 2000.0])
    t5 = np.array([20.0, 17.0, 16.0, 18.0, 18.5, 12.0, 8.0])
    dup = dg.detect_inversion(z5, t5, np.full(7, 50.0))
    assert dup["twi_base_m"] == 500.0 and dup["twi_strength_k"] == pytest.approx(2.5) and dup["n_levels_0_5km"] == 6
    # fewer than 5 levels below 5 km -> no result
    assert dg.detect_inversion(z[:4], t[:4], rh[:4]) == {}


def test_model_rh_matches_dewpoint_rh():
    t_c = np.array([25.0, 10.0, -5.0])
    dpd = np.array([1.0, 6.0, 12.0])
    p = np.array([1.0e5, 8.5e4, 6.0e4])
    td = t_c - dpd
    e = dg.MAGNUS_E0_PA * np.exp(dg.MAGNUS_A * td / (dg.MAGNUS_B + td))
    qv = dg.EP_2 * e / (p - e)
    assert np.allclose(dg.rh_from_mixing_ratio(qv, t_c, p), dg.rh_from_dewpoint_depression(t_c, dpd), atol=1e-10)


def _column_frames(nmem=3, ny=6, nx=7):
    """Horizontally uniform columns with an inversion 800 -> 1000 m and a saturated layer 400-700 m."""
    zw = np.array([0.0, 100.0, 300.0, 500.0, 700.0, 900.0, 1100.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
    zm = 0.5 * (zw[1:] + zw[:-1])          # 50 200 400 600 800 1000 1300 1750 2500 3500 4500 5500
    t_c = np.array([22.0, 21.0, 19.5, 18.5, 17.0, 21.0, 20.0, 17.5, 12.0, 5.0, -2.0, -9.0])
    rh = np.array([70.0, 85.0, 95.0, 92.0, 80.0, 30.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    p = 1.0e5 * np.exp(-zm / 8000.0)
    e = rh / 100 * dg.MAGNUS_E0_PA * np.exp(dg.MAGNUS_A * t_c / (dg.MAGNUS_B + t_c))
    qv = dg.EP_2 * e / (p - e)
    theta = (t_c + 273.15) * (ops.P0 / p) ** ops.RCP - 300.0
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    shape3 = (nmem, len(zm), ny, nx)
    frames = {"QVAPOR": np.broadcast_to(qv[None, :, None, None], shape3).copy(),
              "P": np.broadcast_to(0.2 * p[None, :, None, None], shape3).copy(),
              "PB": np.broadcast_to(0.8 * p[:, None, None], (len(zm), ny, nx)).copy(),
              "PH": np.zeros((nmem, len(zw), ny, nx)),
              "PHB": np.broadcast_to(dg.G_WRF * zw[:, None, None], (len(zw), ny, nx)).copy()}
    frames["T"] = np.broadcast_to(theta[None, :, None, None], shape3).copy()
    geo = {"XLAT": 28.0 + 0.1 * j, "XLONG": -17.0 + 0.1 * i, "HGT": np.full((ny, nx), 25.0)}
    return frames, geo, zm, t_c, rh


def test_model_inversion_on_known_column():
    frames, geo, zm, t_c, rh = _column_frames()
    cols = dg.model_columns(frames, geo, 28.23, -16.67)
    assert np.allclose(cols["z_msl"][0], zm) and np.allclose(cols["z_agl"][0], zm - 25.0)
    assert np.allclose(cols["t_c"][1], t_c, atol=1e-9) and np.allclose(cols["rh_pct"][2], rh, atol=1e-9)
    r = dg.model_inversion(frames, geo, 28.23, -16.67)
    m = r["member_mean_column"]
    assert m["twi_found"] and m["twi_base_m"] == pytest.approx(800.0) and m["twi_top_m"] == pytest.approx(1000.0)
    assert m["twi_strength_k"] == pytest.approx(4.0, abs=1e-9)
    assert m["moist_base_m"] == pytest.approx(400.0) and m["moist_top_m"] == pytest.approx(600.0)
    assert r["summary"]["found_fraction"] == 1.0 and r["summary"]["twi_base_m"]["median"] == pytest.approx(800.0)
    # member 0 gets a plain -6.5 K/km lapse profile (no inversion)
    p = 1.0e5 * np.exp(-zm / 8000.0)
    frames["T"] = frames["T"].copy()
    frames["T"][0] = (((22.0 - 6.5e-3 * zm) + 273.15) * (ops.P0 / p) ** ops.RCP - 300.0)[:, None, None]
    r2 = dg.model_inversion(frames, geo, 28.23, -16.67)
    assert r2["members"][0].get("twi_found") is False and r2["summary"]["found_fraction"] == pytest.approx(2 / 3)
