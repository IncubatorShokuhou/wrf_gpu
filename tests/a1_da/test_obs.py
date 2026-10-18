"""Tests for gpuwrf.assimilation.letkf.obs (lane a1-obs). CPU only, NumPy fp64."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import obs as ob

# ALISIOS read conditions (2026-10-17 01:14Z): obs are read ONLY from the guard_call-admitted copies in guarded_v1
GUARDED = Path("<DATA_ROOT>/wrf_gpu2/scratch/a1-da/obs/guarded_v1")
IGRA = GUARDED / "igra2_SPM00060018_admitted.txt"
MSG_0503 = Path("<DATA_ROOT>/alisios/cloud/msg_clm_corrected_mayaug2026_v1/days/2026-05-03/corrected_msg_clm.npz")


# ---------------------------------------------------------------- thermo / units
def test_saturation_and_mixing_ratio_reference_values():
    # WRF/Bolton at 20 C: 611.2 exp(17.67*20/263.5) = 2336.95 Pa
    assert abs(ob.sat_vapor_pressure(293.15) - 2336.95) < 0.01
    q = ob.qv_from_rh(1.0, 293.15, 1.0e5)
    assert abs(q - 0.621750 * 2336.95 / (1.0e5 - 2336.95)) < 2e-7
    # dewpoint == temperature -> saturation; RH round trip
    assert abs(ob.qv_from_dewpoint(293.15, 9.0e4) - ob.qv_saturation(293.15, 9.0e4)) < 1e-15
    rh = np.array([0.05, 0.4, 0.93])
    t = np.array([250.0, 280.0, 300.0])
    p = np.array([3.0e4, 8.5e4, 1.01e5])
    assert np.allclose(ob.rh_from_qv(ob.qv_from_rh(rh, t, p), t, p), rh, rtol=0, atol=1e-13)
    assert abs(ob.std_atmosphere_pressure(0.0) - 101325.0) < 1e-9
    assert abs(ob.std_atmosphere_pressure(1000.0) - 89874.6) < 1.0


def test_to_datetime64_accepts_posix_seconds_and_aware_datetimes():
    from datetime import datetime, timezone

    aware = datetime(2026, 5, 4, 12, tzinfo=timezone.utc)
    assert ob.to_datetime64(aware.timestamp()) == np.datetime64("2026-05-04T12:00:00")
    assert ob.to_datetime64(aware) == np.datetime64("2026-05-04T12:00:00")
    assert ob.to_datetime64("2026-05-04T13:00+01:00") == np.datetime64("2026-05-04T12:00:00")


def test_wind_components_meteorological_convention():
    u, v = ob.wind_components([10.0, 10.0, np.sqrt(2.0) * 5, 0.0], [0.0, 90.0, 225.0, 270.0])
    assert np.allclose(u, [0.0, -10.0, 5.0, 0.0], atol=1e-12)
    assert np.allclose(v, [-10.0, 0.0, 5.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------- ObsBatch
def small_batch():
    return ob.ObsBatch.from_columns(var=np.array([ob.T, ob.QV, ob.T2, ob.CLDMASK], np.int16),
                                    value=np.array([280.0, 0.005, 290.0, 0.4]), error=np.array([1, 1e-3, 1.75, 0.3]),
                                    lat=np.array([28.3, 28.3, 28.4, 28.0]), lon=np.array([-16.4, -16.4, 343.7, -16.0]),
                                    p_pa=np.array([8.5e4, 8.5e4, np.nan, np.nan]), station="x")


def test_obsbatch_defaults_validation_and_roundtrip():
    b = small_batch()
    assert b.kind.tolist() == [ob.SONDE, ob.SONDE, ob.SURFACE, ob.CLOUD]
    assert b.lon[2] == pytest.approx(-16.3)          # wrapped to [-180, 180)
    assert (b.lh_scale == 1.0).all() and (b.lv_scale == 1.0).all() and (b.qc == 0).all()
    again = ob.ObsBatch.from_dict(json.loads(json.dumps(b.to_dict())))
    for name in ("kind", "var", "value", "error", "lat", "lon", "qc"):
        assert np.array_equal(getattr(again, name), getattr(b, name))
    assert np.isnan(again.p_pa[2]) and again.p_pa[0] == 8.5e4
    with pytest.raises(ValueError, match="kind does not match"):
        ob.ObsBatch.from_columns(var=[ob.T], value=[1.0], lat=[0.0], lon=[0.0], kind=[ob.SURFACE])
    with pytest.raises(ValueError, match="finite values"):
        ob.ObsBatch.from_columns(var=[ob.T], value=[np.nan], lat=[0.0], lon=[0.0])
    flagged = ob.ObsBatch.from_columns(var=[ob.T], value=[np.nan], lat=[0.0], lon=[0.0], qc=[ob.QC_SOURCE])
    assert flagged.obs_id[0].startswith("obs/T/")
    assert len(flagged) == 1
    sub = b.subset(b.kind == ob.SONDE)
    assert len(sub) == 2 and len(ob.concat([b, sub])) == 6 and len(ob.ObsBatch.concat([b, sub])) == 6
    # DESIGN_v2 columns: sigma alias, auto ids, site defaults to station, role/averaging defaults
    assert np.array_equal(b.sigma, b.error) and len(set(b.obs_id)) == 4 and (b.site == "x").all()
    assert (b.role == "assimilate").all() and (b.averaging == "instant").all()
    w = b.with_role(b.kind == ob.SURFACE, "withhold")
    assert w.assimilate.tolist() == [True, True, False, True]
    with pytest.raises(ValueError, match="role"):
        b.replace(role=np.array(["maybe"] * 4, dtype=object))
    with pytest.raises(ValueError, match="sigma"):
        ob.ObsBatch.from_columns(var=[ob.T], value=[280.0], lat=[0.0], lon=[0.0])  # no error given
    timed = b.replace(time=np.array(["2026-05-04T12:00"] * 4, dtype="datetime64[s]"))
    back = ob.ObsBatch.from_dict(json.loads(json.dumps(timed.to_dict())))
    assert (back.time == np.datetime64("2026-05-04T12:00:00")).all() and np.array_equal(back.obs_id, timed.obs_id)
    counts = b.with_qc(b.var == ob.T, ob.QC_GROSS).counts()
    assert counts["SONDE/T"] == {"total": 1, "usable": 0, "stations": 1, "gross": 1}


def test_error_table_overrides_and_humidity_errors():
    table = ob.obs_error_table({"SONDE": {"T": {"sigma": 1.4}}})
    assert ob.error_for(ob.SONDE, ob.T, table) == 1.4 and ob.DEFAULT_OBS_ERRORS["SONDE"]["T"]["sigma"] == 1.0
    with pytest.raises(ValueError):
        ob.error_for(ob.SURFACE, ob.Q2, table)
    sq = ob.error_for(ob.SURFACE, ob.Q2, table, t_k=293.15, p_pa=1e5)
    assert abs(sq - 0.15 * ob.qv_saturation(293.15, 1e5)) < 1e-15
    # dry/cold: floor applies
    assert ob.error_for(ob.SONDE, ob.QV, table, t_k=220.0, p_pa=2e4) == table["q_floor_kg_kg"]


def test_inflate_surface_errors_by_height_mismatch():
    b = ob.ObsBatch.from_columns(var=np.array([ob.T2, ob.U10, ob.T], np.int16), value=np.array([290.0, 1.0, 280.0]),
                                 error=np.array([1.75, 2.5, 1.0]), lat=0.0, lon=0.0, p_pa=np.array([np.nan, np.nan, 8e4]))
    out = ob.inflate_surface_errors(b, np.array([100.0, -200.0, 500.0]))
    assert out.error[0] == pytest.approx(np.hypot(1.75, 0.5)) and out.error[1] == pytest.approx(np.hypot(2.5, 1.0))
    assert out.error[2] == 1.0                      # sondes untouched
    same = ob.inflate_surface_errors(b, np.array([0.0, np.nan, 0.0]))
    assert np.array_equal(same.error, b.error)


# --------------------------------------------------------------------------- QC
def test_gross_check_flags_outlier_and_nonfinite():
    b = ob.ObsBatch.from_columns(var=np.full(3, ob.T2, np.int16), value=np.array([290.0, 290.0, 300.0]),
                                 error=1.0, lat=np.zeros(3), lon=np.zeros(3))
    rng = np.random.default_rng(0)
    hx = 290.0 + 0.5 * rng.standard_normal((10, 3))
    hx[3, 1] = np.nan
    out, counters = ob.gross_check(b, hx, factor=4.0)
    assert out.qc.tolist() == [0, ob.QC_NONFINITE_HX, ob.QC_GROSS]
    assert counters["SURFACE/T2"] == {"checked": 3, "gross": 1, "nonfinite_hx": 1}
    # deletion sensitivity: with a huge factor nothing is gross
    out2, _ = ob.gross_check(b, hx, factor=1e6)
    assert out2.qc[2] == 0
    # ensemble spread enters the limit: d = 5, sigma_o = 1, std(Hx) = 2 -> 4*sqrt(1+4) = 8.9 > 5 accepted
    wide = np.array([[288.0], [292.0]] * 5)  # mean 290, std (ddof=1) ~ 2.1
    one = ob.ObsBatch.from_columns(var=[ob.T2], value=[295.0], error=1.0, lat=[0.0], lon=[0.0])
    assert ob.gross_check(one, wide)[0].qc[0] == 0
    assert ob.gross_check(one, np.full((10, 1), 290.0))[0].qc[0] == ob.QC_GROSS


def test_superob_layers_mean_and_errors():
    p_mass = np.array([1000.0, 900.0, 800.0, 700.0]) * 100
    b = ob.ObsBatch.from_columns(var=np.full(4, ob.T, np.int16), value=np.array([290.0, 288.0, 280.0, 1.0]),
                                 error=np.array([1.0, 1.0, 1.0, 1.0]), lat=28.3, lon=-16.4,
                                 p_pa=np.array([980.0, 960.0, 850.0, 500.0]) * 100, station="igra:X",
                                 qc=np.array([0, 0, 0, ob.QC_OUTSIDE]))
    bins = ob.sonde_layer_bins(b, p_mass)
    assert bins[0] == bins[1] != bins[2]
    out = ob.superob(b, bins, error_mode="keep")
    new = out.subset(out.qc == 0)
    assert len(new) == 2 and sorted(new.value.tolist()) == [280.0, 289.0]
    first = new.subset(new.value == 289.0)
    assert first.p_pa[0] == pytest.approx(np.exp(0.5 * (np.log(98000.0) + np.log(96000.0))))
    assert first.error[0] == 1.0
    assert ((out.qc & ob.QC_THINNED) != 0).sum() == 3 and out.meta["superob"]["groups"] == 2
    sq = ob.superob(b, bins, error_mode="sqrt_n")
    assert sq.subset((sq.qc == 0) & (sq.value == 289.0)).error[0] == pytest.approx(1 / np.sqrt(2))
    untouched = ob.superob(b, np.full(4, -1))
    assert untouched is b
    # withheld obs are never merged with assimilated ones (no leakage into the analysis)
    mixed = b.with_role(np.array([False, True, False, False]), "withhold")
    out_m = ob.superob(mixed, ob.sonde_layer_bins(mixed, p_mass))
    new_m = out_m.subset(out_m.qc == 0)
    assert len(new_m) == 3 and sorted(new_m.role.tolist()) == ["assimilate", "assimilate", "withhold"]
    assert new_m.subset(new_m.role == "withhold").value.tolist() == [288.0]


def test_dedup_colocated_alias_and_radius():
    b = ob.ObsBatch.from_columns(
        var=np.full(5, ob.T2, np.int16), value=np.full(5, 290.0), error=1.0,
        lat=np.array([28.4775, 28.4790, 28.00, 28.10, 28.105]), lon=np.array([-16.3292, -16.3300, -16.5, -16.7, -16.705]),
        height_m=np.array([617.0, 632.0, 50.0, 300.0, 310.0]),
        station=np.array(["metar:60015", "aemet:C447A", "aemet:FAR", "cabildo:Y", "grafcan:X"], dtype=object))
    out, n = ob.dedup_colocated(b)
    assert n == 2
    assert out.qc.tolist() == [ob.QC_DUPLICATE, 0, 0, ob.QC_DUPLICATE, 0]
    # different times are independent
    out2, n2 = ob.dedup_colocated(b.replace(dt_s=np.array([0.0, 3600.0, 0.0, 0.0, 3600.0])))
    assert n2 == 0


def test_protected_stations_refused():
    assert ob.is_protected("aemet:C429I") and ob.is_protected("cabezo:ANY") and ob.is_protected("metar:60025")
    assert not ob.is_protected("aemet:C447A")


# ------------------------------------------------------------------------ IGRA2
def igra_header(sid, y, m, d, h, rel, n, lat, lon):
    return f"#{sid:11s} {y:4d} {m:02d} {d:02d} {h:02d} {rel:4s} {n:4d} {'ncdc-gts':8s} {'':8s} {lat:7d} {lon:8d}\n"


def igra_level(t1, t2, p, gph, temp, rh, dpd, wd, ws):
    return f"{t1}{t2} {-9999:5d} {p:6d}B{gph:5d}B{temp:5d}B{rh:5d} {dpd:5d} {wd:5d} {ws:5d}\n"


def test_read_igra2_synthetic_levels(tmp_path):
    path = tmp_path / "igra.txt"
    lines = [igra_header("SPM00060018", 2026, 5, 3, 0, "2315", 2, 283183, -163822),
             igra_level(2, 1, 101500, 105, 210, -9999, 40, 0, 0),
             igra_level(1, 0, 100000, -9999, 190, -9999, 30, 0, 0),
             igra_header("SPM00060018", 2026, 5, 3, 12, "1115", 5, 283183, -163822),
             igra_level(2, 1, 101000, 100, 200, -9999, 50, 90, 50),
             igra_level(1, 0, 100000, -9999, 180, -9999, 40, 180, 100),
             igra_level(2, 0, 95000, -9999, 150, -9999, -8888, -9999, -9999),
             igra_level(3, 0, -9999, 1500, -9999, -9999, -9999, 270, 100),
             igra_level(1, 0, 85000, 1500, 100, -9999, 200, 270, 200)]
    path.write_text("".join(lines))
    b = ob.read_igra2(path, "SPM00060018", "2026-05-03T12:00Z", analysis_time="2026-05-03T13:00")
    assert b.meta["release_time"] == "2026-05-03T11:15:00Z"
    assert b.meta["skipped"] == {"non_pressure_level": 1, "igra_qc_removed": 1}
    t = b.subset(b.var == ob.T)
    assert np.allclose(t.value, [293.15, 291.15, 288.15, 283.15]) and np.allclose(t.p_pa, [101000, 1e5, 95000, 85000])
    qv = b.subset(b.var == ob.QV)
    assert len(qv) == 3
    assert qv.value[0] == pytest.approx(ob.EP_2 * ob.sat_vapor_pressure(288.15) / (101000 - ob.sat_vapor_pressure(288.15)))
    u, v = b.subset(b.var == ob.U), b.subset(b.var == ob.V)
    assert np.allclose(u.value, [-5.0, 0.0, 20.0], atol=1e-12) and np.allclose(v.value, [0.0, 10.0, 0.0], atol=1e-12)
    assert (b.dt_s == -3600.0).all() and (b.station == "igra:SPM00060018").all()
    assert (b.launch == "SPM00060018@2026-05-03T12Z").all() and len(set(b.obs_id)) == len(b)
    assert (b.time == np.datetime64("2026-05-03T12:00:00")).all()
    assert np.allclose(b.lat, 28.3183) and np.allclose(b.lon, -16.3822)
    # hypsometric height of 1000 hPa from the 1010 hPa surface (Tv with qv)
    tv0 = 293.15 * (1 + 0.608 * qv.value[0])
    tv1 = 291.15 * (1 + 0.608 * qv.value[1])
    z1 = 100.0 + 287.0 * 0.5 * (tv0 + tv1) / 9.81 * np.log(1010.0 / 1000.0)
    assert t.height_m[1] == pytest.approx(z1, abs=1e-9) and t.height_m[3] == 1500.0
    # errors: QV error from RH sigma 10 % of qsat(T, p)
    assert qv.error[0] == pytest.approx(0.1 * ob.qv_saturation(293.15, 101000.0))
    # a 00Z launch with RELTIME 2315 was released on the PREVIOUS day
    early = ob.read_igra2(path, "SPM00060018", "2026-05-03T00:00")
    assert early.meta["release_time"] == "2026-05-02T23:15:00Z"
    lt = ob.read_igra2(path, "SPM00060018", "2026-05-03T00:00", level_time=True, analysis_time="2026-05-03T00:00")
    assert lt.subset(lt.var == ob.T).dt_s[0] == -2700.0               # surface level at 23:15 the day before
    nosfc = ob.read_igra2(path, "SPM00060018", "2026-05-03T12:00", include_surface=False)
    assert len(nosfc.subset(nosfc.var == ob.T)) == 3
    with pytest.raises(LookupError):
        ob.read_igra2(path, "SPM00060018", "2026-05-04T12:00")


def test_read_igra2_drift_constant_wind(tmp_path):
    """Constant 10 m/s westerly, 5 m/s ascent: the 850 hPa level (z 1500 m from 100 m) is 2800 m east."""
    path = tmp_path / "igra.txt"
    lines = [igra_header("SPM00060018", 2026, 5, 3, 12, "1115", 3, 283183, -163822),
             igra_level(2, 1, 101000, 100, 200, -9999, 50, 270, 100),
             igra_level(1, 0, 92500, 800, 150, -9999, 40, 270, 100),
             igra_level(1, 0, 85000, 1500, 100, -9999, 40, 270, 100)]
    path.write_text("".join(lines))
    fixed = ob.read_igra2(path, "SPM00060018", "2026-05-03T12:00")
    moved = ob.read_igra2(path, "SPM00060018", "2026-05-03T12:00", drift=True, ascent_mps=5.0)
    assert np.allclose(fixed.lat, 28.3183) and fixed.meta["drift"] == "ignored (launch site)"
    t = moved.subset(moved.var == ob.T)
    east_m = np.deg2rad(t.lon + 163822 / 1e4) * 6370000.0 * np.cos(np.deg2rad(28.3183))
    assert np.allclose(east_m, [0.0, 1400.0, 2800.0], atol=1e-6) and np.allclose(t.lat, 28.3183)
    assert moved.meta["drift"]["max_km"] == pytest.approx(2.8, abs=1e-3)
    timed = ob.read_igra2(path, "SPM00060018", "2026-05-03T12:00", level_time=True, analysis_time="2026-05-03T12:00")
    tt = timed.subset(timed.var == ob.T)
    # release 11:15, 0 / 140 / 280 s ascent -> dt = -2700, -2560, -2420 s; positions stay at the launch site
    assert tt.dt_s.tolist() == [-2700.0, -2560.0, -2420.0] and np.allclose(tt.lat, 28.3183)
    assert tt.time[2] == np.datetime64("2026-05-03T11:19:40")


@pytest.mark.skipif(not IGRA.exists(), reason="guarded IGRA2 excerpt absent")
def test_read_igra2_real_0503_12z_matches_shared_levels_parquet():
    pd = pytest.importorskip("pandas")
    b = ob.read_igra2(IGRA, "SPM00060018", "2026-05-03T12:00")
    heads = pd.read_parquet(GUARDED / "soundings_admitted.parquet")
    sid = int(heads.loc[heads.time == pd.Timestamp("2026-05-03 12:00"), "sid"].iloc[0])
    lv = pd.read_parquet(GUARDED / "levels_admitted.parquet")
    lv = lv[lv.sid == sid]
    assert b.meta["levels"] == int(lv.p_pa.notna().sum())
    ref_t = lv[lv.p_pa.notna() & lv.t_c.notna()]
    t = b.subset(b.var == ob.T)
    assert np.allclose(np.sort(t.p_pa), np.sort(ref_t.p_pa.to_numpy())) and len(t) == len(ref_t)
    order = np.argsort(-t.p_pa, kind="stable")
    ref_order = np.argsort(-ref_t.p_pa.to_numpy(), kind="stable")
    assert np.allclose(t.value[order], ref_t.t_c.to_numpy()[ref_order] + 273.15, atol=1e-9)
    # significant levels are kept: far more T levels than the 7 mandatory ones below 500 hPa
    assert (t.p_pa > 5.0e4).sum() > 30
    qv = b.subset(b.var == ob.QV)
    ref_q = lv[lv.p_pa.notna() & lv.t_c.notna() & lv.dpd_c.notna()]
    assert len(qv) == len(ref_q)
    rh = ob.rh_from_qv(qv.value, ref_q.t_c.to_numpy() + 273.15, ref_q.p_pa.to_numpy())
    # Bolton/WRF vs the parquet's Magnus (17.625, 243.04): same dewpoint physics, <1 %RH apart
    assert np.max(np.abs(100 * rh - ref_q.rh_pct.to_numpy())) < 1.0
    ref_w = lv[lv.p_pa.notna() & lv.wspd.notna() & lv.wdir.notna()]
    u = b.subset(b.var == ob.U)
    assert len(u) == len(ref_w)
    eu, ev = ob.wind_components(ref_w.wspd.to_numpy(), ref_w.wdir.to_numpy())
    assert np.allclose(u.value, eu, atol=1e-9) and np.allclose(b.subset(b.var == ob.V).value, ev, atol=1e-9)
    assert b.meta["release_time"] == "2026-05-03T11:15:00Z"


# ------------------------------------------------------------- station reader
def write_extract(tmp_path, extra_rows=()):
    pd = pytest.importorskip("pandas")
    stations = pd.DataFrame({
        "station_id": ["aemet:C447A", "metar:60015", "iac:gong", "grafcan:0001", "aemet:FAR", "grafcan:0099"],
        "lat": [28.4775, 28.4789, 28.30, 28.10, 27.0, 28.6],
        "lon": [-16.3292, -16.3300, -16.51, -16.60, -18.9, -16.2],
        "elevation_m": [617.0, 632.0, 2390.0, 400.0, 10.0, np.nan],
        "elevation_source": ["altitude_verified_m"] * 6,
        "altitude_flag": ["ok", "ok", "ok", "ok", "ok", "check_gt30m"]})
    t = "2026-05-03 "
    rows = [("aemet:C447A", t + "11:58:00", "t2_c", 15.0, 0), ("aemet:C447A", t + "12:07:00", "t2_c", 99.0, 0),
            ("aemet:C447A", t + "11:58:00", "rh_pct", 80.0, 0), ("aemet:C447A", t + "11:58:00", "pressure_hpa", 940.0, 0),
            ("aemet:C447A", t + "12:00:00", "wind_speed_ms", 5.0, 0), ("aemet:C447A", t + "12:00:00", "wind_dir_deg", 90.0, 0),
            ("metar:60015", t + "12:00:00", "t2_c", 15.5, 0),
            ("iac:gong", t + "12:00:00", "t2_c", 8.0, 0), ("iac:gong", t + "12:00:00", "wind_speed_ms", 3.0, 0),
            ("iac:gong", t + "12:00:00", "wind_dir_deg", 0.0, 0),
            ("grafcan:0001", t + "12:00:00", "t2_c", 20.0, 0), ("grafcan:0001", t + "12:00:00", "rh_pct", 110.0, 0),
            ("grafcan:0001", t + "13:00:00", "t2_c", 21.0, 1), ("grafcan:0001", t + "13:20:00", "t2_c", 22.0, 0),
            ("grafcan:0001", t + "11:00:00", "wind_speed_ms", 2.0, 0),
            ("aemet:FAR", t + "12:00:00", "t2_c", 25.0, 0),
            ("grafcan:0099", t + "12:00:00", "t2_c", 18.0, 0)]
    rows += list(extra_rows)
    obs = pd.DataFrame(rows, columns=["station_id", "ts_utc", "variable", "value", "qc_flag"])
    stations.to_parquet(tmp_path / "stations.parquet")
    obs.to_parquet(tmp_path / "observations.parquet")
    return tmp_path


def test_read_station_obs_synthetic_extract(tmp_path):
    src = write_extract(tmp_path)
    b = ob.read_station_obs(src, "2026-05-03T11:00", "2026-05-03T13:00", bbox=(27.5, 29.5, -18.5, -13.0),
                            analysis_time="2026-05-03T12:00")
    c = b.meta["counters"]
    assert c["outside_bbox"] == 1 and c["sheltered_wind_excluded"] == 1 and c["rh_out_of_range"] == 1
    assert c["obsdb_qc_flag_dropped"] == 1 and c["wind_pair_missing"] == 1 and c["colocated_duplicates"] == 1
    assert c["pressure_not_used"] == 1 and c["unknown_elevation_rejected"] == 1
    t2 = b.subset((b.var == ob.T2) & (b.station == "aemet:C447A"))
    assert t2.value.tolist() == [288.15] and t2.dt_s.tolist() == [0.0]   # 11:58 beats 12:07
    assert t2.time[0] == np.datetime64("2026-05-03T12:00:00") and t2.site[0] == "site:aemet:C447A"
    q2 = b.subset(b.var == ob.Q2)
    assert len(q2) == 1 and q2.value[0] == pytest.approx(ob.qv_from_rh(0.8, 288.15, 94000.0))
    assert not (b.var == ob.PSFC).any()                # DESIGN_v2: no pressure obs by default
    u = b.subset(b.var == ob.U10)
    assert u.value[0] == pytest.approx(-5.0) and abs(b.subset(b.var == ob.V10).value[0]) < 1e-12
    dup = b.subset(b.station == "metar:60015")
    assert (dup.qc & ob.QC_DUPLICATE).all() and dup.site[0] == "site:aemet:C447A"
    assert b.subset(b.station == "iac:gong").site[0] == "site:aemet:C430E"
    graf = b.subset((b.station == "grafcan:0001") & (b.var == ob.T2))
    assert graf.value.tolist() == [293.15]          # 13:00 flagged, 13:20 outside 10-min tolerance
    assert not np.isin(b.station, ["aemet:FAR"]).any()
    unknown = b.subset(b.station == "grafcan:0099")
    assert len(unknown) == 1 and unknown.qc[0] == ob.QC_SOURCE
    assert b.meta["altitude_flag_rejected_stations"] == ["grafcan:0099"]
    # flagged height alone (finite elevation) is also rejected
    flagged_only = ob.read_station_obs(src, "2026-05-03T12:00", "2026-05-03T12:00",
                                       reject_altitude_flags=("ok",))
    assert (flagged_only.subset(flagged_only.station == "aemet:C447A").qc & ob.QC_SOURCE).all()
    with_p = ob.read_station_obs(src, "2026-05-03T12:00", "2026-05-03T12:00", include_pressure=True)
    assert with_p.subset(with_p.var == ob.PSFC).value.tolist() == [94000.0]
    # the lead's adapter passes POSIX seconds and a single joined parquet
    import pandas as pd
    joined = pd.read_parquet(src / "observations.parquet").merge(pd.read_parquet(src / "stations.parquet"),
                                                                  on="station_id")
    joined.to_parquet(tmp_path / "joined.parquet")
    t12 = ob.to_datetime64("2026-05-03T12:00").astype("int64")
    one = ob.read_station_obs(tmp_path / "joined.parquet", float(t12), float(t12))
    assert sorted(one.obs_id.tolist()) == sorted(ob.read_station_obs(src, "2026-05-03T12:00",
                                                                     "2026-05-03T12:00").obs_id.tolist())


def test_read_station_obs_refuses_protected(tmp_path):
    src = write_extract(tmp_path, [("aemet:C429I", "2026-05-03 12:00:00", "t2_c", 20.0, 0)])
    with pytest.raises(ValueError, match="protected"):
        ob.read_station_obs(src, "2026-05-03T11:00", "2026-05-03T13:00")


# --------------------------------------------------------------- MSG cloud mask
def test_read_msg_clm_synthetic_boxes(tmp_path):
    ny, nx = 7, 8
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    grid = {"XLAT": 28.0 + 0.1 * j, "XLONG": -17.0 + 0.1 * i}
    # 4 pixels (+-0.02 deg) around every INTERIOR mass point; cloudy count = min(i % 5, 4)
    plat, plon, code = [], [], []
    for jj in range(1, ny - 1):
        for ii in range(1, nx - 1):
            for k, (dy, dx) in enumerate(((-0.02, -0.02), (-0.02, 0.02), (0.02, -0.02), (0.02, 0.02))):
                plat.append(28.0 + 0.1 * jj + dy)
                plon.append(-17.0 + 0.1 * ii + dx)
                code.append(255 if (jj, ii) == (1, 1) else (1 if k < ii % 5 else 0))
    plat.append(40.0)
    plon.append(-17.0)  # outside the grid
    code.append(0)
    code = np.array(code, np.uint8)
    times = np.array(["2026-05-03T11:45", "2026-05-03T12:00", "2026-05-03T12:15"], dtype="datetime64[ns]")
    np.savez(tmp_path / "m.npz", time=times.astype("int64"), latitude=np.array(plat, np.float32),
             longitude=np.array(plon, np.float32), cloud_binary=np.stack([code, code, code]),
             sensor_id=np.array(["MSG3"] * 3))
    b = ob.read_msg_clm(tmp_path / "m.npz", "2026-05-03T12:04", grid, stride=1, min_pixels=3,
                        analysis_time="2026-05-03T12:00")
    assert b.meta["slot_time"] == "2026-05-03T12:00:00Z" and b.meta["pixels_invalid"] == 4
    assert b.meta["pixels_outside_grid"] == 1 and len(b) == (ny - 2) * (nx - 2) - 1
    assert (b.var == ob.CLDMASK).all() and (b.error == 0.3).all() and (b.dt_s == 0.0).all()
    assert (b.role == "withhold").all()            # DESIGN_v2: cloud is a diagnostic, not assimilated
    loc_i = np.rint((b.lon + 17.0) / 0.1).astype(int)
    assert np.allclose(b.value, np.minimum(loc_i % 5, 4) / 4.0)
    with pytest.raises(LookupError):
        ob.read_msg_clm(tmp_path / "m.npz", "2026-05-03T14:00", grid)
    coarse = ob.read_msg_clm(tmp_path / "m.npz", "2026-05-03T12:00", grid, stride=2, min_pixels=3)
    assert len(coarse) == 3 * 4 - 1  # box (0, 0) holds only the invalid cell (1, 1)


@pytest.mark.skip(reason="MSG CLM is not in the guard_call-admitted read set (cloud DA cut); "
                         "enable only after an ALISIOS admission for the satellite product")
def test_read_msg_clm_real_0503_on_d01_shaped_grid():
    lat, lon = np.meshgrid(np.linspace(26.6, 30.4, 40), np.linspace(-18.9, -12.6, 60), indexing="ij")
    b = ob.read_msg_clm(MSG_0503, "2026-05-03T12:00", {"XLAT": lat, "XLONG": lon}, stride=1, min_pixels=2)
    assert b.meta["slot_time"] == "2026-05-03T12:00:00Z" and len(b) > 1000
    assert ((b.value >= 0) & (b.value <= 1)).all()
    assert 0.0 < b.value.mean() < 1.0
