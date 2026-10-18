"""Generic observation readers (IGRA2 of any station, ISD-Lite, CSV surface/profile, ObsBatch JSON).

Independent references: hand Bolton (1980) saturation vapour pressure + MetPy mixing ratio and wind components
(MetPy uses epsilon 0.62196 vs WRF's R_d/R_v = 0.62175, so humidity agrees to 4e-4 relative; for RH MetPy's
own Ambaum saturation formula adds ~1e-3) and hand-written unit conversions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import obs as ob
from gpuwrf.assimilation.letkf import readers as rd

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units

T_A = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)
GEO = {"XLAT": np.array([[45.0, 45.0], [48.0, 48.0]]), "XLONG": np.array([[5.0, 11.0], [5.0, 11.0]])}


def _q_metpy(td_k, p_pa):
    """Hand Bolton (1980, eq. 10) saturation vapour pressure — the WRF/real.exe form — then MetPy's mixing ratio.

    MetPy's own saturation_vapor_pressure (Ambaum 2020) differs from Bolton by ~1e-3 at these temperatures,
    MetPy's epsilon (0.62196) from WRF's R_d/R_v (0.62175) by 3.3e-4."""
    e_hpa = 6.112 * np.exp(17.67 * (td_k - 273.15) / (td_k - 273.15 + 243.5))
    return float(mpcalc.mixing_ratio(e_hpa * units.hPa, p_pa * units.pascal).to("kg/kg").magnitude)


def _q_rh_metpy(rh, t_k, p_pa):
    return float(mpcalc.mixing_ratio_from_relative_humidity(p_pa * units.pascal, t_k * units.kelvin,
                                                            rh * units.percent).to("kg/kg").magnitude)


def _uv_metpy(speed, direction):
    u, v = mpcalc.wind_components(speed * units("m/s"), direction * units.degree)
    return float(u.magnitude), float(v.magnitude)


def _p_std(z):
    return 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588


def _rows(batch, var):
    sel = np.asarray(batch.var) == var
    return np.asarray(batch.value)[sel]


def _write(path, header, rows):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows) + "\n")
    return path


# ------------------------------------------------------------------------------------------- CSV surface

def test_csv_surface_values_units_window_and_counters(tmp_path):
    header = ["station_id", "lat", "lon", "elevation_m", "time", "t2_c", "td2_c", "wspd10_ms", "wdir10_deg", "psfc_hpa"]
    rows = [
        ["A", 46.0, 7.0, 500.0, "2023-01-15T00:10:00Z", 2.0, -1.5, 6.0, 250.0, 950.0],
        ["A", 46.0, 7.0, 500.0, "2023-01-14T23:50:00Z", 9.0, 0.0, 1.0, 10.0, 951.0],      # tie: earlier wins
        ["A", 46.0, 7.0, 500.0, "2023-01-15T02:00:00Z", 5.0, 0.0, 1.0, 10.0, 951.0],      # outside 30 min
        ["B", 47.0, 8.0, 1200.0, "2023-01-15T00:00:00+00:00", -4.0, "", 0.0, 0.0, ""],   # calm, no humidity
        ["C", 47.5, 9.0, 300.0, "2023-01-15T00:00:00Z", 1.0, -3.0, 3.0, 0.0, ""],        # dir 0 with speed: invalid
        ["F", 60.0, 9.0, 300.0, "2023-01-15T00:00:00Z", 1.0, -3.0, 3.0, 90.0, ""],       # outside the domain
    ]
    batch = rd.read_csv_surface(_write(tmp_path / "s.csv", header, rows), T_A, bbox=rd._domain_bbox(GEO))
    st = np.asarray(batch.station)
    a = st == "csv:A"
    t2 = np.asarray(batch.value)[a & (np.asarray(batch.var) == ob.T2)]
    assert t2.tolist() == [pytest.approx(9.0 + 273.15)]                     # 23:50 record (tie -> earlier)
    q2 = np.asarray(batch.value)[a & (np.asarray(batch.var) == ob.Q2)][0]
    assert q2 == pytest.approx(_q_metpy(273.15, 95100.0), rel=4e-4)
    u, v = _uv_metpy(1.0, 10.0)
    assert np.asarray(batch.value)[a & (np.asarray(batch.var) == ob.U10)][0] == pytest.approx(u, abs=1e-12)
    assert np.asarray(batch.value)[a & (np.asarray(batch.var) == ob.V10)][0] == pytest.approx(v, abs=1e-12)
    b = st == "csv:B"
    assert sorted(np.asarray(batch.var)[b].tolist()) == [ob.T2, ob.U10, ob.V10]
    assert np.allclose(np.asarray(batch.value)[b & (np.asarray(batch.var) != ob.T2)], 0.0)
    c = st == "csv:C"
    assert sorted(np.asarray(batch.var)[c].tolist()) == [ob.T2, ob.Q2]
    q_c = np.asarray(batch.value)[c & (np.asarray(batch.var) == ob.Q2)][0]
    assert q_c == pytest.approx(_q_metpy(270.15, _p_std(300.0)), rel=4e-4)
    counters = batch.meta["counters"]
    assert counters["outside_window"] == 1 and counters["outside_bbox"] == 1 and counters["wind_invalid"] == 1
    assert counters["q2_std_atmosphere_pressure"] == 1
    dt = np.asarray(batch.dt_s)[a]
    assert np.all(dt == -600.0)
    assert np.all(np.asarray(batch.error) > 0) and np.all(np.asarray(batch.qc) == 0)


def test_csv_surface_rh_and_q_columns(tmp_path):
    header = ["station_id", "lat", "lon", "elevation_m", "time", "t2_k", "rh2_pct", "u10_ms", "v10_ms"]
    rows = [["R", 46.0, 7.0, 0.0, "2023-01-15T00:00:00Z", 280.0, 70.0, 1.5, -2.5],
            ["S", 46.5, 7.5, 0.0, "2023-01-15T00:00:00Z", 280.0, 130.0, "", -2.5]]
    batch = rd.read_csv_surface(_write(tmp_path / "s.csv", header, rows), T_A)
    r = np.asarray(batch.station) == "csv:R"
    q = np.asarray(batch.value)[r & (np.asarray(batch.var) == ob.Q2)][0]
    assert q == pytest.approx(_q_rh_metpy(70.0, 280.0, _p_std(0.0)), rel=2e-3)
    assert np.asarray(batch.value)[r & (np.asarray(batch.var) == ob.U10)][0] == 1.5
    assert batch.meta["counters"]["rh_out_of_range"] == 1 and batch.meta["counters"]["wind_pair_missing"] == 1


@pytest.mark.parametrize("header, row, message", [
    (["station_id", "lat", "lon", "elevation_m", "time", "t2_c"], ["A", 46, 7, 0, "2023-01-15T00:00:00", 1], "UTC offset"),
    (["station_id", "lat", "lon", "elevation_m", "time", "t2_f"], ["A", 46, 7, 0, "2023-01-15T00:00:00Z", 1], "unknown"),
    (["station_id", "lat", "lon", "time", "t2_c"], ["A", 46, 7, "2023-01-15T00:00:00Z", 1], "missing required"),
    (["station_id", "lat", "lon", "elevation_m", "time", "t2_c"], ["A", 46, 7, 0, "2023-01-15T00:00:00Z", "warm"],
     "not a number"),
])
def test_csv_surface_refusals(tmp_path, header, row, message):
    with pytest.raises(ValueError, match=message):
        rd.read_csv_surface(_write(tmp_path / "bad.csv", header, [row]), T_A)


def test_csv_surface_station_moving_is_refused(tmp_path):
    header = ["station_id", "lat", "lon", "elevation_m", "time", "t2_c"]
    rows = [["A", 46, 7, 0, "2023-01-15T00:00:00Z", 1], ["A", 46.1, 7, 0, "2023-01-15T00:05:00Z", 1]]
    with pytest.raises(ValueError, match="changes its coordinates"):
        rd.read_csv_surface(_write(tmp_path / "m.csv", header, rows), T_A)


# ------------------------------------------------------------------------------------------- CSV profile

def test_csv_profile_levels(tmp_path):
    header = ["station_id", "lat", "lon", "time", "p_hpa", "height_m", "t_c", "td_c", "wspd_ms", "wdir_deg"]
    rows = [["P", 46.8, 6.9, "2023-01-14T23:30:00Z", 950.0, 490.0, 3.0, -2.0, 5.0, 225.0],
            ["P", 46.8, 6.9, "2023-01-14T23:30:00Z", 850.0, "", -2.0, -9.0, 12.0, 270.0],
            ["P", 46.8, 6.9, "2023-01-14T23:30:00Z", "", "", -2.0, -9.0, 12.0, 270.0]]
    batch = rd.read_csv_profile(_write(tmp_path / "p.csv", header, rows), T_A)
    assert set(np.asarray(batch.kind).tolist()) == {ob.SONDE} and batch.meta["counters"]["no_pressure"] == 1
    lev = np.asarray(batch.p_pa) == 85000.0
    q = np.asarray(batch.value)[lev & (np.asarray(batch.var) == ob.QV)][0]
    assert q == pytest.approx(_q_metpy(264.15, 85000.0), rel=4e-4)
    u, v = _uv_metpy(12.0, 270.0)
    assert np.asarray(batch.value)[lev & (np.asarray(batch.var) == ob.U)][0] == pytest.approx(u, abs=1e-12)
    assert set(np.asarray(batch.launch).tolist()) == {"P@2023-01-14T23:30:00Z"}
    assert np.all(np.asarray(batch.dt_s) == -1800.0)


# ------------------------------------------------------------------------------------------- ISD-Lite

def test_isd_lite_with_history_station(tmp_path):
    lines = ["2023 01 14 23    15   -20 10213   260    30     0 -9999 -9999",
             "2023 01 15 00   -12   -45 10210   270    50     0 -9999 -9999",
             "2023 01 15 01 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999"]
    data = tmp_path / "066100-99999-2023"
    data.write_text("\n".join(lines) + "\n")
    hist = tmp_path / "isd-history.csv"
    hist.write_text('"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
                    '"066100","99999","PAYERNE","SZ","","LSMP","+46.812","+006.943","+0490.0","19730101","20231231"\n')
    batch = rd.read_isd_lite(data, T_A, isd_history=hist, station_id="066100-99999")
    assert set(np.asarray(batch.station).tolist()) == {"isd:066100-99999"}
    assert _rows(batch, ob.T2)[0] == pytest.approx(-1.2 + 273.15)
    assert _rows(batch, ob.Q2)[0] == pytest.approx(_q_metpy(-4.5 + 273.15, _p_std(490.0)), rel=4e-4)
    assert _rows(batch, ob.U10)[0] == pytest.approx(_uv_metpy(5.0, 270.0)[0], abs=1e-12)
    assert np.asarray(batch.height_m)[0] == 490.0 and np.asarray(batch.lat)[0] == pytest.approx(46.812)
    with pytest.raises(LookupError):
        rd.read_isd_lite(data, T_A, isd_history=hist, station_id="066100-12345")
    (tmp_path / "bad").write_text("2023 01 15 00 1 2 3\n")
    with pytest.raises(ValueError, match="12 integer fields"):
        rd.read_isd_lite(tmp_path / "bad", T_A, station={"id": "x", "lat": 46.0, "lon": 7.0, "elevation_m": 0.0})


# ------------------------------------------------------------------------------------------- IGRA2

def _igra_header(station, y, mo, d, h, rel, nlev, lat, lon):
    line = [" "] * 71
    def put(col, text):            # col = 1-based start column
        for k, ch in enumerate(text):
            line[col - 1 + k] = ch
    put(1, "#"); put(2, station); put(14, f"{y:4d}"); put(19, f"{mo:02d}"); put(22, f"{d:02d}"); put(25, f"{h:02d}")
    put(28, rel); put(33, f"{nlev:4d}"); put(56, f"{int(round(lat * 1e4)):7d}"); put(64, f"{int(round(lon * 1e4)):8d}")
    return "".join(line)


def _igra_level(t1, t2, etime, p, gph, temp, rh, dpd, wdir, wspd):
    line = [" "] * 51
    def put(col, text):
        for k, ch in enumerate(text):
            line[col - 1 + k] = ch
    put(1, str(t1)); put(2, str(t2)); put(4, f"{etime:5d}"); put(10, f"{p:6d}"); put(17, f"{gph:5d}")
    put(23, f"{temp:5d}"); put(29, f"{rh:5d}"); put(35, f"{dpd:5d}"); put(41, f"{wdir:5d}"); put(47, f"{wspd:5d}")
    return "".join(line)


def test_igra2_any_station(tmp_path):
    lev = [_igra_level(2, 1, 0, 101000, 10, 152, -9999, 30, 200, 40),
           _igra_level(1, 0, 300, 85000, 1500, 50, -9999, 80, 250, 100),
           _igra_level(2, 0, 600, 70000, -9999, -45, -9999, 150, -9999, -9999)]
    text = "\n".join([_igra_header("USM00072201", 2023, 1, 14, 12, "1115", 0, 24.5550, -81.7880),
                      _igra_header("USM00072201", 2023, 1, 15, 0, "2315", 3, 24.5550, -81.7880), *lev]) + "\n"
    path = tmp_path / "USM00072201-data.txt"
    path.write_text(text)
    batch = rd.read_source({"format": "igra2", "path": str(path), "station": "USM00072201"}, T_A,
                           {"XLAT": np.array([[20.0, 30.0]]), "XLONG": np.array([[-85.0, -75.0]])})
    t = _rows(batch, ob.T)
    assert sorted(t.tolist()) == sorted([15.2 + 273.15, 5.0 + 273.15, -4.5 + 273.15])
    p = np.asarray(batch.p_pa)
    q850 = np.asarray(batch.value)[(p == 85000.0) & (np.asarray(batch.var) == ob.QV)][0]
    assert q850 == pytest.approx(_q_metpy(5.0 - 8.0 + 273.15, 85000.0), rel=4e-4)
    u850 = np.asarray(batch.value)[(p == 85000.0) & (np.asarray(batch.var) == ob.U)][0]
    assert u850 == pytest.approx(_uv_metpy(10.0, 250.0)[0], abs=1e-12)
    assert batch.meta["release_time"] == "2023-01-14T23:15:00Z" and batch.meta["station"] == "USM00072201"
    assert np.isfinite(np.asarray(batch.height_m)).all()                    # missing GPH filled hypsometrically
    with pytest.raises(LookupError):
        rd.read_source({"format": "igra2", "path": str(path), "station": "USM00072202"}, T_A, GEO)


# ------------------------------------------------------------------------------------------- dispatch

def test_obsbatch_json_round_trip_and_read_sources(tmp_path):
    header = ["station_id", "lat", "lon", "elevation_m", "time", "t2_c"]
    csv_path = _write(tmp_path / "s.csv", header, [["A", 46.0, 7.0, 500.0, "2023-01-15T00:00:00Z", 2.0]])
    batch = rd.read_csv_surface(csv_path, T_A)
    js = tmp_path / "obs.json"
    js.write_text(json.dumps(batch.to_dict(), default=str))
    cfg = {"sources": [{"format": "csv_surface", "path": str(csv_path)}, {"format": "obsbatch_json", "path": str(js)}],
           "sonde_superob": True}
    merged, info = rd.read_sources(cfg, T_A, GEO)
    assert len(merged) == 2 and [s["n"] for s in info["sources"]] == [1, 1]
    assert np.asarray(merged.value).tolist() == [2.0 + 273.15] * 2


def test_error_overrides_reach_the_batch(tmp_path):
    header = ["station_id", "lat", "lon", "elevation_m", "time", "t2_c"]
    csv_path = _write(tmp_path / "s.csv", header, [["A", 46.0, 7.0, 500.0, "2023-01-15T00:00:00Z", 2.0]])
    cfg = {"sources": [{"format": "csv_surface", "path": str(csv_path)}],
           "error_overrides": {"SURFACE": {"T2": {"sigma": 0.9}}}}
    batch, _ = rd.read_sources(cfg, T_A, GEO)
    assert np.asarray(batch.error).tolist() == [0.9]
