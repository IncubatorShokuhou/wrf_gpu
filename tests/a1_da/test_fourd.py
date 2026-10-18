"""Tests for gpuwrf.assimilation.letkf.fourd (4D-LETKF H(x) over the cycle window; lane a1-obs). CPU only."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from gpuwrf.assimilation.letkf import fourd
from gpuwrf.assimilation.letkf import obs as ob
from test_scoring import _grid, _write_wrfout

T_A = np.datetime64("2026-05-03T18:00", "s")
NAMES = {12: "12:00:00", 13: "13:00:18", 14: "14:00:36", 15: "15:00:00", 16: "16:00:18", 17: "17:00:36",
         18: "18:00:00"}


def _members(tmp_path, n=3, missing=((2, 15),)):
    geo, i, j = _grid()
    dirs = []
    for m in range(n):
        d = tmp_path / f"m{m:02d}"
        d.mkdir()
        for h, stamp in NAMES.items():
            if (m, h) in missing:
                continue
            fields = {"T2": 290.0 + 0.1 * i - 0.05 * j + 0.5 * (h - 12) + m, "Q2": 0.008 + 1e-4 * h + 0 * i,
                      "U10": 1.0 + 0.25 * h + 0 * i, "V10": -2.0 + 0 * i, "PSFC": 1.0e5 + 0 * i}
            _write_wrfout(d / f"wrfout_d01_2026-05-03_{stamp}", geo, fields, times=f"2026-05-03_{stamp}")
        dirs.append(d)
    return geo, dirs


def _obs(times, lat=28.15, lon=-16.8):
    n = len(times)
    return ob.ObsBatch.from_columns(var=np.full(n, ob.T2, np.int16), value=np.full(n, 290.0), error=1.0,
                                    lat=np.full(n, lat), lon=np.full(n, lon), height_m=np.full(n, 100.0),
                                    station=np.array([f"grafcan:{k:04d}" for k in range(n)], dtype=object),
                                    time=np.array(times, dtype="datetime64[s]"))


def test_hx_4d_nearest_own_frame_window_and_lazy_reads(tmp_path):
    geo, dirs = _members(tmp_path)
    times = ["2026-05-03T12:00", "2026-05-03T13:00", "2026-05-03T15:20", "2026-05-03T15:40", "2026-05-03T18:00",
             "2026-05-03T18:30", "2026-05-03T12:20"]
    obs = _obs(times)
    res = fourd.hx_4d(obs, dirs, T_A, 6.0, geo)
    i_pt, j_pt = (-16.8 + 17.0) / 0.05, (28.15 - 28.0) / 0.05
    base = 290.0 + 0.1 * i_pt - 0.05 * j_pt
    want_hour = [None, 13, 15, 16, 18, None, None]
    for k, h in enumerate(want_hour):
        for m in range(3):
            if h is None or (m == 2 and h == 15):
                assert np.isnan(res.hx[m, k]), (k, m)
            else:
                assert res.hx[m, k] == pytest.approx(base + 0.5 * (h - 12) + m, abs=1e-4), (k, m)
                assert res.frame_time[m, k] == np.datetime64(f"2026-05-03T{h:02d}:00", "s")
    assert res.in_window.tolist() == [False, True, True, True, True, False, True]
    assert res.qc[0] & ob.QC_OUTSIDE and res.qc[5] & ob.QC_OUTSIDE and res.qc[6] & ob.QC_OUTSIDE
    assert res.qc[1] == res.qc[2] == res.qc[3] == res.qc[4] == 0     # member 2's missing frame -> NaN, not a QC flag
    # only frames with observations are opened (12Z, 14Z, 17Z never)
    opened = {os.path.basename(p) for p in res.files_opened[0]}
    assert opened == {"wrfout_d01_2026-05-03_13:00:18", "wrfout_d01_2026-05-03_15:00:00",
                      "wrfout_d01_2026-05-03_16:00:18", "wrfout_d01_2026-05-03_18:00:00"}
    assert res.frame_offset_s[0]["2026-05-03T13:00:00"] == 18.0 and res.frame_offset_s[0]["2026-05-03T14:00:00"] == 36.0


def test_hx_4d_dt_fallback_shorter_window_and_nonusable(tmp_path):
    geo, dirs = _members(tmp_path, n=2, missing=())
    obs = _obs(["2026-05-03T16:00", "2026-05-03T17:00"])
    no_time = obs.replace(time=np.array(["NaT", "NaT"], dtype="datetime64[s]"), dt_s=np.array([-7200.0, -3600.0]))
    a, b = fourd.hx_4d(obs, dirs, T_A, 6.0, geo), fourd.hx_4d(no_time, dirs, T_A, 6.0, geo)
    assert np.array_equal(a.hx, b.hx) and np.isfinite(a.hx).all()
    short = fourd.hx_4d(obs, dirs, T_A, 1.5, geo)                      # window (16:30, 18:00] -> 16Z obs excluded
    assert np.isnan(short.hx[:, 0]).all() and np.isfinite(short.hx[:, 1]).all()
    flagged = obs.with_qc(np.array([True, False]), ob.QC_DZ)
    c = fourd.hx_4d(flagged, dirs, T_A, 6.0, geo)
    assert np.isnan(c.hx[:, 0]).all() and np.isfinite(c.hx[:, 1]).all()


def test_hx_4d_refuses_alisios_files_without_guard(tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import alisios_guard

    geo, dirs = _members(tmp_path, n=1, missing=())
    monkeypatch.setattr(alisios_guard, "DATA_ROOTS", (tmp_path.resolve(),))
    with pytest.raises(PermissionError, match="ALISIOS data"):
        fourd.hx_4d(_obs(["2026-05-03T17:00"]), dirs, T_A, 6.0, geo)


def test_member_without_frames_in_window_gives_nan(tmp_path):
    geo, dirs = _members(tmp_path, n=2, missing=tuple((1, h) for h in range(13, 19)))   # member 1: only 12Z
    res = fourd.hx_4d(_obs(["2026-05-03T16:00"]), dirs, T_A, 6.0, geo)
    assert np.isfinite(res.hx[0, 0]) and np.isnan(res.hx[1, 0]) and res.files_opened[1] == []


def test_wrfout_frames_from_names_only(tmp_path):
    geo, dirs = _members(tmp_path, n=1, missing=())
    (dirs[0] / "wrfout_d01_2026-05-03_19:07:00").write_text("not netcdf")      # 7 min off a full hour: ignored
    inv = fourd.wrfout_frames(dirs[0])
    assert sorted(inv) == [np.datetime64(f"2026-05-03T{h:02d}:00", "s") for h in range(12, 19)]
    assert inv[np.datetime64("2026-05-03T14:00", "s")][1] == 36.0


def test_window_station_obs_hours(tmp_path):
    from test_obs import write_extract

    src = write_extract(tmp_path)
    b = fourd.window_station_obs(src, "2026-05-03T13:00", 2.0)
    assert set(np.unique(b.time).tolist()) <= {np.datetime64("2026-05-03T12:00", "s").item(),
                                                np.datetime64("2026-05-03T13:00", "s").item()}
    assert (b.dt_s <= 0).all() and (b.dt_s > -7200).all()
