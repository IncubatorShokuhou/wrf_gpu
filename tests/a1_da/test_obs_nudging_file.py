"""ObsBatch -> OBS_DOMAIN101 writer: round trip through the port's strict reader (gpuwrf.io.obs_domain). CPU only."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import obs as ob
from gpuwrf.assimilation.letkf import obs_nudging_file as onf
from gpuwrf.io.obs_domain import read_obs_domain

# ALISIOS read conditions (2026-10-17 01:14Z): obs are read ONLY from the guard_call-admitted copies in guarded_v1
GUARDED = Path("<DATA_ROOT>/wrf_gpu2/scratch/a1-da/obs/guarded_v1")
IGRA = GUARDED / "igra2_SPM00060018_admitted.txt"
STATIONS = GUARDED


def _batch():
    t12, t11 = np.datetime64("2026-05-04T12:00", "s"), np.datetime64("2026-05-04T11:00", "s")
    sonde = ob.ObsBatch.from_columns(
        var=np.array([ob.T, ob.QV, ob.U, ob.V, ob.U, ob.V, ob.T], np.int16),
        value=np.array([288.2, 0.0102, -3.25, 4.5, 7.0, -1.0, 270.4]), error=1.0, lat=28.3183, lon=-16.3822,
        p_pa=np.array([95000.0, 95000.0, 95000.0, 95000.0, 85000.0, 85000.0, 70000.0]),
        height_m=np.array([560.0, 560.0, 560.0, 560.0, 1500.0, 1500.0, 3100.0]), station="igra:SPM00060018",
        launch="SPM00060018@2026-05-04T12Z", time=t12)
    surf = ob.ObsBatch.from_columns(
        var=np.array([ob.T2, ob.Q2, ob.U10, ob.V10, ob.PSFC, ob.T2, ob.Q2, ob.T2, ob.T2], np.int16),
        value=np.array([291.0, 0.009, 2.0, -3.0, 94000.0, 289.5, 0.0085, 290.0, 299.0]), error=1.0,
        lat=np.array([28.47] * 5 + [28.10, 28.10, 28.2, 28.3]), lon=np.array([-16.33] * 5 + [-16.6, -16.6, -16.5, -16.4]),
        height_m=np.array([617.0] * 5 + [400.0, 400.0, 50.0, 60.0]),
        station=np.array(["grafcan:0001"] * 5 + ["grafcan:0002"] * 2 + ["grafcan:0003", "grafcan:0004"], dtype=object),
        time=np.array([t12] * 5 + [t11, t11, t12, t12], dtype="datetime64[s]"),
        role=np.array(["assimilate"] * 8 + ["withhold"], dtype=object),
        qc=np.array([0] * 7 + [ob.QC_DZ, 0]))
    return ob.concat([sonde, surf])


def test_round_trip_synthetic(tmp_path):
    b = _batch()
    receipt = onf.write_obs_domain(b, tmp_path / "OBS_DOMAIN101")
    recs = read_obs_domain(tmp_path / "OBS_DOMAIN101")
    assert receipt["records"] == len(recs) == 3 and receipt["soundings"] == 1
    assert [r.time for r in recs] == sorted(r.time for r in recs)
    assert recs[0].time == datetime(2026, 5, 4, 11) and recs[0].station_id == "grafcan:0002"
    snd = [r for r in recs if r.is_sounding][0]
    assert snd.measurement_count == 3 and snd.platform == "FM-35 TEMP" and abs(snd.latitude - 28.3183) < 5e-5
    vals = onf.records_to_values(recs)
    lv = {round(v["p"]): v for v in vals if v["sounding"]}
    assert lv[95000]["t"] == pytest.approx(288.2, abs=5e-4) and lv[95000]["q"] == pytest.approx(0.0102, rel=2e-5)
    assert lv[95000]["u"] == pytest.approx(-3.25) and lv[95000]["wind_qc"] == 129.0
    assert lv[85000]["t"] is None and lv[85000]["q"] is None and lv[85000]["v"] == pytest.approx(-1.0)
    assert lv[70000]["u"] is None and lv[70000]["t"] == pytest.approx(270.4)
    sf = {(v["station"], v["time"].hour): v for v in vals if not v["sounding"]}
    assert set(sf) == {("grafcan:0001", 12), ("grafcan:0002", 11)}       # QC-rejected and withheld excluded
    s1 = sf[("grafcan:0001", 12)]
    assert s1["p"] == pytest.approx(94000.0) and s1["q"] == pytest.approx(0.009, rel=2e-5) and s1["wind_qc"] == 129.0
    s2 = sf[("grafcan:0002", 11)]
    assert s2["p"] == pytest.approx(float(ob.std_atmosphere_pressure(400.0)), abs=5e-4) and s2["u"] is None
    assert s2["q"] == pytest.approx(0.0085, rel=2e-5) and receipt["psfc_from_std_atmosphere"] == 1
    # missing values are written as the WRF sentinel with sentinel QC
    raw = [m for m in recs[0].measurements[0]]
    assert raw[4].value == onf.MISSING and raw[4].qc == onf.MISSING and raw[8].value == onf.MISSING
    # withheld rows only with the explicit opt-in
    assert onf.write_obs_domain(b, tmp_path / "o2", include_withheld=True)["records"] == 4


def test_wrf_rh2r_and_format_limits(tmp_path):
    # WRF rh2r at 20 C / 1000 hPa (hand value): es = 6.1078 exp(17.2693882*20/(293.15-35.86)) = 23.38 hPa
    es = 6.1078 * np.exp(17.2693882 * (293.15 - 273.16) / (293.15 - 35.86))
    assert onf.wrf_rsat(293.15, 1000.0) == pytest.approx(0.62197 * es / (1000.0 - es))
    assert onf.wrf_rh2r(50.0, 293.15, 1000.0) == pytest.approx(0.5 * onf.wrf_rsat(293.15, 1000.0))
    with pytest.raises(ValueError, match="f11.3"):
        onf._f11_3(1.0e9)
    text = onf.namelist_fdda_text(fdda_end_min=720.0)
    assert "obs_nudge_opt        = 1," in text and "obs_coef_mois        = 0.0006," in text and text.startswith("&fdda")


@pytest.mark.skipif(not (IGRA.exists() and STATIONS.exists()), reason="guarded sonde/station copies absent")
def test_real_0504_obs_round_trip_recovers_every_value(tmp_path):
    t_a = "2026-05-04T12:00"
    sonde = ob.read_igra2(IGRA, "SPM00060018", t_a)
    surf = ob.read_station_obs(STATIONS, t_a, t_a, include_pressure=True)
    b = ob.concat([sonde, surf])
    receipt = onf.write_obs_domain(b, tmp_path / "OBS_DOMAIN101")
    recs = read_obs_domain(tmp_path / "OBS_DOMAIN101")
    vals = onf.records_to_values(recs)
    use = b.subset(b.usable & (b.role == "assimilate"))
    # every usable value is recovered at its location and time
    n_checked = 0
    for k in range(len(use)):
        var, x = int(use.var[k]), use.value[k]
        if var == ob.PSFC:
            continue
        if use.kind[k] == ob.SONDE:
            hits = [v for v in vals if v["sounding"] and abs(v["p"] - use.p_pa[k]) < 5e-4]
        else:
            hits = [v for v in vals if not v["sounding"] and v["station"] == use.station[k]]
        assert len(hits) == 1, (k, use.obs_id[k])
        v = hits[0]
        assert abs(v["lat"] - use.lat[k]) <= 5e-5 and abs(v["lon"] - use.lon[k]) <= 5e-5
        assert np.datetime64(v["time"], "s") == use.time[k]
        got = {ob.T: v["t"], ob.T2: v["t"], ob.U: v["u"], ob.U10: v["u"], ob.V: v["v"], ob.V10: v["v"],
               ob.QV: v["q"], ob.Q2: v["q"]}[var]
        if var in (ob.QV, ob.Q2):
            if got is None:                                  # q without T at that level cannot be written as RH
                assert not any((use.var == ob.T) & (use.p_pa == use.p_pa[k])) if use.kind[k] == ob.SONDE else False
                continue
            # format-implied bound: RH written f11.3 (0.0005 %) -> 5e-6 r_sat; T rounding (0.0005 K) -> ~4e-5 q
            p_rec = v["p"]
            bound = 5e-6 * float(onf.wrf_rsat(v["t"], p_rec / 100.0)) + 5e-5 * x
            assert abs(got - x) <= bound, (use.obs_id[k], got, x, bound)
        else:
            assert got is not None and abs(got - x) <= 5e-4, (use.obs_id[k], got, x)
        n_checked += 1
    assert n_checked > 100 and receipt["soundings"] == 1 and receipt["surface"] > 30
