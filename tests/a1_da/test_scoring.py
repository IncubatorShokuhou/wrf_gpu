"""Tests for gpuwrf.assimilation.letkf.scoring (V5 pre-registered scoring; lane a1-obs). CPU only."""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from gpuwrf.assimilation.letkf import obs as ob
from gpuwrf.assimilation.letkf import scoring as sc

FAST = dataclasses.replace(sc.PROTOCOL, n_boot=400)


# ----------------------------------------------------------------------------- holdouts / leakage
def test_holdout_rule_is_deterministic_order_free_and_near_fraction():
    sites = [f"grafcan:{k:04d}" for k in range(3000)]
    a = {s for s in sites if sc.is_holdout(s)}
    b = {s for s in reversed(sites) if sc.is_holdout(s)}
    assert a == b and abs(len(a) / 3000 - 1 / 3) < 0.03
    assert sc.holdout_score("grafcan:0001") == sc.holdout_score("grafcan:0001")
    other = dataclasses.replace(sc.PROTOCOL, holdout_seed="different")
    assert {s for s in sites if sc.is_holdout(s, other)} != a
    st = pd.DataFrame({"station_id": ["aemet:C447A", "metar:60015", "grafcan:0001"], "lat": [0.0] * 3,
                       "lon": [0.0] * 3, "elevation_m": [600.0, 600.0, 10.0]})
    table = sc.holdout_table(st)
    assert [r["site"] for r in table] == ["grafcan:0001", "site:aemet:C447A"]
    assert table[1]["members"] == ["aemet:C447A", "metar:60015"]


def test_frozen_holdout_list_matches_rule_and_hash():
    import hashlib
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "V5_HOLDOUT_SITES.json"   # frozen copy of the A1 receipt
    raw = path.read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == "e3d51823d711806a6bfa7b6b532c14ade9b43fa463e47df10f5aef5094ff1cdf"
    frozen = json.loads(raw)
    assert frozen["seed"] == sc.PROTOCOL.holdout_seed and frozen["n_holdout"] == 17 and frozen["n_sites"] == 61
    for row in frozen["sites"]:
        assert row["holdout"] == sc.is_holdout(row["site"]), row["site"]


def _surface_batch(sites, values, var=ob.T2, time="2026-05-04T18:00", role="assimilate"):
    n = len(sites)
    return ob.ObsBatch.from_columns(var=np.full(n, var, np.int16), value=np.asarray(values, float), error=1.0,
                                    lat=np.zeros(n), lon=np.zeros(n), height_m=np.zeros(n),
                                    station=np.array(sites, dtype=object), site=np.array(sites, dtype=object),
                                    time=np.datetime64(time, "s"), role=role)


def test_leakage_refused():
    scored = _surface_batch(["grafcan:0002"], [290.0])
    clean = _surface_batch(["grafcan:0003"], [290.0], time="2026-05-04T12:00")
    assert sc.check_no_leakage(scored, {"A2": clean}, {"grafcan:0002"})
    with pytest.raises(ValueError, match="holdout sites"):
        sc.check_no_leakage(scored, {"A2": _surface_batch(["grafcan:0002"], [1.0], time="2026-05-04T12:00")},
                            {"grafcan:0002"})
    with pytest.raises(ValueError, match="scored observations"):
        sc.check_no_leakage(scored, {"A3": scored}, set())
    # withheld rows in an arm receipt are not assimilation
    sc.check_no_leakage(scored, {"A2": scored.with_role(np.array([True]), "withhold")}, {"grafcan:0002"})


# ----------------------------------------------------------------------------- pairs on analytic frames
def _grid(ny=8, nx=9):
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    geo = {"XLAT": 28.0 + 0.05 * j, "XLONG": -17.0 + 0.05 * i, "HGT": np.full((ny, nx), 100.0),
           "COSALPHA": np.ones((ny, nx)), "SINALPHA": np.zeros((ny, nx)), "LANDMASK": np.ones((ny, nx)), "DX": 9000.0}
    return geo, i, j


def _frames(geo, i, j, t2_err=0.0, q2_err=0.0, u_err=0.0, v_err=0.0):
    t2 = 290.0 + 0.1 * i
    return {"T2": (t2 + t2_err)[None], "Q2": (0.008 + q2_err + 0 * i)[None], "U10": (2.0 + u_err + 0 * i)[None],
            "V10": (1.0 + v_err + 0 * i)[None], "PSFC": (1.0e5 + 0 * i)[None]}


def _obs_at(sites, lats, lons, geo, time="2026-05-04T18:00"):
    n = len(sites)
    i_pt = (np.asarray(lons) + 17.0) / 0.05
    t2 = 290.0 + 0.1 * i_pt
    parts = []
    for var, val in ((ob.T2, t2), (ob.Q2, np.full(n, 0.008)), (ob.U10, np.full(n, 2.0)), (ob.V10, np.full(n, 1.0))):
        parts.append(ob.ObsBatch.from_columns(var=np.full(n, var, np.int16), value=val, error=1.0, lat=lats, lon=lons,
                                              height_m=np.full(n, 100.0), station=np.array(sites, dtype=object),
                                              site=np.array(sites, dtype=object), time=np.datetime64(time, "s")))
    return ob.concat(parts)


def test_surface_pairs_exact_losses_and_holdout_filter():
    geo, i, j = _grid()
    sites = ["grafcan:0001", "grafcan:0002", "grafcan:0003"]
    obs = _obs_at(sites, [28.1, 28.2, 28.15], [-16.9, -16.8, -16.75], geo)
    fr = _frames(geo, i, j, t2_err=0.5, q2_err=0.001, u_err=3.0, v_err=4.0)
    pairs = sc.surface_pairs(obs, fr, geo, arm="A2", case="20260504", block="b1", lead_h=6,
                             holdout_sites={"grafcan:0001", "grafcan:0003"})
    assert set(pairs.unit) == {"grafcan:0001", "grafcan:0003"}
    t2 = pairs[pairs["var"] == "T2"]
    assert np.allclose(t2.loss, 0.25) and np.allclose(pairs[pairs["var"] == "Q2"].loss, 1.0)
    w = pairs[pairs["var"] == "WIND10"]
    assert len(w) == 2 and np.allclose(w.loss, 25.0)            # (3, 4) vector error
    # geometry rejects are dropped and counted, a NaN forecast is kept (-> INCOMPLETE later)
    far = _obs_at(["grafcan:0009"], [40.0], [-16.8], geo)
    p2 = sc.surface_pairs(far, fr, geo, arm="A2", case="c", block="b", lead_h=6, holdout_sites={"grafcan:0009"})
    assert len(p2) == 0 and p2.attrs["geometry_dropped"] == 4
    fr_nan = dict(fr, T2=np.full_like(fr["T2"], np.nan))
    p3 = sc.surface_pairs(obs, fr_nan, geo, arm="A2", case="c", block="b", lead_h=6, holdout_sites={"grafcan:0001"})
    assert np.isnan(p3[p3["var"] == "T2"].loss).all()
    with pytest.raises(ValueError, match="2026-07-01"):
        sc.surface_pairs(_obs_at(sites, [28.1] * 3, [-16.9] * 3, geo, time="2026-07-02T00:00"), fr, geo, arm="A2",
                         case="c", block="b", lead_h=6, holdout_sites=set(sites))


def test_inversion_pair_rules():
    hit = sc.inversion_pair({"twi_found": True, "twi_base_m": 900.0}, {"twi_found": True, "twi_base_m": 1100.0},
                            arm="A2", case="c", block="b", lead_h=12, launch="L")
    miss = sc.inversion_pair({"twi_found": True, "twi_base_m": 900.0}, {"twi_found": False},
                             arm="A2", case="c", block="b", lead_h=12, launch="L")
    short = sc.inversion_pair({"twi_found": True, "twi_base_m": 900.0}, {}, arm="A2", case="c", block="b",
                              lead_h=12, launch="L")
    none = sc.inversion_pair({"twi_found": False}, {"twi_found": True, "twi_base_m": 500.0}, arm="A2", case="c",
                             block="b", lead_h=12, launch="L")
    assert hit.loss.tolist() == [200.0] and miss.loss.tolist() == [1000.0] and short.loss.tolist() == [1000.0]
    assert len(none) == 0


# ----------------------------------------------------------------------------- comparisons
def _pairs(n_blocks=10, n_sites=15, ratio=0.5, var="T2", lead=6, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        for s in range(n_sites):
            lr = rng.gamma(2.0, 1.0)
            la = max(lr * ratio * (1 + noise * rng.standard_normal()), 0.0)
            for arm, loss in (("A2_letkf", la), ("A1_zero_update", lr)):
                rows.append((arm, f"case{b}", f"blk{b}", f"site{s}", var, lead, f"site{s}|t", 0.0, 0.0, loss))
    return pd.DataFrame(rows, columns=sc._COLUMNS)


def test_compare_arms_verdicts():
    better = sc.compare_arms(_pairs(ratio=0.7), "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert better["verdict"] == "IMPROVE" and better["ci_upper"] < 0 and better["rel_delta"] == pytest.approx(-0.3, abs=0.02)
    same = sc.compare_arms(_pairs(ratio=1.0, noise=0.0), "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert same["verdict"] == "NONINFERIOR" and same["rel_delta"] == 0.0
    worse = sc.compare_arms(_pairs(ratio=1.3), "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert worse["verdict"] == "FAIL"
    slightly = sc.compare_arms(_pairs(ratio=1.01, noise=0.0), "A2_letkf", "A1_zero_update", var="T2", lead_h=6,
                               protocol=FAST)
    assert slightly["verdict"] == "NONINFERIOR"                  # +1 % < 5 % surface margin
    ni = sc.compare_arms(_pairs(ratio=0.7), "A2_letkf", "A1_zero_update", var="T2", lead_h=6, kind="noninferiority",
                         protocol=FAST)
    assert ni["verdict"] == "NONINFERIOR"                         # noninferiority tests never claim improvement


def test_compare_arms_incomplete_rules():
    p = _pairs()
    p.loc[3, "loss"] = np.nan                                     # E200: one NaN -> INCOMPLETE, never PASS
    r = sc.compare_arms(p, "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert r["verdict"] == "INCOMPLETE" and "non-finite matched losses" in r["reasons"][0]
    p = _pairs().drop(index=0)                                    # pair missing in one arm
    r = sc.compare_arms(p, "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert r["verdict"] == "INCOMPLETE" and "only one arm" in r["reasons"][0]
    r = sc.compare_arms(_pairs(n_blocks=7), "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert r["verdict"] == "INCOMPLETE" and any("blocks" in x for x in r["reasons"])
    r = sc.compare_arms(_pairs(n_blocks=8, n_sites=12), "A2_letkf", "A1_zero_update", var="T2", lead_h=6,
                        protocol=FAST)
    assert r["verdict"] == "INCOMPLETE" and any("pairs" in x for x in r["reasons"])   # 96 < 100
    empty = sc.compare_arms(_pairs().iloc[:0], "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert empty["verdict"] == "INCOMPLETE"
    dup = pd.concat([_pairs(), _pairs().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        sc.compare_arms(dup, "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)


def test_cluster_bootstrap_is_paired_seeded_and_block_aware():
    p = _pairs(ratio=0.8)
    a, r = p[p.arm == "A2_letkf"], p[p.arm == "A1_zero_update"]
    args = (a.loss.to_numpy(), r.loss.to_numpy(), a.block.to_numpy(), a.unit.to_numpy())
    b1 = sc.cluster_bootstrap_rel(*args, n_boot=300, seed=1)
    b2 = sc.cluster_bootstrap_rel(*args, n_boot=300, seed=1)
    assert np.array_equal(b1, b2)
    # constant ratio in every unit -> every resample gives exactly the point estimate (paired draws)
    q = _pairs(ratio=0.8, noise=0.0)
    a, r = q[q.arm == "A2_letkf"], q[q.arm == "A1_zero_update"]
    b = sc.cluster_bootstrap_rel(a.loss.to_numpy(), r.loss.to_numpy(), a.block.to_numpy(), a.unit.to_numpy(),
                                 n_boot=200, seed=3)
    assert np.allclose(b, -0.2)
    # one block carries the whole improvement -> block resampling widens the interval vs pooled units
    rows = _pairs(ratio=1.0, noise=0.0)
    rows.loc[(rows.arm == "A2_letkf") & (rows.block == "blk0"), "loss"] *= 0.01
    a, r = rows[rows.arm == "A2_letkf"], rows[rows.arm == "A1_zero_update"]
    blocked = sc.cluster_bootstrap_rel(a.loss.to_numpy(), r.loss.to_numpy(), a.block.to_numpy(), a.unit.to_numpy(),
                                       n_boot=500, seed=4)
    pooled = sc.cluster_bootstrap_rel(a.loss.to_numpy(), r.loss.to_numpy(), np.zeros(len(a)), np.arange(len(a)),
                                      n_boot=500, seed=4)
    assert np.std(blocked) > 1.5 * np.std(pooled)


def test_lead_windows():
    assert [sc.lead_window(h) for h in (0, 1, 3, 4, 6, 7, 12, 13, 24, 25)] == [None, 3, 3, 6, 6, 12, 12, 24, 24, None]


def test_holm():
    adj = sc.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_evaluate_primary_status():
    frames = []
    for var, lead in sc.PROTOCOL.primary:
        if var in sc.SURFACE_VARS:
            frames.append(_pairs(ratio=0.7, var=var, lead=lead))
        else:
            frames.append(_pairs(n_blocks=10, n_sites=3, ratio=0.7, var=var, lead=lead).assign(
                unit=lambda d: d.case + d.unit))
    pairs = pd.concat(frames, ignore_index=True)
    rep = sc.evaluate(pairs, FAST, comparisons=sc.COMPARISONS[:1])
    prim = [r for r in rep["results"] if r["primary"]]
    assert len(prim) == 4 and rep["primary_status"] == "IMPROVE" and all("p_holm" in r for r in prim)
    # drop one primary variable entirely -> INCOMPLETE overall
    rep2 = sc.evaluate(pairs[pairs["var"] != "Q2"], FAST, comparisons=sc.COMPARISONS[:1])
    assert rep2["primary_status"] == "INCOMPLETE"


# ----------------------------------------------------------------------------- wrfout IO end to end
def _write_wrfout(path, geo, fields, times="2026-05-04_18:00:00"):
    netCDF4 = pytest.importorskip("netCDF4")
    ny, nx = geo["XLAT"].shape
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.setncattr("DX", np.float32(geo["DX"]))
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.array(list(times), dtype="S1")
        for name, arr in list(fields.items()) + [(k, geo[k]) for k in sc.GEO_FIELDS]:
            v = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[0] = np.asarray(arr).reshape(ny, nx)


def test_read_wrfout_and_score_end_to_end(tmp_path):
    geo, i, j = _grid()
    good = {k: v[0] for k, v in _frames(geo, i, j, t2_err=0.3).items()}
    base = {k: v[0] for k, v in _frames(geo, i, j, t2_err=1.0).items()}
    _write_wrfout(tmp_path / "a2.nc", geo, good)
    _write_wrfout(tmp_path / "a1.nc", geo, base)
    fa, ga = sc.read_wrfout(tmp_path / "a2.nc")
    fr, _ = sc.read_wrfout(tmp_path / "a1.nc")
    assert fa["T2"].shape == (1,) + geo["XLAT"].shape and ga["Times"] == "2026-05-04_18:00:00"
    rng = np.random.default_rng(0)
    rows = []
    for b in range(9):
        sites = [f"grafcan:{b:02d}{s:02d}" for s in range(14)]
        lats, lons = rng.uniform(28.05, 28.3, 14), rng.uniform(-16.95, -16.65, 14)
        obs = _obs_at(sites, lats, lons, geo)
        for arm, fr_ in (("A2_letkf", fa), ("A1_zero_update", fr)):
            rows.append(sc.surface_pairs(obs, fr_, ga, arm=arm, case=f"c{b}", block=f"b{b}", lead_h=6,
                                         holdout_sites=set(sites)))
    pairs = pd.concat(rows, ignore_index=True)
    res = sc.compare_arms(pairs, "A2_letkf", "A1_zero_update", var="T2", lead_h=6, protocol=FAST)
    assert res["verdict"] == "IMPROVE" and res["rel_delta"] == pytest.approx(0.09 - 1.0, abs=1e-5)
    # float32 storage: masked/fill values are refused
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(tmp_path / "a1.nc", "a") as ds:
        ds.variables["T2"][0, 0, 0] = np.ma.masked
    with pytest.raises(ValueError, match="masked"):
        sc.read_wrfout(tmp_path / "a1.nc")


def test_case_pairs_windows_missing_files_and_sonde(tmp_path):
    from gpuwrf.assimilation.letkf import diagnostics as dg
    from test_a1_diagnostics import _column_frames

    geo, i, j = _grid()
    t_f = np.datetime64("2026-05-04T12:00", "s")
    sites = ["grafcan:0001", "grafcan:0002"]
    obs = ob.concat([_obs_at(sites, [28.1, 28.2], [-16.9, -16.8], geo, time=str(t_f + np.timedelta64(h, "h")))
                     for h in (2, 5, 9, 20, 30)])
    store = {t_f + np.timedelta64(h, "h"): f"f{h}" for h in (2, 5, 9, 20)}
    fr = _frames(geo, i, j, t2_err=0.5)
    pairs = sc.case_pairs("A2_letkf", "c1", "b1", t_f, store, obs, set(sites), read=lambda path: (fr, geo))
    t2 = pairs[pairs["var"] == "T2"]
    assert sorted(t2.lead_h.unique().tolist()) == [3, 6, 12, 24] and len(t2) == 8 and np.allclose(t2.loss, 0.25)
    del store[t_f + np.timedelta64(9, "h")]
    p2 = sc.case_pairs("A2_letkf", "c1", "b1", t_f, store, obs, set(sites), read=lambda path: (fr, geo))
    assert p2.attrs["missing_valid_times"] == [str(t_f + np.timedelta64(9, "h"))] and 12 not in set(p2.lead_h)
    # sonde +12: profile pairs within 0-3 km and a TWI pair from the model column
    cframes, cgeo, zm, t_c, rh = _column_frames(nmem=1)
    sonde = ob.ObsBatch.from_columns(var=np.array([ob.T, ob.QV, ob.T], np.int16),
                                     value=np.array([290.0, 0.01, 280.0]), error=1.0, lat=28.2, lon=-16.7,
                                     p_pa=1.0e5 * np.exp(-np.array([500.0, 500.0, 3500.0]) / 8000.0),
                                     height_m=np.array([500.0, 500.0, 3500.0]), station="igra:S", launch="S@L")
    store3 = {t_f + np.timedelta64(12, "h"): "fx"}
    p3 = sc.case_pairs("A2_letkf", "c1", "b1", t_f, store3, obs.subset(obs.var < 0), set(),
                       sondes={12: (sonde, {"twi_found": True, "twi_base_m": 700.0})},
                       read=lambda path: (cframes, cgeo))
    assert sorted(p3["var"].tolist()) == ["SONDE_QV", "SONDE_T", "TWI_BASE"]
    assert p3[p3["var"] == "TWI_BASE"].loss.tolist() == [pytest.approx(100.0)]       # model base 800 m vs 700 m
    with pytest.raises(ValueError, match="not pre-registered"):
        sc.case_pairs("A2_letkf", "c1", "b1", t_f, store3, obs, set(), sondes={6: (sonde, {})},
                      read=lambda path: (cframes, cgeo))


def test_wrfout_inventory_rounds_d01_offsets(tmp_path):
    geo, i, j = _grid()
    fields = {k: v[0] for k, v in _frames(geo, i, j).items()}
    _write_wrfout(tmp_path / "a.nc", geo, fields, times="2026-05-04_13:00:18")
    _write_wrfout(tmp_path / "b.nc", geo, fields, times="2026-05-04_14:00:36")
    inv = sc.wrfout_inventory([tmp_path / "a.nc", tmp_path / "b.nc"])
    assert sorted(inv) == [np.datetime64("2026-05-04T13:00:00"), np.datetime64("2026-05-04T14:00:00")]
    _write_wrfout(tmp_path / "c.nc", geo, fields, times="2026-05-04_14:05:00")
    with pytest.raises(ValueError, match="full hour"):
        sc.wrfout_inventory([tmp_path / "c.nc"])
