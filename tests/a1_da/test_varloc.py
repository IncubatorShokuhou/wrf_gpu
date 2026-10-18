"""Variable localisation (lane w41-da-ens): per-group transforms with coupling factors; default path unchanged."""
from __future__ import annotations

from dataclasses import replace
import types

import numpy as np

from gpuwrf.assimilation.letkf import analysis as an
from gpuwrf.assimilation.letkf import varloc


def _setup(seed=0, n=12, nz=5, ny=30, nx=32):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(27.0 + 0.08 * np.arange(ny), -18.0 + 0.09 * np.arange(nx), indexing="ij")
    geo = {"XLAT": lat, "XLONG": lon, "HGT": np.zeros((ny, nx))}
    pb = (1.0e5 * np.exp(-0.12 * np.arange(nz)))[:, None, None] * np.ones((nz, ny, nx))
    common = rng.normal(size=(n, 1, 1, 1))     # makes T and U strongly correlated across members
    frames = {"T": 10 + common + 0.3 * rng.normal(size=(n, nz, ny, nx)),
              "QVAPOR": 0.008 + 0.001 * rng.normal(size=(n, nz, ny, nx)),
              "U": 2 * common + 0.3 * rng.normal(size=(n, nz, ny, nx + 1)),
              "V": rng.normal(size=(n, nz, ny + 1, nx)),
              "P": np.zeros((n, nz, ny, nx)), "PB": np.broadcast_to(pb, (n, nz, ny, nx)).copy(),
              "PSFC": np.full((n, ny, nx), 1.0e5)}
    return frames, geo


def _u10_obs(frames, geo, j, i, delta):
    value = frames["U"][:, 0, j, i].mean() + delta
    return types.SimpleNamespace(kind=np.array([2]), var=np.array([13]), value=np.array([value]),
                                 error=np.array([0.5]), qc=np.array([0]), lat=geo["XLAT"][[j], [i]],
                                 lon=geo["XLONG"][[j], [i]], p_pa=np.array([np.nan]), j=j, i=i)


def _hx(obs, frames, geo):
    return np.asarray(frames["U"])[:, 0, [obs.j], [obs.i]]


PARAMS = an.LetkfParams(stride_h=1, stride_v=1, rtps_alpha=0.0, rebalance=False, variables=("T", "QVAPOR", "U", "V"),
                        diagnostic_variables=(), loc_h_km=40.0, pmax=20)


NAMES = ("T", "QVAPOR", "U", "V")


def test_tables_convert_to_core_var_loc():
    vl = varloc.as_var_loc("surface_split", NAMES + ("T2", "U10"))
    assert vl["T"] == {"surface:U10": 0.0, "surface:V10": 0.0} and vl["QVAPOR"] == vl["T"] == vl["T2"]
    assert vl["U"] == {"surface:T2": 0.0, "surface:Q2": 0.0} and vl["V"] == vl["U"] == vl["U10"]
    strict = varloc.as_var_loc("strict", NAMES)
    assert strict["T"]["sonde:U"] == 0.0 and "sonde:T" not in strict["T"]
    from gpuwrf.assimilation.letkf.config import _localisation
    _localisation("config.localisation", {"variables": vl})          # accepted by the core config schema


def test_surface_wind_does_not_update_temperature_with_surface_split():
    frames, geo = _setup()
    obs = _u10_obs(frames, geo, 15, 16, 1.5)
    plain, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    split, receipt = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx,
                                var_loc=varloc.as_var_loc("surface_split", NAMES))
    assert np.abs(plain["T"] - frames["T"]).max() > 1e-3          # spurious cross-update without varloc
    np.testing.assert_array_equal(split["T"], frames["T"])           # none with it
    np.testing.assert_array_equal(split["QVAPOR"], frames["QVAPOR"])
    np.testing.assert_array_equal(split["U"], plain["U"])           # wind group = the plain transform for this obs
    assert sorted(map(sorted, receipt["hooks"]["var_loc_groups"])) == [["QVAPOR", "T"], ["U", "V"]]


def test_default_maintenance_is_bitwise_the_probe_rtps():
    from gpuwrf.assimilation.letkf.spread import make_maintenance
    frames, geo = _setup(2)
    obs = _u10_obs(frames, geo, 14, 15, 0.8)
    p = replace(PARAMS, rtps_alpha=0.8)
    a, _ = an.analyse(frames, geo, obs, p, model_equivalents=_hx)
    for spec in (None, {"method": "rtps", "alpha": 0.8, "target": "inflated", "rho": 1.0}):
        b, r = an.analyse(frames, geo, obs, p, model_equivalents=_hx, maintenance=make_maintenance(spec))
        for name in NAMES:
            np.testing.assert_array_equal(a[name], b[name])
        assert r["hooks"]["maintenance"].startswith("w41:rtps")


def test_additive_maintenance_is_tapered_seeded_and_centred():
    from gpuwrf.assimilation.letkf.spread import make_maintenance
    frames, geo = _setup(3)
    obs = _u10_obs(frames, geo, 14, 15, 0.8)
    rng = np.random.default_rng(9)
    pool = {"T": rng.normal(size=(6,) + frames["T"].shape[1:])}
    pool["T"] -= pool["T"].mean(0)
    hook = lambda: make_maintenance({"method": "rtps", "alpha": 0.0}, pool=pool, amplitude=0.5, seed=3)  # noqa: E731
    a, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx, maintenance=hook())
    b, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx, maintenance=hook())
    base, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    np.testing.assert_array_equal(a["T"], b["T"])
    d = a["T"] - base["T"]
    assert np.all(d[..., :5, :] == 0) and np.all(d[..., :, -5:] == 0)   # specified zone untouched (analysis taper)
    assert np.abs(d[..., 12:18, 12:18]).max() > 0.1
    np.testing.assert_allclose(a["T"].mean(0), base["T"].mean(0), atol=1e-10)
    np.testing.assert_array_equal(a["U"], base["U"])                  # not in the pool: unchanged
