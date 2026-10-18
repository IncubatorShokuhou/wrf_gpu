"""analysis.analyse end to end on a synthetic WRF-like grid (lane a1-da, DESIGN_v2 V0/V3 plumbing).

The operator is injected (point sampling of T at a mass point), so these tests
check localisation, update, taper, QC and withheld semantics independently of
a1-obs's operators. Oracles: the scalar ensemble Kalman gain, GC support.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import analysis as an
from gpuwrf.assimilation.letkf import core


def _stub_state_io(monkeypatch):
    try:
        from gpuwrf.assimilation.letkf import state_io  # noqa: F401
        return
    except ImportError:
        pass

    def boundary_taper(ny, nx, *, spec_bdy_width=5, ramp=5, stagger=None):
        shape = (ny + (stagger == "y"), nx + (stagger == "x"))
        j = np.minimum(np.arange(shape[0]), shape[0] - 1 - np.arange(shape[0]))
        i = np.minimum(np.arange(shape[1]), shape[1] - 1 - np.arange(shape[1]))
        d = np.minimum(j[:, None], i[None, :]).astype(float)
        x = np.clip((d - spec_bdy_width + 1) / (ramp + 1), 0.0, 1.0)
        return np.where(d < spec_bdy_width, 0.0, 0.5 - 0.5 * np.cos(np.pi * x))

    module = types.ModuleType("gpuwrf.assimilation.letkf.state_io")
    module.boundary_taper = boundary_taper
    monkeypatch.setitem(sys.modules, "gpuwrf.assimilation.letkf.state_io", module)


def _setup(seed=0, n=10, nz=6, ny=32, nx=36):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(27.0 + 0.08 * np.arange(ny), -18.0 + 0.09 * np.arange(nx), indexing="ij")
    geo = {"XLAT": lat, "XLONG": lon, "HGT": np.zeros((ny, nx))}
    pb = (1.0e5 * np.exp(-0.12 * np.arange(nz)))[:, None, None] * np.ones((nz, ny, nx))
    smooth = lambda a: (a + np.roll(a, 1, -1) + np.roll(a, -1, -1) + np.roll(a, 1, -2) + np.roll(a, -1, -2)) / 5  # noqa: E731
    frames = {
        "T": 10 + np.stack([smooth(smooth(rng.normal(size=(nz, ny, nx)))) for _ in range(n)]),
        "QVAPOR": 0.008 + 0.001 * rng.normal(size=(n, nz, ny, nx)),
        "U": rng.normal(size=(n, nz, ny, nx + 1)), "V": rng.normal(size=(n, nz, ny + 1, nx)),
        "P": np.zeros((n, nz, ny, nx)), "PB": np.broadcast_to(pb, (n, nz, ny, nx)).copy(),
        "PSFC": np.full((n, ny, nx), 1.0e5),
    }
    return frames, geo


class _Obs(types.SimpleNamespace):
    pass


def _obs(points, values, sigma):
    """points: list of (k, j, i); sonde-kind T observations at mass points."""
    k, j, i = (np.array(x) for x in zip(*points))
    return _Obs(kind=np.ones(len(k), int), var=np.ones(len(k), int), value=np.asarray(values, float),
                error=np.asarray(sigma, float), k=k, j=j, i=i, qc=np.zeros(len(k), int))


def _bind(obs, frames, geo):
    obs.lat = geo["XLAT"][obs.j, obs.i]
    obs.lon = geo["XLONG"][obs.j, obs.i]
    obs.p_pa = frames["PB"][0][obs.k, obs.j, obs.i]
    return obs


def _hx(obs, frames, geo):
    return np.asarray(frames["T"])[:, obs.k, obs.j, obs.i]


PARAMS = an.LetkfParams(stride_h=1, stride_v=1, rtps_alpha=0.0, rebalance=False, variables=("T", "QVAPOR", "U", "V"),
                        diagnostic_variables=(), loc_h_km=30.0, loc_v_lnp=0.1, loc_v_lnp_low=0.1, pmax=50)


def test_single_observation_increment_matches_scalar_kalman(monkeypatch):
    _stub_state_io(monkeypatch)
    frames, geo = _setup()
    obs = _bind(_obs([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean() + 1.5], [0.5]), frames, geo)
    analysis, receipt = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    hx = frames["T"][:, 2, 16, 18]
    for name, idx in (("T", (2, 16, 18)), ("QVAPOR", (2, 16, 18)), ("T", (2, 16, 19))):
        x = frames[name][(slice(None),) + idx]
        rho = 1.0 if idx == (2, 16, 18) else None
        if rho is None:  # neighbouring column: obs-to-point distance gives the GC weight
            d = np.linalg.norm(core_xyz(geo, 16, 19) - core_xyz(geo, 16, 18))
            rho = float(core.gaspari_cohn(d / core.gc_halfwidth(PARAMS.loc_h_km)))
        cov = np.cov(x, hx, ddof=1)[0, 1]
        gain = cov / (hx.var(ddof=1) + 0.25 / rho)
        expect = x.mean() + gain * (obs.value[0] - hx.mean())
        assert abs(analysis[name][(slice(None),) + idx].mean() - expect) < 1e-9 * max(1.0, abs(expect))
    assert receipt["assimilated"] == 1 and receipt["admissible"]


def core_xyz(geo, j, i):
    from gpuwrf.assimilation.letkf.grid import lonlat_to_xyz_km
    return lonlat_to_xyz_km(geo["XLAT"][j, i], geo["XLONG"][j, i])


def test_localisation_support_and_boundary_taper(monkeypatch):
    _stub_state_io(monkeypatch)
    frames, geo = _setup(1)
    obs = _bind(_obs([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean() + 1.5], [0.5]), frames, geo)
    analysis, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    inc = analysis["T"] - frames["T"]
    d = np.linalg.norm(core_xyz(geo, slice(None), slice(None)) - core_xyz(geo, 16, 18), axis=-1)
    far = d >= 2 * core.gc_halfwidth(PARAMS.loc_h_km)
    assert far.any() and np.all(inc[:, :, far] == 0.0)
    assert np.abs(inc[:, 2, 16, 18]).max() > 0, "observation must be assimilated (not gross-rejected)"
    # obs placed in the specified zone: no increment there
    obs2 = _bind(_obs([(2, 2, 18)], [frames["T"][:, 2, 2, 18].mean() + 1.5], [0.5]), frames, geo)
    analysis2, receipt2 = an.analyse(frames, geo, obs2, PARAMS, model_equivalents=_hx)
    assert receipt2["assimilated"] == 1
    inc2 = analysis2["T"] - frames["T"]
    assert np.abs(inc2[:, :, 5:12, :]).max() > 0
    assert np.all(inc2[:, :, :5, :] == 0.0) and np.all(inc2[:, :, -5:, :] == 0.0)


def test_withheld_observations_do_not_change_the_analysis(monkeypatch):
    _stub_state_io(monkeypatch)
    frames, geo = _setup(2)
    pts = [(2, 16, 18), (3, 19, 14)]
    vals = [frames["T"][:, k, j, i].mean() + 1.0 for k, j, i in pts]
    both = _bind(_obs(pts, vals, [0.5, 0.5]), frames, geo)
    one = _bind(_obs(pts[:1], vals[:1], [0.5]), frames, geo)
    a_with, r_with = an.analyse(frames, geo, both, PARAMS, withheld=np.array([False, True]), model_equivalents=_hx)
    a_one, _ = an.analyse(frames, geo, one, PARAMS, model_equivalents=_hx)
    np.testing.assert_array_equal(a_with["T"], a_one["T"])
    assert r_with["withheld_ok"] == 1 and "sonde:1" in r_with["stats_withheld"]


def test_gross_check_rejects_outlier(monkeypatch):
    _stub_state_io(monkeypatch)
    frames, geo = _setup(3)
    obs = _bind(_obs([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean() + 50.0], [0.5]), frames, geo)
    analysis, receipt = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    assert receipt["qc"]["rejected_gross"] == 1 and receipt["assimilated"] == 0
    np.testing.assert_allclose(analysis["T"], frames["T"], rtol=0, atol=1e-12)


def test_rtps_inflates_toward_prior_spread(monkeypatch):
    _stub_state_io(monkeypatch)
    frames, geo = _setup(4)
    obs = _bind(_obs([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean()], [0.3]), frames, geo)
    from dataclasses import replace
    a0, _ = an.analyse(frames, geo, obs, PARAMS, model_equivalents=_hx)
    a1, _ = an.analyse(frames, geo, obs, replace(PARAMS, rtps_alpha=1.0), model_equivalents=_hx)
    s_b, s0, s1 = (x["T"][:, 2, 16, 18].std(ddof=1) for x in (frames, a0, a1))
    assert s0 < s_b and abs(s1 - s_b) < 1e-12
    np.testing.assert_allclose(a1["T"].mean(0), a0["T"].mean(0), atol=1e-12)


@pytest.mark.parametrize("policy", ["cap", "to_cloud"])
def test_supersaturation_cap_and_to_cloud_are_refused(monkeypatch, policy):
    # w41-da-core critic item 4: applied after the boundary taper they changed protected cells, the fixed-QV
    # rebalance does not load condensate, and member files drop QCLOUD -> refused in this build ('keep' only)
    _stub_state_io(monkeypatch)
    from dataclasses import replace
    frames, geo = _setup(5)
    obs = _bind(_obs([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean() + 0.5], [0.5]), frames, geo)
    with pytest.raises(ValueError, match="refused in this build"):
        an.analyse(frames, geo, obs, replace(PARAMS, supersaturation=policy), model_equivalents=_hx)
