"""Balanced perturbations (lane w41-da-ens): MU-including hydrostatic rebalance vs the port's real.exe balance oracle,
Gaussian pool combinations, recentring, balance diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import perturb, state_io
from test_state_io import blob, synthetic_background, synthetic_geo


@pytest.fixture(scope="module")
def syn():
    geo = synthetic_geo(nz=20, ny=12, nx=14, dtype=np.float64)
    return geo, synthetic_background(geo, seed=0, dtype=np.float64)


def _exact_bg(geo, bg):
    """A background that IS the real.exe balance of (θ, qv, μ): P, PH from the oracle (no non-hydrostatic part)."""
    bal = state_io.balance_state(bg["T"], bg["QVAPOR"], bg["MU"], geo)
    return {**bg, "P": bal["P"], "PH": bal["PH"], "P_HYD": bal["P"] + geo["PB"]}


def test_zero_mu_increment_reproduces_fixed_mu_rebalance(syn):
    geo, bg = syn
    an = {**bg, "T": bg["T"] + 1.2 * blob(bg["MU"].shape)[None], "QVAPOR": bg["QVAPOR"] * (1 + 0.05 * blob(bg["MU"].shape))}
    ref = state_io.rebalance_increment(bg, an, geo)
    out = perturb.rebalance_with_mu(bg, an, geo)
    for name in ("P", "PH", "P_HYD"):
        np.testing.assert_allclose(out[name], ref[name], rtol=1e-9, atol=1e-7)
    np.testing.assert_array_equal(out["MU"], bg["MU"])


def test_pressure_recursion_matches_balance_oracle(syn):
    geo, bg = syn
    bal = state_io.balance_state(bg["T"], bg["QVAPOR"], bg["MU"], geo)
    p = perturb.pressure_recursion(bg["MU"], bg["QVAPOR"], geo)
    np.testing.assert_allclose(p, bal["P"], rtol=0, atol=1e-6 * float(np.abs(geo["PB"]).max()))


@pytest.mark.parametrize("dmu", [400.0, -250.0])
def test_mu_increment_matches_balance_oracle(syn, dmu):
    geo, bg0 = syn
    bg = _exact_bg(geo, bg0)
    shape = bg["MU"].shape
    an = {**bg, "MU": bg["MU"] + dmu * blob(shape), "T": bg["T"] + 0.8 * blob(shape, width=3.5)[None]}
    out = perturb.rebalance_with_mu(bg, an, geo)
    oracle = state_io.balance_state(an["T"], an["QVAPOR"], an["MU"], geo)
    scale_p = float(np.abs(out["P"] - bg["P"]).max())
    scale_ph = float(np.abs(out["PH"] - bg["PH"]).max())
    assert scale_p > 10.0 and scale_ph > 10.0
    np.testing.assert_allclose(out["P"], oracle["P"], rtol=0, atol=1e-6 * scale_p)
    np.testing.assert_allclose(out["PH"], oracle["PH"], rtol=0, atol=1e-6 * scale_ph)
    assert perturb.hydrostatic_residual_change(bg, an, out, geo) < 1e-9


def test_mutant_without_mu_in_hypsometric_factor_is_caught(syn, monkeypatch):
    geo, bg0 = syn
    bg = _exact_bg(geo, bg0)
    an = {**bg, "MU": bg["MU"] + 400.0 * blob(bg["MU"].shape)}
    good = perturb.rebalance_with_mu(bg, an, geo)
    real = perturb._hyps2
    monkeypatch.setattr(perturb, "_hyps2", lambda mu, g: real(bg["MU"], g))   # mutant: frozen-MU layer factor
    bad = perturb.rebalance_with_mu(bg, an, geo)
    oracle = state_io.balance_state(an["T"], an["QVAPOR"], an["MU"], geo)
    tol = 1e-6 * float(np.abs(good["PH"] - bg["PH"]).max())
    assert np.abs(good["PH"] - oracle["PH"]).max() < tol < np.abs(bad["PH"] - oracle["PH"]).max()


def test_refusals(syn):
    geo, bg = syn
    with pytest.raises(ValueError):
        perturb.rebalance_with_mu(bg, {**bg, "QVAPOR": -bg["QVAPOR"]}, geo)
    with pytest.raises(ValueError):
        perturb.rebalance_with_mu(bg, bg, {**geo, "HYPSOMETRIC_OPT": 1})
    with pytest.raises(ValueError):
        perturb.rebalance_with_mu(bg, {**bg, "MU": bg["MU"] - 2.0 * geo["MUB"]}, geo)


def test_combinations_recentre_and_positivity():
    rng = np.random.default_rng(0)
    m, n = 9, 20
    states = {"T": 280 + rng.normal(size=(m, 3, 8, 9)), "QVAPOR": 0.01 * np.exp(0.3 * rng.normal(size=(m, 3, 8, 9))),
              "MU": 50 * rng.normal(size=(m, 8, 9))}
    dev, z = perturb.combination_perturbations(states, n, seed=4)
    dev2, z2 = perturb.combination_perturbations(states, n, seed=4)
    np.testing.assert_array_equal(z, z2)
    for name, d in dev.items():
        np.testing.assert_allclose(d.mean(0), 0.0, atol=1e-12)
    control = {"T": np.full((3, 8, 9), 285.0), "QVAPOR": np.full((3, 8, 9), 0.008), "MU": np.zeros((8, 9))}
    mem = perturb.recentre(control, dev, 0.4)
    assert (mem["QVAPOR"] > 0).all()
    np.testing.assert_allclose(mem["QVAPOR"].mean(0), control["QVAPOR"], rtol=1e-12)
    np.testing.assert_allclose(mem["T"].mean(0), control["T"], rtol=1e-12)
    taper = {"T": np.pad(np.ones((4, 5)), 2)}
    tap = perturb.recentre(control, {"T": dev["T"]}, 0.4, taper=taper)
    assert np.all(tap["T"][..., :2, :] == 285.0)


def test_balance_diagnostics_separate_rotational_from_divergent():
    ny, nx, dx = 40, 44, 3000.0
    y, x = np.mgrid[0:ny + 1, 0:nx + 1] * dx
    psi = np.sin(2 * np.pi * x / (nx * dx)) * np.cos(2 * np.pi * y / (ny * dx)) * 1e6
    chi = psi.copy()
    # rotational: u = -dpsi/dy, v = dpsi/dx (psi on corners); divergent: u = dchi/dx, v = dchi/dy (chi on mass points)
    u_rot = -(psi[1:, :] - psi[:-1, :]) / dx
    v_rot = (psi[:, 1:] - psi[:, :-1]) / dx
    cm = chi[:ny, :nx]
    u_div = np.zeros((ny, nx + 1)); u_div[:, 1:-1] = (cm[:, 1:] - cm[:, :-1]) / dx
    v_div = np.zeros((ny + 1, nx)); v_div[1:-1, :] = (cm[1:, :] - cm[:-1, :]) / dx
    geo = {"DX": dx, "DY": dx}
    rot = perturb.balance_diagnostics({"U": u_rot[None, None], "V": v_rot[None, None]}, geo)["div_over_vort_all"]
    div = perturb.balance_diagnostics({"U": u_div[None, None], "V": v_div[None, None]}, geo)["div_over_vort_all"]
    assert rot < 1e-6 and div > 1e3
    # per height band (critic cr-w41-da-ens 12:26Z: the corner vorticity was indexed with the mass-point mask ->
    # IndexError): 2 members x 3 levels at 500 m / 1.5 km / 3 km AGL -> bands 0-1, 1-2, 2-5 km
    nz = 3
    zw = np.array([0.0, 1000.0, 2000.0, 4000.0]) * 9.81
    frames = {"PH": np.zeros((2, nz + 1, ny, nx)), "PHB": np.broadcast_to(zw[None, :, None, None], (2, nz + 1, ny, nx))}
    geo_f = {**geo, "HGT": np.zeros((ny, nx))}
    lev = lambda f: np.broadcast_to(f[None, None], (2, nz) + f.shape)
    rot_b = perturb.balance_diagnostics({"U": lev(u_rot), "V": lev(v_rot)}, geo_f, frames)["div_over_vort_by_band"]
    div_b = perturb.balance_diagnostics({"U": lev(u_div), "V": lev(v_div)}, geo_f, frames)["div_over_vort_by_band"]
    assert sorted(rot_b) == sorted(div_b) == ["0-1km", "1-2km", "2-5km"]
    assert max(rot_b.values()) < 1e-6 and min(div_b.values()) > 1e3
    # band-resolved: rotational below 2 km, divergent in the 2-5 km level -> the bands must not mix
    mixed = lambda r, d: np.stack([r, r, d])[None].repeat(2, 0)
    mix = perturb.balance_diagnostics({"U": mixed(u_rot, u_div), "V": mixed(v_rot, v_div)}, geo_f,
                                      frames)["div_over_vort_by_band"]
    assert mix["0-1km"] < 1e-6 and mix["1-2km"] < 1e-6 and mix["2-5km"] > 1e3
