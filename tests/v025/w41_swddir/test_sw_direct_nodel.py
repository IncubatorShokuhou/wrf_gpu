"""w41-swddir: RRTMG SW direct output = WRF ``ztdbt_nodel`` on every SW producer (unit level, CPU, seconds).

WRF spcvmc_sw (module_ra_rrtmg_sw.F :8577-8600, :8737) accumulates the DIRECT output from the UNSCALED optical depths:
``zdbt_nodel = (1 - cldf) * exp_tbl(ztauc / mu0) + cldf * exp_tbl((ztauc + ptaormc) / mu0)`` with ``ptaormc`` the
McICA cloud depth before delta scaling (cldprmc_sw taormc, :2433); the adding method keeps the delta-scaled zdbt.  This
module rebuilds that product with an independent NumPy implementation of WRF's exp_tbl lookup and checks the default
Pallas band kernels (``_kernel`` with constant radii, ``_kernel_mp_re`` with Thompson radii; CPU interpret) and the
XLA tile solver against it, with a strongly forward-scattering cloud (delta factor ~0.25) so the scaled beam differs
by a lot.  The genuine mutant (direct carry multiplied by the scaled ``b``, i.e. the pre-fix product) must fail.
The column oracle vs pristine RRTMG_SWRAD lives in test_swddir_oracle.py.
"""
from __future__ import annotations

import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import rad_sw_band_sums as B
from gpuwrf.physics import rrtmg_sw as R

NCOL, NLAY, NG, NB, BAND = 8, 6, 16, 14, 5
LIQ_GRID, ICE_GRID = (2.5, 1.0), (5.0, 3.0)  # cldprmc radius grids (start, step)


def wrf_exp_tbl(od):
    """WRF rrtmg_sw_init exp_tbl + spcvmc_sw lookup / low-tau expansion, REAL (float32), od >= 0."""

    od = np.asarray(od, np.float32)
    bpade, tblint = np.float32(1.0 / 0.278), np.float32(10000.0)
    i = np.arange(10001, dtype=np.float32) / tblint
    with np.errstate(divide="ignore"):
        table = np.exp(-(bpade * i / (np.float32(1.0) - i))).astype(np.float32)
    table[0], table[-1] = 1.0, 1.0e-20
    tau = np.minimum(od, np.float32(500.0))
    idx = (tblint * (tau / (bpade + tau)) + np.float32(0.5)).astype(np.int32)
    expansion = np.float32(1.0) - tau + np.float32(0.5) * tau * tau
    return np.where(tau <= np.float32(0.06), expansion, table[np.clip(idx, 0, 10000)]).astype(np.float32)


def _inputs(seed=7):
    rng = np.random.default_rng(seed)
    f = lambda lo, hi, shape: rng.uniform(lo, hi, shape).astype(np.float32)  # noqa: E731
    tau_gas, tau_ray = f(1e-4, 0.3, (NCOL, NLAY, NG)), f(1e-4, 0.05, (NCOL, NLAY, NG))
    cloud = (rng.uniform(size=(NCOL, NLAY, NB, NG)) < 0.25).astype(np.float32)
    cloud[0] = 0.0  # one clear column: there nodel == the scaled beam
    gmask = np.ones((NB, NG), np.float32)
    gmask[BAND, 12:] = 0.0  # band with fewer g-points: identity lanes
    sflux = f(0.5, 3.0, (NCOL, NB, NG))
    coszen = f(0.15, 0.95, (NCOL,))
    albedo, scale = f(0.05, 0.3, (NCOL,)), f(0.97, 1.03, (NCOL,))
    paths = f(0.0, 6.0, (NCOL, NLAY, 3))  # in-cloud liquid / ice / snow paths [g m-2]: thin clouds (tau ~ 0.1-2)
    return tau_gas, tau_ray, cloud, paths, gmask, sflux, coszen, albedo, scale


def _const_coeffs(rng_seed=3):
    """(12, NB) rows ext(l,i,s), delta tau factor(l,i,s), omega(l,i,s), asymmetry(l,i,s) (caller-formed)."""

    rng = np.random.default_rng(rng_seed)
    ext = rng.uniform(0.05, 0.15, (3, NB))
    ssa, asy = rng.uniform(0.99, 0.9999, (3, NB)), rng.uniform(0.84, 0.88, (3, NB))
    fwd = asy * asy
    den = 1.0 - fwd * ssa
    return np.concatenate([ext, den, ssa * (1 - fwd) / den, (asy - fwd) / (1 - fwd)]).astype(np.float32), ext


def _mp_tables(rng_seed=5):
    """(7, NB, 64) raw radius tables extliq1, ssaliq1, asyliq1, extice3, ssaice3, asyice3, fdlice3 (synthetic)."""

    rng = np.random.default_rng(rng_seed)
    t = np.zeros((7, NB, 64), np.float32)
    t[0], t[3] = rng.uniform(0.05, 0.2, (NB, 64)), rng.uniform(0.02, 0.1, (NB, 64))
    t[1], t[4] = rng.uniform(0.99, 0.9999, (NB, 64)), rng.uniform(0.95, 0.999, (NB, 64))
    t[2], t[5] = rng.uniform(0.84, 0.88, (NB, 64)), rng.uniform(0.75, 0.85, (NB, 64))
    t[6] = rng.uniform(0.0, 0.05, (NB, 64))
    return t


def _reference(tau_gas, tau_ray, cloud, cloud_tau, gmask, sflux, coszen, scale, band=BAND):
    """Direct flux per bottom-up interface: sum_g top_g * prod_{layers above} zdbt_nodel (WRF :8737)."""

    m = gmask[band][None, None, :]
    c = cloud[:, :, band, :]
    tauc = np.maximum(tau_gas + tau_ray, np.float32(1e-10)) * m  # zero-g clear depth (identity delta scaling)
    mu = coszen[:, None, None]
    zdbt = (1 - c) * wrf_exp_tbl(tauc / mu) + c * wrf_exp_tbl((tauc + cloud_tau * c * m) / mu)
    zdbt = np.where(m > 0, zdbt * m, 1.0)
    top = (coszen * scale)[:, None] * sflux[:, band, :]
    out = np.zeros((NCOL, NLAY + 1))
    for lev in range(NLAY + 1):
        out[:, lev] = np.sum(top * np.prod(zdbt[:, lev:, :], axis=1, dtype=np.float64), axis=1)
    return out


def _kernel_module(mutant=False):
    if not mutant:
        return B
    source = Path(B.__file__).read_text()
    assert source.count("dn * bn") == 2  # both kernels (constant radii and Thompson radii)
    mod = types.ModuleType("rad_sw_band_sums_mutant")
    exec(compile(source.replace("dn * bn", "dn * b"), "rad_sw_band_sums_mutant.py", "exec"), mod.__dict__)
    return mod


def _ratio(got, want):
    """Worst |got - want| / (1e-4 + 1e-4 |want|): REAL-rounding agreement (fixture fluxes are O(10))."""
    return float(np.max(np.abs(np.asarray(got, np.float64) - want) / (1e-4 + 1e-4 * np.abs(want))))


@pytest.fixture(scope="module")
def case():
    return _inputs()


@pytest.mark.parametrize("mp_re", [False, True], ids=["const_radii", "thompson_radii"])
def test_band_kernel_direct_is_wrf_nodel_and_mutant_fails(case, mp_re):
    tau_gas, tau_ray, cloud, paths, gmask, sflux, coszen, albedo, scale = case
    if mp_re:
        tables = _mp_tables()
        idx = np.random.default_rng(11).integers(0, 40, (NCOL, NLAY, 3))
        radii = np.stack([LIQ_GRID[0] + LIQ_GRID[1] * idx[..., 0], ICE_GRID[0] + ICE_GRID[1] * idx[..., 1],
                          ICE_GRID[0] + ICE_GRID[1] * idx[..., 2]], axis=-1).astype(np.float32)  # on table nodes
        incloud, coeffs = np.concatenate([paths, radii], axis=-1), tables
        ext = np.stack([tables[0, BAND][idx[..., 0]], tables[3, BAND][idx[..., 1]], tables[3, BAND][idx[..., 2]]], -1)
    else:
        coeffs, ext_band = _const_coeffs()
        incloud, ext = paths, np.broadcast_to(ext_band[:, BAND], paths.shape)
    cloud_tau = np.sum(paths * ext, axis=-1, dtype=np.float32)[..., None]  # unscaled taormc (liquid + ice + snow)
    want = _reference(tau_gas, tau_ray, cloud, cloud_tau, gmask, sflux, coszen, scale)
    args = tuple(jnp.asarray(x) for x in (tau_gas, tau_ray, cloud, incloud, coeffs, gmask, sflux, coszen, albedo, scale))
    results = {}
    for mutant in (False, True):
        mod = _kernel_module(mutant)
        out = mod.sw_band_flux_sums(*args, jnp.int32(BAND), interpret=True)
        results[mutant] = np.asarray(out[2], np.float64)
        assert np.all(np.isfinite(results[mutant]))
    assert _ratio(results[False], want) <= 1.0, _ratio(results[False], want)  # REAL-rounding agreement
    # clear column: nodel == the scaled beam, so the mutant agrees there and misses on the cloudy columns
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=0, atol=0)
    assert _ratio(results[True], want) > 100.0, _ratio(results[True], want)
    # the delta-scaled cloud depth is smaller: the pre-fix beam is too HIGH at the surface
    assert np.max(results[True][1:, 0] - results[False][1:, 0]) > 0.05 * np.max(want[:, 0])


def test_xla_tile_solver_direct_is_wrf_nodel(case, monkeypatch):
    """``rrtmg_sw._sw_band_tile_fluxes`` (chunked and upfront XLA paths) with the unscaled direct depths."""

    monkeypatch.setattr(R, "_FUSED_QUADRATURE", False)  # XLA quadrature (the Pallas one is covered elsewhere)
    tau_gas, tau_ray, cloud, paths, gmask, sflux, coszen, albedo, scale = case
    coeffs, ext_band = _const_coeffs()
    cloud_tau = np.sum(paths * ext_band[:, BAND], axis=-1, dtype=np.float32)[..., None]
    want = _reference(tau_gas, tau_ray, cloud, cloud_tau, gmask, sflux, coszen, scale)
    c = cloud[:, :, BAND:BAND + 1, :]
    m = gmask[BAND:BAND + 1]
    tauc = np.maximum((tau_gas + tau_ray)[:, :, None, :], np.float32(1e-10)) * m
    tau_scaled_cloud = tauc + (cloud_tau[..., None] * coeffs[3, BAND] * c) * m  # any adding-method optics
    omega = np.full_like(tauc, 0.9)
    asym = np.full_like(tauc, 0.5)
    common = (jnp.asarray(tauc), jnp.asarray(omega), jnp.asarray(asym), jnp.asarray(tau_scaled_cloud),
              jnp.asarray(omega), jnp.asarray(asym), jnp.asarray(c), jnp.asarray(m), jnp.asarray(sflux[:, BAND:BAND + 1]),
              jnp.asarray(coszen), jnp.asarray(albedo), jnp.asarray(scale), jnp.float32)
    nodel = R._sw_band_tile_fluxes(*common, direct_taus=(jnp.asarray(tauc), jnp.asarray(tauc + (cloud_tau[..., None] * c) * m)))
    assert _ratio(nodel[2], want) <= 1.0, _ratio(nodel[2], want)
    scaled = R._sw_band_tile_fluxes(*common)  # direct_taus=None: the adding method's (delta-scaled) beam
    assert _ratio(scaled[2], want) > 100.0
    for a, b in zip(nodel[:2], scaled[:2]):  # diffuse/total fluxes do not depend on the direct OUTPUT product
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_allband_launch_direct_is_wrf_nodel(case):
    """BAND_SUMS=2: every band in ONE launch (same per-band solve), summed in band order == sum of WRF nodel beams."""

    tau_gas, tau_ray, cloud, paths, gmask, sflux, coszen, albedo, scale = case
    rng = np.random.default_rng(13)
    gas_all = np.repeat(tau_gas[:, :, None, :], NB, axis=2) * rng.uniform(0.5, 1.5, (1, 1, NB, 1)).astype(np.float32)
    ray_all = np.repeat(tau_ray[:, :, None, :], NB, axis=2)
    coeffs, ext_band = _const_coeffs()
    want = sum(_reference(gas_all[:, :, b], ray_all[:, :, b], cloud,
                          np.sum(paths * ext_band[:, b], axis=-1, dtype=np.float32)[..., None], gmask, sflux, coszen, scale,
                          band=b) for b in range(NB))
    args = tuple(jnp.asarray(x) for x in (gas_all, ray_all, cloud, paths, coeffs, gmask, sflux, coszen, albedo, scale))
    got = np.asarray(B.sw_allband_flux_sums(*args, interpret=True)[2], np.float64)
    assert _ratio(got, want) <= 1.0, _ratio(got, want)
    scaled = np.asarray(_kernel_module(True).sw_allband_flux_sums(*args, interpret=True)[2], np.float64)
    assert _ratio(scaled, want) > 100.0
