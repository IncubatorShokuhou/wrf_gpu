"""w41-aerosol A3: the aer_opt=1 Tegen chain is BYTE-identical to pristine WRF V4.7.1 (AE01 unit oracle).

Each gate has a genuine mutant (E202): a plausible transcription slip that the gate must catch.
"""
from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import wrf_tegen_aerosol as T
from tegen_fixture import UNITS, bound_ratio, load


@pytest.fixture(scope="module")
def u():
    return load(UNITS)


def _grid(u):
    return np.meshgrid(u["in_lat"], u["in_lon"], indexing="ij")


def _aerodm_sha(lat, lon):
    return hashlib.sha256(T.wrf_aerosol_in(lat, lon).tobytes()).hexdigest()


def test_swaerpr_band_tables_bitwise(u):
    for name in ("RSRTAUA", "RSRPIZA", "RSRASYA"):
        assert np.array_equal(getattr(T, name).view(np.uint32), u[name.lower()].view(np.uint32)), name


def test_asset_pressure_levels_and_aerosol_in_bitwise(u):
    assert np.array_equal(T._load_wrf_tegen_aerosol_asset()[0], u["pina"])
    lat, lon = _grid(u)
    assert np.isin(np.float32([-90.0, 89.5, 90.0]), u["in_lat"]).all()
    assert np.isin(np.float32([-180.0, 179.99, 180.0]), u["in_lon"]).all()
    assert _aerodm_sha(lat, lon) == str(u["aerodm_sha256"])


def test_aerosol_in_mutant_without_periodic_longitude_is_caught(u, monkeypatch):
    real = T._interp_vec

    def clamped(locvec, locwant):  # non-periodic interp_vec branch (WRF passes periodic=.true.)
        loc1, loc2, w1, w2 = real(locvec, locwant)
        edge = loc2 == 0
        return np.where(edge, loc1, loc1), np.where(edge, loc1, loc2), np.where(edge, 1, w1).astype(np.float32), \
            np.where(edge, 0, w2).astype(np.float32)

    monkeypatch.setattr(T, "_interp_vec", clamped)
    lat, lon = _grid(u)
    assert _aerodm_sha(lat, lon) != str(u["aerodm_sha256"])


def _time_rows(julians, factors=T.wrf_aer_time_factors):
    import jax

    nm, np1, f1, f2 = (np.asarray(v) for v in jax.vmap(factors)(jnp.asarray(julians, jnp.float32)))  # eager, no jit
    rows = np.zeros((len(julians), 12), np.float32)
    rows[np.arange(len(julians)), nm] += f1
    rows[np.arange(len(julians)), np1] += f2
    return rows


def test_aer_time_int_factors_bitwise_incl_dec_jan_wrap(u):
    jul = u["in_julian"]
    assert jul.min() == 0.0 and jul.max() > 364.9 and 135.0 in jul.tolist()  # both wrap branches + date_oz hit
    assert np.array_equal(_time_rows(jul).view(np.uint32), u["time_unit"].view(np.uint32))


def test_aer_time_int_mutant_without_one_day_offset_is_caught(u):
    def no_offset(julian0):  # WRF adds 1 to the 0-based JULIAN before the month search
        return T.wrf_aer_time_factors(jnp.float32(julian0) - jnp.float32(1.0))

    assert not np.array_equal(_time_rows(u["in_julian"], no_offset), u["time_unit"])


def test_aer_p_int_bitwise_incl_extrapolation_branches(u):
    pmid = u["in_p"] * np.float32(0.01)
    assert (pmid < u["pina"][0]).any() and (pmid > u["pina"][-1]).any()
    assert np.isin(u["pina"], pmid).all()  # every table level hit exactly (kupper initial/stale branch)
    got = np.asarray(T.wrf_aer_p_int(u["in_aerodt"], u["in_p"], u["in_pf"], u["pina"]))
    assert np.array_equal(got.view(np.uint32), u["aerod"].view(np.uint32))


def test_aer_p_int_mutant_layer_mass_from_wrong_faces_is_caught(u):
    pf = u["in_pf"]
    shifted = np.concatenate([pf[:, 1:], pf[:, -1:]], axis=1)  # dpm = pf(k+1) - pf(k+2): off-by-one face
    got = np.asarray(T.wrf_aer_p_int(u["in_aerodt"], u["in_p"], shifted, u["pina"]))
    assert bound_ratio(u["aerod"], got, (1e-9, 1e-3)).max() > 1.0


def test_ecaer_optics_zero_aerosol_is_wrf_iaer6_zero_state():
    """WRF iaer=6 with ecaer = 0 (the extra top layer): ztaua = zomga = zasya = 0 (no division)."""

    tau, omg, asy = (np.asarray(v) for v in T.ecaer_band_optics(np.zeros((3, 6), np.float32)))
    assert tau.shape == (3, 14) and not tau.any() and not omg.any() and not asy.any()
    tau, omg, asy = (np.asarray(v) for v in T.ecaer_band_optics(np.full((1, 6), 0.01, np.float32)))
    assert (tau > 0).all() and ((omg > 0) & (omg < 1)).all() and ((asy > 0) & (asy < 1)).all()
