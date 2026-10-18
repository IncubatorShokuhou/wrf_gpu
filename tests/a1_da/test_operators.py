"""Tests for gpuwrf.assimilation.letkf.operators (lane a1-obs). CPU only, NumPy fp64."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import obs as ob
from gpuwrf.assimilation.letkf import operators as ops

RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260502_18z_a1/run/run")
WRFOUT_D01 = RUN / "wrfout_d01_2026-05-03_12:00:00"
# ALISIOS read conditions (2026-10-17 01:14Z): obs are read ONLY from the guard_call-admitted copies in guarded_v1
IGRA = Path("<DATA_ROOT>/wrf_gpu2/scratch/a1-da/obs/guarded_v1/igra2_SPM00060018_admitted.txt")


# ----------------------------------------------------------------------- fixtures
def affine_grid(ny=9, nx=11, lat0=27.0, lon0=-18.0, dlat=0.08, dlon=0.09, rot_deg=12.0):
    """lat/lon exactly affine in (i, j): the isoparametric inverse is exact."""
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.deg2rad(rot_deg)
    lat = lat0 + dlat * (np.cos(r) * j + np.sin(r) * i * 0.3)
    lon = lon0 + dlon * (np.cos(r) * i - np.sin(r) * j * 0.3)
    return lat, lon


def affine_index(lat, lon, lat_pt, lon_pt):
    """Exact (i, j) of points on an affine grid by solving the 2x2 affine map."""
    a = np.array([[lon[0, 1] - lon[0, 0], lon[1, 0] - lon[0, 0]],
                  [lat[0, 1] - lat[0, 0], lat[1, 0] - lat[0, 0]]])
    rhs = np.stack((lon_pt - lon[0, 0], lat_pt - lat[0, 0]))
    sol = np.linalg.solve(a, rhs)
    return sol[0], sol[1]


def sample_points(lat, lon, n, rng, margin=0.6):
    ny, nx = lat.shape
    i = rng.uniform(margin, nx - 1 - margin, n)
    j = rng.uniform(margin, ny - 1 - margin, n)
    # affine grid: evaluate lat/lon at fractional (i, j) exactly
    lat_pt = lat[0, 0] + (lat[0, 1] - lat[0, 0]) * i + (lat[1, 0] - lat[0, 0]) * j
    lon_pt = lon[0, 0] + (lon[0, 1] - lon[0, 0]) * i + (lon[1, 0] - lon[0, 0]) * j
    return i, j, lat_pt, lon_pt


def analytic_case(nmem=3, ny=9, nx=11, nz=10, alpha_deg=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = affine_grid(ny, nx)
    k = np.arange(nz, dtype=float)[None, :, None, None]
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    mem = np.arange(nmem, dtype=float)[:, None, None, None]
    # ln p varies with k, i, j and member (non-uniform columns)
    lnp = np.log(1.0e5) - 0.12 * k - 0.004 * i[None, None] + 0.003 * j[None, None] - 0.001 * mem
    p_full = np.exp(lnp)
    pb = 0.9 * p_full
    coef = dict(a=0.01, b=2e-4, c=-3e-4, d=0.002)
    qv = coef["a"] + coef["b"] * i + coef["c"] * j + coef["d"] * lnp + 1e-4 * mem
    temp = 250.0 + 0.3 * i - 0.2 * j + 6.0 * lnp + 0.5 * mem
    theta_pert = temp * np.exp(ops.RCP * (np.log(ops.P0) - lnp)) - 300.0
    # staggered winds on horizontally uniform levels (exact in i, j, ln p)
    lnp_lev = np.log(1.0e5) - 0.12 * k
    p_unif = np.broadcast_to(np.exp(lnp_lev), (nmem, nz, ny, nx)).copy()
    ju, iu = np.mgrid[0:ny, 0:nx + 1].astype(float)
    jv, iv = np.mgrid[0:ny + 1, 0:nx].astype(float)
    uu = 3.0 + 0.5 * (iu - 0.5) - 0.25 * ju + 4.0 * lnp_lev + mem
    vv = -2.0 + 0.1 * iv + 0.7 * (jv - 0.5) - 3.0 * lnp_lev - mem
    a = np.deg2rad(alpha_deg)
    geo = {"XLAT": lat, "XLONG": lon, "HGT": 100.0 + 10.0 * i + 5.0 * j,
           "COSALPHA": np.full((ny, nx), np.cos(a)), "SINALPHA": np.full((ny, nx), np.sin(a)),
           "LANDMASK": np.ones((ny, nx)), "DX": 9000.0}
    frames = {"P": p_full - pb, "PB": pb, "T": theta_pert, "QVAPOR": qv}
    wind_frames = {"P": 0.25 * p_unif, "PB": 0.75 * p_unif, "U": uu, "V": vv}
    return rng, lat, lon, geo, frames, wind_frames, lnp_lev


def sonde_batch(var, lat_pt, lon_pt, p_pa):
    n = len(lat_pt)
    return ob.ObsBatch.from_columns(var=np.full(n, var, np.int16), value=np.zeros(n), error=1.0,
                                    lat=lat_pt, lon=lon_pt, p_pa=p_pa, station="s")


# ----------------------------------------------------------------- grid locator
def test_locator_exact_on_affine_grid_and_outside():
    rng = np.random.default_rng(1)
    lat, lon = affine_grid()
    loc = ops.GridLocator(lat, lon)
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 200, rng, margin=0.0)
    fi, fj, inside = loc.locate(lat_pt, lon_pt)
    assert inside.all()
    assert np.max(np.abs(fi - i)) < 1e-9 and np.max(np.abs(fj - j)) < 1e-9
    # outside the domain (beyond the last row) -> not inside, NaN indices
    out_lat = lat[-1, 5] + 2 * (lat[-1, 5] - lat[-2, 5])
    fi, fj, inside = loc.locate([out_lat], [lon[-1, 5]])
    assert not inside[0] and np.isnan(fi[0])


def test_locator_matches_station_sampler_weights_on_curved_grid():
    pd = pytest.importorskip("pandas")
    from gpuwrf.verify.sampling import StationSampler

    ny, nx = 12, 14
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    lat = 27.0 + 0.08 * j + 0.0007 * i ** 2 - 0.0004 * i * j
    lon = -18.0 + 0.09 * i + 0.0006 * j ** 2 + 0.0003 * i * j
    rng = np.random.default_rng(2)
    pts = pd.DataFrame({"station_uid": [f"s{n}" for n in range(30)],
                        "lat": rng.uniform(lat[2:-2, 2:-2].min(), lat[2:-2, 2:-2].max(), 30),
                        "lon": rng.uniform(lon[2:-2, 2:-2].min(), lon[2:-2, 2:-2].max(), 30)})
    loc = ops.GridLocator(lat, lon)
    fi, fj, inside = loc.locate(pts.lat.to_numpy(), pts.lon.to_numpy())
    keep = inside.copy()
    sampler = StationSampler(lat, lon, pts[keep], land_aware=False)
    field = np.sin(i / 3.0) + np.cos(j / 4.0) * 2.0
    jj, ii, w, _ = ops._stencil(fi[keep], fj[keep], ny, nx)
    ours = ops._apply(field, jj, ii, w)
    assert np.max(np.abs(ours - sampler.sample(field))) < 1e-9


# --------------------------------------------------------------- sonde operator
def test_sonde_qv_and_t_exact_for_linear_fields_in_i_j_lnp():
    rng, lat, lon, geo, frames, _, _ = analytic_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 40, rng)
    # pressures inside every member's column at that point
    lnp_obs = np.log(1.0e5) - rng.uniform(0.15, 0.12 * 8, 40) - 0.004 * i + 0.003 * j
    p_obs = np.exp(lnp_obs)
    for var, coeffs in ((ob.QV, None), (ob.T, None)):
        batch = sonde_batch(var, lat_pt, lon_pt, p_obs)
        hx, qc = ops.model_equivalents(batch, frames, geo, return_qc=True)
        mem = np.arange(3)[:, None]
        if var == ob.QV:
            expect = 0.01 + 2e-4 * i - 3e-4 * j + 0.002 * lnp_obs + 1e-4 * mem
            # member offset in ln p shifts the column, not the field definition at given ln p
            expect = 0.01 + 2e-4 * i - 3e-4 * j + 0.002 * lnp_obs + 1e-4 * mem
        else:
            expect = 250.0 + 0.3 * i - 0.2 * j + 6.0 * lnp_obs + 0.5 * mem
        assert (qc == 0).all()
        assert np.max(np.abs(hx - expect)) < 1e-9, (var, np.max(np.abs(hx - expect)))


def test_sonde_linear_in_p_mutant_is_detected(monkeypatch):
    """Deletion sensitivity (E39): interpolating linearly in p instead of ln p must fail the gate."""
    rng, lat, lon, geo, frames, _, _ = analytic_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 20, rng)
    lnp_obs = np.log(1.0e5) - rng.uniform(0.15, 0.9, 20) - 0.004 * i + 0.003 * j
    batch = sonde_batch(ob.QV, lat_pt, lon_pt, np.exp(lnp_obs))
    expect = 0.01 + 2e-4 * i - 3e-4 * j + 0.002 * lnp_obs + 1e-4 * np.arange(3)[:, None]
    orig = ops.vertical_interp_lnp

    def linear_in_p(col_f, col_lnp, lnp_o):
        return orig(col_f, np.exp(col_lnp) / 1e5, np.exp(lnp_o) / 1e5)

    monkeypatch.setattr(ops, "vertical_interp_lnp", linear_in_p)
    hx = ops.model_equivalents(batch, frames, geo)
    assert np.nanmax(np.abs(hx - expect)) > 1e-7


def test_sonde_outside_column_is_nan_and_flagged():
    rng, lat, lon, geo, frames, _, _ = analytic_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 3, rng)
    p_obs = np.array([1.2e5, 1.0, np.exp(np.log(1e5) - 0.5)])  # below surface, above top, inside
    batch = sonde_batch(ob.T, lat_pt, lon_pt, p_obs)
    hx, qc = ops.model_equivalents(batch, frames, geo, return_qc=True)
    assert np.isnan(hx[:, :2]).all() and np.isfinite(hx[:, 2]).all()
    assert (qc[:2] & ob.QC_OUTSIDE).all() and qc[2] == 0
    # out-of-domain location
    far = sonde_batch(ob.T, np.array([lat.max() + 1.0]), np.array([lon.mean()]), np.array([8e4]))
    hx, qc = ops.model_equivalents(far, frames, geo, return_qc=True)
    assert np.isnan(hx).all() and qc[0] & ob.QC_OUTSIDE


@pytest.mark.parametrize("alpha_deg", [0.0, 27.0, -140.0])
def test_sonde_staggered_winds_exact_and_rotated_to_earth(alpha_deg):
    rng, lat, lon, geo, _, wind, _ = analytic_case(alpha_deg=alpha_deg)
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 25, rng)
    lnp_obs = np.log(1.0e5) - rng.uniform(0.2, 0.9, 25)
    mem = np.arange(3)[:, None]
    ug = 3.0 + 0.5 * i - 0.25 * j + 4.0 * lnp_obs + mem
    vg = -2.0 + 0.1 * i + 0.7 * j - 3.0 * lnp_obs - mem
    c, s = np.cos(np.deg2rad(alpha_deg)), np.sin(np.deg2rad(alpha_deg))
    ue, ve = ug * c - vg * s, vg * c + ug * s
    for var, expect in ((ob.U, ue), (ob.V, ve)):
        hx = ops.model_equivalents(sonde_batch(var, lat_pt, lon_pt, np.exp(lnp_obs)), wind, geo)
        assert np.max(np.abs(hx - expect)) < 1e-9
    # inverse rotation recovers grid winds
    gu, gv = ops.earth_to_grid(ue, ve, c, s)
    assert np.max(np.abs(gu - ug)) < 1e-12 and np.max(np.abs(gv - vg)) < 1e-12


def test_sonde_wind_rotation_mutant_is_detected(monkeypatch):
    rng, lat, lon, geo, _, wind, _ = analytic_case(alpha_deg=27.0)
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 10, rng)
    lnp_obs = np.log(1.0e5) - rng.uniform(0.2, 0.9, 10)
    good = ops.model_equivalents(sonde_batch(ob.U, lat_pt, lon_pt, np.exp(lnp_obs)), wind, geo)
    monkeypatch.setattr(ops, "grid_to_earth", lambda u, v, c, s: (u, v))
    bad = ops.model_equivalents(sonde_batch(ob.U, lat_pt, lon_pt, np.exp(lnp_obs)), wind, geo)
    assert np.max(np.abs(good - bad)) > 0.1


def test_stagger_offset_mutant_is_detected():
    """Sampling U as if it were on the mass grid (no +0.5 offset) must break exactness."""
    rng, lat, lon, geo, _, wind, _ = analytic_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 10, rng)
    lnp_obs = np.log(1.0e5) - rng.uniform(0.2, 0.9, 10)
    hx = ops.model_equivalents(sonde_batch(ob.U, lat_pt, lon_pt, np.exp(lnp_obs)), wind, geo)
    expect_unstaggered_bug = 3.0 + 0.5 * (i - 0.5) - 0.25 * j + 4.0 * lnp_obs + np.arange(3)[:, None]
    assert np.min(np.abs(hx - expect_unstaggered_bug)) > 0.2


def test_sonde_layer_superob_uses_model_layers():
    rng, lat, lon, geo, frames, _, _ = analytic_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 1, rng)
    # column at the point: ln p_k = ln 1e5 - 0.12 k - 0.004 i + 0.003 j - 0.001 m (member mean m = 1)
    base = np.log(1.0e5) - 0.004 * i[0] + 0.003 * j[0] - 0.001
    lnp_obs = base - np.array([0.10, 0.13, 0.35, 0.37, 0.39])  # layers 1, 1, 3, 3, 3 (edges at 0.06 + 0.12 k)
    b = sonde_batch(ob.T, np.repeat(lat_pt, 5), np.repeat(lon_pt, 5), np.exp(lnp_obs))
    b = b.replace(value=np.array([1.0, 3.0, 10.0, 20.0, 30.0]), launch=np.array(["L"] * 5, dtype=object))
    out = ops.sonde_layer_superob(b, frames, geo)
    new = out.subset(out.qc == 0)
    assert sorted(new.value.tolist()) == [2.0, 20.0] and ((out.qc & ob.QC_THINNED) != 0).sum() == 5


# ------------------------------------------------------------- surface operator
def surface_case(alpha_deg=20.0):
    rng, lat, lon, geo, _, _, _ = analytic_case(alpha_deg=alpha_deg)
    ny, nx = lat.shape
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    nmem = 2
    mem = np.arange(nmem, dtype=float)[:, None, None]
    frames = {"T2": 290.0 + 0.1 * i - 0.05 * j + mem, "Q2": 0.008 + 1e-4 * i + 0 * mem,
              "U10": 2.0 + 0.2 * i + mem, "V10": -1.0 + 0.3 * j - mem,
              "PSFC": 95000.0 + 10.0 * i - 20.0 * j + mem}
    return rng, lat, lon, geo, frames


def surface_batch(var, lat_pt, lon_pt, z):
    n = len(lat_pt)
    return ob.ObsBatch.from_columns(var=np.full(n, var, np.int16), value=np.zeros(n), error=1.0, lat=lat_pt,
                                    lon=lon_pt, height_m=z, station="st")


def test_surface_t2_lapse_q2_and_dz_rejection():
    rng, lat, lon, geo, frames = surface_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 12, rng)
    zmod = 100.0 + 10.0 * i + 5.0 * j
    z_stn = zmod + np.linspace(-250.0, 350.0, 12)
    z_stn[0] = np.nan  # unknown station height -> rejected
    batch = surface_batch(ob.T2, lat_pt, lon_pt, z_stn)
    res = ops.apply_operators(batch, frames, geo)
    hx, qc = res.hx, res.reason
    expect = 290.0 + 0.1 * i - 0.05 * j + np.arange(2)[:, None] - 0.0065 * (z_stn - zmod)
    ok = np.abs(z_stn - zmod) <= 150.0          # DESIGN_v2 default, every domain
    assert ok.sum() == 6
    assert np.max(np.abs(hx[:, ok] - expect[:, ok])) < 1e-9
    assert np.isnan(hx[:, ~ok]).all() and (qc[~ok] & ob.QC_DZ).all() and (qc[ok] == 0).all()
    assert np.allclose(res.dz_m[1:], (z_stn - zmod)[1:]) and np.isnan(res.dz_m[0])
    assert not res.valid[:, ~ok].any() and res.valid[:, ok].all()
    # explicit looser threshold accepts more (deletion-sensitive on the threshold)
    _, qc300 = ops.model_equivalents(batch, frames, geo, return_qc=True, dz_max_m=300.0)
    assert ((qc300 & ob.QC_DZ) != 0).sum() < ((qc & ob.QC_DZ) != 0).sum()
    hq = ops.model_equivalents(surface_batch(ob.Q2, lat_pt, lon_pt, zmod), frames, geo)
    assert np.max(np.abs(hq - (0.008 + 1e-4 * i))) < 1e-12


def test_operator_result_vertical_coordinate():
    rng, lat, lon, geo, frames = surface_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 4, rng)
    res = ops.apply_operators(surface_batch(ob.T2, lat_pt, lon_pt, 100.0 + 10.0 * i + 5.0 * j), frames, geo)
    psfc_mean = 95000.0 + 10.0 * i - 20.0 * j + 0.5          # members 0, 1 -> mean offset 0.5
    assert np.allclose(res.lnp_vert, np.log(psfc_mean), atol=1e-12)
    _, lat2, lon2, geo2, sframes, _, _ = analytic_case()
    _, _, la, lo = sample_points(lat2, lon2, 3, rng)
    p_obs = np.array([9.0e4, 8.0e4, 7.0e4])
    r2 = ops.apply_operators(sonde_batch(ob.QV, la, lo, p_obs), sframes, geo2)
    assert np.allclose(r2.lnp_vert, np.log(p_obs)) and np.isnan(r2.dz_m).all()
    assert r2.corrections["dz_max_m"] == 150.0


def test_surface_u10_v10_rotated_and_psfc_hydrostatic():
    rng, lat, lon, geo, frames = surface_case(alpha_deg=33.0)
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 8, rng)
    zmod = 100.0 + 10.0 * i + 5.0 * j
    mem = np.arange(2)[:, None]
    ug, vg = 2.0 + 0.2 * i + mem, -1.0 + 0.3 * j - mem
    c, s = np.cos(np.deg2rad(33.0)), np.sin(np.deg2rad(33.0))
    hu = ops.model_equivalents(surface_batch(ob.U10, lat_pt, lon_pt, zmod), frames, geo)
    hv = ops.model_equivalents(surface_batch(ob.V10, lat_pt, lon_pt, zmod), frames, geo)
    assert np.max(np.abs(hu - (ug * c - vg * s))) < 1e-9 and np.max(np.abs(hv - (vg * c + ug * s))) < 1e-9
    ps = 95000.0 + 10.0 * i - 20.0 * j + mem
    hp0 = ops.model_equivalents(surface_batch(ob.PSFC, lat_pt, lon_pt, zmod), frames, geo)
    assert np.max(np.abs(hp0 - ps)) < 1e-7
    hp = ops.model_equivalents(surface_batch(ob.PSFC, lat_pt, lon_pt, zmod + 100.0), frames, geo)
    assert np.isfinite(hp).all()
    t2 = 290.0 + 0.1 * i - 0.05 * j + mem
    tv = (t2 - 0.5 * 0.0065 * 100.0) * (1 + 0.608 * (0.008 + 1e-4 * i))
    assert np.max(np.abs(hp - ps * np.exp(-9.81 * 100.0 / (287.0 * tv)))) < 1e-7
    assert (hp < hp0).all()  # ~ -11 hPa per 100 m


def test_surface_land_aware_uses_only_matching_class():
    rng, lat, lon, geo, frames = surface_case()
    ny, nx = lat.shape
    land = np.ones((ny, nx))
    land[:, 6:] = 0.0  # sea to the east of column 5.x
    geo = dict(geo, LANDMASK=land)
    # point between columns 5 and 6: land-aware must use only column-5 nodes
    lat_pt = lat[0, 0] + (lat[0, 1] - lat[0, 0]) * 5.4 + (lat[1, 0] - lat[0, 0]) * 3.0
    lon_pt = lon[0, 0] + (lon[0, 1] - lon[0, 0]) * 5.4 + (lon[1, 0] - lon[0, 0]) * 3.0
    z = np.array([100.0 + 50.0 + 15.0])
    batch = surface_batch(ob.T2, np.array([lat_pt]), np.array([lon_pt]), z)
    h_land = ops.model_equivalents(batch, frames, geo, lapse_rate=0.0)
    h_plain = ops.model_equivalents(batch, frames, geo, lapse_rate=0.0, land_aware=False)
    assert np.allclose(h_land[:, 0], 290.0 + 0.1 * 5 - 0.05 * 3 + np.arange(2), atol=1e-9)
    assert np.allclose(h_plain[:, 0], 290.0 + 0.1 * 5.4 - 0.05 * 3 + np.arange(2), atol=1e-9)
    # an all-sea cell for a land station -> rejected
    lat_sea = lat[0, 0] + (lat[0, 1] - lat[0, 0]) * 8.5 + (lat[1, 0] - lat[0, 0]) * 3.0
    lon_sea = lon[0, 0] + (lon[0, 1] - lon[0, 0]) * 8.5 + (lon[1, 0] - lon[0, 0]) * 3.0
    hx, qc = ops.model_equivalents(surface_batch(ob.T2, np.array([lat_sea]), np.array([lon_sea]), z), frames,
                                   geo, return_qc=True)
    assert np.isnan(hx).all() and qc[0] & ob.QC_OUTSIDE


# ------------------------------------------------------------------ cloud operator
def test_cloud_column_max_bilinear():
    rng, lat, lon, geo, _, _, _ = analytic_case()
    ny, nx = lat.shape
    nz = 6
    cld = np.zeros((2, nz, ny, nx))
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    cld[0, 3] = np.clip(0.05 * i, 0, 1)      # max over k is level 3
    cld[1, 1] = 0.5
    cld[1, 4] = np.clip(0.04 * j, 0, 0.3)    # max(0.5, ...) = 0.5
    i_pt, j_pt, lat_pt, lon_pt = sample_points(lat, lon, 15, rng)
    batch = ob.ObsBatch.from_columns(var=np.full(15, ob.CLDMASK, np.int16), value=np.zeros(15), error=0.3,
                                     lat=lat_pt, lon=lon_pt)
    hx = ops.model_equivalents(batch, {"CLDFRA": cld}, geo)
    assert np.max(np.abs(hx[0] - 0.05 * i_pt)) < 1e-9 and np.max(np.abs(hx[1] - 0.5)) < 1e-12


def test_frame_index_and_4d_binning():
    rng, lat, lon, geo, frames = surface_case()
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 3, rng)
    zmod = 100.0 + 10.0 * i + 5.0 * j
    batch = surface_batch(ob.T2, lat_pt, lon_pt, zmod).replace(dt_s=np.array([-3600.0, 100.0, 9000.0]))
    later = {k: v + 1.0 for k, v in frames.items()}
    hx, qc, which = ops.model_equivalents_4d(batch, [frames, later], [-3600.0, 0.0], geo)
    assert which.tolist() == [0, 1, -1]
    assert np.allclose(hx[:, 1] - 1.0, 290.0 + 0.1 * i[1] - 0.05 * j[1] + np.arange(2) - 0.0065 * 0.0)
    assert np.isnan(hx[:, 2]).all() and qc[2] & ob.QC_OUTSIDE


# ------------------------------------------------------------------ real files
@pytest.fixture
def guarded_wrfout(tmp_path):
    """ALISIOS rule (manager 2026-10-17 01:25Z): the CPU-WRF file is opened only inside a guard_call session
    (scan + admit CALIBRATION_READ for issue 20260502 / valid 20260503, read spy). No guard -> skip, never read."""
    if not Path("<USER_HOME>/src/alisios/src/alisios/guard_call.py").exists():
        pytest.skip("alisios guard_call unavailable: ALISIOS files must not be read unguarded")
    from gpuwrf.assimilation.letkf import alisios_guard as ag

    with ag.guarded_session({WRFOUT_D01: "20260503"}, ["20260502"], tmp_path / "guard") as session:
        if not WRFOUT_D01.exists():
            pytest.skip("CPU-WRF WN3 0502 run not mounted")
        yield session
    assert session.receipt["spy"]["refused_paths"] == [] and session.receipt["guard_result"]


def _read_frame(path, names):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as ds:
        frames = {n: np.asarray(ds.variables[n][0], dtype=float)[None] for n in names}
        geo = {n: np.asarray(ds.variables[n][0], dtype=float) for n in
               ("XLAT", "XLONG", "HGT", "COSALPHA", "SINALPHA", "LANDMASK")}
        geo["DX"] = float(ds.getncattr("DX"))
    return frames, geo


REAL_POINTS = {  # sampling points (lat, lon, elevation m) on land on several islands; not station truth
    "tenerife_los_rodeos": (28.4775, -16.3292, 617.0),
    "tenerife_canadas": (28.224167, -16.626111, 2149.0),
    "gran_canaria_centre": (27.96, -15.59, 1500.0),
    "la_palma_centre": (28.67, -17.85, 1200.0),
    "fuerteventura_centre": (28.40, -14.00, 200.0),
    "lanzarote_centre": (29.03, -13.65, 300.0),
}


def test_real_surface_hx_matches_verify_extract_wrfout(guarded_wrfout):
    pd = pytest.importorskip("pandas")
    from gpuwrf.verify.wrfout import extract_wrfout

    from gpuwrf.verify.sampling import StationSampler

    stations = pd.DataFrame([{"station_uid": k, "lat": v[0], "lon": v[1], "elevation_m": v[2], "landmask": 1,
                              "wind_height_m": 10.0, "roughness_m": 0.03} for k, v in REAL_POINTS.items()])
    frames, geo = _read_frame(WRFOUT_D01, ("T2", "U10", "V10", "Q2", "PSFC"))
    usable = []
    for n in range(len(stations)):
        try:
            StationSampler(geo["XLAT"], geo["XLONG"], stations.iloc[[n]], landmask=geo["LANDMASK"])
            usable.append(n)
        except ValueError:
            pass
    assert len(usable) >= 4
    stations = stations.iloc[usable].reset_index(drop=True)
    rows, _ = extract_wrfout([WRFOUT_D01], stations, method="bilinear", land_aware=True, lapse_rate=0.0065)
    lat = stations.lat.to_numpy()
    lon = stations.lon.to_numpy()
    z = stations.elevation_m.to_numpy()
    n = len(stations)
    for var, column in ((ob.T2, "t2"), (ob.U10, "u_ms"), (ob.V10, "v_ms")):
        batch = ob.ObsBatch.from_columns(var=np.full(n, var, np.int16), value=np.zeros(n), error=1.0, lat=lat,
                                         lon=lon, height_m=z, station=stations.station_uid.to_numpy())
        hx = ops.model_equivalents(batch, frames, geo, dz_max_m=1e9)
        if column == "t2":
            ref = rows[rows.variable == "t2"].set_index("station_uid").loc[stations.station_uid, "value"] + 273.15
        else:
            ref = rows[rows.variable == "wind"].set_index("station_uid").loc[stations.station_uid, column]
        # verify samples float32 file values in float64 like we do -> identical method, round-off only
        assert np.max(np.abs(hx[0] - ref.to_numpy())) < 1e-6, (column, hx[0], ref.to_numpy())


def test_real_sonde_hx_matches_independent_column_loop(guarded_wrfout):
    if not IGRA.exists():
        pytest.skip("IGRA2 file absent")
    frames, geo = _read_frame(WRFOUT_D01, ("T", "QVAPOR", "U", "V", "P", "PB"))
    sonde = ob.read_igra2(IGRA, "SPM00060018", "2026-05-03T12:00")
    hx, qc = ops.model_equivalents(sonde, frames, geo, return_qc=True)
    ok = qc == 0
    assert ok.sum() >= 50  # mandatory + significant levels inside the d01 column
    # independent scalar implementation: locate by brute force inverse bilinear (verify sampler),
    # interpolate each needed level, then ln-p interpolation with explicit loops
    import pandas as pd
    from gpuwrf.verify.sampling import StationSampler

    st = pd.DataFrame({"station_uid": ["guimar"], "lat": [sonde.lat[0]], "lon": [sonde.lon[0]]})
    smp = StationSampler(geo["XLAT"], geo["XLONG"], st, land_aware=False)
    p = frames["P"][0] + frames["PB"][0]
    nz = p.shape[0]
    lnp = np.array([np.log(smp.sample(p[k])[0]) for k in range(nz)])
    lnp_alt = np.array([smp.sample(np.log(p[k]))[0] for k in range(nz)])
    temp = (frames["T"][0] + 300.0) * (p / 1e5) ** (287.0 / 1004.5)
    tcol = np.array([smp.sample(temp[k])[0] for k in range(nz)])
    qcol = np.array([smp.sample(frames["QVAPOR"][0][k])[0] for k in range(nz)])
    for n in np.flatnonzero(ok & np.isin(sonde.var, [ob.T, ob.QV])):
        x = np.log(sonde.p_pa[n])
        col = tcol if sonde.var[n] == ob.T else qcol
        k = max(kk for kk in range(nz - 1) if lnp_alt[kk] >= x)
        w = (x - lnp_alt[k]) / (lnp_alt[k + 1] - lnp_alt[k])
        ref = col[k] + w * (col[k + 1] - col[k])
        tol = 1e-6 if sonde.var[n] == ob.T else 1e-12
        assert abs(hx[0, n] - ref) < tol, (n, hx[0, n], ref)
    # ln(bilinear p) vs bilinear(ln p): the operator uses the latter. Over Guimar's steep coastal terrain the
    # four corner surface pressures differ by ~6 %, so the two differ by up to 6.3e-4 in ln p (~0.6 hPa) in
    # the lowest levels and <= 3e-6 in the upper half (hybrid coordinate) [M, d01 2026-05-03 12Z].
    assert np.max(np.abs(lnp - lnp_alt)) < 1e-3 and np.max(np.abs(lnp - lnp_alt)[nz // 2:]) < 1e-5


def _lambert_ij(lat, lon, attrs, lat00, lon00):
    """WRF module_llxy llij_lc (sphere R = 6370 km) relative to the (0, 0) mass point; independent locator."""
    r_earth = 6370000.0
    t1, t2 = np.deg2rad(float(attrs["TRUELAT1"])), np.deg2rad(float(attrs["TRUELAT2"]))
    cone = (np.log(np.cos(t1)) - np.log(np.cos(t2))) / (np.log(np.tan(np.pi / 4 - t1 / 2)) -
                                                         np.log(np.tan(np.pi / 4 - t2 / 2)))

    def xy(la, lo):
        dlon = (lo - float(attrs["STAND_LON"]) + 180.0) % 360.0 - 180.0
        rm = r_earth / float(attrs["DX"]) * np.cos(t1) / cone * (
            np.tan(np.deg2rad(90.0 - la) / 2) / np.tan((np.pi / 2 - t1) / 2)) ** cone
        arg = cone * np.deg2rad(dlon)
        return rm * np.sin(arg), -rm * np.cos(arg)

    x, y = xy(lat, lon)
    x0, y0 = xy(lat00, lon00)
    return x - x0, y - y0


def test_real_sonde_column_hand_computed_with_lambert_locator(guarded_wrfout):
    if not IGRA.exists():
        pytest.skip("IGRA2 file absent")
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(WRFOUT_D01) as ds:
        attrs = {k: ds.getncattr(k) for k in ("TRUELAT1", "TRUELAT2", "STAND_LON", "DX", "MAP_PROJ")}
    assert int(attrs["MAP_PROJ"]) == 1
    frames, geo = _read_frame(WRFOUT_D01, ("T", "QVAPOR", "U", "V", "P", "PB"))
    sonde = ob.read_igra2(IGRA, "SPM00060018", "2026-05-03T12:00")
    # locator: isoparametric inverse vs exact Lambert projection on the real grid (float32 XLAT/XLONG)
    i_l, j_l = _lambert_ij(sonde.lat[0], sonde.lon[0], attrs, geo["XLAT"][0, 0], geo["XLONG"][0, 0])
    i_o, j_o, inside = ops.GridLocator(geo["XLAT"], geo["XLONG"]).locate(sonde.lat[:1], sonde.lon[:1])
    assert inside[0] and abs(i_o[0] - i_l) < 2e-3 and abs(j_o[0] - j_l) < 2e-3
    # hand-computed column at the Lambert location: explicit corners, ln p linear interpolation
    i0, j0 = int(np.floor(i_l)), int(np.floor(j_l))
    fx, fy = i_l - i0, j_l - j0

    def bil(f2d, x0=i0, y0=j0, wx=fx, wy=fy):
        return ((1 - wy) * ((1 - wx) * f2d[y0, x0] + wx * f2d[y0, x0 + 1])
                + wy * ((1 - wx) * f2d[y0 + 1, x0] + wx * f2d[y0 + 1, x0 + 1]))

    p = frames["P"][0] + frames["PB"][0]
    nz = p.shape[0]
    lnp = [bil(np.log(p[k])) for k in range(nz)]
    temp = (frames["T"][0] + 300.0) * (p / 1.0e5) ** (287.0 / 1004.5)
    # U on the x-staggered grid: mass i <-> u index i + 0.5; V: mass j <-> v index j + 0.5
    ui, uf = int(np.floor(i_l + 0.5)), i_l + 0.5 - np.floor(i_l + 0.5)
    vj, vf = int(np.floor(j_l + 0.5)), j_l + 0.5 - np.floor(j_l + 0.5)
    cosa, sina = bil(geo["COSALPHA"]), bil(geo["SINALPHA"])
    hx = ops.model_equivalents(sonde, frames, geo)[0]
    checked = 0
    for n in range(len(sonde)):
        x = np.log(sonde.p_pa[n])
        ks = [k for k in range(nz - 1) if lnp[k] >= x > lnp[k + 1] or lnp[k] > x >= lnp[k + 1]]
        if not ks:
            assert np.isnan(hx[n])   # outside the column (surface level, stratosphere)
            continue
        k = ks[0]
        w = (x - lnp[k]) / (lnp[k + 1] - lnp[k])

        def lev(col):
            return col[k] + w * (col[k + 1] - col[k])

        var = sonde.var[n]
        if var == ob.T:
            ref = lev([bil(temp[kk]) for kk in range(nz)])
        elif var == ob.QV:
            ref = lev([bil(frames["QVAPOR"][0][kk]) for kk in range(nz)])
        else:
            ug = [bil(frames["U"][0][kk], ui, j0, uf, fy) for kk in range(nz)]
            vg = [bil(frames["V"][0][kk], i0, vj, fx, vf) for kk in range(nz)]
            ue = [a * cosa - b * sina for a, b in zip(ug, vg)]
            ve = [b * cosa + a * sina for a, b in zip(ug, vg)]
            ref = lev(ue if var == ob.U else ve)
        # location differs from the isoparametric inverse by < 2e-3 grid units -> small field differences
        tol = {ob.T: 2e-3, ob.QV: 2e-6, ob.U: 5e-3, ob.V: 5e-3}[int(var)]
        assert abs(hx[n] - ref) < tol, (n, int(var), hx[n], ref)
        checked += 1
    assert checked >= 50


def test_float32_frames_give_identical_hx_without_full_copies():
    """Gathering in native float32 then casting the corners is bit-identical to casting whole fields first."""
    rng, lat, lon, geo, frames, wind, _ = analytic_case(alpha_deg=21.0)
    _, _, _, _, sframes = surface_case(alpha_deg=21.0)
    f32 = {k: np.asarray(v, np.float32) for k, v in {**frames, **wind, **sframes}.items() if k != "P" and k != "PB"}
    f32["P"], f32["PB"] = np.asarray(frames["P"], np.float32), np.asarray(frames["PB"], np.float32)
    f64 = {k: v.astype(np.float64) for k, v in f32.items()}
    i, j, lat_pt, lon_pt = sample_points(lat, lon, 12, rng)
    lnp_obs = np.log(1.0e5) - rng.uniform(0.2, 0.9, 12)
    sonde = ob.concat([sonde_batch(v, lat_pt, lon_pt, np.exp(lnp_obs)) for v in (ob.T, ob.QV)])
    zmod = 100.0 + 10.0 * i + 5.0 * j
    surf = ob.concat([surface_batch(v, lat_pt, lon_pt, zmod + 20.0) for v in (ob.T2, ob.Q2, ob.U10, ob.V10)])
    for batch, a, b in ((sonde, {k: f32[k] for k in ("P", "PB", "T", "QVAPOR")},
                         {k: f64[k] for k in ("P", "PB", "T", "QVAPOR")}),
                        (surf, {k: f32[k] for k in sframes}, {k: f64[k] for k in sframes})):
        h32, h64 = ops.model_equivalents(batch, a, geo), ops.model_equivalents(batch, b, geo)
        assert h32.dtype == np.float64 and np.array_equal(h32, h64)
