"""LETKF product gates: analytic Kalman limits, the analysis hooks and a Lorenz-96 OSSE.

Independent references: the state-space ensemble Kalman update (``core.kalman_reference``, plain matrix
algebra), the scalar Kalman gain and a free (no-DA) Lorenz-96 run. Hooks are checked against the default path
(bitwise where they must be) and against what they claim to change.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import analysis as an
from gpuwrf.assimilation.letkf import core, grid as lgrid
from gpuwrf.assimilation.letkf import obs as ob

from _synth import PARAMS, concat, ensemble, hx, obs_at


# ------------------------------------------------------------------------------------------- analytic Kalman

@pytest.mark.parametrize("rho", [1.0, 1.7])
def test_inflated_etkf_equals_kalman_with_inflated_covariance(rho):
    rng = np.random.default_rng(11)
    n, nstate, p = 12, 7, 4
    xb = rng.normal(size=(n, nstate)) + np.arange(nstate)
    h = rng.normal(size=(p, nstate))
    yo = rng.normal(size=p)
    sigma = np.full(p, 0.7)
    xa = core.etkf_global(xb, xb @ h.T, yo, sigma, rho=rho)
    mean = xb.mean(0)
    inflated = mean + np.sqrt(rho) * (xb - mean)        # Pb -> rho Pb
    ref_mean, ref_cov = core.kalman_reference(inflated, h, yo, sigma)
    assert np.allclose(xa.mean(0), ref_mean, rtol=0, atol=1e-10)
    assert np.allclose(np.cov(xa.T, ddof=1), ref_cov, rtol=0, atol=1e-10)


def test_per_point_inflation_constant_is_bitwise_scalar_and_varies_per_point():
    rng = np.random.default_rng(3)
    yb = rng.normal(size=(5, 4, 8))
    d = rng.normal(size=(5, 4))
    rinv = rng.uniform(0.1, 1.0, size=(5, 4))
    _, _, t_scalar = core.letkf_transform(yb, d, rinv, rho=1.3)
    _, _, t_array = core.letkf_transform(yb, d, rinv, rho=np.full(5, 1.3))
    assert np.array_equal(t_scalar, t_array)
    rhos = np.array([1.0, 1.1, 1.5, 2.0, 3.0])
    _, _, t_mixed = core.letkf_transform(yb, d, rinv, rho=rhos)
    for i, r in enumerate(rhos):
        _, _, t_one = core.letkf_transform(yb[i:i + 1], d[i:i + 1], rinv[i:i + 1], rho=r)
        assert np.allclose(t_mixed[i], t_one[0], rtol=0, atol=1e-13)
    with pytest.raises(ValueError):
        core.letkf_transform(yb, d, rinv, rho=np.ones(4))


def test_analysis_scalar_kalman_gain_single_obs():
    frames, geo = ensemble(seed=1)
    o = obs_at([(2, 15, 17)], [12.0], [0.5], frames, geo)
    xa, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    xb = frames["T"][:, 2, 15, 17]
    var = xb.var(ddof=1)
    gain = var / (var + 0.25)
    expected = xb.mean() + gain * (12.0 - xb.mean())
    assert receipt["admissible"] and receipt["assimilated"] == 1
    assert abs(xa["T"][:, 2, 15, 17].mean() - expected) < 1e-6
    assert abs(xa["T"][:, 2, 15, 17].var(ddof=1) - (1 - gain) * var) < 1e-6


# ------------------------------------------------------------------------------------------- hooks

def _two_kinds(frames, geo):
    sonde = obs_at([(2, 15, 17), (3, 14, 18)], [11.5, 9.0], [0.6, 0.6], frames, geo)
    surface = obs_at([(0, 16, 16)], [12.5], [0.8], frames, geo, var=ob.T2)
    return concat(sonde, surface)


def test_default_hooks_reproduce_plain_analysis_bitwise():
    frames, geo = ensemble(seed=2)
    o = _two_kinds(frames, geo)
    base, r0 = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    ones = {name: {"*": 1.0} for name in ("T", "QVAPOR", "U", "V")}
    same, r1 = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, var_loc=ones,
                          prior_inflation=np.ones(1), maintenance=lambda name, xb, xa: core.rtps(xb, xa, 0.0))
    for name in ("T", "QVAPOR", "U", "V"):
        assert np.array_equal(base[name], same[name]), name
    assert r1["hooks"]["var_loc_groups"] == [["T", "QVAPOR", "U", "V"]]


def test_variable_localisation_zero_factor_removes_that_kind_for_that_variable_only():
    frames, geo = ensemble(seed=3)
    o = _two_kinds(frames, geo)
    base, _ = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    loc, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, var_loc={"T": {"surface": 0.0}})
    sonde_only = concat(*(obs_at([(2, 15, 17), (3, 14, 18)], [11.5, 9.0], [0.6, 0.6], frames, geo),))
    ref, _ = an.analyse(frames, geo, sonde_only, PARAMS, model_equivalents=hx)
    assert np.allclose(loc["T"], ref["T"], rtol=0, atol=1e-9)
    assert not np.allclose(loc["T"], base["T"], rtol=0, atol=1e-6)         # the surface obs did matter for T
    for name in ("QVAPOR", "U", "V"):                                         # other variables keep every obs
        assert np.array_equal(loc[name], base[name]), name
    assert sorted(map(sorted, receipt["hooks"]["var_loc_groups"])) == [["QVAPOR", "U", "V"], ["T"]]
    with pytest.raises(ValueError):
        an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, var_loc={"W": {"*": 0.5}})


def test_specific_obs_key_beats_kind_and_wildcard():
    frames, geo = ensemble(seed=4)
    o = _two_kinds(frames, geo)
    sel = np.arange(3)
    groups = an.variable_groups(["T", "QVAPOR"], o, sel, {"T": {"*": 0.2, "sonde": 0.5, "surface:T2": 0.9}})
    factors = {g[0]: f for g, f in groups}
    assert factors["T"].tolist() == [0.5, 0.5, 0.9] and factors["QVAPOR"] is None


def test_prior_inflation_field_matches_scalar_inflation_where_equal():
    frames, geo = ensemble(seed=5)
    o = _two_kinds(frames, geo)
    scalar, _ = an.analyse(frames, geo, o, replace(PARAMS, inflation=1.6), model_equivalents=hx)
    field, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx,
                                prior_inflation=lambda agrid: np.full(agrid.shape, 1.6))
    for name in ("T", "QVAPOR", "U", "V"):
        assert np.allclose(field[name], scalar[name], rtol=0, atol=1e-10), name
    assert receipt["hooks"]["prior_inflation"] == "per-point"
    with pytest.raises(ValueError):
        an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, prior_inflation=np.zeros(1))


def test_maintenance_hook_receives_every_analysed_name_once():
    frames, geo = ensemble(seed=6)
    o = _two_kinds(frames, geo)
    seen = []

    def relax_all(name, xb, xa):
        seen.append(name)
        return xb + 0.5 * (xa - xb)

    half, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, maintenance=relax_all)
    full, _ = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    assert sorted(seen) == ["QVAPOR", "T", "U", "V"] and receipt["hooks"]["maintenance"] == "relax_all"
    inc_half = half["T"].astype(float) - frames["T"]
    inc_full = full["T"].astype(float) - frames["T"]
    assert np.allclose(inc_half, 0.5 * inc_full, rtol=0, atol=1e-6)


# ------------------------------------------------------------------------------------------- Lorenz-96 OSSE

def _l96(x, forcing=8.0):
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + forcing


def _rk4(x, dt, steps):
    for _ in range(steps):
        k1 = _l96(x)
        k2 = _l96(x + 0.5 * dt * k1)
        k3 = _l96(x + 0.5 * dt * k2)
        k4 = _l96(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def l96_osse(seed, *, cycles=600, n_vars=40, n_members=20, loc=3.0, rtps_alpha=0.3, assimilate=True,
             transform=core.letkf_transform):
    """40-variable L96 (F=8, dt=0.01, obs of every variable every 5 steps, sigma_o=1), one local analysis per
    variable with GC R-localisation on the periodic index distance (Hunt et al. 2007 setup)."""
    rng = np.random.default_rng(seed)
    truth = _rk4(8.0 + rng.normal(size=n_vars), 0.01, 2000)
    xs = _rk4(truth + rng.normal(size=(n_members, n_vars)), 0.01, 1000)
    dist = np.abs(np.arange(n_vars)[:, None] - np.arange(n_vars)[None, :])
    dist = np.minimum(dist, n_vars - dist)
    rho = core.gaspari_cohn(dist / core.gc_halfwidth(loc))
    rmse, spread = [], []
    for _ in range(cycles):
        truth = _rk4(truth, 0.01, 5)
        xs = _rk4(xs, 0.01, 5)
        if assimilate:
            yo = truth + rng.normal(size=n_vars)
            ybar = xs.mean(0)
            ypert = np.broadcast_to((xs - ybar).T, (n_vars, n_vars, n_members))
            d = np.broadcast_to(yo - ybar, (n_vars, n_vars))
            _, _, t = transform(ypert, d, rho)
            xs = core.rtps(xs, core.apply_transform(xs, t), rtps_alpha)
        rmse.append(float(np.sqrt(np.mean((xs.mean(0) - truth) ** 2))))
        spread.append(float(np.sqrt(np.mean(xs.var(0, ddof=1)))))
    tail = slice(cycles // 4, None)
    return float(np.mean(rmse[tail])), float(np.mean(spread[tail])) / float(np.mean(rmse[tail]))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lorenz96_osse(seed):
    """Declared test setting N=20, L=3, RTPS 0.3 (chosen from the A1 probe sweep; the probe's pre-registered
    RTPS 0.6 run missed its 0.30 bar). Gate: tail analysis RMSE < 0.25 sigma_o, spread/RMSE in [0.8, 1.4]."""
    rmse, ratio = l96_osse(seed)
    assert rmse < 0.25 and 0.8 <= ratio <= 1.4, (rmse, ratio)
    free, _ = l96_osse(seed, cycles=200, assimilate=False)
    assert free > 2.0                                   # the free run has climatological error (~3.6)


def test_lorenz96_mutant_without_square_root_is_caught():
    def no_sqrt(yb, d, rinv, rho=1.0):
        wmean, w, _ = core.letkf_transform(yb, d, rinv, rho=rho)
        n = yb.shape[-1]
        w_bad = np.einsum("nij,njk->nik", w, w) / (n - 1)       # Pa~ (N-1)^-1 W W = Pa~ instead of its root
        return wmean, w_bad, w_bad + wmean[:, :, None]

    rmse, _ = l96_osse(1, cycles=300, transform=no_sqrt)
    assert not rmse < 0.25


# --- critic cr-w41-da-core items 2 (case boundary width) and 3 (observation roles, cloud) ----------------------------

def test_namelist_without_spec_bdy_width_runs_the_wrf_registry_default(tmp_path):
    from gpuwrf.assimilation.letkf.state_io import _namelist_spec_bdy_width

    (tmp_path / "set").write_text("&bdy_control\n spec_bdy_width = 7,\n/\n")
    (tmp_path / "unset").write_text("&bdy_control\n specified = .true.,\n/\n")
    assert _namelist_spec_bdy_width(tmp_path / "set") == (7, str(tmp_path / "set"), False)
    assert _namelist_spec_bdy_width(tmp_path / "unset") == (5, str(tmp_path / "unset"), True)
    assert _namelist_spec_bdy_width(tmp_path / "missing") == (None, "absent", False)
    width, source = an.resolve_spec_bdy_width({"SPEC_BDY_WIDTH": 5, "SPEC_BDY_WIDTH_unset": True})
    assert width == 5 and "WRF Registry default 5" in source
    assert an.resolve_spec_bdy_width({"SPEC_BDY_WIDTH": None}) == (5, "WRF Registry default 5 (no case namelist, "
                                                                       "no explicit value)")


def test_case_boundary_width_binds_the_taper_and_an_explicit_contradiction_is_refused():
    frames, geo = ensemble(seed=1)
    geo["SPEC_BDY_WIDTH"], geo["SPEC_BDY_WIDTH_source"] = 9, "test namelist"
    obs = obs_at([(2, 6, 17)], [12.0], [0.5], frames, geo)
    xa, receipt = an.analyse(frames, geo, obs, PARAMS, model_equivalents=hx)
    delta = xa["T"] - frames["T"]
    assert receipt["spec_bdy_width"] == {"value": 9, "source": "case namelist", "namelist": "test namelist"}
    assert not np.any(delta[:, :, :9]) and np.any(delta[:, :, 9:])
    # the same observation reaches rows 5-8 under a width-5 zone, so the width check above is live
    xa5, _ = an.analyse(frames, {**geo, "SPEC_BDY_WIDTH": None}, obs, replace(PARAMS, spec_bdy_width=5),
                        model_equivalents=hx)
    assert np.count_nonzero((xa5["T"] - frames["T"])[:, :, 5:9]) > 100
    with pytest.raises(ValueError, match="contradicts the case's spec_bdy_width = 9"):
        an.analyse(frames, geo, obs, replace(PARAMS, spec_bdy_width=5), model_equivalents=hx)


def test_role_withheld_and_cloud_rows_never_influence_the_analysis():
    frames, geo = ensemble(seed=5)
    v1, v2 = (float(frames["T"][:, 2, j, i].mean()) + 0.5 for j, i in ((15, 17), (12, 20)))
    base = obs_at([(2, 15, 17), (2, 12, 20)], [v1, v2], [0.5, 0.5], frames, geo)
    only_first = obs_at([(2, 15, 17)], [v1], [0.5], frames, geo)
    xa1, _ = an.analyse(frames, geo, only_first, PARAMS, model_equivalents=hx)
    base.role = np.array(["assimilate", "withhold"], object)           # a source marked the 2nd row withhold
    xa, receipt = an.analyse(frames, geo, base, PARAMS, model_equivalents=hx)
    assert receipt["assimilated"] == 1 and receipt["withheld_ok"] == 1 and receipt["qc"]["role_withheld"] == 1
    assert not np.array_equal(xa1["T"], frames["T"].astype(xa1["T"].dtype))      # the first row is assimilated
    assert np.array_equal(xa["T"], xa1["T"])                            # bitwise: the withheld row has no influence
    cloud = obs_at([(2, 15, 17)], [v1], [0.5], frames, geo)
    cloud.kind = np.full(1, 3)                                          # a cloud-kind row (no cloud analysis)
    xc, rc = an.analyse(frames, geo, cloud, PARAMS, model_equivalents=hx)
    assert rc["assimilated"] == 0 and rc["qc"]["cloud_not_assimilated"] == 1
    assert np.array_equal(xc["T"], frames["T"].astype(xc["T"].dtype))
