"""Adaptive (cycled, debiased, variable x band) inflation and the online obs-bias estimate (lane w41-da-ens)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import inflation as inf


def _surface_obs(n, value, error=1.0, var=11, station=None):
    return types.SimpleNamespace(kind=np.full(n, 2), var=np.full(n, var), value=np.asarray(value, float),
                                 error=np.full(n, float(error)), lat=np.zeros(n), lon=np.zeros(n),
                                 p_pa=np.full(n, np.nan), qc=np.zeros(n, int),
                                 station=np.array(station if station is not None else [f"s{i}" for i in range(n)],
                                                  dtype=object),
                                 site=np.array([""] * n, dtype=object))


def _draw(rng, n, rho_true, sb, so, bias, members=20):
    """Members with spread sb at each obs and innovations d ~ N(bias, rho_true sb² + so²)."""
    z = rng.normal(size=(members, n))
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    hx = sb * z
    d = bias + np.sqrt(rho_true * sb ** 2 + so ** 2) * rng.normal(size=n)
    return hx, d


def test_single_group_fusion_matches_hand_formula():
    rng = np.random.default_rng(0)
    hx, d = _draw(rng, 40, 3.0, 0.5, 1.0, 0.0)
    obs = _surface_obs(40, d)
    ai = inf.AdaptiveInflation(rho0=1.5, var0=0.7, rho_max=50.0)
    out = ai.update(obs, hx, np.ones(40, bool), None, None)
    dd = d - d.mean()
    sb2, so2 = np.mean(hx.var(0, ddof=1)), 1.0
    rho_o = (np.sum(dd ** 2) / 39 - so2) / sb2
    v_o = 2.0 / 39 * ((1.5 * sb2 + so2) / sb2) ** 2
    expect = (1.5 / 0.7 + rho_o / v_o) / (1 / 0.7 + 1 / v_o)
    est = out["estimates"]["T|0-1km"]
    assert abs(est["rho_posterior_raw"] - expect) < 1e-12
    assert abs(est["sd"] ** 2 - 1 / (1 / 0.7 + 1 / v_o)) < 1e-12
    # bands/variables without obs keep the prior exactly
    assert out["estimates"]["T|1-2km"]["rho"] == 1.5 and out["estimates"]["U|0-1km"]["rho"] == 1.5


def test_cycled_estimate_converges_and_debiasing_removes_bias():
    rng = np.random.default_rng(1)
    ai = inf.AdaptiveInflation(rho_max=20.0)
    naive = inf.AdaptiveInflation(rho_max=20.0)
    hist = []
    for _ in range(30):
        hx, d = _draw(rng, 200, 4.0, 0.5, 1.0, bias=-1.8)
        obs = _surface_obs(200, d)
        ai.update(obs, hx, np.ones(200, bool), None, None)
        naive.update(obs, hx, np.ones(200, bool), None, None, bias=np.zeros(200))   # mutant: bias kept in d
        hist.append(ai.rho["T|0-1km"])
    assert abs(np.mean(hist[10:]) - 4.0) < 0.5 and ai.var["T|0-1km"] < 0.3
    assert naive.rho["T|0-1km"] > 10.0           # the -1.8 bias masquerades as spread deficit (R3 (1))


def test_sparse_low_signal_group_shrinks_to_prior():
    """R3/BA: surface T2 spread 0.065 K vs sigma_o 1.75 K, n = 2 -> raw rho is noise (BA: 0228 V10 raw 69)."""
    rng = np.random.default_rng(2)
    hx, d = _draw(rng, 2, 1.0, 0.065, 1.75, 0.0)
    d = np.array([3.0, -3.0])                     # large innovations from a tiny sample
    ai = inf.AdaptiveInflation(rho0=1.2, var0=0.25, rho_max=100.0)
    out = ai.update(_surface_obs(2, d), hx, np.ones(2, bool), None, None)
    est = out["estimates"]["T|0-1km"]
    assert est["parts"]["surface:T2"]["rho_obs"] > 100      # the per-cycle estimate alone would explode
    assert abs(est["rho"] - 1.2) < 0.01                       # fused: stays at the prior


def test_state_roundtrip_and_random_walk_variance():
    ai = inf.AdaptiveInflation(q=0.09)
    rng = np.random.default_rng(3)
    hx, d = _draw(rng, 30, 2.0, 0.5, 1.0, 0.0)
    ai.update(_surface_obs(30, d), hx, np.ones(30, bool), None, None)
    back = inf.AdaptiveInflation.from_dict(ai.to_dict())
    assert back.rho == ai.rho and back.var == ai.var and back.cycles == 1
    rho_b, var_b = back.prior()
    assert abs(var_b["T|1-2km"] - (ai.var["T|1-2km"] + 0.09)) < 1e-15
    with pytest.raises(ValueError):
        inf.AdaptiveInflation.from_dict({"schema": "other"})


def test_obs_bias_converges_and_corrects_only_requested_kinds():
    rng = np.random.default_rng(4)
    ob = inf.ObsBias(apply_kinds=(2,))
    for _ in range(15):
        hx = rng.normal(size=(10, 40))
        d = -1.8 + rng.normal(size=40)
        obs = _surface_obs(40, hx.mean(0) + d)
        ob.update(obs, hx, np.ones(40, bool))
    assert abs(ob.groups["surface:T2"]["bias"] + 1.8) < 0.15
    obs = _surface_obs(3, [10.0, 11.0, 12.0])
    sonde = types.SimpleNamespace(**{**vars(_surface_obs(1, [5.0])), "kind": np.array([1]), "var": np.array([1])})
    fixed, rec = ob.correct(obs)
    np.testing.assert_allclose(fixed.value, obs.value - ob.groups["surface:T2"]["bias"])
    assert rec["applied"] == 3
    fixed_sonde, rec2 = ob.correct(sonde)
    assert fixed_sonde.value[0] == 5.0 and rec2["applied"] == 0
    back = inf.ObsBias.from_dict(ob.to_dict())
    assert back.groups == ob.groups


def test_obs_bias_gain_includes_coherent_forecast_error():
    """Swiss twin cycle 0: an UNBIASED group whose forecast error is coherent over the domain got a -2 m/s 'bias'
    (K ~ 1). The ensemble's variance of the group-mean H(x) enters the estimator variance: one-cycle hand formula,
    and over 40 cycles of zero-mean coherent errors the estimate stays small; the mutant (coherent=False, the v1
    estimator) chases every cycle's error."""
    rng = np.random.default_rng(7)
    ob = inf.ObsBias()
    hx = rng.normal(size=(10, 40)) + 2.0 * rng.normal(size=(10, 1))
    y = hx.mean(0) + 1.5 + rng.normal(size=40)
    rec = ob.update(_surface_obs(40, y), hx, np.ones(40, bool))["surface:T2"]
    d = y - hx.mean(0)
    coh = hx.mean(1).var(ddof=1)
    vb = 0.25 + 0.01
    gain = vb / (vb + coh + max(d.var(ddof=1), 1.0) / 40)
    assert abs(rec["gain"] - gain) < 1e-12 and abs(rec["coherent_fc_var"] - coh) < 1e-12
    new, mutant, est_new, est_mut = inf.ObsBias(), inf.ObsBias(coherent=False), [], []
    for _ in range(40):
        a = rng.normal(size=(10, 1))
        a = 2.0 * (a - a.mean()) / a.std(ddof=1)
        noise = 0.3 * rng.normal(size=(10, 40))
        y = 2.0 * rng.normal() + rng.normal(size=40)          # zero-mean coherent error + obs noise, no bias
        new.update(_surface_obs(40, y), a + noise, np.ones(40, bool))
        mutant.update(_surface_obs(40, y), a + noise, np.ones(40, bool))
        est_new.append(new.groups["surface:T2"]["bias"])
        est_mut.append(mutant.groups["surface:T2"]["bias"])
    assert np.sqrt(np.mean(np.square(est_new[5:]))) < 0.6 < 1.2 < np.sqrt(np.mean(np.square(est_mut[5:])))


def test_profile_ids_set_the_sampling_count():
    """Levels of one profile are not independent: 40 obs from 4 stations -> v_o uses n_eff = 4 * 39/40."""
    rng = np.random.default_rng(8)
    hx, d = _draw(rng, 40, 3.0, 0.5, 1.0, 0.0)
    obs = _surface_obs(40, d, station=[f"s{i // 10}" for i in range(40)])
    out = inf.AdaptiveInflation(rho0=1.5, var0=0.7, rho_max=50.0).update(obs, hx, np.ones(40, bool), None, None)
    part = out["estimates"]["T|0-1km"]["parts"]["surface:T2"]
    sb2 = np.mean(hx.var(0, ddof=1))
    assert abs(part["n_eff"] - 4 * 39 / 40) < 1e-12 and out["n_eff"] == "distinct profiles"
    assert abs(part["sd_obs"] ** 2 - 2.0 / (4 * 39 / 40) * ((1.5 * sb2 + 1.0) / sb2) ** 2) < 1e-9
    assert abs(part["rho_obs"] - (np.sum((d - d.mean()) ** 2) / 39 - 1.0) / sb2) < 1e-9


def test_obs_bias_per_station_keys():
    ob = inf.ObsBias(by="station")
    hx = np.zeros((5, 4))
    obs = _surface_obs(4, [1.0, 1.2, -2.0, -2.2], station=["a", "a", "b", "b"])
    ob.update(obs, hx, np.ones(4, bool))
    assert ob.groups["surface:T2:a"]["bias"] > 0 > ob.groups["surface:T2:b"]["bias"]
    one = inf.ObsBias(by="station")                 # one obs per station and cycle still updates (n_min 1)
    one.update(_surface_obs(2, [1.0, -1.0], station=["a", "b"]), np.zeros((5, 2)) + np.arange(5)[:, None] * 0.1,
               np.ones(2, bool))
    assert one.groups["surface:T2:a"]["bias"] > 0 > one.groups["surface:T2:b"]["bias"]
    with pytest.raises(ValueError, match="obs_bias.v2"):
        inf.ObsBias.from_dict({**one.to_dict(), "schema": "w41.obs_bias.v1"})


def test_config_section_validation():
    ok = inf.validate_config_section({"method": "adaptive", "obs_bias": {"by": "station"},
                                      "maintenance": {"method": "rtps", "alpha": 0.5, "target": "inflated", "rho": 2},
                                      "additive": {"pool_npz": "/x/pool.npz", "amplitude": {"T": 0.2}, "seed": 4}})
    assert ok["adaptive"]["rho_min"] == 1.0 and ok["obs_bias"]["apply_kinds"] == ["surface"]
    assert inf.read_plan(ok) == ["/x/pool.npz"]
    for bad in ({"method": "magic"}, {"method": "probe", "obs_bias": {}}, {"maintenance": {"alpha": 2}},
                {"additive": {"pool_npz": "p"}}, {"method": "adaptive", "adaptive": {"rho_min": 3, "rho_max": 2}},
                {"surprise": 1}):
        with pytest.raises(ValueError):
            inf.validate_config_section(bad)
    from gpuwrf.assimilation.letkf import config
    assert config._delegated("inflation", "config.inflation", {"method": "adaptive"})["method"] == "adaptive"


def test_calibrate_dispatch_probe_path_and_refusal(monkeypatch):
    from gpuwrf.assimilation.letkf import api
    seen = {}
    monkeypatch.setattr(api, "calibrate_scale", lambda f, g, o, s: seen.setdefault("probe", (f, s)) and (f, {"x": 1}))
    frames, ens = {"T": np.zeros((3, 2, 2, 2))}, {"kind": "members", "scale": "innovation"}
    out = inf.calibrate(frames, None, None, ens, None)
    assert out == (frames, {"x": 1}) and seen["probe"][1] is ens
    inf.calibrate(frames, None, None, ens, {"method": "probe"})
    with pytest.raises(ValueError):
        inf.calibrate(frames, None, None, ens, inf.validate_config_section({"method": "adaptive"}))


def test_pool_npz_roundtrip(tmp_path):
    from gpuwrf.assimilation.letkf import spread
    rng = np.random.default_rng(0)
    dev = spread.pool_from_states({"T": rng.normal(size=(5, 2, 3, 4)), "QVAPOR": np.exp(rng.normal(size=(5, 2, 3, 4)))})
    meta = spread.write_pool_npz(tmp_path / "pool.npz", dev, {"source": "test"})
    back = spread.load_pool_npz(tmp_path / "pool.npz")
    assert meta["log_fields"] == ["QVAPOR"] and set(back) == {"T", "QVAPOR"}
    np.testing.assert_allclose(back["T"], dev["T"], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        spread.write_pool_npz(tmp_path / "pool.npz", dev, {})


def test_chain_section_for_cycle_loops(tmp_path):
    sec = inf.validate_config_section({"method": "adaptive", "obs_bias": {}})
    c1 = inf.chain_section(sec, None, tmp_path / "c1")
    assert c1["adaptive"]["state_in"] is None and c1["adaptive"]["state_out"].endswith("c1/adaptive_state.json")
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "adaptive_state.json").write_text("{}")
    c2 = inf.chain_section(sec, tmp_path / "c1", tmp_path / "c2")
    assert c2["adaptive"]["state_in"].endswith("c1/adaptive_state.json") and c2["obs_bias"]["state_in"] is None
    assert sec["adaptive"]["state_in"] is None                       # input section untouched


def test_section_paths_never_resolve_against_the_cwd(tmp_path, monkeypatch):
    """Critic 12:26Z: relative state/pool paths stayed relative (CWD-dependent) while member_files resolve against the
    config directory. Now: absolute kept, relative resolved against base_dir, relative without base_dir refused —
    also through the core config delegation; chain_section emits absolute paths."""
    from gpuwrf.assimilation.letkf import config as dacfg
    raw = {"method": "adaptive", "adaptive": {"state_in": "state/prev.json", "state_out": "/abs/next.json"},
           "additive": {"pool_npz": "pools/p.npz", "amplitude": 0.1, "seed": 1}}
    out = inf.validate_config_section(raw, base_dir="/cfg/dir")
    assert out["adaptive"]["state_in"] == "/cfg/dir/state/prev.json" and out["adaptive"]["state_out"] == "/abs/next.json"
    assert out["additive"]["pool_npz"] == "/cfg/dir/pools/p.npz"
    with pytest.raises(ValueError, match="absolute"):
        inf.validate_config_section(raw)
    with pytest.raises(dacfg.ConfigError, match="absolute"):
        dacfg._delegated("inflation", "config.inflation", raw)
    monkeypatch.chdir(tmp_path)
    sec = inf.validate_config_section({"method": "adaptive", "obs_bias": {}})
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "obs_bias.json").write_text("{}")
    c2 = inf.chain_section(sec, "c1", "c2")                           # relative dirs from a cycle driver
    assert c2["obs_bias"]["state_in"] == str(tmp_path / "c1" / "obs_bias.json")
    assert c2["adaptive"]["state_out"] == str(tmp_path / "c2" / "adaptive_state.json")


def test_rtps_toward_calibrated_transform_rho(monkeypatch):
    """R3 item 4 in the probe mode: rho in the transform AND RTPS relaxing toward sqrt(rho)*sigma_b."""
    from gpuwrf.assimilation.letkf import api
    monkeypatch.setattr(api, "calibrate_scale", lambda f, g, o, s: (f, {"inflation_rho": 4.0}))
    sec = inf.validate_config_section({"maintenance": {"method": "rtps", "alpha": 1.0, "target": "inflated",
                                                       "rho": "calibrated"}})
    hook = inf.analysis_hooks(sec)["maintenance"]          # api builds hooks BEFORE calibrate (lazy rho)
    inf.calibrate({"T": np.zeros((2, 1, 1, 1))}, None, None, {"kind": "members", "scale": "innovation"}, sec)
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(12, 3, 4))
    xa = xb.mean(0) + 0.3 * (xb - xb.mean(0))
    np.testing.assert_allclose(hook("T", xb, xa).std(0, ddof=1), 2.0 * xb.std(0, ddof=1), rtol=1e-12)
    monkeypatch.setattr(inf, "_CALIBRATED", {"rho": None})
    with pytest.raises(ValueError):
        inf.analysis_hooks(sec)["maintenance"]("T", xb, xa)


def test_state_inflation_reaches_member_mass_fields():
    """Critic cr-w41-da-ens P1: inflate_state rescaled T/QV but kept the raw P/PH, and analyse rebalances only
    analysis-minus-prior, so written members missed the inflation's hydrostatic increment (794 m2/s2 PH at rho 4).
    Now the inflated prior equals the raw member + rebalance_increment(raw, inflated); a following analysis
    increment on top reproduces the one-step full rebalance to round-off; rebalance=False is the mutant."""
    from gpuwrf.assimilation.letkf import state_io
    from test_state_io import blob, synthetic_background, synthetic_geo
    geo = synthetic_geo(nz=12, ny=26, nx=28, dtype=np.float64)
    base = synthetic_background(geo, dtype=np.float64)
    members = []
    for a in (-1.5, -0.5, 0.5, 1.5):
        an = {**base, "T": base["T"] + a * blob(base["MU"].shape, width=5)[None]}
        members.append({**an, **state_io.rebalance_increment(base, an, geo)})
    frames = {k: np.stack([m[k] for m in members]) for k in members[0]}
    for k in ("PB", "PHB"):
        frames[k] = np.broadcast_to(geo[k], (4,) + geo[k].shape).copy()
    table = {k: {"rho": (4.0 if k.startswith("T|") else 1.0)} for k in inf.BAND_KEYS}
    prior, cnt = inf.inflate_state(frames, geo, table)
    stale, cnt_off = inf.inflate_state(frames, geo, table, rebalance=False)
    assert cnt["rebalance"].startswith("applied") and cnt_off["rebalance"] == "off"
    worst_missing = 0.0
    for i in range(4):
        raw = {k: v[i] for k, v in frames.items()}
        infl = {k: v[i] for k, v in prior.items()}
        full = state_io.rebalance_increment(raw, infl, geo)
        np.testing.assert_array_equal(infl["PH"], full["PH"])
        np.testing.assert_array_equal(infl["P"], full["P"])
        assert state_io.rebalance_report(raw, infl, full, geo)["residual_change_rel_max"] < 1e-12
        worst_missing = max(worst_missing, float(np.abs(stale["PH"][i] - full["PH"]).max()))
        # an analysis increment on top (what analyse() rebalances) == the one-step rebalance from the raw member
        an = {**infl, "T": infl["T"] + 0.3 * blob(base["MU"].shape, width=4)[None]}
        two_step = state_io.rebalance_increment(infl, an, geo)
        one_step = state_io.rebalance_increment(raw, an, geo)
        assert np.abs(two_step["PH"] - one_step["PH"]).max() < 1e-6 * np.abs(one_step["PH"] - raw["PH"]).max()
    assert worst_missing > 100.0                     # the mutant (no rebalance) misses hundreds of m2/s2


def test_state_inflation_protects_the_case_boundary_width():
    """Sibling of cr-w41-da-core (2): the adaptive prior inflation must leave the CASE specified+relaxation zone
    (geo SPEC_BDY_WIDTH from namelist.input, here 9) untouched, not a configured/default 5; a configured width that
    disagrees with the case is refused."""
    from test_inflation import _frames, _table
    f, geo = _frames()
    geo = {**geo, "SPEC_BDY_WIDTH": 9}
    out, cnt = inf.inflate_state(f, geo, _table({("T", 0): 4.0, ("T", 1): 4.0, ("U", 0): 4.0, ("V", 0): 4.0}))
    assert cnt["spec_bdy_width"] == 9 and cnt["spec_bdy_width_source"] == "case namelist"
    for n in ("T", "U", "V"):
        a, b = out[n], f[n]
        for sl in ((Ellipsis, slice(0, 9), slice(None)), (Ellipsis, slice(-9, None), slice(None)),
                   (Ellipsis, slice(None), slice(0, 9)), (Ellipsis, slice(None), slice(-9, None))):
            assert np.array_equal(a[sl], b[sl]), n
        assert not np.array_equal(a, b), n                           # the interior is inflated
    with pytest.raises(ValueError, match="contradicts the case"):
        inf.inflate_state(f, geo, _table({("T", 0): 4.0}), spec_bdy_width=5)
    sec = inf.validate_config_section({"method": "adaptive"})
    assert sec["adaptive"]["spec_bdy_width"] is None                 # default: follow the case
