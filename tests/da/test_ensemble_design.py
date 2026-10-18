"""Ensemble-design options through the public ``gpuwrf da analyse`` path (lane w41-da-ens).

Reuses da-core's Swiss d01 synthetic-member case (``test_api_cli.case``: real wrfinput structure, 8 perturbed
members, CSV surface + profile obs). Two chained cycles with the ``inflation`` section (adaptive debiased var×band
inflation + obs-bias correction + RTPS 0.3 + seeded additive inflation from a pool npz) and the ``localisation``
section from ``varloc.as_var_loc('surface_split')``: config validation, governance planning of the state/pool
files, the hooks reaching ``analyse`` (receipt), state persistence through ``inflation.chain_section``, the
admissible written mean.
"""
from __future__ import annotations

import json

import numpy as np

from gpuwrf.assimilation.letkf import config as dacfg, inflation, spread, state_io, varloc
from test_api_cli import STAMP, _main, case  # noqa: F401  (module fixture re-used)

NAMES = ("U", "V", "T", "QVAPOR", "T2", "Q2", "U10", "V10", "PSFC")


def _pool(case):
    files = sorted((case / "members").glob(f"m*/wrfout_d01_{STAMP}"))
    states = {n: np.stack([state_io.read_frame(f, STAMP, (n,))[n] for f in files]) for n in ("U", "V", "T", "QVAPOR")}
    path = case / "pool.npz"
    if not path.exists():
        spread.write_pool_npz(path, spread.pool_from_states(states), {"source": "synthetic members"})
    return path


def _config(case, cycle_dir, previous):
    cfg = json.loads((case / "da.json").read_text())
    section = inflation.validate_config_section({
        "method": "adaptive", "obs_bias": {"apply_kinds": ["surface"]},
        "maintenance": {"method": "rtps", "alpha": 0.3},
        "additive": {"pool_npz": str(_pool(case)), "amplitude": 0.1, "seed": 7}})
    cycle_dir.mkdir(exist_ok=True)
    cfg["inflation"] = inflation.chain_section(section, previous, cycle_dir)
    cfg["localisation"] = {"variables": varloc.as_var_loc("surface_split", NAMES)}
    cfg["experiments"] = [{"name": "all", "assimilate": ["surface", "sonde"], "write_mean": True, "land": "cold"}]
    path = case / f"da_{cycle_dir.name}.json"        # relative paths in da.json resolve against its directory
    path.write_text(json.dumps(cfg, indent=1))
    return path


def test_cli_ensemble_design_two_chained_cycles(case):
    c1, c2 = case / "ens_c1", case / "ens_c2"
    assert _main(["da", "analyse", "--config", str(_config(case, c1, None)), "--out", str(c1 / "out")]) == 0
    r1 = json.loads((c1 / "out" / "receipt_all.json").read_text())
    assert r1["admissible"] and r1["analysis_wrfinput"]["written"]
    assert r1["hooks"]["maintenance"].startswith("w41:rtps(alpha=0.3") and "+additive(seed=7)" in r1["hooks"]["maintenance"]
    assert sorted(map(sorted, r1["hooks"]["var_loc_groups"])) == [["PSFC", "Q2", "QVAPOR", "T", "T2"],
                                                                  ["U", "U10", "V", "V10"]]
    cal = r1["calibration"]
    assert cal["method"].startswith("adaptive") and cal["inflation_rho"] == 1.0 and cal["obs_corrected"] is True
    assert cal["state_inflation_counters"]["rebalance"].startswith("applied")   # inflation reaches member P/PH (P1)
    state1 = json.loads((c1 / "adaptive_state.json").read_text())
    assert state1["cycles"] == 1 and (c1 / "obs_bias.json").is_file()
    plan = json.loads((c1 / "out" / "read_plan.json").read_text())
    planned = {e["path"] for e in plan["entries"] if e["role"] == "inflation_state"}
    assert str((case / "pool.npz").resolve()) in planned
    # cycle 2: the persisted estimate is the prior
    assert _main(["da", "analyse", "--config", str(_config(case, c2, c1)), "--out", str(c2 / "out")]) == 0
    r2 = json.loads((c2 / "out" / "receipt_all.json").read_text())
    est = r2["calibration"]["adaptive"]["estimates"]
    for key, value in state1["rho"].items():
        assert abs(est[key]["rho_prior"] - value) < 1e-12
    assert json.loads((c2 / "adaptive_state.json").read_text())["cycles"] == 2
    planned2 = {e["path"] for e in json.loads((c2 / "out" / "read_plan.json").read_text())["entries"]
                if e["role"] == "inflation_state"}
    assert str((c1 / "adaptive_state.json").resolve()) in planned2 and str((c1 / "obs_bias.json").resolve()) in planned2


def test_inflated_prior_and_written_members_stay_hydrostatically_consistent(case):
    """Critic cr-w41-da-ens P1 through the product calibration + analysis: the adaptive state inflation (rho ~ 4
    forced by a confident prior) is carried into each member's P/PH (zero-additional-update case: inflated prior ==
    raw + rebalance_increment(raw, inflated) exactly), and the analysed member fields equal the one-step rebalance
    from the RAW member to round-off (before the fix the inflation increment was missing from written P/PH)."""
    from gpuwrf.assimilation.letkf import analysis as lan, api, readers
    cfg, _, plan = api.prepare(dacfg.load(case / "da.json"))
    frames, geo, _ = api.load_ensemble(cfg, plan)
    obs, _ = readers.read_sources(cfg.data["obs"], cfg.t_a, geo, frames)
    sec = inflation.validate_config_section({"method": "adaptive", "adaptive": {"rho0": 4.0, "var0": 1e-6}})
    prior, info = inflation.calibrate(frames, geo, obs, {"kind": "members", "scale": 1.0}, sec)
    assert info["state_inflation_counters"]["rebalance"].startswith("applied")
    assert min(v["rho"] for v in info["adaptive"]["table"].values()) > 3.0     # inflation really applied
    params = api._params(dict(cfg.data["params"]), {})
    analysis, receipt = lan.analyse(prior, geo, obs, params)
    assert receipt["admissible"], receipt["refusal_reasons"]
    worst_two_step = 0.0
    for m in range(frames["T"].shape[0]):
        raw = {k: v[m] for k, v in frames.items()}
        infl = {k: v[m] for k, v in prior.items()}
        full = state_io.rebalance_increment(raw, infl, geo)
        for k in ("P", "PH"):
            np.testing.assert_array_equal(infl[k], full[k])
        an = {k: v[m] for k, v in analysis.items()}
        one_step = state_io.rebalance_increment(raw, an, geo)
        scale = float(np.abs(np.asarray(one_step["PH"], np.float64) - np.asarray(raw["PH"], np.float64)).max())
        worst_two_step = max(worst_two_step, float(np.abs(np.asarray(an["PH"], np.float64)
                                                          - np.asarray(one_step["PH"], np.float64)).max()) / scale)
    assert worst_two_step < 1e-3


def test_public_config_resolves_inflation_paths_against_the_config_directory(tmp_path, monkeypatch):
    """Critic cr-w41-da-ens delta: the public config.validate(base_dir=...) / config.load must forward the config
    directory into the inflation section like member_files (state_in/out, obs_bias state, additive pool), also
    when launched from another working directory; absolute paths stay as given."""
    from test_config import minimal
    raw = minimal(inflation={
        "method": "adaptive", "adaptive": {"state_in": "c1/adaptive_state.json", "state_out": "c2/adaptive_state.json"},
        "obs_bias": {"state_in": "c1/obs_bias.json", "state_out": "/abs/c2/obs_bias.json"},
        "additive": {"pool_npz": "pools/p.npz", "amplitude": 0.1, "seed": 1}})
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "da.json").write_text(json.dumps(raw))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    got = {}
    for how, call in (("validate", lambda: dacfg.validate(raw, base_dir=cfg_dir)),
                      ("load", lambda: dacfg.load(cfg_dir / "da.json"))):
        try:
            got[how] = call().data["inflation"]
        except dacfg.ConfigError as exc:          # turn a refusal into a value mismatch (genuine assertion)
            got[how] = {"refused": str(exc)}
    want = {"adaptive": (str(cfg_dir / "c1/adaptive_state.json"), str(cfg_dir / "c2/adaptive_state.json")),
            "obs_bias": (str(cfg_dir / "c1/obs_bias.json"), "/abs/c2/obs_bias.json")}
    for how, sec in got.items():
        for key, (s_in, s_out) in want.items():
            assert (sec.get(key, {}).get("state_in"), sec.get(key, {}).get("state_out")) == (s_in, s_out), (how, sec)
        assert sec.get("additive", {}).get("pool_npz") == str(cfg_dir / "pools/p.npz"), (how, sec)
    planned = set(inflation.read_plan(got["load"]))
    assert {str(cfg_dir / "c1/adaptive_state.json"), str(cfg_dir / "c1/obs_bias.json"),
            str(cfg_dir / "pools/p.npz")} <= planned
    assert all(not p.startswith(str(elsewhere)) for p in planned)
