"""``gpuwrf da check|analyse`` end to end on the shipped Switzerland d01 case (no ALISIOS data).

Real WRF file structure (examples/switzerland_d01 wrfinput: hybrid coordinate, base state, Noah-MP fields) with a
synthetic 8-member ensemble (smooth T/QV/U/V perturbations of the real state) and synthetic observations taken from
a different perturbation ("truth") at grid points: CSV surface stations at the model terrain height and one CSV
profile. Checks: CLI exit codes, receipts, O-A < O-B, withheld semantics, zero-increment identity in the
specified zone, hydrostatic rebalance residual, stored positivity, provenance (every opened file planned), and
that the written analysis is admitted by the port's own case preflight (``gpuwrf run --dry-run``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import netCDF4
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "switzerland_d01"
STAMP = "2023-01-15_00:00:00"
N_MEMBERS = 8
STATIONS = [(14, 14), (14, 27), (20, 20), (26, 15), (27, 27), (18, 24)]
PROFILE = (21, 19)


def _smooth(a, passes=3):
    for _ in range(passes):
        a = (a + np.roll(a, 1, -1) + np.roll(a, -1, -1) + np.roll(a, 1, -2) + np.roll(a, -1, -2)) / 5.0
    return a


def _perturb(path, seed, amp=1.0):
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "r+") as ds:
        ds.set_auto_mask(False)
        t = ds["T"][0]
        dt = amp * 0.8 * _smooth(rng.normal(size=t.shape)) / 0.35
        ds["T"][0] = t + dt
        ds["T2"][0] = ds["T2"][0] + dt[0]
        q = ds["QVAPOR"][0]
        fq = np.exp(amp * 0.1 * _smooth(rng.normal(size=q.shape)) / 0.35)
        ds["QVAPOR"][0] = q * fq
        ds["Q2"][0] = ds["Q2"][0] * fq[0]
        for name, sfc in (("U", "U10"), ("V", "V10")):
            v = ds[name][0]
            dv = amp * 1.0 * _smooth(rng.normal(size=v.shape)) / 0.35
            ds[name][0] = v + dv
            d0 = 0.5 * (dv[0][:, 1:] + dv[0][:, :-1]) if name == "U" else 0.5 * (dv[0][1:] + dv[0][:-1])
            ds[sfc][0] = ds[sfc][0] + d0


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if not (CASE / "wrfinput_d01").is_file():
        pytest.skip("examples/switzerland_d01 absent")
    root = tmp_path_factory.mktemp("da_swiss")
    shutil.copy(CASE / "wrfinput_d01", root / "template_wrfinput_d01")
    ctrl = root / "ctrl"
    ctrl.mkdir()
    shutil.copy(CASE / "wrfinput_d01", ctrl / f"wrfout_d01_{STAMP}")
    shutil.copy(CASE / "namelist.input", ctrl / "namelist.input")
    for m in range(N_MEMBERS):
        d = root / "members" / f"m{m:02d}"
        d.mkdir(parents=True)
        shutil.copy(CASE / "wrfinput_d01", d / f"wrfout_d01_{STAMP}")
        _perturb(d / f"wrfout_d01_{STAMP}", seed=100 + m)
    truth = root / "truth.nc"
    shutil.copy(CASE / "wrfinput_d01", truth)
    _perturb(truth, seed=999, amp=1.2)
    with netCDF4.Dataset(truth) as ds:
        ds.set_auto_mask(False)
        f = {n: np.asarray(ds[n][0], np.float64) for n in ("T2", "Q2", "U10", "V10", "HGT", "XLAT", "XLONG",
                                                         "COSALPHA", "SINALPHA", "T", "P", "PB")}
    rows = ["station_id,lat,lon,elevation_m,time,t2_k,q2_kgkg,u10_ms,v10_ms"]
    for n, (j, i) in enumerate(STATIONS):
        ue = f["U10"][j, i] * f["COSALPHA"][j, i] - f["V10"][j, i] * f["SINALPHA"][j, i]
        ve = f["V10"][j, i] * f["COSALPHA"][j, i] + f["U10"][j, i] * f["SINALPHA"][j, i]
        rows.append(f"S{n},{f['XLAT'][j, i]:.6f},{f['XLONG'][j, i]:.6f},{f['HGT'][j, i]:.3f},2023-01-15T00:00:00Z,"
                    f"{f['T2'][j, i]:.4f},{f['Q2'][j, i]:.7f},{ue:.4f},{ve:.4f}")
    (root / "surface.csv").write_text("\n".join(rows) + "\n")
    j, i = PROFILE
    prof = ["station_id,lat,lon,time,p_pa,t_k"]
    for k in (3, 8, 14, 20, 26):
        p = f["P"][k, j, i] + f["PB"][k, j, i]
        t = (f["T"][k, j, i] + 300.0) * (p / 1.0e5) ** (287.0 / 1004.5)
        prof.append(f"PRF,{f['XLAT'][j, i]:.6f},{f['XLONG'][j, i]:.6f},2023-01-15T00:00:00Z,{p:.2f},{t:.4f}")
    (root / "profile.csv").write_text("\n".join(prof) + "\n")
    config = {
        "schema": "gpuwrf.da.v1", "description": "Swiss d01 synthetic-member smoke", "t_a": "2023-01-15T00:00:00Z",
        "template_wrfinput": "template_wrfinput_d01", "control_file": f"ctrl/wrfout_d01_{STAMP}",
        "ensemble": {"kind": "members", "member_glob": f"members/m*/wrfout_d01_{STAMP}"},
        "obs": {"sources": [{"format": "csv_surface", "path": "surface.csv"},
                            {"format": "csv_profile", "path": "profile.csv"}]},
        "params": {"loc_h_km": 40.0, "workers": 2, "rtps_alpha": 0.5},
        "experiments": [
            {"name": "all", "assimilate": ["surface", "sonde"], "write_mean": True, "land": "cold"},
            {"name": "surface_only", "assimilate": ["surface"], "withhold": ["sonde"]},
        ],
    }
    (root / "da.json").write_text(json.dumps(config, indent=1))
    return root


def _main(argv):
    from gpuwrf.cli import main
    return main(argv)


@pytest.fixture(scope="module")
def analysed(case):
    rc = _main(["da", "analyse", "--config", str(case / "da.json"), "--out", str(case / "out")])
    return rc, json.loads((case / "out" / "summary.json").read_text())


def test_check_lists_inputs_without_reading(case, capsys):
    assert _main(["da", "check", "--config", str(case / "da.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["planned_inputs"] == {"control": 1, "namelist": 1, "template": 1, "member": N_MEMBERS, "obs": 2}
    assert report["governance"] == "none" and report["experiments"] == ["all", "surface_only"]


def test_analyse_receipts_and_fit(case, analysed):
    rc, summary = analysed
    assert rc == 0
    # 6 stations x (T2, Q2, U10, V10) + 5 profile levels kept as thinned inputs + their 5 layer super-obs
    assert summary["schema"] == "gpuwrf.da.analysis_summary.v1" and summary["n_obs"] == 6 * 4 + 2 * 5
    assert summary["obs"]["sonde_superob"] == {"n_before": 29, "n_after": 34}
    receipt = json.loads((case / "out" / "receipt_all.json").read_text())
    assert receipt["admissible"] and receipt["assimilated"] == 29 and receipt["n_members"] == N_MEMBERS
    for key in ("surface:11", "surface:13", "sonde:1"):
        st = receipt["stats_assimilated"][key]
        assert st["oma_rms"] < st["omb_rms"], key
    withheld = json.loads((case / "out" / "receipt_surface_only.json").read_text())
    assert withheld["assimilated"] == 24 and withheld["withheld_ok"] == 5
    assert "sonde:1" in withheld["stats_withheld"] and "sonde:1" not in withheld["stats_assimilated"]
    assert receipt["moisture"]["admissible"] and receipt["counters"]["rebalance"] == "applied"


def test_written_mean_is_balanced_positive_and_untouched_in_the_specified_zone(case, analysed):
    from gpuwrf.assimilation.letkf import state_io

    out = case / "out" / "wrfinput_d01"
    assert out.is_file()
    ctrl = state_io.read_frame(case / "ctrl" / f"wrfout_d01_{STAMP}", STAMP,
                               ("U", "V", "T", "QVAPOR", "P", "PH", "MU", "P_HYD"))
    an = state_io.read_frame(out, STAMP, ("U", "V", "T", "QVAPOR", "P", "PH", "MU", "P_HYD"))
    for name in ("T", "QVAPOR", "U", "V"):
        assert np.array_equal(an[name][:, :5, :], ctrl[name][:, :5, :]), name           # spec zone south
        assert np.array_equal(an[name][:, :, -5:], ctrl[name][:, :, -5:]), name         # spec zone east
        assert not np.array_equal(an[name], ctrl[name]), name
    assert np.array_equal(an["MU"], ctrl["MU"]) and an["QVAPOR"].min() >= 0
    geo = state_io.read_geometry(case / "ctrl" / f"wrfout_d01_{STAMP}", base_wrfinput=case / "template_wrfinput_d01")
    report = state_io.rebalance_report(ctrl, {**ctrl, **an}, an, geo)
    assert report["residual_change_rel_max"] < 1e-5 and report["dPH_surface_max_abs"] == 0.0
    with netCDF4.Dataset(out) as ds:
        record = json.loads(ds.getncattr(state_io.PROVENANCE_ATTRIBUTE))
        times = b"".join(ds["Times"][0]).decode()
    assert times == STAMP and record["extra"]["config_sha256"] == analysed[1]["config_sha256"]
    assert "GPUWRF_LAND_CYCLE" not in netCDF4.Dataset(out).ncattrs()                   # land: cold


def test_every_opened_input_was_planned(case, analysed):
    gov = json.loads((case / "out" / "governance" / "governance.json").read_text())
    assert gov["unplanned_opens"] == []
    assert len(gov["inputs"]) == 3 + N_MEMBERS + 2
    assert all(len(row["sha256"]) == 64 for row in gov["inputs"].values())


def test_written_analysis_is_admitted_by_the_port_case_preflight(case, analysed, monkeypatch):
    from gpuwrf.assimilation.letkf.cycle_case import write_cycle_case

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    result = write_cycle_case(CASE, {"d01": case / "out" / "wrfinput_d01"}, STAMP, 6, case / "cycle_case", land="cold")
    admission = result["admission"]
    assert admission["cli_dry_run"]["status"] == "PASS" and admission["ensemble_case"]["status"] == "PASS"


def test_exit_codes(case, capsys):
    assert _main(["da", "analyse", "--config", str(case / "da.json"), "--out", str(case / "out")]) == 2   # exists
    bad = case / "bad.json"
    bad.write_text(json.dumps({"schema": "gpuwrf.da.v1"}))
    assert _main(["da", "check", "--config", str(bad)]) == 2
    assert "missing required key" in capsys.readouterr().err
    from gpuwrf.assimilation.letkf.cli import _refused_outputs
    assert _refused_outputs({"experiments": {"A": {"analysis_wrfinput": {"written": False}}}}) == ["A:analysis_wrfinput"]


def test_cli_is_backend_dark(case):
    # same argv as the console script `gpuwrf da check ...` (gpuwrf/__init__ inspects sys.argv[1] at import)
    code = ("import sys, json; sys.argv = ['gpuwrf', 'da', 'check', '--config', sys.argv[1]]; "
            "from gpuwrf.cli import main; rc = main(sys.argv[1:]); "
            "print(json.dumps({'rc': rc, 'jax': 'jax' in sys.modules}))")
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS",)}
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run([sys.executable, "-c", code, str(case / "da.json")], capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=120)
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last == {"rc": 0, "jax": False}, proc.stderr


@pytest.mark.parametrize("sub,offline", [("check", True), ("analyse", True), ("cycle", False), ("members", False)])
def test_only_host_only_da_subcommands_skip_the_package_bootstrap(sub, offline):
    # da cycle/members run `gpuwrf run` in-process: they must get x64/command-buffer/cache setup like `gpuwrf run`
    code = f"import sys; sys.argv = ['gpuwrf', 'da', {sub!r}]; import gpuwrf; print(gpuwrf._OFFLINE_COMMAND)"
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT),
                          timeout=300)
    assert proc.stdout.strip().splitlines()[-1] == str(offline), proc.stderr


def test_calibration_sees_only_the_experiments_assimilated_observations(case, monkeypatch):
    """Withheld (verification) observations must not tune the prior inflation of the analysis they verify."""
    from gpuwrf.assimilation.letkf import api

    seen = []
    original = api.calibrate

    def spy(frames, geo, obs, cfg):
        seen.append(sorted(set(np.asarray(obs.kind).tolist())))
        return original(frames, geo, obs, cfg)

    monkeypatch.setattr(api, "calibrate", spy)
    summary = api.analyse_case(json.loads((case / "da.json").read_text()) | {"template_wrfinput": str(
        case / "template_wrfinput_d01"), "control_file": str(case / "ctrl" / f"wrfout_d01_{STAMP}"),
        "ensemble": {"kind": "members", "member_glob": str(case / "members" / "m*" / f"wrfout_d01_{STAMP}")},
        "obs": {"sources": [{"format": "csv_surface", "path": str(case / "surface.csv")},
                            {"format": "csv_profile", "path": str(case / "profile.csv")}]},
        "experiments": [{"name": "surface_only", "assimilate": ["surface"], "withhold": ["sonde"]},
                        {"name": "sonde_only", "assimilate": ["sonde"], "withhold": ["surface"]}]},
        case / "out_calibration")
    assert seen == [[2], [1]]
    receipt = json.loads((case / "out_calibration" / "receipt_surface_only.json").read_text())
    assert receipt["calibration"]["calibration_obs"] == 24 and summary["experiments"]["sonde_only"]["withheld_ok"] == 24


def test_inflation_plugin_bias_corrected_obs_and_state_files_are_used(case, monkeypatch):
    """The ensemble component's calibration may return bias-corrected assimilated obs; they must be the ones
    assimilated, and its state files must be in the governance read plan."""
    from gpuwrf.assimilation.letkf import api, inflation

    state = case / "adaptive_state.json"
    state.write_text("{}")
    monkeypatch.setattr(inflation, "validate_config_section", lambda section: dict(section), raising=False)
    monkeypatch.setattr(inflation, "read_plan", lambda section: [section["state_in"]], raising=False)

    def calibrate(frames, geo, obs, ens, section):
        shifted = obs.replace(value=np.asarray(obs.value) + 0.5)          # a fake +0.5 bias correction
        return frames, {"inflation_rho": 1.0, "obs": shifted}
    monkeypatch.setattr(inflation, "calibrate", calibrate, raising=False)
    cfg = json.loads((case / "da.json").read_text())
    cfg.update(template_wrfinput=str(case / "template_wrfinput_d01"),
               control_file=str(case / "ctrl" / f"wrfout_d01_{STAMP}"),
               ensemble={"kind": "members", "member_glob": str(case / "members" / "m*" / f"wrfout_d01_{STAMP}")},
               obs={"sources": [{"format": "csv_surface", "path": str(case / "surface.csv")}]},
               inflation={"state_in": str(state)},
               experiments=[{"name": "s", "assimilate": ["surface"]}])
    api.analyse_case(cfg, case / "out_plugin")
    receipt = json.loads((case / "out_plugin" / "receipt_s.json").read_text())
    base = json.loads((case / "out" / "receipt_surface_only.json").read_text())
    for key in ("surface:11", "surface:13"):
        assert receipt["stats_assimilated"][key]["omb_mean"] == pytest.approx(
            base["stats_assimilated"][key]["omb_mean"] + 0.5, abs=1e-9)
    assert receipt["calibration"]["obs_corrected"] is True
    plan = json.loads((case / "out_plugin" / "read_plan.json").read_text())
    assert [e["path"] for e in plan["entries"] if e["role"] == "inflation_state"] == [str(state.resolve())]


# --- critic cr-w41-da-core items 2-4 on the shipped case ------------------------------------------------------------

def _copy_case(case, dest, *, width=None, experiments=None, params=None):
    shutil.copytree(case, dest, ignore=shutil.ignore_patterns("out*", "cycle_case*", "bad.json", "pytest*"))
    if width is not None:
        nml = dest / "ctrl" / "namelist.input"
        nml.write_text(re.sub(r"spec_bdy_width\s*=\s*5,", f"spec_bdy_width = {width},", nml.read_text()))
    cfg = json.loads((dest / "da.json").read_text())
    if experiments is not None:
        cfg["experiments"] = experiments
    if params:
        cfg["params"] = {**cfg["params"], **params}
    (dest / "da.json").write_text(json.dumps(cfg))
    return dest


def test_case_boundary_width_binds_analysis_mean_and_member_writer(case, tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import api, member_cycle
    from gpuwrf.assimilation.letkf import config as dacfg
    from gpuwrf.assimilation.letkf.member_cycle import spec_zone_changes

    seen = {}

    def fake_member_cycle(**kw):
        seen.update(kw)
        path = tmp_path / "manifest.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(member_cycle, "write_member_cycle", fake_member_cycle)
    root = _copy_case(case, tmp_path / "w9", width=9, experiments=[
        {"name": "all", "assimilate": ["surface", "sonde"], "write_mean": True, "land": "cold",
         "write_members": {"source_case_dir": str(CASE), "hours": 6}}])
    summary = api.analyse_case(dacfg.load(root / "da.json"), root / "out")
    receipt = json.loads((root / "out" / "receipt_all.json").read_text())
    assert receipt["admissible"] and receipt["spec_bdy_width"]["value"] == 9
    assert seen["spec_bdy_width"] == 9                                 # the member writer validates the same zone
    with netCDF4.Dataset(root / "ctrl" / f"wrfout_d01_{STAMP}") as c, netCDF4.Dataset(root / "out" / "wrfinput_d01") as m:
        control = {n: np.asarray(c[n][0]) for n in ("T", "U", "V", "QVAPOR")}
        mean = {n: np.asarray(m[n][0]) for n in control}
    assert spec_zone_changes(control, mean, 9) == {"T": 0, "U": 0, "V": 0, "QVAPOR": 0}
    assert spec_zone_changes(control, mean, 5) == {"T": 0, "U": 0, "V": 0, "QVAPOR": 0}
    assert not np.array_equal(mean["T"], control["T"])
    assert summary["experiments"]["all"]["admissible"]
    bad = _copy_case(case, tmp_path / "w9x", width=9, params={"spec_bdy_width": 5})
    with pytest.raises(dacfg.ConfigError, match="contradicts the case's spec_bdy_width = 9"):
        api.analyse_case(dacfg.load(bad / "da.json"), bad / "out")


def test_role_withheld_rows_never_calibrate_or_update(case, tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import api, readers
    from gpuwrf.assimilation.letkf import config as dacfg

    original = readers.read_sources

    def marked(*args, **kwargs):
        obs, info = original(*args, **kwargs)
        first = np.flatnonzero(np.asarray(obs.kind) == 2)[0]            # one surface row is verification-only
        return obs.with_role(np.arange(len(obs.kind)) == first, "withhold"), info

    monkeypatch.setattr(readers, "read_sources", marked)
    root = _copy_case(case, tmp_path / "roles", experiments=[{"name": "sfc", "assimilate": ["surface"]}])
    api.analyse_case(dacfg.load(root / "da.json"), root / "out")
    receipt = json.loads((root / "out" / "receipt_sfc.json").read_text())
    n_surface = len(STATIONS) * 4                                      # T2, Q2, U10, V10 per station
    assert receipt["calibration"]["calibration_obs"] == n_surface - 1
    assert receipt["qc"]["role_withheld"] == 1 and receipt["withheld_ok"] >= 1
    assert receipt["assimilated"] <= n_surface - 1


def test_cloud_assimilation_is_refused_and_cloud_withholding_allowed(case):
    from gpuwrf.assimilation.letkf import config as dacfg

    raw = json.loads((case / "da.json").read_text())
    with pytest.raises(dacfg.ConfigError, match="cloud observations are not assimilated"):
        dacfg.validate({**raw, "experiments": [{"name": "c", "assimilate": ["cloud"]}]}, base_dir=case)
    ok = dacfg.validate({**raw, "experiments": [{"name": "c", "assimilate": ["surface"], "withhold": ["cloud"]}]},
                        base_dir=case)
    assert ok.data["experiments"][0]["withhold"] == ["cloud"]


def test_mean_writer_refuses_a_change_in_the_boundary_zone(case, tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import api, state_io
    from gpuwrf.assimilation.letkf import config as dacfg

    original = state_io.analysis_mean_fields

    def leaky(*args, **kwargs):
        fields = original(*args, **kwargs)
        fields["T"] = np.array(fields["T"])
        fields["T"][:, 0, :] += 0.1                                    # a mean that touches the south boundary row
        return fields

    monkeypatch.setattr(state_io, "analysis_mean_fields", leaky)
    root = _copy_case(case, tmp_path / "zone", experiments=[
        {"name": "m", "assimilate": ["surface", "sonde"], "write_mean": True, "land": "cold"}])
    summary = api.analyse_case(dacfg.load(root / "da.json"), root / "out")
    written = summary["experiments"]["m"]["analysis_wrfinput"]
    assert written["written"] is False and "specified/relaxation zone" in written["refusal_reasons"][0]
    assert not (root / "out" / "wrfinput_d01").exists()


def test_member_writer_uses_the_case_width_and_refuses_a_contradiction(case, tmp_path):
    from gpuwrf.assimilation.letkf import state_io
    from gpuwrf.assimilation.letkf.member_cycle import ANALYSED, REBALANCED, write_member_cycle

    root = _copy_case(case, tmp_path / "mw9", width=9)
    frames = [root / "ctrl" / f"wrfout_d01_{STAMP}", root / "members" / "m00" / f"wrfout_d01_{STAMP}"]
    names = ANALYSED + REBALANCED
    vals = [state_io.read_frame(f, STAMP, names) for f in frames]
    fields = {n: np.stack([v[n] for v in vals]) for n in names}
    fields["T"][0, :, 6, fields["T"].shape[-1] // 2] += np.float32(0.5)   # row 6: inside width 9, outside width 5
    with pytest.raises(ValueError, match="specified/relaxation zone"):
        write_member_cycle(frames, fields, STAMP, root / "template_wrfinput_d01", CASE, 6, tmp_path / "cyc")
    with pytest.raises(ValueError, match="contradicts the case's spec_bdy_width = 9"):
        write_member_cycle(frames, fields, STAMP, root / "template_wrfinput_d01", CASE, 6, tmp_path / "cyc5",
                           spec_bdy_width=5)
    assert not (tmp_path / "cyc" / "ENSEMBLE_MANIFEST.json").exists()
