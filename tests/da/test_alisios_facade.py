"""The ``alisios`` plug-in on ALISIOS's own reader facade vs its ``guarded_session`` fallback (manager 10:28Z).

The facade is imported from a copy sealed out of the ALISIOS git object store at ``FACADE_COMMIT`` (never the working
tree). The guard is faked in BOTH backends with the same verdicts; the data root is a temporary ALISIOS-like tree
built from the shipped Switzerland d01 case (no ALISIOS data is touched). Parity: same admitted read set, same
analysis bit for bit, DENY refused before any open, every ALISIOS open refused during the staged analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import netCDF4
import numpy as np
import pytest

from gpuwrf.assimilation.letkf import alisios_guard as ag
from gpuwrf.assimilation.letkf import api, governance as gov
from gpuwrf.assimilation.letkf import config as dacfg
from test_api_cli import CASE, STAMP, STATIONS, _perturb

_REAL_FACADE = ag.facade            # captured before the conftest opt-out patches it
N = 4
ISSUE = "20230114"


def _facade_available():
    try:
        subprocess.run(["git", "-C", ag.ALISIOS_REPO, "cat-file", "-e", f"{ag.FACADE_COMMIT}^{{commit}}"],
                       check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ag.FACADE_MODULES)


pytestmark = pytest.mark.skipif(not _facade_available(), reason="ALISIOS repo at FACADE_COMMIT or facade modules absent")


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealed")
    reader, guard, info = _REAL_FACADE(root)
    assert reader is not None, info
    return root, reader, guard, info


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    if not (CASE / "wrfinput_d01").is_file():
        pytest.skip("examples/switzerland_d01 absent")
    base = tmp_path_factory.mktemp("facade_case")
    root = base / "alisios"
    ctrl = root / f"wg_{ISSUE}_18z_a1" / "run" / "run"
    ctrl.mkdir(parents=True)
    shutil.copy(CASE / "wrfinput_d01", ctrl / f"wrfout_d01_{STAMP}")
    shutil.copy(CASE / "wrfinput_d01", ctrl / "wrfinput_d01")
    shutil.copy(CASE / "namelist.input", ctrl / "namelist.input")
    members = []
    for m in range(N):
        d = root / f"wg_{ISSUE}_18z_a{m + 2}" / "run" / "run"
        d.mkdir(parents=True)
        shutil.copy(CASE / "namelist.input", d / "namelist.input")     # run window = admitted coverage metadata
        shutil.copy(CASE / "wrfinput_d01", d / f"wrfout_d01_{STAMP}")
        _perturb(d / f"wrfout_d01_{STAMP}", seed=300 + m)
        members.append(str(d / f"wrfout_d01_{STAMP}"))
    truth = base / "truth.nc"
    shutil.copy(CASE / "wrfinput_d01", truth)
    _perturb(truth, seed=777, amp=1.2)
    with netCDF4.Dataset(truth) as ds:
        f = {n: np.asarray(ds[n][0], np.float64) for n in ("T2", "Q2", "HGT", "XLAT", "XLONG")}
    rows = ["station_id,lat,lon,elevation_m,time,t2_k,q2_kgkg"]
    rows += [f"S{n},{f['XLAT'][j, i]:.6f},{f['XLONG'][j, i]:.6f},{f['HGT'][j, i]:.3f},2023-01-15T00:00:00Z,"
             f"{f['T2'][j, i]:.4f},{f['Q2'][j, i]:.7f}" for n, (j, i) in enumerate(STATIONS)]
    (base / "surface.csv").write_text("\n".join(rows) + "\n")
    raw = {"schema": "gpuwrf.da.v1", "t_a": "2023-01-15T00:00:00Z",
           "template_wrfinput": str(ctrl / "wrfinput_d01"), "control_file": str(ctrl / f"wrfout_d01_{STAMP}"),
           "ensemble": {"kind": "members", "member_files": members},
           "obs": {"sources": [{"format": "csv_surface", "path": str(base / "surface.csv")}]},
           "params": {"loc_h_km": 40.0, "workers": 2, "rtps_alpha": 0.5},
           "experiments": [{"name": "sfc", "assimilate": ["surface"], "write_mean": True, "land": "cold"}]}
    hashes = {str(f.resolve()): ag._sha256(f) for f in root.rglob("*") if f.is_file()}  # before any guard is active
    return base, root, raw, hashes


def _fake(verdicts):
    calls = {"scan": [], "admit": []}

    def scan(days, mode=None, **kw):
        calls["scan"].append(sorted(days))
        return {d: verdicts.get(d, "ALLOW") for d in days}

    def admit(issue_days, valid_days, mode=None, receipt_out=None, **kw):
        calls["admit"].append((sorted(issue_days), sorted(valid_days)))
        if receipt_out:
            Path(receipt_out).write_text("{}")
        return {"result": "ALLOW_CALIBRATION_READ"}

    return scan, admit, calls


@pytest.fixture
def guards(tree, sealed, monkeypatch):
    """Same fake verdicts for the working-tree guard_call (fallback, refine scan) and the sealed one (facade)."""
    root = tree[1]
    _, _, sealed_guard, _ = sealed
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    monkeypatch.setattr(ag, "facade", lambda sealed_root=None: sealed[1:])

    def install(verdicts):
        scan, admit, calls = _fake(verdicts)
        fake = types.SimpleNamespace(scan=scan, admit=admit, GuardRefused=sealed_guard.GuardRefused)
        monkeypatch.setattr(ag, "_guard_call", lambda: fake)
        monkeypatch.setattr(sealed_guard, "scan", scan)
        monkeypatch.setattr(sealed_guard, "admit", admit)
        return calls

    return install


def _cfg(tree, facade):
    base, _, raw, _ = tree
    return dacfg.validate({**raw, "governance": {"plugin": "alisios", "options": {"facade": facade}}}, base_dir=base)


def _fields(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {n: np.asarray(v[:]) for n, v in ds.variables.items()}, json.loads(ds.getncattr("GPUWRF_LETKF_ANALYSIS"))


@pytest.mark.alisios_facade
def test_facade_and_fallback_admit_the_same_set_and_give_the_same_analysis(tree, sealed, guards, tmp_path):
    calls_off = guards({})
    off = api.analyse_case(_cfg(tree, "off"), tmp_path / "off")
    calls_on = guards({})
    try:
        on = api.analyse_case(_cfg(tree, "require"), tmp_path / "on")
    except PermissionError as exc:      # the staged analysis must never reach an original ALISIOS file
        raise AssertionError(f"facade-backend analysis opened an original ALISIOS file: {exc}") from exc

    gov_off, gov_on = off["governance"], on["governance"]
    assert gov_off["backend"] == {"kind": "guarded_session", "fallback_reason": "governance.options.facade = 'off'"}
    assert gov_on["backend"]["kind"] == "guarded_read" and gov_on["backend"]["facade"]["commit"] == ag.FACADE_COMMIT
    # the same read set is admitted: fallback in one admit, facade one admit per whole file
    files = gov_on["backend"]["files"]
    assert len(files) == 3 + N
    assert len(json.loads(Path(gov_on["backend"]["guard_admit_receipts"]).read_text())) == len(files)
    # before that, each run's namelist was read once as admitted metadata (issue day) for the coverage
    assert sorted(a for a in calls_on["admit"] if a == ([ISSUE], [ISSUE])) == [([ISSUE], [ISSUE])] * (1 + N + 1)
    union = lambda pairs: (sorted({d for i, _ in pairs for d in i}), sorted({d for _, v in pairs for d in v}))  # noqa
    assert union(calls_on["admit"]) == union(calls_off["admit"]) == ([ISSUE], [ISSUE, "20230115"])
    # every staged byte went through the facade, which logged it; the stage is gone
    assert gov_on["backend"]["counters"]["authorized_attempts"] == len(files)
    assert gov_on["backend"]["stage_removed"] and not (tmp_path / "on" / "governance" / "alisios_stage").exists()
    events = [json.loads(line)["event"] for line in Path(gov_on["backend"]["read_spy"]).read_text().splitlines()]
    assert events[0] == "start" and events[-1] == "end" and events.count("authorized_attempt") == len(files)
    for original, row in files.items():
        assert row["sha256"] == tree[3][original]
    # the analysis phase opened nothing in the ALISIOS roots (all reads came from the stage)
    assert gov_on["alisios_guard"]["spy"]["observed"] == {} and gov_on["alisios_guard"]["spy"]["refused_paths"] == []
    assert gov_off["alisios_guard"]["spy"]["observed"]
    # identical analysis
    r_off = json.loads((tmp_path / "off" / "receipt_sfc.json").read_text())
    r_on = json.loads((tmp_path / "on" / "receipt_sfc.json").read_text())
    for key in ("admissible", "qc", "stats_assimilated", "moisture", "mean_moisture", "local_obs", "analysis_grid"):
        assert r_on[key] == r_off[key], key
    (f_off, rec_off), (f_on, rec_on) = _fields(tmp_path / "off" / "wrfinput_d01"), _fields(tmp_path / "on" / "wrfinput_d01")
    assert f_off.keys() == f_on.keys()
    for name in f_off:
        assert np.array_equal(f_off[name], f_on[name]), name
    # provenance names the ORIGINAL inputs in both (the facade run marks that the bytes came from its stage)
    for key in ("template", "template_sha256", "member_wrfout", "member_sha256"):
        assert rec_on[key] == rec_off[key], key
    assert rec_off["template"] == str(Path(_cfg(tree, "off").data["template_wrfinput"]).resolve())
    assert rec_on.get("inputs_staged") is True and "inputs_staged" not in rec_off
    for key in ("control_file", "geometry_file", "members"):
        assert on["ensemble_meta"][key] == off["ensemble_meta"][key], key
    assert "alisios_stage" not in json.dumps(on["ensemble_meta"]) + (tmp_path / "on" / "receipt_sfc.json").read_text()
    # the sealed import left any working-tree ``alisios`` package alone
    assert all(not str(getattr(m, "__file__", "")).startswith(str(sealed[0])) for k, m in sys.modules.items()
               if k == "alisios" or k.startswith("alisios."))


@pytest.mark.alisios_facade
@pytest.mark.parametrize("facade", ["off", "require"])
def test_a_deny_day_is_refused_before_any_open_in_both_backends(tree, guards, tmp_path, facade):
    # at planning time: refine refuses (shared by both backends), nothing is opened or staged
    calls = guards({"20230115": "DENY"})
    with pytest.raises(gov.PlanError, match="non-ALLOW"):
        api.analyse_case(_cfg(tree, facade), tmp_path / facade)
    assert all("20230115" not in admit[1] for admit in calls["admit"])
    assert not (tmp_path / facade / "wrfinput_d01").exists()
    assert not (tmp_path / facade / "governance" / "alisios_stage").exists()


@pytest.mark.alisios_facade
@pytest.mark.parametrize("facade", ["off", "require"])
def test_a_deny_day_appearing_after_planning_is_refused_by_the_session(tree, guards, tmp_path, facade):
    guards({})
    cfg, policy, plan = api.prepare(_cfg(tree, facade), receipt_dir=tmp_path / "gov")
    calls = guards({"20230115": "DENY"})                        # the guard changed between planning and reading
    with pytest.raises(PermissionError):
        with policy.session(plan, tmp_path / "gov"):
            raise AssertionError("the session must refuse before yielding")
    assert calls["admit"] == []
    if facade == "require":
        spy = [json.loads(line) for line in
               (tmp_path / "gov" / "alisios_guarded_read" / "READ_SPY.jsonl").read_text().splitlines()]
        assert [e["event"] for e in spy if e["event"] not in ("start", "scan")] == ["refused", "end"]
        assert spy[-2]["reason"] == "scan_DENY_file_discarded"
        assert spy[-1]["counters"] == {"authorized_attempts": 0, "denied_targets": 1, "refused": 1}
        assert not (tmp_path / "gov" / "alisios_stage").exists()


@pytest.mark.alisios_facade
def test_facade_admits_a_rollover_template_under_its_start_day(tmp_path, sealed, guards, monkeypatch):
    root = tmp_path / "alisios"
    run = root / "wg_20260501_18z_a1" / "run" / "run"
    run.mkdir(parents=True)
    (run / "namelist.input").write_text("&time_control\n start_year = 2026, start_month = 5, start_day = 2, "
                                        "start_hour = 0,\n run_hours = 24,\n/\n")
    template = run / "wrfinput_d01"
    template.write_bytes(b"synthetic")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    plan = gov.ReadPlan([gov.ReadEntry(str(template.resolve()), "template")])
    policy = gov.AlisiosPolicy({"facade": "require", "receipt_dir": str(tmp_path / "rc")})
    guards({"20260502": "DENY"})
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    with pytest.raises(gov.PlanError, match="20260502"):
        policy.refine(plan, None)
    guards({})
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    refined = gov.AlisiosPolicy({"facade": "require", "receipt_dir": str(tmp_path / "rc2")}).refine(plan, None)
    with policy.session(refined, tmp_path / "s") as session:
        pass
    target = session.receipt["backend"]["files"][str(template.resolve())]
    assert target["issue_days"] == ["20260501"] and target["valid_days"] == ["20260502"]
    # the namelist itself was read through the facade as metadata (its own READ_SPY)
    spies = list((tmp_path / "rc2" / "guard_metadata").rglob("READ_SPY.jsonl"))
    assert len(spies) == 1 and '"authorized_attempt"' in spies[0].read_text()


@pytest.mark.alisios_facade
def test_staged_analysis_refuses_every_original_alisios_open(tree, guards, tmp_path):
    guards({})
    cfg, policy, plan = api.prepare(_cfg(tree, "require"), receipt_dir=tmp_path / "gov")
    control = cfg.data["control_file"]
    with policy.session(plan, tmp_path / "gov") as session:
        run_cfg, run_plan = session.localize(cfg, plan)
        staged = run_cfg.data["control_file"]
        assert staged != control and Path(staged).is_file()
        assert {e.role for e in run_plan.entries if "alisios_stage" in e.path} == {"control", "template", "namelist", "member"}
        with netCDF4.Dataset(staged):
            pass
        with pytest.raises(PermissionError):
            netCDF4.Dataset(control)
        with pytest.raises(PermissionError):
            open(Path(control).parent / "namelist.input").read()
    assert session.receipt["alisios_guard"]["spy"]["refused_paths"] == [str(Path(control).resolve()),
                                                                       str((Path(control).parent / "namelist.input").resolve())]


def test_backend_choice_falls_back_with_a_reason_and_require_refuses(tmp_path, monkeypatch):
    root = tmp_path / "alisios"
    monkeypatch.setattr(ag, "DATA_ROOTS", (root,))
    staged_ok = gov.ReadPlan([gov.ReadEntry(str(root / "wg_20230114_18z_a1/run/run/wrfout_d01_x"), "control")])
    case = gov.ReadPlan(staged_ok.entries + [gov.ReadEntry(str(root / "wg_20230114_18z_a1/run/run/wrfbdy_d01"), "case")])
    available = (object(), object(), {"commit": ag.FACADE_COMMIT})
    monkeypatch.setattr(ag, "facade", lambda sealed_root=None: available)
    assert gov.AlisiosPolicy({}).backend(staged_ok) == ("guarded_read", None)
    kind, reason = gov.AlisiosPolicy({}).backend(case)
    assert kind == "guarded_session" and "['case']" in reason
    with pytest.raises(gov.PlanError, match="require"):
        gov.AlisiosPolicy({"facade": "require"}).backend(case)
    assert gov.AlisiosPolicy({"facade": "off"}).backend(staged_ok)[0] == "guarded_session"
    monkeypatch.setattr(ag, "facade", lambda sealed_root=None: (None, None, {"unavailable": "ImportError: x"}))
    assert gov.AlisiosPolicy({}).backend(staged_ok) == ("guarded_session", "ImportError: x")
    with pytest.raises(gov.PlanError, match="auto|require|off"):
        gov.AlisiosPolicy({"facade": "yes"}).backend(staged_ok)


def test_sealed_copy_is_the_pinned_git_blob_and_a_tampered_copy_is_replaced(tmp_path):
    dest, blobs = ag.seal_facade(tmp_path)
    for rel, blob in blobs.items():
        expect = subprocess.run(["git", "-C", ag.ALISIOS_REPO, "rev-parse", f"{ag.FACADE_COMMIT}:{rel}"],
                                check=True, capture_output=True, text=True).stdout.strip()
        assert blob == expect == ag._git_blob_id((dest / rel).read_bytes())
    reader_file = dest / "src/alisios/guarded_read.py"
    reader_file.write_text(reader_file.read_text() + "\nTAMPERED = True\n")
    ag.seal_facade(tmp_path)
    assert "TAMPERED" not in reader_file.read_text()
    reader, guard = ag._import_sealed(dest / "src")
    assert Path(reader.__file__).parent == Path(guard.__file__).parent == dest / "src" / "alisios"
    assert reader.guard_call is guard and not hasattr(reader, "TAMPERED")
