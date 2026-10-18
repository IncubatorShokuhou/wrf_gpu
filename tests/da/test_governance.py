"""Read governance: plan before open, provenance receipts, ALISIOS plug-in, custom plug-ins.

The ALISIOS plug-in is exercised with a fake ``guard_call`` and a temporary data root: no ALISIOS tool or data
is touched here.
"""
from __future__ import annotations

import hashlib
import json
import types

import netCDF4
import numpy as np
import pytest

from gpuwrf.assimilation.letkf import alisios_guard as ag
from gpuwrf.assimilation.letkf import config as dacfg
from gpuwrf.assimilation.letkf import governance as gov


def _nc(path, value=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 2)
        ds.createVariable("v", "f4", ("x",))[:] = value
    return path


def _cfg(tmp_path, **changes):
    raw = {"schema": "gpuwrf.da.v1", "t_a": "2023-01-15T00:00:00Z",
           "ensemble": {"kind": "members", "member_glob": str(tmp_path / "mem" / "m*" / "wrfout_d01_*")},
           "obs": {"sources": [{"format": "csv_surface", "path": str(tmp_path / "obs.csv")}]},
           "experiments": [{"name": "A", "assimilate": ["surface"]}]}
    raw.update(changes)
    return dacfg.validate(raw, base_dir=tmp_path)


@pytest.fixture
def tree(tmp_path):
    for m in range(3):
        _nc(tmp_path / "mem" / f"m{m}" / "wrfout_d01_2023-01-15_00:00:00", m)
    (tmp_path / "mem" / "m0" / "namelist.input").write_text("&domains\n spec_bdy_width = 5\n/\n")
    (tmp_path / "obs.csv").write_text("station_id,lat,lon,elevation_m,time,t2_c\n")
    return tmp_path


def test_plan_lists_every_input_with_roles_and_opens_nothing(tree, monkeypatch):
    opened = []
    monkeypatch.setattr(netCDF4, "Dataset", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError))
    plan = gov.plan_reads(_cfg(tree))
    roles = [e.role for e in plan.entries]
    assert roles.count("member") == 3 and roles.count("obs") == 1 and roles.count("namelist") == 1
    assert not opened


def test_plan_refuses_missing_inputs(tree):
    (tree / "obs.csv").unlink()
    with pytest.raises(gov.PlanError, match="obs file does not exist"):
        gov.plan_reads(_cfg(tree))
    with pytest.raises(gov.PlanError, match=">= 2 member files"):
        gov.plan_reads(_cfg(tree, ensemble={"kind": "members", "member_glob": str(tree / "none*")}))


def test_open_policy_receipt_hashes_and_records_opens(tree):
    cfg = _cfg(tree)
    policy = gov.make_policy(cfg.data["governance"])
    plan = policy.refine(gov.plan_reads(cfg), cfg)
    stray = _nc(tree / "stray.nc")
    with policy.session(plan, tree / "rc") as session:
        with netCDF4.Dataset(tree / "mem" / "m1" / "wrfout_d01_2023-01-15_00:00:00"):
            pass
        open(tree / "obs.csv").read()
        with netCDF4.Dataset(stray):
            pass
    receipt = json.loads((tree / "rc" / "governance.json").read_text())
    assert receipt == json.loads(json.dumps(session.receipt, default=str))
    member1 = str((tree / "mem" / "m1" / "wrfout_d01_2023-01-15_00:00:00").resolve())
    assert receipt["inputs"][member1]["sha256"] == hashlib.sha256(open(member1, "rb").read()).hexdigest()
    assert receipt["opened"][member1]["planned"] and receipt["opened"][member1]["kind"] == "netcdf_open"
    assert receipt["unplanned_opens"] == [str(stray.resolve())]
    with policy.session(plan, tree / "rc2"):          # the session can be re-entered after it closed
        pass


def test_open_policy_refuses_alisios_paths(tree, monkeypatch):
    monkeypatch.setattr(ag, "DATA_ROOTS", ((tree / "mem").resolve(),))
    cfg = _cfg(tree)
    with pytest.raises(gov.PlanError, match="governance.plugin = 'alisios'"):
        gov.make_policy(cfg.data["governance"]).refine(gov.plan_reads(cfg), cfg)


class _Refused(PermissionError):
    pass


def _fake_guard(verdicts):
    calls = {"admit": [], "scan": []}

    def scan(days, mode=None, **kw):
        calls["scan"].append(sorted(days))
        return {d: verdicts.get(d, "ALLOW") for d in days}

    def admit(issue_days, valid_days, mode=None, receipt_out=None, **kw):
        calls["admit"].append((sorted(issue_days), sorted(valid_days)))
        if receipt_out:
            open(receipt_out, "w").write("{}")
        return {"result": "ALLOW"}

    return types.SimpleNamespace(scan=scan, admit=admit, GuardRefused=_Refused), calls


def _namelist(run_dir, start, *, hours=72, interval=60, frames=1, extra=""):
    """A run's namelist.input (the governance reads the run window from it as admitted metadata)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    y, m, d, h = start
    (run_dir / "namelist.input").write_text(
        f"&time_control\n start_year = {y}, start_month = {m}, start_day = {d}, start_hour = {h},\n"
        f" run_hours = {hours},\n history_interval = {interval},\n frames_per_outfile = {frames},\n{extra}/\n"
        "&domains\n max_dom = 1,\n/\n&bdy_control\n spec_bdy_width = 5,\n/\n")


def _run(root, issue):
    run = root / f"wg_{issue}_18z_a1" / "run" / "run"
    _namelist(run, (int(issue[:4]), int(issue[4:6]), int(issue[6:]), 18))
    return run


def _archive(root):
    run_a, run_b = _run(root, "20260501"), _run(root, "20260502")
    for run, days in ((run_a, ("2026-05-02", "2026-05-03")), (run_b, ("2026-05-03",))):
        for day in days:
            _nc(run / f"wrfout_d01_{day}_12:00:00")
    control = _nc(_run(root, "20260503") / "wrfout_d01_2026-05-04_12:00:00")
    return control


def test_alisios_policy_drops_deny_days_before_open_and_admits_exact_set(tmp_path, monkeypatch):
    root = tmp_path / "alisios"
    control = _archive(root)
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, calls = _fake_guard({"20260502": "DENY"})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    (tmp_path / "obs.csv").write_text("station_id,lat,lon,elevation_m,time,t2_c\n")
    cfg = _cfg(tmp_path, t_a="2026-05-04T12:00:00Z", control_file=str(control),
               ensemble={"kind": "same_phase", "runs_glob": str(root / "wg_*" / "run" / "run")},
               governance={"plugin": "alisios"})
    policy = gov.make_policy(cfg.data["governance"], receipt_dir=tmp_path / "rc")
    try:
        plan = policy.refine(gov.plan_reads(cfg), cfg)
    except Exception as exc:         # a DENY-issue run must be dropped before its run metadata is even requested
        raise AssertionError(f"a DENY issue-day archive run refused the whole plan: {exc!r}") from exc
    metadata = tmp_path / "rc" / "guard_metadata"
    assert sorted(p.name for p in metadata.iterdir()) == ["wg_20260501_18z_a1", "wg_20260503_18z_a1"]
    archive = [e for e in plan.entries if e.role == "archive"]
    # run 0501: its 0502 frame (DENY valid day) dropped, 0503 frame kept; run 0502: DENY issue day -> dropped
    assert [(e.path.split("/")[-4], e.valid_days) for e in archive] == [("wg_20260501_18z_a1", ("20260503",))]
    assert sorted(d["path"].split("/")[-4] for d in plan.dropped) == ["wg_20260501_18z_a1", "wg_20260502_18z_a1"]
    assert plan.notes["issue_days"] == ["20260501", "20260503"]
    with policy.session(plan, tmp_path / "rc") as session:
        for e in archive:
            with netCDF4.Dataset(e.path):
                pass
        with pytest.raises(_Refused):
            netCDF4.Dataset(root / "wg_20260501_18z_a1" / "run" / "run" / "wrfout_d01_2026-05-02_12:00:00")
    assert calls["admit"] and calls["admit"][-1] == (["20260501", "20260503"], ["20260503", "20260504"])
    receipt = json.loads((tmp_path / "rc" / "governance.json").read_text())
    assert receipt["plugin"] == "alisios" and receipt["alisios_guard"]["spy"]["refused_paths"]


def test_alisios_policy_refuses_raw_observation_files(tmp_path, monkeypatch):
    root = tmp_path / "alisios"
    control = _archive(root)
    raw_obs = root / "obs" / "stations.csv"
    raw_obs.parent.mkdir()
    raw_obs.write_text("x\n")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, _ = _fake_guard({})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    cfg = _cfg(tmp_path, t_a="2026-05-04T12:00:00Z", control_file=str(control),
               ensemble={"kind": "same_phase", "runs_glob": str(root / "wg_*" / "run" / "run")},
               obs={"sources": [{"format": "csv_surface", "path": str(raw_obs)}]}, governance={"plugin": "alisios"})
    with pytest.raises(gov.PlanError, match="guarded extract"):
        gov.make_policy(cfg.data["governance"]).refine(gov.plan_reads(cfg), cfg)


class SitePolicy(gov.OpenPolicy):
    name = "site"


def site_factory(options):
    return SitePolicy(options)


def bad_factory(options):
    return object()


def test_custom_plugin_factory(tree):
    cfg = _cfg(tree, governance={"plugin": f"{__name__}:site_factory", "options": {"k": 1}})
    policy = gov.make_policy(cfg.data["governance"])
    assert isinstance(policy, SitePolicy) and policy.options == {"k": 1}
    with pytest.raises(TypeError):
        gov.make_policy({"plugin": f"{__name__}:bad_factory"})


def test_max_members_limits_the_admitted_set_after_deny_filtering(tmp_path, monkeypatch):
    root = tmp_path / "alisios"
    control = _archive(root)
    for issue, day in (("20260429", "2026-04-30"), ("20260430", "2026-05-01")):
        _nc(_run(root, issue) / f"wrfout_d01_{day}_12:00:00")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, calls = _fake_guard({"20260502": "DENY"})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    (tmp_path / "obs.csv").write_text("station_id,lat,lon,elevation_m,time,t2_c\n")
    raw = {"schema": "gpuwrf.da.v1", "t_a": "2026-05-04T12:00:00Z", "control_file": str(control),
           "ensemble": {"kind": "same_phase", "runs_glob": str(root / "wg_*" / "run" / "run"), "max_members": 2},
           "obs": {"sources": [{"format": "csv_surface", "path": str(tmp_path / "obs.csv")}]},
           "governance": {"plugin": "alisios"}, "experiments": [{"name": "A", "assimilate": ["surface"]}]}
    from gpuwrf.assimilation.letkf import api
    cfg, policy, plan = api.prepare(dacfg.validate(raw, base_dir=tmp_path), receipt_dir=tmp_path / "rc")
    archive = [e.path.split("/")[-4] + "/" + e.path[-19:-9] for e in plan.entries if e.role == "archive"]
    # DENY frames go first (run 0502, frame 05-02); of the three ALLOW frames only the newest two are admitted
    assert archive == ["wg_20260430_18z_a1/2026-05-01", "wg_20260501_18z_a1/2026-05-03"]
    assert plan.notes["issue_days"] == ["20260430", "20260501", "20260503"]
    assert [d["path"].split("/")[-4] for d in plan.dropped if d["reason"].startswith("beyond max_members")] == [
        "wg_20260429_18z_a1"]
    with policy.session(plan, tmp_path / "rc"):
        pass
    assert calls["admit"][-1] == (["20260430", "20260501", "20260503"], ["20260501", "20260503", "20260504"])


# --- complete valid-day coverage from the run's admitted namelist (critic cr-w41-da-core item 1) -------------------

def _template_plan(tmp_path, monkeypatch, verdicts, *, start=(2026, 5, 2, 0), extra=""):
    root = tmp_path / "alisios"
    run = root / "wg_20260501_18z_a1" / "run" / "run"
    _namelist(run, start, hours=24, extra=extra)
    template = _nc(run / "wrfinput_d01")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, calls = _fake_guard(verdicts)
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    plan = gov.ReadPlan([gov.ReadEntry(str(template.resolve()), "template")])
    cfg = types.SimpleNamespace(t_a=None)
    return gov.AlisiosPolicy({"facade": "off", "receipt_dir": str(tmp_path / "rc")}), plan, cfg, template, calls


def test_template_issued_the_day_before_is_admitted_under_its_start_day(tmp_path, monkeypatch):
    # issued 2026-05-01 18Z, the run (and its wrfinput) starts 2026-05-02 00Z
    policy, plan, cfg, template, calls = _template_plan(tmp_path, monkeypatch, {})
    refined = policy.refine(plan, cfg)
    assert refined.entries[0].valid_days == ("20260502",)
    with policy.session(refined, tmp_path / "rc") as session:
        open(template, "rb").read()
    assert calls["admit"][-1] == (["20260501"], ["20260502"])
    assert session.receipt["alisios_guard"]["valid_days"] == ["20260502"]


def test_template_on_a_deny_start_day_is_refused_before_any_open(tmp_path, monkeypatch):
    policy, plan, cfg, template, calls = _template_plan(tmp_path, monkeypatch, {"20260502": "DENY"})
    seen = []
    monkeypatch.setattr(netCDF4, "Dataset", lambda *a, **k: seen.append(a) or (_ for _ in ()).throw(AssertionError))
    with pytest.raises(gov.PlanError, match=r"wrfinput_d01 \(template\) covers non-ALLOW day\(s\) \['20260502'\]"):
        policy.refine(plan, cfg)
    assert not seen
    assert all(admit[1] != ["20260502"] for admit in calls["admit"])     # only the namelist (issue day) was admitted


def _frames_plan(tmp_path, monkeypatch, verdicts, *, frames, interval, role="archive"):
    root = tmp_path / "alisios"
    run = root / "wg_20260501_18z_a1" / "run" / "run"
    _namelist(run, (2026, 5, 1, 18), hours=72, interval=interval, frames=frames)
    path = _nc(run / "wrfout_d01_2026-05-02_12:00:00")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, _ = _fake_guard(verdicts)
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    meta = {"valid": "2026050212", "run": str(run)} if role == "archive" else {}
    return gov.AlisiosPolicy({"facade": "off", "receipt_dir": str(tmp_path / "rc")}), \
        gov.ReadPlan([gov.ReadEntry(str(path.resolve()), role, (), meta)]), types.SimpleNamespace(t_a=None)


def test_history_file_coverage_includes_every_frame_it_can_hold(tmp_path, monkeypatch):
    # 10 frames every 180 min from 05-02 12Z reach 05-03 15Z: two valid days, not just the file-name day
    policy, plan, cfg = _frames_plan(tmp_path, monkeypatch, {}, frames=10, interval=180)
    assert policy.refine(plan, cfg).entries[0].valid_days == ("20260502", "20260503")
    policy, plan, cfg = _frames_plan(tmp_path / "one", monkeypatch, {}, frames=1, interval=60)
    assert policy.refine(plan, cfg).entries[0].valid_days == ("20260502",)


def test_a_deny_day_on_a_later_frame_drops_the_archive_frame_and_refuses_a_control(tmp_path, monkeypatch):
    policy, plan, cfg = _frames_plan(tmp_path, monkeypatch, {"20260503": "DENY"}, frames=10, interval=180)
    refined = policy.refine(plan, cfg)
    assert refined.entries == [] and "coverage day(s) ['20260503']" in refined.dropped[-1]["reason"]
    policy, plan, cfg = _frames_plan(tmp_path / "c", monkeypatch, {"20260503": "DENY"}, frames=10, interval=180,
                                     role="control")
    with pytest.raises(gov.PlanError, match="20260503"):
        policy.refine(plan, cfg)


def test_wrf_registry_defaults_and_unknown_coverage(tmp_path, monkeypatch):
    # no frames_per_outfile/history_interval: WRF defaults (10 frames, interval unset) -> conservatively to run end
    root = tmp_path / "alisios"
    run = root / "wg_20260501_18z_a1" / "run" / "run"
    run.mkdir(parents=True)
    (run / "namelist.input").write_text("&time_control\n start_year = 2026, start_month = 5, start_day = 1, "
                                        "start_hour = 18,\n run_hours = 48,\n/\n")
    path = _nc(run / "wrfout_d01_2026-05-02_12:00:00")
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    fake, _ = _fake_guard({})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    policy = gov.AlisiosPolicy({"facade": "off", "receipt_dir": str(tmp_path / "rc")})
    entry = gov.ReadEntry(str(path.resolve()), "control")
    assert policy.refine(gov.ReadPlan([entry]), None).entries[0].valid_days == ("20260502", "20260503")
    # history_interval set, frames_per_outfile unset: WRF's 10 frames x 180 min from 05-02 12Z reach 05-03 15Z
    (run / "namelist.input").write_text("&time_control\n start_year = 2026, start_month = 5, start_day = 1, "
                                        "start_hour = 18,\n run_hours = 48,\n history_interval = 180,\n/\n")
    policy = gov.AlisiosPolicy({"facade": "off", "receipt_dir": str(tmp_path / "rc1")})
    assert policy.refine(gov.ReadPlan([entry]), None).entries[0].valid_days == ("20260502", "20260503")
    late = _nc(run / "wrfout_d01_2026-05-04_12:00:00")             # after the 48 h run end
    with pytest.raises(gov.PlanError, match="outside its run window"):
        policy.refine(gov.ReadPlan([gov.ReadEntry(str(late.resolve()), "control")]), None)
    (run / "namelist.input").unlink()
    with pytest.raises(gov.PlanError, match="no namelist.input"):
        gov.AlisiosPolicy({"facade": "off", "receipt_dir": str(tmp_path / "rc2")}).refine(gov.ReadPlan([entry]), None)
