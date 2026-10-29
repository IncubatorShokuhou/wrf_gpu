"""CPU stub forecasts, with real engine readers and serialized consumer receipts."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.optimize import contrasts as c


def _times(ds):
    ds.createDimension("DateStrLen", 19)
    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(
        [list(f"2026-01-01_{h:02d}:00:00") for h in range(len(ds.dimensions["Time"]))], dtype="S1")


@pytest.fixture
def study(tmp_path):
    root = tmp_path/"control"
    root.mkdir()
    (root/"namelist.input").write_text("""&time_control
 start_year=2026, start_month=1, start_day=1, start_hour=0,
 io_form_auxinput4=2, auxinput4_interval=60,
 /
 &domains
 max_dom=1, time_step=60,
 /
 &physics
 sst_update=1,
 /
""")
    for filename, nt in (("wrfinput_d01", 1), ("wrfbdy_d01", 3), ("wrflowinp_d01", 3)):
        with Dataset(root/filename, "w") as ds:
            ds.createDimension("Time", nt)
            ds.createDimension("y", 2)
            ds.createDimension("x", 2)
            ds.MAP_PROJ = 1
            ds.GRID_ID = 1
            _times(ds)
            if filename.startswith("wrfinput"):
                names = c.IC_REQUIRED | c.ATMOSPHERE | {"HGT"}
            elif filename.startswith("wrfbdy"):
                names = {f"{f}_{s}" for f in ("U", "V", "T", "QVAPOR", "PH", "MU", "W")
                         for s in ("BXS", "BXE", "BYS", "BYE", "BTXS", "BTXE", "BTYS", "BTYE")}
            else:
                names = {"SST", "SEAICE", "VEGFRA", "ALBBCK"}
            for name in sorted(names):
                var = ds.createVariable(name, "f4", ("Time", "y", "x"))
                var[:] = np.arange(nt)[:, None, None] + (290 if name == "SST" else .25)
                if name == "SEAICE":
                    var[:] = 0
    raw = {"schema": c.SCHEMA, "hours": 2, "member_timeout_s": 60,
           "control": {"id": "control", "input_dir": "control"}, "arms": []}
    path = tmp_path/"study.json"
    return path, raw


def arm(study, kind, name="candidate"):
    path, raw = study
    source, target = path.parent/"control", path.parent/name
    shutil.copytree(source, target)
    raw["arms"].append({"id": name, "kind": kind, "input_dir": name})
    if kind == "ic":
        filename, field = "wrfinput_d01", "QVAPOR"
    elif kind == "lbc":
        filename, field = "wrfbdy_d01", "T_BXS"
    elif kind == "sst-held":
        filename, field = "wrflowinp_d01", "SST"
    else:
        filename, field = "wrfinput_d01", "U"
    with Dataset(target/filename, "r+") as ds:
        if kind == "sst-held":
            ds[field][:] = ds[field][0]
        else:
            ds[field][:] += .5
    path.write_text(json.dumps(raw))
    return target


def engine_reads(member, *, skip=None):
    """Stub stepping; actual IC materializer, bdy decoder and aux4 loader run."""
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.boundary_replay import decode_wrfbdy
    from gpuwrf.io.lower_boundary import load_lower_boundary
    run = Gen2Run(member["input_dir"])
    for filename, record in member["files"].items():
        if filename.startswith("wrfinput"):
            for field in record["fields"]:
                if field != "Times" and (filename, field) != skip:
                    # Actual host-read boundary used by device materialization.
                    run.load_wrfinput("d01", field, lazy=False)
        elif filename.startswith("wrfbdy") and skip != (filename, "*"):
            for i in range(3):
                decode_wrfbdy(record["identity"]["path"], time_index=i)
    if skip != ("wrflowinp_d01", "*"):
        boundary = load_lower_boundary(member["input_dir"], member["config"], "d01",
                                       run_start=datetime(2026, 1, 1), dt_s=60, shape=(2, 2))
        assert boundary.sst.shape == (3, 2, 2)
    return 0


@pytest.mark.parametrize("kind", ["ic", "lbc", "sst-held"])
def test_plan_run_and_parse_real_loader_receipts(study, kind, tmp_path):
    arm(study, kind)
    plan = c.load_plan(study[0])
    assert plan["planned_case_hours"] == 4
    assert len(plan["members"][1]["changes"]) == 1
    original = {f["identity"]["path"]: Path(f["identity"]["path"]).read_bytes()
                for m in plan["members"] for f in m["files"].values()}
    def runner(p, m, out):
        def model(argv):
            assert argv == ["run", "--input-dir", m["input_dir"], "--domains-from-namelist",
                            "--hours", str(p["hours"]), "--output-dir", str(out/"forecast")]
            return engine_reads(m)
        c.member_run(p, m, out, model_main=model)
    result = c.run(plan, tmp_path/"result", runner=runner, member_env={"JAX_PLATFORMS": "cpu"})
    assert result["control"] == "control"
    assert [m["kind"] for m in result["members"]] == ["control", kind]
    for m in result["members"]:
        receipt = json.loads(Path(m["receipt"]["path"]).read_text())
        assert receipt["status"] == "CONSUMED" and receipt["platform"].startswith("stub")
        assert {r["loader"] for r in receipt["reads"]} == {
            "Gen2Run._read_variable", "decode_wrfbdy", "load_lower_boundary"}
    assert all(Path(p).read_bytes() == data for p, data in original.items())


@pytest.mark.parametrize("skip", [("wrfinput_d01", "QVAPOR"), ("wrfbdy_d01", "*"), ("wrflowinp_d01", "*")])
def test_successful_process_without_consumption_refuses(study, tmp_path, skip):
    arm(study, "ic")
    plan = c.load_plan(study[0])
    member = plan["members"][1]
    with pytest.raises(ValueError, match="not consumed"):
        c.member_run(plan, member, tmp_path/"unconsumed", model_main=lambda argv: engine_reads(member, skip=skip))
    assert not (tmp_path/"unconsumed"/"consumer.json").exists()


@pytest.mark.parametrize("mutation", ["foreign", "sha", "deleted", "input", "horizon", "rc"])
def test_receipt_and_input_refusals(study, tmp_path, mutation):
    target = arm(study, "ic")
    plan = c.load_plan(study[0])
    member = plan["members"][1]
    if mutation == "input":
        with Dataset(target/"wrfinput_d01", "r+") as ds:
            ds["QVAPOR"][:] += .5
        with pytest.raises(ValueError, match="changed"):
            c.member_run(plan, member, tmp_path/"refused", model_main=lambda argv: 0)
        return
    if mutation == "rc":
        with pytest.raises(ValueError, match="exit 2"):
            c.member_run(plan, member, tmp_path/"refused", model_main=lambda argv: 2)
        return
    receipt = c.member_run(plan, member, tmp_path/"ok", model_main=lambda argv: engine_reads(member))
    if mutation == "foreign":
        receipt["member"] = "another-arm"
    elif mutation == "sha":
        receipt["reads"][0]["sha256"] = "0"*64
    elif mutation == "deleted":
        receipt["reads"] = receipt["reads"][1:]
    else:
        # run() also checks the exact requested horizon on serialized receipts.
        receipt["hours"] = 1
        def runner(p, m, out):
            out.mkdir()
            (out/"consumer.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="horizon"):
            c.run(plan, tmp_path/"foreign", runner=runner, member_env={"JAX_PLATFORMS": "cpu"})
        return
    with pytest.raises(ValueError):
        c.validate_receipt(member, receipt)


@pytest.mark.parametrize("mutation", ["lbc", "ic", "metadata", "namelist", "unheld", "duplicate", "nochange", "sstoff"])
def test_declarations_fail_closed(study, mutation):
    target = arm(study, "sst-held")
    path, raw = study
    if mutation in ("lbc", "ic", "metadata"):
        filename = "wrfbdy_d01" if mutation == "lbc" else "wrfinput_d01"
        with Dataset(target/filename, "r+") as ds:
            if mutation == "metadata":
                ds.MAP_PROJ = 3
            else:
                ds["T_BXS" if mutation == "lbc" else "U"][:] += 1
    elif mutation in ("namelist", "sstoff"):
        nml = target/"namelist.input"
        nml.write_text(nml.read_text().replace("sst_update=1", "sst_update=0" if mutation == "sstoff" else "sst_update=1, radt=5"))
    elif mutation == "unheld":
        with Dataset(target/"wrflowinp_d01", "r+") as ds:
            ds["SST"][1] += 1
    elif mutation == "duplicate":
        raw["arms"][0]["id"] = "control"
    else:
        raw["arms"][0]["input_dir"] = "control"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        c.load_plan(path)


def test_warm_start_requires_real_cycling_provenance(study):
    target = arm(study, "warm-start")
    nml = target/"namelist.input"
    nml.write_text(nml.read_text().replace("&time_control", "&time_control\n cycling=.true.,"))
    with pytest.raises(ValueError, match="provenance"):
        c.load_plan(study[0])


def warm_case(study):
    import hashlib
    target = arm(study, "warm-start")
    nml = target/"namelist.input"
    nml.write_text(nml.read_text().replace("&time_control", "&time_control\n cycling=.true.,"))
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        fields = {}
        for name in c.ATMOSPHERE:
            a = np.asarray(ds[name][0])
            fields[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
        ds.GPUWRF_ATMOSPHERE_CYCLE = json.dumps({"schema": 1, "grid_id": 1,
                                                "valid_time": "2026-01-01_00:00:00", "fields": fields,
                                                "wrfinput_sha256": c._identity(study[0].parent/"control"/"wrfinput_d01")["sha256"]})
    return target


def test_warm_plan_admits_provenance_and_reads(study, tmp_path):
    warm_case(study)
    plan = c.load_plan(study[0])
    member = plan["members"][1]
    receipt = c.member_run(plan, member, tmp_path/"warm", model_main=lambda argv: engine_reads(member))
    c.validate_receipt(member, json.loads(json.dumps(receipt)))
    assert receipt["kind"] == "warm-start"


def test_warm_does_not_change_continued_lbc(study):
    target = warm_case(study)
    with Dataset(target/"wrfbdy_d01", "r+") as ds:
        ds["U_BXS"][1] += 1
    with pytest.raises(ValueError, match="continued LBC"):
        c.load_plan(study[0])


def test_warm_first_boundary_rebalance_is_allowed(study):
    target = warm_case(study)
    with Dataset(target/"wrfbdy_d01", "r+") as ds:
        ds["U_BXS"][0] += 1
        ds["U_BTXS"][0] -= 1/3600
    assert len(c.load_plan(study[0])["members"][1]["changes"]) == 3


def test_native_warm_input_can_retain_history_dimensions_and_title(study):
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        ds.TITLE = "OUTPUT FROM WRF"
        ds.createDimension("retained_diagnostic", 3)
        ds.createVariable("U10", "f4", ("Time", "retained_diagnostic"))[:] = 0
    member = c.load_plan(study[0])["members"][1]
    assert member["kind"] == "warm-start"
    assert {"file": "wrfinput_d01", "field": "U10", "role": "retained-diagnostic"} in member["changes"]


def test_warm_provenance_must_bind_control_template(study):
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        marker = json.loads(ds.GPUWRF_ATMOSPHERE_CYCLE)
        marker["wrfinput_sha256"] = "0"*64
        ds.GPUWRF_ATMOSPHERE_CYCLE = json.dumps(marker)
    with pytest.raises(ValueError, match="control analysis"):
        c.load_plan(study[0])


def test_cli_plan_uses_contrast_dispatch(study, tmp_path, capsys):
    from gpuwrf.cli import main
    arm(study, "ic")
    out = tmp_path/"cli-plan.json"
    assert main(["optimize", "contrasts", "plan", "--study", str(study[0]), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == c.PLAN
    assert json.loads(capsys.readouterr().out)["planned_case_hours"] == 4


def test_symlink_to_alisios_refuses_before_native_reader(study, monkeypatch):
    arm(study, "ic")
    from gpuwrf.assimilation.letkf import alisios_guard
    original = alisios_guard.is_alisios_data
    monkeypatch.setattr(alisios_guard, "is_alisios_data", lambda path: Path(path).name == "wrfinput_d01" or original(path))
    from gpuwrf.ens import workflows
    monkeypatch.setattr(workflows, "_case", lambda *a: pytest.fail("native read before ALISIOS admission"))
    with pytest.raises(ValueError, match="facade"):
        c.load_plan(study[0])


def test_observer_is_restored_on_exception(study):
    from gpuwrf.io.gen2_accessor import Gen2Run
    arm(study, "ic")
    member = c.load_plan(study[0])["members"][0]
    original = Gen2Run._read_variable
    with pytest.raises(RuntimeError):
        with c.consumer_reads(c.Consumer(member)):
            raise RuntimeError("model crashed")
    assert Gen2Run._read_variable is original


def test_actual_readback_mismatch_refuses(study):
    arm(study, "ic")
    member = c.load_plan(study[0])["members"][0]
    reader = c.Consumer(member)
    with pytest.raises(ValueError, match="readback differs"):
        reader.observe(Path(member["input_dir"])/"wrfinput_d01", "U", 0, np.zeros((2, 2)), "Gen2Run._read_variable")
    with pytest.raises(ValueError, match="undeclared input"):
        reader.observe("/foreign/wrfinput_d01", "U", 0, np.zeros((2, 2)), "Gen2Run._read_variable")


def test_default_runner_preserves_environment_and_hard_timeout(study, tmp_path, monkeypatch):
    arm(study, "ic")
    plan = c.load_plan(study[0])
    seen = []
    def child(argv, *, check, timeout, env, pass_fds):
        assert argv[:3] == [c.sys.executable, "-m", "gpuwrf.optimize.contrasts"]
        assert argv[3] == "member" and check and timeout == 60
        assert env["JAX_PLATFORMS"] == "cpu" and env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
        assert pass_fds == ()
        m = next(m for m in plan["members"] if m["id"] == argv[argv.index("--arm")+1])
        out = Path(argv[argv.index("--output")+1])
        seen.append(m["id"])
        c.member_run(plan, m, out, model_main=lambda args: engine_reads(m))
    monkeypatch.setattr(c.subprocess, "run", child)
    c.run(plan, tmp_path/"native-launch", member_env={"JAX_PLATFORMS": "cpu"})
    assert seen == ["control", "candidate"]


def test_timeout_never_publishes_result(study, tmp_path, monkeypatch):
    arm(study, "ic")
    plan = c.load_plan(study[0])
    def timeout(*args, **kw):
        raise c.subprocess.TimeoutExpired("contrast", 60)
    monkeypatch.setattr(c.subprocess, "run", timeout)
    with pytest.raises(c.subprocess.TimeoutExpired):
        c.run(plan, tmp_path/"timed-out", member_env={"JAX_PLATFORMS": "cpu"})
    assert not (tmp_path/"timed-out"/"result.json").exists()


def test_gpu_lock_refusal_happens_before_launch(study, tmp_path):
    arm(study, "ic")
    with pytest.raises(ValueError, match="with_gpu_lock"):
        c.run(c.load_plan(study[0]), tmp_path/"unlocked", member_env={"JAX_PLATFORMS": "cuda"},
              runner=lambda *args: pytest.fail("unlocked forecast"))
    assert not (tmp_path/"unlocked").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.ma.masked_all((2, 2))])
def test_nonfinite_readbacks_refuse(value):
    with pytest.raises(ValueError, match="nonfinite|missing"):
        c._hash(value)


@pytest.mark.parametrize("field", ["HGT", "SST", "SMOIS", "UNKNOWN_PHYSICAL_INPUT"])
def test_critic_f1_untransported_warm_change_refuses(study, field):
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        if field not in ds.variables:
            ds.createVariable(field, "f4", ("Time", "y", "x"))[:] = 0
        ds[field][:] += 100
        marker = json.loads(ds.GPUWRF_ATMOSPHERE_CYCLE)
        assert field not in marker["fields"]  # Exact critic HGT witness alongside transported U.
    with pytest.raises(ValueError, match="undeclared field change"):
        c.load_plan(study[0])


def test_critic_f1_legacy_plan_cannot_report_unread_hgt_consumed(study, tmp_path):
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        ds["HGT"][:] += 100
    # Even a legacy already-admitted member must not filter HGT out at finish.
    member = c._case({"id": "candidate", "kind": "warm-start", "input_dir": str(target)}, study[0].parent, 2)
    member["changes"] = [{"file": "wrfinput_d01", "field": f} for f in ("U", "HGT")]
    with pytest.raises(ValueError, match="unsupported physical warm-input"):
        c.member_run({"hours": 2}, member, tmp_path/"unread-HGT",
                     model_main=lambda argv: engine_reads(member, skip=("wrfinput_d01", "HGT")))
    assert not (tmp_path/"unread-HGT"/"consumer.json").exists()


def test_warm_changed_transport_still_requires_consumer_read(study, tmp_path):
    warm_case(study)
    plan = c.load_plan(study[0])
    member = plan["members"][1]
    with pytest.raises(ValueError, match="not consumed"):
        c.member_run(plan, member, tmp_path/"unread-U",
                     model_main=lambda argv: engine_reads(member, skip=("wrfinput_d01", "U")))
    assert not (tmp_path/"unread-U"/"consumer.json").exists()


def test_retained_diagnostic_is_explicitly_not_a_consumed_intervention(study, tmp_path):
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        ds.createVariable("U10", "f4", ("Time", "y", "x"))[:] = 0
    plan = c.load_plan(study[0])
    member = plan["members"][1]
    receipt = c.member_run(plan, member, tmp_path/"diagnostic",
                           model_main=lambda argv: engine_reads(member, skip=("wrfinput_d01", "U10")))
    c.validate_receipt(member, json.loads(json.dumps(receipt)))
    assert not any(row["field"] == "U10" for row in receipt["reads"])
    assert next(change for change in member["changes"] if change["field"] == "U10")["role"] == "retained-diagnostic"
    assert receipt["retained_diagnostics"] == [{"file": "wrfinput_d01", "field": "U10", "status": "RETAINED_ONLY_NOT_CONSUMER_PROOF"}]
    receipt.pop("retained_diagnostics")
    with pytest.raises(ValueError, match="disclosure"):
        c.validate_receipt(member, receipt)


def test_preexisting_diagnostic_cannot_hide_changed_input(study):
    with Dataset(study[0].parent/"control"/"wrfinput_d01", "r+") as ds:
        ds.createVariable("U10", "f4", ("Time", "y", "x"))[:] = 0
    target = warm_case(study)
    with Dataset(target/"wrfinput_d01", "r+") as ds:
        ds["U10"][:] += 2
    with pytest.raises(ValueError, match="undeclared field change"):
        c.load_plan(study[0])


def test_critic_f2_private_descriptor_survives_native_dispatch(study, tmp_path, monkeypatch):
    import os
    from gpuwrf.ens import workflows
    arm(study, "ic")
    plan = c.load_plan(study[0])
    private = tmp_path/"private-capability"
    unrelated = tmp_path/"unrelated-capability"
    private.touch()
    unrelated.touch()
    monkeypatch.setattr(workflows, "GPU_LOCK_FILE", private)  # No official file/token/lease touched.
    fd, other = os.open(private, os.O_RDONLY), os.open(unrelated, os.O_RDONLY)
    real_run = c.subprocess.run
    seen = []
    try:
        os.set_inheritable(fd, True)
        os.set_inheritable(other, True)
        identity = [(os.fstat(n).st_dev, os.fstat(n).st_ino) for n in (fd, other)]
        code = """import json,os,sys
seen=[]
for fd,dev,ino in json.loads(sys.argv[1]):
 try:
  st=os.fstat(fd); seen.append((st.st_dev,st.st_ino)==(dev,ino))
 except OSError: seen.append(False)
print(json.dumps(seen))
"""
        probe = [c.sys.executable, "-c", code, json.dumps([(n, *st) for n, st in zip((fd, other), identity)])]
        env = {"JAX_PLATFORMS": "cpu", "GPUWRF_GPU_LOCK_HELD": "1", "GPUWRF_GPU_LOCK_FD": str(fd)}
        # Critic's negative control: inheritable alone does not survive subprocess defaults.
        lost = real_run(probe, env=env, check=True, timeout=10, capture_output=True, text=True)
        assert json.loads(lost.stdout) == [False, False]
        def child(argv, *, check, timeout, env, pass_fds):
            assert pass_fds == (fd,)
            retained = real_run(probe, env=env, check=check, timeout=timeout, pass_fds=pass_fds,
                                capture_output=True, text=True)
            assert json.loads(retained.stdout) == [True, False]
            seen.append(pass_fds)
            member = next(m for m in plan["members"] if m["id"] == argv[argv.index("--arm")+1])
            c.member_run(plan, member, Path(argv[argv.index("--output")+1]),
                         model_main=lambda args: engine_reads(member))
        monkeypatch.setattr(c.subprocess, "run", child)
        c.run(plan, tmp_path/"fd-wave", member_env=env)
    finally:
        os.close(fd)
        os.close(other)
    assert seen == [(fd,), (fd,)]


@pytest.mark.parametrize("bad", ["", "0", "1", "2", "-1", "abc", "closed", "unrelated", "unheld", "missing-official"])
def test_invalid_lock_descriptor_refuses_before_child(study, tmp_path, monkeypatch, bad):
    import os
    from gpuwrf.ens import workflows
    arm(study, "ic")
    plan = c.load_plan(study[0])
    private, other = tmp_path/"private-lock", tmp_path/"unrelated"
    private.touch()
    other.touch()
    monkeypatch.setattr(workflows, "GPU_LOCK_FILE", private)
    fd = os.open(other if bad == "unrelated" else private, os.O_RDONLY)
    env = {"JAX_PLATFORMS": "cpu", "GPUWRF_GPU_LOCK_HELD": "1",
           "GPUWRF_GPU_LOCK_FD": str(fd) if bad in ("closed", "unrelated", "unheld", "missing-official") else bad}
    if bad == "unheld":
        env.pop("GPUWRF_GPU_LOCK_HELD")
    if bad == "closed":
        os.close(fd)
    if bad == "missing-official":
        monkeypatch.setattr(workflows, "GPU_LOCK_FILE", tmp_path/"missing-lock")
    try:
        with pytest.raises(ValueError, match="lock descriptor"):
            c.run(plan, tmp_path/"invalid-fd", member_env=env,
                  runner=lambda *args: pytest.fail("invalid descriptor reached a child"))
    finally:
        if bad != "closed":
            os.close(fd)
    assert not (tmp_path/"invalid-fd").exists()


def test_cpu_without_lock_never_stats_official_path(monkeypatch):
    from gpuwrf.ens import workflows
    class Forbidden:
        def stat(self):
            pytest.fail("CPU/no-lock execution must not inspect the official lock")
    monkeypatch.setattr(workflows, "GPU_LOCK_FILE", Forbidden())
    assert c._lock_fds({"JAX_PLATFORMS": "cpu"}) == ()
