"""Port of cr-w44-launchers 27c27087's 29 controls, plus RD078/lease seams.

Every cross, clock, df, lock, lease, notice and registry fixture is PRIVATE.
No tests read operational plans, registry storage, GPU state or manager paths.
"""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUB = Path(__file__).with_name("gate_stubs.py")
MODULE = ROOT / "scripts/gpu_gate.py"
spec = importlib.util.spec_from_file_location("gpu_gate", MODULE)
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
GIB = 2**30


def epoch(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


START = epoch("2026-10-27T01:00:00Z")
LATEST = epoch("2026-10-27T03:30:00Z")
END = epoch("2026-10-27T07:30:00Z")


def stamp(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(p, data):
    p.write_text(json.dumps(data, indent=2))


def calls(root, mode=None):
    p = root / "calls.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rows if mode is None or r["mode"] == mode]


def fixture(root, **c):
    for d in ("bin", "cross", "leases", "entries"):
        (root / d).mkdir()
    c.setdefault("end_epoch", END)
    write(root / "case.json", c)
    (root / "clock").write_text(str(c.get("start", START)))
    (root / "locked").write_text("1" if c.get("locked") else "0")
    (root / "stat").write_text("cpu3 100 0 0 100 0 0 0 0 0 0\n")
    if c.get("trigger", True):
        (root / "cross/GO.md").write_text(c.get("trigger_text", "GO\n"))
    if c.get("lease", "present") is not None:
        (root / "lease").write_text("expires-utc: " + c.get("lease", "2026-10-27T07:40:00Z") + "\n")
    for name in ("date", "df", "lslocks", "sleep"):
        p = root / "bin" / name
        p.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{STUB}" {name} "$@"\n')
        p.chmod(0o700)
    env = {**os.environ, "PRIVATE_CASE": str(root), "JAX_PLATFORMS": "cpu",
           "PATH": str(root / "bin") + ":" + os.environ["PATH"], "PYTHONPATH": str(ROOT / "src"),
           "DBUS_SESSION_BUS_ADDRESS": "unix:path=/private/bus", "XDG_RUNTIME_DIR": str(root),
           "LD_LIBRARY_PATH": "/private/lib", "GPUWRF_STALE": "1", "XLA_STALE": "1",
           "JAX_STALE": "1", "TF_STALE": "1", "W44R_STALE": "1", "WRF_GPU2_SEQ_SMOKE": "1",
           "W44R_LAUNCH_SMOKE": "1", "CUDA_ROOT": "/private/cuda", "CUDA_VISIBLE_DEVICES": "",
           "LD_PRELOAD": "", "RELEASE_APIKEY": "fake-test-key"}
    env.update(c.get("extra_env", {}))
    plan = {"steps": [dict(lane="s" + str(i), path="private", sha256="stub", cap_min=30) for i in range(5)]}
    write(root / "plan.json", plan)
    cfg = dict(plan=str(root / "plan.json"), cross_dir=str(root / "cross"), run_dir=str(root / "run"),
               go_glob="GO*.md", not_before="2026-10-26T00:00:00Z", not_after=stamp(LATEST), hard_end=stamp(END),
               blocked_intervals=[["2026-10-26T19:40:00Z", "2026-10-26T22:00:00Z"]],
               lease=str(root / "lease"), lock_path=str(root / "gpu.lock"), disk_path=str(root),
               disk_floor_bytes=365*GIB, disk_need_bytes=43*GIB,
               sequencer=[sys.executable, str(STUB), "seq"],
               cpu_lease=dict(command=[sys.executable, str(STUB), "lease"], directory=str(root / "leases"),
                              cpus="3", stat_path=str(root / "stat")),
               cross_names={s: s + ".md" for s in ("DONE", "STOPPED", "NOSTART")},
               notice_hooks=[dict(min_free_bytes=423*GIB, lane="s4", entry_glob=str(root / "entries/*/ENTRY.json"),
                                  entry="rw3", pattern="ARM O3 rc=0 try", command=[sys.executable, str(STUB), "notice"])])
    cfg.update(c.get("config", {}))
    write(root / "config.json", cfg)
    return env, cfg


def gate(root, env, program=None):
    out = root / "output.log"
    with out.open("w") as f:
        command = [sys.executable, str(program), "gate"] if program else ["bash", str(ROOT / "scripts/gpu_gate.sh")]
        p = subprocess.Popen([*command, str(root / "config.json")], env=env,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=12)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
    assert (root / "run/result.json").exists(), out.read_text()
    return rc, json.loads((root / "run/result.json").read_text())


# Names retained verbatim from rw_gate_20261027_v4/probe.py (29/29).
CASES = {
    "all_five_zero": {},
    "one_step_rc7": dict(one_rc7=True, expected="STOPPED"),
    "partial_success_log": dict(end_count=1, expected="STOPPED"),
    "duplicate_step_log": dict(duplicate_steps=True, expected="STOPPED"),
    "newer_unrelated_log": dict(one_rc7=True, foreign_log=True, expected="STOPPED"),
    "trigger_absent_expiry": dict(trigger=False, start=LATEST-1, clock_queue=[LATEST+1], expected="NOSTART"),
    "trigger_arrives": dict(trigger=False, trigger_after_sleep=True),
    "lock_busy_then_free": dict(locked=True, unlock_after_sleep=True),
    "blocked_waits_until22": dict(start=epoch("2026-10-26T19:40:00Z"), clock_queue=[epoch("2026-10-26T22:00:00Z")]),
    "past_latest_start": dict(start=LATEST+1, expected="NOSTART"),
    "latest_exact": dict(start=LATEST),
    "late_after_preflight": dict(start=LATEST-2, preflight_advance=LATEST+1, expected="NOSTART"),
    "disk407": dict(free_gib=407, expected="NOSTART"),
    "disk407point5_rounded": dict(free_gib=407.5, expected="NOSTART"),
    "disk408_watcher": dict(free_gib=408, watch=True, expected="STOPPED"),
    "disk422_watcher": dict(free_gib=422, watch=True, expected="STOPPED"),
    "disk422point5_rounded": dict(free_gib=422.5, watch=True, expected="STOPPED"),
    "disk423_optional_allowed": dict(free_gib=423),
    "unrelated_session_O3": dict(free_gib=420, watch=True, unrelated_O3=True, expected="STOPPED"),
    "watcher_after_early_failure": dict(free_gib=420, one_rc7=True, expected="STOPPED"),
    "lease_missing": dict(lease=None, expected="NOSTART"),
    "lease_ends_before_hard_end": dict(lease="2026-10-27T07:00:00Z", expected="NOSTART"),
    "earlier_trigger_hard_end": dict(trigger_text="GO\nHARD_END_UTC=2026-10-27T06:30:00Z\n"),
    "earlier_end_no_time": dict(trigger_text="GO\nHARD_END_UTC=2026-10-27T03:00:00Z\n", expected="NOSTART"),
    "preflight_refuses": dict(preflight_rc=2, expected="NOSTART"),
    "unrelated_session_after_RW3_RUN": dict(free_gib=420, watch=True, unrelated_after_RUN=True, expected="STOPPED"),
    "ENTRY_written_before_watcher_observes_RUN": dict(free_gib=420, watch=True, old_ENTRY=True, expected="STOPPED"),
    "foreign_ENTRY_wrong_gate_ID": dict(free_gib=420, watch=True, foreign_ENTRY_ID=True, expected="STOPPED"),
    "stale_SEQ_SMOKE": dict(extra_env={"SEQ_SMOKE": "1"}),
}


@pytest.mark.parametrize("name", CASES)
def test_reviewed_29_controls(tmp_path, name):
    case = CASES[name]
    env, cfg = fixture(tmp_path, **case)
    rc, result = gate(tmp_path, env)
    expected = case.get("expected", "DONE")
    assert result["status"] == expected, (result, (tmp_path / "output.log").read_text())
    assert (rc == 0) == (expected == "DONE")
    assert sorted(p.stem for p in (tmp_path / "cross").glob("*.md") if p.stem != "GO") == [expected]
    actual = calls(tmp_path, "actual")
    if expected == "NOSTART":
        assert not actual
    if actual:
        got = actual[0]["env"]
        for key in ("SEQ_SMOKE", "WRF_GPU2_SEQ_SMOKE", "PREFLIGHT_ONLY", "GPUWRF_STALE", "JAX_STALE", "W44R_STALE", "TF_STALE", "XLA_STALE", "CUDA_ROOT", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "RELEASE_APIKEY"):
            assert got[key] is None
        for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "HOME", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH"):
            assert got[key] == env[key]
        assert epoch(got["HARD_END_UTC"]) <= END
    if case.get("watch"):
        notices = calls(tmp_path, "notice")
        assert len(notices) == 1
        assert notices[0]["entry"] == str(tmp_path / "entries/current/ENTRY.json")
        assert notices[0]["gid"] == result["gate_id"]
    else:
        assert not calls(tmp_path, "notice")
    assert not list((tmp_path / "leases").iterdir())


def sequence_fixture(tmp_path, **changes):
    env, cfg = fixture(tmp_path, **changes)
    package = tmp_path / "src/gpuwrf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# module-resolution double; never imported\n")
    env["PYTHONPATH"] = str(tmp_path / "src")
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{STUB}" launcher "$@"\n')
    phases = ["model", "e41", "check", "registry_end"]
    step = dict(lane="test", path=str(launcher), sha256=lib.hashlib.sha256(launcher.read_bytes()).hexdigest(), cap_min=1,
                args=["space arg", "$(touch BAD)", "'quoted'"], phases=phases,
                registry_smoke=[sys.executable, str(STUB), "registry", changes.get("forcing", "WN3")])
    write(tmp_path / "plan.json", dict(steps=[step]))
    env.update(SEQ_RUN_DIR=str(tmp_path / "sequence"), HARD_END_UTC=stamp(END))
    for key in lib.SELECTORS:
        env.pop(key, None)
    return env, step


def seq(root, env):
    return subprocess.run(["bash", str(ROOT / "scripts/night_seq_v10.sh"), str(root / "plan.json")],
                          env=env, capture_output=True, text=True, timeout=12)


def test_rd078_go_before_window_fails_legacy_passes_v10(tmp_path):
    nb = START + 120
    # Executable counterfactual: omit both not-before checks, as in the old gate.
    old, new = tmp_path / "legacy", tmp_path / "v10"
    old.mkdir(); new.mkdir()
    mutant = tmp_path / "legacy_gate.py"
    source = MODULE.read_text()
    source = source.replace('require(nb <= t <= latest, "outside start window")', 'require(t <= latest, "outside start window")')
    source = source.replace('if t >= nb and triggers and lock_free', 'if triggers and lock_free')
    mutant.write_text(source)
    env, _ = fixture(old, clock_queue=[START], config=dict(not_before=stamp(nb)))
    rc, _ = gate(old, env, mutant)
    assert rc == 0 and calls(old, "actual")[0]["time"] < nb  # historical false start
    env, _ = fixture(new, clock_queue=[nb], config=dict(not_before=stamp(nb)))
    rc, _ = gate(new, env)
    assert rc == 0
    assert calls(new, "actual")[0]["time"] >= nb


def test_rd078_systemd_path_without_python_fails_legacy_passes_v10(tmp_path):
    env, step = sequence_fixture(tmp_path)
    # Legacy bash -n/sha check passes; env python would die with rc127.
    step["env"] = dict(PATH="/usr/bin:/bin", PYTHONPATH=str(ROOT / "src"))
    assert not lib.shutil.which("python", path=step["env"]["PATH"])
    assert subprocess.run(["bash", "-n", step["path"]]).returncode == 0
    legacy = subprocess.run(["/usr/bin/env", "python", "-c", "pass"], env={**env, **step["env"]}, capture_output=True)
    assert legacy.returncode == 127  # old preflight missed the actual interpreter
    write(tmp_path / "plan.json", dict(steps=[step]))
    result = seq(tmp_path, env)
    assert result.returncode != 0 and "PATH lacks python" in result.stderr
    assert not calls(tmp_path, "launcher")
    del step["env"]
    write(tmp_path / "plan.json", dict(steps=[step]))
    assert seq(tmp_path, env).returncode == 0


def test_rd078_bad_registry_schema_fails_legacy_passes_v10(tmp_path):
    env, step = sequence_fixture(tmp_path, forcing="ERA5 v043 E2W300I")
    # The old smoke skipped the schema call; the SAME bad payload's launcher
    # succeeds, reproducing the missed rejection before testing the new route.
    legacy_env = {**env, "WRF_GPU2_SEQ_SMOKE": "1", "WRF_GPU2_SMOKE_RECEIPT": str(tmp_path / "legacy_receipt.json")}
    assert subprocess.run(["bash", step["path"], *step["args"]], env=legacy_env).returncode == 0
    (tmp_path / "calls.jsonl").unlink()
    env["SEQ_SMOKE"] = "1"
    result = seq(tmp_path, env)
    assert result.returncode != 0
    assert calls(tmp_path, "registry")[0]["smoke"] == "1"
    assert not calls(tmp_path, "launcher")
    step["registry_smoke"][-1] = "ERA5"
    write(tmp_path / "plan.json", dict(steps=[step]))
    assert seq(tmp_path, env).returncode == 0


@pytest.mark.parametrize("changes,smoke", [(dict(launch_rc=7), False), (dict(launch_rc=7), True), (dict(), False)])
def test_step_rc_propagates_even_step_only(tmp_path, changes, smoke):
    env, _ = sequence_fixture(tmp_path, **changes)
    env["STEP_ONLY"] = "1"
    if smoke:
        env["SEQ_SMOKE"] = "1"
    result = seq(tmp_path, env)
    assert (result.returncode == 0) == (not changes)
    assert "END test rc=" + str(changes.get("launch_rc", 0)) in result.stdout
    if not changes:
        assert calls(tmp_path, "launcher")[0]["args"] == ["space arg", "$(touch BAD)", "'quoted'"]
        assert not (tmp_path / "BAD").exists()


def test_skipped_step_returns_nonzero(tmp_path):
    env, _ = sequence_fixture(tmp_path)
    env["HARD_END_UTC"] = stamp(START+100)
    result = seq(tmp_path, env)
    assert result.returncode != 0 and "SKIP test" in result.stdout


@pytest.mark.parametrize("override", [dict(completed_phases=["model"]), dict(real_lock_used=True), dict(device="gpu"), dict(production_registry_writes=1)])
def test_smoke_all_phases_private_proof_required(tmp_path, override):
    env, _ = sequence_fixture(tmp_path, receipt_override=override)
    env["SEQ_SMOKE"] = "1"
    assert seq(tmp_path, env).returncode != 0


@pytest.mark.parametrize("changes,expected", [
    (dict(acquire_fail=True), "NOSTART"),
    (dict(busy_cpu=True), "NOSTART"),
    (dict(acquire_advance=LATEST+1), "NOSTART"),
    (dict(release_fail=True), "STOPPED"),
    (dict(lock_after_preflight=True), "NOSTART"),
    (dict(unexpected_step=True), "STOPPED"),
    (dict(hang=True), "STOPPED"),
    (dict(expire_quiet=True), "NOSTART"),
    (dict(free_gib=420, watch=True, hang_notice=True), "STOPPED"),
])
def test_lease_and_admission_residuals(tmp_path, changes, expected):
    env, _ = fixture(tmp_path, **changes)
    rc, result = gate(tmp_path, env)
    assert rc != 0 and result["status"] == expected
    if expected == "NOSTART":
        assert not calls(tmp_path, "actual")
    if changes.get("release_fail"):
        assert result["cpu_lease_retained"] and len(calls(tmp_path, "release")) == 2
    else:
        assert not list((tmp_path / "leases").iterdir())


@pytest.mark.parametrize("phase,marker", [("signal_acquire", "acquiring"), ("signal_quiet", "sampling")])
def test_signal_during_admission_is_terminal_releases_owned_only(tmp_path, phase, marker):
    env, _ = fixture(tmp_path, **{phase: True})
    foreign = tmp_path / "leases/foreign-cpu.lease"
    foreign.write_text("foreign lease")
    out = (tmp_path / "output.log").open("w")
    p = subprocess.Popen(["bash", str(ROOT / "scripts/gpu_gate.sh"), str(tmp_path / "config.json")],
                         env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        until = time.monotonic()+8
        while not (tmp_path / marker).exists() and time.monotonic() < until:
            time.sleep(.01)
        assert (tmp_path / marker).exists()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=5) == 130
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        out.close()
    assert not calls(tmp_path, "actual")
    assert list((tmp_path / "leases").iterdir()) == [foreign]
    assert (tmp_path / "run/result.json").exists()


def test_quiet_admission_and_no_standing_cpu_lease(tmp_path):
    env, _ = fixture(tmp_path, trigger=False, trigger_after_sleep=True)
    rc, _ = gate(tmp_path, env)
    assert rc == 0
    rows = calls(tmp_path)
    modes = [r["mode"] for r in rows]
    assert modes.index("sleep") < modes.index("acquire") < modes.index("actual") < modes.index("release")
    assert any(r["mode"] == "sleep" and r["seconds"] == 60 for r in rows)


def test_thirty_second_cpu_pin_watcher_lead_required(tmp_path):
    env, _ = fixture(tmp_path, fresh_lease=True)
    rc, _ = gate(tmp_path, env)
    assert rc == 0
    rows = calls(tmp_path)
    actual = next(i for i, r in enumerate(rows) if r["mode"] == "actual")
    assert any(r["mode"] == "sleep" and r["seconds"] == 30 for r in rows[:actual])


def test_smoke_cannot_reuse_stale_success_receipt(tmp_path):
    env, _ = sequence_fixture(tmp_path)
    env["SEQ_SMOKE"] = "1"
    assert seq(tmp_path, env).returncode == 0
    write(tmp_path / "case.json", dict(omit_receipt=True))
    result = seq(tmp_path, env)
    assert result.returncode != 0 and "SMOKE REFUSE" in result.stdout


def test_byte_boundary_one_byte_below(tmp_path):
    env, cfg = fixture(tmp_path)
    cfg["disk_need_bytes"] = 65*GIB+1  # 430 - (65+epsilon) <365
    write(tmp_path / "config.json", cfg)
    rc, result = gate(tmp_path, env)
    assert rc == 4 and result["status"] == "NOSTART"


def test_child_import_resolution_required(tmp_path):
    env, step = sequence_fixture(tmp_path)
    (tmp_path / "src/gpuwrf/__init__.py").unlink()
    (tmp_path / "src/gpuwrf").rmdir()
    result = seq(tmp_path, env)
    assert result.returncode != 0 and "module resolution failed" in result.stderr
    assert not calls(tmp_path, "launcher")


@pytest.mark.parametrize("needle,replacement,changes", [
    ('require(free - c["disk_need_bytes"] >= c["disk_floor_bytes"], "disk byte floor")', 'pass', dict(free_gib=407.5)),
    ('rc == 0 and completion(output.read_text(), steps)', 'rc == 0', dict(one_rc7=True)),
])
def test_deleting_admission_or_completion_guard_is_detected(tmp_path, needle, replacement, changes):
    source = MODULE.read_text()
    assert source.count(needle) == 1
    mutant = tmp_path / "mutant.py"
    mutant.write_text(source.replace(needle, replacement))
    root = tmp_path / "case"
    root.mkdir()
    env, _ = fixture(root, **changes)
    rc, result = gate(root, env, mutant)
    # The ordinary negative control's assertion must fail on the mutant.
    with pytest.raises(AssertionError):
        assert rc != 0 and result["status"] != "DONE"


def finished_between_polls(module, monkeypatch, tmp_path):
    class Finished:
        returncode = 0
        def poll(self):
            return self.returncode
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: Finished())
    monkeypatch.setattr(module, "stop_group", lambda proc: None)
    monkeypatch.setattr(module, "now", lambda: END+1)
    return module.run(["private-stub"], {}, tmp_path / "finished.log", END)


def test_post_child_deadline_checks_exit_between_polls(tmp_path, monkeypatch):
    assert finished_between_polls(lib, monkeypatch, tmp_path) == 124


def test_finished_sequence_past_hard_end_is_stopped(tmp_path):
    env, _ = fixture(tmp_path, finish_past_end=True)
    rc, result = gate(tmp_path, env)
    assert rc != 0 and result["status"] == "STOPPED"
    assert not list((tmp_path / "leases").iterdir())


def test_finished_step_past_hard_end_is_nonzero(tmp_path):
    env, _ = sequence_fixture(tmp_path, finish_past_end=True)
    result = seq(tmp_path, env)
    assert result.returncode != 0 and "END test rc=124" in result.stdout


def terminal_release_cleanup(root, program=None):
    env, _ = fixture(root, signal_release=True)
    foreign = root / "leases/foreign-cpu.lease"
    foreign.write_text("foreign lease")
    command = [sys.executable, str(program), "gate"] if program else ["bash", str(ROOT / "scripts/gpu_gate.sh")]
    with (root / "output.log").open("w") as out:
        p = subprocess.Popen([*command, str(root / "config.json")], env=env, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            until = time.monotonic()+8
            while not (root / "releasing").exists() and time.monotonic() < until:
                time.sleep(.01)
            assert (root / "releasing").exists(), "release helper not reached"
            p.send_signal(signal.SIGTERM)
            rc = p.wait(timeout=5)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    assert rc == 130
    assert list((root / "leases").iterdir()) == [foreign], "owned lease retained after terminal signal"
    assert foreign.read_text() == "foreign lease"
    assert (root / "run/result.json").exists(), "terminal result missing after signal"
    result = json.loads((root / "run/result.json").read_text())
    assert result["status"] == "STOPPED" and not result["cpu_lease_retained"]
    assert (root / "cross/STOPPED.md").is_file() and not (root / "cross/DONE.md").exists()
    assert len(calls(root, "release")) == 2


def test_signal_during_terminal_release_retries_and_publishes(tmp_path):
    terminal_release_cleanup(tmp_path)


def test_deleting_post_child_deadline_guard_is_detected(tmp_path, monkeypatch):
    guard = "            if deadline is not None and now() >= deadline:\n                return 124\n"
    source = MODULE.read_text()
    assert source.count(guard) == 1
    path = tmp_path / "deadline_mutant.py"
    path.write_text(source.replace(guard, ""))
    spec = importlib.util.spec_from_file_location("deadline_mutant", path)
    mutant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutant)
    with pytest.raises(AssertionError):
        assert finished_between_polls(mutant, monkeypatch, tmp_path) == 124


def test_deleting_terminal_signal_deferral_is_detected(tmp_path):
    needle = "signal.signal(sig, defer)"
    source = MODULE.read_text()
    assert source.count(needle) == 1
    path = tmp_path / "release_mutant.py"
    path.write_text(source.replace(needle, "signal.getsignal(sig)"))
    case = tmp_path / "case"
    case.mkdir()
    with pytest.raises(AssertionError, match="owned lease retained|terminal result missing"):
        terminal_release_cleanup(case, path)
