"""Wave gates use analytic stub forecasts; registry uses real ALISIOS validation on PRIVATE stores only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gpuwrf.optimize import continuous
from gpuwrf.optimize.registry import Invocation, load_hooks
from gpuwrf.optimize.study import _atomic_json
from gpuwrf.optimize.waves import Waves, digest

ROOT = Path(__file__).resolve().parents[1]


def fixture_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = fixture_module("_v45_continuous_fixture", ROOT / "w41_optcont" / "test_continuous.py")
study_files = old.study_files
discrete = fixture_module("_v45_discrete_fixture", ROOT / "test_optimize.py")
discrete_study_files = discrete.study_files
profile_fixture = fixture_module("_v45_composed_profile_fixture", ROOT / "test_v45_diag_profiles.py")
profile_history = profile_fixture.history


def approve(output, **changes):
    """External test reviewer, never called by production code."""
    state = json.loads((output / "wave_state.json").read_text())
    p = state["pending"]
    token = {"schema": "gpuwrf.optimize.wave-go.v1", "decision": "GO", "run_id": state["run_id"],
             "study_sha256": state["study_sha256"], "wave": p["wave"], "report_sha256": p["report_sha256"],
             "checkpoint_sha256": p["checkpoint_sha256"], "approved_by": "review:unit-test", **changes}
    path = output / p["go_file"]
    path.write_text(json.dumps(token))
    return path


def test_eki_wave_stop_exact_checkpoint_and_single_use_go(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.surface_params import eki

    study = continuous.load_continuous(study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", old.make_fake_launch(calls))
    whole = continuous.run(study, tmp_path / "whole", wave_mode=False)
    out = tmp_path / "waves"
    updates, update = [], eki.update

    def observe(*a, **k):
        updates.append(k["seed"])
        return update(*a, **k)

    monkeypatch.setattr(eki, "update", observe)
    first = continuous.run(study, out)
    assert first["status"] == "PAUSED"
    assert len(updates) == 1
    report = json.loads((out / "waves/wave-0000.json").read_text())
    assert {"ensemble", "scores", "impact_matrix", "admissibility", "resources"} <= report.keys()
    assert len(report["ensemble"]) == 8 and len(report["scores"]) == 8
    assert report["admissibility"]["status"] == "NOT_EVALUATED"
    assert not (out / "result.json").exists()
    n = len(calls)
    with pytest.raises(ValueError, match="GO missing"):
        continuous.run(study, out, resume=True)
    assert len(calls) == n
    old_go = approve(out).read_text()
    second = continuous.run(study, out, resume=True)
    assert second["status"] == "PAUSED" and len(updates) == 2
    pending = json.loads((out / "wave_state.json").read_text())["pending"]
    (out / pending["go_file"]).write_text(old_go)
    with pytest.raises(ValueError, match="foreign"):
        continuous.run(study, out, resume=True)
    approve(out)
    assert continuous.run(study, out, resume=True)["status"] == "PAUSED"
    approve(out)
    done = continuous.run(study, out, resume=True)
    for key in ("estimate_multipliers", "evaluation", "iterations", "ensemble_final_multipliers"):
        assert done[key] == whole[key]
    assert len(updates) == 3  # no EKI step repeated on any resume
    state = json.loads((out / "wave_state.json").read_text())
    assert len(state["approvals"]) == 3 and state["pending"] is None
    count = len(calls)
    assert continuous.run(study, out, resume=True) == done
    assert len(calls) == count  # DONE replay is idempotent too


def test_resume_interrupted_authorized_wave_is_exact(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows

    study = continuous.load_continuous(study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", old.make_fake_launch(calls))
    out = tmp_path / "run"
    assert continuous.run(study, out)["status"] == "PAUSED"
    approve(out)
    monkeypatch.setattr(workflows, "launch", old.make_fake_launch(calls, interrupt=len(calls)))
    with pytest.raises(KeyboardInterrupt):
        continuous.run(study, out, resume=True)
    before = json.loads((out / "wave_state.json").read_text())
    assert before["pending"] is None and len(before["approvals"]) == 1
    monkeypatch.setattr(workflows, "launch", old.make_fake_launch(calls))
    assert continuous.run(study, out, resume=True)["status"] == "PAUSED"
    after = json.loads((out / "wave_state.json").read_text())
    assert after["checkpoint"]["iteration"] == 2
    assert len(after["approvals"]) == 1  # original authorization reused, not consumed twice


def test_discrete_rung_stops_and_resumes_with_identical_result(discrete_study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import load_study, run

    study = load_study(discrete_study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", discrete.make_fake_launch(calls))
    whole = run(study, tmp_path / "whole", wave_mode=False)
    out = tmp_path / "waves"
    report = run(study, out)
    assert report["status"] == "PAUSED"
    first = json.loads((out / "waves/wave-0000.json").read_text())
    assert first["kind"] == "rung" and first["scores"] == whole["rounds"][0]["scores"]
    for _ in study["rounds"]:
        approve(out)
        report = run(study, out, resume=True)
    assert report["rounds"] == whole["rounds"]
    assert report["held_out"] == whole["held_out"] and report["recommended"] == whole["recommended"]


@pytest.fixture
def boundary(tmp_path):
    from gpuwrf.optimize.study import Pause
    w = Waves(tmp_path, {"study": "A"}, resume=False, atomic_json=_atomic_json)
    with pytest.raises(Pause):
        w.complete(0, kind="eki", ensemble=[{"id": "a", "theta": [0.1]}, {"id": "b", "theta": [0.2]}],
                   scores=[], checkpoint={"theta": [[0.12345678901234567]]},
                   ledger={"members": {}, "scheduled_case_hours": 0})
    return tmp_path


@pytest.mark.parametrize("changes", [
    {"schema": "bad"}, {"decision": "INVALID"}, {"run_id": "foreign"}, {"study_sha256": "foreign"},
    {"wave": 1}, {"wave": False}, {"report_sha256": "foreign"}, {"checkpoint_sha256": "foreign"},
    {"approved_by": "optimizer"}, {"approved_by": []}, {"extra": "unsupported"},
])
def test_each_go_binding_refuses(boundary, changes):
    approve(boundary, **changes)
    with pytest.raises(ValueError, match="foreign"):
        Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert json.loads((boundary / "wave_state.json").read_text())["pending"] is not None


@pytest.mark.parametrize("artifact", ["report", "checkpoint", "missing_state", "symlink_go", "invalid_json"])
def test_missing_or_changed_review_artifacts_refuse(boundary, artifact):
    go = approve(boundary)
    state_path = boundary / "wave_state.json"
    state = json.loads(state_path.read_text())
    if artifact == "report":
        (boundary / state["pending"]["report"]).write_text('{}')
    elif artifact == "checkpoint":
        state["checkpoint"]["theta"][0][0] += 1
        state_path.write_text(json.dumps(state))
    elif artifact == "missing_state":
        state_path.unlink()
    elif artifact == "symlink_go":
        target = boundary / "external.json"
        go.rename(target)
        go.symlink_to(target)
    else:
        go.write_text("no JSON")
    with pytest.raises((ValueError, OSError)):
        Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)


def test_consumption_commits_before_work_and_survives_restart(boundary):
    approve(boundary)
    w = Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert len(w.state["approvals"]) == 1 and w.state["pending"] is None
    # Simulated crash before the first member of the authorized wave: no second GO needed.
    again = Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    np.testing.assert_array_equal(again.checkpoint["theta"], [[0.12345678901234567]])
    assert len(again.state["approvals"]) == 1


def assess(output, scores, *, status="EVALUATED", objective="a"*64, decision="GO"):
    state = json.loads((output / "wave_state.json").read_text())
    p = state["pending"]
    scores = [{"status": "ADMISSIBLE" if row.get("admissible") else "HARM_VETO", **row} for row in scores]
    review = {"schema": "gpuwrf.optimize.wave-assessment.v1", "run_id": state["run_id"], "wave": p["wave"],
              "report_sha256": p["report_sha256"],
              "assessment": {"schema": "gpuwrf.optimize.multiobjective.v1", "objective_identity": objective,
                             "status": status, "scores": scores, "impact_matrix": []}}
    (output / f"waves/wave-{p['wave']:04d}.review.json").write_text(json.dumps(review))
    approve(output, assessment_sha256=digest(review), decision=decision)


def test_harm_veto_best_persists_and_stop_is_idempotent(boundary):
    # The lower scalar loss is harmful: authority's admissible flag vetoes it.
    assess(boundary, [{"id": "a", "loss": 0.0, "admissible": False},
                      {"id": "b", "loss": 2.0, "admissible": True}], decision="STOP")
    w = Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    best = json.loads((boundary / "best.json").read_text())
    assert best["candidate"]["id"] == "b" and w.state["stopped"] is True
    again = Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert again.state == w.state
    assert json.loads((boundary / "best.json").read_text()) == best


def test_missing_truth_never_promotes_a_candidate(boundary):
    assess(boundary, [{"id": "a", "admissible": False, "status": "UNAVAILABLE"}], status="UNAVAILABLE")
    Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert json.loads((boundary / "best.json").read_text())["status"] == "NO_ADMISSIBLE_SOLUTION"


@pytest.mark.parametrize("row", [{"id": "a", "admissible": True, "loss": 0.0},
                                 {"id": "foreign", "admissible": False}, {"id": "a", "loss": 0.0}])
def test_unavailable_or_foreign_assessment_refuses(boundary, row):
    assess(boundary, [row], status="UNAVAILABLE")
    with pytest.raises(ValueError):
        Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert json.loads((boundary / "best.json").read_text())["candidate"] is None


def test_contradictory_harm_status_cannot_win(boundary):
    assess(boundary, [{"id": "a", "admissible": True, "loss": 0.0, "status": "HARM_VETO"}])
    with pytest.raises(ValueError, match="contradicts"):
        Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert json.loads((boundary / "best.json").read_text())["candidate"] is None


def test_changed_objective_denominator_refuses_and_preserves_best(boundary):
    from gpuwrf.optimize.study import Pause
    assess(boundary, [{"id": "a", "admissible": True, "loss": 1.0}])
    w = Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    before = (boundary / "best.json").read_bytes()
    with pytest.raises(Pause):
        w.complete(1, kind="eki", ensemble=["a"], scores=[], checkpoint={"iteration": 2},
                   ledger={"members": {}, "scheduled_case_hours": 0})
    assess(boundary, [{"id": "a", "admissible": True, "loss": 0.0}], objective="b"*64)
    with pytest.raises(ValueError, match="denominator"):
        Waves(boundary, {"study": "A"}, resume=True, atomic_json=_atomic_json)
    assert (boundary / "best.json").read_bytes() == before


def test_profile_caller_uses_flat_held_histories_once(tmp_path, monkeypatch):
    import types
    from gpuwrf.optimize.study import Pause
    calls = []
    fake = types.ModuleType("gpuwrf.optimize.diagnostics")
    def write(paths, points, output):
        calls.append((paths, points))
        output.write_text('{"status":"UNAVAILABLE","reason":"test has no native taps"}')
    fake.write_profile_report = write
    fake.wave_report_entry = lambda path: {"path": str(path), "sha256": digest(json.loads(path.read_text())),
                                          "status": "UNAVAILABLE"}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    histories = tmp_path / "histories"
    histories.mkdir()
    path = histories / "wrfout_d03_2026-01-01_01:00:00"
    path.write_bytes(b"analytic stub, not WRF")
    points = [{"id": "north", "region": "north_windward", "domain": "d03", "j": 0, "i": 0}]
    ledger = {"members": {"a|case": {"files": [{"path": str(path)}]}}, "scheduled_case_hours": 3}
    waves = Waves(tmp_path, {}, resume=False, atomic_json=_atomic_json, profile_points=points)
    with pytest.raises(Pause):
        waves.complete(0, kind="rung", ensemble=["a"], scores=[], checkpoint={}, ledger=ledger)
    report = json.loads((tmp_path / "waves/wave-0000.json").read_text())
    assert report["profiles"]["members"]["a|case"]["status"] == "UNAVAILABLE"
    assert calls == [([path], points)]
    approve(tmp_path)
    waves = Waves(tmp_path, {}, resume=True, atomic_json=_atomic_json, profile_points=points)
    waves.complete(0, kind="rung", ensemble=["a"], scores=[], checkpoint={}, ledger=ledger)
    assert len(calls) == 1


def test_retry_and_case_hour_caps_never_launch_again(discrete_study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import load_study, run
    study = load_study(discrete_study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", lambda *a, **k: calls.append(k) or 7)
    first_cold_hours = len(study["candidates"])*study["hours"]
    for name, retries, budget in (("retry", 0, study["budget_case_hours"]), ("casehours", 1, first_cold_hours)):
        arm = {**study, "budget_case_hours": budget}
        out = tmp_path / name
        with pytest.raises(ValueError, match="ensemble batch failed"):
            run(arm, out, max_retries=retries)
        n = len(calls)
        if retries == 0:
            try:
                result = run(arm, out, resume=True, max_retries=retries)
            except ValueError as exc:
                raise AssertionError("retry cap allowed another failed model launch") from exc
            assert result["status"] == "PAUSED" and "retry budget exhausted" in result["reason"]
        else:
            with pytest.raises(ValueError, match="compute budget exhausted"):
                run(arm, out, resume=True, max_retries=retries)
        assert len(calls) == n
        assert json.loads((out / "best.json").read_text())["candidate"] is None


def test_wallclock_cap_interrupts_host_work(discrete_study_files, tmp_path):
    import time
    from gpuwrf.optimize.study import load_study, run
    study = load_study(discrete_study_files[0])
    def slow(**kwargs):
        time.sleep(2)
        raise AssertionError("hard wall timer did not interrupt host work")
    result = run(study, tmp_path / "wall", driver=slow, wall_seconds=0.05)
    assert result["status"] == "PAUSED" and "hard wallclock" in result["reason"]


def test_concurrent_resume_cannot_double_start(discrete_study_files, tmp_path, monkeypatch):
    import fcntl
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import load_study, run
    study = load_study(discrete_study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", lambda *a, **k: calls.append(k) or 0)
    out = tmp_path / "locked"
    with out.with_name(out.name+".optimizer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="concurrent resume"):
            run(study, out)
    assert not out.exists()
    assert calls == []


def test_public_run_stop_and_best_cli_never_start_work(discrete_study_files, tmp_path, capsys):
    from gpuwrf.cli import main
    from gpuwrf.optimize.study import load_study, run
    study = load_study(discrete_study_files[0])
    out = tmp_path / "stopped"
    def boundary_driver(*, waves, ledger, **kwargs):
        waves.complete(0, kind="rung", ensemble=["a"], scores=[], checkpoint={}, ledger=ledger)
    assert run(study, out, driver=boundary_driver)["status"] == "PAUSED"
    approve(out, decision="STOP")
    def forbidden(**kwargs):
        raise AssertionError("STOP must refuse further driver/model work")
    assert run(study, out, driver=forbidden, resume=True)["status"] == "STOPPED"
    assert main(["optimize", "best", "--output", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NO_ADMISSIBLE_SOLUTION"


@pytest.mark.parametrize("capacity", [False, True])
def test_dense_cloud_native_refused_before_model_work(study_files, monkeypatch, capacity):
    from gpuwrf.surface_params import eki
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", capacity)
    study = continuous.load_continuous(study_files[0])
    study["scorer"]["kind"] = "cloud"
    study["scorer"]["observation_operator"] = "native"
    message = "ALISIOS correlation contract" if capacity else "full-native.*refused"
    with pytest.raises(ValueError, match=message):
        continuous.driver(study)


def test_dense_observation_cap_before_covariance(study_files, monkeypatch):
    from gpuwrf.surface_params import eki
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", False)
    study = continuous.load_continuous(study_files[0])
    obs, sigma = np.zeros(4097), np.ones(4097)
    with pytest.raises(ValueError, match="4096 observations before covariance"):
        continuous._problem(study, obs, sigma)
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", True, raising=False)
    # Capability handshake only: no dense gamma/covariance materialized here.
    assert continuous._problem(study, obs, sigma).obs.shape == (4097,)


def test_merged_ensemble_space_capacity_does_not_admit_native_correlations(study_files):
    from gpuwrf.surface_params import eki
    assert eki.ENSEMBLE_SPACE_UPDATE is True
    study = continuous.load_continuous(study_files[0])
    study["scorer"]["kind"] = "cloud"
    study["scorer"]["observation_operator"] = "native"
    study["scorer"]["cloud"] = {"superobs": {"correlation_length_km": 10., "contract": "BARE_LABEL_NOT_NATIVE_GAMMA"}}
    with pytest.raises(ValueError, match="ALISIOS correlation contract"):
        continuous.driver(study)


def test_merged_ensemble_space_large_vectors_with_stub_admitted_contract(study_files, monkeypatch):
    from gpuwrf.optimize import cloud
    from gpuwrf.surface_params import eki
    assert eki.ENSEMBLE_SPACE_UPDATE is True
    study = continuous.load_continuous(study_files[0])
    contract = {"id": "SYNTHETIC_INDEPENDENT_NATIVE_ERRORS", "admitted_by": "review:unit-test",
                "covariance": "Gamma=I for these synthetic errors only"}
    study["scorer"].update(kind="cloud", observation_operator="native", cloud={"stub_admitted_contract": contract})
    admissions = []

    def admit_stub(config, operator):
        assert operator == "native" and config == {"stub_admitted_contract": contract}
        admissions.append(contract["id"])

    # A test-only admission dependency; no production native contract is invented.
    monkeypatch.setattr(cloud, "require_correlation_contract", admit_stub)
    assert callable(continuous.driver(study))
    assert admissions == [contract["id"]]
    m = 25200
    problem = continuous._problem(study, np.zeros(m), np.ones(m))
    assert problem.obs_cov.shape == (m,)
    state = eki.initial_ensemble(problem, 4, seed=7)
    predictions = state.theta[:, :1] + np.linspace(-0.1, 0.1, m)[None, :]
    first = eki.update(problem, state, predictions, dt=0.5, seed=19)
    replay = eki.update(problem, state, predictions, dt=0.5, seed=19)
    assert first.iteration == 1 and np.isfinite(first.theta).all()
    np.testing.assert_array_equal(first.theta, replay.theta)


def test_wave_hook_uses_merged_profile_reader(profile_history, tmp_path):
    import hashlib
    from gpuwrf.optimize.study import Pause
    before = hashlib.sha256(profile_history.read_bytes()).hexdigest()
    output = tmp_path / "composed"
    output.mkdir()
    ledger = {"members": {"member|case": {"files": [{"path": str(profile_history)}]}}, "scheduled_case_hours": 3}
    waves = Waves(output, {}, resume=False, atomic_json=_atomic_json, profile_points=profile_fixture.POINTS)
    with pytest.raises(Pause):
        waves.complete(0, kind="rung", ensemble=["member"], scores=[], checkpoint={}, ledger=ledger)
    report = json.loads((output / "waves/wave-0000.json").read_text())
    entry = report["profiles"]["members"]["member|case"]
    assert entry["status"] == "COMPLETE_WITH_UNAVAILABLE_TERMS"
    assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert hashlib.sha256(profile_history.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("mutation", ["legacy-unrecorded", "solver-source", "numpy-version"])
def test_resume_refuses_changed_optimizer_implementation(study_files, tmp_path, monkeypatch, mutation):
    from gpuwrf.optimize.study import Pause
    study = continuous.load_continuous(study_files[0])
    calls = []
    def driver(bound):
        def pause(**kwargs):
            calls.append(bound)
            raise Pause("private checkpoint setup; no model run")
        return pause
    monkeypatch.setattr(continuous, "driver", driver)
    output = tmp_path / "versioned"
    assert continuous.run(study, output)["status"] == "PAUSED"
    saved = json.loads((output / "study.json").read_text())
    assert "optimizer_implementation" in saved
    assert saved["optimizer_implementation"]["source_sha256"]
    assert saved["optimizer_implementation"]["numpy_version"] == np.__version__
    assert "optimizer_implementation" not in study  # caller's validated study is unchanged
    if mutation == "legacy-unrecorded":
        saved.pop("optimizer_implementation")
    elif mutation == "solver-source":
        saved["optimizer_implementation"]["source_sha256"] = "0"*64
    else:
        saved["optimizer_implementation"]["numpy_version"] = "different-version"
    (output / "study.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="EKI implementation/NumPy version changed or was not recorded"):
        continuous.run(study, output, resume=True)
    assert len(calls) == 1


def test_actual_member_bookkeeping_does_not_fragment_trace_key(discrete_study_files, tmp_path, monkeypatch):
    import os
    from unittest.mock import patch
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import load_study, run
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
    study = load_study(discrete_study_files[0])
    keys = []
    def launch(*args, env, **kwargs):
        names = {k for k in env if k.startswith(("ALISIOS_OPTIMIZE_", "GPUWRF_OPTIMIZE_"))}
        assert names == {"ALISIOS_OPTIMIZE_REGISTRY", "ALISIOS_OPTIMIZE_WALL_DEADLINE"}
        # Only hashes and the two known bookkeeping names can reach assertion output.
        with patch.dict(os.environ, env, clear=True):
            keys.append(global_trace_env_hash())
            os.environ["ALISIOS_OPTIMIZE_REGISTRY"] = str(tmp_path / "another-batch/registry.json")
            os.environ["ALISIOS_OPTIMIZE_WALL_DEADLINE"] = "123456789.0"
            keys.append(global_trace_env_hash())
            # Positive control: the old prefix really would change the product key.
            os.environ["GPUWRF_OPTIMIZE_REGISTRY"] = "old-batch-a"
            old_a = global_trace_env_hash()
            os.environ["GPUWRF_OPTIMIZE_REGISTRY"] = "old-batch-b"
            assert global_trace_env_hash() != old_a
        return 7  # no model/subprocess/production registration
    monkeypatch.setattr(workflows, "launch", launch)
    with pytest.raises(ValueError, match="ensemble batch failed"):
        run(study, tmp_path / "keys")
    assert len(keys) == 2 and keys[0] == keys[1]


@pytest.fixture
def private_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(load_hooks(), "PRODUCTION_ATLAS", tmp_path / "private-production-sentinel")
    atlas = tmp_path / "private-atlas"
    # Empty private atlas = no production metadata reads, hashes or rows.
    data = {"schema": "gpuwrf.optimize.registry.v1", "policy": {"atlas_dir": str(atlas), "smoke": True,
            "forcing": "OTHER"}, "purpose": "Private wave unit test, NO_SCORE",
            "members": {str(tmp_path / "case"): {"issue": "20260101_00z", "lead_h": 3, "nests": "1 domain"}}}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    return path, tmp_path / "case", atlas


@pytest.mark.parametrize("rc,aborted,status", [(0, False, "DONE_HELD"), (7, False, "FAILED"),
                                               (-15, False, "ABORTED"), (1, True, "ABORTED")])
def test_real_registry_success_crash_abort_and_no_duplicate_end(private_registry, tmp_path, rc, aborted, status):
    path, case, atlas = private_registry
    run = Invocation(path, case, tmp_path / "out")
    env = run.begin({"JAX_PLATFORMS": "cpu"})
    events = run.registry._events()
    assert len(events) == 1 and events[0]["event"] == "start"
    assert env["ALISIOS_WRF_RUN_ID"] == events[0]["run_id"]
    assert Path(env["ALISIOS_ATLAS_DIR"]) == atlas
    run.finish(rc, aborted=aborted)
    run.finish(rc)  # finally after normal poll is a no-op
    events = run.registry._events()
    assert [e["event"] for e in events] == ["start", "end"]
    assert events[-1]["status"] == status
    assert events[-1]["raw_deleted"] is False  # actual optimizer histories are retained for scoring


def test_registry_precheck_refuses_before_run(private_registry, tmp_path):
    path, case, _ = private_registry
    data = json.loads(path.read_text())
    data["policy"]["forcing"] = "invented-forcing"
    path.write_text(json.dumps(data))
    run = Invocation(path, case, tmp_path / "out")
    with pytest.raises(load_hooks().RegistryError):
        run.begin({})
    assert run.registry._events() == []


def test_terminal_failure_cannot_be_salvaged_as_success(tmp_path):
    from gpuwrf.optimize.study import _member_entry
    out = tmp_path / "batch"
    histories = out / "runs/m/wrfout"
    histories.mkdir(parents=True)
    (histories / "wrfout_d01_2026-01-01_01:00:00").write_bytes(b"complete analytic stub")
    (out / "plan.json").write_text("{}")
    (out / "runs/m/receipt.json").write_text(json.dumps({"rc": 0, "registry_run_id": "original:run",
                                                       "registry_complete": False}))
    with pytest.raises(ValueError, match="reconcile the original.*do not rerun"):
        _member_entry(out, {"out_root": str(out / "runs")}, {"id": "m"}, "candidate", {"domain": "d01", "id": "case"})


def test_atomic_checkpoint_is_durable_before_publication(tmp_path, monkeypatch):
    import os
    calls = []
    fsync, replace = os.fsync, os.replace
    def sync(fd):
        calls.append("fsync")
        return fsync(fd)
    def publish(*args):
        calls.append("replace")
        return replace(*args)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", publish)
    path = tmp_path / "checkpoint.json"
    _atomic_json(path, {"theta": [0.12345678901234567]})
    assert calls == ["fsync", "replace", "fsync"]
    assert json.loads(path.read_text())["theta"] == [0.12345678901234567]


@pytest.mark.parametrize("kind,status", [("success", "DONE_HELD"), ("crash", "FAILED"),
                                         ("abort", "ABORTED"), ("spawn_error", "FAILED"), ("expired", None)])
def test_scheduler_registers_real_process_attempts(private_registry, tmp_path, monkeypatch, kind, status):
    pc = fixture_module("_v45_parallel_cases", ROOT.parent / "scripts/parallel_cases.py")
    path, case, atlas = private_registry
    case.mkdir()
    cli = tmp_path / "stub.py"
    cli.write_text("import os,json\nprint(json.dumps({'effective_hours':3}))\n"
                   "assert os.environ['ALISIOS_WRF_RUN_ID'].startswith('optimize:')\n"
                   "from pathlib import Path\n"
                   "rows=[json.loads(x) for x in (Path(os.environ['ALISIOS_ATLAS_DIR'])/'WRF_RUN_REGISTRY.jsonl').read_text().splitlines()]\n"
                   "assert rows[-1]['event']=='start' and rows[-1]['run_id']==os.environ['ALISIOS_WRF_RUN_ID']\n" +
                   {"success": "", "crash": "raise RuntimeError('stub crash')\n",
                    "abort": "import signal,time;os.kill(os.getppid(),signal.SIGTERM);time.sleep(2)\n",
                    "spawn_error": "", "expired": ""}[kind])
    monkeypatch.setenv("ALISIOS_OPTIMIZE_REGISTRY", str(path))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    if kind == "expired":
        monkeypatch.setenv("ALISIOS_OPTIMIZE_WALL_DEADLINE", "0")
    args = ["--out-root", str(tmp_path / "scheduler"), "--pool-gib", "1", "--host-gb-per-case", "0.001",
            "--gpu-log-interval", "0", "--min-mem-available-gb", "0", "--cli-json",
            json.dumps([sys.executable, str(cli)]), str(case)]
    if kind == "spawn_error":
        original_popen = pc.subprocess.Popen
        def refuse(argv, *a, **k):
            if len(argv) > 1 and argv[1] == str(cli):
                raise OSError("cannot spawn")
            return original_popen(argv, *a, **k)
        monkeypatch.setattr(pc.subprocess, "Popen", refuse)
        with pytest.raises(OSError, match="cannot spawn"):
            pc.main(args, gpu=lambda: (32*2**30, 32*2**30), vram=lambda: {})
    else:
        assert pc.main(args, gpu=lambda: (32*2**30, 32*2**30), vram=lambda: {}) == (0 if kind == "success" else 1)
    if kind == "expired":
        assert not (atlas / "WRF_RUN_REGISTRY.jsonl").exists()
        return
    assert (atlas / "WRF_RUN_REGISTRY.jsonl").is_file(), "member start registry missing"
    rows = [json.loads(line) for line in (atlas / "WRF_RUN_REGISTRY.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "end"]
    assert rows[-1]["status"] == status
    receipt = json.loads((Path(rows[0]["run_dir"]) / "receipt.json").read_text())
    assert receipt["registry_complete"] is True
