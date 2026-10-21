"""Host suspend detection in run receipts (ALISIOS robustness item 3, lane w43-robust); CPU-only, model stubbed."""
import json
from pathlib import Path

import pytest
from gpuwrf import cli
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import host_suspend


class _Clock:
    """CLOCK_BOOTTIME − CLOCK_MONOTONIC offset under test control."""

    def __init__(self, offset=1000.0):
        self.offset = offset

    def __call__(self):
        return self.offset


def test_offset_growth_is_the_suspended_time(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(host_suspend, "boottime_offset_s", clock)
    watch = host_suspend.SuspendWatch()
    clock.offset += 0.2                                   # clock noise, no suspend
    assert watch.summary() == {"suspended_s": 0.2, "timing_valid": True, "clock": "CLOCK_BOOTTIME-CLOCK_MONOTONIC"}
    assert watch.failure_note(checkpoint_dir="/ck") == ""
    clock.offset += 21300.0                               # 5 h 55 min hibernation
    summary = watch.summary()
    assert summary["suspended_s"] == pytest.approx(21300.2) and summary["timing_valid"] is False
    note = watch.failure_note(checkpoint_dir="/ck")
    assert "suspended for 21300 s" in note and "--resume-checkpoint" in note and "/ck" in note
    assert "--checkpoint-dir" in watch.failure_note()


def test_unavailable_clock_reports_unknown(monkeypatch):
    monkeypatch.setattr(host_suspend, "boottime_offset_s", lambda: None)
    watch = host_suspend.SuspendWatch()
    assert watch.summary() == {"suspended_s": None, "timing_valid": None, "clock": "unavailable"}
    assert watch.failure_note() == ""


def test_real_clocks_report_no_suspend_now():
    watch = host_suspend.SuspendWatch()
    suspended = watch.suspended_s()
    assert suspended is not None and 0.0 <= suspended < host_suspend.SUSPEND_THRESHOLD_S


class _RawClocks:
    """Stand-in for the ``time`` module: CLOCK_BOOTTIME and CLOCK_MONOTONIC under test control."""
    CLOCK_BOOTTIME, CLOCK_MONOTONIC = 7, 1

    def __init__(self):
        self.boot, self.mono = 5000.0, 4000.0

    def clock_gettime(self, clock):
        return {self.CLOCK_BOOTTIME: self.boot, self.CLOCK_MONOTONIC: self.mono}[clock]

    def run(self, seconds):                               # awake time advances both clocks
        self.boot += seconds
        self.mono += seconds

    def sleep_host(self, seconds):                        # suspend advances only CLOCK_BOOTTIME
        self.boot += seconds


def test_detector_reads_boottime_minus_monotonic(monkeypatch):
    clocks = _RawClocks()
    monkeypatch.setattr(host_suspend, "time", clocks)
    assert host_suspend.boottime_offset_s() == 1000.0
    watch = host_suspend.SuspendWatch()
    clocks.run(900.0)                                     # a long awake run is not a suspend
    assert watch.suspended_s() == 0.0 and watch.summary()["timing_valid"] is True
    clocks.sleep_host(3599.0)
    clocks.run(30.0)
    assert watch.suspended_s() == 3599.0 and watch.summary()["timing_valid"] is False


class _NoBoottime:
    CLOCK_MONOTONIC = 1

    @staticmethod
    def clock_gettime(clock):
        return 1.0


class _BrokenClocks(_RawClocks):
    def clock_gettime(self, clock):
        raise OSError("clock not supported")


@pytest.mark.parametrize("fake", [_NoBoottime(), _BrokenClocks()])
def test_detector_without_boottime_is_unknown(monkeypatch, fake):
    monkeypatch.setattr(host_suspend, "time", fake)
    assert host_suspend.boottime_offset_s() is None
    assert host_suspend.SuspendWatch().summary()["timing_valid"] is None


def _nested_args(tmp_path):
    fixture = Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist"
    (tmp_path / "namelist.input").write_bytes(fixture.read_bytes())
    return ["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"), "--proof-dir", str(tmp_path / "proof"),
            "--domains-from-namelist", "--hours", "3", "--force-gpu-run"]


def test_nested_receipt_records_a_suspend_during_the_run(tmp_path, monkeypatch, capsys):
    clock = _Clock()
    monkeypatch.setattr(host_suspend, "boottime_offset_s", clock)

    def execute(config):
        clock.offset += 600.0                             # host sleeps while the forecast runs
        return {"verdict": "PIPELINE_GREEN", "metadata": {}}

    monkeypatch.setattr(pipeline, "execute_nested_pipeline", execute)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert cli.main(_nested_args(tmp_path)) == 0
    receipt = json.loads((tmp_path / "proof" / "nested_pipeline_run.json").read_text())
    assert receipt["host_suspend"]["suspended_s"] == pytest.approx(600.0)
    assert receipt["host_suspend"]["timing_valid"] is False
    assert "host was suspended for 600 s" in capsys.readouterr().err


def test_nested_receipt_without_suspend_is_timing_valid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(host_suspend, "boottime_offset_s", _Clock())
    monkeypatch.setattr(pipeline, "execute_nested_pipeline", lambda config: {"verdict": "PIPELINE_GREEN", "metadata": {}})
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert cli.main(_nested_args(tmp_path)) == 0
    receipt = json.loads((tmp_path / "proof" / "nested_pipeline_run.json").read_text())
    assert receipt["host_suspend"] == {"suspended_s": 0.0, "timing_valid": True, "clock": "CLOCK_BOOTTIME-CLOCK_MONOTONIC"}
    assert "suspended" not in capsys.readouterr().err


def test_nested_failure_after_a_suspend_names_it(tmp_path, monkeypatch, capsys):
    clock = _Clock()
    monkeypatch.setattr(host_suspend, "boottime_offset_s", clock)

    def execute(config):
        clock.offset += 300.0
        raise RuntimeError("CUDA_ERROR_ILLEGAL_STATE")      # context lost across the suspend

    monkeypatch.setattr(pipeline, "execute_nested_pipeline", execute)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    args = _nested_args(tmp_path) + ["--checkpoint-dir", str(tmp_path / "ck"), "--checkpoint-interval-steps", "10"]
    assert cli.main(args) == 1
    err = capsys.readouterr().err
    assert "nested forecast failed (RuntimeError: CUDA_ERROR_ILLEGAL_STATE)" in err
    assert "host was suspended for 300 s" in err and "--resume-checkpoint" in err and str(tmp_path / "ck") in err


def test_single_domain_replay_payload_records_a_suspend(tmp_path, monkeypatch, capsys):
    from gpuwrf.integration import daily_pipeline as daily
    source = Path(__file__).resolve().parents[1] / "examples/switzerland_d01"
    (tmp_path / "namelist.input").write_bytes((source / "namelist.input").read_bytes())
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        (tmp_path / name).symlink_to(source / name)
    for hour in (0, 1):                                   # replay detection reads filenames only
        (tmp_path / f"wrfout_d01_2023-01-15_{hour:02d}:00:00").touch()
    clock = _Clock()
    monkeypatch.setattr(host_suspend, "boottime_offset_s", clock)

    def execute(config):
        clock.offset += 120.0
        return {"verdict": "PIPELINE_GREEN", "wrfout_files": []}

    monkeypatch.setattr(daily, "execute_daily_pipeline", execute)
    monkeypatch.setattr(pipeline, "execute_nested_pipeline", lambda config: pytest.fail("replay selected native"))
    assert cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                     "--domain", "d01", "--hours", "1", "--scratch-dir", str(tmp_path / "scratch")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["init_mode"] == "cpu_wrf_replay"
    assert payload["host_suspend"]["suspended_s"] == pytest.approx(120.0) and payload["host_suspend"]["timing_valid"] is False
