"""Host-only admission tests for the pre-registered diagnostic launcher."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rain_launch", ROOT / "scripts/rw3_rainfloor/launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_window_requires_both_bounds_and_full_cap(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setenv("GA_START_UTC", (now - timedelta(minutes=1)).isoformat())
    monkeypatch.setenv("GA_END_UTC", (now + timedelta(minutes=15)).isoformat())
    launch.window(720)
    monkeypatch.setenv("GA_START_UTC", (now + timedelta(minutes=1)).isoformat())
    with pytest.raises(ValueError, match="outside booked"):
        launch.window(720)
    monkeypatch.setenv("GA_START_UTC", (now - timedelta(minutes=1)).isoformat())
    monkeypatch.setenv("GA_END_UTC", (now + timedelta(minutes=10)).isoformat())
    with pytest.raises(ValueError, match="insufficient cap"):
        launch.window(720)


def test_no_booking_cannot_launch_gpu(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch.py", "NO_QSQ", "warm"])
    monkeypatch.setattr(launch, "verify_sources", lambda arm: None)
    monkeypatch.setattr(launch, "window", lambda cap: None)
    monkeypatch.delenv("WRF_GPU2_SEQ_SMOKE", raising=False)
    monkeypatch.delenv("V45_RAINFLOOR_BOOKING", raising=False)
    with pytest.raises(KeyError, match="V45_RAINFLOOR_BOOKING"):
        launch.main()


def test_registered_revert_patches_and_deletion_sensitive_pin(monkeypatch):
    # Immutable archive tests use the original shared object store read-only;
    # this performs no checkout, index mutation, model import or GPU access.
    for arm in json.loads((launch.HERE / "SOURCE_VARIANTS.json").read_text())["arms"]:
        launch.verify_sources(arm)
        mutated = {**arm, "src_tree": "0" * 40}
        with pytest.raises(ValueError, match="source tree mismatch"):
            launch.verify_sources(mutated)


def test_smoke_preserves_real_duration_argv(tmp_path):
    normal = launch.argv(ROOT, tmp_path, tmp_path, 72, False)
    smoke = launch.argv(ROOT, tmp_path, tmp_path, 72, True)
    assert smoke == normal + ["--dry-run"]
    assert normal[-2:] == ["--hours", "72"]
    assert "--domains-from-namelist" in normal and "--emit-initial-history" in normal


@pytest.mark.parametrize("mutation", [{"E41": "UNKNOWN"}, {"E41": "FAIL"},
                                      {"platform": "cpu"}, {"source_pin": "wrong"}])
def test_warm_gate_requires_actual_gpu_pass_on_exact_pin(tmp_path, mutation):
    p = tmp_path / "COLD_GATE_PASS.json"
    receipt = dict(source_pin="expected", E41="PASS", platform="gpu")
    p.write_text(json.dumps(receipt))
    launch.check_cold_gate(p, "expected")
    p.write_text(json.dumps({**receipt, **mutation}))
    with pytest.raises(ValueError, match="pin-bound"):
        launch.check_cold_gate(p, "expected")
