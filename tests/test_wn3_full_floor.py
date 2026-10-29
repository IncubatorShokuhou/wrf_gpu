"""Archive identity and fail-closed RW3 gate tests; no model fidelity claims."""
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor = load("wn3_floor")
scorer = load("wn3_score")


def hermetic_cli(tmp_path):
    """Test adapter only: real scorer/parser/math with private provenance bytes.

    The scorer hashes (but never parses) its twin-threshold file. This fixture
    copies the archived provenance object from the committed W42 summary; it
    contains no invented scientific thresholds. No production option changes.
    """
    raw = (ROOT / "tests/fixtures/wn3_twin_threshold_metadata.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "fbea7fc40c5ecf587a979b20aec754318705a6fc379edbf3e363742c9f510894"
    private = tmp_path / "twin_threshold_metadata.json"
    private.write_bytes(raw)
    attempted = tmp_path / "protected_attempt.txt"
    standin = tmp_path / "protected_standin.json"
    standin.write_text("private sentinel control, never a live ALISIOS file\n")
    bootstrap = tmp_path / "private_scorer_cli.py"
    bootstrap.write_text('''import importlib.util, os, sys
from pathlib import Path
script, metadata, marker, standin, mode = sys.argv[1:6]
protected = ("<DATA_ROOT>/alisios", "<DATA_ROOT>/server/work/src/alisios", "<USER_HOME>/src/alisios", standin)
def guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.abspath(os.fsdecode(args[0]))
    if any(path == root or path.startswith(root + "/") for root in protected):
        Path(marker).write_text(path + "\\n")
        # AssertionError is deliberate: the scorer catches OSError for missing
        # provenance, which must never hide a protected-data attempt here.
        raise AssertionError("protected data access before open: " + path)
sys.addaudithook(guard)
if mode == "sentinel-control":
    Path(standin).read_bytes()
    raise AssertionError("sentinel did not stop the private negative control")
spec = importlib.util.spec_from_file_location("private_wn3_score", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TWIN_THRESHOLDS = Path(metadata)
sys.argv = [script, *sys.argv[6:]]
module.main()
''')
    base = [sys.executable, str(bootstrap), str(ROOT / "scripts/wn3_score.py"),
            str(private), str(attempted), str(standin)]
    probe = subprocess.run(base + ["sentinel-control"], capture_output=True, text=True, timeout=30)
    assert probe.returncode != 0 and "protected data access before open" in probe.stderr
    assert attempted.read_text().strip() == str(standin)
    attempted.unlink()
    return base + ["score"], attempted, {"path": str(private), "sha256": hashlib.sha256(raw).hexdigest()}


def summary():
    return dict(pass_=False, domains={d: dict(
        pairing=dict(paired_file_count=73, common_leads_h=list(range(73)),
                     init_time_utc="2026-02-28T00:00:00+00:00", unmatched_cpu_count=0, unmatched_gpu_count=0),
        missing_required_fields={}, gpu_only_nonfinite=[], policy_failures=[],
        **{"pass": True, "summaries": {"verdict": "PASS"}}) for d in floor.DOMAINS},
        output_integrity_pass=True)


def breach(s, field="RAINNC", lead=67, x=1.0600553865983122, limit=1.0, metric="rmse"):
    s.pop("pass_", None)
    s["pass"] = False
    s["domains"]["d03"]["pass"] = False
    s["domains"]["d03"]["summaries"]["verdict"] = "FAIL"
    s["domains"]["d03"]["policy_failures"].append(dict(
        kind="frame_tolerance", field=field, lead_h=lead,
        detail=[dict(metric=metric, value=x, limit=limit)]))
    return s


def test_full_archive_exact_and_six_frozen_values():
    ref = floor.load_reference()
    assert floor.encode(floor.generate()) == floor.REFERENCE.read_text()
    assert ref["pair_values"] == 2406
    assert len(ref["domains"]) == 3
    for d, fields in ref["domains"].items():
        assert len(fields) == 11
        assert all(len(v["C"]) == 73 for v in fields.values())
        assert fields["RAINNC_hourly"]["C"][0] is None
    expected = ["3ff1301cddf34d3b", "3ff8268cb9431a2f", "3fffc8fef332fde5", "4000ca6a497d00f2", "4000e515bf3487b1"]
    for lead, bits in zip(range(68, 73), expected):
        assert struct.pack(">d", ref["domains"]["d03"]["RAINNC"]["C"][lead]).hex() == bits
    assert struct.pack(">d", ref["domains"]["d03"]["W"]["C"][69]).hex() == "3fd51e1647aae7f0"
    assert struct.pack(">d", ref["domains"]["d03"]["RAINNC"]["C"][67]).hex() == "3fec7fc810333fc1"


def test_deleted_or_changed_full_lead_is_rejected(tmp_path, monkeypatch):
    ref = floor.load_reference()
    ref["domains"]["d03"]["RAINNC"]["C"][67] = None
    damaged = tmp_path / "deleted.json"
    damaged.write_text(json.dumps(ref))
    monkeypatch.setattr(floor, "REFERENCE", damaged)
    with pytest.raises(ValueError, match="differs from original"):
        floor.load_reference()


def test_recovered_real_c67_wiring_preserves_raw_and_integrity(tmp_path):
    s = breach(summary())
    s["output_integrity_pass"] = False  # QGRAUP is a separate gate, never waived here
    assert scorer.write_floor72(s, tmp_path) is True
    out = json.loads((tmp_path / "FLOOR72.json").read_text())
    assert out["counts"] == {"FLOOR-LIMITED": 1, "REAL FAIL": 0}
    assert out["rows"][0]["C"] == 0.8905983272943488
    assert out["rows"][0]["C_over_X"] == 0.8401432024719517
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["pass"] is False and saved["binding72_A3"] == "PASS"
    assert saved["output_integrity_pass"] is False


@pytest.mark.parametrize("change", ["date", "lead", "domain", "missing", "nan", "other", "unmatched"])
def test_floor_never_hides_coverage_nonfinite_or_wrong_case(change):
    s = breach(summary())
    d = s["domains"]["d03"]
    if change == "date":
        d["pairing"]["init_time_utc"] = "2026-04-21T00:00:00Z"
        with pytest.raises(ValueError, match="different case"):
            floor.score_floor(s)
        return
    if change == "lead": d["pairing"]["common_leads_h"].pop()
    if change == "domain": s["domains"].pop("d02")
    if change == "missing": d["missing_required_fields"] = {"RAINNC": ["frame"]}
    if change in ("nan", "other"):
        d["policy_failures"].append(dict(kind="nonfinite_cpu" if change == "nan" else "incomplete_field"))
    if change == "unmatched": d["pairing"]["unmatched_gpu_count"] = 1
    assert floor.score_floor(s)["binding72_A3"] == "FAIL"


@pytest.mark.parametrize("field,lead,metric", [("UNARCHIVED", 67, "rmse"), ("RAINNC", 73, "rmse"),
                                                  ("RAINNC", 67, "max_abs"), ("RAINNC_hourly", 0, "rmse")])
def test_absent_or_wrong_metric_c_is_real_failure(field, lead, metric):
    result = floor.score_floor(breach(summary(), field, lead, metric=metric))
    assert result["binding72_A3"] == "FAIL" and result["rows"][0]["C"] is None


def test_original_or_and_ratio_boundary_and_qv_units():
    assert floor.score_floor(breach(summary(), lead=68, x=100.0))["binding72_A3"] == "PASS"  # C>limit
    c = 0.8905983272943488
    assert floor.score_floor(breach(summary(), x=c / .75))["binding72_A3"] == "PASS"
    assert floor.score_floor(breach(summary(), x=c / .749))["binding72_A3"] == "FAIL"
    result = floor.score_floor(breach(summary(), "QVAPOR", 67, x=.001, limit=.0001))
    archived = floor.load_reference()["domains"]["d03"]["QVAPOR"]["C"][67]
    assert result["rows"][0]["C"] == archived / 1000


def test_actual_scorer_cli_reads_floor_and_keeps_integrity_binding(tmp_path):
    """Tiny synthetic history is gate-wiring evidence only, never WRF fidelity."""
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    cpu.mkdir(); gpu.mkdir()
    mandatory = json.loads(scorer.TOLERANCE.read_text())["score_policy"]["mandatory_presence_fields_for_final_runs"]
    for domain in floor.DOMAINS:
        for h in range(73):
            t = datetime(2026, 2, 28) + timedelta(hours=h)
            name = f"wrfout_{domain}_{t:%Y-%m-%d_%H:%M:%S}"
            for directory in (cpu, gpu):
                with Dataset(directory / name, "w") as ds:
                    ds.createDimension("Time", 1); ds.createDimension("DateStrLen", 19)
                    ds.createDimension("south_north", 1); ds.createDimension("west_east", 2)
                    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.array(list(f"{t:%Y-%m-%d_%H:%M:%S}"), "S1")
                    for attr in scorer.REQUIRED_GLOBAL_ATTRS: ds.setncattr(attr, 1)
                    for field in mandatory:
                        values = np.array([[0., 1.]], "f4")
                        if directory == gpu and domain == "d03" and field == "RAINNC" and h == 67:
                            values += np.float32(1.06)
                        ds.createVariable(field, "f4", ("Time", "south_north", "west_east"))[0] = values
    out = tmp_path / "score"
    cli, attempted, private_ref = hermetic_cli(tmp_path)
    cmd = [*cli, "--cpu-dir", str(cpu),
           "--gpu-dir", str(gpu), "--out", str(out), "--hours", "72", "--rw3-floor"]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert not attempted.exists(), "CLI child attempted a protected ALISIOS path"
    assert run.returncode == 0, run.stderr + run.stdout[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["twin_thresholds"] == private_ref
    for domain in floor.DOMAINS:
        assert json.loads((out / f"{domain}_early_hours.json").read_text())["twin_thresholds"] == private_ref
    assert s["pass"] is False and s["binding72_A3"] == "PASS" and s["output_integrity_pass"] is True
    assert json.loads((out / "FLOOR72.json").read_text())["rows"][0]["lead_h"] == 67
    # One zeroed GPU field fails integrity even when the floor itself passes.
    with Dataset(gpu / "wrfout_d03_2026-03-03_00:00:00", "a") as ds: ds["RAINNC"][:] = 0
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert not attempted.exists(), "CLI child attempted a protected ALISIOS path"
    assert run.returncode == 1
    s = json.loads((out / "summary.json").read_text())
    assert s["twin_thresholds"] == private_ref
    assert s["binding72_A3"] == "PASS" and s["output_integrity_pass"] is False
