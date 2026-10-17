from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gpuwrf.validation.savepoint_schema import load_tolerance_ladder


def test_m6b0r_synthetic_dryrun_passes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/m6b0r_synthetic_dryrun.py"
    # Keep the real schema/tamper probe in a fresh process, with private output.
    invoke = (
        "import runpy,sys; from pathlib import Path; "
        "probe=runpy.run_path(sys.argv[1])['main']; "
        "probe.__globals__['SPRINT']=Path(sys.argv[2]); "
        "raise SystemExit(probe())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", invoke, str(script), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["perturbation_caught"] is True
    assert payload["schema_version_mismatch_caught"] is True
    assert payload["tamper_detection_caught"] is True


def test_tolerance_ladder_has_required_calc_coef_fields():
    ladder = load_tolerance_ladder()
    for field in ("a", "alpha", "gamma"):
        entry = ladder["fields"][field]
        assert entry["units"]
        assert entry["dtype"] == "float64"
        assert entry["abs"] is not None
        assert entry["ulp"] is not None
