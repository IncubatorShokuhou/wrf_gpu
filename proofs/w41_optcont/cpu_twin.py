"""CPU record of the analytic twin (tests/w41_optcont fake launcher; NOT the WRF model): numbers behind the test.

  JAX_PLATFORMS=cpu python proofs/w41_optcont/cpu_twin.py WORKDIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"src"), str(REPO/"tests"/"w41_optcont"), str(REPO/"tests"/"w41_trackp")]


def main(work, out):
    import pytest
    import test_continuous as t
    from gpuwrf.ens import workflows
    from gpuwrf.optimize import continuous

    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    path, _ = t.study_files.__wrapped__(work)
    study = continuous.load_continuous(path)
    calls = []
    mp = pytest.MonkeyPatch()
    mp.setattr(workflows, "launch", t.make_fake_launch(calls))
    try:
        report = continuous.run(study, work/"run")
    finally:
        mp.undo()
    record = {"scope": "CPU, analytic fake launcher (log-linear T2/wind response to z0), NOT the WRF model",
              "truth": t.TRUTH, "estimate": report["estimate_multipliers"], "posterior_sd_log": report["posterior_sd_log"],
              "abs_log_error": report["twin"]["abs_log_error"], "verdict": report["twin"]["verdict"],
              "misfit_history": [h["normalised_misfit"] for h in report["iterations"]],
              "evaluation": report["evaluation"], "program": report["program"],
              "batch_sizes": [len(c["members"]) for c in calls], "scheduled_case_hours": report["scheduled_case_hours"]}
    Path(out).write_text(json.dumps(record, indent=1)+"\n")
    print(json.dumps({k: record[k] for k in ("estimate", "abs_log_error", "verdict", "misfit_history", "batch_sizes")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
