"""Runtime-file forwarding and frozen directions survive externally approved waves."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.optimize import continuous
from .test_cloud_controls import fixtures, study_files


def approve(output):
    state = json.loads((output/"wave_state.json").read_text())
    pending = state["pending"]
    token = {"schema": "gpuwrf.optimize.wave-go.v1", "decision": "GO", "run_id": state["run_id"],
             "study_sha256": state["study_sha256"], "wave": pending["wave"],
             "report_sha256": pending["report_sha256"], "checkpoint_sha256": pending["checkpoint_sha256"],
             "approved_by": "review:cloud-runtime-integration"}
    (output/pending["go_file"]).write_text(json.dumps(token))


def test_runtime_parameter_waves_preserve_null_freeze_and_exact_resume(study_files, tmp_path, monkeypatch):
    """Analytic surface forward model intentionally cannot observe the runtime scalar."""
    from gpuwrf.ens import workflows
    path, raw = study_files
    raw = copy.deepcopy(raw)
    raw["parameters"] = [{"name": "roughness", "control": "z0mvt"},
                         {"name": "mixing", "control": "mynn_parameters.alp1"}]
    raw["eki"] = {"members": 3, "iterations": 2, "seed": 11}
    raw["scorer"]["truth"] = {"roughness": 1.2, "mixing": 1.0}
    path.write_text(json.dumps(raw))
    study = continuous.load_continuous(path)

    def launcher(calls):
        forward = fixtures.make_fake_launch(calls)

        def launch(plan_path, **kwargs):
            plan = json.loads(Path(plan_path).read_text())
            assert "--case-runtime-parameters" in plan["launcher_options"]
            for member in plan["members"]:
                case = Path(member["case"])
                manifest = json.loads((case/"member_manifest.json").read_text())
                assert any(row["name"] == workflows.RUNTIME_PARAMETERS_NAME for row in manifest["inputs"])
                payload = json.loads((case/workflows.RUNTIME_PARAMETERS_NAME).read_text())["domains"]["d01"]
                cid = Path(manifest["source_case"]).parent.name
                if cid.startswith("i") or cid == "posterior":
                    assert payload["mynn_parameters"]["alp1"] == float(np.float32(.23))
            return forward(plan_path, **kwargs)
        return launch

    calls = []
    monkeypatch.setattr(workflows, "launch", launcher(calls))
    whole = continuous.run(study, tmp_path/"whole", wave_mode=False)
    assert whole["identifiability"]["rank"] == 1

    calls = []
    monkeypatch.setattr(workflows, "launch", launcher(calls))
    output = tmp_path/"waves"
    assert continuous.run(study, output)["status"] == "PAUSED"
    state = json.loads((output/"wave_state.json").read_text())
    assert state["checkpoint"]["iteration"] == 1
    np.testing.assert_array_equal(np.asarray(state["checkpoint"]["theta"])[:, 1], 0)
    count = len(calls)
    with pytest.raises(ValueError, match="GO missing"):
        continuous.run(study, output, resume=True)
    assert len(calls) == count

    approve(output)
    assert continuous.run(study, output, resume=True)["status"] == "PAUSED"
    state = json.loads((output/"wave_state.json").read_text())
    assert state["checkpoint"]["iteration"] == 2
    np.testing.assert_array_equal(np.asarray(state["checkpoint"]["theta"])[:, 1], 0)
    approve(output)
    done = continuous.run(study, output, resume=True)
    for key in ("estimate_multipliers", "evaluation", "iterations", "ensemble_final_multipliers"):
        assert done[key] == whole[key]
    assert done["scheduled_case_hours"] == study["planned_case_hours"]
    count = len(calls)
    assert continuous.run(study, output, resume=True) == done
    assert len(calls) == count
