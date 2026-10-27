"""Public options reach the shared native driver and preserve checkpoint routing."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf import cli
from gpuwrf.integration import nested_pipeline
from gpuwrf.runtime_parameters import SCHEMA
from gpuwrf.surface_params.controls import save_controls


def test_family_catalog_matches_admitted_names():
    from gpuwrf.io.scheme_catalog import RUNTIME_PARAMETER_FAMILIES
    from gpuwrf.runtime_parameters import DIFFUSION, MYNN
    from gpuwrf.surface_params.controls import ALL_CONTROLS
    for family, controls in (("surface_controls", ALL_CONTROLS), ("diffusion_parameters", DIFFUSION),
                             ("mynn_parameters", MYNN)):
        assert set(RUNTIME_PARAMETER_FAMILIES[family]["parameters"]) == set(controls)


@pytest.mark.parametrize("family", ["surface", "diffusion"])
def test_cli_control_file_reaches_checkpoint_driver(tmp_path, monkeypatch, family, capsys):
    source = Path(__file__).resolve().parents[2] / "examples/switzerland_d01"
    configs = []

    def execute(config):
        configs.append(config)
        return {"verdict": "PIPELINE_GREEN", "wrfout_files": [], "metadata": {}}

    monkeypatch.setattr(nested_pipeline, "execute_nested_pipeline", execute)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    monkeypatch.delenv("GPUWRF_BATCH_ENSEMBLE", raising=False)
    if family == "surface":
        path = tmp_path / "surface.npz"
        save_controls(path, {"d01": {"ctopo": np.ones((42, 42), np.float32)}})
        option = "--surface-controls"
    else:
        path = tmp_path / "parameters.json"
        path.write_text(json.dumps({"schema": SCHEMA, "domains": {
            "d01": {"diffusion_parameters": {"diff_6th_factor": .24}}}}))
        option = "--runtime-parameters"
    rc = cli.main(["run", "--input-dir", str(source), "--output-dir", str(tmp_path / "out"),
                     "--domain", "d01", "--hours", "1", "--scratch-dir", str(tmp_path / "scratch"),
                     "--checkpoint-dir", str(tmp_path / "checkpoints"), "--checkpoint-interval-steps", "1",
                     option, str(path)])
    if family == "diffusion":
        assert rc != 0 and not configs
        assert "failed the GPU qualification" in capsys.readouterr().err
        return
    assert rc == 0
    assert len(configs) == 1 and configs[0].checkpoint_dir == tmp_path / "checkpoints"
    assert getattr(configs[0], "surface_controls" if family == "surface" else "runtime_parameters") == path
