"""Actual API call-order and effective CLI selection controls; no GPU work."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwrf.io import cammg_config


@pytest.fixture
def refused_mp11(monkeypatch):
    # mp11 is GPU-admitted since the stock-Triton column (t-cammgws); keep the
    # admission-order infrastructure covered with a refused set of {11}.
    monkeypatch.setattr(cammg_config, "GPU_REFUSED_MP_PHYSICS", frozenset({11}))


@pytest.mark.parametrize("entry", ["nested", "nested_private", "loader", "batch_loader", "daily", "daily_builder"])
@pytest.mark.parametrize("resume", [False, True])
def test_api_refuses_before_initialization(refused_mp11, tmp_path, monkeypatch, entry, resume):
    import jax
    from gpuwrf.integration import nested_pipeline as nested, daily_pipeline as daily
    from gpuwrf.dfi import driver as dfi
    from gpuwrf.io import wrfbdy_coverage
    from gpuwrf.timeseries import runtime as timeseries
    from gpuwrf.streams import runtime as streams

    (tmp_path / "namelist.input").write_text("&physics\n mp_physics=8,11,\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "1")
    events = []
    validate = cammg_config.validate_selection
    def admitted(options):
        events.append("admission")
        return validate(options)
    monkeypatch.setattr(cammg_config, "validate_selection", admitted)
    def forbidden(*args, **kwargs):
        events.append("device/DFI/model/init")
        raise AssertionError("INITIALIZATION_BEFORE_CAMMG_ADMISSION")
    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(dfi, "apply_dfi_prepass", forbidden)
    monkeypatch.setattr(nested, "build_replay_case", forbidden)
    monkeypatch.setattr(daily, "build_replay_case", forbidden)
    monkeypatch.setattr(daily, "detect_init_mode", forbidden)
    monkeypatch.setattr(timeseries, "require_supported_run", lambda *a, **k: None)
    monkeypatch.setattr(streams, "require_supported_run", lambda *a, **k: None)
    monkeypatch.setattr(wrfbdy_coverage, "require_wrfbdy_coverage", lambda *a, **k: None)
    cfg = nested.NestedPipelineConfig(input_dir=tmp_path, output_dir=tmp_path/"out",
        proof_dir=tmp_path/"proof", hours=1, max_dom=2, aot_prefetch=True,
        resume_checkpoint=tmp_path/"checkpoint" if resume else None)
    dc = daily.DailyPipelineConfig(run_id=str(tmp_path), output_dir=tmp_path/"out",
                                 proof_dir=tmp_path/"proof", hours=1)
    calls = {"nested": lambda: nested.execute_nested_pipeline(cfg),
             "nested_private": lambda: nested._execute_nested_pipeline(cfg, _prefetch_handles=[]),
             "loader": lambda: nested._load_domains(cfg, ("d01", "d02")),
             "batch_loader": lambda: nested._load_batched_domains(cfg, ("d01", "d02"), batch_size=1),
             "daily": lambda: daily.execute_daily_pipeline(dc),
             "daily_builder": lambda: daily._build_real_case(dc)}
    with pytest.raises(ValueError, match="CAM-MG"):
        calls[entry]()
    assert events == ["admission"]


@pytest.mark.parametrize("location", ["config", "environment"])
def test_all_batch_inputs_admitted_before_dfi(refused_mp11, tmp_path, monkeypatch, location):
    from gpuwrf.integration import nested_pipeline as nested
    from gpuwrf.dfi import driver as dfi
    safe, bad = tmp_path/"safe", tmp_path/"bad"
    for directory, value in ((safe, "8,8"), (bad, "8,11")):
        directory.mkdir()
        (directory/"namelist.input").write_text(f"&physics\n mp_physics={value},\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2")
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    def forbidden(*a, **k):
        raise AssertionError("DFI_BEFORE_ALL_BATCH_INPUTS_ADMITTED")
    monkeypatch.setattr(dfi, "apply_dfi_prepass", forbidden)
    cfg = nested.NestedPipelineConfig(input_dir=safe, output_dir=tmp_path/"out",
        proof_dir=tmp_path/"proof", hours=1, max_dom=2,
        batch_input_dirs=(safe, bad) if location == "config" else None)
    if location == "environment":
        monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{safe},{bad}")
    with pytest.raises(ValueError, match="CAM-MG"):
        nested.execute_nested_pipeline(cfg)


@pytest.mark.parametrize("backend,value", [("cpu", 11), ("cuda", 8), ("cuda", 11)])
def test_api_admitted_control_reaches_dfi(tmp_path, monkeypatch, backend, value):
    from gpuwrf.integration import nested_pipeline as nested
    from gpuwrf.dfi import driver as dfi
    (tmp_path/"namelist.input").write_text(f"&physics\n mp_physics={value},\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: backend)
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "1")
    def reached(*a, **k):
        raise RuntimeError("ADMITTED_TO_DFI")
    monkeypatch.setattr(dfi, "apply_dfi_prepass", reached)
    cfg = nested.NestedPipelineConfig(input_dir=tmp_path, output_dir=tmp_path/"out",
                                     proof_dir=tmp_path/"proof", hours=1, max_dom=2)
    with pytest.raises(RuntimeError, match="ADMITTED_TO_DFI"):
        nested.execute_nested_pipeline(cfg)


@pytest.mark.parametrize("key", ["namelist", "input-dir"])
@pytest.mark.parametrize("last_bad", [False, True])
def test_effective_options_use_actual_argparse(refused_mp11, tmp_path, monkeypatch, key, last_bad):
    from gpuwrf.cli import build_parser
    safe, bad = tmp_path/"safe", tmp_path/"bad"
    for directory, value in ((safe, 8), (bad, 11)):
        directory.mkdir()
        (directory/"namelist.input").write_text(f"&physics\n mp_physics={value},\n/\n")
    paths = [safe, bad] if last_bad else [bad, safe]
    values = [str(p/"namelist.input" if key == "namelist" else p) for p in paths]
    args = ["--output-dir", str(tmp_path/"out")]
    if key == "namelist":
        args += ["--input-dir", str(safe)]
    args += [f"--{key}", values[0], f"--{key}="+values[1]]
    parsed = build_parser().parse_args(["run"]+args)
    effective = parsed.namelist or parsed.input_dir/"namelist.input"
    assert effective == (paths[-1]/"namelist.input")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    outcome = cammg_config.bootstrap_refusal(["gpuwrf", "run"]+args)
    assert outcome == (cammg_config.REFUSAL if last_bad else None)


@pytest.mark.parametrize("spelling", ["--name", "--input", "--name=bad.input", "--input=bad"])
def test_abbreviations_refuse_directly(refused_mp11, monkeypatch, spelling):
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    outcome = cammg_config.bootstrap_refusal(["gpuwrf", "run", spelling, "unused"])
    assert outcome and "full --namelist / --input-dir spellings" in outcome


@pytest.mark.parametrize("key", ["namelist", "input-dir"])
@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("last_bad", [False, True])
def test_cli_last_occurrence_matches_argparse(tmp_path, key, equals, last_bad):
    root = Path(__file__).resolve().parents[1]
    safe, bad = tmp_path/"safe", tmp_path/"bad"
    for directory, value in ((safe, "8"), (bad, "8,11")):
        directory.mkdir()
        (directory/"namelist.input").write_text(f"&physics\n mp_physics={value},\n/\n")
    values = [safe, bad] if last_bad else [bad, safe]
    values = [str(p/"namelist.input" if key == "namelist" else p) for p in values]
    args = [arg for v in values for arg in ([f"--{key}={v}"] if equals else [f"--{key}", v])]
    _bootstrap_child(root, args, refusal=False)  # mp11 GPU-admitted (empty refused set)


@pytest.mark.parametrize("key", ["--name", "--input"])
@pytest.mark.parametrize("last_bad", [False, True])
def test_cli_abbreviations_fail_closed_both_directions(tmp_path, key, last_bad):
    root = Path(__file__).resolve().parents[1]
    safe, bad = tmp_path/"safe.input", tmp_path/"bad.input"
    safe.write_text("&physics\n mp_physics=8,\n/\n")
    bad.write_text("&physics\n mp_physics=11,\n/\n")
    values = [safe, bad] if last_bad else [bad, safe]
    _bootstrap_child(root, [key, str(values[0]), key+"="+str(values[1])],
                     refusal=True, message="full --namelist / --input-dir spellings")


@pytest.mark.parametrize("source", ["case", "batch"])
def test_cli_safe_override_cannot_hide_actual_mp11_input(tmp_path, monkeypatch, source):
    root = Path(__file__).resolve().parents[1]
    safe, bad = tmp_path/"safe", tmp_path/"bad"
    for directory, value in ((safe, 8), (bad, 11)):
        directory.mkdir()
        (directory/"namelist.input").write_text(f"&physics\n mp_physics={value},\n/\n")
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "2" if source == "batch" else "1")
    if source == "batch":
        monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", f"{safe},{bad}")
    _bootstrap_child(root, ["--namelist", str(safe/"namelist.input"),
        "--input-dir", str(safe if source == "batch" else bad)], refusal=False)


def _bootstrap_child(root, args, *, refusal, message="RAINNCV/SNOWNCV"):
    source = f'''import sys,os,importlib.abc
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise AssertionError('REACHED_JAX')
sys.meta_path.insert(0,NoJax())
os.environ['JAX_PLATFORMS']='cuda'
sys.argv=['gpuwrf','run']+{args!r}
try:
    import gpuwrf
except SystemExit as exc:
    assert {refusal!r} and exc.code == 2
except AssertionError as exc:
    assert not {refusal!r} and str(exc) == 'REACHED_JAX'
else:
    raise AssertionError('missing bootstrap outcome')
'''
    result = subprocess.run([sys.executable, "-c", source], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root/"src")), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    if refusal:
        assert message in result.stderr
