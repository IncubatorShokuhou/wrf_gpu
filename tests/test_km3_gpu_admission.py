"""km3 stays CPU-admitted; every forecast entry refuses GPU before initialization."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from gpuwrf.io import cammg_config, km3_config


@pytest.mark.parametrize("backend", ["cuda", "rocm", "gpu"])
@pytest.mark.parametrize("value", [3, [4, 3]])
def test_km3_gpu_refused(backend, value):
    errors = km3_config.selection_errors({"dynamics": {"km_opt": value}}, backend=backend)
    assert errors == [km3_config.REFUSAL]


def test_km3_cpu_and_default_admitted():
    assert km3_config.selection_errors({"dynamics": {"km_opt": [3,3]}}, backend="cpu") == []
    assert km3_config.selection_errors({"dynamics": {"km_opt": [4,4]}}, backend="cuda") == []


@pytest.mark.parametrize("max_dom", [1,2])
@pytest.mark.parametrize("resume", [False,True])
@pytest.mark.parametrize("private", [False,True])
def test_km3_nested_and_single_refuse_before_device(tmp_path, monkeypatch, max_dom, resume, private):
    import jax
    from gpuwrf.integration import nested_pipeline as nested
    from gpuwrf.dfi import driver as dfi
    from gpuwrf.io import wrfbdy_coverage
    from gpuwrf.timeseries import runtime as timeseries
    from gpuwrf.streams import runtime as streams
    values = "3" if max_dom == 1 else "4,3"
    (tmp_path/"namelist.input").write_text(f"&dynamics\n diff_opt=2, km_opt={values},\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "1")
    def forbidden(*a, **k):
        raise AssertionError("INITIALIZATION_BEFORE_KM3_ADMISSION")
    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(dfi, "apply_dfi_prepass", forbidden)
    monkeypatch.setattr(timeseries, "require_supported_run", lambda *a, **k: None)
    monkeypatch.setattr(streams, "require_supported_run", lambda *a, **k: None)
    monkeypatch.setattr(wrfbdy_coverage, "require_wrfbdy_coverage", lambda *a, **k: None)
    config = nested.NestedPipelineConfig(input_dir=tmp_path, output_dir=tmp_path/"out",
        proof_dir=tmp_path/"proof", hours=1,max_dom=max_dom, aot_prefetch=True,
        resume_checkpoint=tmp_path/"checkpoint" if resume else None)
    with pytest.raises(ValueError, match="km_opt=3.*CPU-ONLY"):
        if private:
            nested._execute_nested_pipeline(config, _prefetch_handles=[])
        else:
            nested.execute_nested_pipeline(config)


@pytest.mark.parametrize("builder", [False,True])
def test_km3_daily_refuses_before_model(tmp_path, monkeypatch, builder):
    from gpuwrf.integration import daily_pipeline as daily
    (tmp_path/"namelist.input").write_text("&dynamics\n diff_opt=2, km_opt=3,\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    def forbidden(*a, **k):
        raise AssertionError("MODEL_BEFORE_KM3_ADMISSION")
    monkeypatch.setattr(daily, "build_replay_case", forbidden)
    monkeypatch.setattr(daily, "detect_init_mode", forbidden)
    config = daily.DailyPipelineConfig(run_id=str(tmp_path),output_dir=tmp_path/"out",
                                     proof_dir=tmp_path/"proof",hours=1)
    with pytest.raises(ValueError, match="km_opt=3.*CPU-ONLY"):
        (daily._build_real_case if builder else daily.execute_daily_pipeline)(config)


def test_km3_validator_and_binder_refuse(monkeypatch):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.namelist_binding import bind_namelist_options
    monkeypatch.setattr(km3_config, "target_backend", lambda: "cuda")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    options = {"dynamics": {"diff_opt": 2, "km_opt": [4,3]}}
    with pytest.raises(ValueError, match="km_opt=3"):
        validate_operational_namelist(options)
    with pytest.raises(ValueError, match="km_opt=3"):
        bind_namelist_options(None, options, "d02")


@pytest.mark.parametrize("backend", ["cpu","cuda"])
def test_admitted_api_control_reaches_dfi(tmp_path, monkeypatch, backend):
    from gpuwrf.integration import nested_pipeline as nested
    from gpuwrf.dfi import driver as dfi
    value = 3 if backend == "cpu" else 4
    (tmp_path/"namelist.input").write_text(f"&dynamics\n diff_opt=2,km_opt={value},\n/\n")
    monkeypatch.setattr(cammg_config, "target_backend", lambda: backend)
    monkeypatch.setenv("GPUWRF_BATCH_ENSEMBLE", "1")
    def reached(*a, **k):raise RuntimeError("ADMITTED_TO_DFI")
    monkeypatch.setattr(dfi,"apply_dfi_prepass",reached)
    config = nested.NestedPipelineConfig(input_dir=tmp_path,output_dir=tmp_path/"out",
                                        proof_dir=tmp_path/"proof",hours=1,max_dom=1)
    with pytest.raises(RuntimeError,match="ADMITTED_TO_DFI"):
        nested.execute_nested_pipeline(config)


@pytest.mark.parametrize("values", ["3","4,3"])
def test_km3_cli_pre_jax(tmp_path, values):
    root = Path(__file__).resolve().parents[1]
    (tmp_path/"namelist.input").write_text(f"&dynamics\n diff_opt=2,km_opt={values},\n/\n")
    source=f'''import sys,os,importlib.abc
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='jax' or fullname.startswith('jax.'):
            raise AssertionError('JAX_BEFORE_KM3_REFUSAL')
sys.meta_path.insert(0,NoJax())
os.environ['JAX_PLATFORMS']='cuda'
sys.argv=['gpuwrf','run','--input-dir',{str(tmp_path)!r}]
try:import gpuwrf
except SystemExit as error:assert error.code==2
else:raise AssertionError('GPU accepted km3')
'''
    result=subprocess.run([sys.executable,'-c',source],cwd=root,
        env=dict(os.environ,PYTHONPATH=str(root/'src')),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'km_opt=3' in result.stderr and 'non-finite at 1 h' in result.stderr
