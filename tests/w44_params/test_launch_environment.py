import importlib.util
import os
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'proofs/w44_params/launch_environment.py'
SPEC=importlib.util.spec_from_file_location('params_launch_environment',PATH)
ENV=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ENV)


@pytest.fixture
def product(monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS','cuda')
    monkeypatch.delenv('JAX_PLATFORM_NAME',raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS','1')
    monkeypatch.setenv('GPUWRF_AOT_SLIM_MODULE','0')
    monkeypatch.setenv('GPUWRF_GPU_LOCK_HELD','0')


def test_records_product_before_only_device_substitution(product):
    result=ENV.select_device(True)
    assert result['product_child_env']['JAX_PLATFORMS']=='cuda'
    assert result['device_override']=={'JAX_PLATFORMS':'cpu'}
    assert os.environ['JAX_PLATFORMS']=='cpu'


@pytest.mark.parametrize('bad',['gpu','rocm','cpu'])
def test_bad_product_backend_cannot_be_hidden_by_cpu_smoke(product,monkeypatch,bad):
    monkeypatch.setenv('JAX_PLATFORMS',bad)
    with pytest.raises(AssertionError):ENV.select_device(True)
    assert os.environ['JAX_PLATFORMS']==bad


def test_real_device_path_requires_official_lock_marker(product):
    with pytest.raises(AssertionError):ENV.select_device(False)
