"""CAM-MG GPU admission: mp11 runs on GPU again (stock-Triton native column,
lane t-cammgws); the backend-dark refusal mechanism stays covered with a
monkeypatched refused set."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpuwrf.io import cammg_config


def test_mp11_is_not_gpu_refused():
    assert cammg_config.GPU_REFUSED_MP_PHYSICS == frozenset()


@pytest.mark.parametrize("backend", ["cuda", "rocm", "gpu", "unknown"])
@pytest.mark.parametrize("value", [11, [8, 11], [11, 11]])
def test_gpu_selection_admitted(monkeypatch, backend, value):
    monkeypatch.setattr(cammg_config, "target_backend", lambda: backend)
    cammg_config.validate_selection({"physics": {"mp_physics": value}})
    assert cammg_config.selection_errors({"physics": {"mp_physics": value}}) == []


@pytest.mark.parametrize("backend", ["cuda", "rocm", "gpu", "unknown"])
@pytest.mark.parametrize("value", [11, [8, 11], [11, 11]])
def test_refusal_mechanism_still_refuses_a_listed_scheme(monkeypatch, backend, value):
    monkeypatch.setattr(cammg_config, "GPU_REFUSED_MP_PHYSICS", frozenset({11}))
    monkeypatch.setattr(cammg_config, "target_backend", lambda: backend)
    with pytest.raises(ValueError, match="RAINNCV/SNOWNCV"):
        cammg_config.validate_selection({"physics": {"mp_physics": value}})


def test_cpu_and_other_microphysics_admitted(monkeypatch):
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cpu")
    cammg_config.validate_selection({"physics": {"mp_physics": [11, 11]}})
    assert cammg_config.selection_errors(SimpleNamespace(mp_physics=11), backend="cpu") == []
    assert cammg_config.selection_errors({"physics": {"mp_physics": [8, 6]}}, backend="cuda") == []


@pytest.mark.parametrize("argument", ["input", "namelist", "equals"])
def test_cli_admits_mp11_and_reaches_jax(tmp_path, argument):
    root = Path(__file__).resolve().parents[1]
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics=8, 2*11,\n/\n")
    args = {"input": ["--input-dir", str(tmp_path)],
            "namelist": ["--namelist", str(namelist)],
            "equals": ["--namelist=" + str(namelist)]}[argument]
    source = f'''import sys, os, importlib.abc
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise AssertionError('REACHED_JAX')
sys.meta_path.insert(0, NoJax())
os.environ['JAX_PLATFORMS'] = 'cuda'
sys.argv = ['gpuwrf', 'run'] + {args!r}
try:
    import gpuwrf
except SystemExit as error:
    raise AssertionError('GPU refused CAM-MG: %s' % error.code)
except AssertionError as error:
    assert str(error) == 'REACHED_JAX', error
else:
    raise AssertionError('missing bootstrap outcome')
'''
    result = subprocess.run([sys.executable, "-c", source], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root / "src")),
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU-ONLY" not in result.stderr


def test_operational_validator_and_binding_admit_mp11(monkeypatch):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.namelist_binding import bind_namelist_options
    monkeypatch.setattr(cammg_config, "target_backend", lambda: "cuda")
    config = {"physics": {"mp_physics": [8, 11]}}
    for call in (lambda: validate_operational_namelist(config), lambda: bind_namelist_options(None, config, "d02")):
        try:
            call()
        except (ValueError, TypeError, AttributeError) as error:
            # Later, unrelated validation of this minimal config may still object;
            # the CAM-MG GPU admission must not be the reason.
            assert "CAM-MG" not in str(error) and "CPU-ONLY" not in str(error), error
