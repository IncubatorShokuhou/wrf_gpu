"""&dfi_control admission (pre-JAX): TDFI (dfi_opt=3) accepted when reproducible, all else refused.

No numerical model execution here (the TDFI driver itself: tests/test_o2_dfi_*.py).
"""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from gpuwrf.io.namelist_check import (
    UnsupportedSchemeError,
    validate_namelist,
    validate_operational_namelist,
)


@pytest.mark.parametrize("value", [1, 2, 3, -1, "3", [0, 3]])
def test_unreproducible_dfi_refused(value):
    # Bare active DFI without a time step/window is invalid; -1 and [0,3] are invalid too.
    with pytest.raises(UnsupportedSchemeError) as exc:
        validate_operational_namelist({"dfi_control": {"dfi_opt": value}})
    message = str(exc.value)
    assert "dfi_control.dfi_opt" in message
    assert "digital filter initialization" in message
    assert "dfi_opt=0" in message
    assert "CPU-WRF" in message


@pytest.mark.parametrize("config", [{}, {"dfi_opt": 0}, {"dfi_opt": "0"},
                                  {"dfi_control": {"dfi_opt": [0]}}])
def test_dfi_disabled_or_absent_passes(config):
    validate_operational_namelist(config)


SWISS_TDFI = Path(__file__).parent / "fixtures/o2_dfi_swiss_tdfi.namelist"


def test_reproducible_tdfi_namelist_is_accepted():
    validate_operational_namelist(SWISS_TDFI.read_text())


@pytest.mark.parametrize("old, new, needle", [
    (" dfi_nfilter                         = 7,", " dfi_nfilter = 8,", "RHOFIL"),
    (" dfi_bckstop_day                     = 14,", " dfi_bckstop_day = 15,", "dfi_bckstop"),
    (" dfi_bckstop_hour                    = 23,", " dfi_bckstop_hour = 1,", "dfi_time_dim"),
    (" dfi_opt                             = 3,", " dfi_opt = 4,", "dfi_opt=4"),
])
def test_unreproducible_tdfi_variants_are_refused_with_the_reason(old, new, needle):
    text = SWISS_TDFI.read_text()
    assert old in text
    with pytest.raises(UnsupportedSchemeError) as exc:
        validate_operational_namelist(text.replace(old, new))
    assert needle in str(exc.value) and "dfi_opt=0" in str(exc.value)


def test_reference_validator_and_unknown_keys_unchanged():
    validate_namelist({"dfi_control": {"dfi_opt": 3}})
    validate_operational_namelist({"unregistered_control": 3})


def test_standard_wn3_dfi0_passes():
    text = (Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist").read_text()
    validate_operational_namelist(text)
    validate_operational_namelist(text + "\n&dfi_control\n dfi_opt=0,\n/\n")


def test_cli_refuses_dfi_before_backend_or_model_import(tmp_path):
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&dfi_control\n dfi_opt=3,\n/\n")
    code = '''
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts/v025"))
import cpu_guard

class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("jax_plugins", "jaxlib.cuda", "cupy", "triton")):
            raise ImportError("CPU admission test excludes GPU plugins: " + fullname)
        if fullname.startswith(("gpuwrf.integration", "gpuwrf.physics",
                                "gpuwrf.runtime.operational_mode")):
            raise AssertionError("heavy import before DFI refusal: " + fullname)
sys.meta_path.insert(0, NoHeavyImports())
from gpuwrf import cli
from jax._src import xla_bridge
def forbidden(*args, **kwargs):
    raise AssertionError("backend initialization before DFI refusal")
xla_bridge.get_backend = forbidden
xla_bridge.backends = forbidden
assert cli.main(["run", "--namelist", sys.argv[1], "--input-dir", sys.argv[2],
                 "--output-dir", sys.argv[3], "--hours", "3"]) == 2
assert not xla_bridge._backends
'''
    repo = Path(__file__).resolve().parents[1]
    # E152: the child must import the source tree under test explicitly (package not installed in snapshots).
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(repo / "src"), os.environ.get("PYTHONPATH")])))
    result = subprocess.run(
        [sys.executable, "-c", code, str(namelist), str(tmp_path),
         str(tmp_path / "output")], env=env, capture_output=True, text=True, timeout=30,
        cwd=repo,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dfi_control.dfi_opt" in result.stderr
    assert "dfi_opt=0" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("shape, needle", [(["--max-dom", "2"], "nested"), (["--domain", "d02"], "replay")])
def test_cli_refuses_non_native_tdfi_before_backend(tmp_path, shape, needle):
    """A reproducible TDFI namelist outside the native single-domain d01 driver is refused pre-backend."""

    namelist = tmp_path / "namelist.input"
    namelist.write_text(SWISS_TDFI.read_text())
    code = '''
import sys
from gpuwrf import cli
from jax._src import xla_bridge
def forbidden(*args, **kwargs):
    raise AssertionError("backend initialization before DFI refusal")
xla_bridge.get_backend = forbidden
xla_bridge.backends = forbidden
assert cli.main(["run", "--namelist", sys.argv[1], "--input-dir", sys.argv[2],
                 "--output-dir", sys.argv[3], "--hours", "3", *sys.argv[4:]]) == 2
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"),
                                         env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", code, str(namelist), str(tmp_path), str(tmp_path / "output"), *shape],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert needle in result.stderr and "dfi_opt=0" in result.stderr
    assert not (tmp_path / "output").exists()
