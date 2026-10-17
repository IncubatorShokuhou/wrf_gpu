"""WRF activates SPP sub-switches independently, including on child domains."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_operational_namelist
from gpuwrf.io.scheme_catalog import classify_feature_switch, classify_scheme, SupportStatus

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", ["skebs", "sppt", "spp", "rand_perturb", "stoch_force_opt",
                                 "spp_conv", "spp_pbl", "spp_lsm"])
@pytest.mark.parametrize("domain_values", [[1, 0], [0, 1]])
def test_independent_stochastic_switch_is_admitted_with_honest_qualification(key, domain_values):
    config = {"stoch": {"spp": [0, 0], key: domain_values}, "domains": {"max_dom": 2}}
    assert classify_feature_switch(key, 1) is None
    support = classify_scheme(key, 1)
    assert support.status == SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified; GPU/coupled-forecast qualification pending" in support.reason
    validate_operational_namelist(config)


def test_stochastic_off_options_preserve_release_admission():
    validate_operational_namelist({"stoch": {key: [0, 0] for key in
        ("skebs", "sppt", "spp", "spp_conv", "spp_pbl", "spp_lsm", "rand_perturb", "stoch_force_opt")}})


@pytest.mark.parametrize("key", ["spp_conv", "spp_pbl", "spp_lsm"])
def test_invalid_child_subswitch_cli_refused_before_model_import(tmp_path, key):
    namelist = tmp_path / "namelist.input"
    namelist.write_text(f"&domains\n max_dom=2,\n/\n&stoch\n spp=0,0,\n {key}=0,2,\n/\n")
    script = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name == "jax" or name.startswith("jax."):
            raise AssertionError("admission imported JAX")
sys.meta_path.insert(0,Block())
arguments=sys.argv[1:]
sys.argv=["gpuwrf","manifest"] # pure-host bootstrap; run path must reject before JAX
from gpuwrf.cli import main
raise SystemExit(main(arguments))
'''
    result = subprocess.run([sys.executable, "-c", script, "run", "--namelist", str(namelist),
                            "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")], cwd=REPO,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert key in result.stderr and "admission imported JAX" not in result.stderr
