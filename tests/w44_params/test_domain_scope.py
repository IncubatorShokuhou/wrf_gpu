"""Per-domain values exist in WRF only for max_domains rconfig entries (critic FIX-5, w44-docs 2026-10-24).

c_s, diff_6th_factor, dampcoef and zdamp are Registry max_domains entries; w_crit_cfl is a count-1 rconfig,
w_alpha/w_beta are module_model_constants PARAMETERs and the five MYNN coefficients are module_bl_mynnedmf
PARAMETERs. A parameter file may therefore vary only the first four between domains.
"""
import json
from pathlib import Path

import pytest

from gpuwrf import cli
from gpuwrf import runtime_parameters as rp

PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


def write(tmp_path, domains):
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"schema": rp.SCHEMA, "domains": domains}))
    return path


def test_wrf_scope_classification_is_pinned():
    assert rp.PER_DOMAIN_IN_WRF == {"c_s", "diff_6th_factor", "dampcoef", "zdamp"}
    assert rp.WRF_GLOBAL == {"w_alpha", "w_beta", "w_crit_cfl", "sqfac", "alp1", "cphm_unst", "am_st", "entrainment"}
    assert rp.WRF_GLOBAL | rp.PER_DOMAIN_IN_WRF == set(rp.DIFFUSION) | set(rp.MYNN)


@pytest.mark.parametrize("family,name,first,second", [
    ("diffusion_parameters", "w_crit_cfl", .5, .6),
    ("diffusion_parameters", "w_alpha", .3, .4),
    ("diffusion_parameters", "w_beta", 1., 2.),
    ("mynn_parameters", "alp1", .23, .253),
    ("mynn_parameters", "sqfac", 3., 6.),
])
def test_file_refuses_differing_or_missing_wrf_global_values(tmp_path, family, name, first, second):
    differing = {"d01": {family: {name: first}}, "d02": {family: {name: second}}}
    with pytest.raises(ValueError, match="one value for all domains in WRF"):
        rp.load_parameters(write(tmp_path, differing))
    with pytest.raises(ValueError, match="one value for all domains in WRF"):
        rp.bind_domain_parameters(None, differing, "d01")  # programmatic dict source
    missing = {"d01": {family: {name: first}}, "d02": {"diffusion_parameters": {"dampcoef": .2}}}
    with pytest.raises(ValueError, match="missing \\['d02'\\]"):
        rp.load_parameters(write(tmp_path, missing))


def test_identical_globals_and_differing_per_domain_values_are_accepted(tmp_path):
    domains = {
        "d01": {"diffusion_parameters": {"w_crit_cfl": .5, "dampcoef": .2, "zdamp": 5000., "c_s": .25,
                                         "diff_6th_factor": .12},
                "mynn_parameters": {"alp1": .253}},
        "d02": {"diffusion_parameters": {"w_crit_cfl": .5, "dampcoef": .4, "zdamp": 6000., "c_s": .5,
                                         "diff_6th_factor": .24},
                "mynn_parameters": {"alp1": .253}},
    }
    assert rp.check_wrf_domain_scope(domains) == domains
    with pytest.raises(ValueError, match="failed the GPU qualification"):
        rp.load_parameters(write(tmp_path, domains))
    with pytest.raises(ValueError, match="missing \\['d02'\\]"):
        rp.check_wrf_domain_scope({"d01": domains["d01"]}, {"d01", "d02"})
    assert rp.check_wrf_domain_scope({"d01": domains["d01"]}, {"d01"})


@pytest.mark.skipif(not (PROD / "namelist.input").exists(), reason="PROD case unavailable")
def test_cli_refuses_a_global_value_that_leaves_a_run_domain_on_the_wrf_default(tmp_path, capsys):
    path = write(tmp_path, {"d01": {"mynn_parameters": {"sqfac": 3.}}})
    argv = ["run", "--input-dir", str(PROD), "--output-dir", str(tmp_path / "out"), "--max-dom", "2", "--hours", "3",
            "--runtime-parameters", str(path), "--dry-run"]
    assert cli.main(argv) != 0
    assert "sqfac is one value for all domains in WRF" in capsys.readouterr().err
    path = write(tmp_path, {"d01": {"diffusion_parameters": {"dampcoef": .4}}})  # per-domain in WRF, but excluded from this release
    assert cli.main(argv) != 0
    assert "failed the GPU qualification" in capsys.readouterr().err
