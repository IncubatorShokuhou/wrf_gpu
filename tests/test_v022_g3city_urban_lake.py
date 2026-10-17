"""Graduated BEP/BEM admission and unseeded urban/Lake guards.

Lake has graduated to its own original-WRF numerical and operational tests.
The saved G3 inventory remains historical evidence, not current qualification.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection
from gpuwrf.io.namelist_check import NotOperationallyWiredError, validate_namelist, validate_operational_namelist
from gpuwrf.io.scheme_catalog import SupportStatus, assert_catalog_consistent, classify_scheme
from gpuwrf.io.wrf_scheme_catalog import is_recognized_wrf_option, wrf_scheme_name
from gpuwrf.physics import lake_model, urban_bep_bem
from gpuwrf.runtime.operational_mode import _SCAN_UNWIRED_REASON, _resolve_operational_suite


def test_g3_catalog_recognizes_graduated_urban_and_lake() -> None:
    assert_catalog_consistent()
    assert classify_scheme("sf_urban_physics", 0).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("sf_lake_physics", 0).status is SupportStatus.IMPLEMENTED

    for key, code, name in (
        ("sf_urban_physics", 2, "BEP"),
        ("sf_urban_physics", 3, "BEM"),
    ):
        assert is_recognized_wrf_option(key, code)
        support = classify_scheme(key, code)
        assert support.status is SupportStatus.IMPLEMENTED
        assert name.lower() in (support.reason + support.alternative + (support.wrf_name or "")).lower()


def test_g3_namelist_admits_qualified_urban() -> None:
    for key, code, expected in (
        ("sf_urban_physics", 2, "BEP"),
        ("sf_urban_physics", 3, "BEM"),
    ):
        validate_namelist({"physics": {key: [code]}})
        validate_operational_namelist({"physics": {key: [code]}})


def test_g3_operational_scan_path_fails_closed_even_if_validator_is_bypassed() -> None:
    for key, code, expected in (
        ("sf_urban_physics", 2, "BEP"),
        ("sf_urban_physics", 3, "BEM"),
        ("sf_lake_physics", 1, "lake"),
    ):
        nml = _minimal_operational_namelist(**{key: code})
        with pytest.raises(UnsupportedSchemeSelection) as excinfo:
            _resolve_operational_suite(nml)
        text = str(excinfo.value)
        assert f"{key}={code}" in text
        assert expected.lower() in text.lower()
        assert f"{key}={code}" in _SCAN_UNWIRED_REASON


def test_g3_scaffold_modules_raise_not_silent() -> None:
    assert len(urban_bep_bem.BEP_REGISTRY_STATE) >= 25
    assert len(urban_bep_bem.BEP_BEM_REGISTRY_STATE) > len(urban_bep_bem.BEP_REGISTRY_STATE)
    assert "lake model" in wrf_scheme_name("sf_lake_physics", 1).name
    assert len(lake_model.LAKE_CARRY_MEMBERS) >= 20

    with pytest.raises(TypeError):
        urban_bep_bem.bep_step()
    with pytest.raises(TypeError):
        urban_bep_bem.bep_bem_step()
    # Lake graduated from this historical scaffold gate; its real timestep
    # requires explicit columns/forcing, qualified by tests/test_s1_lake_*.
    with pytest.raises(TypeError):
        lake_model.lake_step()


def test_g3_proof_gate_reports_partial_scaffold_honestly() -> None:
    import json
    report=json.loads((Path(__file__).resolve().parents[1]/'proofs/v023/feature_sprints/g3_urban_lake_oracle_check.json').read_text())
    assert report["gate_pass"] is True
    assert report["full_physics_landed"] is False
    assert report["default_unchanged"] is True
    assert report["small_grid_static_plausibility"]["passed"] is True
    assert report["small_grid_static_plausibility"]["physics_executed"] is False

    for piece in ("bep", "bem"):
        entry = report["pieces"][piece]
        assert entry["catalog_status"] == SupportStatus.REFERENCE_ONLY.value
        assert entry["oracle_status"] in {"reference_only_inventory", "absent_fail_closed"}
        assert entry["stub_raises"] is True
        assert entry["landed"] is False
        assert entry["scaffold"] is True


def _minimal_operational_namelist(**overrides) -> SimpleNamespace:
    from dataclasses import MISSING, fields
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    # Bypass constructor validation deliberately, with the real inactive defaults
    # for newly added optional features. The suite resolver must still refuse.
    base = {field.name: field.default for field in fields(OperationalNamelist)
            if field.default is not MISSING}
    base.update(
        mp_physics=8,
        bl_pbl_physics=5,
        sf_sfclay_physics=5,
        cu_physics=0,
        sf_surface_physics=0,
        sf_urban_physics=0,
        sf_lake_physics=0,
        use_noahmp=False,
        use_flux_advection=False,
        moist_adv_opt=0,
        ra_sw_physics=4,
        ra_lw_physics=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "proofs/v023/feature_sprints/g3_urban_lake_oracle_check.py"
    spec = importlib.util.spec_from_file_location("g3city_urban_lake_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
