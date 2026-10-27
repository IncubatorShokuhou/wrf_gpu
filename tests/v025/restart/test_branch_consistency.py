"""Shared branch admission on original WN3 real.exe arrays, no forecast claim."""
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from gpuwrf.runtime.checkpoint_branch import apply_overrides
from gpuwrf.runtime.checkpoint_consistency import checkpoint_validator, STATE_TO_WRF, LAND_TO_WRF
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfinput_consistency import WrfinputConsistencyError

CASE = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227")


@pytest.fixture(params=["d01", "d02", "d03"])
def real_input(request):
    name = request.param
    path = CASE / f"wrfinput_{name}"
    if not path.is_file():
        pytest.skip("staged original WN3 0227 real.exe input is not mounted")
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        state = {leaf: np.asarray(ds[key][0]) for leaf, key in STATE_TO_WRF.items() if key in ds.variables}
        state["theta"] = ((np.asarray(ds["T"][0], np.float64) + 300) *
                          (1 + (461.6 / 287.) * state["qv"])).astype(state["qv"].dtype)
        for field, perturbation, base in (("p_total", "P", "PB"), ("ph_total", "PH", "PHB"), ("mu_total", "MU", "MUB")):
            state[field] = np.asarray(ds[perturbation][0]) + np.asarray(ds[base][0])
        base = SimpleNamespace(**{leaf: np.asarray(ds[key][0]) for leaf, key in (("pb", "PB"), ("phb", "PHB"), ("mub", "MUB"))})
        land = {leaf: np.asarray(ds[key][0]) for leaf, key in LAND_TO_WRF.items() if key in ds.variables}
    payload = {"state_fields": state, "noahmp_land_fields": land,
               "extra_fields": {"base_state": base}, "scratch_fields": {}}
    snapshot = {"carries": {name: payload}, "own_steps": {name: 0},
                "driver_state": {"outputs": {name: []}, "forecast_hours": 3}}
    bundles = {name: SimpleNamespace(namelist=SimpleNamespace(sf_surface_physics=4))}
    return name, snapshot, checkpoint_validator(CASE, bundles)


def controls(name, state=None, land=None, *, waivers=None):
    return {"schema_version": 1, "domains": {name: {"state": state or {}, "land": land or {}}},
            "waivers": {name: waivers or {}}}


def test_qv_edit_uses_shared_thermodynamic_checks_and_preserves_other_arrays(real_input):
    name, old, validate = real_input
    before = {k: v.tobytes() for k, v in old["carries"][name]["state_fields"].items()}
    changes = controls(name, {"qv": {"multiply": .99}}, waivers={
        "hydrostatic_consistency": "Declared unbalanced branch sensitivity", "supersaturation_bound": "Source model-state research admission"})
    new, report = apply_overrides(old, changes, validate=validate)
    assert report["consistency"][name]["passed"]
    assert any(c["name"] == "hydrostatic_consistency" and c["status"] == "WAIVED" for c in report["consistency"][name]["checks"])
    assert new["carries"][name]["state_fields"]["qv"].tobytes() != before["qv"]
    for leaf, value in new["carries"][name]["state_fields"].items():
        if leaf != "qv":
            assert value.tobytes() == before[leaf], leaf
    assert {k: v.tobytes() for k, v in old["carries"][name]["state_fields"].items()} == before


def test_qv_edit_without_balance_waiver_refuses(real_input):
    name, old, validate = real_input
    with pytest.raises(WrfinputConsistencyError, match="hydrostatic_consistency"):
        apply_overrides(old, controls(name, {"qv": {"multiply": .99}}), validate=validate)


def test_noop_qv_does_not_readmit_unchanged_thermodynamics(real_input):
    name, old, validate = real_input
    new, report = apply_overrides(old, controls(name, {"qv": {"multiply": 1.}}), validate=validate)
    assert report["consistency"][name]["passed"]
    assert not any(c["name"] in {"hydrostatic_consistency", "supersaturation_bound"} for c in report["consistency"][name]["checks"])
    assert new["carries"][name]["state_fields"]["qv"].tobytes() == old["carries"][name]["state_fields"]["qv"].tobytes()


@pytest.mark.parametrize("field,value,match", [("qv", -1., "moisture_positivity"), ("qke", -1., "field_ranges"),
                                              ("qv", float("nan"), "NaN/Inf|nonfinite")])
def test_invalid_edit_refuses(real_input, field, value, match):
    name, old, validate = real_input
    if field == "qke" and field not in old["carries"][name]["state_fields"]:
        # real.exe lacks QKE for some schemes. Use its actual TKE variable when
        # present; absence itself must fail closed rather than manufacture it.
        with pytest.raises(ValueError, match="no state/qke leaf"):
            apply_overrides(old, controls(name, {field: {"set": value}}), validate=validate)
        return
    with pytest.raises(ValueError, match=match):
        apply_overrides(old, controls(name, {field: {"set": value}}, waivers={
            "hydrostatic_consistency": "Explicit research test", "supersaturation_bound": "Explicit research test"}), validate=validate)


def test_negative_land_store_refuses(real_input):
    name, old, validate = real_input
    assert "smois" in old["carries"][name]["noahmp_land_fields"]
    with pytest.raises(WrfinputConsistencyError, match="soil_porosity|land_water_mask"):
        apply_overrides(old, controls(name, land={"smois": {"set": -1.}}), validate=validate)


def test_ph_edit_requires_waiver_and_retains_mu(real_input):
    name, old, validate = real_input
    mu = old["carries"][name]["state_fields"]["mu_perturbation"].tobytes()
    changes = controls(name, {"ph_perturbation": {"add": .125}})
    with pytest.raises(WrfinputConsistencyError, match="hydrostatic_consistency"):
        apply_overrides(old, changes, validate=validate)
    changes["waivers"][name] = {"hydrostatic_consistency": "Declared fixed-MU research branch"}
    new, report = apply_overrides(old, changes, validate=validate)
    assert report["consistency"][name]["passed"]
    assert new["carries"][name]["state_fields"]["mu_perturbation"].tobytes() == mu


def test_state_skin_validation_cannot_be_hidden_by_retained_land(real_input):
    name, old, validate = real_input
    with pytest.raises(WrfinputConsistencyError, match="field_ranges"):
        apply_overrides(old, controls(name, {"t_skin": {"set": -1.}}), validate=validate)
