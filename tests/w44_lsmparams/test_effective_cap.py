"""Shared checker effective-cap semantics, independent file/array known answers."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w44_override"))
from test_override import make_input
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io import wrfinput_consistency as check


def outcome(fn, *args, **kwargs):
    try:
        return {"passed": True, "report": fn(*args, **kwargs)}
    except check.WrfinputConsistencyError as exc:
        return {"passed": False, "report": exc.report}


@pytest.fixture
def data(tmp_path):
    path = make_input(tmp_path / "wrfinput_d01")
    with Dataset(path) as ds:
        fields, meta = check._file_view(ds)
        arrays = {name: np.asarray(value).copy() for name, value in fields.items()}
    return path, arrays, meta


def test_explicit_cap_is_checked_without_a_soil_edit(data):
    path, fields, meta = data
    fields["SMOIS"][:, :, 0, 0] = .4
    assert outcome(check.validate_fields, fields, meta)["passed"]
    refused = outcome(check.validate_fields, fields, meta, effective_smcmax=.3)
    assert not refused["passed"], "ignoring effective_smcmax would admit retained water above the new cap"
    assert any(row["name"] == "soil_porosity" and row["status"] == "FAIL" for row in refused["report"]["checks"])


def test_controlled_cap_replaces_static_stas_for_joint_soil_edits(data):
    path, fields, meta = data
    fields["SMOIS"][:, :, 0, 0] = .4
    changed = np.zeros(fields["SMOIS"].shape, bool)
    changed[:, :, 0, 0] = True
    assert not outcome(check.validate_fields, fields, meta, changed_fields={"SMOIS": changed})["passed"]
    accepted = outcome(check.validate_fields, fields, meta, changed_fields={"SMOIS": changed}, effective_smcmax=.5)
    assert accepted["passed"], "explicit physical cap must replace the static table cap"
    assert accepted["report"]["soil_cap_source"] == "explicit effective SMCMAX"


def test_file_wrapper_forwards_cap_without_soil_edits(data):
    path, _, _ = data
    with Dataset(path, "r+") as ds:
        ds["SMOIS"][:, :, 0, 0] = .4
    assert outcome(check.validate_wrfinput, path)["passed"]
    assert not outcome(check.validate_wrfinput, path, effective_smcmax=.3)["passed"]
    assert outcome(check.validate_wrfinput, path, effective_smcmax=.5)["passed"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -.1, 1.1])
def test_explicit_bad_cap_cannot_pass(data, bad):
    _, fields, meta = data
    assert not outcome(check.validate_fields, fields, meta, effective_smcmax=bad)["passed"]


def test_glacier_and_seaice_keep_original_caller_predicate(data):
    _, fields, meta = data
    fields["LU_INDEX"][0, 0, 0] = fields["IVGTYP"][0, 0, 0] = 15
    fields["SMOIS"][:, :, 0, 0] = 1.
    assert outcome(check.validate_fields, fields, meta, effective_smcmax=.3, surface_physics=4)["passed"]
    assert not outcome(check.validate_fields, fields, meta, effective_smcmax=.3, surface_physics=2)["passed"]
    fields["SEAICE"] = np.zeros(fields["LU_INDEX"].shape, np.float32)
    meta["variable_dimensions"]["SEAICE"] = ("Time", "south_north", "west_east")
    fields["SEAICE"][0, 0, 0] = .5
    assert not outcome(check.validate_fields, fields, meta, effective_smcmax=.3)["passed"]


def test_water_cells_keep_original_exclusion(data):
    _, fields, meta = data
    fields["SMOIS"][:, :, 0, 1] = 1.5
    assert outcome(check.validate_fields, fields, meta, effective_smcmax=.3)["passed"]


def test_default_none_is_exact_to_omitted_for_both_apis(data):
    path, fields, meta = data
    assert outcome(check.validate_fields, fields, meta) == outcome(check.validate_fields, fields, meta, effective_smcmax=None)
    assert outcome(check.validate_wrfinput, path) == outcome(check.validate_wrfinput, path, effective_smcmax=None)
