"""Host array API and checkpoint-only admission; no temporary-file adapter."""
import copy
import json
import shutil

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfinput_consistency import validate_fields, validate_wrfinput, WrfinputConsistencyError
from test_override import make_input, XY, XYZ, WAIVER


@pytest.fixture
def view(tmp_path):
    path = make_input(tmp_path / "input")
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        fields = {name: np.asarray(var[:]) for name, var in ds.variables.items()}
        meta = {"dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
                "variable_dimensions": {name: var.dimensions for name, var in ds.variables.items()},
                "attributes": {name: ds.getncattr(name) for name in ds.ncattrs()}}
    return fields, meta, path


def add(fields, meta, name, dims, value):
    shape = tuple(meta["dimensions"][dim] for dim in dims)
    fields[name] = np.full(shape, value, np.float32)
    meta["variable_dimensions"][name] = dims


def test_array_and_file_results_identical_and_arrays_unchanged(view):
    fields, meta, path = view
    before = {k: v.tobytes() for k, v in fields.items()}
    old_meta = copy.deepcopy(meta)
    assert validate_fields(fields, meta, changed_fields=["T"], waivers=WAIVER) == validate_wrfinput(path, changed_fields=["T"], waivers=WAIVER)
    assert {k: v.tobytes() for k, v in fields.items()} == before
    assert meta == old_meta


def test_array_api_never_opens_netcdf(view, monkeypatch):
    fields, meta, _ = view
    def forbidden(*args, **kwargs):
        raise AssertionError("array checks opened a NetCDF file")
    monkeypatch.setattr("gpuwrf.io.wrfinput_consistency.Dataset", forbidden)
    assert validate_fields(fields, meta, checkpoint=True)["passed"]


def test_checkpoint_empty_supersaturation_is_retained_but_nonfinite_refuses(view):
    fields, meta, _ = view
    fields["QVAPOR"][:] = .1
    result = validate_fields(fields, meta, checkpoint=True)
    assert result["passed"]
    assert not any(c["name"] in {"supersaturation_bound", "hydrostatic_consistency"} for c in result["checks"])
    fields["TSK"][0, 0, 0] = np.inf
    with pytest.raises(WrfinputConsistencyError, match="finite"):
        validate_fields(fields, meta, checkpoint=True)


@pytest.mark.parametrize("name,dims", [("U", XYZ[:-1] + ("west_east_stag",)),
    ("V", XYZ[:2] + ("south_north_stag", "west_east")), ("W", ("Time", "bottom_top_stag", "south_north", "west_east")),
    ("QKE", XYZ)])
def test_checkpoint_wind_and_qke_fields(view, name, dims):
    fields, meta, _ = view
    meta["dimensions"].update(west_east_stag=4, south_north_stag=3)
    add(fields, meta, name, dims, .2)
    assert validate_fields(fields, meta, changed_fields=[name], checkpoint=True)["passed"]
    if name == "QKE":
        fields[name][0, 0, 0, 0] = -.1
        with pytest.raises(WrfinputConsistencyError, match="field_ranges"):
            validate_fields(fields, meta, changed_fields=[name], checkpoint=True)


def test_checkpoint_ph_edit_requires_balance_waiver_and_valid_geometry(view):
    fields, meta, _ = view
    fields["PH"][0, 1] += 10
    with pytest.raises(WrfinputConsistencyError, match="hydrostatic_consistency"):
        validate_fields(fields, meta, changed_fields=["PH"], checkpoint=True)
    assert validate_fields(fields, meta, changed_fields=["PH"], checkpoint=True, waivers=WAIVER)["passed"]
    fields["PH"][0, 1] = -10000
    with pytest.raises(WrfinputConsistencyError, match="mass_geometry"):
        validate_fields(fields, meta, changed_fields=["PH"], checkpoint=True, waivers=WAIVER)


@pytest.mark.parametrize("name", ["MU", "MUB", "PB", "PHB", "HGT"])
def test_checkpoint_base_and_mu_edits_refuse(view, name):
    fields, meta, _ = view
    with pytest.raises(WrfinputConsistencyError, match="unsupported field"):
        validate_fields(fields, meta, changed_fields=[name], checkpoint=True, waivers=WAIVER)


@pytest.mark.parametrize("name", ["SNOWH", "SNOW", "CANLIQ", "CANICE", "TV", "TG", "TAH", "SAI"])
def test_checkpoint_land_stores_and_temperatures(view, name):
    fields, meta, _ = view
    add(fields, meta, name, XY, 280 if name in {"TV", "TG", "TAH"} else .1)
    mask = np.array([[[1, 0, 1], [1, 1, 0]]], dtype=bool)
    assert validate_fields(fields, meta, changed_fields={name: mask}, checkpoint=True)["passed"]
    fields[name][0, 0, 0] = -1
    with pytest.raises(WrfinputConsistencyError, match="field_ranges"):
        validate_fields(fields, meta, changed_fields={name: mask}, checkpoint=True)


def test_checkpoint_liquid_soil_bounded_by_total_soil_water(view):
    fields, meta, _ = view
    dims = ("Time", "soil_layers_stag", "south_north", "west_east")
    add(fields, meta, "SH2O", dims, .1)
    mask = np.broadcast_to(np.array([[[1, 0, 1], [1, 1, 0]]], bool), fields["SH2O"].shape)
    assert validate_fields(fields, meta, changed_fields={"SH2O": mask}, checkpoint=True)["passed"]
    fields["SH2O"][0, :, 0, 0] = .3
    with pytest.raises(WrfinputConsistencyError, match="soil_porosity"):
        validate_fields(fields, meta, changed_fields={"SH2O": mask}, checkpoint=True)


@pytest.mark.parametrize("name", ["TSNO", "SNICE", "SNLIQ"])
def test_checkpoint_snow_layers(view, name):
    fields, meta, _ = view
    meta["dimensions"]["snow_layers"] = 3
    dims = ("Time", "snow_layers", "south_north", "west_east")
    for field, value in (("TSNO", 270), ("SNICE", .1), ("SNLIQ", .01)):
        add(fields, meta, field, dims, value)
    mask = np.broadcast_to(np.array([[[1, 0, 1], [1, 1, 0]]], bool), fields[name].shape)
    assert validate_fields(fields, meta, changed_fields={name: mask}, checkpoint=True)["passed"]
    fields[name][0, 0, 0, 0] = -1
    with pytest.raises(WrfinputConsistencyError, match="field_ranges"):
        validate_fields(fields, meta, changed_fields={name: mask}, checkpoint=True)


def test_array_shape_must_match_metadata(view):
    fields, meta, _ = view
    fields["T"] = fields["T"][:, :, :, :-1]
    with pytest.raises(WrfinputConsistencyError, match="array shape"):
        validate_fields(fields, meta, checkpoint=True)


def inherited_state(fields):
    """Literal WRF-shaped source: two outer-ring negative-number/RH cells."""
    fields["QNRAIN"][0, 0, 0, 0] = np.float32(-6.19e-29)
    fields["QNRAIN"][0, 1, 1, 2] = np.float32(-3.e-30)
    temperature = 300 * .9 ** (2 / 7)
    e = 1.0049 * 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    q = np.float32((287 / 461.6) * e / (90000 - e))
    fields["QVAPOR"][0, 0, 0, 0] = q
    fields["QVAPOR"][0, 1, 1, 2] = q


def assert_inherited_report(report):
    assert report["passed"] and report["waivers"] == {}
    assert all(c["status"] == "PASS" for c in report["checks"])
    assert not any(c["name"] in {"moisture_positivity", "supersaturation_bound"} for c in report["checks"])
    source = report["inherited_source"]
    number = source["moisture"]["QNRAIN"]
    assert number["min"] == float(np.float32(-6.19e-29))
    assert number["negative_cells"] == number["outer_ring_negative_cells"] == 2
    assert number["nonfinite_cells"] == 0
    rh = source["relative_humidity_over_liquid"]
    assert rh["max"] == pytest.approx(1.0049, abs=1e-7)
    assert rh["supersaturated_cells"] == rh["outer_ring_supersaturated_cells"] == rh["over_bound_cells"] == 2
    assert rh["nonfinite_cells"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("scoped", [False, True])
def test_verified_transport_reports_inherited_negatives_and_rh_without_waiver(view, scoped):
    fields, meta, _ = view
    inherited_state(fields)
    source = {name: a.copy() for name, a in fields.items()}
    before = {name: a.tobytes() for name, a in fields.items()}
    old_meta = copy.deepcopy(meta)
    changed = ["T", "P", "QVAPOR", "QNRAIN"] if scoped else ()
    result = validate_fields(fields, meta, changed_fields=changed, transport_reference=(source, meta))
    assert_inherited_report(result)
    assert before == {name: a.tobytes() for name, a in fields.items()}
    assert before == {name: a.tobytes() for name, a in source.items()}
    assert meta == old_meta


def test_file_transport_reports_inherited_values_and_preserves_storage(view, tmp_path):
    fields, _, path = view
    inherited_state(fields)
    with Dataset(path, "r+") as ds:
        for name in ("QNRAIN", "QVAPOR"):
            ds[name][:] = fields[name]
    target = tmp_path / "transport"
    shutil.copyfile(path, target)
    before = path.read_bytes(), target.read_bytes()
    assert_inherited_report(validate_wrfinput(target, transport_reference=path))
    assert before == (path.read_bytes(), target.read_bytes())


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
def test_actually_edited_negative_moisture_is_strict_even_with_transport_parameter(view, checkpoint, with_reference):
    fields, meta, _ = view
    source = {name: a.copy() for name, a in fields.items()}
    fields["QNRAIN"][0, 0, 0, 0] = np.float32(-6.19e-29)
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_fields(fields, meta, changed_fields=["QNRAIN"], checkpoint=checkpoint,
                        transport_reference=(source, meta) if with_reference else None)
    assert any(c["name"] == "moisture_positivity" and c["status"] == "FAIL" for c in exc.value.report["checks"])
    assert "inherited_source" not in exc.value.report
    if with_reference:
        assert any(c["name"] == "source_state_transport" and c["status"] == "FAIL" for c in exc.value.report["checks"])


@pytest.mark.parametrize("field,value,check", [("QVAPOR", .1, "supersaturation_bound"), ("T", .05, "hydrostatic_consistency")])
@pytest.mark.parametrize("with_reference", [False, True])
def test_analysis_rh_and_balance_checks_are_not_relaxed(view, field, value, check, with_reference):
    fields, meta, _ = view
    source = {name: a.copy() for name, a in fields.items()}
    fields[field][:] = value
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_fields(fields, meta, changed_fields=[field], checkpoint=True,
                        transport_reference=(source, meta) if with_reference else None)
    assert any(c["name"] == check and c["status"] == "FAIL" for c in exc.value.report["checks"])


@pytest.mark.parametrize("field,value,check", [("QNRAIN", np.nan, "finite"), ("P", np.inf, "finite"),
                                              ("MU", -90000, "mass_geometry"), ("PH", -1e4, "mass_geometry")])
def test_inherited_mode_still_refuses_nonfinite_storage_and_bad_geometry(view, field, value, check):
    fields, meta, _ = view
    if field == "PH":
        fields[field][0, 1] = value
    else:
        fields[field][:] = value
    source = {name: a.copy() for name, a in fields.items()}
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_fields(fields, meta, transport_reference=(source, meta))
    assert any(c["name"] == check and c["status"] == "FAIL" for c in exc.value.report["checks"])


def test_transport_parameter_does_not_allow_false_changed_masks_to_hide_analysis(view):
    fields, meta, _ = view
    source = {name: a.copy() for name, a in fields.items()}
    fields["QNRAIN"][0, 0, 0, 0] = -.01
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_fields(fields, meta, changed_fields={"QNRAIN": np.zeros(fields["QNRAIN"].shape, bool)},
                        transport_reference=(source, meta), checkpoint=True)
    assert any(c["name"] == "moisture_positivity" and c["status"] == "FAIL" for c in exc.value.report["checks"])
