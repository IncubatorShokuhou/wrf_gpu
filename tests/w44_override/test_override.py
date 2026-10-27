"""Independent known answers for the file seam, not forecast fidelity tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.field_override import override_inputs, sha256
from gpuwrf.io.wrfinput_consistency import validate_wrfinput, WrfinputConsistencyError

WAIVER = {"hydrostatic_consistency": "controlled sensitivity experiment; not a balanced initialization"}
XY = ("Time", "south_north", "west_east")
XYZ = ("Time", "bottom_top", "south_north", "west_east")


def make_input(path, nest=1):
    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in {"Time": None, "DateStrLen": 19, "south_north": 2, "west_east": 3,
                           "bottom_top": 2, "bottom_top_stag": 3, "soil_layers_stag": 4}.items():
            ds.createDimension(name, size)
        ds.setncatts({"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21, "GRID_ID": nest,
                      "DX": 9000. / nest, "DY": 9000. / nest, "USE_THETA_M": 1})
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.frombuffer(b"2023-02-27_00:00:00", "S1")
        for name in ("T", "P", "PB", "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "QNRAIN"):
            v = ds.createVariable(name, "f4", XYZ)
            v.units = "K" if name == "T" else "Pa" if name in {"P", "PB"} else "kg kg-1"
            v[0] = {"T": 0, "P": 100, "PB": 89900, "QVAPOR": .003}.get(name, 0)
        for name in ("PH", "PHB"):
            ds.createVariable(name, "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))[0] = np.array([0, 1000, 2000])[:, None, None]
        for name in ("MU", "MUB", "HGT", "SST", "TSK", "LANDMASK", "XLAND", "LAI", "VEGFRA", "EMISS", "ALBBCK", "ZNT"):
            value = {"MU": 100, "MUB": 80000, "HGT": 500, "SST": 290, "TSK": 289, "LAI": 2,
                     "VEGFRA": 60, "EMISS": .95, "ALBBCK": .2, "ZNT": .1}.get(name, 1)
            v = ds.createVariable(name, "f4", XY)
            v[0] = value
        ds["LANDMASK"][0] = [[1, 0, 1], [1, 1, 0]]
        ds["XLAND"][0] = [[1, 2, 1], [1, 1, 2]]
        for name in ("LU_INDEX", "IVGTYP", "ISLTYP"):
            ds.createVariable(name, "i4", XY)[0] = ([[5, 17, 5], [5, 5, 21]] if name != "ISLTYP" else 1)
        for name, value in (("SMOIS", .2), ("TSLB", 280)):
            ds.createVariable(name, "f4", ("Time", "soil_layers_stag", "south_north", "west_east"))[0] = value
    return path


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for nest in (1, 2):
        make_input(root / f"wrfinput_d{nest:02d}", nest)
    return root


def run(case, tmp_path, field, values, **kwargs):
    item = {"nest": 2, "field": field, "values": values, **kwargs}
    spec = {"overrides": [item]}
    if field in {"T", "P", "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP"}:
        spec["waivers"] = WAIVER
    return override_inputs(case, tmp_path / "out", spec)


def test_empty_spec_is_whole_file_byte_identity(case, tmp_path):
    source = {p.name: p.read_bytes() for p in case.iterdir()}
    report = override_inputs(case, tmp_path / "out", {})
    for name, raw in source.items():
        assert (tmp_path / "out" / name).read_bytes() == raw
        assert (case / name).read_bytes() == raw
        assert report["files"][name]["sha256_before"] == report["files"][name]["sha256_after"]
    assert json.loads((tmp_path / "out" / "OVERRIDE_PROTOCOL.json").read_text()) == report


@pytest.mark.parametrize("op,value,expected", [("set", 281., 281.), ("scale", 2., 578.), ("add", -2., 287.)])
def test_per_nest_known_answers_and_untouched_storage(case, tmp_path, op, value, expected):
    before = {p.name: sha256(p) for p in case.iterdir()}
    report = run(case, tmp_path, "TSK", value, op=op, mask=[[1, 0, 0], [0, 1, 0]])
    assert sha256(tmp_path / "out" / "wrfinput_d01") == before["wrfinput_d01"]
    assert {p.name: sha256(p) for p in case.iterdir()} == before
    with Dataset(tmp_path / "out" / "wrfinput_d02") as result, Dataset(case / "wrfinput_d02") as original:
        np.testing.assert_array_equal(result["TSK"][0], [[expected, 289, 289], [289, expected, 289]])
        assert set(result.variables) == set(original.variables)
        assert result.ncattrs() == original.ncattrs()
        for name in original.variables:
            assert result[name].dimensions == original[name].dimensions
            assert result[name].dtype == original[name].dtype
            assert result[name].ncattrs() == original[name].ncattrs()
            if name != "TSK":
                assert result[name][:].tobytes() == original[name][:].tobytes()
    edit = report["files"]["wrfinput_d02"]["edits"][0]
    assert edit["cells_changed"] == 2
    assert edit["sha256_before"] != edit["sha256_after"]
    assert all(c["status"] == "PASS" for c in edit["checks"])


@pytest.mark.parametrize("field,value,mask,check", [
    ("QVAPOR", -.01, None, "moisture_positivity"),
    ("QCLOUD", -.001, None, "moisture_positivity"),
    ("QICE", -.001, None, "moisture_positivity"),
    ("QNRAIN", -1., None, "moisture_positivity"),
    ("QVAPOR", .1, None, "supersaturation_bound"),
    ("SMOIS", .34, "land", "soil_porosity"),  # STAS sand porosity=.339
    ("SMOIS", -.01, "land", "soil_porosity"),
    ("SST", 300., None, "land_water_mask"),
    ("TSLB", 282., None, "land_water_mask"),
    ("SMOIS", .25, None, "land_water_mask"),
    ("ISLTYP", 20, None, "modis_classes"),
    ("LU_INDEX", 22, None, "modis_classes"),
    ("IVGTYP", 22, None, "modis_classes"),
    ("VEGFRA", 101, "land", "field_ranges"),
    ("LAI", -1, "land", "field_ranges"),
    ("TSK", 0, None, "field_ranges"),
])
def test_mandatory_check_refuses_transactionally(case, tmp_path, field, value, mask, check):
    before = {p.name: sha256(p) for p in case.iterdir()}
    kwargs = {} if mask is None else {"mask": mask}
    with pytest.raises(WrfinputConsistencyError) as exc:
        run(case, tmp_path, field, value, **kwargs)
    assert any(c["name"] == check and c["status"] == "FAIL" for c in exc.value.report["checks"])
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.override-*"))
    assert {p.name: sha256(p) for p in case.iterdir()} == before


@pytest.mark.parametrize("field", ["T", "P", "QVAPOR", "QCLOUD", "QICE"])
def test_hydrostatic_edits_require_explicit_recorded_waiver(case, tmp_path, field):
    spec = {"overrides": [{"nest": 1, "field": field, "op": "add", "values": .001}]}
    with pytest.raises(WrfinputConsistencyError, match="hydrostatic_consistency"):
        override_inputs(case, tmp_path / "refused", spec)
    spec["waivers"] = WAIVER
    report = override_inputs(case, tmp_path / "accepted", spec)
    check = [c for c in report["files"]["wrfinput_d01"]["validation"]["checks"] if c["name"] == "hydrostatic_consistency"]
    assert check == [{"name": "hydrostatic_consistency", "status": "WAIVED", "detail":
        "T/P/moist-mass edits require real.exe-consistent MU/PH rebalance; unavailable (explicit waiver required)"}]
    assert report["files"]["wrfinput_d01"]["validation"]["waivers"] == WAIVER


def test_named_water_mask_and_porosity_class_known_answers(case, tmp_path):
    report = run(case, tmp_path, "SST", 2., op="add", mask="water")
    with Dataset(tmp_path / "out" / "wrfinput_d02") as ds:
        np.testing.assert_array_equal(ds["SST"][0], [[290, 292, 290], [290, 290, 292]])
    assert report["files"]["wrfinput_d02"]["edits"][0]["cells_changed"] == 2
    with Dataset(case / "wrfinput_d01", "r+") as ds:
        ds["ISLTYP"][:] = 2
        ds["SMOIS"][:] = .421
    changed = np.broadcast_to(np.array([[[1, 0, 1], [1, 1, 0]]], dtype=bool), (1, 4, 2, 3))
    result = validate_wrfinput(case / "wrfinput_d01", changed_fields={"SMOIS": changed})
    assert next(c for c in result["checks"] if c["name"] == "soil_porosity")["status"] == "PASS"
    assert result["soil_table"]["sha256"] == sha256(result["soil_table"]["path"])


def test_modis_alias_and_land_water_checks(case, tmp_path):
    spec = {"overrides": [{"nest": 1, "field": n, "values": 6, "mask": "land"} for n in ("LU_INDEX", "IVGTYP")]}
    assert override_inputs(case, tmp_path / "out", spec)["files"]["wrfinput_d01"]["validation"]["passed"]
    with Dataset(case / "wrfinput_d01", "r+") as ds:
        ds["LU_INDEX"][0, 0, 0] = 17
        ds["IVGTYP"][0, 0, 0] = 17
    with pytest.raises(WrfinputConsistencyError, match="water/lake"):
        validate_wrfinput(case / "wrfinput_d01", changed_fields=["LU_INDEX"])


@pytest.mark.parametrize("field", ["EMISS", "ALBBCK", "ZNT"])
def test_noahmp_no_effect_refusal_and_forced_warning(case, tmp_path, field):
    spec = {"overrides": [{"nest": 1, "field": field, "values": .5}]}
    with pytest.raises(ValueError, match="NO EFFECT under Noah-MP; use --surface-controls"):
        override_inputs(case, tmp_path / "refused", spec)
    with pytest.warns(UserWarning, match="NO EFFECT"):
        report = override_inputs(case, tmp_path / "forced", spec, force_no_effect=True)
    assert report["force_no_effect"]
    assert "NO EFFECT" in report["files"]["wrfinput_d01"]["edits"][0]["warning"]


@pytest.mark.parametrize("field", ["ALBEDO", "SH2O"])
def test_rebuilt_unsupported_fields_explain_surface_controls(case, tmp_path, field):
    for forced in (False, True):
        with pytest.raises(ValueError, match="recomputed by Noah-MP.*use --surface-controls") as exc:
            override_inputs(case, tmp_path / "out", {"overrides": [{"nest": 1, "field": field, "values": .4}]}, force_no_effect=forced)
        if field == "SH2O":
            assert "SMOIS + TSLB" in str(exc.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("waived", [False, True])
def test_glacier_class_change_rechecks_retained_saturated_soil(case, tmp_path, waived):
    with Dataset(case / "wrfinput_d01", "r+") as ds:
        for name in ("LU_INDEX", "IVGTYP"):
            ds[name][0, 0, 0] = 15
        ds["SMOIS"][0, :, 0, 0] = 1
    spec = {"overrides": [{"nest": 1, "field": n, "values": 5, "mask": [[1, 0, 0], [0, 0, 0]]}
                          for n in ("LU_INDEX", "IVGTYP")]}
    if waived:
        spec["waivers"] = {"soil_porosity": "deliberately retain saturated soil for controlled class experiment"}
        report = override_inputs(case, tmp_path / "out", spec)
        checks = report["files"]["wrfinput_d01"]["validation"]["checks"]
        assert any(c["name"] == "soil_porosity" and c["status"] == "WAIVED" for c in checks)
    else:
        with pytest.raises(WrfinputConsistencyError, match="soil_porosity"):
            override_inputs(case, tmp_path / "out", spec)
        assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["HGT", "PH", "PHB", "MUB", "U", "NOT_A_FIELD"])
def test_unsupported_fields_refused(case, tmp_path, field):
    with pytest.raises(ValueError, match="unsupported field"):
        run(case, tmp_path, field, 1)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1.e100])
def test_nonfinite_or_storage_overflow_refused(case, tmp_path, value):
    with pytest.raises(ValueError, match="finite|NaN/Inf"):
        run(case, tmp_path, "TSK", value)
    assert not (tmp_path / "out").exists()


def test_nonfinite_checked_dependencies_cannot_be_waived(case):
    path = case / "wrfinput_d01"
    with Dataset(path, "r+") as ds:
        ds["PB"][0, 0, 0, 0] = np.nan
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_wrfinput(path, changed_fields=["T"], waivers=WAIVER)
    assert any(c["name"] == "finite" and c["status"] == "FAIL" for c in exc.value.report["checks"])
    for name in ("finite", "schema", "source_state_transport", "unknown"):
        with pytest.raises(ValueError, match="waiver"):
            validate_wrfinput(path, waivers={name: "cannot waive"})


def test_noop_thermo_does_not_require_waiver_and_byte_exact(case, tmp_path):
    report = override_inputs(case, tmp_path / "out", {"overrides": [{"nest": 2, "field": "T", "values": 0., "op": "add"}]})
    assert sha256(case / "wrfinput_d02") == sha256(tmp_path / "out" / "wrfinput_d02")
    assert report["files"]["wrfinput_d02"]["edits"][0]["cells_changed"] == 0


def test_file_values_and_mask_provenance(case, tmp_path):
    np.save(tmp_path / "values.npy", np.full((2, 3), 285., np.float32))
    np.savez(tmp_path / "mask.npz", selected=[[1, 0, 0], [0, 0, 0]])
    spec = {"overrides": [{"nest": 2, "field": "TSK", "file": "values.npy", "mask": {"file": "mask.npz", "variable": "selected"}}]}
    report = override_inputs(case, tmp_path / "out", spec, spec_base=tmp_path)
    edit = report["files"]["wrfinput_d02"]["edits"][0]
    assert edit["provenance"]["sha256"] == sha256(tmp_path / "values.npy")
    assert edit["mask"]["sha256"] == sha256(tmp_path / "mask.npz")
    assert edit["cells_changed"] == 1
    with Dataset(tmp_path / "out" / "wrfinput_d02") as ds:
        np.testing.assert_array_equal(ds["TSK"][0], [[285, 289, 289], [289, 289, 289]])


@pytest.mark.parametrize("patch", [{"nest": 0}, {"nest": True}, {"nest": "d01"}, {"nest": 3},
    {"values": [1, 2]}, {"op": "multiply"}, {"typo": 1}, {"mask": [1, 2, 0]},
    {"file": "extra.npy"}, {"target": "wrfbdy"}, {"field": "ISLTYP", "values": 1.5}])
def test_malformed_spec_refuses_without_output(case, tmp_path, patch):
    item = {"nest": 1, "field": "TSK", "values": 280, **patch}
    with pytest.raises((ValueError, OSError)):
        override_inputs(case, tmp_path / "out", {"overrides": [item]})
    assert not (tmp_path / "out").exists()


def test_refusal_never_replaces_existing_destination(case, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "user-file").write_text("retained")
    with pytest.raises(ValueError):
        override_inputs(case, out, {})
    assert (out / "user-file").read_text() == "retained"


def test_source_transport_retains_nonhydrostatic_residual_without_waiver(case, tmp_path):
    source, target = case / "wrfinput_d01", tmp_path / "transport"
    shutil.copyfile(source, target)
    with Dataset(target, "r+") as ds:
        # Small base changes and the exact opposite perturbation retain totals.
        ds["PB"][:] += np.float32(.0078125)
        ds["P"][:] -= np.float32(.0078125)
    report = validate_wrfinput(target, changed_fields=["T", "P", "QVAPOR"], transport_reference=source)
    assert report["passed"] and report["waivers"] == {}
    check = next(c for c in report["checks"] if c["name"] == "hydrostatic_consistency")
    assert check["status"] == "PASS" and check["P_total_max_delta"] == 0


@pytest.mark.parametrize("field,delta", [("T", 1), ("QVAPOR", .001), ("PH", 100), ("P", 10),
                                      ("MU", 100), ("HGT", 1), ("PHB", 100), ("PB", 10), ("MUB", 100)])
def test_source_transport_cannot_justify_analysis_or_material_base_changes(case, tmp_path, field, delta):
    source, target = case / "wrfinput_d01", tmp_path / "transport"
    shutil.copyfile(source, target)
    with Dataset(target, "r+") as ds:
        ds[field][:] += delta
    with pytest.raises(WrfinputConsistencyError, match="source_state_transport") as exc:
        validate_wrfinput(target, changed_fields=["T", "P", "QVAPOR"], transport_reference=source, waivers=WAIVER)
    assert any(c["name"] == "source_state_transport" and c["status"] == "FAIL" for c in exc.value.report["checks"])


@pytest.mark.parametrize("field,value", [("MU", -90000), ("PH", -100000)])
def test_invalid_positive_mass_or_layer_thickness_refused(case, field, value):
    path = case / "wrfinput_d01"
    with Dataset(path, "r+") as ds:
        if field == "MU":
            ds[field][:] = value
        else:
            ds[field][0, 1] = value
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_wrfinput(path)
    assert any(c["name"] == "mass_geometry" and c["status"] == "FAIL" for c in exc.value.report["checks"])


def test_restart_current_slots_and_ambiguous_refusal(case, tmp_path):
    source, target = tmp_path / "restart", case / "wrfinput_d01"
    shutil.copyfile(target, source)
    with Dataset(source, "r+") as ds:
        for name in ("PH", "MU"):
            ds.renameVariable(name, name + "_2")
    assert validate_wrfinput(target, transport_reference=source)["passed"]
    with Dataset(source, "r+") as ds:
        ds.createVariable("MU", "f4", XY)[:] = 200
    with pytest.raises(WrfinputConsistencyError, match="ambiguous current slot"):
        validate_wrfinput(target, transport_reference=source)


def test_aux4_multiple_records_and_water_footprint(case, tmp_path):
    with Dataset(case / "wrflowinp_d02", "w") as ds:
        for dim, size in (("Time", 3), ("south_north", 2), ("west_east", 3)):
            ds.createDimension(dim, size)
        ds.createVariable("SST", "f4", XY)[:] = 290
    mask = "water"
    spec = {"overrides": [{"nest": 2, "target": "wrflowinp", "field": "SST", "op": "add", "values": 1, "mask": mask}]}
    report = override_inputs(case, tmp_path / "out", spec)
    assert report["files"]["wrflowinp_d02"]["edits"][0]["cells_changed"] == 6
    with Dataset(tmp_path / "out" / "wrflowinp_d02") as ds:
        np.testing.assert_array_equal(ds["SST"][:], np.broadcast_to([[290, 291, 290], [290, 290, 291]], (3, 2, 3)))
    spec["overrides"][0].pop("mask")
    with pytest.raises(WrfinputConsistencyError, match="land_water_mask"):
        override_inputs(case, tmp_path / "refused", spec)


@pytest.mark.parametrize("field", ["T", "QVAPOR", "TSK", "SMOIS", "PHB"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_empty_spec_rejects_nonfinite_retained_fields(case, tmp_path, field, value):
    with Dataset(case / "wrfinput_d01", "r+") as ds:
        key = (0,) * ds[field].ndim
        ds[field][key] = value
    with pytest.raises(WrfinputConsistencyError) as exc:
        override_inputs(case, tmp_path / "out", {})
    assert any(c["name"] == "finite" and c["status"] == "FAIL" for c in exc.value.report["checks"])


def test_porosity_glacier_exception_matches_noahmp_init(case):
    path = case / "wrfinput_d01"
    with Dataset(path, "r+") as ds:
        for name in ("LU_INDEX", "IVGTYP"):
            ds[name][0, 0, 0] = 15
        ds["SMOIS"][0, :, 0, 0] = 1
    mask = np.zeros((1, 4, 2, 3), bool)
    mask[0, :, 0, 0] = True
    assert validate_wrfinput(path, changed_fields={"SMOIS": mask})["passed"]
    with pytest.raises(WrfinputConsistencyError, match="soil_porosity"):
        validate_wrfinput(path, changed_fields={"SMOIS": mask}, surface_physics=2)


def test_schema_refusal_is_not_waivable(case):
    path = case / "wrfinput_d01"
    with Dataset(path, "r+") as ds:
        ds.renameVariable("PHB", "PHB_deleted")
    with pytest.raises(WrfinputConsistencyError) as exc:
        validate_wrfinput(path)
    assert any(c["name"] == "schema" and c["status"] == "FAIL" for c in exc.value.report["checks"])


def test_rh_report_uses_dry_theta_with_wrf_constants(case):
    # Literal pressure=90000, dry theta=300: physical T=291.103673555 K.
    # Independently evaluate vapour pressure with WRF ep_2=287/461.6.
    report = validate_wrfinput(case / "wrfinput_d01", changed_fields=["QVAPOR"], waivers=WAIVER)
    rh = next(c for c in report["checks"] if c["name"] == "supersaturation_bound")["max_rh"]
    temperature = 300 * .9 ** (2 / 7)
    expected = (90000 * float(np.float32(.003)) / (287 / 461.6 + float(np.float32(.003)))) / (611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65)))
    assert rh == pytest.approx(expected, rel=1e-12)


def test_cli_is_backend_dark_for_actual_edit(case, tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("overrides:\n  - nest: 1\n    field: TSK\n    op: add\n    values: 1\n")
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax','jaxlib','cupy','torch','triton'}:
            raise AssertionError('backend import: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['gpuwrf','override',*sys.argv[1:]]
from gpuwrf.cli import main
raise SystemExit(main())
"""
    result = subprocess.run([sys.executable, "-c", code, "--input-dir", str(case), "--output-dir", str(tmp_path / "out"), "--spec", str(spec)],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0",
             "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["files"]["wrfinput_d01"]["edits"][0]["cells_changed"] == 6
