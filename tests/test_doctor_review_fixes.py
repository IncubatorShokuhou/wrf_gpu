"""rv-u1-doctor's five findings: pristine-WRF coefficient/axis/Registry semantics."""
import json
import shutil

import netCDF4
import numpy as np
import pytest

from test_doctor import case, codes, inspect_case, create_manifest, reproduce, offline_budget


def second_record(case):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        for var in list(ds.variables.values()):
            if var.dimensions[0] == "Time":
                var[1] = var[0]
        ds.variables["Times"][1] = np.asarray(list("2026-01-01_01:00:00"), dtype="S1")


def test_wrf_sigma_derivatives_are_not_pressure(case):
    second_record(case)
    # Pristine initialize_ideal sets C1F=1, C2F=0 for sigma. Face pressure
    # is eta*mu+ptop; a uniform .01 kg/kg column is .01*column dry mass.
    result = offline_budget([case / "wrfinput_d01"])
    assert result["records"][0]["air_water_kg"] == pytest.approx(12e6*100000/9.80665*.01)


def test_wrf_hybrid_pressure_weights(case):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.HYBRID_OPT = 2
        # B=eta², eta=(1,.5,0), reference pressure span 90000 Pa.
        # Pristine C3=B, C4=(eta-B)*90000; C1=dB/deta, C2=(1-C1)*90000.
        ds.variables["C3F"][0] = [1,.25,0]
        ds.variables["C4F"][0] = [0,22500,0]
        ds.variables["C1F"][0] = [2,1,0]
        ds.variables["C2F"][0] = [-90000,0,90000]
        ds.variables["QVAPOR"][0,0] = .02
        ds.variables["QVAPOR"][0,1] = .004
    second_record(case)
    result = offline_budget([case / "wrfinput_d01"])
    # Independently integrated faces: 100000,47500,0 Pa (+ constant ptop).
    expected = 12e6 / 9.80665 * (52500*.02 + 47500*.004)
    assert result["records"][0]["air_water_kg"] == pytest.approx(expected)


def test_explicit_sigma_znw_fallback(case):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.renameVariable("C3F", "unused_C3F")
        ds.renameVariable("C4F", "unused_C4F")
    second_record(case)
    result = offline_budget([case / "wrfinput_d01"])
    assert result["records"][0]["air_water_kg"] == pytest.approx(12e6*100000/9.80665*.01)
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.HYBRID_OPT = 2
    with pytest.raises(ValueError, match="requires C3F/C4F"):
        offline_budget([case / "wrfinput_d01"])


@pytest.mark.parametrize("faces", [[1,1,0], [0,.5,1], [1,np.nan,0], [1,np.inf,0]])
def test_reject_invalid_pressure_layers(case,faces):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.variables["C3F"][0] = faces
    with pytest.raises(ValueError, match="positive|NaN/Inf"):
        offline_budget([case / "wrfinput_d01"])


@pytest.mark.parametrize("mode", ["same_size", "added", "removed"])
def test_mptable_manifest_inventory(case,mode):
    table = case / "MPTABLE.TBL"
    if mode != "added":
        table.write_text("old")
    manifest = create_manifest(case)
    if mode != "added":
        assert any(e["relative_path"] == "MPTABLE.TBL" for e in manifest["inputs"])
    path = case / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert reproduce(path)["status"] == "PASS"
    if mode == "removed":
        table.unlink()
    else:
        table.write_text("new")
    result = reproduce(path)
    assert result["status"] == "FAIL"
    if mode != "same_size":
        assert not next(c for c in result["checks"] if c["check"] == "input_inventory")["matches"]


@pytest.mark.parametrize("axis", ["south_north", "west_east_stag", "bottom_top_stag"])
def test_boundary_axis_names(case,axis):
    with netCDF4.Dataset(case / "wrfbdy_d01", "a") as ds:
        ds.renameDimension(axis, "wrong_axis")
    result = inspect_case(case)
    assert "FORCING_INVALID" in codes(result)
    assert result["status"] == "FAIL"


def test_boundary_v_stagger_is_required(case):
    with netCDF4.Dataset(case / "wrfbdy_d01", "a") as ds:
        ds.renameVariable("V_BXS", "former_V_BXS")
        var = ds.createVariable("V_BXS", "f8", ("Time","bdy_width","bottom_top","south_north"))
        var[:] = 0
    assert "FORCING_INVALID" in codes(inspect_case(case))


def test_boundary_staggered_length_matches_ic(case):
    with netCDF4.Dataset(case / "wrfbdy_d01", "a") as ds:
        ds.renameDimension("south_north_stag", "former_y_stag")
        ds.createDimension("south_north_stag", 5)  # IC has four.
        for suffix in ("BXS","BXE","BTXS","BTXE"):
            name = "V_" + suffix
            ds.renameVariable(name, "former_" + name)
            var = ds.createVariable(name,"f8",("Time","bdy_width","bottom_top","south_north_stag"))
            var[:] = 0
    assert "FORCING_INVALID" in codes(inspect_case(case))


def test_boundary_width_matches_namelist(case):
    path = case / "namelist.input"
    path.write_text(path.read_text().replace("spec_bdy_width=1", "spec_bdy_width=2"))
    assert "FORCING_INVALID" in codes(inspect_case(case))


def test_accumulated_shallow_rain_is_included(case):
    second_record(case)
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        rain = ds.createVariable("RAINSH","f8",("Time","south_north","west_east"))
        rain[0] = 0
        rain[1] = 2
    result = offline_budget([case / "wrfinput_d01"])
    assert result["changes"]["precip_accumulated_kg"] == pytest.approx(24_000_000.)
    assert result["changes"]["water_plus_precip_kg"] == pytest.approx(24_000_000.)
    assert "RAINSH" in result["records"][0]["included_precip_accumulators"]


def test_omitted_parent_time_step_ratio_is_one(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("max_dom=1", "max_dom=2, parent_id=1,1"))
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.variables["W"][:] = 20
    shutil.copyfile(case / "wrfinput_d01",case / "wrfinput_d02")
    report = inspect_case(case)
    assert report["status"] == "PASS", report
    child = report["domains"][1]["advice"]
    assert child["dt_s"] == 6  # Registry.EM_COMMON:2323, default 1.
    assert child["initial_vertical_cfl"] == pytest.approx(1.2)


def test_nontrivial_parent_time_step_chain(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("max_dom=1", "max_dom=4, parent_id=1,1,1,2, parent_time_step_ratio=1,2,3,4"))
    for domain in ("d02","d03","d04"):
        shutil.copyfile(case / "wrfinput_d01",case / ("wrfinput_" + domain))
    report = inspect_case(case)
    assert report["status"] == "PASS", report
    # d03 attaches to root, d04 to d02: do not divide by every preceding nest.
    assert [d["advice"]["dt_s"] for d in report["domains"]] == [6,3,2,.75]
