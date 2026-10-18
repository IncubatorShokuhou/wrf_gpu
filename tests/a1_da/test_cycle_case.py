"""Case construction/admission tests; synthetic arrays test IO, not WRF fidelity."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwrf.assimilation.letkf.cycle_case import write_cycle_case
from gpuwrf.cycling.core import CORE, MP_REQUIRED, cycle, sha256
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

START = datetime(2026, 5, 4)
ANALYSIS = START + timedelta(hours=12)
FORMAT = "%Y-%m-%d_%H:%M:%S"


def _stamps(ds, name, times):
    var = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
    for i, stamp in enumerate(times):
        var[i] = np.frombuffer(stamp.strftime(FORMAT).encode(), dtype="S1")


def _forcing(path, *, lower=False, stop=72):
    times = [START + timedelta(hours=h) for h in range(0, stop + 1, 6)]
    with Dataset(path, "w") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", 2), ("west_east", 3)):
            ds.createDimension(name, size)
        ds.START_DATE = START.strftime(FORMAT)
        ds.SIMULATION_START_DATE = START.strftime(FORMAT)
        ds.unchanged_attribute = "literal metadata"
        _stamps(ds, "Times", times)
        if not lower:
            _stamps(ds, "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", times)
            _stamps(ds, "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", [t + timedelta(hours=6) for t in times])
        fields = ("SST", "SEAICE", "VEGFRA", "ALBBCK") if lower else ("U_BXS", "U_BTXS")
        for name in fields:
            var = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"), zlib=True)
            for i in range(len(times)):
                var[i] = {"SST": 290 + i, "SEAICE": 0, "VEGFRA": 60, "ALBBCK": .2}.get(name, i * .125)
        # Packed/raw and static values must survive without scale/offset rounding.
        var = ds.createVariable("packed", "i2", ("Time", "west_east"))
        var.scale_factor, var.add_offset = .13, 273.15
        var.set_auto_maskandscale(False)
        var[:] = np.arange(len(times) * 3, dtype=np.int16).reshape(-1, 3)
        ds.createVariable("static", "f4", ("west_east",))[:] = [0., -0., .12345678]


def _input(path):
    with Dataset(path, "w") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", 2), ("west_east", 3),
                           ("soil_layers_stag", 4), ("snow_layers_stag", 3), ("snso_layers_stag", 7)):
            ds.createDimension(name, size)
        ds.setncatts({"DX": 9000., "DY": 9000., "MAP_PROJ": 1, "GRID_ID": 1,
                      "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21,
                      "ISWATER": 17, "ISICE": 15, "SF_SURFACE_PHYSICS": 4,
                      "START_DATE": ANALYSIS.strftime(FORMAT), "SIMULATION_START_DATE": ANALYSIS.strftime(FORMAT)})
        _stamps(ds, "Times", [ANALYSIS])
        for name, value in {"XLAT": 28., "XLONG": -16., "HGT": 500., "LANDMASK": 1,
                            "XLAND": 1, "LU_INDEX": 5, "IVGTYP": 5, "ISLTYP": 3, "FVEG": .6}.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = value
        for name, value in (("ZS", [.05, .25, .7, 1.5]), ("DZS", [.1, .3, .6, 1.])):
            ds.createVariable(name, "f4", ("Time", "soil_layers_stag"))[0] = value
        for name in CORE + MP_REQUIRED:
            layer = (("soil_layers_stag",) if name in ("SMOIS", "SH2O", "TSLB") else
                     ("snow_layers_stag",) if name in ("TSNO", "SNICE", "SNLIQ") else
                     ("snso_layers_stag",) if name == "ZSNSO" else ())
            var = ds.createVariable(name, "i4" if name == "ISNOW" else "f4", ("Time",) + layer + ("south_north", "west_east"))
            var[0] = {"SMOIS": .4, "SH2O": .2, "TSLB": 280., "TSK": 280., "TV": 280.,
                      "TG": 280., "TAH": 280., "EAH": 1200., "LAI": 2., "XSAI": .3}.get(name, 0)
        ds["ZSNSO"][0] = np.broadcast_to(np.array([0, 0, 0, -.1, -.4, -1., -2.], np.float32)[:, None, None], (7, 2, 3))


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    root = tmp_path / "source"
    root.mkdir()
    namelist_text = """&time_control
 run_days = 0, run_hours = 72, run_minutes = 0, run_seconds = 0,
 start_year = 3*2026, start_month = 3*5, start_day = 3*4, start_hour = 3*0,
 end_year = 3*2026, end_month = 3*5, end_day = 3*7, end_hour = 3*0,
 interval_seconds = 21600,
 auxinput4_interval = 3*360, auxinput4_inname = 'wrflowinp_d<domain>',
 io_form_auxinput4 = 2, io_form_input = 2, io_form_boundary = 2,
 history_interval = 3*60, frames_per_outfile = 1,
 ! keep this comment and the quoted ! and / below
 history_outname = 'wrfout!/<domain>',
/
&domains
 max_dom = 3, time_step = 54, e_we = 4, e_sn = 3, e_vert = 45,
 dx = 9000, dy = 9000, feedback = 0,
/
&physics
 mp_physics = 8, ra_lw_physics = 4, ra_sw_physics = 4,
 sf_sfclay_physics = 5, sf_surface_physics = 4, bl_pbl_physics = 5,
 cu_physics = 1, sst_update = 1,
/
&dynamics
 hybrid_opt = 2, moist_adv_opt = 1, scalar_adv_opt = 1,
/
&bdy_control
 specified = .true., nested = .false., spec_bdy_width = 5,
/
"""
    # The retained CLI validator still expects one assignment per line.
    (root / "namelist.input").write_text(re.sub(r", (?=[a-z_]+\s*=)", ",\n ", namelist_text))
    _forcing(root / "wrfbdy_d01")
    _forcing(root / "wrflowinp_d01", lower=True)
    (root / "LANDUSE.TBL").write_text("table\n")
    (root / "ozone.formatted").write_text("ozone\n")
    (root / "RRTM_DATA_DBL").write_bytes(b"data")
    (root / "wrfout_d01_ignore").write_bytes(b"never link history")
    _input(root / "wrfinput_d01")
    analysis = tmp_path / "analysis"
    cycle(root / "wrfinput_d01", root / "wrfinput_d01", analysis)
    return root, analysis


def test_admission_roundtrip_and_exact_records(source, tmp_path):
    root, analysis = source
    before = {p: sha256(p) for p in root.iterdir() if p.is_file()}
    out = tmp_path / "cycle"
    manifest = write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, out)
    assert manifest["max_dom"] == 1 and manifest["source_max_dom"] == 3
    assert manifest["start"] == "2026-05-04_12:00:00" and manifest["end"] == "2026-05-05_12:00:00"
    assert manifest["slicing"]["wrfbdy_d01"]["records"] == [2, 3, 4, 5, 6]
    assert manifest["admission"]["cli_dry_run"]["plan"]["init_mode"] == "standalone_native_init"
    assert manifest["admission"]["lower_boundary"]["records"] == 5
    assert manifest["admission"]["land_cycle_marker"]["fields"] >= len(CORE + MP_REQUIRED)
    assert json.loads((out / "cycle_case_manifest.json").read_text()) == manifest
    assert not (out / "scratch").exists() and not (out / "forecast").exists()
    assert not (out / "wrfout_d01_ignore").exists()
    assert sha256(out / "wrfinput_d01") == sha256(analysis)
    for name in ("LANDUSE.TBL", "ozone.formatted", "RRTM_DATA_DBL"):
        assert (out / name).is_symlink() and sha256(out / name) == before[root / name]
    old, new = parse_namelist(root / "namelist.input"), parse_namelist(out / "namelist.input")
    assert new["domains"] == {**old["domains"], "max_dom": 1}
    for name in ("physics", "dynamics", "bdy_control"):
        assert old[name] == new[name]
    assert new["time_control"]["start_hour"] == 12 and new["time_control"]["run_hours"] == 24
    assert "! keep this comment" in (out / "namelist.input").read_text()
    for filename in ("wrfbdy_d01", "wrflowinp_d01"):
        with Dataset(root / filename) as src, Dataset(out / filename) as dst:
            src.set_auto_maskandscale(False)
            dst.set_auto_maskandscale(False)
            assert set(src.variables) == set(dst.variables)
            assert dst.START_DATE == "2026-05-04_12:00:00"
            assert dst.unchanged_attribute == src.unchanged_attribute
            for name in src.variables:
                a = np.ascontiguousarray(src[name][2:7] if "Time" in src[name].dimensions else src[name][:])
                b = np.ascontiguousarray(dst[name][:])
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
                assert src[name].ncattrs() == dst[name].ncattrs()
    assert before == {p: sha256(p) for p in before}


@pytest.mark.parametrize("hours", [1, 4, 0, -3, 3.0, True])
def test_invalid_hours(source, tmp_path, hours):
    root, analysis = source
    with pytest.raises(ValueError, match="multiple of 3"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, hours, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_phase_refusals(source, tmp_path):
    root, analysis = source
    with pytest.raises(ValueError, match="wrfbdy interval"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS + timedelta(hours=1), 24, tmp_path / "out")
    path = root / "namelist.input"
    path.write_text(path.read_text().replace("3*360", "3*1440"))
    with pytest.raises(ValueError, match="auxinput4 interval"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "out")


@pytest.mark.parametrize("filename", ["wrfbdy_d01", "wrflowinp_d01"])
def test_short_coverage(source, tmp_path, filename):
    root, analysis = source
    (root / filename).unlink()
    _forcing(root / filename, lower=filename == "wrflowinp_d01", stop=30)
    with pytest.raises(ValueError, match="coverage too short"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "out")


def test_boundary_metadata_refusal(source, tmp_path):
    root, analysis = source
    with Dataset(root / "wrfbdy_d01", "r+") as ds:
        ds["md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"][2] = ds["Times"][2]
    with pytest.raises(ValueError, match="disagrees"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "out")


@pytest.mark.parametrize("domain", ["d02", "two"])
def test_nested_analysis_refusal(source, tmp_path, domain):
    root, analysis = source
    mapping = {domain: analysis} if domain == "d02" else {"d01": analysis, "d02": analysis}
    with pytest.raises(ValueError, match="max_dom=1"):
        write_cycle_case(root, mapping, ANALYSIS, 24, tmp_path / "out")


def test_never_overwrite_and_failed_admission_atomic(source, tmp_path):
    root, analysis = source
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, out)
    out.rmdir()
    with Dataset(root / "wrflowinp_d01", "r+") as ds:
        ds["SEAICE"][3] = 1
    with pytest.raises(ValueError, match="SEAICE"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, out)
    assert not out.exists() and not list(tmp_path.glob(".out.building-*"))


def test_marker_validation(source, tmp_path):
    root, analysis = source
    with Dataset(analysis, "r+") as ds:
        ds["TSLB"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="provenance"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_analysis_clock_attributes_refused(source, tmp_path):
    root, analysis = source
    with Dataset(analysis, "r+") as ds:
        ds.START_DATE = START.strftime(FORMAT)
    with pytest.raises(ValueError, match="START_DATE"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "out")


def test_source_alias_cannot_be_output_parent(source, tmp_path):
    root, analysis = source
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="read-only source"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, alias / "out")
    assert not (root / "out").exists()


def test_publish_race_does_not_clobber(source, tmp_path, monkeypatch):
    import gpuwrf.assimilation.letkf.cycle_case as module
    root, analysis = source
    out = tmp_path / "out"
    publish = module._publish
    def raced(temporary, output):
        output.mkdir()
        publish(temporary, output)
    monkeypatch.setattr(module, "_publish", raced)
    with pytest.raises(FileExistsError):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, out)
    assert out.is_dir() and not list(out.iterdir())
    assert not list(tmp_path.glob(".out.building-*"))
