"""Independent WRF header conventions and hand-integrated hydrostatic columns."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

import netCDF4
import numpy as np
import pytest

from gpuwrf.doctor.case import inspect_case, vram_estimate
from gpuwrf.doctor.budget import offline_budget
from gpuwrf.doctor.repro import create_manifest, reproduce

ROOT = Path(__file__).resolve().parents[1]


def times(ds, labels, name="Times"):
    var = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
    var[:] = np.asarray([list(label) for label in labels], dtype="S1")


def variable(ds, name, dims, val, units=""):
    var = ds.createVariable(name, "f8", dims)
    var[0] = val
    var.units = units


@pytest.fixture
def case(tmp_path):
    (tmp_path / "namelist.input").write_text("""&time_control
 start_year=2026, start_month=1, start_day=1, start_hour=0,
 run_hours=1, interval_seconds=3600,
/
&domains
 max_dom=1, time_step=6, e_we=5, e_sn=4, e_vert=3, dx=1000, dy=1000,
/
&physics
 mp_physics=8, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5,
 sf_surface_physics=4, ra_lw_physics=4, ra_sw_physics=4, num_soil_layers=4,
 num_land_cat=3,
/
&dynamics
 diff_opt=1, km_opt=4,
/
&bdy_control
 spec_bdy_width=1,
/
""")
    with netCDF4.Dataset(tmp_path / "wrfinput_d01", "w") as ds:
        for key, size in {"Time": None, "DateStrLen": 19, "west_east": 4, "south_north": 3,
                          "bottom_top": 2, "bottom_top_stag": 3, "west_east_stag": 5,
                          "south_north_stag": 4, "soil_layers_stag": 4}.items():
            ds.createDimension(key, size)
        ds.DX = ds.DY = 1000.
        ds.NUM_LAND_CAT = 3
        ds.NUM_SOIL_CAT = 16
        ds.ISWATER = 2
        ds.MMINLU = "TEST"
        ds.SOURCE = "GFS"
        times(ds, ["2026-01-01_00:00:00"])
        xy = ("Time", "south_north", "west_east")
        xyz = ("Time", "bottom_top", "south_north", "west_east")
        land = np.ones((3,4)); land[0,0] = 0
        lu = 2 - land
        for key, val in {"LANDMASK":land, "XLAND":2-land, "LU_INDEX":lu, "IVGTYP":lu,
                         "ISLTYP":1, "MU":0, "MUB":100000, "MAPFAC_M":1, "RAINC":0, "RAINNC":0}.items():
            variable(ds, key, xy, val)
        variable(ds, "HGT", xy, 0, "m")
        variable(ds, "SST", xy, 280, "K")
        for key,val in (("T",0),("P",0),("PB",90000)):
            variable(ds,key,xyz,val)
        variable(ds,"U",("Time","bottom_top","south_north","west_east_stag"),1)
        variable(ds,"V",("Time","bottom_top","south_north_stag","west_east"),1)
        variable(ds, "QVAPOR", xyz, .01, "kg kg-1")
        for key in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL"):
            variable(ds,key,xyz,0,"kg kg-1")
        for key,val in (("TSLB",280),("SMOIS",.3),("SH2O",.3)):
            variable(ds,key,("Time","soil_layers_stag","south_north","west_east"),val,
                     "K" if key == "TSLB" else "m3 m-3")
        variable(ds, "PH", ("Time","bottom_top_stag","south_north","west_east"), np.arange(3)[:,None,None]*980.665)
        variable(ds, "PHB", ("Time","bottom_top_stag","south_north","west_east"), 0)
        variable(ds, "W", ("Time","bottom_top_stag","south_north","west_east"), 0)
        # Sigma: pressure B(eta)=eta, while its coupling derivative dB/deta=1.
        ds.HYBRID_OPT = 0
        variable(ds, "ZNW", ("Time","bottom_top_stag"), [1,.5,0])
        variable(ds, "C1F", ("Time","bottom_top_stag"), [1,1,1])
        variable(ds, "C2F", ("Time","bottom_top_stag"), [0,0,0])
        variable(ds, "C3F", ("Time","bottom_top_stag"), [1,.5,0])
        variable(ds, "C4F", ("Time","bottom_top_stag"), [0,0,0])
    with netCDF4.Dataset(tmp_path / "wrfbdy_d01", "w") as ds:
        ds.createDimension("Time", None); ds.createDimension("DateStrLen",19)
        times(ds,["2026-01-01_00:00:00"])
        times(ds,["2026-01-01_01:00:00"],"md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        ds.SOURCE = "GFS"
        for name,size in (("bdy_width",1),("south_north",3),("west_east",4),("bottom_top",2),
                          ("south_north_stag",4),("west_east_stag",5),("bottom_top_stag",3)):
            ds.createDimension(name,size)
        for key in ("U","V","T","PH","MU","QVAPOR"):
            for suffix in ("BXS","BXE","BYS","BYE","BTXS","BTXE","BTYS","BTYE"):
                vertical = () if key == "MU" else ("bottom_top_stag" if key == "PH" else "bottom_top",)
                along = ("south_north_stag" if key == "V" else "south_north") if suffix.endswith(("XS","XE")) else ("west_east_stag" if key == "U" else "west_east")
                dims = ("Time","bdy_width",*vertical,along)
                variable(ds,f"{key}_{suffix}",dims,0)
    (tmp_path / "LANDUSE.TBL").write_text("TEST\n3,1, 'parameters'\nSUMMER\n1, .1, 'Forest'\n2, .1, 'Water'\n3, .1, 'Urban'\n")
    return tmp_path


def codes(report):
    return {entry["code"] for entry in report["issues"]}


def test_valid_case_conventions(case):
    report = inspect_case(case)
    assert report["status"] == "PASS", report
    assert report["domains"][0]["advice"]["typical_dt_s"] == 6.
    assert report["domains"][0]["advice"]["initial_vertical_cfl"] == 0.


def test_vram_calibration_matches_recorded_prod_measurement():
    # FINDINGS A58 measured PROD live memory is 4.00 decimal GB.
    prod_cells = (120*70 + 267*117)*44
    assert vram_estimate(prod_cells)["live_gib"] == pytest.approx(4_000_000_000 / 2**30)
    assert "INFERRED" in vram_estimate(prod_cells)["label"]


def test_landmask_failure(case):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.variables["XLAND"][0,0,0] = 1
    report = inspect_case(case)
    assert "LANDMASK_XLAND" in codes(report)
    assert report["status"] == "FAIL"


@pytest.mark.parametrize("name,val", [("QVAPOR",np.nan),("W",np.inf),("HGT",-np.inf),("ISLTYP",99),
                                       ("IVGTYP",4),("LU_INDEX",31),("SMOIS",1.5),("PH",-10)])
def test_bad_fields(case,name,val):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.variables[name][:] = val
    assert inspect_case(case)["status"] == "FAIL"


def test_boundary_coverage(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("run_hours=1", "run_hours=2"))
    assert "FORCING_COVERAGE" in codes(inspect_case(case))


def test_boundary_start(case):
    with netCDF4.Dataset(case / "wrfbdy_d01", "a") as ds:
        ds.variables["Times"][:] = np.asarray([list("2026-01-01_01:00:00")],dtype="S1")
    assert "FORCING_INVALID" in codes(inspect_case(case))


def test_soil_dims_and_grid(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("num_soil_layers=4","num_soil_layers=6").replace("e_we=5","e_we=6"))
    assert {"GRID_DIMENSION","SOIL_LEVELS"} <= codes(inspect_case(case))


def test_source_mismatch_is_heuristic(case):
    with netCDF4.Dataset(case / "wrfbdy_d01", "a") as ds:
        ds.SOURCE = "ECMWF"
    report = inspect_case(case)
    assert "SOURCE_MISMATCH" in codes(report)
    assert report["status"] == "PASS"


def test_sst_update_missing_forcing(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("mp_physics=8", "sst_update=1, mp_physics=8"))
    assert "FORCING_INVALID" in codes(inspect_case(case))


def test_support_reuses_catalog(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("mp_physics=8","mp_physics=999"))
    report = inspect_case(case)
    assert "NAMELIST_SUPPORT" in codes(report)
    entry = next(e for e in report["support"]["options"] if e["key"] == "mp_physics")
    assert entry["status"] == "recognized_fail_closed"


def test_doctor_gecros_refusal_links_upstream_evidence(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text() + "\n&noah_mp\n opt_crop=2,\n/\n")
    report = inspect_case(case)
    assert report["status"] == "FAIL"
    entry = next(e for e in report["support"]["options"] if e["key"] == "opt_crop")
    assert entry["status"] == "recognized_fail_closed"
    assert "EMERG" in entry["reason"] and "8/8" in entry["reason"]
    assert "proofs/s2_gecros/UPSTREAM_WRF_BUGS.md" in entry["reason"]
    assert any(i["code"] == "NAMELIST_SUPPORT" and entry["reason"] in i["message"]
               for i in report["issues"])


def test_terrain_vertical_cfl(case):
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.variables["HGT"][0,:,1:] = 600
        ds.variables["W"][:] = 20
    advice = inspect_case(case)["domains"][0]["advice"]
    assert advice["max_adjacent_terrain_slope"] == pytest.approx(.6)
    assert advice["initial_vertical_cfl"] == pytest.approx(1.2)


def test_manifest_roundtrip_hash_failure_inventory(case,tmp_path):
    manifest = create_manifest(case)
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    assert reproduce(path)["status"] == "PASS"
    with (case / "wrfinput_d01").open("ab") as handle:
        handle.write(b"changed")
    assert reproduce(path)["status"] == "FAIL"
    (case / "wrfinput_d02").write_bytes(b"new input")
    assert not next(c for c in reproduce(path)["checks"] if c["check"] == "input_inventory")["matches"]


def test_manifest_environment_and_metadata(case,monkeypatch):
    manifest = create_manifest(case)
    path = case / "manifest.json"; path.write_text(json.dumps(manifest))
    monkeypatch.setenv("GPUWRF_DYN_FP32","0")
    assert reproduce(path)["status"] == "FAIL"
    monkeypatch.setenv("GPUWRF_SECRET_TOKEN","never save this")
    assert "never save this" not in json.dumps(create_manifest(case))
    manifest["namelist"]["domains"]["time_step"] = 99
    path.write_text(json.dumps(manifest))
    assert reproduce(path)["status"] == "FAIL"


def test_manifest_relocated_case(case):
    path = case / "manifest.json"
    path.write_text(json.dumps(create_manifest(case)))
    relocated = case.parent / (case.name + "_relocated")
    shutil.copytree(case, relocated)
    assert reproduce(path, relocated)["status"] == "PASS"


def test_manifest_standard_sha256_and_same_size_tampering(case):
    # Standard SHA256('abc') test vector, independent of our hashing helper.
    data = case / "wrfinput_d02"
    data.write_bytes(b"abc")
    manifest = create_manifest(case)
    entry = next(e for e in manifest["inputs"] if e["relative_path"] == "wrfinput_d02")
    assert entry["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path = case / "manifest.json"
    path.write_text(json.dumps(manifest))
    data.write_bytes(b"abd")
    result = reproduce(path)
    assert result["status"] == "FAIL"
    check = next(c for c in result["checks"] if c.get("path") == str(data))
    assert not check["matches"]


def test_manifest_source_and_schema_refusals(case):
    manifest = create_manifest(case)
    path = case / "manifest.json"
    manifest["gpuwrf"]["python_source_sha256"] = "0"*64
    path.write_text(json.dumps(manifest))
    assert reproduce(path)["status"] == "FAIL"
    manifest["schema_version"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="schema"):
        reproduce(path)
    path.write_text("[]")
    with pytest.raises(ValueError,match="schema"):
        reproduce(path)


def test_budget_missing_metadata_clean_error(case,capsys):
    from gpuwrf.cli import main
    with netCDF4.Dataset(case / "wrfinput_d01","a") as ds:
        ds.delncattr("DX")
    assert main(["doctor",str(case),"--budget",str(case / "wrfinput_d01")]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_lcz_detection_does_not_mislabel_nlcd(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("num_land_cat=3", "num_land_cat=41"))
    (case / "LANDUSE.TBL").unlink()
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        ds.NUM_LAND_CAT=41
        ds.variables["LU_INDEX"][0,1,1] = 31
        ds.variables["IVGTYP"][0,1,1] = 31
    report = inspect_case(case)
    assert report["status"] == "PASS"
    assert "LCZ_WITHOUT_URBAN" in codes(report)
    nl.write_text(nl.read_text().replace("mp_physics=8", "use_wudapt_lcz=.true., mp_physics=8"))
    assert inspect_case(case)["status"] == "FAIL"


def test_sst_update_good_and_short(case):
    nl = case / "namelist.input"
    nl.write_text(nl.read_text().replace("mp_physics=8", "sst_update=1, mp_physics=8"))
    with netCDF4.Dataset(case / "wrflowinp_d01","w") as ds:
        for key,size in (("Time",None),("DateStrLen",19),("south_north",3),("west_east",4)):
            ds.createDimension(key,size)
        times(ds,["2026-01-01_00:00:00","2026-01-01_01:00:00"])
        variable(ds,"SST",("Time","south_north","west_east"),280,"K")
        ds.variables["SST"][1] = 280
    assert inspect_case(case)["status"] == "PASS"
    with netCDF4.Dataset(case / "wrflowinp_d01","a") as ds:
        ds.variables["Times"][1] = np.asarray(list("2026-01-01_00:30:00"),dtype="S1")
    assert "FORCING_COVERAGE" in codes(inspect_case(case))


@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
def test_budget_rejects_nonfinite(case,bad):
    with netCDF4.Dataset(case / "wrfinput_d01","a") as ds:
        ds.variables["QVAPOR"][0,0,0,0] = bad
    with pytest.raises(ValueError,match="NaN/Inf"):
        offline_budget([case / "wrfinput_d01"])


def test_budget_hydrostatic_column_independent(case):
    # Twelve 1 km² cells, 100000 Pa/g dry columns. 10 g/kg water;
    # next record +1 g/kg and +2 mm precip. These numbers come from
    # hydrostatic dp/g, NOT another invocation of the production reducer.
    with netCDF4.Dataset(case / "wrfinput_d01", "a") as ds:
        for var in ds.variables.values():
            if var.dimensions[0] == "Time":
                var[1] = var[0]
        ds.variables["Times"][1] = np.asarray(list("2026-01-01_01:00:00"),dtype="S1")
        ds.variables["QVAPOR"][1] = .011
        ds.variables["RAINNC"][1] = 2
    report = offline_budget([case / "wrfinput_d01"])
    dry = 12 * 1e6 * 100000 / 9.80665
    assert report["records"][0]["dry_mass_kg"] == pytest.approx(dry)
    assert report["records"][0]["air_water_kg"] == pytest.approx(dry*.01)
    assert report["changes"]["water_plus_precip_kg"] == pytest.approx(dry*.001+24e6)
    assert report["closure_status"].startswith("UNKNOWN")


@pytest.mark.parametrize("command", ["doctor", "manifest", "reproduce", "budget"])
def test_cli_prejax(case,command):
    # Actively prohibit JAX imports; success is stronger than backend not used.
    path = case / "manifest.json"
    path.write_text(json.dumps(create_manifest(case)))
    arguments = [command, str(path) if command == "reproduce" else str(case)]
    if command == "budget":
        history = case / "history.nc"
        shutil.copyfile(case / "wrfinput_d01",history)
        with netCDF4.Dataset(history,"a") as ds:
            ds.variables["Times"][0] = np.asarray(list("2026-01-01_01:00:00"),dtype="S1")
        arguments = ["doctor",str(case),"--budget",str(case / "wrfinput_d01"),str(history)]
    script = """import sys
import importlib.abc
class BlockJax(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'jax' or fullname.startswith('jax.'):
   raise AssertionError('JAX imported by offline command')
sys.meta_path.insert(0, BlockJax())
from gpuwrf.cli import main
raise SystemExit(main())
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    # Parent pytest import applies defaults. Child gets identical effective flags.
    result = subprocess.run([sys.executable,"-c",script,*arguments],env=env,cwd=ROOT,
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["schema"].startswith("Gpuwrf")
