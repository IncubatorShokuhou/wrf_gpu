"""Independent projection/forcing/CPU-preprocessor gates; no GPU required."""
from datetime import datetime
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from gpuwrf.cli import main
from gpuwrf.onboard import forcing
from gpuwrf.onboard.pipeline import _metgrid_levels, check_real
from gpuwrf.onboard.region import namelists, plan_region, profiles


def plan(**kw):
    return plan_region(kw.pop("bbox", [-17, 28, -16, 28.6]), kw.pop("resolution_km", 9),
                       kw.pop("date", "2026-10-12T00"), kw.pop("hours", 3), **kw)


@pytest.mark.parametrize("bbox", [[-17, 28, -16, 28.6], [7, 46, 8, 47], [150, -34, 151, -33]])
def test_projection_against_proj(bbox):
    # Independent PROJ implementation, on WRF's spherical Earth.
    from pyproj import Proj
    p = plan(bbox=bbox)
    proj = Proj(proj="merc", R=6370000, lat_ts=p["truelat1"], lon_0=p["ref_lon"])
    x0, y0 = proj(bbox[0], bbox[1])
    x1, y1 = proj(bbox[2], bbox[3])
    _, cy = proj(p["ref_lon"], p["ref_lat"])
    assert p["nx"] * p["dx_m"] >= x1 - x0 + 24 * p["dx_m"]
    assert p["ny"] * p["dx_m"] >= y1 - y0 + 24 * p["dx_m"]
    assert cy == pytest.approx((y0 + y1) / 2, abs=1e-5)
    assert p["e_we"] == p["nx"] + 1
    assert 3600 % p["time_step"] == 0


@pytest.mark.parametrize("kwargs", [dict(bbox=[0, 0, float("nan"), 1]), dict(bbox=[170, 0, -170, 1]),
    dict(bbox=[0, 71, 1, 72]), dict(resolution_km=float("inf")), dict(resolution_km=0),
    dict(hours=1), dict(hours=121), dict(date="2026-10-12T01"),
    dict(bbox=[-20, 20, 20, 40], resolution_km=1)])
def test_refuse_bad_plan(kwargs):
    with pytest.raises(ValueError):
        plan(**kwargs)


def test_profiles_independent_measured_denominator():
    # Freeze the actual measured baseline rather than calling the estimator to
    # derive the expected value (E98).
    rows = profiles()
    assert [r["vram_gib"] for r in rows] == [12, 16, 24, 32]
    coefficient = 4e9 / (39639 * 44)
    for row in rows:
        n = row["square_mass_grid"]
        need = 1 + 1.25 * n * n * 44 * coefficient / 1024**3
        assert need <= row["vram_gib"] - 2
        assert 1 + 1.25 * (n + 1)**2 * 44 * coefficient / 1024**3 > row["vram_gib"] - 2
        assert row["label"] == "I" and row["measured_limit"] is False


def test_namelist_binding_and_real_duration(tmp_path):
    from gpuwrf.io.namelist_check import _parse_wrf_namelist, validate_operational_namelist
    p = plan(date="2026-12-31T18", hours=9)
    wps, wrf = namelists(p, tmp_path, 35)
    path = tmp_path / "namelist.input"
    path.write_text(wrf)
    nl = _parse_wrf_namelist(wrf)
    validate_operational_namelist(nl)
    assert nl["domains"]["num_metgrid_levels"] == [35]
    assert nl["time_control"]["end_year"] == [2027]
    assert nl["time_control"]["end_hour"] == [3]
    assert "geog_data_res='lowres'" in wps
    assert nl["physics"]["mp_physics"] == [8]
    assert nl["physics"]["sf_surface_physics"] == [4]


def test_subset_covers_padded_domain_and_endpoint():
    p = plan()
    query = parse_qs(urlparse(forcing.gfs_url(p, 3)).query)
    assert query["file"] == ["gfs.t00z.pgrb2.0p25.f003"]
    assert float(query["leftlon"][0]) < p["domain_bbox"][0]
    assert float(query["toplat"][0]) > p["domain_bbox"][3]
    assert p["boundary_hours"] == [0, 3]
    assert query["all_var"] == ["on"] and query["all_lev"] == ["on"]


class Response(io.BytesIO):
    headers = {}


def test_download_cap_and_html_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(forcing, "urlopen", lambda *a, **k: Response(b"GRIB" + b"x"*20 + b"7777"))
    with pytest.raises(ValueError, match="budget"):
        forcing.download("https://example.com", tmp_path / "x", 16)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(forcing, "urlopen", lambda *a, **k: Response(b"<html>missing cycle</html>"))
    with pytest.raises(ValueError, match="GRIB"):
        forcing.download("https://example.com", tmp_path / "x", 100)
    assert not list(tmp_path.iterdir())


def test_eccodes_numeric_surface_code(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(h, "typeOfFirstFixedSurface", 1)
        path = tmp_path / "sfc.grib2"
        with path.open("wb") as out:
            eccodes.codes_write(h, out)
        # Library's string accessor returns 'sfc'; WPS Vtables use numeric 1.
        assert eccodes.codes_get(h, "typeOfFirstFixedSurface") == "sfc"
        assert forcing.inventory_grib(path)[0]["typeOfFirstFixedSurface"] == 1
    finally:
        eccodes.codes_release(h)


def test_ecmwf_index_missing_soil_is_explicit(monkeypatch):
    records = [dict(param=p, levtype="sfc") for p in ("2t", "2d", "10u", "10v", "sp", "msl", "skt", "lsm", "sd", "z")]
    blob = b"\n".join(json.dumps(r).encode() for r in records)
    monkeypatch.setattr(forcing, "urlopen", lambda *a, **k: Response(blob))
    report = forcing.ecmwf_report(plan(), "aifs")
    assert report["status"] == "MISSING_FIELDS"
    assert len(report["missing"]) == 8
    assert "soil_temperature_layer_4" in report["missing"]
    assert "/aifs-single/" in report["url"]


@pytest.mark.parametrize("layers", [2, 4])
def test_ecmwf_actual_sol_sot_vsw_names(monkeypatch, layers):
    # Exact current parameter/level conventions from the original 20261012 00Z
    # open-data indexes (IFS cf2a59ad..., AIFS dbf1ae49...), not surface-only lists.
    records = [dict(param=p, levtype="sfc") for p in ("2t", "2d", "10u", "10v", "sp", "msl", "skt", "lsm", "sd", "z")]
    records += [dict(param=param, levtype="sol", levelist=str(layer))
                for param in ("sot", "vsw") for layer in range(1, layers + 1)]
    blob = b"\n".join(json.dumps(r).encode() for r in records)
    monkeypatch.setattr(forcing, "urlopen", lambda *a, **k: Response(blob))
    report = forcing.ecmwf_report(plan(), "ifs" if layers == 4 else "aifs")
    if layers == 4:
        assert report["missing"] == []
        assert report["status"] == "UNQUALIFIED_VTABLE"
    else:
        assert sorted(report["missing"]) == [f"{kind}_layer_{layer}" for kind in
                 ("soil_moisture", "soil_temperature") for layer in (3, 4)]


def complete_fields():
    # Construct GRIB metadata from WPS Vtable.GFS, independently of matcher.
    def field(d, c, n, surface, level=0, **kw):
        return dict(discipline=d, parameterCategory=c, parameterNumber=n,
                    typeOfFirstFixedSurface=surface, level=level,
                    validityDate=20261012, validityTime=0, **kw)
    rows = [field(0, c, n, 100, level) for level in (50, 100, 200, 300, 500, 700, 850, 1000)
            for c, n in ((0, 0), (2, 2), (2, 3), (3, 5), (1, 1))]
    rows += [field(0, 0, 0, 103, 2), field(0, 1, 1, 103, 2),
             field(0, 2, 2, 103, 10), field(0, 2, 3, 103, 10),
             field(0, 3, 0, 1), field(2, 0, 218, 1), field(0, 0, 0, 1),
             field(0, 3, 5, 1), field(0, 1, 13, 1)]
    rows += [field(d, c, n, 106, first_depth_m=top, second_depth_m=bottom)
             for d, c, n in ((2, 0, 2), (2, 0, 192))
             for top, bottom in ((0, .1), (.1, .4), (.4, 1), (1, 2))]
    return rows


def test_field_completeness_and_aifs_missing_soil():
    rows = complete_fields()
    valid = datetime(2026, 10, 12)
    assert forcing.field_report(rows, valid)["status"] == "PASS"
    # AIFS/IFS one-level soil is insufficient: no extrapolated deep fields.
    rows = [f for f in rows if f.get("first_depth_m", 0) < .4]
    report = forcing.field_report(rows, valid)
    assert report["status"] == "MISSING_FIELDS"
    assert "soil_moisture_1_2m" in report["missing"]
    assert forcing.field_report(complete_fields(), datetime(2026, 10, 13))["status"] == "MISSING_FIELDS"


def test_profiles_and_plan_cli_are_side_effect_free(tmp_path, capsys):
    assert main(["quickstart", "--profiles"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    assert main(["quickstart", "--bbox", "-17", "28", "-16", "28.6", "--date", "2026-10-12T00",
                 "--hours", "3", "--case-dir", str(tmp_path / "unused")]) == 0
    assert not (tmp_path / "unused").exists()
    assert json.loads(capsys.readouterr().out)["gpu_qualified"] is False


def test_forecast_handoff_keeps_process_and_descriptor(tmp_path, monkeypatch):
    import os
    import sys
    from gpuwrf import cli
    from gpuwrf.onboard import pipeline
    args = cli.build_parser().parse_args(["quickstart", "--bbox", "-17", "28", "-16", "28.6",
        "--date", "2026-10-12T00", "--hours", "3", "--run", "--wrf-root", str(tmp_path)])
    monkeypatch.setattr(pipeline, "prepare", lambda *_: dict(forecast_argv=[sys.executable, "-m", "gpuwrf.cli", "run", "--hours", "3"]))
    pid = os.getpid()
    with (tmp_path / "descriptor").open("w") as stream:
        inode = os.fstat(stream.fileno()).st_ino
        def run_main(argv):
            assert argv == ["run", "--hours", "3"]
            assert os.getpid() == pid
            assert os.fstat(stream.fileno()).st_ino == inode
            assert os.environ["GPUWRF_WRF_ROOT"] == str(tmp_path)
            return 7
        monkeypatch.setattr(cli, "main", run_main)
        monkeypatch.setenv("GPUWRF_WRF_ROOT", "before")
        assert pipeline.command(args) == 7


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["XLAT", "XLONG"])
def test_real_check_rejects_nonfinite(tmp_path, bad, field):
    import netCDF4
    import numpy as np
    p = plan()
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        with netCDF4.Dataset(tmp_path / name, "w") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("DateStrLen", 19)
            t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
            t[:] = np.array(list(p["start"]), "S1")[None]
            if name == "wrfinput_d01":
                ds.createDimension("west_east", p["nx"])
                ds.createDimension("south_north", p["ny"])
                for name_, lo, hi in (("XLAT", 27, 30), ("XLONG", -18, -15)):
                    var = ds.createVariable(name_, "f4", ("Time", "south_north", "west_east"))
                    var[0] = np.linspace(lo, hi, p["nx"])[None, :]
    assert check_real(tmp_path, p)["status"] == "PASS"
    with netCDF4.Dataset(tmp_path / "wrfinput_d01", "a") as ds:
        ds.variables[field][0, 1, 1] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        check_real(tmp_path, p)


def test_geography_archive_rejects_traversal(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import tarfile
    source = Path(__file__).resolve().parents[1] / "scripts/onboarding/download_geog.py"
    spec = importlib.util.spec_from_file_location("download_geog", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("../escaped")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: Response(archive.getvalue()))
    monkeypatch.setattr(sys, "argv", ["download_geog.py", str(tmp_path / "geog")])
    with pytest.raises(ValueError, match="unsafe"):
        module.main()
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "geog/download.tar.gz.part").exists()
