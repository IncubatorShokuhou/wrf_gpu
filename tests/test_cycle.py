"""Independent, literal array-copy reference and original CPU-WRF savepoints."""
from pathlib import Path
import json
import os
import shutil
import hashlib
import io
import subprocess
import sys
import tarfile

import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.cycling.core import CORE, MP_REQUIRED, MARKER, cycle, sha256, restore_land_state


def make_case(root, mp=False):
    root.mkdir(exist_ok=True, parents=True)
    source, target = root / "wrfrst", root / "wrfinput_d01"
    for path, previous in ((source, True), (target, False)):
        with Dataset(path, "w") as ds:
            for name, size in (("Time", None), ("DateStrLen", 19), ("soil_layers_stag", 4),
                               ("south_north", 2), ("west_east", 3), ("snow_layers_stag", 3),
                               ("snso_layers_stag", 7)):
                ds.createDimension(name, size)
            ds.setncatts({"DX": 3000., "DY": 3000., "MAP_PROJ": 1, "GRID_ID": 1,
                          "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21,
                          "ISWATER": 17, "ISICE": 15, "SF_SURFACE_PHYSICS": 4 if mp else 2})
            v = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
            stamps = ["2023-01-15_00:00:00", "2023-01-15_03:00:00"] if previous else ["2023-01-15_03:00:00"]
            for i, stamp in enumerate(stamps):
                v[i] = np.frombuffer(stamp.encode(), dtype="S1")
            static = {"XLAT": np.full((2, 3), 46.5), "XLONG": np.full((2, 3), 8.5),
                      "HGT": np.arange(6).reshape(2, 3) + 500,
                      "LANDMASK": [[1, 0, 1], [1, 1, 1]], "XLAND": [[1, 2, 1], [1, 1, 1]],
                      "LAKEMASK": [[0, 0, 1], [0, 0, 0]],
                      "LU_INDEX": [[5, 17, 21], [15, 5, 5]],
                      "IVGTYP": [[5, 17, 21], [15, 5, 5]], "ISLTYP": np.full((2, 3), 3),
                      "ZS": [.05, .25, .7, 1.5], "DZS": [.1, .3, .6, 1.]}
            for name, value in static.items():
                dims = ("soil_layers_stag",) if name in ("ZS", "DZS") else ("south_north", "west_east")
                var = ds.createVariable(name, "f4", ("Time",) + dims)
                for i in range(len(stamps)):
                    var[i] = value
            names = list(CORE) + ([*MP_REQUIRED, "TAUSS", "CANWAT", "FVEG"] if mp else [])
            # Deliberately omit TAUSS from real.exe-like template to test creation.
            if not previous and mp:
                names.remove("TAUSS")
            for name in names:
                layer = (("soil_layers_stag",) if name in ("SMOIS", "SH2O", "TSLB") else
                         ("snow_layers_stag",) if name in ("TSNO", "SNICE", "SNLIQ") else
                         ("snso_layers_stag",) if name == "ZSNSO" else ())
                var = ds.createVariable(name, "i4" if name == "ISNOW" else "f4", ("Time",) + layer + ("south_north", "west_east"))
                units = {"SMOIS": "m3 m-3", "SH2O": "m3 m-3", "TSLB": "K", "TSK": "K", "SNOW": "kg m-2", "SNOWH": "m"}
                var.units = units.get(name, "")
                for i in range(len(stamps)):
                    value = {"SMOIS": .4, "SH2O": .2, "TSLB": 270., "TSK": 269., "SNOW": 20., "SNOWH": .1,
                             "SNOWC": .8, "ISNOW": -3, "TSNO": 268., "SNICE": 6., "SNLIQ": .1,
                             "ZSNSO": -1., "SNEQVO": 19., "ALBOLD": .65, "TV": 270., "TG": 269.,
                             "TAH": 271., "EAH": 1200., "CANLIQ": 0., "CANICE": 0., "FWET": 0.,
                             "LAI": 2., "XSAI": .3, "TAUSS": .02, "CANWAT": 0., "FVEG": .6}[name]
                    # Final source record differs from both earlier state and template.
                    if previous and i == 1 and name in ("TSLB", "TSK", "TSNO", "TV", "TG", "TAH"):
                        value += 2
                    var[i] = value
            if mp:
                # Literal REAL4 Noah-MP column: 0.5 m / 100 kg m-2 snow,
                # depths .05/.20/.25 m, followed by the four DZS soil layers.
                # Build soil interfaces/thicknesses in the WRF operation order.
                zs = np.empty(4, np.float32)
                zs[0] = -np.float32(.1)
                for k, dz in enumerate((.3, .6, 1.), start=1):
                    zs[k] = zs[k - 1] - np.float32(dz)
                dzsoil = np.r_[-zs[0], zs[:-1] - zs[1:]].astype(np.float32)
                interfaces = np.cumsum(-np.r_[np.array([.05, .20, .25], np.float32), dzsoil], dtype=np.float32)
                for i in range(len(stamps)):
                    ds["SNOW"][i] = 100
                    ds["SNOWH"][i] = .5
                    ds["SNICE"][i] = np.broadcast_to(np.array([10, 40, 50], np.float32)[:, None, None], (3, 2, 3))
                    ds["SNLIQ"][i] = 0
                    ds["ZSNSO"][i] = np.broadcast_to(interfaces[:, None, None], (7, 2, 3))
                    ds["FVEG"][i, 1, 0] = 0
                    if previous:
                        for name in ("TV", "TAH", "EAH", "FWET", "LAI", "XSAI"):
                            ds[name][i, 1, 0] = -1e36  # WRF inactive glacier sentinel
    return source, target


@pytest.fixture
def pair(tmp_path):
    return make_case(tmp_path)


def test_independent_literal_copy_reference(pair, tmp_path):
    source, target = pair
    before = (sha256(source), sha256(target))
    report = cycle(source, target, tmp_path / "cycled")
    with Dataset(tmp_path / "cycled") as ds:
        # Literal independent reference: water and lake remain 269 K; four land
        # cells take the evolved 271 K. No calls to production mask helpers.
        np.testing.assert_array_equal(ds["TSK"][0], [[271, 269, 269], [271, 271, 271]])
        np.testing.assert_array_equal(ds["TSLB"][0], np.broadcast_to([[272, 270, 270], [272, 272, 272]], (4, 2, 3)))
        assert json.loads(ds.getncattr(MARKER))["source_frame"] == 1
        assert b"".join(ds["Times"][0]).decode() == "2023-01-15_03:00:00"
        np.testing.assert_array_equal(ds["HGT"][0], np.arange(6).reshape(2, 3) + 500)
    assert before == (sha256(source), sha256(target))
    assert report["valid_time"] == "2023-01-15_03:00:00" and report["land_cells"] == 4
    assert report["source_sha256"] == before[0]
    assert report["output_sha256"] == sha256(tmp_path / "cycled")


def test_noahmp_copies_valid_zero_and_preserves_inactive_glacier(tmp_path):
    source, target = make_case(tmp_path, mp=True)
    with Dataset(target, "r+") as ds:
        ds["CANLIQ"][:] = .75
    report = cycle(source, target, tmp_path / "cycled")
    with Dataset(tmp_path / "cycled") as ds:
        np.testing.assert_array_equal(ds["TV"][0], [[272, 270, 270], [270, 272, 272]])
        np.testing.assert_array_equal(ds["CANLIQ"][0], [[0, .75, .75], [0, 0, 0]])
        assert float(ds["TAUSS"][0, 0, 0]) == np.float32(.02)
        assert not np.any(ds["TV"][:] < 150)
        assert report["fields"]["TV"]["cells"] == 3


@pytest.mark.parametrize("name", ["XLAT", "XLONG", "HGT", "LANDMASK", "XLAND", "LU_INDEX", "IVGTYP", "ISLTYP", "ZS", "DZS", "LAKEMASK"])
def test_mismatch_refuses_without_output(pair, tmp_path, name):
    source, target = pair
    with Dataset(source, "r+") as ds:
        v = ds[name]
        key = (1,) + (0,) * (len(v.dimensions) - 1)
        v[key] += 1
    with pytest.raises(ValueError, match="differ"):
        cycle(source, target, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["DX", "DY", "MAP_PROJ", "MMINLU", "NUM_LAND_CAT", "SF_SURFACE_PHYSICS"])
def test_metadata_mismatch(pair, tmp_path, name):
    source, target = pair
    with Dataset(source, "r+") as ds:
        ds.setncattr(name, "OTHER" if name == "MMINLU" else ds.getncattr(name) + 1)
    with pytest.raises(ValueError, match="differ"):
        cycle(source, target, tmp_path / "out")


@pytest.mark.parametrize("name", CORE)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(pair, tmp_path, name, bad):
    source, target = pair
    with Dataset(source, "r+") as ds:
        ds[name][(1,) + (0,) * (len(ds[name].dimensions) - 1)] = bad
    with pytest.raises(ValueError, match="non-finite"):
        cycle(source, target, tmp_path / "out")


@pytest.mark.parametrize("name", CORE)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_template_rejected(pair, tmp_path, name, bad):
    source, target = pair
    with Dataset(target, "r+") as ds:
        ds[name][(0,) + (0,) * (len(ds[name].dimensions) - 1)] = bad
    with pytest.raises(ValueError, match="non-finite"):
        cycle(source, target, tmp_path / "out")


@pytest.mark.parametrize("name,bad", [("SMOIS", 1.1), ("SH2O", -.1), ("SNOW", -1), ("SNOWH", -1), ("SNOWC", 1.1), ("TSLB", 0), ("TSK", 400)])
def test_sanity_no_clipping(pair, tmp_path, name, bad):
    source, target = pair
    with Dataset(source, "r+") as ds:
        ds[name][(1,) + (0,) * (len(ds[name].dimensions) - 1)] = bad
    with pytest.raises(ValueError, match="sanity"):
        cycle(source, target, tmp_path / "out")


def test_liquid_exceeds_soil_refused(pair, tmp_path):
    source, target = pair
    with Dataset(source, "r+") as ds:
        ds["SH2O"][1] = .5
    with pytest.raises(ValueError, match="SH2O exceeds"):
        cycle(source, target, tmp_path / "out")


def test_missing_exact_time_and_duplicate_refused(pair, tmp_path):
    source, target = pair
    with Dataset(target, "r+") as ds:
        ds["Times"][0] = np.frombuffer(b"2023-01-15_02:00:00", dtype="S1")
    with pytest.raises(ValueError, match="exactly one frame"):
        cycle(source, target, tmp_path / "out")
    with Dataset(source, "r+") as ds, Dataset(target, "r+") as inp:
        ds["Times"][0] = ds["Times"][1]
        inp["Times"][0] = ds["Times"][1]
    with pytest.raises(ValueError, match="exactly one frame"):
        cycle(source, target, tmp_path / "out")


def test_never_overwrite_inputs_or_output(pair, tmp_path):
    source, target = pair
    for output in (source, target):
        with pytest.raises(ValueError, match="new file"):
            cycle(source, target, output)
    out = tmp_path / "out"
    out.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="new file"):
        cycle(source, target, out)
    assert out.read_bytes() == b"preserve"


def test_write_failure_atomic(pair, tmp_path, monkeypatch):
    import gpuwrf.cycling.core as core
    def fail(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(core.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        cycle(*pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".gpuwrf-cycle-*"))


@pytest.mark.parametrize("name", [n for n in MP_REQUIRED if n != "ISNOW"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_noahmp_nonfinite_rejected(tmp_path, name, bad):
    source, target = make_case(tmp_path, mp=True)
    with Dataset(source, "r+") as ds:
        ds[name][(1,) + (0,) * (len(ds[name].dimensions) - 1)] = bad
    with pytest.raises(ValueError, match="non-finite"):
        cycle(source, target, tmp_path / "out")


@pytest.mark.parametrize("name,bad", [("ISNOW", -4), ("ISNOW", 1), ("TSNO", 0), ("FWET", 1.1), ("FVEG", 1.1)])
def test_noahmp_invalid_active_state(tmp_path, name, bad):
    source, target = make_case(tmp_path, mp=True)
    with Dataset(source, "r+") as ds:
        ds[name][(1,) + (0,) * (len(ds[name].dimensions) - 1)] = bad
    with pytest.raises(ValueError, match="(ISNOW|TSNO|FWET|FVEG)"):
        cycle(source, target, tmp_path / "out")


def test_tampered_provenance_refused(tmp_path):
    source, target = make_case(tmp_path, mp=True)
    output = tmp_path / "out"
    cycle(source, target, output)
    # The first changed value must be refused before any model-state access.
    with Dataset(output, "r+") as ds:
        ds["ALBOLD"][0, 0, 0] = .3
    with Dataset(output) as ds, pytest.raises(ValueError, match="no longer matches provenance"):
        restore_land_state(ds, object())


def test_packed_fields_refused(pair, tmp_path):
    source, target = pair
    with Dataset(target, "r+") as ds:
        ds["SMOIS"].scale_factor = 1.
    with pytest.raises(ValueError, match="packed"):
        cycle(source, target, tmp_path / "out")


def test_frozen_swiss_cpu_copy_bytes(tmp_path):
    """Tracked pristine-WRF 6x6 snow/glacier/vegetated crop, all three hours."""
    fixture = Path(__file__).parent / "fixtures/cycling"
    manifest = json.loads((fixture / "swiss_cpu_land.json").read_text())
    assert sha256(fixture / "swiss_cpu_land.npz") == manifest["fixture_sha256"]
    with np.load(fixture / "swiss_cpu_land.npz", allow_pickle=False) as data:
        for hour in (1, 2, 3):
            paths = []
            for entry in (manifest["sources"][hour], manifest["sources"][0]):
                path = tmp_path / f"{entry['prefix']}_{hour}.nc"
                paths.append(path)
                with Dataset(path, "w") as ds:
                    for name, size in (("Time", None), ("DateStrLen", 19), ("soil_layers_stag", 4),
                                       ("snow_layers_stag", 3), ("snso_layers_stag", 7), ("south_north", 6), ("west_east", 6)):
                        ds.createDimension(name, size)
                    ds.setncatts(entry["attrs"])
                    var = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
                    var[0] = np.frombuffer(f"2023-01-15_0{hour}:00:00".encode(), dtype="S1")
                    for name, meta in entry["variables"].items():
                        a = data[f"{entry['prefix']}__{name}"]
                        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == meta["sha256"]
                        var = ds.createVariable(name, a.dtype, meta["dimensions"])
                        var.units = meta["units"]
                        var[0] = a
            report = cycle(*paths, tmp_path / f"out{hour}")
            with Dataset(report["output"]) as ds:
                for name in CORE + ("ISNOW", "TSNO", "SNICE", "SNLIQ", "ZSNSO", "TAUSS", "SNEQVO", "ALBOLD"):
                    actual = np.ascontiguousarray(ds[name][0]).tobytes()
                    expected = np.ascontiguousarray(data[f"source{hour}__{name}"]).tobytes()
                    assert actual == expected, (hour, name)


def test_frozen_fixture_from_clean_git_archive(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    archive = subprocess.run(["git", "archive", "HEAD", "src", "tests/test_cycle.py", "tests/fixtures/cycling"],
                             cwd=repo, capture_output=True, check=True, timeout=60).stdout
    unpacked = tmp_path / "archive"
    unpacked.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(unpacked, filter="data")
    env = {**os.environ, "PYTHONPATH": str(unpacked / "src"), "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_cycle.py", "-q", "-k", "test_frozen_swiss_cpu_copy_bytes",
                           "--basetemp", str(tmp_path / "archive-pytest")],
                          env=env, cwd=unpacked, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "1 passed" in proc.stdout, proc.stdout + proc.stderr


def test_cli_and_backend_dark(pair, tmp_path):
    source, target = pair
    repo = Path(__file__).resolve().parents[1]
    code = """import sys
sys.argv=['gpuwrf','cycle']+sys.argv[1:]
from gpuwrf.cli import main
rc=main(sys.argv[1:])
assert 'jax' not in sys.modules, 'offline cycle imported JAX'
raise SystemExit(rc)
"""
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0"}
    proc = subprocess.run([sys.executable, "-c", code, "--source", str(source), "--wrfinput", str(target), "--output", str(tmp_path / "out")],
                          env=env, cwd=repo, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["land_cells"] == 4


def test_real_swiss_consecutive_cpu_frames(tmp_path):
    """Full real WRF t1->t2->t3 state; next-init templates retimed explicitly.

    Does not claim independently regenerated atmospheric initial conditions.
    """
    repo = Path(__file__).resolve().parents[1]
    reference = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h")
    if not (reference / "wrfout_d01_2023-01-15_03:00:00").exists():
        pytest.skip("full original CPU-WRF Swiss savepoints not mounted; tracked crop tests remain mandatory")
    for hour in (1, 2, 3):
        stamp = f"2023-01-15_0{hour}:00:00"
        template = tmp_path / f"new{hour}"
        shutil.copyfile(repo / "examples/switzerland_d01/wrfinput_d01", template)
        with Dataset(template, "r+") as ds:
            ds["Times"][0] = np.frombuffer(stamp.encode(), dtype="S1")
            ds.START_DATE = stamp
        source = reference / f"wrfout_d01_{stamp}"
        report = cycle(source, template, tmp_path / f"cycle{hour}" / "wrfinput_d01")
        shutil.copyfile(repo / "examples/switzerland_d01/namelist.input", tmp_path / f"cycle{hour}" / "namelist.input")
        with Dataset(source) as src, Dataset(report["output"]) as dst, Dataset(template) as cold:
            for name in CORE:
                np.testing.assert_array_equal(dst[name][0], src[name][0], err_msg=name)
            for name in ("U", "V", "T", "QVAPOR", "PSFC", "SST", "Times", "LU_INDEX"):
                np.testing.assert_array_equal(dst[name][:], cold[name][:], err_msg=name)
        # The actual model input loader must retain the transferred state.
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
        land, _, meta = build_noahmp_land_state(tmp_path / f"cycle{hour}", "d01")
        with Dataset(report["output"]) as ds:
            for name, attr in (("SH2O", "sh2o"), ("TSLB", "tslb"), ("SMOIS", "smois"), ("TSNO", "tsno"),
                               ("SNICE", "snice"), ("SNLIQ", "snliq"), ("ISNOW", "isnow"), ("ZSNSO", "zsnso"),
                               ("SNOW", "sneqv"), ("SNOWH", "snowh"), ("TAUSS", "tauss")):
                actual = np.asarray(getattr(land, attr), dtype=ds[name].dtype)
                expected = np.asarray(ds[name][0])
                assert actual.tobytes(order="C") == expected.tobytes(order="C"), name
        assert meta["land_cycle"]["valid_time"] == stamp
        assert meta["source"] == "soil/snow cycle from previous forecast"


def test_real_swiss_native_real_loader(tmp_path, monkeypatch):
    import jax
    jax.clear_caches()
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    try:
        test_real_swiss_consecutive_cpu_frames(tmp_path)
    finally:
        jax.clear_caches()


COLUMN_CORRUPTIONS = ("zero_soil_thickness", "zero_snow_thickness", "zero_layer_mass", "inactive_layers",
                      "wrong_soil_thickness", "wrong_snow_mass", "wrong_snow_depth")


def corrupt_column(ds, index, kind):
    """Critic's four original-source controls, plus nonzero consistency errors."""
    if kind == "zero_soil_thickness":
        ds["ZSNSO"][index, 4, 0, 0] = ds["ZSNSO"][index, 3, 0, 0]
        return ("ZSNSO",)
    if kind == "zero_snow_thickness":
        ds["ZSNSO"][index, 2, 0, 0] = ds["ZSNSO"][index, 1, 0, 0]
        return ("ZSNSO",)
    if kind == "zero_layer_mass":
        ds["SNICE"][index, :, 0, 0] = 0
        ds["SNLIQ"][index, :, 0, 0] = 0
        return ("SNICE", "SNLIQ")
    if kind == "inactive_layers":
        ds["ISNOW"][index, 0, 0] = 0
        return ("ISNOW",)
    if kind == "wrong_soil_thickness":
        ds["ZSNSO"][index, 4, 0, 0] -= .05
        return ("ZSNSO",)
    if kind == "wrong_snow_mass":
        ds["SNOW"][index, 0, 0] += 1
        return ("SNOW",)
    assert kind == "wrong_snow_depth"
    ds["SNOWH"][index, 0, 0] += .01
    return ("SNOWH",)


def column_pair(tmp_path, original_cpu):
    if not original_cpu:
        source, target = make_case(tmp_path, mp=True)
        return source, target, 1
    repo = Path(__file__).resolve().parents[1]
    reference = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h/wrfout_d01_2023-01-15_01:00:00")
    if not reference.exists():
        pytest.skip("original CPU-WRF Swiss frame not mounted; synthetic column checks remain mandatory")
    source, target = tmp_path / "source", tmp_path / "wrfinput_d01"
    shutil.copyfile(reference, source)
    shutil.copyfile(repo / "examples/switzerland_d01/wrfinput_d01", target)
    with Dataset(target, "r+") as ds:
        ds["Times"][0] = np.frombuffer(b"2023-01-15_01:00:00", dtype="S1")
    return source, target, 0


@pytest.mark.parametrize("original_cpu", [False, True], ids=["literal", "cpu-wrf"])
@pytest.mark.parametrize("kind", COLUMN_CORRUPTIONS)
def test_inconsistent_column_refused_before_publication(tmp_path, original_cpu, kind):
    source, target, index = column_pair(tmp_path, original_cpu)
    with Dataset(source, "r+") as ds:
        corrupt_column(ds, index, kind)
    before = (sha256(source), sha256(target))
    with pytest.raises(ValueError, match="soil/snow column"):
        cycle(source, target, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".gpuwrf-cycle-*"))
    assert before == (sha256(source), sha256(target))


@pytest.mark.parametrize("original_cpu", [False, True], ids=["literal", "cpu-wrf"])
@pytest.mark.parametrize("kind", COLUMN_CORRUPTIONS)
def test_internally_hashed_bad_column_refused_before_restore(tmp_path, original_cpu, kind):
    source, target, _ = column_pair(tmp_path, original_cpu)
    output = tmp_path / "out"
    cycle(source, target, output)
    with Dataset(output, "r+") as ds:
        changed = corrupt_column(ds, 0, kind)
        # Deliberately reseal all modified values. The hash gate alone cannot
        # catch these self-consistent records: physical admission must do so.
        report = json.loads(ds.getncattr(MARKER))
        land = (np.asarray(ds["LANDMASK"][0]) == 1) & (np.asarray(ds["LAKEMASK"][0]) == 0)
        for name in changed:
            values = np.asarray(ds[name][0])
            mask = np.broadcast_to(land, values.shape)
            report["fields"][name]["sha256"] = hashlib.sha256(np.ascontiguousarray(values[mask]).tobytes()).hexdigest()
        ds.setncattr(MARKER, json.dumps(report))
    class NoRestoreFields:
        def __getattr__(self, name):
            raise AssertionError("invalid column reached model fields before admission")

    # Require physical refusal before touching any device/fallback buffers.
    with Dataset(output) as ds, pytest.raises(ValueError, match="soil/snow column"):
        restore_land_state(ds, NoRestoreFields())


@pytest.mark.parametrize("layers,height,swe", [(0, 0., 0.), (0, .01, 2.), (1, .03, 7.3), (2, .15, 31.7), (3, .5, 102.3)])
def test_valid_thin_empty_and_layered_columns_preserve_bytes(tmp_path, layers, height, swe):
    source, target = make_case(tmp_path, mp=True)
    with Dataset(source, "r+") as ds:
        # Literal WRF SNOW_INIT branches (not production validation helpers).
        f = np.float32
        snow_dz = np.zeros(3, f)
        if layers == 1:
            snow_dz[2] = f(height)
        elif layers == 2:
            snow_dz[1], snow_dz[2] = f(.05), f(height) - f(.05)
        elif layers == 3:
            snow_dz[:] = [f(.05), f(.20), (f(height) - f(.20)) - f(.05)]
        zsoil = -np.cumsum(np.array([.1, .3, .6, 1.], f), dtype=f)
        soil_dz = np.r_[-zsoil[0], zsoil[:-1] - zsoil[1:]].astype(f)
        depths = np.cumsum(-np.r_[snow_dz, soil_dz], dtype=f)
        ds["ISNOW"][1, 0, 0] = -layers
        ds["SNOW"][1, 0, 0] = f(swe)
        ds["SNOWH"][1, 0, 0] = f(height)
        ds["SNOWC"][1, 0, 0] = 1 if height else 0
        ds["ZSNSO"][1, :, 0, 0] = depths
        ds["SNICE"][1, :, 0, 0] = snow_dz * (f(swe) / f(height)) if layers else 0
        ds["SNLIQ"][1, :, 0, 0] = 0
        ds["TSNO"][1, :, 0, 0] = np.where(np.arange(3) >= 3 - layers, f(270), f(0))
    report = cycle(source, target, tmp_path / "out")
    with Dataset(source) as src, Dataset(report["output"]) as dst:
        land = (np.asarray(src["LANDMASK"][1]) == 1) & (np.asarray(src["LAKEMASK"][1]) == 0)
        for name in ("ISNOW", "SNOW", "SNOWH", "SNICE", "SNLIQ", "TSNO", "ZSNSO", "SH2O", "SMOIS"):
            assert np.asarray(src[name][1])[..., land].tobytes() == np.asarray(dst[name][0])[..., land].tobytes(), name
    # Independently populate the old fallback state only to test restore bytes.
    from types import SimpleNamespace
    from gpuwrf.cycling.core import LAND_LEAVES
    with Dataset(report["output"]) as dst:
        fallback = SimpleNamespace(**{attr: np.zeros(dst[name].shape[1:], dtype=dst[name].dtype)
                                    for name, attr in LAND_LEAVES.items() if name in dst.variables})
        restore_land_state(dst, fallback)
        for name in ("ISNOW", "SNOW", "SNOWH", "SNICE", "SNLIQ", "TSNO", "ZSNSO", "SH2O", "SMOIS"):
            actual = np.asarray(getattr(fallback, LAND_LEAVES[name]), dtype=dst[name].dtype)
            # Only this literal land cell is needed; water/lake fallbacks are
            # intentionally different and remain outside the transfer mask.
            assert actual[..., 0, 0].tobytes() == np.asarray(dst[name][0])[..., 0, 0].tobytes(), name


@pytest.mark.parametrize("name,delta,admitted", [("SNOW", "ulp", True), ("SNOW", .01, False),
                                                ("SNOWH", "ulp", True), ("SNOWH", .001, False)])
def test_derived_rounding_allows_real_ulp_but_refuses_resolved_errors(tmp_path, name, delta, admitted):
    source, target = make_case(tmp_path, mp=True)
    with Dataset(source, "r+") as ds:
        value = np.float32(ds[name][1, 0, 0])
        ds[name][1, 0, 0] = (np.nextafter(value, np.float32(np.inf)) if delta == "ulp" else value + delta)
    if admitted:
        report = cycle(source, target, tmp_path / "out")
        with Dataset(source) as src, Dataset(report["output"]) as dst:
            assert np.asarray(src[name][1, 0, 0]).tobytes() == np.asarray(dst[name][0, 0, 0]).tobytes()
    else:
        with pytest.raises(ValueError, match="soil/snow column"):
            cycle(source, target, tmp_path / "out")
