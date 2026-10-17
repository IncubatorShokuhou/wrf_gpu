"""Independent affine wind fields and literal WRF coupling/rate identities."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.perturb.inputs import factors, prepare

REPO = Path(__file__).resolve().parents[1]


def case_files(path, *, boundary=True, nests=2):
    path.mkdir()
    (path / "namelist.input").write_text(f"&domains\n max_dom={nests},\n/\n")
    for number in range(1, nests + 1):
        with Dataset(path / f"wrfinput_d{number:02d}", "w", format="NETCDF3_64BIT_OFFSET") as ds:
            for name, size in (("Time", 1), ("bottom_top", 2), ("south_north", 3),
                               ("south_north_stag", 4), ("west_east", 4), ("west_east_stag", 5)):
                ds.createDimension(name, size)
            for name, dims in (("U", ("Time", "bottom_top", "south_north", "west_east_stag")),
                               ("V", ("Time", "bottom_top", "south_north_stag", "west_east"))):
                field = ds.createVariable(name, "f4", dims)
                field.units = "m s-1"
                shape = tuple(len(ds.dimensions[d]) for d in dims)
                field[:] = np.arange(np.prod(shape), dtype="f4").reshape(shape)*.125 + number
            ds.createVariable("MUB", "f4", ("Time", "south_north", "west_east"))[:] = 75000
            ds.description = "unchanged metadata"
    if boundary:
        with Dataset(path / "wrfbdy_d01", "w") as ds:
            for name, size in (("Time", 3), ("bottom_top", 2), ("bdy_width", 2),
                               ("south_north", 3), ("south_north_stag", 4),
                               ("west_east", 4), ("west_east_stag", 5)):
                ds.createDimension(name, size)
            for wind in "UV":
                for side in ("XS", "XE", "YS", "YE"):
                    along = ("south_north" + ("_stag" if wind == "V" else "") if side[0] == "X"
                             else "west_east" + ("_stag" if wind == "U" else ""))
                    dims = ("Time", "bdy_width", "bottom_top", along)
                    shape = tuple(len(ds.dimensions[d]) for d in dims)
                    initial = np.arange(np.prod(shape[1:]), dtype="f4").reshape(shape[1:]) + 120000
                    # Exact affine records and derivatives (1-hour spacing).
                    records = initial + np.arange(3, dtype="f4")[:, None, None, None] * 3600 * 2
                    ds.createVariable(f"{wind}_B{side}", "f4", dims)[:] = records
                    ds.createVariable(f"{wind}_BT{side}", "f4", dims)[:] = 2
            ds.createVariable("MU_BXS", "f4", ("Time", "bdy_width", "south_north"))[:] = 20
    (path / "wrflowinp_d01").write_bytes(b"unchanged lower boundary file")
    return path


def test_sample_multiplier_definition_and_reproducible_stream():
    actual = factors(11, 451, .05)
    # Independent public PCG64 normal draws + mathematical definition.
    noise = np.random.Generator(np.random.PCG64(451)).normal(size=11)
    center = sum(noise) / 11
    norm = (sum((v - center)**2 for v in noise) / 10)**.5
    expected = np.array([1 + .05 * (v - center) / norm for v in noise])
    np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=2e-16)
    assert np.mean(actual) == pytest.approx(1, abs=3e-16)
    assert np.std(actual, ddof=1) == pytest.approx(.05)
    np.testing.assert_array_equal(actual, factors(11, 451, .05))
    assert not np.array_equal(actual, factors(11, 452, .05))


def test_case_copies_coherent_wind_and_preserves_mass_metadata_sources(tmp_path):
    case = case_files(tmp_path / "source")
    original = {p.name: p.read_bytes() for p in case.iterdir()}
    result = prepare(case, tmp_path / "ensemble", members=3, seed=52, wind_std=.05)
    assert len(result["members"]) == 3 and result["ic_only"] is False
    assert json.loads((tmp_path / "ensemble/perturb_manifest.json").read_text()) == result
    for member in result["members"]:
        dest = Path(member["case"])
        scale = np.float32(member["factor"])
        assert scale == member["factor_real32"]
        for name in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"):
            assert not (dest / name).is_symlink()
            with Dataset(case / name) as src, Dataset(dest / name) as got:
                assert src.file_format == got.file_format
                assert src.ncattrs() == got.ncattrs()
                for variable in src.variables:
                    selected = variable in {"U", "V"} or variable.startswith(("U_B", "V_B"))
                    expected = src[variable][:] * scale if selected else src[variable][:]
                    np.testing.assert_array_equal(got[variable][:], expected)
                if name == "wrfbdy_d01":
                    for wind in "UV":
                        for side in ("XS", "XE", "YS", "YE"):
                            # Rounding of independent REAL multiplication is allowed;
                            # scale commutes with prescribed mass coupling and time derivative.
                            rate = np.diff(got[f"{wind}_B{side}"][:], axis=0) / 3600
                            np.testing.assert_allclose(rate, got[f"{wind}_BT{side}"][:-1], atol=5e-6, rtol=1e-6)
        assert (dest / "namelist.input").is_symlink()
        assert (dest / "wrflowinp_d01").read_bytes() == original["wrflowinp_d01"]
    assert {p.name: p.read_bytes() for p in case.iterdir()} == original


def test_ic_only_is_explicit_and_retains_boundary_byte_identity(tmp_path):
    case = case_files(tmp_path / "source")
    result = prepare(case, tmp_path / "members", members=2, seed=4, wind_std=.02, ic_only=True)
    for member in result["members"]:
        path = Path(member["case"]) / "wrfbdy_d01"
        assert path.is_symlink() and path.read_bytes() == (case / "wrfbdy_d01").read_bytes()
    no_bdy = case_files(tmp_path / "no-bdy", boundary=False)
    with pytest.raises(ValueError, match="require wrfbdy"):
        prepare(no_bdy, tmp_path / "refused", members=2, seed=1, wind_std=.1)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("kwargs", [
    {"members": 1}, {"members": True}, {"members": 2.5}, {"seed": -1}, {"seed": 2**64},
    {"seed": True}, {"wind_std": 0}, {"wind_std": -1}, {"wind_std": np.nan},
    {"wind_std": np.inf}, {"wind_std": 1}, {"wind_std": True},
    {"wind_std": 1e-20},
])
def test_invalid_recipe_rejected(kwargs):
    args = {"members": 3, "seed": 4, "wind_std": .1, **kwargs}
    with pytest.raises(ValueError):
        factors(**args)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -9999.])
def test_corrupt_selected_inputs_refused_before_output(tmp_path, bad):
    case = case_files(tmp_path / "source")
    with Dataset(case / "wrfinput_d02", "r+") as ds:
        if bad == -9999:
            ds["U"].missing_value = np.float32(bad)
        ds["U"][0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="NaN/Inf|missing"):
        prepare(case, tmp_path / "out", members=3, seed=1, wind_std=.1)
    assert not (tmp_path / "out").exists()


def test_overwrite_and_partial_boundary_refused(tmp_path):
    case = case_files(tmp_path / "source")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_bytes(b"existing")
    with pytest.raises(ValueError, match="refusing overwrite"):
        prepare(case, out, members=2, seed=1, wind_std=.1)
    assert (out / "keep").read_bytes() == b"existing"
    with Dataset(case / "wrfbdy_d01", "r+") as ds:
        ds.renameVariable("V_BTXE", "REMOVED")
    with pytest.raises(ValueError, match="missing V_BTXE"):
        prepare(case, tmp_path / "out2", members=2, seed=1, wind_std=.1)


def test_real32_overflow_removes_only_new_output_and_preserves_source(tmp_path):
    case = case_files(tmp_path / "source")
    with Dataset(case / "wrfinput_d01", "r+") as ds:
        ds["U"][:] = np.finfo("float32").max
    original = (case / "wrfinput_d01").read_bytes()
    with pytest.raises(ValueError, match="produced NaN/Inf"):
        prepare(case, tmp_path / "failed", members=2, seed=451, wind_std=.05)
    assert not (tmp_path / "failed").exists()
    assert (case / "wrfinput_d01").read_bytes() == original


@pytest.mark.parametrize("file,dimension", [("wrfbdy_d01", "bdy_width"),
                                           ("wrfinput_d01", "west_east_stag")])
def test_misdeclared_layout_refused(tmp_path, file, dimension):
    case = case_files(tmp_path / "source")
    with Dataset(case / file, "r+") as ds:
        ds.renameDimension(dimension, "wrong_dimension")
    with pytest.raises(ValueError, match="layout"):
        prepare(case, tmp_path / "failed", members=2, seed=1, wind_std=.1)


def test_cli_offline_no_jax_import_and_native_command(tmp_path):
    case = case_files(tmp_path / "source")
    script = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name == "jax" or name.startswith("jax."):
            raise AssertionError("offline perturb imported JAX")
sys.meta_path.insert(0,Block())
sys.argv=["gpuwrf","perturb"]+sys.argv[1:]
from gpuwrf.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    command = [sys.executable, "-c", script, "--case", str(case), "--output", str(tmp_path / "members"),
               "--members", "3", "--seed", "7", "--wind-std", "0.05"]
    result = subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src"),
                            "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0"},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == "gpuwrf.perturb.wind.v1"
