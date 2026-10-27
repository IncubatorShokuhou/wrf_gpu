"""File-seam controls; coupled physics qualification uses original WRF separately."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.cycling.core import cycle, sha256, _times
from gpuwrf.cycling.atmosphere import MARKER, BASE_FIELDS, MASS_DIMS
from gpuwrf.io.netcdf_lock import Dataset
from test_cycle import make_case


FIELDS = ("U", "V", "W", "T", "THM", "P", "PH", "MU", "QVAPOR", "QCLOUD", "QRAIN",
          "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "QKE", "qke_adv", "TSQ", "QSQ", "COV")


def fixture_case(root, domain=1):
    source, template = make_case(root)
    for path, old in ((source, True), (template, False)):
        with Dataset(path, "r+") as ds:
            nt = len(ds.dimensions["Time"])
            for dim, size in (("bottom_top", 3), ("bottom_top_stag", 4),
                              ("west_east_stag", 4), ("south_north_stag", 3)):
                ds.createDimension(dim, size)
            ds.setncatts({"GRID_ID": domain, "PARENT_ID": 1, "I_PARENT_START": 1,
                          "DX": 3000. if domain == 1 else 1000., "DY": 3000. if domain == 1 else 1000.,
                          "J_PARENT_START": 1, "PARENT_GRID_RATIO": 1 if domain == 1 else 3,
                          "MP_PHYSICS": 8, "BL_PBL_PHYSICS": 5, "HYBRID_OPT": 2,
                          "HYPSOMETRIC_OPT": 2, "USE_THETA_M": 1})
            for name, values in {"ZNU": [.85, .5, .15], "ZNW": [1, .7, .3, 0],
                                  "DNW": [-.3, -.4, -.3], "RDNW": [-1/.3, -1/.4, -1/.3],
                                  "C1H": [1.1, .8, .2], "C2H": [200, 200, 200],
                                  "C3H": [.85, .5, .15], "C4H": [0, 0, 0],
                                  "C1F": [1.2, 1, .5, 0], "C2F": [200]*4,
                                  "C3F": [1, .7, .3, 0], "C4F": [0]*4}.items():
                dim = "bottom_top_stag" if len(values) == 4 else "bottom_top"
                var = ds.createVariable(name, "f4", ("Time", dim))
                var[:nt] = np.broadcast_to(np.array(values), (nt, len(values)))
            ds.createVariable("P_TOP", "f4", ("Time",))[:] = 10000
            for name in ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V"):
                dims = (("south_north", "west_east_stag") if name.endswith("U") else
                        ("south_north_stag", "west_east") if name.endswith("V") else MASS_DIMS[1:])
                ds.createVariable(name, "f4", ("Time",) + dims)[:] = 1.05
            dims_for = {"U": ("bottom_top", "south_north", "west_east_stag"),
                        "V": ("bottom_top", "south_north_stag", "west_east"),
                        "W": ("bottom_top_stag", "south_north", "west_east"),
                        "PH": ("bottom_top_stag", "south_north", "west_east"),
                        "PHB": ("bottom_top_stag", "south_north", "west_east"),
                        "MU": MASS_DIMS[1:], "MUB": MASS_DIMS[1:]}
            for name in (*FIELDS, "PB", "PHB", "MUB", "AL", "ALB", "P_HYD"):
                dims = dims_for.get(name, MASS_DIMS)
                var = ds.createVariable(name, "f4", ("Time",) + dims)
                var.units = ("Pa" if name in ("P", "PB", "MU", "MUB", "P_HYD") else
                             "K" if name in ("T", "THM") else "")
                var[:] = 0
            # Independent column construction, using EOS and layer-integrated
            # hydrostatic pressure. It exercises copying, not WRF parity.
            for i in range(len(ds.dimensions["Time"])):
                mu = np.arange(6).reshape(2, 3) * 100 + 90000 + (100 if old else 0)
                pf = np.array([1, .7, .3, 0])[:, None, None] * mu + 10000
                pm = np.array([.85, .5, .15])[:, None, None] * mu + 10000
                theta = np.full((3, 2, 3), 290. + old)
                qv = np.broadcast_to(np.array([.001, 1e-5, 1e-10])[:, None, None], theta.shape)
                alpha = 287/1e5 * theta * (1+461.6/287*qv) * (pm/1e5)**(-2.5/3.5)
                thick = alpha * pm * np.log(pf[:-1]/pf[1:])
                surface = 9.81 * np.asarray(ds["HGT"][i], np.float64)
                phi = np.concatenate((surface[None], surface[None] + np.cumsum(thick, axis=0)))
                ds["MUB"][i] = 90000
                ds["MU"][i] = mu - 90000
                ds["PB"][i] = 50000
                ds["P"][i] = pm - 50000
                ds["PHB"][i] = np.arange(4)[:, None, None] * 1000 + surface
                ds["PH"][i] = phi - ds["PHB"][i]
                ds["T"][i] = theta - 300
                ds["THM"][i] = theta * (1+461.6/287*qv) - 300
                ds["QVAPOR"][i] = qv
                ds["QCLOUD"][i] = .001 * old
                ds["QNICE"][i] = 1000
                ds["QNRAIN"][i] = 2000
                ds["QKE"][i] = 2
                ds["qke_adv"][i] = 2
                ds["TSQ"][i] = .1
                ds["QSQ"][i] = 1e-8
                ds["COV"][i] = -1e-5
                ds["ALB"][i] = 1
                ds["AL"][i] = alpha - 1
                ds["P_HYD"][i] = pm
                for name in ("U", "V", "W"):
                    ds[name][i] = (np.arange(ds[name][i].size).reshape(ds[name][i].shape) + 1) * .1
    bdy = root / "wrfbdy_d01"
    make_boundary(template, bdy)
    return source, template, bdy


def literal_coupled(ds, name):
    """Deliberately scalar reference for WRFDA's coupling, not the implementation."""
    chosen = "THM" if name == "T" and ds.USE_THETA_M else name
    raw = np.asarray(ds[chosen][0])
    if name == "MU":
        return raw.copy()
    result = np.empty_like(raw)
    mass = np.asarray(ds["MU"][0]) + np.asarray(ds["MUB"][0])
    for k, j, i in np.ndindex(raw.shape):
        mu = mass[j, i] if name not in ("U", "V") else (
            np.float32(.5)*(mass[j, min(i, 2)]+mass[j, max(i-1, 0)]) if name == "U" else
            np.float32(.5)*(mass[min(j, 1), i]+mass[max(j-1, 0), i]))
        level = "F" if name in ("PH", "W") else "H"
        a = np.float32(np.float32(ds[f"C1{level}"][0,k]*mu) + ds[f"C2{level}"][0,k])
        result[k,j,i] = np.float32(raw[k,j,i]*a)
        if name in ("U", "V", "W"):
            result[k,j,i] /= ds["MAPFAC_"+(name if name != "W" else "M")][0,j,i]
    return result


def literal_edge(raw, side):
    if side == "XS": return np.stack((raw[..., 0], raw[..., 1]))
    if side == "XE": return np.stack((raw[..., -1], raw[..., -2]))
    if side == "YS": return np.stack((raw[..., 0, :], raw[..., 1, :]))
    return np.stack((raw[..., -1, :], raw[..., -2, :]))


def make_boundary(initial, path, records=2):
    with Dataset(initial) as src, Dataset(path, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, records if name == "Time" else len(dim))
        dst.createDimension("bdy_width", 2)
        times = ["2023-01-15_03:00:00", "2023-01-15_09:00:00", "2023-01-15_15:00:00"]
        for name in ("Times", "thisbdytime", "nextbdytime"):
            var = dst.createVariable(name, "S1", ("Time", "DateStrLen"))
            for i in range(records):
                var[i] = np.frombuffer(times[i+(name == "nextbdytime")].encode(), dtype="S1")
        for name in ("U", "V", "W", "T", "PH", "MU", "QVAPOR", "QCLOUD", "QNICE", "qke_adv"):
            raw = literal_coupled(src, name)
            for side in ("XS", "XE", "YS", "YE"):
                cross = ("south_north_stag" if name == "V" else "south_north") if side[0] == "X" else (
                    "west_east_stag" if name == "U" else "west_east")
                dims = ("Time", "bdy_width") + (("bottom_top_stag" if name in ("PH", "W") else "bottom_top",)
                                                if raw.ndim == 3 else ()) + (cross,)
                for tendency in (False, True):
                    var = dst.createVariable(f'{name}_B{"T" if tendency else ""}{side}', "f4", dims)
                    for i in range(records):
                        value = literal_edge(raw, side)
                        var[i] = np.full_like(value, .01) if tendency else value + i * np.float32(216)


def run_cycle(files, output, **kw):
    source, template, boundary = files
    return cycle(source, template, output, atmosphere=True, wrfbdy=boundary,
                 boundary_output=output.with_name(output.name + "_bdy"), **kw)


def test_transfer_and_boundary_endpoint(tmp_path):
    files = fixture_case(tmp_path)
    before = [sha256(p) for p in files]
    output = tmp_path / "cycled"
    report = run_cycle(files, output)
    assert [sha256(p) for p in files] == before
    with Dataset(files[0]) as src, Dataset(files[1]) as fresh, Dataset(output) as result:
        stored = json.loads(result.getncattr(MARKER))
        assert set(stored["fields"]) == set(FIELDS)
        for name in FIELDS:
            np.testing.assert_array_equal(result[name][0], src[name][1])
        for name in BASE_FIELDS:
            np.testing.assert_array_equal(result[name][:], fresh[name][:])
        assert stored["fields"]["QCLOUD"]["sha256"] == report["domains"][0]["fields"]["QCLOUD"]["sha256"]
        with Dataset(files[2]) as old, Dataset(tmp_path / "cycled_bdy") as bdy:
            for name, record in report["boundary"]["fields"].items():
                field, side = name.rsplit("_B", 1)
                expected = literal_edge(literal_coupled(result, field), side)
                np.testing.assert_array_equal(bdy[name][0], expected)
                tendency = name.replace("_B", "_BT")
                np.testing.assert_array_equal(bdy[tendency][0], (old[name][1] - expected) / np.float32(21600))
                np.testing.assert_array_equal(bdy[name][1:], old[name][1:])
                np.testing.assert_array_equal(bdy[tendency][1:], old[tendency][1:])
                assert record["endpoint_source"] == "second value"


def test_identity_all_variable_storage(tmp_path):
    _, template, bdy = fixture_case(tmp_path)
    output = tmp_path / "roundtrip"
    run_cycle((template, template, bdy), output)
    with Dataset(template) as src, Dataset(output) as out:
        assert set(src.variables) == set(out.variables)
        for name in src.variables:
            assert np.asarray(src[name][:]).tobytes() == np.asarray(out[name][:]).tobytes(), name


@pytest.mark.parametrize("name", ["T", "MU", "PH", "P", "QVAPOR", "QCLOUD", "QNICE", "QKE", "U", "PB", "HGT"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_refuses(tmp_path, name, bad):
    files = fixture_case(tmp_path)
    with Dataset(files[0], "r+") as ds:
        ds[name][1] = bad
    with pytest.raises(ValueError, match="non-finite|nonfinite|finite"):
        run_cycle(files, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".gpuwrf-atmosphere-*"))


@pytest.mark.parametrize("name", ["QVAPOR", "QRAIN", "QNICE", "QNRAIN", "QKE", "TSQ"])
def test_missing_state_refuses(tmp_path, name):
    files = fixture_case(tmp_path)
    with Dataset(files[0], "r+") as ds:
        ds.renameVariable(name, "REMOVED_" + name)
    with pytest.raises(ValueError, match="missing"):
        run_cycle(files, tmp_path / "bad")


@pytest.mark.parametrize("kind", ["time", "grid", "nest", "coordinate", "mu", "ph", "theta"])
def test_incompatible_state_refuses(tmp_path, kind):
    files = fixture_case(tmp_path)
    with Dataset(files[0], "r+") as ds:
        if kind == "time": ds["Times"][1] = np.frombuffer(b"2023-01-15_04:00:00", dtype="S1")
        elif kind == "grid": ds["XLAT"][1] += .1
        elif kind == "nest": ds.I_PARENT_START = 2
        elif kind == "coordinate": ds["C3H"][1] += .1
        elif kind == "mu": ds["MU"][1] = -1e6
        elif kind == "ph": ds["PH"][1] *= -1
        elif kind == "theta": ds["THM"][1] += 10
    with pytest.raises(ValueError):
        run_cycle(files, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_material_terrain_and_base_changes_refuse(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[1], "r+") as ds:
        ds["PB"][:] += 500
        ds["MUB"][:] += 100
        ds["HGT"][:] += 15
        ds["PHB"][:] += np.float32(9.81*15)
    out = tmp_path / "rebased"
    with pytest.raises(ValueError, match="material"):
        run_cycle(files, out)
    assert not out.exists()


def test_roundoff_rebase_preserves_totals(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[1], "r+") as ds:
        for name in ("PB", "MUB", "PHB"):
            ds[name][:] = np.nextafter(np.asarray(ds[name][:]), np.float32(np.inf))
    out = tmp_path / "rebased"
    run_cycle(files, out)
    with Dataset(files[0]) as src, Dataset(files[1]) as fresh, Dataset(out) as dst:
        for prime, base in (("P", "PB"), ("MU", "MUB")):
            np.testing.assert_allclose(dst[prime][0]+dst[base][0], src[prime][1]+src[base][1], atol=.01, rtol=0)
            np.testing.assert_array_equal(dst[base][:], fresh[base][:])
        source_phi = np.asarray(src["PH"][1], np.float64)+src["PHB"][1]
        target_phi = np.asarray(dst["PH"][0], np.float64)+dst["PHB"][0]
        np.testing.assert_allclose(np.diff(target_phi,axis=0), np.diff(source_phi,axis=0), atol=.02, rtol=0)
        np.testing.assert_array_equal(dst["HGT"][:], fresh["HGT"][:])


def test_restart_current_slots(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[0], "r+") as ds:
        for name in ("U", "V", "W", "PH", "MU", "THM"):
            ds.renameVariable(name, name + "_2")
    run_cycle(files, tmp_path / "out")


def test_unrecorded_qke_adv_comes_from_qke(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[0], "r+") as ds:
        ds.renameVariable("qke_adv", "REMOVED")
    result = run_cycle(files, tmp_path / "out")
    assert result["domains"][0]["fields"]["qke_adv"]["source_variable"] == "QKE"
    with Dataset(files[0]) as src, Dataset(tmp_path / "out") as dst:
        np.testing.assert_array_equal(src["QKE"][1], dst["qke_adv"][0])


def test_encoded_boundary_clocks(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[2], "r+") as ds:
        for name in ("thisbdytime", "nextbdytime"):
            ds.renameVariable(name, "md___"+name+"e_x_t_d_o_m_a_i_n_m_e_t_a_data_")
    run_cycle(files, tmp_path / "out")


def test_subnormal_boundary_tendency_uses_ieee_absolute_error_term(tmp_path):
    from gpuwrf.cycling.boundary import update_first_interval
    files=fixture_case(tmp_path)
    with Dataset(files[1],"r+") as ds:
        ds["QCLOUD"][0]=np.float32(1.6e-43)
    with Dataset(files[2],"r+") as ds:
        for side in ("XS","XE","YS","YE"):
            ds[f"QCLOUD_B{side}"][:]=0
            ds[f"QCLOUD_BT{side}"][:]=0
    with Dataset(files[1]) as initial,Dataset(files[2],"r+") as boundary:
        report=update_first_interval(initial,boundary,allowed_fields=set(FIELDS))
        records=[report["fields"][f"QCLOUD_B{side}"] for side in ("XS","XE","YS","YE")]
        assert all(r["endpoint_underflow_bound"]>0 for r in records)
        assert any(r["endpoint_max_abs_error"]>0 for r in records)
        assert all(r["endpoint_max_abs_error"]<=r["endpoint_underflow_bound"] for r in records)


def test_single_record_extrapolated_endpoint(tmp_path):
    files = fixture_case(tmp_path)
    files[2].unlink()
    make_boundary(files[1], files[2], records=1)
    report = run_cycle(files, tmp_path / "out")
    assert all(v["endpoint_source"] == "original extrapolation" for v in report["boundary"]["fields"].values())


def test_missing_boundary_pair_and_bad_clock_refuse(tmp_path):
    files = fixture_case(tmp_path)
    with Dataset(files[2], "r+") as ds:
        ds.renameVariable("PH_BTXE", "REMOVED")
    with pytest.raises(ValueError, match="missing"):
        run_cycle(files, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_nested_preflight_and_publication_rollback(tmp_path, monkeypatch):
    from test_native_atmosphere_cycle import native_fixture
    first = native_fixture(tmp_path / "d01")
    second = native_fixture(tmp_path / "d02", domain=2)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    original = os.link
    def fail_second(a, b):
        if Path(b) == out2:
            raise OSError("injected publication failure")
        return original(a, b)
    monkeypatch.setattr(os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        run_cycle(first, out1, nests=[(*second[:2], out2)], native_atmosphere=True)
    assert not out1.exists() and not out2.exists()
    assert not (tmp_path / "out1_bdy").exists()


def _loaded_source_path():
    # The child must import the SAME gpuwrf tree as this test (E152): pytest's
    # pythonpath setting is not inherited by subprocesses.
    import gpuwrf
    src = str(Path(gpuwrf.__file__).resolve().parents[1])
    return os.pathsep.join([src, *filter(None, [os.environ.get("PYTHONPATH")])])


def test_cli_backend_dark(tmp_path):
    files = fixture_case(tmp_path)
    code = """import sys
sys.argv=['gpuwrf','cycle']+sys.argv[1:]
from gpuwrf.cli import main
rc=main()
assert 'jax' not in sys.modules, 'offline atmospheric cycle imported JAX'
raise SystemExit(rc)
"""
    p = subprocess.run([sys.executable, "-c", code, "--source", str(files[0]), "--wrfinput", str(files[1]),
                        "--output", str(tmp_path / "out"), "--atmosphere", "--wrfbdy", str(files[2]),
                        "--boundary-output", str(tmp_path / "out_bdy")], capture_output=True, text=True,
                       env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": _loaded_source_path()},
                       timeout=45)
    assert p.returncode == 0, p.stderr
