"""Literal WRF-coupling controls for the mass-frozen LBC IO prototype."""
from datetime import datetime, timedelta
import hashlib

import numpy as np
from netCDF4 import Dataset
import pytest

from gpuwrf.assimilation.letkf.lbc_perturb import perturb_wrfbdy
from gpuwrf.io.boundary_replay import decode_wrfbdy

R = np.float32
RV = float(R(R(461.6) / R(287.)))
SIDES = ("XS", "XE", "YS", "YE")
FIELDS = ("U", "V", "T", "QVAPOR")
START = datetime(2026, 5, 3)
FMT = "%Y-%m-%d_%H:%M:%S"


def strips(a):
    # Independent indexed stuffing, including reversed east/north storage.
    if a.ndim == 2:
        a = a[None]
        return {s: x[:, 0] for s, x in strips(a).items()}
    z, y, x = a.shape
    out = {s: np.empty((2, z, y if s in ("XS", "XE") else x), a.dtype) for s in SIDES}
    for b in range(2):
        out["XS"][b], out["XE"][b] = a[:, :, b], a[:, :, x - 1 - b]
        out["YS"][b], out["YE"][b] = a[:, b, :], a[:, y - 1 - b, :]
    return out


def factors(case, index):
    # Scalar-loop literal of pristine calc_mu_staggered/couple. No production
    # factor/strip helper is imported; maps are deliberately anisotropic.
    mu, mub, c1, c2, mapu, mapv = (case[k] for k in ("mu", "mub", "c1", "c2", "mapu", "mapv"))
    mu = mu[index]
    nz, ny, nx = len(c1), *mub.shape
    f = {"U": np.empty((nz, ny, nx + 1), float), "V": np.empty((nz, ny + 1, nx), float),
         "T": np.empty((nz, ny, nx), float)}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                f["T"][k, j, i] = R(R(c1[k] * mu[j, i]) + R(R(c1[k] * mub[j, i]) + c2[k]))
        for j in range(ny):
            for i in range(nx + 1):
                m = R(mu[j, min(i, nx - 1)] + mub[j, min(i, nx - 1)]) if i in (0, nx) else R(R(.5) * R(R(R(mu[j, i] + mu[j, i - 1]) + mub[j, i]) + mub[j, i - 1]))
                f["U"][k, j, i] = float(R(R(c1[k] * m) + c2[k])) / float(mapu[j, i])
        for j in range(ny + 1):
            for i in range(nx):
                m = R(mu[min(j, ny - 1), i] + mub[min(j, ny - 1), i]) if j in (0, ny) else R(R(.5) * R(R(R(mu[j, i] + mu[j - 1, i]) + mub[j, i]) + mub[j - 1, i]))
                f["V"][k, j, i] = float(R(R(c1[k] * m) + c2[k])) / float(mapv[j, i])
    f["QVAPOR"] = f["T"]
    return {name: strips(a) for name, a in f.items()}


def decoded(path, case, index):
    raw = decode_wrfbdy(path, variables=FIELDS, time_index=index)["variables"]
    m = factors(case, index)
    side_names = {"XS": "W", "XE": "E", "YS": "S", "YE": "N"}
    out = {name: {s: raw[name]["sides"][side_names[s]]["boundary"] / m[name][s] for s in SIDES} for name in FIELDS}
    for s in SIDES:
        theta = out["T"][s] + 300.
        out["T"][s] = theta / (1. + RV * out["QVAPOR"][s]) if case["theta_m"] else theta
    return out


@pytest.fixture(params=[0, 1])
def case(tmp_path, request):
    source = tmp_path / "source"
    source.mkdir()
    nz, ny, nx = 3, 6, 7
    c = {"source": source / "wrfbdy_d01", "context": source / "wrfinput_d01", "theta_m": request.param,
         "c1": np.array([.73, .31, .07], np.float32), "c2": np.array([5000, 12000, 3000], np.float32),
         "mub": (80000 + np.arange(ny * nx).reshape(ny, nx) * 11).astype(np.float32),
         "mapu": np.full((ny, nx + 1), 1.17, np.float32), "mapv": np.full((ny + 1, nx), .87, np.float32),
         "mu": [(100 + k * 31 + np.arange(ny * nx).reshape(ny, nx) * .7).astype(np.float32) for k in range(4)]}
    attrs = {"GRID_ID": 1, "DX": 9000., "DY": 9000., "HYBRID_OPT": 2, "USE_THETA_M": request.param}
    with Dataset(c["context"], "w") as ds:
        for name, size in (("Time", 1), ("bottom_top", nz), ("south_north", ny), ("west_east", nx), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1)):
            ds.createDimension(name, size)
        ds.setncatts(attrs)
        for name, value, dims in (("MUB", c["mub"], ("south_north", "west_east")), ("C1H", c["c1"], ("bottom_top",)),
                                  ("C2H", c["c2"], ("bottom_top",)), ("MAPFAC_UY", c["mapu"], ("south_north", "west_east_stag")),
                                  ("MAPFAC_VX", c["mapv"], ("south_north_stag", "west_east"))):
            ds.createVariable(name, "f4", ("Time",) + dims)[0] = value
    times = [START + timedelta(hours=6 * k) for k in range(4)]
    c["deviations"] = {t.strftime(FMT): {"U": np.full((nz, ny, nx + 1), .5 * (k + 1), np.float32),
                                         "V": np.full((nz, ny + 1, nx), -.25 * (k + 1), np.float32),
                                         "T": np.full((nz, ny, nx), .25 * (k + 1), np.float32),
                                         "QVAPOR": np.full((nz, ny, nx), .0001 * (k + 1), np.float32)} for k, t in enumerate(times)}
    values = {name: {s: [] for s in SIDES} for name in (*FIELDS, "MU")}
    for k in range(4):
        m = factors(c, k)
        full = {"U": np.full((nz, ny, nx + 1), 5. + k, np.float32), "V": np.full((nz, ny + 1, nx), -3. - k, np.float32),
                "T": np.full((nz, ny, nx), 290. + k, np.float32), "QVAPOR": np.full((nz, ny, nx), .01 + .001 * k, np.float32)}
        full["U"][:, 0, 0] = R(-0.)
        for name in FIELDS:
            for side, data in strips(full[name]).items():
                field = data.astype(float)
                if name == "T":
                    q = strips(full["QVAPOR"])[side]
                    field = R(R(data * R(R(1.) + R(R(RV) * q))) - R(300.)) if request.param else R(data - R(300.))
                if name in ("U", "V"):
                    mapping = strips(c["mapu"] if name == "U" else c["mapv"])[side][..., None, :]
                    mass = R(m[name][side] * mapping)
                    encoded = R(R(R(field) * mass) / mapping)
                else:
                    encoded = R(R(field) * R(m[name][side]))
                values[name][side].append(encoded)
        for side, data in strips(c["mu"][k]).items():
            values["MU"][side].append(data)
    with Dataset(c["source"], "w") as ds:
        for name, size in (("Time", 3), ("DateStrLen", 19), ("bdy_width", 2), ("bottom_top", nz), ("south_north", ny),
                           ("west_east", nx), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1)):
            ds.createDimension(name, size)
        ds.setncatts(attrs)
        ds.literal_attribute = "preserve exactly"
        for name, stamps in (("Times", times[:-1]), ("md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", times[:-1]),
                             ("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", times[1:])):
            v = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
            for k, t in enumerate(stamps):
                v[k] = np.frombuffer(t.strftime(FMT).encode(), dtype="S1")
        for name in (*FIELDS, "MU"):
            for side in SIDES:
                tangent = ("south_north_stag" if name == "V" else "south_north") if side[0] == "X" else ("west_east_stag" if name == "U" else "west_east")
                dims = ("Time", "bdy_width") + (() if name == "MU" else ("bottom_top",)) + (tangent,)
                a = np.asarray(values[name][side])
                ds.createVariable(f"{name}_B{side}", "f4", dims)[:] = a[:-1]
                ds.createVariable(f"{name}_BT{side}", "f4", dims)[:] = np.float32(np.float32(a[1:] - a[:-1]) / R(21600))
        ds.createVariable("passive", "f4", ("Time", "west_east"))[:] = np.arange(3 * nx).reshape(3, nx)
    return c


def test_zero_is_wholefile_byte_identical_including_signed_zero(case, tmp_path):
    out = tmp_path / "out" / "zero"
    zeros = {t: {} for t in case["deviations"]}
    report = perturb_wrfbdy(case["source"], zeros, out)
    assert out.read_bytes() == case["source"].read_bytes()
    assert report["zero_deviation"] and not report["changed_variables"]


def test_independent_physical_decode_and_tendency_gate(case, tmp_path):
    out = tmp_path / "out" / "perturbed"
    before = hashlib.sha256(case["source"].read_bytes()).hexdigest()
    report = perturb_wrfbdy(case["source"], case["deviations"], out)
    for k, record in enumerate(list(case["deviations"].values())[:-1]):
        old, new = decoded(case["source"], case, k), decoded(out, case, k)
        for name in FIELDS:
            for side, delta in strips(record[name]).items():
                assert np.isfinite(old[name][side]).all() and np.isfinite(new[name][side]).all()
                scale = np.maximum(np.abs(old[name][side]), np.abs(old[name][side] + delta)).astype(np.float32)
                assert np.all(np.abs((new[name][side] - old[name][side]) - delta) <= np.abs(np.spacing(scale)))
    with Dataset(case["source"]) as src, Dataset(out) as dst:
        assert src.ncattrs() == dst.ncattrs()
        for name in src.variables:
            if name not in report["changed_variables"]:
                assert np.asarray(src[name][:]).tobytes() == np.asarray(dst[name][:]).tobytes(), name
        for name in FIELDS:
            for side in SIDES:
                a = np.asarray(dst[f"{name}_B{side}"][:])
                bt = np.asarray(dst[f"{name}_BT{side}"][:-1])
                assert np.array_equal(bt.view("u4"), np.float32(np.float32(a[1:] - a[:-1]) / R(21600)).view("u4"))
    assert before == hashlib.sha256(case["source"].read_bytes()).hexdigest()


def test_qv_increment_converts_moist_theta_without_heating_dry_theta(case, tmp_path):
    for fields in case["deviations"].values():
        fields.pop("U"); fields.pop("V"); fields.pop("T")
    out = tmp_path / "out" / "qv"
    report = perturb_wrfbdy(case["source"], case["deviations"], out)
    changed_t = any(v.startswith("T_") for v in report["changed_variables"])
    assert changed_t == bool(case["theta_m"])
    for k in range(3):
        old, new = decoded(case["source"], case, k), decoded(out, case, k)
        for side in SIDES:
            assert np.all(np.abs(new["T"][side] - old["T"][side]) <= np.spacing(old["T"][side].astype(np.float32)))


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_refusal(case, tmp_path, field, bad):
    next(iter(case["deviations"].values()))[field].flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        perturb_wrfbdy(case["source"], case["deviations"], tmp_path / "out" / "bad")
    assert not list(tmp_path.glob("out/.lbc-perturb-*"))


def test_missing_endpoint_and_mass_perturbation_refuse(case, tmp_path):
    records = dict(case["deviations"])
    records.pop(list(records)[-1])
    with pytest.raises(ValueError, match="endpoint"):
        perturb_wrfbdy(case["source"], records, tmp_path / "out" / "bad")
    next(iter(case["deviations"].values()))["MU"] = np.ones_like(case["mub"])
    with pytest.raises(ValueError, match="MU is frozen"):
        perturb_wrfbdy(case["source"], case["deviations"], tmp_path / "out" / "mass")


def test_negative_moisture_and_overwrite_refuse(case, tmp_path):
    for fields in case["deviations"].values():
        fields["QVAPOR"][:] = -.5
    out = tmp_path / "out" / "bad"
    with pytest.raises(ValueError, match="negative QVAPOR"):
        perturb_wrfbdy(case["source"], case["deviations"], out)
    assert not out.exists()
    out.write_text("sentinel")
    with pytest.raises(ValueError, match="overwrite"):
        perturb_wrfbdy(case["source"], case["deviations"], out)
    assert out.read_text() == "sentinel"


@pytest.mark.parametrize("mutant", ["shape", "metadata", "map", "packed", "physical_overflow"])
def test_refusal_controls(case, tmp_path, mutant):
    if mutant == "shape":
        next(iter(case["deviations"].values()))["U"] = np.zeros((2, 2))
    elif mutant == "metadata":
        with Dataset(case["source"], "r+") as ds:
            ds["md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"][0] = ds["Times"][0]
    elif mutant == "packed":
        with Dataset(case["source"], "r+") as ds:
            ds["U_BXS"].scale_factor = 2.
    else:
        with Dataset(case["context"], "r+") as ds:
            if mutant == "map":
                ds["MAPFAC_UY"][0, 0, 0] = 0
            else:
                ds["C1H"][:] = 0
                ds["C2H"][:] = 1e-40
    with pytest.raises(ValueError):
        perturb_wrfbdy(case["source"], case["deviations"], tmp_path / "out" / "bad")
