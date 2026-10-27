"""Deletion-sensitive VALUE witnesses for atmospheric cycling (critic F3, E39/E202/E98).

* WRFDA first boundary interval (da_update_bc.f90 :555, :911-992): with >= 2
  records the endpoint is the SECOND RECORD VALUE (time_level > 1), with one
  record it is first + tend*bdyfrq; new tend = (scnd - frst)/bdyfrq. The
  fixture's second record is NOT first + tendency*dt, so an extrapolated
  endpoint is visible in the values.
* WRFDA mass/map-factor coupling and the start_em use_input_w W gate are
  compared byte-for-byte with frozen outputs of the ORIGINAL compiled WRF/WRFDA
  Fortran (tests/data/w44_warmstart/f3b_generate.py, provenance in
  f3b_provenance.json); never with the port's own helpers.
"""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.cycling.boundary import coupled_field
from gpuwrf.cycling.core import sha256
from gpuwrf.io.netcdf_lock import Dataset
from test_atmosphere_cycle import fixture_case, literal_coupled, literal_edge, make_boundary, run_cycle


DATA = Path(__file__).resolve().parent / "data" / "w44_warmstart"
WITNESS = DATA / "f3b_witness.npz"
DT = 21600.


def _witness():
    provenance = json.loads((DATA / "f3b_provenance.json").read_text())
    assert sha256(WITNESS) == provenance["npz_sha256"], "frozen witness differs from its provenance record"
    assert provenance["imports_gpuwrf"] is False
    return np.load(WITNESS)


def _bits(a):
    a = np.asarray(a)
    assert a.dtype == np.float32
    return a.view(np.uint32)


# ---------------------------------------------------------------- M4 ----------
def _scale_tendencies(path, factor):
    with Dataset(path, "r+") as ds:
        for name in ds.variables:
            if "_BT" in name:
                ds[name][:] = np.asarray(ds[name][:], np.float32) * np.float32(factor)


def test_first_interval_endpoint_is_second_record_value_not_extrapolation(tmp_path):
    files = fixture_case(tmp_path)
    _scale_tendencies(files[2], 1.1)          # stored tendency .011, second record = first + 216 (= .01*dt)
    with Dataset(files[2]) as old:
        value_names = [n for n in old.variables if "_B" in n and "_BT" not in n and n not in ("Times",)]
        discriminating = 0
        for name in value_names:
            extrap = np.asarray(old[name][0]) + np.asarray(old[name.replace("_B", "_BT")][0]) * np.float32(DT)
            discriminating += int(not np.array_equal(extrap, np.asarray(old[name][1])))
        assert discriminating == len(value_names) > 0     # fixture precondition
    output = tmp_path / "cycled"
    report = run_cycle(files, output)
    assert report["boundary"]["fields"]
    with Dataset(output) as result, Dataset(files[2]) as old, Dataset(tmp_path / "cycled_bdy") as bdy:
        for name, record in report["boundary"]["fields"].items():
            field, side = name.rsplit("_B", 1)
            tendency = name.replace("_B", "_BT")
            start = literal_edge(literal_coupled(result, field), side)
            second = np.asarray(old[name][1])
            np.testing.assert_array_equal(bdy[name][0], start)
            # da_update_bc.f90:984 tend = (scnd - frst)/bdyfrq with scnd = record-2 VALUE (:555, time_level>1)
            np.testing.assert_array_equal(bdy[tendency][0], (second - start) / np.float32(DT), err_msg=tendency)
            np.testing.assert_array_equal(bdy[name][1:], old[name][1:])
            np.testing.assert_array_equal(bdy[tendency][1:], old[tendency][1:])
            assert record["endpoint_source"] == "second value"


def test_single_record_endpoint_is_wrfda_extrapolation(tmp_path):
    files = fixture_case(tmp_path)
    one = tmp_path / "wrfbdy_one"
    make_boundary(files[1], one, records=1)
    _scale_tendencies(one, 1.1)
    output = tmp_path / "cycled"
    report = run_cycle((files[0], files[1], one), output)
    with Dataset(output) as result, Dataset(one) as old, Dataset(tmp_path / "cycled_bdy") as bdy:
        for name, record in report["boundary"]["fields"].items():
            field, side = name.rsplit("_B", 1)
            tendency = name.replace("_B", "_BT")
            start = literal_edge(literal_coupled(result, field), side)
            # da_update_bc.f90:926 scnd = frst + tend*bdyfrq (REAL), then :984
            second = np.asarray(old[name][0]) + np.asarray(old[tendency][0]) * np.float32(DT)
            np.testing.assert_array_equal(bdy[tendency][0], (second - start) / np.float32(DT), err_msg=tendency)
            assert record["endpoint_source"] == "original extrapolation"


# ---------------------------------------------------------------- M5 ----------
_DIMS = {"U": ("bottom_top", "south_north", "west_east_stag"), "V": ("bottom_top", "south_north_stag", "west_east"),
         "W": ("bottom_top_stag", "south_north", "west_east"), "PH": ("bottom_top_stag", "south_north", "west_east"),
         "THM": ("bottom_top", "south_north", "west_east"), "QVAPOR": ("bottom_top", "south_north", "west_east"),
         "MU": ("south_north", "west_east"), "MUB": ("south_north", "west_east"),
         "MAPFAC_M": ("south_north", "west_east"), "MAPFAC_U": ("south_north", "west_east_stag"),
         "MAPFAC_V": ("south_north_stag", "west_east"), "C1H": ("bottom_top",), "C2H": ("bottom_top",),
         "C1F": ("bottom_top_stag",), "C2F": ("bottom_top_stag",)}


def _write_crop(path, z, crop):
    names = [k.split("__", 1)[1] for k in z.files if k.startswith(crop + "__") and "__ref_" not in k]
    ny, nx = z[f"{crop}__MU"].shape
    nz = z[f"{crop}__C1H"].shape[0]
    with Dataset(path, "w") as ds:
        for dim, size in (("Time", None), ("bottom_top", nz), ("bottom_top_stag", nz + 1), ("south_north", ny),
                          ("south_north_stag", ny + 1), ("west_east", nx), ("west_east_stag", nx + 1)):
            ds.createDimension(dim, size)
        ds.setncatts({"USE_THETA_M": 1})
        for name in names:
            ds.createVariable(name, "f4", ("Time",) + _DIMS[name])[0] = z[f"{crop}__{name}"]


_COUPLING_CASES = ([("d01_ytransect", f) for f in ("U", "V")] + [("d01_xtransect", f) for f in ("U", "V")]
                   + [("d03_2d", f) for f in ("U", "V", "T", "PH", "QVAPOR", "W")])


@pytest.mark.parametrize("crop,field", _COUPLING_CASES)
def test_coupling_equals_compiled_original_wrfda(tmp_path, crop, field):
    z = _witness()
    if field in ("U", "V"):    # fixture precondition: staggered map factors are not the mass map factor
        stag, mass = z[f"{crop}__MAPFAC_{field}"], z[f"{crop}__MAPFAC_M"]
        assert not np.array_equal(stag[:, :-1] if field == "U" else stag[:-1], mass)
    path = tmp_path / f"{crop}.nc"
    _write_crop(path, z, crop)
    with Dataset(path) as ds:
        result = coupled_field(ds, field)
    reference = z[f"{crop}__ref_{field}"]
    assert result.shape == reference.shape
    np.testing.assert_array_equal(_bits(result), _bits(reference), err_msg=f"{crop} {field} vs compiled WRFDA")


# ---------------------------------------------------------------- M8 ----------
_START_FIELDS = ("T", "THM", "ALB", "C1F", "C2F", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN",
                 "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")


def _write_start_input(path, z):
    ny, nx = z["setw__HGT"].shape
    nz = z["setw__RDNW"].shape[0]
    with Dataset(path, "w") as ds:
        for dim, size in (("Time", None), ("bottom_top", nz), ("bottom_top_stag", nz + 1),
                          ("south_north", ny), ("west_east", nx)):
            ds.createDimension(dim, size)
        ds.setncatts({"USE_THETA_M": 1})
        for name in _START_FIELDS:
            value = z[f"setw__{name}"]
            dims = ("bottom_top_stag",) if name in ("C1F", "C2F", "C3F", "C4F") else (
                ("bottom_top",) if value.ndim == 1 else ("bottom_top", "south_north", "west_east"))
            ds.createVariable(name, "f4", ("Time",) + dims)[0] = value
        ds.createVariable("P_TOP", "f4", ("Time",))[0] = z["setw__P_TOP"]


@pytest.mark.parametrize("w_input,use_input_w", [("W_NONZERO_SFC", False), ("W_NONZERO_SFC", True),
                                                  ("W_ZERO_SFC", False), ("W_ZERO_SFC", True)])
def test_start_domain_cycling_w_equals_compiled_original_wrf(tmp_path, w_input, use_input_w):
    from gpuwrf.io.cycling_init import start_domain_cycling
    z = _witness()
    label = f"{'nonzero' if w_input == 'W_NONZERO_SFC' else 'zero'}_sfc_use{int(use_input_w)}"
    w_in = z[f"setw__{w_input}"].copy()
    wrfinput = tmp_path / "wrfinput_d01"
    _write_start_input(wrfinput, z)
    run = SimpleNamespace(namelist={"dynamics": {"use_input_w": use_input_w}},
                          wrfinput_file=lambda domain: wrfinput, _read_dataset=lambda path: Dataset(path))
    dx, dy = (float(v) for v in z["setw__DXDY"])
    grid = SimpleNamespace(terrain_height=z["setw__HGT"], vertical=SimpleNamespace(eta_levels=z["setw__ZNW"]),
                           projection=SimpleNamespace(dx_m=dx, dy_m=dy))
    cf = z["setw__CF"]
    metrics = SimpleNamespace(msftx=z["setw__MAPFAC_MX"], msfty=z["setw__MAPFAC_MY"],
                              cf1=cf[0:1], cf2=cf[1:2], cf3=cf[2:3])
    theta = z["setw__THM"].astype(np.float64) + 300.
    *_, w_out, meta = start_domain_cycling(
        run, "d01", grid=grid, metrics=metrics, pb=z["setw__PB"], phb=z["setw__PHB"], mub=z["setw__MUB"],
        mu=z["setw__MU"], prognostic_theta=theta, qv=z["setw__QVAPOR"], u=z["setw__U"], v=z["setw__V"], w=w_in)
    reference = z[f"setw__ref_W_{label}"]
    np.testing.assert_array_equal(_bits(w_out), _bits(reference), err_msg=f"W {label} vs compiled start_em/set_w_surface")
    assert meta["w_reinitialized"] == bool(int(z[f"setw__ref_set_{label}"])), label
    assert meta["use_input_w"] is use_input_w
