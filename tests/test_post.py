"""Independent equations/upstream MetPy on tracked real CPU and GPU history."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from gpuwrf.post import read_wrfout, run_products, write_products
from gpuwrf.post.common import earth_wind, height_agl, temperature
from gpuwrf.post.products import max_random_overlap
from gpuwrf.post.registry import PRODUCTS
from gpuwrf.post.thermo import first_zero, heat_index_nws

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/post"


@pytest.fixture(scope="module")
def _cpu_history():
    return read_wrfout(sorted(FIXTURES.glob("cpu_[0-9].nc")))


@pytest.fixture
def cpu(_cpu_history):
    return _cpu_history.copy(deep=True)


def test_fixture_integrity():
    import hashlib
    manifest = json.loads((FIXTURES/"manifest.json").read_text())
    for records in manifest.values():
        for record in records:
            path = ROOT / record["fixture"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["fixture_sha256"]


def test_real_gpu_speed_and_frame_refusal():
    ds = read_wrfout(sorted(FIXTURES.glob("gpu_surface_*.nc")))
    out = run_products(ds, ["wind-speed10"])
    # Independent magnitude equation on original stored fields (not product helper).
    np.testing.assert_allclose(out.wind_speed_10m, np.sqrt(ds.U10.values**2 + ds.V10.values**2), rtol=2e-7)
    with pytest.raises(ValueError, match="earth-relative"):
        run_products(ds, ["wind10"])
    with pytest.raises(ValueError, match="missing WRF fields"):
        run_products(ds, ["instability"])


@pytest.mark.parametrize("names", [("wind-speed10", "wind10"), ("wind10", "wind-speed10")])
def test_real_wind_products_compose_in_both_orders(cpu, names, tmp_path):
    from gpuwrf.cli import main
    assert np.any(cpu.SINALPHA.values != 0)  # Real rotation exposes the old mismatch.
    speed = run_products(cpu, ["wind-speed10"])
    wind = run_products(cpu, ["wind10"])
    combined = run_products(cpu, names)
    xr.testing.assert_identical(combined.wind_speed_10m, speed.wind_speed_10m)
    xr.testing.assert_identical(combined.wind_speed_10m, wind.wind_speed_10m)
    xr.testing.assert_identical(combined.wind_from_direction_10m, wind.wind_from_direction_10m)
    expected = np.sqrt(cpu.U10.values.astype(float)**2 + cpu.V10.values.astype(float)**2)
    np.testing.assert_allclose(combined.wind_speed_10m, expected, rtol=2e-15, atol=0)
    path = tmp_path/"combined.nc"
    assert main(["post", "--input", *map(str, sorted(FIXTURES.glob("cpu_[0-9].nc"))),
                 "--product", ",".join(names), "--output", str(path)]) == 0
    with xr.open_dataset(path) as stored:
        # CF decodes the source auxiliary XTIME minutes to datetimes. Compare
        # wind values/dimensions and the primary Time coordinate exactly.
        xr.testing.assert_equal(stored.wind_speed_10m.reset_coords(drop=True),
                                combined.wind_speed_10m.reset_coords(drop=True))
        xr.testing.assert_equal(stored.wind_from_direction_10m.reset_coords(drop=True),
                                combined.wind_from_direction_10m.reset_coords(drop=True))


def test_cpu_wind_metpy_and_destagger(cpu):
    calc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    out = run_products(cpu, ["wind10", "wind-height"], {"heights": [50, 100]})
    c, s = cpu.COSALPHA.values, cpu.SINALPHA.values
    u10, v10 = cpu.U10.values*c-cpu.V10.values*s, cpu.V10.values*c+cpu.U10.values*s
    np.testing.assert_allclose(out.wind_speed_10m, calc.wind_speed(u10*units("m/s"), v10*units("m/s")).magnitude, rtol=2e-7)
    np.testing.assert_allclose(out.wind_from_direction_10m, calc.wind_direction(u10*units("m/s"), v10*units("m/s")).magnitude, atol=2e-5)
    # Reconstruct C-grid winds and geopotential independently, interpolate with NumPy.
    raw_u = cpu.U.values.astype(float)
    raw_v = cpu.V.values.astype(float)
    ug = .5*(raw_u[..., :-1]+raw_u[..., 1:])
    vg = .5*(raw_v[:, :, :-1]+raw_v[:, :, 1:])
    u, v = ug*c[:, None]-vg*s[:, None], vg*c[:, None]+ug*s[:, None]
    geop = cpu.PH.values.astype(float)+cpu.PHB.values.astype(float)
    zm = (geop[:, 1:]+geop[:, :-1])/19.62-cpu.HGT.values[:, None]
    for it, y, x in np.ndindex(3, 2, 2):
        z = np.r_[10., zm[it, :, y, x]]
        ue = np.interp([50., 100.], z, np.r_[u10[it, y, x], u[it, :, y, x]])
        ve = np.interp([50., 100.], z, np.r_[v10[it, y, x], v[it, :, y, x]])
        np.testing.assert_allclose(out.eastward_wind.values[it, :, y, x], ue, atol=1e-6)
        np.testing.assert_allclose(out.northward_wind.values[it, :, y, x], ve, atol=1e-6)
        assert out.bulk_wind_shear.values[it, y, x] == pytest.approx(np.linalg.norm([ue[1]-ue[0], ve[1]-ve[0]]), abs=1e-6)
        dirs = calc.wind_direction(ue*units("m/s"), ve*units("m/s")).magnitude
        assert out.wind_veer.values[it, y, x] == pytest.approx((dirs[1]-dirs[0]+180)%360-180, abs=1e-5)


def test_wind_boundary_and_calm(cpu):
    single = run_products(cpu, ["wind-height"], {"heights": [100]})
    assert single.sizes["height"] == 1 and "wind_veer" not in single
    with pytest.raises(ValueError, match="outside"):
        run_products(cpu, ["wind-height"], {"heights": [5, 100]})
    with pytest.raises(ValueError, match="increasing"):
        run_products(cpu, ["wind-height"], {"heights": [100, 50]})
    cpu["U10"][:] = 0
    cpu["V10"][:] = 0
    out = run_products(cpu, ["wind-height"], {"heights": [10, 100]})
    assert (out.wind_from_direction.isel(height=0) == 0).all()
    assert (out.wind_veer_valid == 0).all()
    assert (out.wind_veer == -9999).all()


def test_max_random_published_definition():
    # Contiguous blocks [.2,.6] and [.3,.5]: 1-(1-.6)*(1-.5)=.8;
    # adjacent cloudy levels overlap maximally, separated layers randomly.
    cf = np.array([.2, .6, 0, .3, .5]).reshape(1, 5, 1, 1)
    assert max_random_overlap(cf).item() == pytest.approx(.8)
    assert max_random_overlap(np.zeros_like(cf)).item() == 0
    assert max_random_overlap(np.ones_like(cf)).item() == 1


def test_real_cloud_independent_blocks(cpu):
    out = run_products(cpu, ["cloud"])
    assert np.count_nonzero(cpu.CLDFRA) > 0  # Active real cloud, not an empty fixture.
    for it, y, x in np.ndindex(3, 2, 2):
        profile = cpu.CLDFRA.values[it, :, y, x]
        blocks = []
        current = []
        for value in profile:
            if value > 0:
                current.append(float(value))
            elif current:
                blocks.append(max(current))
                current = []
        if current:
            blocks.append(max(current))
        expect = 1-np.prod([1-v for v in blocks])
        assert out.cloud_cover_total.values[it, y, x] == pytest.approx(expect, abs=1e-12)
    assert out.cloud_cover_low.attrs["pressure_lower_bound_pa"] == 68000
    assert out.cloud_cover_mid.attrs["pressure_upper_bound_pa"] == 68000
    assert out.cloud_cover_high.attrs["pressure_upper_bound_pa"] == 44000


def test_cloud_pressure_edges(cpu):
    d = cpu.isel(bottom_top=slice(0, 4)).copy(deep=True)
    d.PB.values[:] = 0
    d.P.values[:] = np.array([90000., 68000., 44000., 43999.])[None, :, None, None]
    d.CLDFRA.values[:] = np.array([.1, .4, .3, .6])[None, :, None, None]
    out = run_products(d, ["cloud"])
    np.testing.assert_allclose(out.cloud_cover_low, .4, atol=2e-8)
    np.testing.assert_allclose(out.cloud_cover_mid, .3, atol=2e-8)
    np.testing.assert_allclose(out.cloud_cover_high, .6, atol=3e-8)
    np.testing.assert_allclose(out.cloud_cover_total, .6, atol=3e-8)
    d.P.values[:] = 40000
    out = run_products(d, ["cloud"])
    assert (out.cloud_cover_low == 0).all() and (out.cloud_layer_low_sampled == 0).all()
    d.CLDFRA.values.flat[0] = 1.1
    with pytest.raises(ValueError, match="CLDFRA"):
        run_products(d, ["cloud"])


def test_bucket_interval_rolling(cpu):
    cpu = cpu.copy(deep=True)
    for name, values in {"RAINNC": [9., 3., 10.], "RAINC": [0., 0., 0.], "RAINSH": [0., 1., 3.], "I_RAINNC": [0, 1, 1], "I_RAINC": [0, 0, 0]}.items():
        cpu[name] = xr.DataArray(np.broadcast_to(np.array(values)[:, None, None], (3, 2, 2)).copy(), dims=("Time", "south_north", "west_east"))
    cpu.attrs["BUCKET_MM"] = 10.
    out = run_products(cpu, ["precip"], {"rolling_hours": [1, 2, 3]})
    np.testing.assert_array_equal(out.precipitation_accumulated[:, 0, 0], [9, 14, 23])
    np.testing.assert_array_equal(out.precipitation_interval[:, 0, 0], [-9999, 5, 9])
    np.testing.assert_array_equal(out.precipitation_rate[:, 0, 0], [-9999, 5, 4.5])
    np.testing.assert_array_equal(out.precipitation_rolling_1h[:, 0, 0], [-9999, 5, -9999])
    np.testing.assert_array_equal(out.precipitation_rolling_2h[:, 0, 0], [-9999, -9999, 9])
    np.testing.assert_array_equal(out.precipitation_rolling_3h[:, 0, 0], [-9999, -9999, 14])
    assert out.precipitation_rolling_3h_valid.values.tolist() == [0, 0, 1]
    assert out.time_bounds.values[2, 0] == cpu.Time.values[1]
    cpu.attrs["BUCKET_MM"] = -1.
    with pytest.raises(ValueError, match="bucket"):
        run_products(cpu, ["precip"])


def test_bucket_restore_is_deletion_sensitive(cpu):
    # Residuals also increase: deleting restoration cannot merely raise a reset
    # error. It must produce the wrong numbers and fail this genuine assertion.
    for name, values in {"RAINNC": [1., 3., 5.], "RAINC": [0., 0., 0.], "RAINSH": [0., 0., 0.], "I_RAINNC": [0, 1, 2], "I_RAINC": [0, 0, 0]}.items():
        cpu[name] = xr.DataArray(np.broadcast_to(np.array(values)[:, None, None], (3, 2, 2)).copy(), dims=("Time", "south_north", "west_east"))
    out = run_products(cpu, ["precip"], {"bucket_mm": 10., "rolling_hours": [3.]})
    np.testing.assert_array_equal(out.precipitation_interval.values[1:, 0, 0], [12., 12.])


def test_real_precip_and_snow_fields(cpu):
    out = run_products(cpu, ["precip", "snow", "precip-type"])
    total = cpu.RAINNC.values.astype(float)+cpu.RAINC.values.astype(float)+cpu.RAINSH.values.astype(float)
    np.testing.assert_array_equal(out.precipitation_interval[1:], np.diff(total, axis=0))
    assert np.count_nonzero(out.precipitation_interval.values[1:]) > 0
    np.testing.assert_array_equal(out.snow_depth, cpu.SNOWH)
    np.testing.assert_array_equal(out.snow_water_equivalent, cpu.SNOW)
    np.testing.assert_array_equal(out.snowfall_interval_water_equivalent[1:], np.diff(cpu.SNOWNC.values.astype(float), axis=0))


def test_resets_and_bucket_metadata_fail(cpu):
    cpu.RAINNC.values[1] = cpu.RAINNC.values[0]-1
    with pytest.raises(ValueError, match="negative|decreased"):
        run_products(cpu, ["precip"])
    cpu.RAINNC.values[:] = 0
    cpu.attrs["BUCKET_MM"] = 100.
    with pytest.raises(ValueError, match="I_RAINNC"):
        run_products(cpu, ["precip"])


def test_individual_reset_cannot_be_hidden_by_other_precip(cpu):
    cpu.RAINNC.values[:] = np.array([2., 1., 4.])[:, None, None]
    cpu.RAINC.values[:] = np.array([0., 5., 7.])[:, None, None]
    with pytest.raises(ValueError, match="RAINNC: accumulation decreased"):
        run_products(cpu, ["precip"])


def test_precip_type_defined_partition(cpu):
    one = cpu.isel(Time=[0])
    d = xr.concat([one]*5, dim="Time").assign_coords(Time=np.datetime64("2023-01-15")+np.arange(5)*np.timedelta64(1, "h"))
    for name, values in {"RAINNC": [0, 0, 2, 4, 6], "SNOWNC": [0, 0, 0, 2, 3], "GRAUPELNC": [0, 0, 0, 0, 0], "HAILNC": [0, 0, 0, 0, 0]}.items():
        d[name] = xr.DataArray(np.broadcast_to(np.array(values)[:, None, None], (5, 2, 2)).copy(), dims=("Time", "south_north", "west_east"))
    out = run_products(d, ["precip-type"])
    np.testing.assert_array_equal(out.grid_scale_precipitation_phase[:, 0, 0], [-1, 0, 1, 2, 3])
    d.attrs["MP_PHYSICS"] = 6
    with pytest.raises(ValueError, match="Thompson"):
        run_products(d, ["precip-type"])


def test_thermal_against_metpy_and_nws_example(cpu):
    calc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    out = run_products(cpu, ["thermal"])
    p, t, q = cpu.PSFC.values.astype(float), cpu.T2.values.astype(float), cpu.Q2.values.astype(float)
    e = p*q/(.6219569100577033+q)  # MetPy Rd/Rv; dry-air mixing ratio equation.
    td = calc.dewpoint(e*units.Pa)
    wet = calc.wet_bulb_temperature(p*units.Pa, t*units.K, td).to("K").magnitude
    np.testing.assert_allclose(out.wet_bulb_temperature_2m, wet, atol=1e-5)
    # Published NOAA heat index chart: 90 F / 70% RH -> about 106 F.
    got_f = (heat_index_nws(np.array(305.3722222222), np.array(.7))-273.15)*1.8+32
    assert got_f == pytest.approx(105.9220206, abs=1e-6)
    hot = np.array([305.3722222222, 310.9277777778, 302.5944444444])
    rh = np.array([.7, .1, .9])
    np.testing.assert_allclose(heat_index_nws(hot, rh), calc.heat_index(hot*units.K, rh*units.dimensionless, mask_undefined=False).to("K").magnitude, atol=1e-6)


def test_real_supersaturation_refused_for_thermal_indices():
    pytest.importorskip("metpy.calc")
    ds = read_wrfout(sorted(FIXTURES.glob("supersaturated_surface_*.nc")))
    with pytest.raises(ValueError, match="relative humidity"):
        run_products(ds, ["thermal"])


def test_instability_real_against_independent_metpy_wiring(cpu):
    calc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    out = run_products(cpu.isel(south_north=[0], west_east=[0]), ["instability"])
    for it in range(3):
        pc = cpu.P.values[it, :, 0, 0].astype(float)+cpu.PB.values[it, :, 0, 0].astype(float)
        tc = (cpu.T.values[it, :, 0, 0].astype(float)+300)*(pc/1e5)**(287/1004.5)
        p = np.r_[cpu.PSFC.values[it, 0, 0], pc]*units.Pa
        t = np.r_[cpu.T2.values[it, 0, 0], tc]*units.K
        q = np.r_[cpu.Q2.values[it, 0, 0], cpu.QVAPOR.values[it, :, 0, 0].astype(float)]
        td = calc.dewpoint(calc.vapor_pressure(p, q*units.dimensionless))
        cape, cin = calc.surface_based_cape_cin(p, t, td)
        assert out.surface_based_cape.values[it, 0, 0] == pytest.approx(cape.magnitude, abs=1e-7)
        assert out.surface_based_cin.values[it, 0, 0] == pytest.approx(cin.magnitude, abs=1e-7)
        lcl, _ = calc.lcl(p[0], t[0], td[0])
        assert out.lcl_pressure.values[it, 0, 0] == pytest.approx(lcl.to("Pa").magnitude, abs=1e-7)
        parcel = calc.parcel_profile(p, t[0], td[0])
        expected_li = calc.lifted_index(p, t, parcel).to("delta_degC").magnitude.item()
        assert out.lifted_index.values[it, 0, 0] == pytest.approx(expected_li, abs=1e-7)
        lfc, _ = calc.lfc(p, t, td, parcel_temperature_profile=parcel, which="bottom")
        if np.isfinite(lfc.magnitude):
            assert out.lfc_pressure.values[it, 0, 0] == pytest.approx(lfc.to("Pa").magnitude, abs=1e-7)
            assert out.lfc_pressure_valid.values[it, 0, 0] == 1
        else:
            assert out.lfc_pressure_valid.values[it, 0, 0] == 0
            assert out.lfc_pressure.values[it, 0, 0] == -9999


def test_active_parcel_reference_wiring(cpu):
    calc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    d = cpu.isel(Time=[0], south_north=[0], west_east=[0]).copy(deep=True)
    # Labelled analytic unstable sounding on real pressure/height levels.
    # Reference moist parcel physics is upstream MetPy, not the tested package.
    p = d.P.values.astype(float)+d.PB.values.astype(float)
    geop = d.PH.values.astype(float)+d.PHB.values.astype(float)
    z = (geop[:, 1:]+geop[:, :-1])/19.62-d.HGT.values[:, None]
    t = np.maximum(203., 305.-.007*z)
    d.T.values[:] = t/(p/1e5)**(287/1004.5)-300
    q = calc.mixing_ratio(calc.saturation_vapor_pressure((t-8)*units.K), p*units.Pa).magnitude
    d.QVAPOR.values[:] = q
    d.T2.values[:] = 305.
    d.Q2.values[:] = calc.mixing_ratio(calc.saturation_vapor_pressure(300*units.K), d.PSFC.values*units.Pa).magnitude
    out = run_products(d, ["instability"])
    pp = np.r_[float(d.PSFC.values.item()), p.ravel()]*units.Pa
    # Expected profile uses the actual float32 WRF representation after assignment.
    tt = np.r_[305., (d.T.values.astype(float).ravel()+300)*(p.ravel()/1e5)**(287/1004.5)]*units.K
    qq = np.r_[d.Q2.values.item(), d.QVAPOR.values.ravel().astype(float)]
    td = calc.dewpoint(calc.vapor_pressure(pp, qq*units.dimensionless))
    cape, cin = calc.surface_based_cape_cin(pp, tt, td)
    assert cape.magnitude > 100
    assert out.surface_based_cape.item() == pytest.approx(cape.magnitude, abs=1e-7)
    assert out.surface_based_cin.item() == pytest.approx(cin.magnitude, abs=1e-7)
    assert out.lfc_pressure_valid.item() == 1


def test_srh_against_metpy_and_explicit_motion(cpu):
    calc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    with pytest.raises(ValueError, match="explicit storm"):
        run_products(cpu, ["srh"])
    # Construct an analytic hodograph with unit rotation on the real height grid.
    cpu.COSALPHA.values[:] = 1
    cpu.SINALPHA.values[:] = 0
    cpu.U10.values[:] = 3
    cpu.V10.values[:] = 1
    cpu.U.values[:] = 8
    cpu.V.values[:] = 4
    out = run_products(cpu, ["srh"], {"storm_u": 2, "storm_v": -1})
    # Only the ground-to-first-model segment contributes: (8-2)*(1+1)-(3-2)*(4+1)=7.
    np.testing.assert_allclose(out.srh_0_1km, 7., atol=1e-12)
    np.testing.assert_allclose(out.srh_0_3km, 7., atol=1e-12)
    ref = calc.storm_relative_helicity(np.array([0, 10, 50, 3000])*units.m,
        np.array([3, 3, 8, 8])*units("m/s"), np.array([1, 1, 4, 4])*units("m/s"), 3000*units.m,
        storm_u=2*units("m/s"), storm_v=-1*units("m/s"))[2]
    assert ref.magnitude == pytest.approx(7)


def test_freezing_crossing_and_real_wet_bulb(cpu):
    pytest.importorskip("metpy.calc")
    z = np.array([0, 1000, 2000]).reshape(1, 3, 1, 1)
    t = np.array([278.15, 268.15, 260]).reshape(1, 3, 1, 1)
    values, status = first_zero(z, t)
    assert values.item() == 500 and status.item() == 1
    assert first_zero(z, t-10)[0].item() == 0
    assert first_zero(z, t+100)[1].item() == 0
    out = run_products(cpu.isel(Time=[0], south_north=[0], west_east=[0]), ["freezing"])
    assert out.freezing_level_status.item() in (1, 2)
    assert out.wet_bulb_zero_status.item() in (1, 2)
    assert out.wet_bulb_zero_height_agl.item() <= out.freezing_level_height_agl.item()+1e-6


INPUTS = sorted(set().union(*(set(p.required+p.optional) for p in PRODUCTS.values())))


@pytest.mark.parametrize("name", INPUTS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_every_consumed_field_rejects_nonfinite(cpu, name, value):
    product = next(k for k, p in PRODUCTS.items() if name in p.required+p.optional)
    if name not in cpu:
        cpu[name] = xr.zeros_like(cpu.RAINNC)
    cpu[name].values.flat[0] = value
    with pytest.raises(ValueError, match="NaN/Inf"):
        run_products(cpu, [product])


def test_history_identity_and_time_refusal(cpu, tmp_path):
    path = tmp_path/"one.nc"
    # Duplicate times are refused even if the files are equal.
    with pytest.raises(ValueError, match="duplicate"):
        read_wrfout([FIXTURES/"cpu_0.nc"]*2)
    one = xr.open_dataset(FIXTURES/"cpu_1.nc", decode_times=False).load()
    one.XLAT.values[0, 0, 0] += 1
    one.to_netcdf(path)
    with pytest.raises(ValueError, match="grid changed"):
        read_wrfout([FIXTURES/"cpu_0.nc", path])
    reversed_ds = read_wrfout(list(reversed(sorted(FIXTURES.glob("cpu_*.nc")))))
    np.testing.assert_array_equal(reversed_ds.Time, cpu.Time)
    with pytest.raises(ValueError, match="strictly increase"):
        run_products(cpu.isel(Time=[1, 0]), ["wind-speed10"])


def test_empty_history_is_not_a_product(cpu, tmp_path):
    with pytest.raises(ValueError, match="Time coordinate"):
        run_products(cpu.isel(Time=slice(0, 0)), ["wind-speed10"])
    with xr.open_dataset(FIXTURES/"cpu_0.nc", decode_times=False) as raw:
        raw.isel(Time=slice(0, 0)).to_netcdf(tmp_path/"empty.nc")
    with pytest.raises(ValueError, match="invalid WRF timestamps"):
        read_wrfout([tmp_path/"empty.nc"])


def test_cf_roundtrip_and_cli(cpu, tmp_path):
    from gpuwrf.cli import main
    path = tmp_path/"products.nc"
    rc = main(["post", "--input", *map(str, sorted(FIXTURES.glob("cpu_*.nc"))), "--product", "wind10,precip,snow,cloud", "--output", str(path)])
    assert rc == 0
    with xr.open_dataset(path) as out:
        assert out.attrs["Conventions"] == "CF-1.10"
        assert out.latitude.attrs["standard_name"] == "latitude"
        assert out.wind_speed_10m.attrs["units"] == "m s-1"
        assert out.precipitation_interval.attrs["cell_methods"] == "time: sum"
        assert np.isnan(out.precipitation_interval.values[0]).all()
        assert out.interval_valid.values.tolist() == [0, 1, 1]
        assert out.Time.attrs["bounds"] == "time_bounds"
        np.testing.assert_array_equal(out.Time, cpu.Time)
    assert main(["post", "--list-products"]) == 0
    assert main(["post", "--product", "utci", "--input", str(FIXTURES/"cpu_0.nc"), "--output", str(path)]) == 2
    assert main(["post", "--input", str(path), "--product", "wind10", "--output", str(path)]) == 2


def test_post_does_not_initialize_jax_backend(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    code = "from gpuwrf.cli import main; main(['post','--list-products']); import jax._src.xla_bridge as b; assert not b.backends_are_initialized()"
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr


def test_model_files_untouched():
    # K0 proof is additionally sealed by git tree IDs in the handoff.
    import ast
    for p in (ROOT/"src/gpuwrf/post").glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(("jax", "gpuwrf.physics", "gpuwrf.runtime")) for n in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("jax", "gpuwrf.physics", "gpuwrf.runtime"))


def test_registry_preserves_reduced_sampling_and_science_attrs(cpu, monkeypatch):
    from gpuwrf.post.registry import Product
    def daily(ds, options):
        return xr.Dataset({"daily": ds.T2.isel(Time=[1])}, attrs={"method": "independent test runner", "previous_code_date": "2023-01-14"})
    monkeypatch.setitem(PRODUCTS, "test_daily", Product(daily, ("T2",)))
    out = run_products(cpu, ["test_daily"])
    assert out.sizes["Time"] == 1
    np.testing.assert_array_equal(out.Time, cpu.Time.isel(Time=[1]))
    np.testing.assert_array_equal(out.latitude, cpu.XLAT.isel(Time=[1]))
    assert json.loads(out.attrs["product_metadata"])["test_daily"]["previous_code_date"] == "2023-01-14"
    with pytest.raises(ValueError):
        run_products(cpu, ["wind-speed10", "test_daily"])
