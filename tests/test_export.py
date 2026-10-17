"""Independent CF Appendix F, WRF destagger/rotation and MetPy interpolation checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from gpuwrf.export.core import convert, mapping, load
from gpuwrf.export.writers import write


@pytest.fixture
def history():
    nt, nz, ny, nx = 2, 3, 3, 4
    surf = ("Time", "south_north", "west_east")
    prof = ("Time", "bottom_top", "south_north", "west_east")
    ds = xr.Dataset(coords={"Time": np.array(["2026-01-01", "2026-01-01T01:00"], dtype="datetime64[ns]")},
                    attrs={"MAP_PROJ": 6, "USE_THETA_M": 0})
    lat, lon = np.meshgrid([28., 29., 30.], [-18., -17., -16., -15.], indexing="ij")
    def add(name, values, dims=surf):
        shape = (nt,nz,ny,nx) if dims == prof else (nt,ny,nx)
        ds[name] = (dims, np.broadcast_to(values, shape).copy())
    add("XLAT",lat); add("XLONG",lon); add("HGT",200.)
    add("T2", np.arange(12).reshape(3,4)+280.)
    add("Q2", .01); add("PSFC",100000.)
    add("U10", 4.); add("V10",3.); add("COSALPHA",.6); add("SINALPHA",.8)
    p = np.array([95000., 80000., 60000.])[None,:,None,None]
    add("P",p-50000.,prof); add("PB",50000.,prof); add("T",0.,prof)
    add("QVAPOR",np.log(p)*.001,prof)
    ds["U"] = (("Time","bottom_top","south_north","west_east_stag"),
               np.broadcast_to(np.arange(nx+1,dtype=float), (nt,nz,ny,nx+1)).copy())
    ds["V"] = (("Time","bottom_top","south_north_stag","west_east"),
               np.broadcast_to(np.arange(ny+1)[None,None,:,None]*2, (nt,nz,ny+1,nx)).copy())
    ds["PHB"] = (("Time","bottom_top_stag","south_north","west_east"),
                 np.broadcast_to(np.arange(nz+1)[None,:,None,None]*9.81*1000,(nt,nz+1,ny,nx)).copy())
    ds["PH"] = xr.zeros_like(ds.PHB)
    return ds


def test_surface_rotation_cf_sampling(history):
    out = convert(history,["T2","Q2","PSFC","U10","V10","HGT"])
    np.testing.assert_array_equal(out.temperature_2m, history.T2)
    np.testing.assert_allclose(out.eastward_wind_10m,0.,atol=1e-14)
    np.testing.assert_allclose(out.northward_wind_10m,5.)
    assert out.temperature_2m.encoding["coordinates"] == "latitude longitude height_2m"
    assert out.height_2m == 2 and out.height_10m == 10
    assert out.mixing_ratio_2m.attrs["standard_name"] == "humidity_mixing_ratio"
    assert out.crs.attrs == {"earth_radius":6370000.,"grid_mapping_name":"latitude_longitude"}


def test_pressure_against_metpy_and_analytic_stagger(history):
    log_interpolate_1d = pytest.importorskip("metpy.interpolate").log_interpolate_1d
    out = convert(history,["T","U","V","QVAPOR","PH"],[850,700])
    pressure = np.array([95000.,80000.,60000.])
    reference = log_interpolate_1d(np.array([85000.,70000.]), pressure, 300*(pressure/100000)**(287/1004.5))
    np.testing.assert_allclose(out.temperature[0,:,0,0],reference,rtol=1e-14)
    np.testing.assert_allclose(out.mixing_ratio[0,:,0,0],np.log([85000,70000])*.001)
    # U adjacent face average j+.5; V adjacent face average 2*i+1, then rotation.
    u = np.broadcast_to(np.arange(4)+.5,(3,4))
    v = np.broadcast_to((2*np.arange(3)+1)[:,None],(3,4))
    np.testing.assert_allclose(out.eastward_wind[0,0],.6*u-.8*v)
    np.testing.assert_allclose(out.northward_wind[0,0],.8*u+.6*v)
    z = log_interpolate_1d(np.array([85000.,70000.]),pressure,np.array([500.,1500.,2500.]))
    np.testing.assert_allclose(out.geopotential_height[0,:,0,0],z)
    assert out.pressure.attrs["positive"] == "down"


@pytest.mark.parametrize("proj,definition,attrs",[
 (1,"+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-100 +R=6370000",dict(TRUELAT1=30,TRUELAT2=60,MOAD_CEN_LAT=40,STAND_LON=-100)),
 (2,"+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-100 +R=6370000",dict(TRUELAT1=60,STAND_LON=-100)),
 (3,"+proj=merc +lat_ts=30 +lon_0=-100 +R=6370000",dict(TRUELAT1=30,STAND_LON=-100)),
 (2,"+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=20 +R=6370000",dict(TRUELAT1=-60,STAND_LON=20)),
])
def test_projection_against_explicit_proj(history,proj,definition,attrs):
    pytest.importorskip("pyproj")
    from pyproj import CRS, Transformer
    crs = CRS.from_user_input(definition)
    x,y = np.meshgrid(np.arange(4)*9000.+10000,np.arange(3)*9000.+10000)
    lon,lat=Transformer.from_crs(crs,crs.geodetic_crs,always_xy=True).transform(x,y)
    history["XLAT"][:] = lat; history["XLONG"][:] = lon
    history.attrs.update(MAP_PROJ=proj,**attrs)
    out=convert(history,["T2"])
    np.testing.assert_allclose(out.x,x[0],atol=1e-7)
    np.testing.assert_allclose(out.y,y[:,0],atol=1e-7)
    assert out.temperature_2m.attrs["grid_mapping"] == "crs"


@pytest.mark.parametrize("fmt",["netcdf","zarr"])
def test_saved_cf_rules_reopen(history,tmp_path,fmt):
    ds = convert(history,["T2","T","U"],[850,700])
    path=tmp_path/("export.nc" if fmt=="netcdf" else "export.zarr")
    write(ds,path,fmt,chunk=2)
    with (xr.open_dataset(path) if fmt=="netcdf" else xr.open_zarr(path)) as actual:
        np.testing.assert_array_equal(actual.time,ds.time)
        np.testing.assert_array_equal(actual.temperature,ds.temperature)
        assert actual.attrs["Conventions"] == "CF-1.10"
        assert actual.latitude.attrs["units"] == "degrees_north"
        assert actual.longitude.attrs["units"] == "degrees_east"
        assert actual.pressure.attrs["units"] == "Pa"
        assert "height_2m" not in actual.temperature.encoding["coordinates"]
        assert "height_2m" in actual.temperature_2m.encoding["coordinates"]
        for name in ("temperature_2m","temperature","eastward_wind"):
            assert actual[name].attrs["units"]
            assert actual[name].attrs["grid_mapping"] in actual
            assert {"latitude","longitude"} <= set(actual[name].coords)
        if fmt=="zarr":
            assert actual.temperature.encoding["chunks"]==(1,1,2,2)


def test_grib2_roundtrip_metadata_values(history,tmp_path):
    ec = pytest.importorskip("eccodes")
    ds=convert(history,["T2","PSFC","Q2","U10","V10","T","U","V","QVAPOR","PH"],[850,700])
    path=tmp_path/"export.grib2"
    write(ds,path,"grib2")
    count=0
    with path.open("rb") as stream:
        for it in range(2):
            for name,da in ds.data_vars.items():
                if name=="crs": continue
                for iz in range(da.sizes.get("pressure",1)):
                    gid=ec.codes_grib_new_from_file(stream)
                    assert gid is not None
                    try:
                        # WMO 4.2: independently fixed parameter identities (including
                        # mixing ratio, which must never be labelled specific humidity).
                        parameter = (0,0) if name.startswith("temperature") else (1,2) if "mixing_ratio" in name else (2,2) if name.startswith("eastward") else (2,3) if name.startswith("northward") else (3,0) if name == "surface_pressure" else (3,5)
                        assert (ec.codes_get_long(gid,"parameterCategory"),ec.codes_get_long(gid,"parameterNumber")) == parameter
                        assert ec.codes_get(gid,"edition")==2
                        assert ec.codes_get(gid,"dataDate")==20260101
                        assert ec.codes_get(gid,"dataTime")==it*100
                        assert ec.codes_get(gid,"forecastTime")==0
                        assert ec.codes_get(gid,"significanceOfReferenceTime")==2
                        assert ec.codes_get(gid,"typeOfGeneratingProcess")==13
                        assert ec.codes_get_long(gid,"centre")==65535
                        assert ec.codes_get(gid,"uvRelativeToGrid")==0
                        assert ec.codes_get(gid,"radius")==6370000
                        expected=da.isel(time=it,**({"pressure":iz} if "pressure" in da.dims else {}))
                        np.testing.assert_allclose(ec.codes_get_values(gid),expected.values.ravel(),rtol=1e-14)
                        np.testing.assert_allclose(ec.codes_get_array(gid,"latitudes"),ds.latitude.values.ravel(),atol=1e-6)
                        np.testing.assert_allclose(ec.codes_get_array(gid,"longitudes"),ds.longitude.values.ravel()%360,atol=1e-6)
                        if "pressure" in da.dims:
                            assert ec.codes_get(gid,"typeOfLevel")=="isobaricInhPa"
                            assert ec.codes_get(gid,"level")==[850,700][iz]
                        elif name=="temperature_2m":
                            assert ec.codes_get(gid,"shortName")=="2t"
                        count+=1
                    finally: ec.codes_release(gid)
        assert ec.codes_grib_new_from_file(stream) is None
    assert count==30


def test_grib_explicit_grid_linear_reference(history,tmp_path):
    pytest.importorskip("scipy")
    ec = pytest.importorskip("eccodes")
    ds=convert(history,["T2"])
    ds["temperature_2m"][:]=2*ds.latitude.values+3*ds.longitude.values
    path=tmp_path/"regrid.grib2"
    write(ds,path,"grib2",grib_grid=[-17.5,-15.5,28.5,29.5,3,2])
    with path.open("rb") as stream:
        gid=ec.codes_grib_new_from_file(stream)
        try:
            x,y=np.meshgrid([-17.5,-16.5,-15.5],[28.5,29.5])
            np.testing.assert_allclose(ec.codes_get_values(gid), (2*y+3*x).ravel())
        finally: ec.codes_release(gid)


def test_crop_includes_adjacent_faces_and_file_variable_selection(history,tmp_path):
    # read_wrfout honours decoded timestamps via a Times char/string field.
    history["Times"]=("Time",np.array(["2026-01-01_00:00:00","2026-01-01_01:00:00"],dtype="S19"))
    path=tmp_path/"history.nc"; history.to_netcdf(path)
    ds=load([path],["U","T2"],[850],[-17.1,-15.9,28.9,30.1])
    assert ds.sizes=={"time":2,"y":2,"x":2,"pressure":1}
    expected=history.T2.values[:,1:,1:3]
    np.testing.assert_array_equal(ds.temperature_2m,expected)


@pytest.mark.parametrize("name",["T2","XLAT","COSALPHA","P","U","PHB"])
@pytest.mark.parametrize("value",[np.nan,np.inf,-np.inf])
def test_nonfinite_refused(history,name,value):
    history[name].values.flat[0]=value
    with pytest.raises(ValueError): convert(history,["T2","U","T","PH"],[850])


@pytest.mark.parametrize("levels",[[1000],[500],[850,850],[float("nan")],[-1]])
def test_extrapolation_bad_levels(history,levels):
    with pytest.raises(ValueError): convert(history,["T"],levels)


def test_missing_wind_rotation_and_moist_theta(history):
    with pytest.raises(ValueError,match="COSALPHA"): convert(history.drop_vars("COSALPHA"),["U10"])
    history.attrs["USE_THETA_M"]=1
    with pytest.raises(ValueError,match="moist theta"): convert(history,["T"],[850])


def test_atomic_refusal(history,tmp_path):
    pytest.importorskip("eccodes")
    pytest.importorskip("scipy")
    ds=convert(history,["T2"])
    path=tmp_path/"out.grib2"; path.write_bytes(b"keep")
    with pytest.raises(ValueError,match="extrapolation"):
        write(ds,path,"grib2",grib_grid=[-30,-29,20,21,2,2])
    assert path.read_bytes()==b"keep"
    assert sorted(p.name for p in tmp_path.iterdir())==["out.grib2"]


def test_cli_backend_dark(history,tmp_path):
    history["Times"]=("Time",np.array(["2026-01-01_00:00:00","2026-01-01_01:00:00"],dtype="S19"))
    path=tmp_path/"history.nc"; history.to_netcdf(path)
    root=Path(__file__).resolve().parents[1]
    script="""import sys
from gpuwrf.cli import main
status=main(sys.argv[1:])
assert 'jax' not in sys.modules, sorted(k for k in sys.modules if k.startswith('jax'))
raise SystemExit(status)
"""
    run=subprocess.run([sys.executable,"-c",script,"export","--input",str(path),"--output",str(tmp_path/"out.nc"),"--variables","T2"],
        env={**os.environ,"PYTHONPATH":str(root/"src"),"JAX_PLATFORMS":"cpu","GPUWRF_JAX_CACHE":"0"},cwd=root,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)["variables"]==["temperature_2m"]


def test_raw_crop_preserves_history_values_attrs(history,tmp_path):
    from gpuwrf.export.core import raw_subset
    history["Times"]=("Time",np.array(["2026-01-01_00:00:00","2026-01-01_01:00:00"],dtype="S19"))
    history["U"].attrs={"units":"m s-1","stagger":"X"}
    path=tmp_path/"history.nc"; history.to_netcdf(path)
    result=raw_subset([path],["U","QVAPOR"],[-17.1,-15.9,28.9,30.1])
    np.testing.assert_array_equal(result.U,history.U.values[:,:,1:,1:4])
    np.testing.assert_array_equal(result.QVAPOR,history.QVAPOR.values[:,:,1:,1:3])
    assert result.U.attrs == history.U.attrs
    assert "Conventions" not in result.attrs
    with pytest.raises(ValueError,match="missing"):raw_subset([path],["NOT_A_FIELD"])


def test_grib_fractional_pa_refused(history,tmp_path):
    pytest.importorskip("eccodes")
    ds=convert(history,["T"],[850.123])
    with pytest.raises(ValueError,match="whole Pa"):
        write(ds,tmp_path/"out.grib2","grib2")


@pytest.mark.parametrize("selection", [[850,700], [700,850]])
def test_pressure_coordinate_order_preserves_slabs(history,selection):
    result=convert(history,["QVAPOR"],selection)
    np.testing.assert_array_equal(result.pressure.values,np.asarray(selection)*100)
    np.testing.assert_allclose(result.mixing_ratio[0,:,0,0],np.log(np.asarray(selection)*100)*.001)


@pytest.mark.parametrize("selection", [[850,650,750], [650,850,750]])
def test_unordered_pressure_coordinate_refused(history,selection):
    with pytest.raises(ValueError,match="monotonic"):
        convert(history,["QVAPOR"],selection)


def test_netcdf_dimension_coordinates_have_no_fill_metadata(history,tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    # Explicit geographic PROJ definition, independent of the export mapping.
    pyproj=pytest.importorskip("pyproj")
    crs=pyproj.CRS.from_user_input("+proj=merc +lat_ts=30 +lon_0=-100 +R=6370000")
    x,y=np.meshgrid(np.arange(4)*9000.+10000,np.arange(3)*9000.+10000)
    lon,lat=pyproj.Transformer.from_crs(crs,crs.geodetic_crs,always_xy=True).transform(x,y)
    history["XLAT"][:]=lat;history["XLONG"][:]=lon
    history.attrs.update(MAP_PROJ=3,TRUELAT1=30,STAND_LON=-100)
    ds=convert(history,["T2","QVAPOR"],[850,700])
    ds.x.attrs["missing_value"]=-9999.
    ds.pressure.encoding["_FillValue"]=-9999.
    path=tmp_path/"cf.nc"
    write(ds,path,"netcdf")
    with Dataset(path) as nc:
        for name in ("x","y","pressure","time"):
            assert nc[name].dimensions==(name,)
            assert not {"_FillValue","missing_value"}&set(nc[name].ncattrs()),name
        np.testing.assert_array_equal(nc["pressure"][:],[85000,70000])
        np.testing.assert_array_equal(nc["mixing_ratio"][:],ds.mixing_ratio.values)
    # The metadata-only serialization correction must not mutate caller state.
    assert ds.x.attrs["missing_value"]==-9999.
    assert ds.pressure.encoding["_FillValue"]==-9999.
