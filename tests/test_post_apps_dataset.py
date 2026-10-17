"""End-to-end WRF application adapters and CLI contracts."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from gpuwrf.post import run_products
from gpuwrf.post.apps import PRODUCT_NAMES, register_products
from gpuwrf.post.apps.dataset import fwi_product

SURFACE = ("Time", "south_north", "west_east")
PROFILE = ("Time", "bottom_top", "south_north", "west_east")
FWI_OPTIONS = dict(ffmc_previous=85, dmc_previous=6, dc_previous=15, utc_offset_hours=0)


@pytest.fixture
def wrf():
    """Small WRF grid; independent known heights/staggered fields, two fire days."""
    stamps = np.arange(np.datetime64("1985-04-12T12"), np.datetime64("1985-04-14T13"), np.timedelta64(1,"h"))
    nt = len(stamps)
    ds = xr.Dataset(coords={"Time": stamps})
    def surf(name, value):
        ds[name] = xr.DataArray(np.broadcast_to(np.asarray(value).reshape((-1,1,1)) if np.ndim(value) else value, (nt,2,2)).copy(), dims=SURFACE)
    for name, value in dict(XLAT=40., XLONG=-100., HGT=0., T2=290.15, TSK=272.15,
                            PSFC=100000., U10=25/3.6, V10=0., RAINNC=0., RAINC=0.,
                            SWDOWN=500., COSALPHA=1., SINALPHA=0.).items():
        surf(name,value)
    temps = np.full(nt, 290.15);temps[-1]=293.15;surf("T2",temps)
    rh = np.full(nt,42.);rh[-1]=21.
    e = rh/100*611.2*np.exp(17.67*(temps-273.15)/(temps-29.65))
    surf("Q2", .622*e/(100000-e))
    rain = np.maximum(0,np.arange(nt)-24)*.1;surf("RAINNC",rain)
    p = np.array([99000.,98000.,97000.,90000.])[None,:,None,None]
    temperature = np.array([269.,268.,267.,260.])[None,:,None,None]
    def profile(name,a):
        ds[name] = xr.DataArray(np.broadcast_to(a,(nt,4,2,2)).copy(),dims=PROFILE)
    profile("P",p-90000);profile("PB",90000.)
    profile("T",temperature/(p/100000)**(287/1004.5)-300)
    for name,value in dict(QVAPOR=.002,QCLOUD=.0002,QRAIN=.0001,QICE=.00001,QSNOW=.0001).items():
        profile(name,value)
    for name,values in {"PH":np.zeros(5), "PHB":np.array([0,60,100,180,1420])*9.81}.items():
        ds[name] = xr.DataArray(np.broadcast_to(values[None,:,None,None],(nt,5,2,2)).copy(), dims=("Time","bottom_top_stag","south_north","west_east"))
    ds["U"] = xr.DataArray(np.broadcast_to(np.array([4.,6.,9.,12.])[None,:,None,None],(nt,4,2,3)).copy(), dims=("Time","bottom_top","south_north","west_east_stag"))
    ds["V"] = xr.DataArray(np.broadcast_to(np.array([0.,2.,4.,6.])[None,:,None,None],(nt,4,3,2)).copy(), dims=("Time","bottom_top","south_north_stag","west_east"))
    text = [np.datetime_as_string(stamp, unit="s").replace("T", "_") for stamp in stamps]
    ds["Times"] = xr.DataArray(np.array([list(s) for s in text], dtype="S1"), dims=("Time", "DateStrLen"))
    return ds


def test_all_apps_registered_idempotently():
    from gpuwrf.post.registry import PRODUCTS
    register_products(); register_products()
    assert set(PRODUCT_NAMES).issubset(PRODUCTS)


def test_wind_stagger_height_and_density_energy(wrf,tmp_path):
    path=tmp_path/"curve.csv";path.write_text("speed_ms,power_w\n0,0\n10,1000\n25,0\n")
    result=run_products(wrf,["app_wind"],dict(power_curve=str(path),hub_height=100,rotor_diameter=100))
    # Model masses at30/80/140/800m; 100m component interpolation = u7,v8/3.
    assert_allclose(result.hub_wind_speed, np.hypot(7,8/3),rtol=1e-14)
    assert np.all(result.wind_power>0)
    assert_allclose(result.wind_energy[-1], result.wind_power[0]*48,rtol=1e-14)
    metadata=json.loads(result.attrs["product_metadata"])["app_wind"]
    assert metadata["hub_height_m"]==100
    assert "Instantaneous" in metadata["limitations"]


def test_solar_native_vs_erbs_and_metadata(wrf):
    pytest.importorskip("pvlib")
    wrf=wrf.isel(Time=slice(0,2)).copy()
    wrf["SWDDNI"]=xr.full_like(wrf.SWDOWN,200)
    wrf["SWDDIF"]=xr.full_like(wrf.SWDOWN,300)
    native=run_products(wrf,["app_solar"],dict(cell_temperature_c=25))
    assert_allclose(native.dni,200)
    assert_allclose(native.dhi,300)
    assert_allclose(native.pv_dc_power,native.poa_irradiance)
    meta=json.loads(native.attrs["product_metadata"])["app_solar"]
    assert meta["irradiance_method"]=="WRF SWDDNI/SWDDIF"
    decomp=run_products(wrf.drop_vars(["SWDDNI","SWDDIF"]),["app_solar"])
    assert "Erbs" in json.loads(decomp.attrs["product_metadata"])["app_solar"]["irradiance_method"]
    with pytest.raises(ValueError,match="both SWDDNI"):
        run_products(wrf.drop_vars("SWDDNI"),["app_solar"])


def test_icing_frost_fire_visibility_compatible_merge(wrf):
    result=run_products(wrf,["app_icing","app_frost","app_fire","app_visibility"])
    assert np.all(result.icing_potential==1)
    assert np.all(result.air_frost==0)
    assert np.all(result.surface_frost==1)
    assert np.all(result.visibility<1000)
    assert np.all(result.hdw_instantaneous>0)
    assert result.supercooled_liquid_water.dims==PROFILE
    for key in ["visibility","air_frost","hdw_instantaneous"]:
        assert result[key].attrs["cell_methods"]=="time: point"
        assert "units" in result[key].attrs


def test_fwi_noon_subset_against_published_two_day_example(wrf):
    result=run_products(wrf,["app_fwi"],FWI_OPTIONS)
    assert result.sizes["Time"]==2
    assert_allclose(result.fwi_rain_24h[:,0,0],[0,2.4],atol=1e-12)
    path=Path(__file__).parent/"fixtures/post_apps/fwi_1985.csv"
    with path.open() as stream:
        rows=list(csv.DictReader(stream))[:2]
    for i,row in enumerate(rows):
        for name in ("ffmc","dmc","dc","isi","bui","fwi"):
            assert_allclose(result["fwi_"+name][i],float(row[name.upper()]),atol=.051,rtol=0)
    assert np.all(result.time_bounds[:,1]-result.time_bounds[:,0]==np.timedelta64(24,"h"))
    metadata=json.loads(result.attrs["product_metadata"])["app_fwi"]
    assert metadata["previous_code_date_utc"].startswith("1985-04-12T12")
    # The core must not silently expand daily codes to hourly NaN fields.
    assert np.isfinite(result.fwi_fwi).all()
    with pytest.raises(ValueError):
        run_products(wrf,["app_fwi","app_frost"],FWI_OPTIONS)


def test_fwi_bucket_counter_equivalence(wrf):
    regular=fwi_product(wrf,FWI_OPTIONS)
    bucketed=wrf.copy(deep=True);bucketed.attrs["BUCKET_MM"]=1
    for key in ("RAINNC","RAINC"):
        bucketed["I_"+key]=np.floor(bucketed[key]).astype(int)
        bucketed[key]=bucketed[key]%1
    result=fwi_product(bucketed,FWI_OPTIONS)
    assert_allclose(result.fwi_fwi,regular.fwi_fwi,rtol=1e-14)
    with pytest.raises(ValueError,match="missing WRF field I_RAINNC"):
        fwi_product(bucketed.drop_vars("I_RAINNC"),FWI_OPTIONS)


def test_fwi_shallow_rain_is_unbucketed_like_pristine_wrf(wrf):
    wrf["RAINSH"] = xr.DataArray(np.broadcast_to((np.arange(49)*.02)[:,None,None],(49,2,2)),dims=SURFACE)
    wrf.attrs["BUCKET_MM"] = 1
    for key in ("RAINNC","RAINC"):
        wrf["I_"+key] = np.floor(wrf[key]).astype(int)
        wrf[key] = wrf[key]%1
    # WRF Registry has I_RAINNC/I_RAINC, and no I_RAINSH.
    result=run_products(wrf,["app_fwi"],FWI_OPTIONS)
    assert_allclose(result.fwi_rain_24h[:,0,0],[.48,2.88],atol=1e-14)


def test_fwi_nonzero_bucket_counter_without_bucket_metadata_refused(wrf):
    wrf["I_RAINNC"] = xr.ones_like(wrf.RAINNC,dtype=int)
    with pytest.raises(ValueError,match="nonzero rain bucket counters"):
        run_products(wrf,["app_fwi"],FWI_OPTIONS)


def test_fwi_local_standard_noon_is_not_utc_noon(wrf):
    shifted = wrf.assign_coords(Time=wrf.Time.values - np.timedelta64(2,"h"))
    options = dict(FWI_OPTIONS, utc_offset_hours=2)
    result = run_products(shifted,["app_fwi"],options)
    assert_allclose(result.fwi_ffmc[:,0,0],[87.6,86.2],atol=.051,rtol=0)
    assert result.Time.values[0] == np.datetime64("1985-04-13T10")


@pytest.mark.parametrize("change",["no_codes","no_offset","no_antecedent","missing_noon","gap","reset","nan_rain","bad_counter"])
def test_fwi_bad_inputs_refused(wrf,change):
    options=dict(FWI_OPTIONS)
    if change=="no_codes":options.pop("ffmc_previous")
    elif change=="no_offset":options.pop("utc_offset_hours")
    elif change=="no_antecedent":wrf=wrf.isel(Time=slice(-24,None))
    elif change=="missing_noon":wrf=wrf.isel(Time=[i for i in range(49) if i!=24])
    elif change=="gap":
        # Extend to a third day but remove day2; a recurrence cannot jump days.
        day3=wrf.isel(Time=slice(-1,None)).assign_coords(Time=[np.datetime64("1985-04-15T12")])
        wrf=xr.concat([wrf,day3],dim="Time").isel(Time=[i for i in range(50) if i!=48])
    elif change=="reset":wrf["RAINNC"].values[-1]=0
    elif change=="nan_rain":wrf["RAINNC"].values[0,0,0]=np.nan
    elif change=="bad_counter":
        wrf.attrs["BUCKET_MM"]=1
        wrf["I_RAINNC"]=xr.full_like(wrf.RAINNC,.5)
        wrf["I_RAINC"]=xr.zeros_like(wrf.RAINC)
    with pytest.raises(ValueError):
        fwi_product(wrf,options)


@pytest.mark.parametrize("product,fieldname",[("app_wind","U"),("app_solar","SWDOWN"),("app_icing","QCLOUD"),("app_fire","QVAPOR"),("app_frost","T2"),("app_visibility","QSNOW"),("app_fwi","RAINC")])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_apps_reject_nonfinite_field(wrf,product,fieldname,bad):
    wrf[fieldname].values.flat[0]=bad
    with pytest.raises(ValueError,match="NaN/Inf|finite"):
        run_products(wrf,[product],FWI_OPTIONS)


def test_missing_optional_power_thermo_refused_when_used(wrf,tmp_path):
    path=tmp_path/"curve.csv";path.write_text("speed_ms,power_w\n0,0\n25,1000\n")
    with pytest.raises(ValueError,match="missing WRF field QVAPOR"):
        run_products(wrf.drop_vars("QVAPOR"),["app_wind"],dict(power_curve=str(path)))


def test_icing_negative_trace_is_explicitly_refused(wrf):
    wrf["QCLOUD"].values[0,0,0,0] = -1.49e-19
    with pytest.raises(ValueError, match="qc must be >= 0"):
        run_products(wrf,["app_icing"])


def _cli(args,cwd):
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN","PASSWORD","SECRET","API_KEY","CREDENTIAL"))}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"),JAX_PLATFORMS="cpu",GPUWRF_JAX_CACHE="0")
    return subprocess.run([sys.executable,"-m","gpuwrf.cli","post",*args],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)


def test_cli_all_products_and_fwi_subset_metadata(wrf,tmp_path):
    pytest.importorskip("pvlib")
    path=tmp_path/"wrfout.nc";wrf.to_netcdf(path)
    out=tmp_path/"applications.nc"
    result=_cli(["--input",str(path),"--output",str(out),"--product","app_wind,app_solar,app_icing,app_fire,app_frost,app_visibility"],tmp_path)
    assert result.returncode==0,result.stderr
    with xr.open_dataset(out) as output:
        assert output.attrs["Conventions"].startswith("CF")
        assert "app_solar" in json.loads(output.attrs["product_metadata"])
        assert output.pv_dc_power.attrs["units"]=="W"
        assert np.isfinite(output.pv_dc_power).all()
    fwiout=tmp_path/"fwi.nc"
    args=["--input",str(path),"--output",str(fwiout),"--product","app_fwi"]
    for k,v in FWI_OPTIONS.items():args.extend(["--option",f"{k}={v}"])
    result=_cli(args,tmp_path)
    assert result.returncode==0,result.stderr
    with xr.open_dataset(fwiout) as output:
        assert output.sizes["Time"]==2
        assert np.isfinite(output.fwi_fwi).all()
        assert output.time_bounds.dtype.kind=="M"
    result=_cli(["--input",str(path),"--output",str(out),"--product","app_fwi"],tmp_path)
    assert result.returncode==2
    assert "previous codes" in result.stderr


def test_archive_contains_references():
    # The archive runner loads every fixture without replaying numeric gates.
    root = Path(__file__).parent / "fixtures/post_apps"
    for name in ("fwi_1985.csv","fwi_reference.json","REFERENCES.json"):
        assert (root/name).is_file()
    provenance = json.loads((root/"REFERENCES.json").read_text())
    assert hashlib.sha256((root/"fwi_1985.csv").read_bytes()).hexdigest() == provenance["fwi_1985.csv"]["sha256"]
    with (root/"fwi_1985.csv").open() as stream:
        assert next(csv.DictReader(stream))["FFMC"] == "87.6"
    fixture = json.loads((root/"fwi_reference.json").read_text())
    assert len(fixture["cases"]) == 108
    assert fixture["source_sha256"] == "e572ac19b5f0d1ba76aa78b3b0d41119a71a1b44e98bb334696351d0ccfc7aaa"
