"""Pristine-WRF operator and public history/restart seams, including active resets."""
from dataclasses import replace
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.diagnostics.precipitation import (
    PrecipitationBuckets, history_fields, seed_buckets, thompson_amounts, update_buckets,
)
from gpuwrf.io.namelist_binding import bind_namelist_options, namelist_honour_report
from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
from test_m7_netcdf_writer import writer_authority

FIXTURE = Path(__file__).resolve().parents[1] / "proofs/u2_diag/precipitation_oracle.json"


@pytest.mark.parametrize("compiled", [False, True])
def test_pristine_sequential_bucket_operator(compiled):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["precision"] == "WRF REAL=4" and len(fixture["source_sha256"]) == 64
    operator = jax.jit(update_buckets) if compiled else update_buckets
    for case in fixture["cases"]:
        buckets = PrecipitationBuckets(*(jnp.asarray(x, jnp.float32) for x in case["initial"]))
        for clock, inputs, expected in zip(case["clocks"], case["inputs"], case["expected"], strict=True):
            buckets = operator(buckets, *map(lambda x: jnp.asarray(x, jnp.float32), inputs), clock, case["interval"])
            actual, expected = np.asarray(buckets), np.asarray(expected, np.float32)
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            assert actual.dtype == np.float32
            np.testing.assert_array_max_ulp(actual, expected, maxulp=2 if compiled else 0)


def test_thompson_raw_amounts_and_spec_zone():
    precip = {k: jnp.full((5,6), v, jnp.float32) for k,v in
              dict(rain=.18, snow=.09, graupel=.036, ice=.018).items()}
    total, snow = thompson_amounts(precip, spec_zone=1)
    # Independent literal WRF module_mp_thompson.F expressions in NumPy REAL.
    expected = ((np.float32(.18)+np.float32(.09))+np.float32(.036))+np.float32(.018)
    np.testing.assert_array_equal(np.asarray(total)[1:-1,1:-1], expected)
    np.testing.assert_array_equal(np.asarray(snow)[1:-1,1:-1], np.float32(.09)+np.float32(.018))
    for a in (total, snow):
        assert np.all(np.asarray(a)[[0,-1],:] == 0) and np.all(np.asarray(a)[:,[0,-1]] == 0)
    assert np.float32(1e8+.18)-np.float32(1e8) == 0 and float(total[1,1]) > 0


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", False)
    grid = make_ideal_grid(4,3,5)
    state = _state(grid)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, prec_acc_dt=60.)
    shape = state.t_skin.shape
    buckets = PrecipitationBuckets(*(jnp.arange(np.prod(shape),dtype=jnp.float32).reshape(shape)/32+i for i in range(3)))
    return grid, nl, initial_operational_carry(state).replace(precipitation_buckets=buckets)


def save(path, fmt, grid, nl, carry):
    if fmt == "pickle":
        rp.write_restart(carry,nl,grid,2,path)
    else:
        nc.write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:30:00",
                             run_start="2026-06-03_00:00:00",step_index=2)


def read(path,fmt,nl=None):
    return (rp.read_restart(path) if fmt == "pickle" else nc.read_wrfrst_carry(path,namelist=nl))[0]


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_buckets_restart_exact(case,tmp_path,fmt):
    grid,nl,carry = case
    path = tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    restored = read(path,fmt)
    assert restored.precipitation_buckets._fields == carry.precipitation_buckets._fields
    for x,y in zip(restored.precipitation_buckets,carry.precipitation_buckets,strict=True):
        assert np.asarray(x).dtype == np.float32
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("damage", ["marker", "group", "leaf"])
def test_missing_active_restart_buckets_refused(case,tmp_path,fmt,damage):
    grid,nl,carry = case
    path = tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    if fmt == "pickle":
        payload=pickle.loads(path.read_bytes())
        c=payload["carry"]
        if damage == "marker":
            c.pop("precipitation_buckets_schema_version")
        elif damage == "group":
            c["extra_fields"].pop("precipitation_buckets")
            c["carry_field_order"].remove("precipitation_buckets")
        else:
            c["extra_fields"]["precipitation_buckets"] = c["extra_fields"]["precipitation_buckets"]._replace(prec_acc_c=None)
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path,"a") as ds:
            if damage == "marker":
                ds.delncattr("GPUWRF_PRECIPITATION_BUCKETS_SCHEMA_VERSION")
            elif damage == "group":
                for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER","GPUWRF_OPTIONAL_CARRY_KIND"):
                    value=json.loads(ds.getncattr(attr))
                    value.pop("precipitation_buckets")
                    ds.setncattr(attr,json.dumps(value))
            else:
                ds.renameVariable("GPUWRF_PRECIPITATION_BUCKETS_LEAF_000","REMOVED_LEAF")
    with pytest.raises(ValueError,match="precipitation buckets.*E78|missing exact optional carry variables"):
        read(path,fmt)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_old_inactive_restart_buckets_remain_readable(case,tmp_path,fmt):
    grid,nl,carry=case
    nl=replace(nl,prec_acc_dt=0.)
    carry=carry.replace(precipitation_buckets=None)
    path=tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    if fmt == "pickle":
        p=pickle.loads(path.read_bytes())
        p["carry"].pop("precipitation_buckets_schema_version")
        p["carry"]["extra_fields"].pop("precipitation_buckets")
        p["carry"]["carry_field_order"].remove("precipitation_buckets")
        path.write_bytes(pickle.dumps(p))
    else:
        with Dataset(path,"a") as ds:
            ds.delncattr("GPUWRF_PRECIPITATION_BUCKETS_SCHEMA_VERSION")
            ds.delncattr("GPUWRF_PRECIPITATION_BUCKETS_CONFIG")
            for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER","GPUWRF_OPTIONAL_CARRY_KIND"):
                value=json.loads(ds.getncattr(attr)); value.pop("precipitation_buckets")
                ds.setncattr(attr,json.dumps(value))
    assert read(path,fmt).precipitation_buckets is None
    if fmt == "netcdf":
        with pytest.raises(ValueError,match="precipitation buckets.*E78"):
            read(path,fmt,replace(nl,prec_acc_dt=60.))


def test_binding_and_static_tree(case):
    _,nl,_=case
    bound=bind_namelist_options(replace(nl,prec_acc_dt=0.), {"physics":{"prec_acc_dt":[15.,30.]}}, "d02")
    assert bound.prec_acc_dt == 30.
    leaves,tree=jax.tree.flatten(bound)
    assert jax.tree.unflatten(tree,leaves).prec_acc_dt == 30.
    wrfnl={"physics":{"mp_physics":8,"cu_physics":1,"prec_acc_dt":30.}}
    errors,warnings=namelist_honour_report(wrfnl,("d01",))
    assert not errors and not any("prec_acc_dt" in x for x in warnings)
    assert any("replay clock" in e for e in namelist_honour_report(wrfnl,("d01",),replay=True)[0])
    for value in (float("nan"),float("inf"),-1.):
        errors,_=namelist_honour_report({"physics":{"prec_acc_dt":value}},("d01",))
        assert any("finite and nonnegative" in e for e in errors)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_wrong_surface_shape_cannot_restart(case,tmp_path,fmt):
    grid,nl,carry=case
    wrong=PrecipitationBuckets(*(a[:,:-1] for a in carry.precipitation_buckets))
    with pytest.raises(ValueError,match="surface grid.*E78"):
        save(tmp_path/fmt,fmt,grid,nl,carry.replace(precipitation_buckets=wrong))


def test_writer_schema_against_original_cpu_history():
    from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS
    manifest=json.loads((FIXTURE.parent/"cpu_reference_manifest.json").read_text())
    names={"PREC_ACC_C","PREC_ACC_NC","SNOW_ACC_NC"}
    for frame in manifest["frames"]:
        for name in names:
            expected=frame["fields"][name]
            assert expected["dtype"] == "float32" and expected["shape"] == [1,42,42]
            assert WRFOUT_VARIABLE_SPECS[name].units == expected["units"] == "mm"
    assert set(history_fields(PrecipitationBuckets(None,None,None))) == names


@pytest.mark.parametrize("config,key",[
    ({"time_control":{"nwp_diagnostics":2}},"nwp_diagnostics"),
    ({"afwa":{"afwa_diag_opt":1}},"afwa_diag_opt"),
    ({"afwa":{"afwa_severe_opt":1}},"afwa_severe_opt"),
    ({"afwa":{"afwa_radar_opt":1}},"afwa_radar_opt"),
    ({"physics":{"do_radar_ref":2}},"do_radar_ref"),
    ({"physics":{"swint_opt":2}},"swint_opt"),
])
def test_unqualified_online_diagnostics_refused(config,key):
    errors,_=namelist_honour_report(config,("d01",))
    assert any(key in e for e in errors)


@pytest.mark.parametrize("own_step",[0,200])
def test_actual_nested_history_callback_transcribes_buckets(tmp_path,own_step):
    from types import SimpleNamespace
    from test_initial_history_wrf_t0 import _writer
    writer,state=_writer(tmp_path,lambda *a,**kw: {})
    shape=np.shape(state.t_skin)
    scale=1 if own_step else 0
    buckets=PrecipitationBuckets(*(np.full(shape,scale*(i+.125),np.float32) for i in range(3)))
    output=writer("d01",own_step,SimpleNamespace(state=state,precipitation_buckets=buckets))
    with Dataset(output["wrfout"]) as ds:
        for name,value in zip(buckets._fields,buckets,strict=True):
            np.testing.assert_array_equal(ds[name.upper()][0],value)


@pytest.mark.parametrize("full",[False,True])
@pytest.mark.parametrize("subset",[None,("PREC_ACC_C","PREC_ACC_NC","SNOW_ACC_NC")])
def test_written_history_fields(case,tmp_path,full,subset):
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    metadata=json.loads((Path(__file__).resolve().parents[1]/"proofs/u2_diag/history_metadata.json").read_text())["fields"]
    grid,nl,carry=case
    path=tmp_path/"wrfout"
    write_wrfout_netcdf(carry.state,grid,nl,path,domain="d02",domain_authority=writer_authority(grid),lead_hours=.5,
                       diagnostics=history_fields(carry.precipitation_buckets),
                       valid_time="2026-06-03_00:30:00",run_start="2026-06-03_00:00:00",
                       full_variable_set=full,variable_subset=subset)
    with Dataset(path) as ds:
        for name in carry.precipitation_buckets._fields:
            var=ds.variables[name.upper()]
            assert var.units == "mm" and var.dimensions == ("Time","south_north","west_east")
            assert {key:var.getncattr(key) for key in metadata[name.upper()]}==metadata[name.upper()]
            np.testing.assert_array_equal(var[0],getattr(carry.precipitation_buckets,name))


@pytest.mark.parametrize("name",PrecipitationBuckets._fields)
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_nonfinite_bucket_history_rejected(case,tmp_path,name,bad):
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    grid,nl,carry=case
    fields=history_fields(carry.precipitation_buckets)
    fields[name.upper()]=np.full(carry.state.t_skin.shape,bad,np.float32)
    with pytest.raises(ValueError,match="nonfinite precipitation history"):
        write_wrfout_netcdf(carry.state,grid,nl,tmp_path/"bad",domain="d02",domain_authority=writer_authority(grid),
                           lead_hours=.5,diagnostics=fields,
                           valid_time="2026-06-03_00:30:00",run_start="2026-06-03_00:00:00")
