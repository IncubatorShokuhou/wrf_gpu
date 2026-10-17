"""Composed opt-in NWP/radar public binding, restart and history seams.

History schema is checked against pristine CPU-WRF's retained manifest;
operator numerics are separately gated by test_online_nwp and radar tests.
"""
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.diagnostics.nwp import NwpDiagnostics,history_fields
from gpuwrf.diagnostics.online import history_due,validate_options
from gpuwrf.diagnostics.precipitation import PrecipitationBuckets
from gpuwrf.io import restart as rp,wrfrst_netcdf as nc
from gpuwrf.io.namelist_binding import bind_namelist_options,namelist_honour_report
from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS,write_wrfout_netcdf
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_initial_history_wrf_t0 import _writer
from test_m7_netcdf_writer import writer_authority
from test_v0110_wrfrst_netcdf import _state

NAMES=("WSPD10MAX","W_UP_MAX","W_DN_MAX","UP_HELI_MAX","W_MEAN","GRPL_MAX",
       "REFD_MAX","HAIL_MAXK1","HAIL_MAX2D","REFL_10CM")
GROUPS=("nwp_diagnostics","radar_reflectivity")
MANIFEST=Path(__file__).resolve().parents[1]/"proofs/u2_diag/cpu_reference_manifest.json"


def populated(state):
    shape=state.t_skin.shape
    surface=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/np.float32(32)
    nwp=NwpDiagnostics(*(jnp.asarray(surface+np.float32(i+.125)) for i in range(10)))
    radar=jnp.asarray(np.arange(np.prod(state.qv.shape),dtype=np.float32).reshape(state.qv.shape)/np.float32(8)-np.float32(35))
    return nwp,radar


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD","0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re,"_MP_RE",False)
    grid=make_ideal_grid(5,4,5)
    state=_state(grid)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,mp_physics=8,
                           nwp_diagnostics=1,do_radar_ref=1,diagnostic_history_interval=30.,prec_acc_dt=15.)
    nwp,radar=populated(state)
    buckets=PrecipitationBuckets(*(jnp.full(state.t_skin.shape,i+.25,jnp.float32) for i in range(3)))
    carry=initial_operational_carry(state).replace(nwp_diagnostics=nwp,radar_reflectivity=radar,
                                                 precipitation_buckets=buckets)
    return grid,nl,carry


def save(path,fmt,grid,nl,carry):
    if fmt=="pickle":
        rp.write_restart(carry,nl,grid,200,path)
    else:
        nc.write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:30:00",
                             run_start="2026-06-03_00:00:00",step_index=200)


def read(path,fmt,nl=None):
    return (rp.read_restart(path) if fmt=="pickle" else nc.read_wrfrst_carry(path,namelist=nl))[0]


def test_composed_binding_static_tree_and_honour(case):
    _,nl,_=case
    original=replace(nl,nwp_diagnostics=0,do_radar_ref=0,prec_acc_dt=0.,diagnostic_history_interval=60.)
    # Registry do_radar_ref has dimension1 (one value shared by all domains).
    wrf={"physics":{"mp_physics":8,"cu_physics":1,"do_radar_ref":1,"prec_acc_dt":[15.,30.]},
         "time_control":{"nwp_diagnostics":1,"history_interval":[60,30]}}
    bound=bind_namelist_options(original,wrf,"d02")
    assert (bound.nwp_diagnostics,bound.do_radar_ref,bound.prec_acc_dt,bound.diagnostic_history_interval)==(1,1,30.,30.)
    leaves,tree=jax.tree.flatten(bound)
    restored=jax.tree.unflatten(tree,leaves)
    assert (restored.nwp_diagnostics,restored.do_radar_ref,restored.prec_acc_dt,restored.diagnostic_history_interval)==(1,1,30.,30.)
    assert jax.tree.structure(bound)!=jax.tree.structure(original)
    # Inactive diagnostic cadence must not alter the released static program.
    assert jax.tree.structure(replace(original,diagnostic_history_interval=17.))==jax.tree.structure(original)
    # Compose optional static suffixes introduced on main with both diagnostic
    # suffixes; unflatten must consume them in reverse append order.
    obs_marker=object()  # structural child; no observation/forecast claim
    combined=replace(bound,ccn_conc=2.e8,lakedepth_default=77.,observation_nudging=obs_marker)
    combined_leaves,combined_tree=jax.tree.flatten(combined)
    combined_restored=jax.tree.unflatten(combined_tree,combined_leaves)
    assert (combined_restored.ccn_conc,combined_restored.lakedepth_default,
            combined_restored.prec_acc_dt,combined_restored.nwp_diagnostics,
            combined_restored.do_radar_ref)==(2.e8,77.,30.,1,1)
    assert combined_restored.observation_nudging is obs_marker
    errors,warnings=namelist_honour_report(wrf,("d01","d02"))
    assert not errors
    assert not any(any(key in warning for key in ("nwp_diagnostics","do_radar_ref","prec_acc_dt")) for warning in warnings)
    assert any("native mode" in error for error in namelist_honour_report(wrf,("d02",),replay=True)[0])
    validate_options(bound,specified=True,nested=False,periodic=False)


@pytest.mark.parametrize("radar",[None,0,1])
def test_nwp_binding_implies_radar_like_wrf_check_a_mundo(case,radar):
    _,nl,_=case
    wrf={"time_control":{"nwp_diagnostics":1},"physics":{"mp_physics":8}}
    if radar is not None:
        wrf["physics"]["do_radar_ref"]=radar
    bound=bind_namelist_options(replace(nl,nwp_diagnostics=0,do_radar_ref=0),wrf,"d01")
    assert (bound.nwp_diagnostics,bound.do_radar_ref)==(1,1)


@pytest.mark.parametrize("patch,flags",[
    ({"mp_physics":10},{}),({"run_physics":False},{}),({"diagnostic_history_interval":np.nan},{}),
    ({"diagnostic_history_interval":np.inf},{}),({"diagnostic_history_interval":0.},{}),
    ({"nwp_diagnostics":2},{}),({"do_radar_ref":2},{}),
    ({"do_radar_ref":0},{}),
    ({},{"periodic":True}),({},{"specified":False,"nested":False}),
])
def test_unsupported_composed_options_refused(case,patch,flags):
    _,nl,_=case
    with pytest.raises(ValueError,match="diagnostics|nwp_diagnostics|do_radar_ref|NWP helicity"):
        validate_options(replace(nl,**patch),**{**dict(specified=True,nested=False,periodic=False),**flags})


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
def test_nwp_restart_requires_wrf_radar_selection(case,tmp_path,fmt):
    grid,nl,carry=case
    # check_a_mundo forces radar for NWP; a manually constructed carry or an
    # old inconsistent selection must not bypass that invariant through IO.
    with pytest.raises(ValueError,match="NWP.*do_radar_ref=1.*E78"):
        save(tmp_path/fmt,fmt,grid,replace(nl,do_radar_ref=0),
             carry.replace(radar_reflectivity=None))


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
def test_composed_restart_preserves_exact_leaves_and_selection(case,tmp_path,fmt):
    grid,nl,carry=case
    path=tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    restored=read(path,fmt,nl)
    for group in (*GROUPS,"precipitation_buckets"):
        expected=getattr(carry,group)
        actual=getattr(restored,group)
        assert jax.tree.structure(actual)==jax.tree.structure(expected)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
            assert np.asarray(a).dtype==np.float32
            assert np.asarray(a).shape==np.asarray(b).shape
            assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    if fmt=="pickle":
        saved_nl=rp.read_restart(path)[1]
        assert (saved_nl.nwp_diagnostics,saved_nl.do_radar_ref,saved_nl.diagnostic_history_interval)==(1,1,30.)
    else:
        with Dataset(path) as ds:
            assert ds.GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION==1
            assert json.loads(ds.GPUWRF_ONLINE_DIAGNOSTICS_CONFIG)=={
                "nwp_diagnostics":1,"do_radar_ref":1,"history_interval":30.}


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
@pytest.mark.parametrize("group",GROUPS)
@pytest.mark.parametrize("damage",["marker","group","leaf","precision","shape"])
def test_active_restart_damage_refused(case,tmp_path,fmt,group,damage):
    grid,nl,carry=case
    path=tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    if fmt=="pickle":
        payload=pickle.loads(path.read_bytes())
        schema=payload["carry"]
        if damage=="marker":
            schema.pop("online_diagnostics_schema_version")
        elif damage=="group":
            schema["extra_fields"].pop(group)
            schema["carry_field_order"].remove(group)
        else:
            value=schema["extra_fields"][group]
            leaf=value.w_mean if group=="nwp_diagnostics" else value
            changed=(None if damage=="leaf" else np.asarray(leaf,np.float64) if damage=="precision" else np.asarray(leaf)[...,:-1])
            schema["extra_fields"][group]=value._replace(w_mean=changed) if group=="nwp_diagnostics" else changed
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path,"a") as ds:
            if damage=="marker":
                ds.delncattr("GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION")
            elif damage=="group":
                for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER","GPUWRF_OPTIONAL_CARRY_KIND"):
                    values=json.loads(ds.getncattr(attr));values.pop(group)
                    ds.setncattr(attr,json.dumps(values))
            else:
                name="GPUWRF_"+group.upper()+"_LEAF_000"
                var=ds[name]
                values=np.asarray(var[:]);dims=var.dimensions
                ds.renameVariable(name,"REMOVED_DIAGNOSTIC")
                if damage=="precision":
                    ds.createVariable(name,"f8",dims)[:]=values
                elif damage=="shape":
                    ds.createDimension("damaged_width",values.shape[-1]-1)
                    ds.createVariable(name,"f4",(*dims[:-1],"damaged_width"))[:]=values[...,:-1]
    with pytest.raises(ValueError,match="E78|missing exact optional carry variables"):
        read(path,fmt,nl)


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
@pytest.mark.parametrize("group",GROUPS)
@pytest.mark.parametrize("damage",["precision","shape"])
def test_wrong_diagnostic_leaf_refused_before_write(case,tmp_path,fmt,group,damage):
    grid,nl,carry=case
    value=getattr(carry,group)
    leaf=value.uh if group=="nwp_diagnostics" else value
    damaged=leaf.astype(jnp.float64) if damage=="precision" else leaf[...,:-1]
    value=value._replace(uh=damaged) if group=="nwp_diagnostics" else damaged
    with pytest.raises(ValueError,match="E78"):
        save(tmp_path/fmt,fmt,grid,nl,carry.replace(**{group:value}))


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
def test_inactive_legacy_diagnostics_readable(case,tmp_path,fmt):
    grid,nl,carry=case
    nl=replace(nl,nwp_diagnostics=0,do_radar_ref=0)
    carry=carry.replace(nwp_diagnostics=None,radar_reflectivity=None)
    path=tmp_path/fmt
    save(path,fmt,grid,nl,carry)
    if fmt=="pickle":
        payload=pickle.loads(path.read_bytes());schema=payload["carry"]
        schema.pop("online_diagnostics_schema_version")
        for group in GROUPS:
            schema["extra_fields"].pop(group);schema["carry_field_order"].remove(group)
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path,"a") as ds:
            for attr in ("GPUWRF_ONLINE_DIAGNOSTICS_CONFIG","GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION"):
                ds.delncattr(attr)
            for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER","GPUWRF_OPTIONAL_CARRY_KIND"):
                values=json.loads(ds.getncattr(attr))
                for group in GROUPS: values.pop(group)
                ds.setncattr(attr,json.dumps(values))
    restored=read(path,fmt,nl)
    assert restored.nwp_diagnostics is None and restored.radar_reflectivity is None
    if fmt=="netcdf":
        with pytest.raises(ValueError,match="E78"):
            read(path,fmt,replace(nl,nwp_diagnostics=1,do_radar_ref=1))


def test_requested_restart_cadence_mismatch_refused(case,tmp_path):
    grid,nl,carry=case
    path=tmp_path/"wrfrst"
    save(path,"netcdf",grid,nl,carry)
    with pytest.raises(ValueError,match="cadence/selection differs.*E78"):
        read(path,"netcdf",replace(nl,diagnostic_history_interval=15.))


@pytest.mark.parametrize("full",[False,True])
@pytest.mark.parametrize("subset",[None,NAMES])
def test_public_history_names_values_units_match_original_cpu_schema(case,tmp_path,full,subset):
    grid,nl,carry=case
    fields={**history_fields(carry.nwp_diagnostics),"REFL_10CM":carry.radar_reflectivity}
    assert tuple(fields)==NAMES and "UH" not in fields
    manifest=json.loads(MANIFEST.read_text())
    metadata=json.loads(MANIFEST.with_name("history_metadata.json").read_text())["fields"]
    path=tmp_path/"wrfout"
    write_wrfout_netcdf(carry.state,grid,nl,path,domain="d02",domain_authority=writer_authority(grid),
        lead_hours=.5,diagnostics=fields,valid_time="2026-06-03_00:30:00",run_start="2026-06-03_00:00:00",
        full_variable_set=full,variable_subset=subset)
    with Dataset(path) as ds:
        for name in NAMES:
            actual=ds[name]
            expected=manifest["frames"][-1]["fields"][name]
            assert actual.units==expected["units"]==WRFOUT_VARIABLE_SPECS[name].units
            assert {key:actual.getncattr(key) for key in metadata[name]}==metadata[name]
            assert np.dtype(actual.dtype)==np.dtype(expected["dtype"])==np.dtype("float32")
            assert actual.dimensions==(("Time","bottom_top","south_north","west_east") if name=="REFL_10CM" else ("Time","south_north","west_east"))
            np.testing.assert_array_equal(actual[0],fields[name])


@pytest.mark.parametrize("own_step",[0,200])
def test_actual_nested_callback_transcribes_online_carry(tmp_path,own_step):
    writer,state=_writer(tmp_path,lambda *a,**kw:{})
    nwp,radar=populated(state)
    result=writer("d01",own_step,SimpleNamespace(state=state,nwp_diagnostics=nwp,radar_reflectivity=radar))
    with Dataset(result["wrfout"]) as ds:
        for name,value in {**history_fields(nwp),"REFL_10CM":radar}.items():
            np.testing.assert_array_equal(ds[name][0],value)


@pytest.mark.parametrize("name",["WSPD10MAX","HAIL_MAX2D","REFL_10CM"])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("subset",[None,("T2",)])
def test_nested_callback_rejects_nonfinite_online_fields_even_unrequested(tmp_path,name,bad,subset):
    writer,state=_writer(tmp_path,lambda *a,**kw:{})
    writer._variable_subset=subset
    nwp,radar=populated(state)
    if name=="REFL_10CM":
        radar=np.full(radar.shape,bad,np.float32)
    else:
        nwp=nwp._replace(**{name.lower():np.full(state.t_skin.shape,bad,np.float32)})
    with pytest.raises(ValueError,match="nonfinite.*online diagnostic|nonfinite source"):
        writer("d01",200,SimpleNamespace(state=state,nwp_diagnostics=nwp,radar_reflectivity=radar))
    assert not list(writer.output_dir.glob("wrfout*"))


@pytest.mark.parametrize("dt,steps,expected",[
    (18.,[0,199,200,201,399,400],[True,False,True,False,False,True]),
    (54.,[65,66,67,68,133,134],[False,False,True,False,False,True]),
])
def test_history_alarm_crossing_matches_wrf_boundary(dt,steps,expected):
    # The 54s alarm rings at 3618s: truncating3600/54 to66 must not ring.
    np.testing.assert_array_equal(history_due(jnp.asarray(steps,jnp.int32),dt,60.),expected)
