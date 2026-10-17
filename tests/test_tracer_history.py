"""Tracer output metadata and stream selection, independently read from NetCDF."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf, bind_wrfout_domain_authority
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.tracers.config import UserTracerConfig, TracerSpecies
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
from test_m7_netcdf_writer import authenticated_source_grid


@pytest.mark.parametrize("full",[False,True])
@pytest.mark.parametrize("subset",[None,("tr17_1","tr17_5","SMOKE")])
def test_tracer_history_and_auxhist_exact(tmp_path,full,subset):
    grid=make_ideal_grid(5,4,4)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,tracer_opt=2,
        tracer_config=UserTracerConfig((TracerSpecies("SMOKE"),),(),"a"*64))
    q=np.arange(9*5*4*4,dtype=np.float32).reshape(9,5,4,4)/np.float32(1024)
    state=_state(grid).replace(tracers=q,tracer_budget=np.zeros((9,4),np.float32))
    authority=bind_wrfout_domain_authority("d01",authenticated_source_grid(grid,"d01"),grid)
    path=tmp_path/"history"
    write_wrfout_netcdf(state,grid,nl,path,domain="d01",domain_authority=authority,
        valid_time="2026-06-03_00:00:18",run_start="2026-06-03_00:00:00",lead_hours=.005,
        full_variable_set=full,variable_subset=subset)
    with Dataset(path) as ds:
        expected=tuple(f"tr17_{i}" for i in range(1,9))+("SMOKE",)
        for i,name in enumerate(expected):
            if subset is not None and name not in subset:
                assert name not in ds.variables
                continue
            var=ds.variables[name]
            np.testing.assert_array_equal(var[0],q[i])
            assert var.dimensions==("Time","bottom_top","south_north","west_east")
            assert var.units==("Dimensionless" if name.startswith("tr17_") else "kg kg-1")
            assert var.description==(name if name.startswith("tr17_") else "Passive tracer SMOKE")
            assert var.MemoryOrder=="XYZ"
            assert var.stagger==""


def test_custom_tracer_cannot_replace_existing_diagnostic():
    from gpuwrf.tracers.history import register_specs
    grid=make_ideal_grid(5,4,4)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,
        tracer_config=UserTracerConfig((TracerSpecies("UP_HELI_MAX"),),(),"a"*64))
    with pytest.raises(ValueError,match="collides"):
        register_specs(nl)


def _device_history_case():
    grid=make_ideal_grid(5,4,4)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,tracer_opt=2,
        tracer_config=UserTracerConfig((TracerSpecies("SMOKE"),),(),"a"*64))
    # Distinct from every meteorological input, so actual host-batch contents
    # identify transferred tracer volumes without converting extra device data.
    q=np.arange(9*5*4*4,dtype=np.float32).reshape(9,5,4,4)+np.float32(1e6)
    state=_state(grid).replace(tracers=jnp.asarray(q),tracer_budget=jnp.zeros((9,4),jnp.float32))
    authority=bind_wrfout_domain_authority("d01",authenticated_source_grid(grid,"d01"),grid)
    return grid,nl,state,q,authority


def _observe_host_batch(monkeypatch):
    real=jax.device_get
    batches=[]
    def observe(tree):
        host=real(tree)
        batches.append((tuple(jax.tree.leaves(tree)),tuple(jax.tree.leaves(host))))
        return host
    monkeypatch.setattr(jax,"device_get",observe)
    return batches


def _prepare_device_history(tmp_path,grid,nl,state,authority,subset):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    return prepare_wrfout_payload(state,grid,nl,tmp_path/"selected",domain="d01",domain_authority=authority,
        valid_time="2026-06-03_00:00:18",run_start="2026-06-03_00:00:00",lead_hours=.005,
        diagnostics={"T2":np.full((4,4),280,np.float32)},variable_subset=subset)


@pytest.mark.parametrize("subset,indices",[(('T2',),()),(('tr17_1','SMOKE'),(0,8))])
def test_actual_host_batch_copies_only_requested_tracer_volumes(tmp_path,monkeypatch,subset,indices):
    grid,nl,state,q,authority=_device_history_case()
    batches=_observe_host_batch(monkeypatch)
    prepared=_prepare_device_history(tmp_path,grid,nl,state,authority,subset)
    assert len(batches)==1,"all requested output and tracer health must share one host batch"
    device,host=batches[0]
    assert all(value.shape!=q.shape for value in device),"unrequested tracer stack copied to host"
    transferred=[i for value in host if np.shape(value)==q.shape[1:]
                 for i in range(len(q)) if np.array_equal(value,q[i])]
    assert transferred==list(indices),"only selected species may be transferred, once each"
    assert sum(q[i].nbytes for i in transferred)==len(indices)*5*4*4*4
    assert any(np.shape(value)==() and np.asarray(value).dtype==np.bool_ and bool(value) for value in host),\
        "whole-stack finite health must transfer as a scalar in the same batch"
    for name,i in zip((name for name in subset if name!='T2'),indices,strict=True):
        np.testing.assert_array_equal(prepared.fields[name],q[i])
        # Context lookup must reuse the exact batched selected view; rebuilding
        # its slice would implicitly copy it again outside the host batch.
        assert any(prepared.fields[name] is value for value in host)


@pytest.mark.parametrize("subset",[('T2',),('tr17_1',)])
@pytest.mark.parametrize("bad",[np.nan,np.inf])
def test_unrequested_nonfinite_tracer_refuses_output_without_volume_copy(tmp_path,monkeypatch,subset,bad):
    grid,nl,state,q,authority=_device_history_case()
    state=state.replace(tracers=state.tracers.at[8,0,0,0].set(bad))
    batches=_observe_host_batch(monkeypatch)
    with pytest.raises(ValueError,match="tracer history.*nonfinite"):
        _prepare_device_history(tmp_path,grid,nl,state,authority,subset)
    assert len(batches)==1
    device,host=batches[0]
    assert all(value.shape!=q.shape for value in device)
    assert any(np.shape(value)==() and np.asarray(value).dtype==np.bool_ and not bool(value) for value in host)
    assert all(not np.any(~np.isfinite(value)) for value in host if np.asarray(value).dtype.kind in 'fc'),\
        "the bad unrequested concentration must stay on device"


def test_invalid_tracer_shape_refused_before_any_host_batch(tmp_path,monkeypatch):
    grid,nl,state,_,authority=_device_history_case()
    state=state.replace(tracers=state.tracers[:8])
    batches=_observe_host_batch(monkeypatch)
    with pytest.raises(ValueError,match="tracer history shape"):
        _prepare_device_history(tmp_path,grid,nl,state,authority,('T2',))
    assert batches==[],"shape metadata must be checked before output transfers"
