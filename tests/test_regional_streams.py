"""Independent arithmetic, original WRF auxhist, device and restart gates."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.streams.config import FILENAME, read_streams, region_points, FIELDS
from gpuwrf.streams.device import seed, sample, record_buffer, record_step, validate_schema
from gpuwrf.streams.runtime import prepare_streams, require_supported_run


def spec_file(tmp_path, **updates):
    data = dict(name="valley", domain="d01", interval_steps=2, buffer_frames=3,
                region={"stations": [{"id": "a", "i": 1, "j": 1}, {"id": "b", "i": 3, "j": 2}]},
                variables={"T2": ["instant", "min", "max", "mean"], "RAINNC": ["instant", "accumulation"]})
    data.update(updates)
    path = tmp_path / FILENAME
    path.write_text(json.dumps({"version": 1, "streams": [data]}))
    return read_streams(path)[0]


def carry_case():
    from gpuwrf.validation.tier2 import make_ideal_grid
    from test_v0110_wrfrst_netcdf import _state
    from gpuwrf.runtime.operational_state import initial_operational_carry, HistoryDiagnostics
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    grid = make_ideal_grid(4, 3, 5)
    state = _state(grid)
    a = jnp.ones(state.t_skin.shape, jnp.float32)
    carry = initial_operational_carry(state).replace(history_diagnostics=HistoryDiagnostics(*(a*i for i in range(1,7))))
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18.)
    return grid, nl, carry


def exact(a, b):
    aa, at = jax.tree.flatten(a)
    bb, bt = jax.tree.flatten(b)
    assert at == bt
    for x,y in zip(aa,bb,strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_independent_window_arithmetic_and_resume_midwindow(tmp_path):
    spec = spec_file(tmp_path)
    _, _, carry = carry_case()
    buffer = seed(carry,spec,[0,1],[0,2],2,3)
    # Non-monotonic temperature + NONZERO initial cumulative rainfall, then a
    # negative endpoint difference: no sum-of-cumulative or masking happy path.
    baseline = np.asarray(buffer.baseline).copy()
    baseline[-2:] = [[10.,20.],[10.,20.]]
    buffer = replace(buffer,baseline=jnp.asarray(baseline))
    temperature = np.asarray([[8,4],[2,7],[6,1],[3,9],[5,2],[0,4]],np.float32)
    rain = np.asarray([[11,21],[13,25],[17,26],[18,28],[18,28],[16,27]],np.float32)
    operands = np.stack([temperature]*4+[rain]*2,axis=1)
    step = jax.jit(record_buffer)
    partial = step(buffer,jnp.asarray(operands[0]),jnp.int32(1))
    resumed = jax.tree.map(jnp.asarray, pickle.loads(pickle.dumps(jax.device_get(partial))))
    for k in range(1,6):
        partial = step(partial,jnp.asarray(operands[k]),jnp.int32(k+1))
        resumed = step(resumed,jnp.asarray(operands[k]),jnp.int32(k+1))
    exact(partial,resumed)
    actual = np.asarray(resumed.frames)
    for n, window in enumerate(temperature.reshape(3,2,2)):
        expected = np.stack([window[-1],np.min(window,axis=0),np.max(window,axis=0),np.mean(window,axis=0),
                             rain[2*n+1],rain[2*n+1] - (np.array([10,20]) if n == 0 else rain[2*n-1])])
        np.testing.assert_array_equal(actual[n],expected)
    np.testing.assert_array_equal(resumed.end_steps,[2,4,6])
    validate_schema((resumed,),1)


def test_regions_bbox_polygon_stations_and_boundaries(tmp_path):
    lat,lon = np.meshgrid([0.,1.,2.],[10.,11.,12.,13.],indexing="ij")
    spec = spec_file(tmp_path,region={"bbox":[11,0,12,1]})
    y,x,ids = region_points(spec,lat,lon)
    np.testing.assert_array_equal(np.column_stack((y,x)), [[0,1],[0,2],[1,1],[1,2]])
    spec = spec_file(tmp_path,region={"polygon":[[11,0],[12,0],[12,1],[11,1]]})
    yy,xx,_ = region_points(spec,lat,lon)
    np.testing.assert_array_equal(yy,y); np.testing.assert_array_equal(xx,x)
    spec = spec_file(tmp_path,region={"stations":[{"id":"ll","lat":.9,"lon":12.1},{"id":"ij","i":1,"j":3}]})
    y,x,ids = region_points(spec,lat,lon)
    assert ids == ("ll","ij")
    np.testing.assert_array_equal(np.column_stack((y,x)),[[1,2],[2,0]])


@pytest.mark.parametrize("update", [dict(name="../bad"),dict(domain="d99x"),dict(interval_steps=0),
    dict(interval_steps=True),dict(variables=["VISIBILITY"]),dict(variables={"T2":["accumulation"]}),
    dict(variables={"T2":["median"]}),dict(buffer_frames=0),dict(region={"bbox":[],"polygon":[]}),
    dict(bad_key=1)])
def test_configuration_fail_closed(tmp_path,update):
    with pytest.raises(ValueError):
        spec_file(tmp_path,**update)


BAD_PREFLIGHT_REGIONS = [
    pytest.param({"bbox": []}, "bbox is", id="empty-bbox"),
    pytest.param({"bbox": [20, 0, 10, 1]}, "bbox must be ordered", id="reversed-bbox"),
    pytest.param({"polygon": [[10, 0], [11, 1]]}, "at least three", id="two-vertex-polygon"),
    pytest.param({"stations": [{"id": "a", "i": 0, "j": 1}]}, "station i", id="zero-station-index"),
]


@pytest.mark.parametrize("region,error", BAD_PREFLIGHT_REGIONS + [
    ({"bbox": [10, 2, 11, 1]}, "bbox must be ordered"),
    ({"bbox": [10, 0, 181, 1]}, "bbox must be ordered"),
    ({"bbox": [10, -91, 11, 1]}, "bbox must be ordered"),
    ({"bbox": [10, 0, True, 1]}, "finite numeric"),
    ({"polygon": [[10, 0], [11], [12, 1]]}, "vertices must be"),
    ({"polygon": [[10, 0], [11, 91], [12, 1]]}, "out of bounds"),
    ({"polygon": [[-100, 0], [100, 0], [100, 1]]}, "dateline wrapping"),
    ({"polygon": [[10, 0], [11, "1"], [12, 1]]}, "finite numeric"),
    ({"stations": []}, "nonempty list"),
    ({"stations": [{"id": "../bad", "i": 1, "j": 1}]}, "station id"),
    ({"stations": [{"id": "a", "i": 1, "j": 1}] * 2}, "station id"),
    ({"stations": [{"id": "a", "i": 1, "j": False}]}, "station j"),
    ({"stations": [{"id": "a", "i": 1, "lat": 0}]}, "either i/j"),
    ({"stations": [{"id": "a", "lat": 0, "lon": "10"}]}, "finite numeric"),
    ({"stations": [{"id": "a", "i": 1, "j": 1, "extra": 0}]}, "expected object"),
])
def test_region_payload_rejected_by_parser(tmp_path,region,error):
    with pytest.raises(ValueError,match=error):
        spec_file(tmp_path,region=region)


@pytest.mark.parametrize("region", [
    {"bbox": [20, 0, 21, 1]},
    {"polygon": [[20, 0], [21, 0], [20, 1]]},
    {"stations": [{"id": "a", "i": 10, "j": 1}]},
    {"stations": [{"id": "a", "lat": 0, "lon": 20}]},
])
def test_region_membership_checked_with_grid(tmp_path,region):
    spec = spec_file(tmp_path,region=region)  # Structurally valid; grid not loaded yet.
    lat,lon = np.meshgrid([0.,1.,2.],[10.,11.,12.,13.],indexing="ij")
    with pytest.raises(ValueError,match="no mass-grid centers|outside domain"):
        region_points(spec,lat,lon)


def test_bbox_inclusive_single_center_remains_valid(tmp_path):
    spec = spec_file(tmp_path,region={"bbox": [11, 1, 11, 1]})
    lat,lon = np.meshgrid([0.,1.,2.],[10.,11.,12.,13.],indexing="ij")
    y,x,ids = region_points(spec,lat,lon)
    assert y.tolist() == [1] and x.tolist() == [1] and ids == ("j2_i2",)


def test_seconds_alignment_and_empty_default(tmp_path):
    spec = spec_file(tmp_path)
    path = tmp_path / FILENAME
    data = json.loads(path.read_text())
    s = data["streams"][0]
    del s["interval_steps"]
    s["interval_seconds"] = 36.
    path.write_text(json.dumps(data))
    spec = read_streams(path)[0]
    assert spec.steps(18.) == 2
    with pytest.raises(ValueError,match="align"):
        spec.steps(50.)
    path.unlink()
    original = {"d01": object()}
    assert prepare_streams(NS(input_dir=tmp_path),(),original,{},None) == (None,original)
    assert prepare_streams(NS(input_dir=tmp_path),(),original,{},None)[1] is original


@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_nonfinite_interior_instant_sample_rejected(tmp_path,bad):
    spec = spec_file(tmp_path,variables=["RAINNC"])
    _,_,carry = carry_case()
    b = seed(carry,spec,[0],[0],2,3)
    b = record_buffer(b,jnp.asarray([[bad]],jnp.float32),jnp.int32(1))
    b = record_buffer(b,jnp.asarray([[3]],jnp.float32),jnp.int32(2))
    assert not np.isfinite(np.asarray(b.frames[0])).all()
    from gpuwrf.streams.writer import write_frames
    with pytest.raises(ValueError,match="NaN/Inf"):
        write_frames(tmp_path,spec,([0],[0]),[0.],[0.],("a",),18.,datetime(2023,1,15),np.array([2]),np.asarray(b.frames[:1]))
    assert not list(tmp_path.glob("*.nc"))


def test_post_solve_observer_hook_and_default_no_work(tmp_path,monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    _,nl,carry = carry_case()
    spec = spec_file(tmp_path,variables=["RAINNC"],interval_steps=1)
    carry = carry.replace(regional_streams=(seed(carry,spec,[0],[0],1,3),))
    evolved = carry.replace(state=carry.state.replace(rain_acc=jnp.full_like(carry.state.rain_acc,7.)))
    monkeypatch.setattr(om,"_physics_boundary_step_with_limiter_diagnostics",lambda *a,**kw:(evolved,None))
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32","0")
    monkeypatch.setenv("GPUWRF_DYN_CARRY_FP32","0")
    result = om._physics_boundary_step(carry,nl,1,run_radiation=False)
    assert int(result.regional_streams[0].count) == 1
    np.testing.assert_array_equal(result.regional_streams[0].frames[0],[[7.]])
    result = om._physics_boundary_step(carry.replace(regional_streams=None),nl,1,run_radiation=False)
    assert result is evolved  # No observer replace/call when OFF.


@pytest.mark.parametrize("fmt",["pickle","netcdf"])
@pytest.mark.parametrize("damage",["marker","group"])
def test_real_restart_preserves_partial_window_and_rejects_missing_schema(tmp_path,fmt,damage):
    from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
    grid,nl,carry = carry_case()
    spec = spec_file(tmp_path)
    b = seed(carry,spec,[0,1],[0,2],2,3)
    b = record_buffer(b,jnp.full_like(b.running,3.),jnp.int32(1))
    carry = carry.replace(regional_streams=(b,))
    path = tmp_path / fmt
    if fmt == "pickle":
        rp.write_restart(carry,nl,grid,1,path)
        restored = rp.read_restart(path)[0]
        payload = pickle.loads(path.read_bytes())
        if damage == "marker":
            del payload["carry"]["regional_streams_schema_version"]
        else:
            payload["carry"]["extra_fields"]["regional_streams"] = None
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError,match="regional stream"):
            rp.read_restart(path)
    else:
        nc.write_wrfrst_carry(carry,grid,nl,path,valid_time="2023-01-15_00:00:18",run_start="2023-01-15_00:00:00",step_index=1)
        restored = nc.read_wrfrst_carry(path)[0]
        with Dataset(path,"r+") as ds:
            if damage == "marker":
                ds.delncattr("GPUWRF_REGIONAL_STREAMS_SCHEMA_VERSION")
            else:
                kinds = json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND)
                kinds["regional_streams"] = "none"
                ds.GPUWRF_OPTIONAL_CARRY_KIND = json.dumps(kinds)
                order = json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER)
                order["regional_streams"] = []
                ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER = json.dumps(order)
        with pytest.raises(ValueError,match="regional stream"):
            nc.read_wrfrst_carry(path)
    exact(carry,restored)
    after = record_buffer(restored.regional_streams[0],jnp.full_like(b.running,5.),jnp.int32(2))
    expected = record_buffer(b,jnp.full_like(b.running,5.),jnp.int32(2))
    exact(after,expected)


def write_grid(path,ny,nx):
    with Dataset(path,"w") as ds:
        ds.createDimension("Time",1); ds.createDimension("y",ny); ds.createDimension("x",nx)
        for name,values in [("XLAT",np.broadcast_to(np.arange(ny)[:,None],(ny,nx))),
                             ("XLONG",np.broadcast_to(np.arange(nx)[None,:],(ny,nx)))]:
            ds.createVariable(name,"f4",("Time","y","x"))[0] = values


def test_native_setup_flush_durable_replay_and_foreign_protection(tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    spec = spec_file(tmp_path)
    _,_,carry = carry_case()
    write_grid(tmp_path/"wrfinput_d01",*carry.state.theta.shape[-2:])
    config = NS(input_dir=tmp_path,output_dir=tmp_path/"out")
    store = RestartStore(tmp_path/"restart",10**8,3,0)
    identity = {"output_dir":str(config.output_dir.resolve()),"test":"streams"}
    start = datetime(2023,1,15,tzinfo=timezone.utc)
    runtime,carries = prepare_streams(config,("d01",),{"d01":carry},{"d01":18.},start,
                                      restart_store=store,restart_identity=identity)
    assert runtime.max_root_steps == 6
    def advance(carries,a,z):
        for n in range(a,z):
            c = carries["d01"]
            c = c.replace(state=c.state.replace(rain_acc=jnp.full_like(c.state.rain_acc,n*2.)),
                          history_diagnostics=c.history_diagnostics._replace(t2=jnp.full_like(c.state.t_skin,280.+n)))
            carries = {"d01":record_step(c,n)}
        return carries
    carries = runtime.flush(advance(carries,1,4))  # interval1 complete, sample3 pending
    assert carries["d01"].regional_streams[0].count.committed
    saved = runtime.checkpoint_state()
    generation,_ = store.save(carries,{"d01":3},{"d01":18.},identity,driver_state={"regional_streams":saved})
    expected = runtime.flush(advance(carries,4,7))
    runtime.close()  # completed later files must be authenticated and rolled back
    before = sorted((config.output_dir/"streams").glob("*.nc"))
    assert len(before) == 2
    restored,_ = store.read(generation)
    runtime,carries = prepare_streams(config,("d01",),restored["carries"],{"d01":18.},start,resumed=True,
                                      restart_store=store,restart_identity=identity,saved=saved)
    assert len(list((config.output_dir/"streams").glob("*.nc"))) == 1
    actual = runtime.flush(advance(carries,4,7))
    runtime.close()
    exact(expected,actual)
    frames = sorted((config.output_dir/"streams").glob("*.nc"))
    with Dataset(frames[1]) as ds:
        np.testing.assert_array_equal(ds["time"][:],[72.,108.])
        np.testing.assert_array_equal(ds["RAINNC_accumulation"][:],[[4,4],[4,4]])
        np.testing.assert_array_equal(ds["T2_mean"][:],[[283.5,283.5],[285.5,285.5]])
    # A foreign collision is refused; its contents are retained.
    frames[1].unlink(); frames[1].write_bytes(b"foreign")
    with pytest.raises(ValueError,match="later history frame is foreign or corrupt"):
        prepare_streams(config,("d01",),restored["carries"],{"d01":18.},start,resumed=True,
                        restart_store=store,restart_identity=identity,saved=saved)
    assert frames[1].read_bytes() == b"foreign"


def test_batch_replay_and_missing_active_state_refused(tmp_path):
    spec_file(tmp_path)
    with pytest.raises(ValueError,match="native B=1"):
        require_supported_run(tmp_path,batch_size=2)
    with pytest.raises(ValueError,match="native B=1"):
        require_supported_run(tmp_path,replay=True)
    _,_,carry = carry_case()
    with pytest.raises(ValueError,match="lacks regional stream state"):
        prepare_streams(NS(input_dir=tmp_path),("d01",),{"d01":carry},{"d01":18.},None,resumed=True)


def test_native_driver_stream_hooks_restart_and_main_alarms(tmp_path,monkeypatch):
    """Actual two-domain orchestration + real stream IO, stub numerical solver.

    This is a segmentation/restart gate, not a nested forecast fidelity claim.
    """
    import gpuwrf.integration.nested_pipeline as pipeline
    import gpuwrf.io.gen2_accessor as accessor
    import gpuwrf.runtime.restart_store as storage
    import gpuwrf.diagnostics.census as census
    from gpuwrf.runtime.domain_tree import DomainTreeResult
    grid,nl,carry = carry_case()
    inputs = tmp_path/"inputs"
    inputs.mkdir()
    spec_file(inputs,variables=["RAINNC"],buffer_frames=1)
    path = inputs/FILENAME
    config_data = json.loads(path.read_text())
    child = dict(config_data["streams"][0],name="child",domain="d02")
    config_data["streams"].append(child)
    path.write_text(json.dumps(config_data))
    names = ("d01","d02")
    for name in names:
        write_grid(inputs/f"wrfinput_{name}",grid.ny,grid.nx)
    dt = {"d01":54.,"d02":18.}
    bundles = {n:NS(state=carry.state,grid=grid,namelist=replace(nl,dt_s=dt[n])) for n in names}
    hierarchy = NS(order=names,nests=())
    tree = NS(persistent_state_bytes=lambda:{n:carry.state.bytes() for n in names})
    monkeypatch.setattr(pipeline.DomainTree,"from_domains",lambda *a,**k:tree)
    monkeypatch.setattr(pipeline,"_load_domains",lambda *a,**k:(hierarchy,bundles,{},datetime(2023,1,15),dt,{n:carry for n in names}))
    monkeypatch.setattr(accessor,"Gen2Run",lambda _:None)
    monkeypatch.setattr(pipeline,"_output_cadence_steps_by_domain",lambda *a:({"d01":2,"d02":6},dict.fromkeys(names,1.8)))
    alarms = {"d01":(2,4,6),"d02":(6,12,18)}
    monkeypatch.setattr(pipeline,"_output_alarm_steps_by_domain",lambda *a:(alarms,alarms))
    monkeypatch.setattr(pipeline,"maybe_prewarm_defused_nest",lambda *a,**k:{})
    monkeypatch.setattr(pipeline,"_prepare_operational_domain_tree_runtime",lambda *a,**k:None)
    monkeypatch.setattr(storage,"nested_identity",lambda cfg,*a:{"output_dir":str(cfg.output_dir.resolve()),"test":"actual-stream-hooks"})
    monkeypatch.setattr(census,"write_segment",lambda *a:None)
    for flag in ("GPUWRF_NESTED_ASYNC_OUTPUT","GPUWRF_NEST_OUTPUT_PIPELINE","GPUWRF_NEST_PERF_TIMERS"):
        monkeypatch.setenv(flag,"0")
    class Writer:
        def __init__(self,*,output_dir,**kwargs):
            self.output_dir=output_dir
            self.written={n:[] for n in names}
            self.writer_static_latlon_metadata={}
        def __call__(self,name,step,c):
            path = self.output_dir/f"wrfout_{name}_{step}"
            path.write_bytes(np.asarray(c.state.theta).tobytes())
            self.written[name].append(str(path))
    monkeypatch.setattr(pipeline,"_PerDomainWrfoutWriter",Writer)
    completed=[]
    def advance(tree,*,root_steps,carries,initial_own_steps,output,**kwargs):
        assert root_steps == 1  # Child minimum capacity admits at most one root trip.
        new,own={},{}
        for name,c in carries.items():
            own[name]=initial_own_steps[name]+(1 if name=="d01" else 3)
            for step in range(initial_own_steps[name]+1,own[name]+1):
                c=c.replace(state=c.state.replace(rain_acc=jnp.full_like(c.state.rain_acc,float(step))))
                c=record_step(c,step)
            new[name]=c
            if own[name] in alarms[name]:
                output(name,own[name],c)
        completed.append(new)
        return DomainTreeResult(new,{n:c.state for n,c in new.items()},own,(),(),cascade_counts={})
    monkeypatch.setattr(pipeline,"run_operational_domain_tree",advance)
    config = pipeline.NestedPipelineConfig(inputs,tmp_path/"control",tmp_path/"proof",.09,2)
    control=pipeline.execute_nested_pipeline(config)
    expected=completed[-1]
    interrupted = replace(config,output_dir=tmp_path/"resumed",checkpoint_dir=tmp_path/"restart",
                          checkpoint_interval_steps=3,checkpoint_reserve_bytes=0)
    save = storage.RestartStore.save
    def stop(*args,**kwargs):
        result=save(*args,**kwargs)
        raise OSError("simulated crash after verified stream checkpoint")
    monkeypatch.setattr(storage.RestartStore,"save",stop)
    with pytest.raises(OSError,match="verified stream checkpoint"):
        pipeline.execute_nested_pipeline(interrupted)
    monkeypatch.setattr(storage.RestartStore,"save",save)
    store=storage.RestartStore(interrupted.checkpoint_dir,10**8,2,0)
    resumed=pipeline.execute_nested_pipeline(replace(interrupted,resume_checkpoint=store.latest()))
    assert control["verdict"] == resumed["verdict"] == "PIPELINE_GREEN"
    assert resumed["hierarchy"]["observed_own_steps"] == {"d01":6,"d02":18}
    assert all(row["wrfout_count"]==3 for row in resumed["per_domain"].values())
    exact(expected,completed[-1])
    def rows(directory,name):
        result=[]
        for file in sorted((directory/"streams").glob(f"wrfout_stream_{name}_*.nc")):
            with Dataset(file) as ds:
                result.extend(zip(ds["time"][:].tolist(),ds["RAINNC"][:].tolist()))
        return result
    for name,count in [("valley",3),("child",9)]:
        assert rows(config.output_dir,name) == rows(interrupted.output_dir,name)
        assert len(rows(config.output_dir,name)) == count
    # A public Store generation can be structurally valid while omitting active
    # optional state. The actual native driver must refuse before any solve.
    snapshot,manifest=store.read(store.latest())
    missing={name:c.replace(regional_streams=None) for name,c in snapshot["carries"].items()}
    generation,_=store.save(missing,snapshot["own_steps"],dt,manifest["identity"],driver_state=snapshot["driver_state"])
    before=len(completed)
    with pytest.raises(ValueError,match="lacks regional stream state"):
        pipeline.execute_nested_pipeline(replace(interrupted,resume_checkpoint=generation))
    assert len(completed)==before


def test_cli_check(tmp_path,capsys):
    from gpuwrf.cli import main
    spec_file(tmp_path)
    assert main(["stream-check","--input",str(tmp_path/FILENAME),"--dt","18"]) == 0
    assert json.loads(capsys.readouterr().out)["streams"][0]["interval_steps"] == 2
    assert main(["stream-check","--input",str(tmp_path/FILENAME),"--dt","0"]) == 2


@pytest.mark.parametrize("region,error", [(None, None)] + BAD_PREFLIGHT_REGIONS)
def test_stream_check_subprocess_is_backend_dark(tmp_path,region,error):
    import os
    import subprocess
    import sys
    spec_file(tmp_path)
    if region is not None:
        path = tmp_path / FILENAME
        data = json.loads(path.read_text())
        data["streams"][0]["region"] = region
        path.write_text(json.dumps(data))
    root=Path(__file__).resolve().parents[1]
    program='''import sys,importlib.abc
class NoJax(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'jax','jaxlib'}: raise AssertionError('stream-check imported JAX')
sys.meta_path.insert(0,NoJax())
sys.argv=['gpuwrf','stream-check','--input',sys.argv[1]]
from gpuwrf.cli import main
raise SystemExit(main())
'''
    env=dict(os.environ,PYTHONPATH=str(root/'src'),JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0')
    run=subprocess.run([sys.executable,'-c',program,str(tmp_path/FILENAME)],env=env,cwd=root,
                       text=True,capture_output=True,timeout=30)
    assert run.returncode == (0 if region is None else 2), (run.stdout,run.stderr)
    if region is None:
        assert json.loads(run.stdout)['streams'][0]['name']=='valley'
    else:
        assert error in run.stdout


def test_observer_trace_is_real32_without_host_callbacks(tmp_path):
    spec = spec_file(tmp_path)
    _,_,carry = carry_case()
    b = seed(carry,spec,[0,1],[0,2],2,3)
    traced = jax.make_jaxpr(record_buffer)(b,jnp.ones_like(b.running),jnp.int32(1))
    def walk(value):
        if isinstance(value,jax.extend.core.ClosedJaxpr):
            yield from walk(value.jaxpr)
        elif isinstance(value,jax.extend.core.Jaxpr):
            for variable in (*value.invars,*value.outvars):
                assert getattr(variable.aval,"dtype",None) != np.dtype("float64")
            for eqn in value.eqns:
                assert eqn.primitive.name not in {"io_callback","pure_callback","debug_callback"}
                for variable in eqn.outvars:
                    assert getattr(variable.aval,"dtype",None) != np.dtype("float64")
                for parameter in eqn.params.values():
                    yield from walk(parameter)
                yield eqn.primitive.name
        elif isinstance(value,(tuple,list)):
            for item in value:
                yield from walk(item)
    assert len(list(walk(traced))) > 10


def test_region_and_name_changes_reuse_code_with_same_shapes(tmp_path):
    spec = spec_file(tmp_path)
    _,_,carry = carry_case()
    first = seed(carry,spec,[0,1],[0,2],2,3)
    other = spec_file(tmp_path,name="other_airport",region={"stations":[{"id":"c","i":2,"j":2},{"id":"d","i":4,"j":3}]})
    second = seed(carry,other,[1,2],[1,3],2,3)
    assert other.signature != spec.signature
    # This is a cache-count gate, so reserve shared JIT capacity (E183) and use
    # a private callable; prior whole-suite compilations must not evict/alias it.
    jax.clear_caches()
    fn = jax.jit(lambda buffer, values, step: record_buffer(buffer, values, step))
    values=jnp.ones_like(first.running)
    fn(first,values,jnp.int32(1)).running.block_until_ready()
    fn(second,values,jnp.int32(1)).running.block_until_ready()
    assert fn._cache_size()==1  # Genuine mutant: full spec signature in seed =>2.
    # The former full-config aux string never entered the compiled operations.
    old = replace(first,signature=spec.signature)
    assert str(jax.make_jaxpr(record_buffer)(old,values,jnp.int32(1))) == str(jax.make_jaxpr(record_buffer)(first,values,jnp.int32(1)))
    assert str(fn.lower(old,values,jnp.int32(1)).compiler_ir()) == str(fn.lower(first,values,jnp.int32(1)).compiler_ir())


def test_original_cpu_wrf_auxhist_values_and_windows(tmp_path):
    oracle = Path(__file__).resolve().parents[1]/"proofs/u2_streams/oracle"
    manifest = json.loads((oracle/"manifest.json").read_text())
    assert manifest["reference"] == "original CPU-WRF auxhist7"
    with np.load(oracle/"frames.npz") as raw:
        channels = tuple((v,stat) for v in manifest["fields"] for stat in ("instant","min","max","mean"))
        from gpuwrf.streams.device import StreamBuffer
        data = np.stack([raw[var] for var,_ in channels],axis=1)
        def gather(fields):
            groups = {group: {} for group,_,_ in FIELDS.values()}
            for name,array in fields.items():
                group,member,_ = FIELDS[name]
                groups[group][member] = array
            return sample(NS(**{g:NS(**v) for g,v in groups.items()}),channels,jnp.array([0,1]),jnp.array([1,0]))
        for n in range(5):
            arrays = {name:jnp.asarray([[999.,raw[name][n,0]],[raw[name][n,1],-999.]],jnp.float32)
                      for name in manifest["fields"]}
            np.testing.assert_array_equal(jax.jit(gather)(arrays),data[n])
        shape = data.shape[1:]
        b = StreamBuffer(jnp.array([0,1],jnp.int32),jnp.array([0,0],jnp.int32),jnp.zeros(shape,jnp.float32),
                         jnp.asarray(data[0]),jnp.zeros((2,*shape),jnp.float32),jnp.zeros(2,jnp.int32),
                         jnp.int32(0),jnp.int32(0),jnp.int32(0),channels,2,"oracle")
        for n in range(1,5):
            b = jax.jit(record_buffer)(b,jnp.asarray(data[n]),jnp.int32(n))
        for k,(var,stat) in enumerate(channels):
            frames = raw[var][1:].reshape(2,2,2)
            expected = {"instant":lambda a:a[:,-1],"min":lambda a:a.min(1),"max":lambda a:a.max(1),"mean":lambda a:a.mean(1)}[stat](frames)
            np.testing.assert_allclose(b.frames[:,k],expected,rtol=2.e-7,atol=0)
