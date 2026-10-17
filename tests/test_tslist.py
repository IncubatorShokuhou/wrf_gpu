"""WRF time series gates: independent original CPU-WRF Swiss files and fields."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.timeseries.files import (PROFILE_FIELDS, Station, StationLocation,
                                    TimeSeriesWriter, read_tslist)
from gpuwrf.timeseries.locations import latlon_to_ij, locate_stations
from gpuwrf.timeseries.runtime import require_supported_run, prepare_time_series, TimeSeriesRuntime
from gpuwrf.timeseries.sampling import sample_fields, seed_buffer, record_step

ORACLE = Path(__file__).resolve().parents[1] / 'proofs/u2_tslist/oracle'


def surface_records(prefix):
    # Parse WRF's independent numeric stream (not the implementation's formatter).
    return np.loadtxt(ORACLE / f'{prefix}.d01.TS', skiprows=1)


def locations():
    stations = read_tslist(ORACLE / 'tslist')
    manifest = json.loads((ORACLE / 'manifest.json').read_text())
    with np.load(ORACLE / 'columns.npz') as data:
        return locate_stations(stations, 1, data['XLAT'], data['XLONG'], data['HGT'], manifest['projection'])


def test_original_wrf_locations():
    locs = locations()
    assert [(l.station.prefix, l.i, l.j, l.station.station_id) for l in locs] == [('ctr', 20, 20, 1), ('west', 9, 12, 2)]
    # The WRF grid's SW point projects to 1,1, not 0,0.
    manifest = json.loads((ORACLE / 'manifest.json').read_text())
    with np.load(ORACLE / 'columns.npz') as data:
        lat, lon = data['XLAT'][0, 0], data['XLONG'][0, 0]
    np.testing.assert_allclose(latlon_to_ij(lat, lon, manifest['projection'], lat, lon), [1, 1], atol=2.e-4)


def test_wrf_files_byte_exact_full_and_partial_flush(tmp_path):
    locs = locations()
    records = np.stack([surface_records(l.station.prefix) for l in locs], axis=1)
    profiles = np.stack([np.stack([np.loadtxt(ORACLE/f'{loc.station.prefix}.d01.{ending}', skiprows=1)[:, 1:]
                                   for ending in PROFILE_FIELDS], axis=1) for loc in locs], axis=1)
    writer = TimeSeriesWriter(tmp_path, locs, '2023-01-15_00:00:00', 5)
    writer.write(records[:3, 0, 1], records[:3, :, 5:], profiles[:3])
    writer.write(records[3:, 0, 1], records[3:, :, 5:], profiles[3:])
    assert writer.rows == 4
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (ORACLE / path.name).read_bytes(), path.name


def patch_fields(frame, *, cloudy=False):
    """Pack independent WRF station neighborhoods side by side in a tiny grid."""
    with np.load(ORACLE / 'columns.npz') as raw:
        # (station,z,y,x) -> (z,y,station*x); retain actual staggered operand values.
        fields = {key: np.concatenate(tuple(raw[key][frame]), axis=-1) for key in raw.files
                  if key not in {'ZNW', 'XLAT', 'XLONG', 'HGT'}}
        eta = raw['ZNW']
    f = np.float32
    dnw = np.diff(eta)
    dn = f(.5)*(dnw[:-1]+dnw[1:])
    metrics = NS(cosa=jnp.asarray(fields['COSALPHA']), sina=jnp.asarray(fields['SINALPHA']),
                 fnm=jnp.asarray(np.r_[f(0), f(.5)*dnw[:-1]/dn]),
                 fnp=jnp.asarray(np.r_[f(0), f(.5)*dnw[1:]/dn]))
    state = NS(**{name: jnp.asarray(fields[key]) for name, key in
                  {'u':'U', 'v':'V', 'w':'W', 'qv':'QVAPOR', 'qc':'QCLOUD', 'qr':'QRAIN',
                   'qi':'QICE', 'qs':'QSNOW', 'qg':'QGRAUP', 'hfx':'HFX', 't_skin':'TSK',
                   'rain_acc':'RAINNC', 'rainc_acc':'RAINC'}.items()},
               theta=jnp.asarray(fields['THM']+f(300)),
               p_total=jnp.asarray(fields['P']+fields['PB']),
               ph_total=jnp.asarray(fields['PH']+fields['PHB']))
    if cloudy:
        # All species and all levels active, including levels above max_ts_level.
        for idx, name in enumerate(('qc', 'qr', 'qi', 'qs', 'qg'), 1):
            setattr(state, name, jnp.full_like(state.qv, idx*2.e-5))
    history = NS(**{name: jnp.asarray(fields[key]) for name, key in
                    {'t2':'T2', 'q2':'Q2', 'u10':'U10', 'v10':'V10', 'psfc':'PSFC', 'lh':'LH'}.items()})
    radiation = NS(glw=jnp.asarray(fields['GLW']), swdown=jnp.asarray(fields['SWDOWN']), swup=jnp.asarray(fields['SWUPB']))
    soil = jnp.asarray(fields['TSLB'][0, 0, [0, 2]])
    return state, history, radiation, soil, metrics


@pytest.mark.parametrize('frame', range(4))
def test_sampler_same_input_original_wrf(frame):
    state, hist, rad, soil, metrics = patch_fields(frame)
    # Close over ordinary containers; only array operands enter the compiled sampler.
    sample = jax.jit(lambda x,y: sample_fields(state,hist,rad,soil,metrics,x,y,5))
    actual, profiles = sample(jnp.array([0, 2]), jnp.array([0, 0]))
    expected = np.stack([surface_records(p)[frame, 5:] for p in ('ctr', 'west')])
    # WRF formatted REAL fields have 5 decimal digits; 1 ulp at PSFC = 0.0078125 Pa.
    np.testing.assert_allclose(actual, expected, atol=5.1e-6, rtol=2.e-7)
    for index, prefix in enumerate(('ctr', 'west')):
        for k, ending in enumerate(PROFILE_FIELDS):
            expected = np.loadtxt(ORACLE / f'{prefix}.d01.{ending}', skiprows=1)[frame, 1:]
            np.testing.assert_allclose(profiles[index, k], expected, atol=5.1e-6, rtol=2.e-7,
                                       err_msg=f'{prefix} {ending}')


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_writer_rejects_nonfinite_before_append(tmp_path, bad):
    writer = TimeSeriesWriter(tmp_path, locations(), '2023-01-15_00:00:00', 5)
    for operand in range(3):
        values = [np.array([.005]), np.zeros((1, 2, 14)), np.zeros((1, 2, 7, 5))]
        values[operand].flat[0] = bad
        before = {p: p.read_bytes() for p in tmp_path.iterdir()}
        with pytest.raises(ValueError, match='NaN/Inf'):
            writer.write(*values)
        assert all(p.read_bytes() == content for p, content in before.items())


def test_writer_refuses_existing_stream(tmp_path):
    writer = TimeSeriesWriter(tmp_path, locations(), '2023-01-15_00:00:00', 5)
    before = writer.path(locations()[0], 'TS').read_bytes()
    with pytest.raises(FileExistsError):
        TimeSeriesWriter(tmp_path, locations(), '2023-01-15_00:00:00', 5)
    assert writer.path(locations()[0], 'TS').read_bytes() == before


def test_ij_parser_id_preservation_limits_and_outside(tmp_path):
    path = tmp_path / 'tslist'
    path.write_text('#\n# 24 characters for name | pfx |   I   |    J   |\n#\n'
                    + f'{"outside":25} {"out":5} {999:7d} {999:8d}\n'
                    + f'{"inside":25} {"in":5} {2:7d} {2:8d}\n')
    stations = read_tslist(path)
    assert stations[0].ij
    field = np.zeros((3,3))
    locs = locate_stations(stations, 2, field, field, field, {'MAP_PROJ':1})
    assert [(l.domain_id, l.station.station_id, l.i, l.j) for l in locs] == [(2,2,2,2)]
    with pytest.warns(UserWarning):
        assert len(read_tslist(path, 1)) == 1
    assert read_tslist(path, 0) == ()


@pytest.mark.parametrize('record', ['../x ', 'abc.d', '     '])
def test_unsafe_prefix_refused(tmp_path, record):
    path = tmp_path / 'tslist'
    path.write_text('#\n# 24 characters for name | pfx |   I   |    J   |\n#\n'
                    + f'{"bad":25} {record} {2:7d} {2:8d}\n')
    with pytest.raises(ValueError, match='prefix'):
        read_tslist(path)


def test_empty_default_path_returns_same_carry_without_backend(tmp_path):
    carries = {'d01': object()}
    assert prepare_time_series(NS(input_dir=tmp_path), (), {}, carries, {}, None, {}) == (None, carries)
    assert prepare_time_series(NS(input_dir=tmp_path), (), {}, carries, {}, None, {})[1] is carries


@pytest.mark.parametrize('mode', [{'restart':True}, {'batch_size':2}, {'replay':True}])
def test_unsupported_modes_refuse_before_model_work(tmp_path, mode):
    require_supported_run(tmp_path, **mode)  # no tslist: old path unchanged
    (tmp_path / 'tslist').write_text('')
    with pytest.raises(ValueError, match='fresh native B=1'):
        require_supported_run(tmp_path, **mode)


class FakeCarry(NS):
    def replace(self, **kwargs):
        return FakeCarry(**(vars(self) | kwargs))


def test_record_scan_flush_and_continue(tmp_path):
    state, hist, rad, _soil, metrics = patch_fields(0)
    locs = tuple(replace(l, i=i+1, j=1) for i,l in zip((0,2), locations()))
    buffer = seed_buffer(locs, 5, 3, land=[True, True])
    # A production-shaped pytree seam; changing station count changes only the observer.
    # Use the small buffer itself as scan carry (ordinary test containers close over arrays).
    carry = FakeCarry(state=state, history_diagnostics=hist, radiation_diagnostics=rad,
                      noahmp_land=NS(tslb=jnp.full((4,2,4),272.,jnp.float32)), timeseries=buffer)
    nl = NS(metrics=metrics, dt_s=18.)
    scan = jax.jit(lambda b: jax.lax.fori_loop(1,4, lambda n,b: record_step(carry.replace(timeseries=b),nl,n).timeseries, b))
    result = scan(buffer)
    assert int(result.count) == 3
    np.testing.assert_allclose(result.hours, [.005,.010,.015], rtol=2.e-7)
    writer = TimeSeriesWriter(tmp_path, locs, '2023-01-15_00:00:00', 5)
    runtime = TimeSeriesRuntime({'d01':writer}, 3)
    carries = runtime.flush({'d01':carry.replace(timeseries=result)})
    assert int(carries['d01'].timeseries.count) == 0
    assert carries['d01'].timeseries.count.committed
    result = jax.jit(lambda b: record_step(carry.replace(timeseries=b), nl, 4).timeseries)(carries['d01'].timeseries)
    runtime.flush({'d01':carry.replace(timeseries=result)})
    assert writer.rows == 4
    for loc in locs:
        data = np.loadtxt(writer.path(loc, 'TS'), skiprows=1)
        np.testing.assert_array_equal(data[:,1], [.005,.010,.015,.020])


def test_hook_runs_after_physics_and_off_has_no_observer(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    import gpuwrf.timeseries.sampling as sampling
    events = []
    original = FakeCarry(timeseries=object(), state='entry')
    final = FakeCarry(timeseries=None, state='post_solve')
    monkeypatch.setattr(om, '_physics_boundary_step_with_limiter_diagnostics',
                        lambda *a,**k: (events.append('solve') or final, None))
    for key in ('GPUWRF_DYN_RK_FP32','GPUWRF_DYN_CARRY_FP32'):
        monkeypatch.setenv(key, '0')
    def observe(carry,nl,step):
        assert carry.state == 'post_solve'
        assert carry.timeseries is original.timeseries
        assert step == 7
        events.append('sample')
        return carry
    monkeypatch.setattr(sampling,'record_step',observe)
    om._physics_boundary_step(original, None, 7, run_radiation=False)
    assert events == ['solve','sample']
    events.clear()
    assert om._physics_boundary_step(original.replace(timeseries=None), None, 7, run_radiation=False) is final
    assert events == ['solve']


def test_active_clw_uses_pristine_pressure_faces_all_species_and_levels():
    state, hist, rad, soil, metrics = patch_fields(0)
    with np.load(ORACLE / 'active_clw.npz') as oracle:
        for mm, name in enumerate(('qc','qr','qi','qs','qg')):
            # Spread the two real station columns into the packed test grid.
            q = np.zeros_like(state.qv)
            q[:,0,0] = oracle['moist'][0,:,0,mm]
            q[:,0,2] = oracle['moist'][1,:,0,mm]
            setattr(state,name,jnp.asarray(q))
        expected = oracle['clw']
    fields, _ = jax.jit(lambda: sample_fields(state,hist,rad,soil,metrics,jnp.array([0,2]),jnp.array([0,0]),5))()
    assert np.all(expected > 3.)
    # Literal 220-term REAL accumulation: f32 XLA FMA may move a few ulps.
    np.testing.assert_allclose(fields[:,-1],expected,rtol=8*np.finfo(np.float32).eps,atol=0)


def test_ij_unstagger_original_wrf_format_and_samples(tmp_path):
    fixture = ORACLE / 'ij_unstagger'
    stations = read_tslist(fixture/'tslist')
    with np.load(ORACLE/'columns.npz') as raw:
        locs = locate_stations(stations,1,raw['XLAT'],raw['XLONG'],raw['HGT'],{'MAP_PROJ':1})
    writer = TimeSeriesWriter(tmp_path,locs,'2023-01-15_00:00:00',5)
    records = np.stack([np.loadtxt(fixture/f'{p}.d01.TS',skiprows=1) for p in ('ctr','west')],axis=1)
    profiles = np.stack([np.stack([np.loadtxt(fixture/f'{p}.d01.{e}',skiprows=1)[:,1:]
                                  for e in PROFILE_FIELDS],axis=1) for p in ('ctr','west')],axis=1)
    writer.write(records[:,0,1],records[:,:,5:],profiles)
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (fixture/path.name).read_bytes()
    for frame in range(4):
        state,hist,rad,soil,metrics = patch_fields(frame)
        fields, actual = jax.jit(lambda: sample_fields(state,hist,rad,soil,metrics,
                                   jnp.array([0,2]),jnp.array([0,0]),5,unstagger=True))()
        np.testing.assert_allclose(actual,profiles[frame],atol=5.1e-6,rtol=2.e-7)
        np.testing.assert_allclose(fields,records[frame,:,5:],atol=5.1e-6,rtol=2.e-7)


def test_buffered_tslist_restart_refused():
    from gpuwrf.io.restart import _carry_to_payload
    with pytest.raises(ValueError,match='tslist stream checkpoint'):
        # E78 now validates the preceding inactive epsilon-history contract.
        _carry_to_payload(FakeCarry(timeseries=object(), history_diagnostics=None,
            state=NS(epsilon_pbl=None, t_skin=np.zeros((1, 1), np.float32))))


@pytest.mark.parametrize('child_nz',[44,88])
def test_native_setup_nested_capacity_levels_and_committed_seed(tmp_path,child_nz):
    from datetime import datetime
    from gpuwrf.io.netcdf_lock import Dataset
    path = tmp_path/'tslist'
    path.write_text('#\n# 24 characters for name | pfx |   I   |    J   |\n#\n'
                    + f'{"station":25} {"sta":5} {2:7d} {1:8d}\n')
    carries,bundles = {},{}
    for name in ('d01','d02'):
        with Dataset(tmp_path/f'wrfinput_{name}','w') as ds:
            ds.createDimension('Time',1)
            ds.createDimension('south_north',2)
            ds.createDimension('west_east',4)
            ds.MAP_PROJ=1
            for key in ('XLAT','XLONG','HGT','LANDMASK'):
                ds.createVariable(key,'f4',('Time','south_north','west_east'))[:] = 1
        nz=44 if name=='d01' else child_nz
        carries[name] = FakeCarry(state=NS(theta=jnp.zeros((nz,2,4),jnp.float32)),noahmp_land=object(),
                                 history_diagnostics=object(),radiation_diagnostics=object())
        bundles[name] = NS(namelist=NS(sf_surface_physics=4))
    config = NS(input_dir=tmp_path,output_dir=tmp_path/'output')
    runtime,seeded = prepare_time_series(config,('d01','d02'),bundles,carries,{'d01':54.,'d02':18.},
                                         datetime(2023,1,15),{'domains':{'ts_buf_size':2,'max_ts_level':99}})
    assert runtime.max_root_steps == 1
    assert seeded['d01'].timeseries.hours.shape == (2,)
    assert seeded['d02'].timeseries.hours.shape == (3,)
    assert seeded['d02'].timeseries.profiles.shape == (3,1,7,44)
    assert all(x.committed for name in seeded for x in jax.tree.leaves(seeded[name].timeseries))
    assert len(list(config.output_dir.iterdir())) == 16


@pytest.mark.parametrize('format',['csv','netcdf'])
def test_selected_variable_interval_export_against_original_wrf(tmp_path,format):
    from gpuwrf.timeseries.export import export_surface
    target=tmp_path/f'out.{format}'
    receipt=export_surface(ORACLE/'ctr.d01.TS',target,variables=['T2','U10','CLW'],interval_seconds=36,format=format)
    expected=surface_records('ctr')[[1,3]][:,[1,5,7,18]]
    assert receipt['rows']==2
    if format=='csv':
        actual=np.loadtxt(target,delimiter=',',skiprows=1)[:,[1,5,6,7]]
    else:
        from gpuwrf.io.netcdf_lock import Dataset
        with Dataset(target) as ds:
            assert ds['T2'].units=='K'
            actual=np.stack([ds[key][:] for key in ['time','T2','U10','CLW']],axis=1)
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(FileExistsError):
        export_surface(ORACLE/'ctr.d01.TS',target,format=format)


def test_export_cli_stays_backend_dark(tmp_path):
    import os,subprocess,sys
    source=str(ORACLE/'ctr.d01.TS')
    target=str(tmp_path/'station.csv')
    code='''import sys
from jax._src import xla_bridge
def refuse(*args,**kwargs):
    raise AssertionError('offline export attempted backend initialization')
xla_bridge._initialize_backend=refuse
from gpuwrf.cli import main
assert main(sys.argv[1:]) == 0
assert not xla_bridge._backends
'''
    env=dict(os.environ,PYTHONPATH=str(ORACLE.parents[2]/'src'),JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0')
    result=subprocess.run([sys.executable,'-c',code,'tslist-export','--input',source,'--output',target,
                           '--variables','T2,U10','--interval-seconds','36'],cwd=ORACLE.parents[2],env=env,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['rows']==2


def test_projection_pristine_source_oracle_both_hemispheres_dateline():
    oracle=json.loads((ORACLE/'projection.json').read_text())
    for case,expected in zip(oracle['cases'],oracle['expected_ij']):
        code,tl1,tl2,stdlon,dx,lat1,lon1,lat,lon=case
        attrs={'MAP_PROJ':code,'TRUELAT1':tl1,'TRUELAT2':tl2,'STAND_LON':stdlon,'DX':dx}
        actual=latlon_to_ij(lat,lon,attrs,lat1,lon1)
        # REAL trig/power implementations can differ at a few ulps of the
        # large pole-coordinate subtraction. Integer WRF placement is exact.
        np.testing.assert_allclose(actual,expected,atol=.001,rtol=0)
        nint=lambda x:np.copysign(np.floor(np.abs(x)+.5),x)
        np.testing.assert_array_equal(nint(np.asarray(actual)),nint(np.asarray(expected)))


def test_tracked_oracle_fixture_checksums():
    import hashlib,yaml
    manifest=yaml.safe_load((ORACLE/'fixture.yaml').read_text())
    assert len(manifest['variables'])==22
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((ORACLE/name).read_bytes()).hexdigest()==sha,name


def test_station_csv_consumed_directly_by_u1_verification(tmp_path):
    import math
    from gpuwrf.timeseries.export import export_surface
    from gpuwrf.verify.observations import read_forecast_csv
    target=tmp_path/'verify.csv'
    receipt=export_surface(ORACLE/'ctr.d01.TS',target,format='station-csv',station_uid='swiss-centre',
                           variables=['T2','U10','V10'])
    assert receipt['rows']==12
    data=read_forecast_csv(target)
    assert set(data.station_uid)=={'swiss-centre'}
    assert set(data.variable)=={'t2','wind','wind_direction'}
    raw=surface_records('ctr')
    np.testing.assert_allclose(data[data.variable=='t2'].value,raw[:,5]-273.15,atol=1.e-12)
    wind=data[data.variable=='wind']
    np.testing.assert_array_equal(wind.u_ms,raw[:,7])
    np.testing.assert_array_equal(wind.v_ms,raw[:,8])
    np.testing.assert_allclose(wind.value,[math.hypot(u,v) for u,v in raw[:,[7,8]]],atol=1.e-12)
    np.testing.assert_allclose(data[data.variable=='wind_direction'].value,
                               [math.degrees(math.atan2(-u,-v))%360 for u,v in raw[:,[7,8]]],atol=1.e-12)
    assert str(data.valid_time.iloc[0])=='2023-01-15 00:00:18+00:00'


def test_station_csv_rejects_ambiguous_wind_and_uid(tmp_path):
    from gpuwrf.timeseries.export import export_surface
    with pytest.raises(ValueError,match='explicit station_uid'):
        export_surface(ORACLE/'ctr.d01.TS',tmp_path/'a.csv',format='station-csv',variables=['T2'])
    with pytest.raises(ValueError,match='U10,V10 pair'):
        export_surface(ORACLE/'ctr.d01.TS',tmp_path/'a.csv',format='station-csv',station_uid='x',variables=['U10'])


def test_calc_p8w_one_level_run_refused_before_reading_wrfinput(tmp_path):
    from datetime import datetime
    (tmp_path/'tslist').write_bytes((ORACLE/'tslist').read_bytes())
    carry=FakeCarry(state=NS(theta=jnp.zeros((1,2,4),jnp.float32)),noahmp_land=object(),
                    history_diagnostics=object(),radiation_diagnostics=object())
    with pytest.raises(ValueError,match='at least two mass levels'):
        prepare_time_series(NS(input_dir=tmp_path,output_dir=tmp_path/'out'),('d01',),
                            {'d01':NS(namelist=NS(sf_surface_physics=4))},{'d01':carry},
                            {'d01':18.},datetime(2023,1,15),{})


def test_all_outside_stations_leave_forecast_unsegmented(tmp_path):
    from datetime import datetime
    from gpuwrf.io.netcdf_lock import Dataset
    (tmp_path/'tslist').write_text('#\n# 24 characters for name | pfx |   I   |    J   |\n#\n'
                                 + f'{"outside":25} {"out":5} {999:7d} {1:8d}\n')
    with Dataset(tmp_path/'wrfinput_d01','w') as ds:
        for key,n in [('Time',1),('south_north',2),('west_east',4)]:
            ds.createDimension(key,n)
        ds.MAP_PROJ=1
        for key in ('XLAT','XLONG','HGT','LANDMASK'):
            ds.createVariable(key,'f4',('Time','south_north','west_east'))[:]=1
    carry=FakeCarry(state=NS(theta=jnp.zeros((44,2,4),jnp.float32)),noahmp_land=object(),
                    history_diagnostics=object(),radiation_diagnostics=object())
    carries={'d01':carry}
    runtime,result=prepare_time_series(NS(input_dir=tmp_path,output_dir=tmp_path/'out'),('d01',),
                                       {'d01':NS(namelist=NS(sf_surface_physics=4))},carries,
                                       {'d01':54.},datetime(2023,1,15),{})
    assert runtime is None and result is carries
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('solar',[1,[1]])
def test_solar_extended_format_refused_before_model_or_output(tmp_path,solar):
    (tmp_path/'tslist').write_bytes((ORACLE/'tslist').read_bytes())
    with pytest.raises(ValueError,match='solar'):
        prepare_time_series(NS(input_dir=tmp_path,output_dir=tmp_path/'out'),(),{}, {},{},None,
                            {'diags':{'solar_diagnostics':solar}})
    assert not (tmp_path/'out').exists()
