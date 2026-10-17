"""Actual daily host lifecycle with recorded chunks and the real restart codec.

These tests replace model arithmetic and output IO, not the daily dispatcher,
retained stepper, global clock, boundary guard or checkpoint implementation.
Scientific model validation is carried forward from the original WRF packet.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state,_namelist
from test_m7_daily_pipeline import _synthetic_case
from gpuwrf.coupling.cuplus import CuPlusCarry,initial_carry
from gpuwrf.coupling.scan_adapters import initial_camuw_carry
from gpuwrf.integration import daily_pipeline as daily
from gpuwrf.io import restart
from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics

PROFILES=((10,5,5,4),(11,5,5,4),(7,9,5,4),(7,2,2,0))


def _output_hooks(monkeypatch):
    monkeypatch.setattr(daily,'write_wrfout_netcdf',lambda *a,**k:None)
    monkeypatch.setattr(daily,'prepare_wrfout_payload',lambda *a,**k:object())
    monkeypatch.setattr(daily,'_emit_auxhist_frame',lambda **k:k['output_dir']/str(k['lead_minutes']))
    monkeypatch.setattr(daily,'_commit_to_operational_device',lambda s:s)


def _same(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert (x.dtype,x.shape,x.tobytes())==(y.dtype,y.shape,y.tobytes())


def _case(tmp_path,cu,bl,sf,lsm,**options):
    grid=_grid(nz=4);state=_state(grid)
    nml=_namelist(grid,dt_s=18.,cu_physics=cu,bl_pbl_physics=bl,sf_sfclay_physics=sf,
        sf_surface_physics=lsm,use_noahmp=lsm==4,rad_rk_tendf=1,**options)
    return replace(_synthetic_case(tmp_path),state=state,grid=grid,namelist=nml,metadata={})


def _carry(state,cu,bl):
    value=initial_carry(state,cu)
    held={k:(v if v.dtype==jnp.int32 else v+jnp.float32(.125)) for k,v in value.held.items()}
    if cu in (10,11):held['NCA']=jnp.full_like(held['NCA'],10000.)
    carry=initial_operational_carry(state).replace(cuplus=CuPlusCarry(value.w0avg+1.,held))
    return carry.replace(camuw_pbl=initial_camuw_carry(state)) if bl==9 else carry


@pytest.mark.parametrize('cu,bl,sf,lsm',PROFILES+((0,5,5,4),))
def test_actual_daily_route_includes_cuplus_and_preserves_default(monkeypatch,tmp_path,cu,bl,sf,lsm):
    class SelectedRoute(Exception):pass
    nml=SimpleNamespace(cu_physics=cu,bl_pbl_physics=bl,sf_sfclay_physics=sf,
        sf_surface_physics=lsm,sf_ocean_physics=0,isftcflx=0)
    case=SimpleNamespace(state=SimpleNamespace(),namelist=nml,metadata={})
    def full(*a,**k):raise SelectedRoute('full-carry')
    def state_only(*a,**k):raise SelectedRoute('State-only')
    monkeypatch.setattr(daily,'_RucDailyStepper',full)
    monkeypatch.setattr(daily,'_capture_boundary_leaves',state_only)
    with pytest.raises(SelectedRoute) as route:
        daily._run_forecast_sequence(daily.DailyPipelineConfig(hours=2),output_dir=tmp_path/'out',
            forecast_fn=daily._segmented_forecast_fn,case_builder=lambda cfg:(case,tmp_path))
    assert str(route.value)==('State-only' if cu==0 else 'full-carry')


@pytest.mark.parametrize('cu,bl,sf,lsm',PROFILES)
@pytest.mark.parametrize('aux_minutes',[None,15])
def test_daily_preserves_nonzero_cloud_rates_timer_clock_and_checkpoint(
        monkeypatch,tmp_path,cu,bl,sf,lsm,aux_minutes):
    _output_hooks(monkeypatch)
    case=_case(tmp_path,cu,bl,sf,lsm,radiation_cadence_steps=200)
    seeds=[];calls=[];outputs=[];checkpoints=[];clock=object()
    def seed(state,nml):
        seeds.append(1)
        carry=_carry(state,cu,bl)
        if nml.sf_sfclay_physics==2:
            from gpuwrf.coupling.myj_surface import initial_myj_memory
            # Resident MYJSFC memory, as _initial_carry_for_run seeds it (t-cfix-sfc F12, restart E78).
            carry=carry.replace(history_diagnostics=seed_history_diagnostics(state,myjsfc=True),
                                qnse=initial_myj_memory(state))
        return carry
    def advance(c,n,start,cb,*,n_steps,cadence):
        start=int(start);calls.append(start)
        assert cb is clock and cadence==200
        np.testing.assert_array_equal(c.cuplus.w0avg,np.float32(1.+(start-1)/4096.))
        np.testing.assert_array_equal(c.cuplus.held['RTHCUTEN'],np.float32(.125+(start-1)/8192.))
        if cu in (10,11):
            np.testing.assert_array_equal(c.cuplus.held['NCA'],np.float32(10000.-18.*(start-1)))
        held=dict(c.cuplus.held)
        held['RTHCUTEN']=held['RTHCUTEN']+jnp.float32(n_steps/8192.)
        if cu in (10,11):held['NCA']=held['NCA']-jnp.float32(18.*n_steps)
        cloud='CLDFRA_CUP' if cu==10 else 'CLDFRA_DP_KF' if cu==11 else 'DLF_OUT'
        assert np.all(np.asarray(held[cloud])!=0.)
        return c.replace(cuplus=CuPlusCarry(c.cuplus.w0avg+jnp.float32(n_steps/4096.),held))
    def output(state,nml,run_start,**kw):
        c=kw['operational_carry'];outputs.append(c)
        assert np.all(np.asarray(c.cuplus.held['RTHCUTEN'])!=0.)
        return {}
    write_real=restart.write_restart
    def checkpoint(c,n,g,step,path,**kw):
        write_real(c,n,g,step,path,**kw)
        restored,_,_,saved_step=restart.read_restart(path)
        _same(c,restored)
        assert saved_step==step==200
        checkpoints.append(Path(path))
    monkeypatch.setattr(om,'_committed_initial_carry_for_run',seed)
    monkeypatch.setattr(om,'build_clock_base',lambda n:clock)
    monkeypatch.setattr(om,'_advance_chunk',advance)
    monkeypatch.setattr(daily,'_surface_diagnostics_for_output',output)
    monkeypatch.setattr(restart,'write_restart',checkpoint)
    monkeypatch.setattr(daily,'write_checkpoint',lambda *a,**k:pytest.fail('CuPlus State-only checkpoint'))
    cfg=daily.DailyPipelineConfig(hours=2,dt_s=18.,async_output=False,score=False,
        refresh_land_state_hourly=False,auxhist=AuxhistStreamConfig(interval_minutes=aux_minutes) if aux_minutes else None)
    result=daily._run_forecast_sequence(cfg,output_dir=tmp_path/'out',checkpoint_at_hour=1,
        forecast_fn=daily._segmented_forecast_fn,case_builder=lambda cfg:(case,tmp_path))
    stride=50 if aux_minutes else 200
    assert calls==list(range(1,401,stride))
    assert seeds==[1] and len(checkpoints)==1 and len(outputs)==400//stride
    assert result.checkpoint['step_index']==200
    np.testing.assert_array_equal(outputs[-1].cuplus.w0avg,np.float32(1.+400/4096.))


@pytest.mark.parametrize('bl,sf,lsm',[(9,5,4),(2,2,0)])
@pytest.mark.parametrize('aux_minutes',[None,15])
def test_daily_selected_stepper_reports_sticky_cam_entropy(monkeypatch,tmp_path,bl,sf,lsm,aux_minutes):
    _output_hooks(monkeypatch)
    cadence=25 if aux_minutes else 100
    case=_case(tmp_path,7,bl,sf,lsm,radiation_cadence_steps=cadence)
    starts=[]
    def advance(c,n,start,clock,*,n_steps,cadence):
        starts.append(int(start))
        held=dict(c.cuplus.held)
        if len(starts)==1:
            held['ENTROPY_FAILED']=held['ENTROPY_FAILED'].at[1].set(1)
        else:
            assert int(held['ENTROPY_FAILED'][1])==1
            held['RTHCUTEN']=jnp.zeros_like(held['RTHCUTEN'])
        return c.replace(cuplus=CuPlusCarry(c.cuplus.w0avg,held))
    monkeypatch.setattr(om,'_committed_initial_carry_for_run',lambda s,n:_carry(s,7,bl))
    monkeypatch.setattr(om,'build_clock_base',lambda n:None)
    monkeypatch.setattr(om,'_advance_chunk',advance)
    cfg=daily.DailyPipelineConfig(hours=2,dt_s=18.,async_output=False,score=False,
        refresh_land_state_hourly=False,auxhist=AuxhistStreamConfig(interval_minutes=aux_minutes) if aux_minutes else None)
    end=50 if aux_minutes else 200
    with pytest.raises(RuntimeError,match=rf'CAM-ZM entropy.*domain=daily.*step={end}.*column=1'):
        daily._run_forecast_sequence(cfg,output_dir=tmp_path/'out',forecast_fn=daily._segmented_forecast_fn,
            case_builder=lambda cfg:(case,tmp_path))
    assert starts==[1,cadence+1]
