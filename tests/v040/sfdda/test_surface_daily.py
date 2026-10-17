"""Actual daily host loop preserves auxinput9 carry and global time across output."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.integration import daily_pipeline as daily,nested_pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.assimilation.fdda_grid import initial_fdda_carry
from gpuwrf.assimilation.fdda_surface import read_surface_record
from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
from gpuwrf.io import restart
from test_m7_daily_pipeline import _synthetic_case
from test_surface_io import storage_case,exact


@pytest.mark.parametrize('aux',[None,15])
def test_real_daily_host_loop_surface_cursor_checkpoint(monkeypatch,tmp_path,aux):
    g,nl,c=storage_case()
    nl=replace(nl,use_mp_re=0,radiation_cadence_steps=200)
    state=c.state
    case=replace(_synthetic_case(tmp_path),state=state,grid=g,namelist=nl)
    monkeypatch.setattr(daily,'write_wrfout_netcdf',lambda *a,**k:None)
    monkeypatch.setattr(daily,'prepare_wrfout_payload',lambda *a,**k:object())
    monkeypatch.setattr(daily,'_emit_auxhist_frame',lambda **k:k['output_dir']/str(k['lead_minutes']))
    monkeypatch.setattr(daily,'_commit_to_operational_device',lambda s:s)
    monkeypatch.setattr(om,'_resolve_operational_suite',lambda n:None)
    monkeypatch.setattr(om,'build_clock_base',lambda n:None)
    seeds=[];starts=[];outputs=[]
    def seed(s,n):
        seeds.append(1)
        return initial_operational_carry(s).replace(fdda=initial_fdda_carry(n.data_assimilation,jnp.float32))
    def advance(c,n,start,clock,*,n_steps,cadence):
        starts.append((int(start),int(c.fdda.surface_cursor),float(c.fdda.rthndgdten[0,0,0])))
        group=read_surface_record(c.fdda,n.data_assimilation,start,jnp.float32)
        return c.replace(fdda=group.replace(rthndgdten=jnp.full_like(group.rthndgdten,int(start))))
    def output(s,n,t,**kw):
        outputs.append(1)
        return {'T2':np.full((g.ny,g.nx),289.,np.float32)}
    original=restart.write_restart
    def write(c,n,g,step,path,**kw):
        original(c,n,g,step,path,**kw)
        back,rnl,_,restep=restart.read_restart(path)
        exact(back.fdda,c.fdda);assert restep==step==200
    monkeypatch.setattr(om,'_committed_initial_carry_for_run',seed)
    monkeypatch.setattr(om,'_advance_chunk',advance)
    monkeypatch.setattr(nested_pipeline,'_noahmp_surface_diagnostics_for_output',output)
    monkeypatch.setattr(restart,'write_restart',write)
    monkeypatch.setattr(daily,'write_checkpoint',lambda *a,**k:pytest.fail('State-only surface FDDA checkpoint'))
    cfg=daily.DailyPipelineConfig(hours=2,dt_s=18,async_output=False,score=False,refresh_land_state_hourly=False,
        auxhist=AuxhistStreamConfig(interval_minutes=aux) if aux else None)
    daily._run_forecast_sequence(cfg,output_dir=tmp_path/'out',checkpoint_at_hour=1,
        forecast_fn=daily._segmented_forecast_fn,case_builder=lambda _: (case,tmp_path))
    stride=50 if aux else 200
    assert [a for a,b,h in starts]==list(range(1,401,stride))
    assert seeds==[1] and len(outputs)==400//stride
    assert starts[200//stride][0]==201 and starts[200//stride][2]>0
    assert starts[-1][1]==(1 if aux else 0)  # cursor read at step201 survives the next aux boundary
