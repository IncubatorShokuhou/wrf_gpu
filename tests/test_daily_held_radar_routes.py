"""Actual daily carry selection and generic held-radar relay; no physics claim."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
from test_m7_daily_pipeline import _synthetic_case


ROUTES=[('p3_operational',52,4,0,5,'operational_carry')]
for mp in (18,52):
    ROUTES += [(name,mp,sf,lake,bl,key) for name,sf,lake,bl,key in (
        ('ruc3',3,0,5,'ruc_carry'),('ssib8',8,0,5,'ruc_carry'),
        ('lake',4,1,5,'ruc_carry'),('temf',4,0,10,'operational_carry'),
        ('ruc_temf',3,0,10,'ruc_carry'),('lake_temf',4,1,10,'ruc_carry'))]


@pytest.mark.parametrize('name,mp,sf,lake,bl,expected_key',ROUTES)
def test_actual_daily_route_publishes_same_held_radar(tmp_path,monkeypatch,name,mp,sf,lake,bl,expected_key):
    from gpuwrf.coupling import ruc_surface_hook,ssib_surface_hook,temf,lake_surface_hook
    base=_synthetic_case(tmp_path)
    nl=NS(grid=base.grid,mp_physics=mp,sf_surface_physics=sf,sf_lake_physics=lake,
          bl_pbl_physics=bl,ruc_static=object(),lake_bundle=object(),run_boundary=False)
    case=replace(base,namelist=nl)
    radar=np.arange(base.state.qv.size,dtype=np.float32).reshape(base.state.qv.shape)/8-99
    radar.flat[-1]=41.25 # Both no-echo and positive values must survive unchanged.
    carry=NS(state=base.state,radar_reflectivity=radar,noahmp_land=None,noahmp_rad=None,
        radiation_diagnostics=None,history_diagnostics=None,ruc_land=object(),
        lake=object() if lake else None,temf=object() if bl==10 else None,
        ssib_land=object() if sf==8 else None)
    class Stepper:
        def __init__(self,state,namelist,**kw):self.carry=carry;self.next_step=1
        def __call__(self,state,namelist,hours):self.next_step+=200;return state
    monkeypatch.setattr(daily,'_RucDailyStepper',Stepper)
    monkeypatch.setattr(nested,'_noahmp_surface_diagnostics_for_output',lambda *a,**kw:{'SURFACE_SENTINEL':7})
    monkeypatch.setattr(nested,'_history_surface_diagnostics_for_output',lambda *a,**kw:{'SURFACE_SENTINEL':7})
    monkeypatch.setattr(ruc_surface_hook,'ruc_history_fields',lambda *a,**kw:{})
    monkeypatch.setattr(ssib_surface_hook,'ssib_history_fields',lambda *a,**kw:{})
    monkeypatch.setattr(temf,'history_fields',lambda *a,**kw:{})
    monkeypatch.setattr(lake_surface_hook,'overlay_lake_history',lambda values,*a,**kw:values)
    actual_surface=daily._surface_diagnostics_for_output
    routed=[];published=[]
    def surface(*args,**kwargs):
        # Observe the REAL loop's routing rather than copying its condition.
        keys=[key for key in ('ruc_carry','operational_carry') if kwargs.get(key) is carry]
        assert keys==[expected_key]
        routed.extend(keys)
        values=actual_surface(*args,**kwargs)
        assert values['SURFACE_SENTINEL']==7
        assert values.get('REFL_10CM') is radar
        return values
    monkeypatch.setattr(daily,'_surface_diagnostics_for_output',surface)
    def writer(state,grid,nl,target,**kwargs):
        value=kwargs['diagnostics']['REFL_10CM']
        assert value is radar and value.tobytes()==radar.tobytes()
        published.append(value);target.touch()
    monkeypatch.setattr(daily,'write_wrfout_netcdf',writer)
    config=daily.DailyPipelineConfig(hours=1,output_dir=tmp_path/'out',refresh_land_state_hourly=False,async_output=False)
    result=daily._run_forecast_sequence(config,output_dir=config.output_dir,
        case_builder=lambda cfg:(case,tmp_path))
    assert result.status=='PASS' and routed==[expected_key] and len(published)==1
