"""Actual c40 noon atmosphere and prepared soil -> real radiation -> LSM caller."""
import dataclasses
from datetime import datetime,timezone
from pathlib import Path
import json
from conftest import cpu_pallas_interpret
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from real_crop import real_atmosphere

OUT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-ysuqc/c40_review')
class Stop(Exception):
    pass

def test_actual_c40_noon_swdown_reaches_surface(cpu_pallas_interpret,tmp_path,monkeypatch):
    from test_s2_ssib_wiring import _case
    from gpuwrf.io.ssib_land_init import load_ssib_land
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.coupling import ssib_surface_hook as hook
    from gpuwrf.coupling.physics_couplers import RRTMGRadiationStatic
    from gpuwrf.runtime import operational_mode as om
    run=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c40/cpu')
    case=run/'wrfout_d01_2026-10-13_12:00:00'
    grid,state=real_atmosphere(case)
    _,_,nml,_=_case(tmp_path)
    static,land=load_ssib_land(run,'d01')
    full_shape=np.asarray(static.xland).shape
    def crop(x):
        if hasattr(x,'shape') and len(x.shape)>=2 and tuple(x.shape[-2:])==full_shape:
            return x[...,20:22,20:23]
        return x
    static,land=jax.tree.map(crop,(static,land))
    with Dataset(case) as ds:
        lat=np.asarray(ds['XLAT'][0,20:22,20:23]);lon=np.asarray(ds['XLONG'][0,20:22,20:23])
        terrain=np.asarray(ds['HGT'][0,20:22,20:23])
    z=np.zeros((2,3),np.float32)
    rad_static=RRTMGRadiationStatic(xlat_deg=lat,xlong_deg=lon,terrain_height_m=terrain,
        slope_rad=z,slope_azimuth_rad=z,sina=z,cosa=z+1.)
    nml=dataclasses.replace(nml,grid=grid,metrics=grid.metrics,dt_s=50.,ssib_static=static,ssib_land=land,
        time_utc=datetime(2026,10,13,12,tzinfo=timezone.utc),radiation_static=rad_static)
    carry=om._initial_carry_for_run(state,nml)
    _,diag=hook.ssib_radiation(state,carry.ssib_land,static,nml,0.,None)
    down=np.asarray(diag.swdown);normal=np.asarray(diag.swnorm)
    assert np.isfinite(down).all() and down.min()>100.
    np.testing.assert_array_equal(normal,np.zeros_like(normal))
    seen={}
    def step(*args,**kwargs):
        seen.update(kwargs)
        raise Stop
    monkeypatch.setattr(hook,'ssib_surface_step',step)
    with pytest.raises(Stop):
        om._physics_step_forcing(carry.replace(radiation_diagnostics=diag),nml,0.,
            run_radiation=False,first_timestep=True)
    got=np.asarray(seen['radiation'][0])
    np.testing.assert_array_equal(got,down.astype(np.float32))
    (OUT/'ssib_real_columns.json').write_text(json.dumps(dict(
        source=str(case),crop=[20,22,20,23],atmosphere='original CPU-WRF noon crop and native vertical metrics',
        soil='original c40 real.exe prepared input crop',swdown_min=float(down.min()),swdown_max=float(down.max()),
        swnorm_all_zero=True,caller_swdown_byte_equal=True,scope='producer/caller interface; not a corrected forecast D6'),indent=2)+'\n')
