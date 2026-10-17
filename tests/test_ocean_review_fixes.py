"""Deletion-sensitive admission and real NetCDF read gates for rv-s2-oml."""
from dataclasses import replace
from datetime import datetime, timezone

import jax
import numpy as np
import pytest

from gpuwrf.io.namelist_binding import namelist_honour_report
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime.operational_mode import _resolve_operational_suite
from test_ocean_wiring import scene


@pytest.mark.parametrize('option', [1, 2])
@pytest.mark.parametrize('lsm,urban,pbl,sf,reason', [
    (3,0,5,5,'RUC'),
    (4,1,5,1,'SLUCM'), (4,1,5,5,'SLUCM'), (4,1,5,91,'SLUCM'),
    (4,2,1,1,'BEP/BEM'), (4,3,1,1,'BEP/BEM'),
    (4,0,16,1,'epsilon-PBL'), (4,0,17,1,'epsilon-PBL'),
])
def test_unqualified_hurricane_pairings_refused_at_both_entries(option,lsm,urban,pbl,sf,reason):
    physics=dict(sf_ocean_physics=0,isftcflx=option,sf_surface_physics=lsm,
                 sf_sfclay_physics=sf,sf_urban_physics=urban,bl_pbl_physics=pbl,
                 mp_physics=8,cu_physics=0,ra_lw_physics=4,ra_sw_physics=4)
    errors,_=namelist_honour_report({'physics':physics,'dynamics':{'diff_opt':1,'km_opt':4}},('d01',))
    assert any('isftcflx' in e and reason in e for e in errors),errors
    _,_,n,_=scene()
    n=replace(n,**physics,ocean_initial=None,use_noahmp=lsm==4)
    with pytest.raises(ValueError,match=reason):
        _resolve_operational_suite(n)
    # Zero selector must not trip this new admission rule.
    from gpuwrf.io.ocean_options import validate_options
    validate_options({'isftcflx':0},sfclay=sf,lsm=lsm,urban=urban,pbl=pbl)


@pytest.mark.parametrize('group',['ocean','hurricane_ustm','lh'])
@pytest.mark.parametrize('mutation',['float64','shape_1x1'])
@pytest.mark.parametrize('supply_namelist',[False,True])
def test_raw_netcdf_ocean_precision_and_grid_fail_closed(tmp_path,monkeypatch,group,mutation,supply_namelist):
    from gpuwrf.io import wrfrst_netcdf as rst
    grid,_,n,carry=scene()
    n=replace(n,mp_physics=0)
    original=rst._write_optional_carry_variables
    def malformed(x):
        raw=np.asarray(x)
        return raw.astype(np.float64) if mutation=='float64' else raw[:1,:1]
    def emit_malformed(dataset,healthy,dimensions):
        # Only the extension emitter is changed: all normal writer validation,
        # metadata, State fields and actual disk/read boundaries remain intact.
        if group=='ocean':
            changed=healthy.replace(ocean=jax.tree.map(malformed,healthy.ocean))
        elif group=='hurricane_ustm':
            changed=healthy.replace(hurricane_ustm=malformed(healthy.hurricane_ustm))
        else:
            changed=healthy.replace(history_diagnostics=healthy.history_diagnostics._replace(
                lh=malformed(healthy.history_diagnostics.lh)))
        return original(dataset,changed,dimensions)
    monkeypatch.setattr(rst,'_write_optional_carry_variables',emit_malformed)
    path=tmp_path/'malformed.nc';stamp=datetime(2026,7,25,tzinfo=timezone.utc)
    rst.write_wrfrst_carry(carry,grid,n,path,step_index=5,run_start=stamp,valid_time=stamp,lead_hours=0.)
    with Dataset(path) as ds:
        if group=='lh':
            # History is a positional NamedTuple in the stored pytree.
            index=carry.history_diagnostics._fields.index('lh')
            variable=rst._optional_variable_name('history_diagnostics',f'leaf_{index:03d}')
        else:
            variable=rst._optional_variable_name(group,'leaf_000')
        raw=np.asarray(ds[variable][...])
        assert raw.dtype==np.float64 if mutation=='float64' else raw.shape==(1,1)
    with jax.enable_x64(False):
        assert not jax.config.x64_enabled
        with pytest.raises(ValueError,match='E78'):
            rst.read_wrfrst_carry(path,namelist=n if supply_namelist else None)


def test_healthy_raw_ocean_restart_with_x64_disabled(tmp_path):
    from gpuwrf.io import wrfrst_netcdf as rst
    grid,_,n,carry=scene();n=replace(n,mp_physics=0)
    path=tmp_path/'healthy.nc';stamp=datetime(2026,7,25,tzinfo=timezone.utc)
    rst.write_wrfrst_carry(carry,grid,n,path,step_index=5,run_start=stamp,valid_time=stamp,lead_hours=0.)
    with jax.enable_x64(False):
        restored,_=rst.read_wrfrst_carry(path,namelist=n)
    for a,b in zip(jax.tree.leaves((carry.ocean,carry.hurricane_ustm,carry.history_diagnostics)),
                   jax.tree.leaves((restored.ocean,restored.hurricane_ustm,restored.history_diagnostics)),strict=True):
        np.testing.assert_array_equal(a,b)
    assert restored.ocean.tml.shape==restored.state.t_skin.shape==(2,3)
