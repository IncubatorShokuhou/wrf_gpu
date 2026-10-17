"""Caller operands, not forecast physics: selected ALBEDO/EMISS reach Goddard."""
from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import goddard_radiation as goddard
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.io.namelist_check import UnsupportedSchemeError,validate_operational_namelist
from gpuwrf.runtime import operational_mode as om
from gpuwrf.validation.tier2 import make_ideal_grid
from test_goddard_operational import case
from test_v0110_wrfrst_netcdf import _state


@pytest.fixture
def real_ruc_case(case):
    # Critic R06's actual build_ruc_bundles path, with matching atmosphere shape.
    from test_v034_ruclsm import _bundles
    static,land=_bundles()
    grid=make_ideal_grid(nz=5,ny=2,nx=3);z=jnp.zeros((2,3),jnp.float32)
    rad=pc.RRTMGRadiationStatic(jnp.full_like(z,28),jnp.full_like(z,-16),z,z,z,z,jnp.ones_like(z))
    nl=replace(case[1],grid=grid,metrics=grid.metrics,radiation_static=rad)
    carry=om._initial_carry_for_run(_state(grid),nl)
    nl=replace(nl,sf_surface_physics=3,sf_sfclay_physics=5,bl_pbl_physics=5,
               ruc_static=static,ruc_land=land)
    carry=carry.replace(ruc_land=land,state=carry.state.replace(xland=jnp.asarray(static.xland)))
    return grid,nl,carry


def test_real_ruc_pair_is_admitted(real_ruc_case):
    _,nl,_=real_ruc_case
    validate_operational_namelist({'physics':{'mp_physics':8,'use_mp_re':1,
        'ra_lw_physics':5,'ra_sw_physics':5,'sf_surface_physics':3,
        'sf_sfclay_physics':5,'bl_pbl_physics':5,'cu_physics':0}})
    om._resolve_operational_suite(nl)


def capture_refresh(monkeypatch,carry,nl):
    seen=[]
    def capture(state,grid,**kwargs):
        seen.append(kwargs['land_state'])
        return jnp.zeros_like(carry.rthraten),carry.radiation_diagnostics
    monkeypatch.setattr(goddard,'paired_radiation',capture)
    refreshed=om._refresh_rrtmg_driver(carry,nl,600.,True,None)
    held=om._refresh_rrtmg_driver(refreshed,nl,618.,False,None)
    assert len(seen)==1 and held.radiation_diagnostics is refreshed.radiation_diagnostics
    return seen[0]


def test_real_ruc_surface_operands_reach_refresh(real_ruc_case,monkeypatch):
    _,nl,carry=real_ruc_case
    surface=capture_refresh(monkeypatch,carry,nl)
    assert surface is not None,'Goddard dropped the admitted RUC ALBEDO/EMISS view'
    assert surface.albedo is carry.ruc_land.alb and surface.emiss is carry.ruc_land.emiss
    actual=pc._surface_radiation_properties(carry.state,surface)
    fallback=pc._surface_radiation_properties(carry.state,None)
    land=np.asarray(carry.state.xland)<1.5
    assert land.any()
    for a,expected in zip(actual,(carry.ruc_land.alb,carry.ruc_land.emiss),strict=True):
        np.testing.assert_array_equal(np.asarray(a)[land],np.asarray(expected)[land])
    assert any(np.any(np.asarray(a)[land]!=np.asarray(b)[land]) for a,b in zip(actual,fallback))


class Surface(NamedTuple):
    albedo:object
    emiss:object
    t_skin:object
    def replace(self,**kwargs):return self._replace(**kwargs)


@pytest.mark.parametrize('urban',[0,1])
def test_noahmp_and_slucm_merged_surface_is_forwarded(case,monkeypatch,urban):
    _,nl,carry=case
    # Both urban_slucm_hook and Noah-MP update these same resident land fields.
    land=Surface(jnp.full_like(carry.state.t_skin,.41),jnp.full_like(carry.state.t_skin,.93),carry.state.t_skin)
    nl=replace(nl,sf_surface_physics=4,use_noahmp=True,sf_urban_physics=urban)
    assert capture_refresh(monkeypatch,carry.replace(noahmp_land=land),nl) is land


@pytest.mark.parametrize('key,value',[('sf_surface_physics',1),('sf_surface_physics',2),
    ('sf_surface_physics',7),('sf_urban_physics',2),('sf_urban_physics',3),('sf_lake_physics',1)])
def test_unqualified_surface_pairs_refused_before_execution(case,key,value):
    options={'ra_lw_physics':5,'ra_sw_physics':5,'mp_physics':8,'use_mp_re':1,key:value}
    with pytest.raises(UnsupportedSchemeError,match='Goddard'):
        validate_operational_namelist({'physics':options})
    with pytest.raises(ValueError,match='Goddard'):
        om._resolve_operational_suite(replace(case[1],**{key:value}))
