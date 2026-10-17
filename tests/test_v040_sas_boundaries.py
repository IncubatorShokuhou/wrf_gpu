"""Original Fortran consumer ranges through actual operational SAS slots."""
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import test_v013_operational_smoke as smoke
from gpuwrf.runtime import operational_mode as om
from gpuwrf.coupling import native_sas_adapter, sas94_adapter, osas95_adapter
from gpuwrf.coupling.sas_consumers import apply_sas_scalar_rates, add_sas_momentum

FIX=Path(__file__).resolve().parents[1]/'proofs/v040/sas_family/fixtures'


def test_original_boundary_fixture_provenance():
    meta=json.loads((FIX/'boundary_manifest.json').read_text())
    assert meta['original_bodies_unchanged']
    assert hashlib.sha256((FIX/'boundary.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    assert meta['counts']['specified_scalar']==36
    assert meta['counts']['specified_u']==45 and meta['counts']['specified_v']==48


@pytest.mark.parametrize('option',(14,94,95,96))
@pytest.mark.parametrize('nested',(False,True))
def test_actual_operational_consumer_matches_original_and_keeps_full_holds(monkeypatch,option,nested):
    grid=smoke._grid(nz=3,ny=5,nx=6)
    grid=dataclasses.replace(grid,bc=dataclasses.replace(grid.bc,source='WRF'))
    state=smoke._convective_state(grid)
    nl=smoke._namelist(grid,dt_s=18.,mp_physics=0,bl_pbl_physics=0,sf_sfclay_physics=0,
                      cu_physics=option,ra_sw_physics=0,ra_lw_physics=0)
    nl=dataclasses.replace(nl,run_boundary=True,cumulus_cadence_steps=3,
                          boundary_config=SimpleNamespace(force_geopotential=not nested))
    assert om._acoustic_lateral_bc_flags(nl)==(False,not nested,nested)
    carry=om._initial_carry_for_run(state,nl)
    rates=list(carry.cumulus_tendencies)
    rates[0]=jnp.full_like(rates[0],1.e-3)
    rates[1]=jnp.full_like(rates[1],1.e-6)
    rates[6]=jnp.full_like(rates[6],.001)
    if option in (14,96):
        rates[7]=jnp.full_like(rates[7],.001);rates[8]=jnp.full_like(rates[8],.002)
        monkeypatch.setattr(native_sas_adapter,'native_sas_rates',lambda *a,**kw:tuple(rates))
    else:
        adapter=sas94_adapter if option==94 else osas95_adapter
        monkeypatch.setattr(adapter,'sas94_rates' if option==94 else 'osas95_rates',
                            lambda *a,**kw:tuple(rates[i] for i in (0,1,2,4,6)))
    with np.load(FIX/'boundary.npz') as original:
        for step in (1,2):
            result=om._physics_step_forcing(carry,nl,18.*(step-1),run_radiation=False)
            arm='nested' if nested else 'specified'
            for field,source in [('theta','scalar'),('qv','scalar')]+([('u','u'),('v','v')] if option in (14,96) else []):
                observed=np.asarray(getattr(result.state,field))!=np.asarray(getattr(carry.state,field))
                np.testing.assert_array_equal(observed,original[f'{arm}_{source}']!=0)
            # Source ownership masks consumers, never stored rates or rain.
            for got,want in zip(result.carry.cumulus_tendencies,rates):np.testing.assert_array_equal(got,want)
            np.testing.assert_allclose(result.state.rainc_acc-carry.state.rainc_acc,.018,rtol=1.e-6)
            carry=result.carry.replace(state=result.state)


@pytest.mark.parametrize('arm',('specified','nested','specified_periodic','nested_periodic'))
def test_periodic_override_and_all_scalar_consumers_against_original(arm):
    grid=smoke._grid(nz=3,ny=5,nx=6);state=smoke._convective_state(grid)
    rates=(jnp.ones_like(state.theta),)*6+(jnp.ones_like(state.rainc_acc),)
    periodic='periodic' in arm
    result=apply_sas_scalar_rates(state,rates,1.,entry_state=state,
                                  specified_or_nested=True,periodic_x=periodic)
    result=add_sas_momentum(result,jnp.ones_like(state.theta),jnp.ones_like(state.theta),
                             specified_or_nested=True,periodic_x=periodic,periodic_y=periodic)
    with np.load(FIX/'boundary.npz') as original:
        for field,source in (('theta','scalar'),('qv','scalar'),('qc','scalar'),('qi','scalar'),('u','u'),('v','v')):
            observed=np.asarray(getattr(result,field))!=np.asarray(getattr(state,field))
            np.testing.assert_array_equal(observed,original[f'{arm}_{source}']!=0)
