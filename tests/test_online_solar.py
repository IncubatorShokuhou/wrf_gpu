"""Original-WRF Solar operator sequence and exact REAL carry contracts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import solar
from gpuwrf.diagnostics.solar import (
    SolarCarry,SolarInterpolationState,SolarSurfaceFluxes,history_fields,
    reconstruct_swdown,seed_solar,solar_step,validate_restart_solar,
)

FIXTURE=Path(__file__).resolve().parents[1]/"proofs/u2_diag/solar_oracle.json"
SURFACE_NAMES=("SWDOWN","SWDDIR","SWDDNI","SWDDIF","GSW")
PARAMETER_NAMES=("swddir_ref","bb","Bx","swdown_ref","gg","Gx","coszen_ref")


def _check_sequence(backend):
    fixture=json.loads(FIXTURE.read_text())
    assert fixture["precision"]=="WRF REAL=4; no DOUBLE"
    assert len(fixture["source_sha256"])==len(fixture["block_sha256"])==64
    assert all(value>0 for value in fixture["regimes"].values())
    xp=np if backend=="numpy" else jnp
    operator=lambda state,operands,run,first:solar_step(state,*operands,run_radiation=run,first_step=first,xp=xp)
    if backend=="jit":
        operator=jax.jit(operator)
    for case in fixture["cases"]:
        state=SolarInterpolationState(*(xp.asarray(a,xp.float32) for a in case["initial"]))
        for operands,run,first,expected in zip(case["inputs"],case["run_radiation"],case["first_step"],case["expected"],strict=True):
            previous=state
            with np.errstate(divide="ignore",invalid="ignore"):
                state,fluxes=operator(state,xp.asarray(operands,xp.float32),xp.asarray(run) if backend=="jit" else run,first)
            actual=np.asarray((*state,*fluxes))
            assert actual.dtype==np.float32 and np.isfinite(actual).all()
            np.testing.assert_allclose(actual,np.asarray(expected,np.float32),rtol=2e-6,atol=2e-5)
            if not run:
                for before,after in zip(previous,state,strict=True):
                    np.testing.assert_array_equal(before,after)
            assert set(history_fields(fluxes))=={"SWDOWN","SWDDIR","SWDDNI","SWDDIF","GSW"}


@pytest.mark.parametrize("backend",["numpy","jax","jit"])
def test_original_wrf_solar_sequence(backend):
    _check_sequence(backend)


def test_original_source_fixture_provenance():
    fixture=json.loads(FIXTURE.read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()=="df70cbebad9a4d4050fc9ac8e2810f4ee4f07195c234f6fef7cec94a1efa7742"
    assert fixture["source_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert fixture["source_sha256"]=="4f0177189eb616c9220fe2022186dd6cc310666ca29977557dfe53d4409d733b"
    assert fixture["provenance"]=="verbatim unmodified WRF update/interp routines; no gpuwrf/JAX import"
    assert fixture["precision"]=="WRF REAL=4; no DOUBLE"
    assert fixture["output_order"]==[*PARAMETER_NAMES,*(name.lower() for name in SURFACE_NAMES)]
    assert "-ffp-contract=off" in fixture["compiler"] and "-fno-fast-math" in fixture["compiler"]
    assert np.asarray([case["expected"] for case in fixture["cases"]]).shape==(2,11,12,16)
    assert fixture["regimes"]["negative_diffuse"]==24
    assert np.min(np.asarray(fixture["cases"][0]["expected"])[:,10])<0


def test_no_parameter_update_mutant_is_killed(monkeypatch):
    # Genuine deletion of the source update. The unmodified original-WRF
    # control passes; the mutated operator must fail a numerical comparison.
    _check_sequence("numpy")
    monkeypatch.setattr(solar,"update_parameters",lambda state,*args,**kwargs:state)
    with pytest.raises(AssertionError,match="Not equal to tolerance"):
        _check_sequence("numpy")


@pytest.fixture
def group():
    return seed_solar(SimpleNamespace(t_skin=np.zeros((3,4),np.float32)),xp=np)


def test_solar_carry_seed_and_public_fields(group):
    assert isinstance(group,SolarCarry)
    assert group.parameters._fields==PARAMETER_NAMES
    assert group.fluxes._fields==tuple(name.lower() for name in SURFACE_NAMES)
    assert len(jax.tree.leaves(group))==12
    assert sum(value.nbytes for value in jax.tree.leaves(group))==48*3*4
    for value in jax.tree.leaves(group):
        assert value.shape==(3,4) and value.dtype==np.float32 and not value.any()
    assert tuple(history_fields(group))==SURFACE_NAMES
    assert tuple(history_fields(group.fluxes))==SURFACE_NAMES
    assert history_fields(None)=={}
    with pytest.raises(TypeError,match="Solar history"):
        history_fields(group.parameters)
    validate_restart_solar(group,True,1,(3,4))
    validate_restart_solar(None,False,None,(3,4))


@pytest.mark.parametrize("member,name",[("parameters",name) for name in PARAMETER_NAMES]+
                         [("fluxes",name.lower()) for name in SURFACE_NAMES])
@pytest.mark.parametrize("damage",["missing","precision","shape"])
def test_restart_rejects_each_damaged_real_leaf(group,member,name,damage):
    obj=getattr(group,member)
    leaf=getattr(obj,name)
    damaged=(None if damage=="missing" else leaf.astype(np.float64) if damage=="precision" else leaf[:,:-1])
    group=group._replace(**{member:obj._replace(**{name:damaged})})
    with pytest.raises(ValueError,match="Solar interpolation.*E78"):
        validate_restart_solar(group,True,1,(3,4))


def test_restart_missing_schema_groups_rank_and_selection_refused(group):
    wrong_rank=SolarCarry(SolarInterpolationState(*(a.ravel() for a in group.parameters)),
                          SolarSurfaceFluxes(*(a.ravel() for a in group.fluxes)))
    for value,active,version,shape in ((None,True,1,(3,4)),(group,True,None,(3,4)),
        (group,True,0,(3,4)),(group,False,1,(3,4)),(group._replace(parameters=None),True,1,(3,4)),
        (group._replace(fluxes=None),True,1,(3,4)),(wrong_rank,True,1,None),(group,True,1,(3,5))):
        with pytest.raises(ValueError,match="Solar.*E78"):
            validate_restart_solar(value,active,version,shape)


@pytest.mark.parametrize("compiled",[False,True])
def test_original_real_gsw_reconstruction_boundary(compiled):
    # Literal independent REAL assignment and subtraction/division from
    # pristine radiation_driver:3000; storage-fp64 operands must first round.
    gsw=np.array([676.6251,11.1000001,0.],np.float64)
    albedo=np.array([.20000001,.33,.45],np.float64)
    expected=np.asarray(gsw,np.float32)/(np.float32(1)-np.asarray(albedo,np.float32))
    operator=jax.jit(reconstruct_swdown) if compiled else reconstruct_swdown
    actual=operator(jnp.asarray(gsw),jnp.asarray(albedo))
    assert actual.dtype==jnp.float32
    np.testing.assert_array_max_ulp(actual,expected,maxulp=2 if compiled else 0)


def test_reconstruction_has_no_invented_denominator_floor():
    with np.errstate(divide="ignore",invalid="ignore"):
        result=reconstruct_swdown(np.array([1,0],np.float32),np.ones(2,np.float32),xp=np)
    assert np.isinf(result[0]) and np.isnan(result[1])
