"""Cycling control/initialization contracts, separate from forecast fidelity."""
from dataclasses import dataclass, replace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as couplers
from gpuwrf.io.cycling_init import cycling_enabled, admit_cycling_input
from gpuwrf.io.namelist_binding import bound_options, namelist_honour_report, NamelistNotHonouredError
from gpuwrf.io.restart_schema import validate_cycling_restart
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.io.netcdf_lock import Dataset
from test_atmosphere_cycle import fixture_case, run_cycle


@dataclass(frozen=True)
class State:
    qke: object
    qsq: object = None
    qc_bl: object = None
    qi_bl: object = None
    cldfra_bl: object = None
    el_pbl: object = None

    def replace(self, **updates):
        return replace(self, **updates)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_columnwise_zero_cold_boundary_and_compiled_first_flag(monkeypatch, dtype):
    def seed(s,g,*,initialize_qke,return_moments):
        assert return_moments
        mask=initialize_qke[None]
        return (jnp.where(mask,jnp.full_like(s.qke,7),jnp.zeros_like(s.qke)),
                s.el_pbl+jnp.full_like(s.qke,3),
                jnp.where(mask,s.qsq+jnp.full_like(s.qke,.25),jnp.zeros_like(s.qke)))
    monkeypatch.setattr(couplers, "mynn_coldstart_qke_from_state", seed)
    boundary = np.float32(.0002)
    values = np.array([0, np.nextafter(boundary, np.float32(0)), boundary, .5, 1.], np.float32)
    qke = jnp.broadcast_to(jnp.asarray(values, dtype=dtype)[None, None], (4, 1, 5))
    qsq = jnp.ones_like(qke)
    state = State(qke, qsq, qsq, qsq, qsq, qsq)
    def call(q, flag):
        out = couplers._mynn_state_with_first_call_qke(state.replace(qke=q), None, flag, cycling=True)
        return out.qke, out.qsq, out.qc_bl, out.qi_bl, out.cldfra_bl, out.el_pbl
    compiled = jax.jit(call)
    result = compiled(qke, jnp.asarray(True))
    expected = np.broadcast_to(np.array([7,7,0,0,0], np.float32)[None,None], (4,1,5))
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[1],np.broadcast_to(np.array([.25,.25,0,0,0])[None,None],(4,1,5)))
    for field in result[2:5]:
        np.testing.assert_array_equal(field, np.zeros((4,1,5)))
    np.testing.assert_array_equal(result[5],np.full((4,1,5),3))
    later = compiled(qke, jnp.asarray(False))
    np.testing.assert_array_equal(later[0], qke)
    for field in later[1:]:
        np.testing.assert_array_equal(field, qsq)


def test_restart_and_nonfirst_cycling_keep_state(monkeypatch):
    def bad_seed(*args):
        raise AssertionError("seed executed outside first-call cycling")
    monkeypatch.setattr(couplers, "mynn_coldstart_qke_from_state", bad_seed)
    state = State(jnp.full((3,2,2), 2.))
    assert couplers._mynn_state_with_first_call_qke(state,None,False,cycling=True) is state
    assert couplers._mynn_state_with_first_call_qke(state,None,True,restart=True,cycling=True) is state
    with pytest.raises(TypeError, match="lifecycle flag"):
        couplers._mynn_state_with_first_call_qke(state,None,True,cycling=jnp.asarray(True))


@pytest.mark.parametrize("cycling", [False, True])
def test_inactive_precipitation_and_ruc_packages_have_no_bucket_leaves(monkeypatch, cycling):
    from gpuwrf.validation.tier2 import make_ideal_grid
    from test_v0110_wrfrst_netcdf import _state
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    grid = make_ideal_grid(nx=5, ny=4, nz=4, dx_m=1000., dy_m=1000.)
    state = _state(grid)
    namelist = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18.,
        acoustic_substeps=4, use_noahmp=True, sf_surface_physics=4, run_physics=True,
        mp_physics=8, ra_sw_physics=0, ra_lw_physics=0, cu_physics=0,
        bl_pbl_physics=0, prec_acc_dt=0., cycling=cycling)
    # Calling the actual carry factory must leave disabled packages absent,
    # including in the cycling mode that triggered the upstream WRF overrun.
    carry = _initial_carry_for_run(state, namelist)
    assert carry.precipitation_buckets is None
    assert carry.ruc_land is None
    assert namelist.ruc_land is None


@pytest.mark.parametrize("shape",[(4,6),(2,4,3,5)])
def test_cycling_requires_state_level_y_x_layout(shape):
    with pytest.raises(ValueError,match="bottom_top, south_north, west_east"):
        couplers._mynn_state_with_first_call_qke(State(jnp.ones(shape)),None,True,cycling=True)


@pytest.mark.parametrize("shape",[(44,70,120),(44,117,267),(44,93,111),(9,2,3)])
def test_every_domain_state_contract_has_levels_first(shape):
    from gpuwrf.contracts.state import _state_field_shapes
    nz,ny,nx=shape
    grid=SimpleNamespace(nz=nz,ny=ny,nx=nx)
    assert _state_field_shapes(grid)["qke"] == (nz,ny,nx)


def test_native_cycling_does_not_blend_resolved_terrain(monkeypatch):
    from gpuwrf.integration import d02_replay as loader
    meta=SimpleNamespace(parent_id=1,parent_grid_ratio=3)
    run=SimpleNamespace(grid=lambda domain:meta)
    hgt=jnp.ones((2,3))
    grid=SimpleNamespace(terrain_height=hgt)
    metrics=object()
    parent=SimpleNamespace(metadata={"domain":"d01"})
    expected=tuple(object() for _ in range(5))
    seen=[]
    def base(*args,**kwargs):
        seen.append(kwargs["hgt"])
        return expected
    def bad_blend(*args,**kwargs):
        raise AssertionError("source WRF resolved terrain was blended a second time")
    monkeypatch.setattr(loader,"_wrf_start_domain_base_from_hgt",base)
    monkeypatch.setattr(loader,"sint_to_child_reference",bad_blend)
    result=loader._apply_live_nest_base_init(run,domain="d02",grid=grid,metrics=metrics,parent_case=parent,
             child_mub=None,child_phb=None,blend_from_real=False)
    assert result[:2] == (grid,metrics)
    # The base builder returns PB,MUB,PHB; the public live-init contract is
    # PB,PHB,MUB (3-D mass, 3-D interfaces, 2-D column mass).
    assert result[2:5] == (expected[0],expected[2],expected[1])
    assert seen == [hgt]
    assert result[-1]["save_topo_from_real"] == 0


def test_default_ingest_reexports_identical_host_math():
    from gpuwrf.integration import d02_replay as loader
    from gpuwrf.io import wrf_init_math as pure
    assert loader._WRF_INIT_LIBM32 is pure._WRF_INIT_LIBM32
    assert loader._WRFInitLibm32 is pure._WRFInitLibm32
    assert loader._wrf_start_domain_t_init_alb32 is pure._wrf_start_domain_t_init_alb32


def test_cycling_flag_aux_only_when_on():
    default = OperationalNamelist(grid=object(), tendencies=None, metrics=None)
    off = replace(default, cycling=False)
    on = replace(default, cycling=True)
    children, aux = default.tree_flatten()
    assert off.tree_flatten() == (children, aux)
    assert on.tree_flatten() == (children, aux + (("cycling", True),))
    restored = OperationalNamelist.tree_unflatten(aux, children)
    assert restored.cycling is False
    assert OperationalNamelist.tree_unflatten(on.tree_flatten()[1], children).cycling is True


def test_cycling_separates_actual_aot_keys_and_lowerings():
    from gpuwrf.runtime import aot_cheap_key as key
    from gpuwrf.validation.tier2 import make_ideal_grid
    grid=make_ideal_grid(nz=4,ny=3,nx=5)
    default=OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=None)
    @jax.jit
    def probe(value,namelist):
        return value+(1 if namelist.cycling else 0)
    value=jnp.ones((1,),jnp.float32)
    versions=[default,replace(default,cycling=False),replace(default,cycling=True)]
    static=[key.static_config_hash(n)for n in versions]
    actual=[key.cheap_key(probe,(value,n),{},n)for n in versions]
    hlo=[probe.lower(value,n).compiler_ir('hlo').as_hlo_text()for n in versions]
    assert static[0]==static[1]and static[0]!=static[2]
    assert actual[0]==actual[1]and actual[0]!=actual[2]
    assert hlo[0]==hlo[1]and hlo[0]!=hlo[2]


@pytest.mark.parametrize("value,expected", [(None,False),(False,False),(True,True),(0,False),(1,True)])
def test_global_logical_binding(value, expected):
    nml = {} if value is None else {"time_control":{"cycling":value}}
    assert cycling_enabled(nml) == expected
    bound = bound_options(nml, "d01")
    assert bool(bound.get("cycling", False)) == expected


@pytest.mark.parametrize("value", [2,-1,"true",1.0,[True,False],[]])
def test_invalid_global_logical_refuses(value):
    nml={"time_control":{"cycling":value}}
    with pytest.raises(ValueError):
        cycling_enabled(nml)
    if value != []:
        with pytest.raises(ValueError):
            bound_options(nml,"d01")


@pytest.mark.parametrize("group", [{"dfi_control":{"dfi_opt":3}}, {"physics":{"mp_physics":10}},
                                   {"physics":{"bl_pbl_physics":1}}])
def test_unsupported_cycling_combination_refuses_before_init(group):
    nml={"time_control":{"cycling":True},**group}
    with pytest.raises(NamelistNotHonouredError):
        bound_options(nml,"d01")
    errors,_=namelist_honour_report(nml,("d01",))
    assert errors


def test_use_input_w_honoured_for_cycling():
    nml={"time_control":{"cycling":True},"dynamics":{"use_input_w":True}}
    errors,_=namelist_honour_report(nml,("d01",))
    assert not errors, errors


@pytest.mark.parametrize("enabled,version,accept", [(False,None,True),(True,1,True),(True,None,False),
                                                   (False,1,False),(True,2,False)])
def test_restart_requires_cycling_marker(enabled, version, accept):
    nml=SimpleNamespace(cycling=enabled)
    if accept:
        validate_cycling_restart(nml,version)
    else:
        with pytest.raises(ValueError,match="cycling"):
            validate_cycling_restart(nml,version)


def test_marker_inventory_and_tampering_refusal(tmp_path):
    files=fixture_case(tmp_path)
    with pytest.raises(ValueError,match="provenance"):
        admit_cycling_input(files[1])
    output=tmp_path/"cycled"
    run_cycle(files,output)
    report=admit_cycling_input(output)
    assert report["required_namelist"] == {"time_control":{"cycling":True}}
    with Dataset(output,"r+") as ds:
        ds["QVAPOR"][0] *= .99
    with pytest.raises(ValueError,match="QVAPOR.*differs"):
        admit_cycling_input(output)


def test_public_loader_binds_lifecycle_and_refuses_missing_key(tmp_path, monkeypatch):
    from gpuwrf.integration import d02_replay as loader
    files=fixture_case(tmp_path)
    output=tmp_path/"cycled"
    run_cycle(files,output)
    nml={"physics":{"bl_pbl_physics":5},"time_control":{"cycling":True}}
    run=SimpleNamespace(path=tmp_path,namelist=nml,
          wrfinput_file=lambda domain:output, input_read_scope=lambda domain:nullcontext(),
          _read_dataset=lambda path:Dataset(path))
    monkeypatch.setattr(loader,"Gen2Run",lambda path:run)
    monkeypatch.setattr(loader,"_build_replay_case",lambda *a,**kw:kw)
    assert loader.build_replay_case(tmp_path,domain="d01",standalone=True)["cycling"] is True
    nml["time_control"]["cycling"]=False
    with pytest.raises(ValueError,match="cycling=.true"):
        loader.build_replay_case(tmp_path,domain="d01",standalone=True)
