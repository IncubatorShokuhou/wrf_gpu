"""Public checkpoint requests and reduced readers must preserve land memory."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def checkpoint_case():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.coupling.sst_skin_hook import SSTSkinState
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.carbon import CarbonState
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    from gpuwrf.physics.noahmp.irrigation import seed_irrigation
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_noahmp_checkpoint_v2 import _namelist, _land
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template();state=_state(grid);shape=state.t_skin.shape
    row=lambda value:jnp.full(shape,value,jnp.float32)
    base=replace(_namelist(grid),use_noahmp=True,sf_surface_physics=4)
    carry=initial_operational_carry(state).replace(noahmp_land=_land(*shape))
    cases={
        'sst_skin':(replace(base,sst_skin=1),carry.replace(sst_skin=SSTSkinState(row(290.),row(291.),row(.2))),
                    lambda nl:replace(nl,sst_skin=0)),
        'noahmp_irrigation':(replace(base,noahmp_options=NoahMPOptions(opt_irr=1,opt_irrm=1)),
                    carry.replace(noahmp_irrigation=seed_irrigation(shape)._replace(irwatsi=row(.001))),
                    lambda nl:replace(nl,noahmp_options=replace(nl.noahmp_options,opt_irr=2))),
        'noahmp_carbon':(replace(base,noahmp_options=NoahMPOptions(dveg=2)),
                    carry.replace(noahmp_carbon=CarbonState(*(row(v) for v in (61.,501.,67.,501.,1001.,1001.)))),
                    lambda nl:replace(nl,noahmp_options=replace(nl.noahmp_options,dveg=5))),
        'noahmp_groundwater':(replace(base,noahmp_options=NoahMPOptions(opt_run=1,fsatmx=.38,timean=10.5)),
                    carry.replace(noahmp_groundwater=GroundwaterState(row(2.49),row(4901.),row(4901.))),
                    lambda nl:replace(nl,noahmp_options=replace(nl.noahmp_options,opt_run=2))),
        'noahmp_tile_drain':(replace(base,noahmp_options=NoahMPOptions(opt_tdrn=1,drain_layer_opt=4)),
                    carry.replace(noahmp_tile_drain=row(1.25)),
                    lambda nl:replace(nl,noahmp_options=replace(nl.noahmp_options,opt_tdrn=2))),
    }
    return state,grid,base,carry,cases


GROUPS=('sst_skin','noahmp_irrigation','noahmp_carbon','noahmp_groundwater','noahmp_tile_drain')


@pytest.mark.parametrize('group',GROUPS)
def test_full_reader_checks_requested_selection_and_roundtrips(tmp_path,checkpoint_case,group):
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    state,grid,_,_,cases=checkpoint_case;nl,carry,changed=cases[group];path=tmp_path/'land.checkpoint'
    write_checkpoint(state,nl,grid,3,path,runtime_state=carry,land_state=carry.noahmp_land)
    for requested in (None,nl):
        *_,restored=read_checkpoint_with_runtime_state(path,namelist=requested)
        for a,b in zip(jax.tree.leaves(carry),jax.tree.leaves(restored),strict=True):
            assert np.asarray(a).dtype==np.asarray(b).dtype
            assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    with pytest.raises(ValueError,match='restart.*selection|restart options differ'):
        read_checkpoint_with_runtime_state(path,namelist=changed(nl))


@pytest.mark.parametrize('group',GROUPS)
@pytest.mark.parametrize('reader_name',('read_checkpoint','read_checkpoint_with_land_state'))
def test_reduced_reader_refuses_discarding_active_groups(tmp_path,checkpoint_case,group,reader_name):
    from gpuwrf.runtime import checkpoint
    state,grid,_,_,cases=checkpoint_case;nl,carry,_=cases[group];path=tmp_path/'land.checkpoint'
    checkpoint.write_checkpoint(state,nl,grid,3,path,runtime_state=carry,land_state=carry.noahmp_land)
    with pytest.raises(ValueError,match=group+'.*read_checkpoint_with_runtime_state'):
        getattr(checkpoint,reader_name)(path,namelist=nl)


def test_stateless_selectors_differ_despite_identical_bound_tables(tmp_path,checkpoint_case):
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.io.noahmp_restart_identity import bound_table_identity
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    state,grid,base,carry,_=checkpoint_case
    nl=replace(base,noahmp_options=NoahMPOptions(dveg=8))
    requested=replace(nl,noahmp_options=NoahMPOptions(dveg=9))
    assert bound_table_identity(nl)==bound_table_identity(requested)
    path=tmp_path/'land.checkpoint'
    write_checkpoint(state,nl,grid,3,path,runtime_state=carry,land_state=carry.noahmp_land)
    read_checkpoint_with_runtime_state(path,namelist=nl)
    with pytest.raises(ValueError,match='restart options differ'):
        read_checkpoint_with_runtime_state(path,namelist=requested)


@pytest.mark.parametrize('reader_name',('read_checkpoint','read_checkpoint_with_land_state'))
@pytest.mark.parametrize('runtime', (False,True))
def test_reduced_legacy_and_inactive_reads_stay_available(tmp_path,checkpoint_case,reader_name,runtime):
    from gpuwrf.runtime import checkpoint
    state,grid,nl,carry,_=checkpoint_case;path=tmp_path/'inactive.checkpoint'
    checkpoint.write_checkpoint(state,nl,grid,3,path,
        **dict(runtime_state=carry,land_state=carry.noahmp_land) if runtime else {})
    restored=getattr(checkpoint,reader_name)(path,namelist=nl)
    assert restored[3]==3
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored[0]),strict=True):
        np.testing.assert_array_equal(a,b)


def test_restarted_input_lai_below_threshold_is_not_cold_initialized(tmp_path,checkpoint_case):
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    state,grid,base,carry,_=checkpoint_case
    nl=replace(base,noahmp_options=NoahMPOptions(dveg=7))
    carry=carry.replace(noahmp_land=carry.noahmp_land.replace(lai=jnp.full(state.t_skin.shape,.01)))
    path=tmp_path/'evolved.checkpoint'
    write_checkpoint(state,nl,grid,8,path,runtime_state=carry,land_state=carry.noahmp_land)
    *_,restored=read_checkpoint_with_runtime_state(path,namelist=nl)
    np.testing.assert_array_equal(restored.noahmp_land.lai,carry.noahmp_land.lai)
