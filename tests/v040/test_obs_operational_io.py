"""Observation admission, both loaders and restart completeness (regression)."""
from dataclasses import replace
import inspect,pickle
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.assimilation.obs_driver import ObservationInputs,initial_observation_carry
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.io.obs_namelist import DEFAULTS,observation_options,observation_namelist_errors
from gpuwrf.io.restart_schema import validate_observations
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_m7_restart_checkpoint_roundtrip import _synthetic_state,_synthetic_tendencies


def _inputs():
    opts=dict(DEFAULTS,obs_nudge_opt=1,max_obs=100,obs_rinxy=30.,obs_twindo=.5,fdda_end=60.,
              obs_nudge_wind=1,obs_nudge_temp=1,obs_nudge_mois=1,
              obs_coef_wind=.0003,obs_coef_temp=.0003,obs_coef_mois=.0003)
    return ObservationInputs(jnp.zeros(1,jnp.float32),jnp.zeros(1,jnp.int32),
        jnp.full(1,2147483647,jnp.int32),jnp.full(1,3.,jnp.float32),jnp.full(1,3.,jnp.float32),
        jnp.ones((1,1),jnp.float32),jnp.ones((1,1,1,6),jnp.float32),
        jnp.zeros((1,1),jnp.int32),jnp.ones(1,jnp.int32),tuple(sorted(opts.items())),1,'ab'*32)


@pytest.mark.parametrize('kind',['pickle','netcdf','checkpoint'])
def test_observation_carry_is_exact_and_required_on_restart(tmp_path,kind):
    grid=GridSpec.canary_3km_template();state=_synthetic_state(grid);inputs=_inputs()
    nl=OperationalNamelist(grid=grid,tendencies=_synthetic_tendencies(grid),metrics=grid.metrics,
                           use_mp_re=0,observation_nudging=inputs)
    obs=initial_observation_carry(inputs,state.theta.shape)
    obs=obs.replace(savwt_u=jnp.full_like(obs.savwt_u,1.25),savwt_q=jnp.full_like(obs.savwt_q,-.001))
    carry=initial_operational_carry(state).replace(obs=obs)
    path=tmp_path/'restart'
    if kind=='pickle':
        from gpuwrf.io.restart import write_restart,read_restart
        write=lambda c:write_restart(c,nl,grid,2,path)
        read=lambda:read_restart(path)[0]
    elif kind=='netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        write=lambda c:write_wrfrst_carry(c,grid,nl,path,valid_time='2023-01-15_00:00:36',
                                         run_start='2023-01-15_00:00:00',step_index=2)
        read=lambda:read_wrfrst_carry(path,namelist=nl)[0]
    else:
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
        write=lambda c:write_checkpoint(state,nl,grid,2,path,runtime_state=c)
        read=lambda:read_checkpoint_with_runtime_state(path)[-1]
    with pytest.raises(ValueError,match='observation.*E78'):
        write(carry.replace(obs=None))
    write(carry);restored=read()
    for a,b in zip(jax.tree.leaves(obs),jax.tree.leaves(restored.obs),strict=True):
        assert np.asarray(a).dtype==np.asarray(b).dtype
        assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    with pytest.raises(ValueError,match='fingerprint changed'):
        validate_observations(restored.obs,replace(nl,observation_nudging=replace(inputs,fingerprint='cd'*32)))


def test_cli_admission_and_omitted_child_values():
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    from gpuwrf.io.namelist_check import validate_operational_namelist
    opts=dict(_inputs().options)
    config=dict(physics=dict(mp_physics=8,bl_pbl_physics=5,sf_sfclay_physics=5,
                           sf_surface_physics=4,ra_sw_physics=4,ra_lw_physics=4,cu_physics=1),
                dynamics=dict(diff_opt=1,km_opt=4),fdda=opts)
    validate_operational_namelist(config);require_namelist_honoured(config,('d01',))
    assert observation_options(config,'d02')['obs_nudge_opt']==0
    assert classify_scheme('obs_nudge_opt',1).status==SupportStatus.IMPLEMENTED
    assert 'GPU/coupled-forecast qualification pending' in classify_scheme('obs_nudge_opt',1).reason
    assert observation_namelist_errors(dict(config,fdda=dict(opts,obs_twindo=float('nan'))),'d01')
    assert observation_namelist_errors(dict(config,fdda=dict(opts,obs_ionf=1.5)),'d01')
    assert observation_namelist_errors(dict(config,physics=dict(config['physics'],bl_pbl_physics=1)),'d01')
    assert observation_namelist_errors(dict(config,fdda=dict(opts,obs_nudge_opt=[1,1])),'d02')


def test_both_pipeline_consumers_bind_the_loader_and_separate_grid_fdda():
    from gpuwrf.integration import nested_pipeline,daily_pipeline
    from gpuwrf.coupling.obs_rk import observation_nudging_active
    # Nested execution is independently exercised by the real Swiss smoke.
    for fn in (nested_pipeline._load_domains,daily_pipeline._build_real_case):
        source=inspect.getsource(fn)
        assert 'load_observation_nudging(' in source and 'observation_nudging=' in source
    assert observation_nudging_active(SimpleNamespace(observation_nudging=_inputs(),data_assimilation=object()))
    assert not observation_nudging_active(SimpleNamespace(observation_nudging=None,data_assimilation=object()))


def test_changed_stream_checksum_reuses_executable_shape():
    a=_inputs();b=replace(a,fingerprint='cd'*32)
    assert jax.tree.structure(a)==jax.tree.structure(b)
    f=jax.jit(lambda inputs:inputs.times+1)
    np.testing.assert_array_equal(np.asarray(f(a)),np.asarray(f(b)))
    assert f._cache_size()==1
