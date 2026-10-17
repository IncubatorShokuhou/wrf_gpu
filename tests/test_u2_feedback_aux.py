"""Registry mapping uses saved pristine-WRF operands/results and fail-closed IO."""
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling.boundary_feedback import build_feedback_weights,point_feedback_weights,StateFeedbackWeights
from gpuwrf.coupling.feedback_aux import feedback_group,radiation_for_sums,refresh_fluxes,refresh_surface,FLUXES
from gpuwrf.io.nesting_options import admit_feedback,require_feedback_restart
from gpuwrf.runtime.history_accumulators import PackedFields

def test_aux_average_point_and_held_mapping_from_pristine():
    here=Path(__file__).parent/'fixtures'
    with np.load(here/'u2_feedback_oracle.npz') as f:
        p,c,area=(jnp.asarray(f[k][0]) for k in ('p0','c0','o0_0'))
    mass=build_feedback_weights(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,
        parent_we=22,parent_sn=20,child_we=36,child_sn=30)
    weights=StateFeedbackWeights(mass,mass,mass,smooth_option=0,point=point_feedback_weights(mass))
    with np.load(here/'u2_point_feedback_oracle.npz') as f:
        pp,cc,point=(jnp.asarray(f[k]) for k in ('p0','c0','o0'))
    parent=PackedFields.pack({'SWDNB':p,'TH2':p,'ACSWDNB':p,'TV':pp,'T2':p,'RTHRATEN':p})
    child=PackedFields.pack({'SWDNB':c,'TH2':c,'ACSWDNB':c,'TV':cc,'T2':c,'RTHRATEN':c})
    got=feedback_group(parent,child,weights)
    for name in ('SWDNB','TH2','ACSWDNB','T2'):np.testing.assert_array_equal(got[name],area)
    np.testing.assert_array_equal(got['TV'],point)
    np.testing.assert_array_equal(got['RTHRATEN'],p)


@pytest.mark.parametrize('names,nl',[
    (('d01','d02','d03'),{}),
    (('d01','d02'),{'physics':{'mp_physics':40}}),
    (('d01','d02'),{'physics':{'sst_update':1}}),
    (('d01','d02'),{'physics':{'sf_surface_physics':2}}),
    (('d01','d02'),{'domains':{'parent_grid_ratio':[1,4]}}),
    (('d01','d02'),{'domains':{'parent_grid_ratio':[1,1]}}),
    (('d01','d02'),{'physics':{'swint_opt':1}}),
    (('d01','d02'),{'physics':{'prec_acc_dt':.5}}),
    (('d01','d02'),{'physics':{'do_radar_ref':1}}),
    (('d01','d02'),{'time_control':{'nwp_diagnostics':1}}),
])
def test_unqualified_feedback_refused(names,nl):
    with pytest.raises(ValueError,match='two-way'):admit_feedback(nl,names,feedback=True)
    admit_feedback(nl,names,feedback=False)


def test_feedback_restart_requires_both_independent_groups():
    for base,history in ((None,None),(SimpleNamespace(alb=None),{}),(SimpleNamespace(alb=1),None)):
        with pytest.raises(ValueError,match='E78'):
            require_feedback_restart({'d01':SimpleNamespace(base_state=base,feedback_history=history)})
    shape=(2,3,4)
    valid=SimpleNamespace(state=SimpleNamespace(qv=np.zeros(shape),t_skin=np.zeros(shape[1:])),
        base_state=SimpleNamespace(alb=np.ones(shape,np.float32)),
        feedback_history=PackedFields(('TH2',*FLUXES),np.ones((17,*shape[1:]),np.float32)))
    require_feedback_restart({'d01':valid})


def test_independent_base_and_aux_restart_both_formats(tmp_path):
    import jax
    from gpuwrf.contracts.state import BaseState
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid,build_neutral_state
    grid=build_flat_grid(nx=5,ny=4,nz=3,dx_m=3000.)
    state=build_neutral_state(grid)
    base=BaseState(pb=state.p_total-state.p_perturbation,phb=state.ph_total-state.ph_perturbation,
        mub=state.mu_total-state.mu_perturbation,t0=jnp.zeros_like(state.theta),
        theta_base=jnp.full_like(state.theta,300.),alb=jnp.full(state.theta.shape,1.2345,jnp.float32))
    aux=PackedFields.pack({'TH2':jnp.full(state.t_skin.shape,299.123,jnp.float32),
                          **{name:jnp.full(state.t_skin.shape,451.234,jnp.float32) for name in FLUXES}})
    carry=initial_operational_carry(state).replace(base_state=base,feedback_history=aux)
    nl=OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=None,dt_s=18,acoustic_substeps=4,gwd_opt=0)
    write_restart(carry,nl,grid,3,tmp_path/'carry.pkl')
    write_wrfrst_carry(carry,grid,{},tmp_path/'carry.nc',
        valid_time='2026-07-25_18:00:54',run_start='2026-07-25_18:00:00',step_index=3)
    for restored in (read_restart(tmp_path/'carry.pkl')[0],read_wrfrst_carry(tmp_path/'carry.nc')[0]):
        require_feedback_restart({'d01':restored})
        for before,after in zip(jax.tree.leaves(carry),jax.tree.leaves(restored),strict=True):
            assert np.asarray(before).dtype==np.asarray(after).dtype
            assert np.asarray(before).tobytes()==np.asarray(after).tobytes()


@pytest.mark.parametrize('own_step', [0, 1])
@pytest.mark.parametrize('enabled', [False, True])
def test_writer_retains_pristine_upfed_aux_at_lead_zero(tmp_path,monkeypatch,own_step,enabled):
    """Actual output callback must retain independently computed WRF feedback."""
    from datetime import datetime
    from gpuwrf.integration import nested_pipeline as pipeline
    fixture=Path(__file__).parent/'fixtures/u2_feedback_oracle.npz'
    with np.load(fixture) as original:
        parent=np.array(original['p0'][0])
        expected=np.array(original['o0_0'][0])
    assert np.any(parent!=expected)  # Active pristine copy_fcn, not zeros.
    writer=object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.run_start=datetime(2026,2,28)
    writer.bundles={'d01':SimpleNamespace(grid=SimpleNamespace(),
        namelist=SimpleNamespace(use_noahmp=False))}
    writer.writer_diagnostics={}
    writer._initial_surface_fields=lambda _:dict(TH2=parent,SWDNB=parent)
    writer._surface_diagnostics_for_output=lambda *a,**kw:dict(TH2=parent,SWDNB=parent)
    writer._merge_output_diagnostics=lambda _,surface:dict(surface)
    writer.domain_authorities={'d01':None}
    writer._variable_subset=None
    writer._full_variable_set=True
    writer.census_io_ledger=None
    # This is an output-wiring gate; no synthetic atmospheric timestep is run.
    monkeypatch.delenv('GPUWRF_DYN_RK_FP32',raising=False)
    monkeypatch.setattr(pipeline,'assert_state_finite_at_boundary',lambda *a,**kw:None)
    captured={}
    class Captured(Exception):pass
    def capture(*a,**kw):
        captured.update(kw['diagnostics'])
        raise Captured
    monkeypatch.setattr(pipeline,'prepare_wrfout_payload',capture)
    history=PackedFields.pack({'TH2':jnp.asarray(expected),'SWDNB':jnp.asarray(expected)}) if enabled else None
    carry=SimpleNamespace(state=SimpleNamespace(),feedback_history=history)
    with pytest.raises(Captured):
        writer._materialize_and_submit(name='d01',own_step=own_step,carry=carry,
            valid_time=writer.run_start,lead_seconds=own_step*54,lead_hours=own_step*54/3600,
            path=tmp_path/'history.nc')
    for name in ('TH2','SWDNB'):
        np.testing.assert_array_equal(captured[name],expected if enabled else parent)
