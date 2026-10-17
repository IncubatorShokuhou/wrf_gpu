"""Original cal_cldfra1 flags and ordered transient convective cloud mediation."""
import json,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state


def test_flags_are_the_original_cal_cldfra1_branch_flags():
    from gpuwrf.physics.cloud_fraction import wrf_resolved_cloud_fraction
    path=Path(__file__).resolve().parents[3]/'proofs/t_camuw/cloud_oracle.json'
    cases=[c for c in json.loads(path.read_text())['cases'] if c['input'][0]==8]
    a=np.asarray([c['input'][1:] for c in cases],np.float32)
    t,p,qv,qc,qi,qs=a.T
    _,flags=jax.jit(lambda *a:wrf_resolved_cloud_fraction(*a,mp_physics=8,return_flags=True))(qv,qc,qi,qs,t,p)
    np.testing.assert_array_equal(np.asarray(flags),np.asarray([c['branch'] for c in cases],np.int32))


@pytest.mark.parametrize('scheme,lead',[(11,0.),(11,60.),(10,60.)])
def test_original_merge_order_reaches_both_radiation_columns_without_state_mutation(monkeypatch,scheme,lead):
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.coupling.cuplus import initial_carry
    grid=_grid(nz=4);state=_state(grid)
    zero=jnp.zeros_like(state.qc)
    state=state.replace(qc=zero,qi=zero,qs=zero,qg=zero,
        cldfra_bl=jnp.full_like(zero,.5),qc_bl=jnp.full_like(zero,1e-4),qi_bl=jnp.full_like(zero,1e-5))
    held=dict(initial_carry(state,scheme).held)
    if scheme==11:
        for key,value in (('CLDFRA_DP_KF',.2),('CLDFRA_SH_KF',.15),('QC_KF',.002),('QI_KF',.0003)):
            held[key]=jnp.full_like(held[key],value)
        f=np.float32;conv=f(f(.2)+f(.15))
        expected_cf=np.full((3,3,4),conv if lead==0 else f(.5),f)
        expected_qc=np.full((3,3,4),f(.002)*conv,f)
        expected_qi=np.full((3,3,4),f(.0003)*conv,f)
    else:
        held.update(CLDFRA_CUP=jnp.full_like(held['CLDFRA_CUP'],.2),ISHALL=jnp.ones_like(held['ISHALL']),
                    CUBOT=jnp.full_like(held['CUBOT'],2.),CUTOP=jnp.full_like(held['CUTOP'],3.))
        expected_cf=np.zeros((3,3,4),np.float32);expected_cf[...,1:3]=np.float32(.2)
        expected_qc=np.full((3,3,4),1e-4,np.float32);expected_qc[...,1:3]=np.float32(.2)*np.float32(1e-3)
        expected_qi=np.full((3,3,4),1e-5,np.float32)
    monkeypatch.setattr(pc,'sgs_cloud_enabled',lambda:True)
    sw,lw,*extra=pc._rrtmg_column_inputs(state,grid,time_utc='2023-01-15T04:00:00Z',
        lead_seconds=lead,use_mp_re=0,cuplus_clouds=held)
    for result in (sw,lw):
        np.testing.assert_allclose(np.asarray(result.cloud_fraction),expected_cf,rtol=2*np.finfo(np.float32).eps,atol=0.)
        np.testing.assert_allclose(np.asarray(result.qc),expected_qc,rtol=2*np.finfo(np.float32).eps,atol=0.)
        np.testing.assert_allclose(np.asarray(result.qi),expected_qi,rtol=2*np.finfo(np.float32).eps,atol=0.)
    np.testing.assert_array_equal(np.asarray(state.qc),0.)
    np.testing.assert_array_equal(np.asarray(state.qi),0.)
    if scheme==10:np.testing.assert_array_equal(np.asarray(extra[-1]),np.asarray(held['CLDFRA_CUP']))
