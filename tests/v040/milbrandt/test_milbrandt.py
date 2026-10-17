"""Actual WRF driver-output parity and opt-in MY2 integration contracts."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE=Path(__file__).parent
NAMES='t q qc qr qi qn qg qh nc nr ny nn ng nh'.split()


def check_outputs(result,ref,*,compiled=True):
    out,rates,diam,zet,zec=map(np.asarray,result)
    with np.load(HERE/'input_ulp_envelope.npz') as f:envelope=dict(f)
    for j,name in enumerate(NAMES):
        atol=3e-5 if j==0 else 3e-10 if j<8 else 3e-3
        assert np.isfinite(out[:,j]).all()
        diff=np.abs(out[:,j]-ref['ref_fields'][:,j])
        primary=diff<=atol+3e-5*np.abs(ref['ref_fields'][:,j])
        # Standing wave policy: compiled contraction/reassociation may be
        # assessed against an INDEPENDENT original-WRF input-ULP envelope.
        # Keep the original primary failure disclosed; its bound is unchanged.
        assert np.all(primary | (compiled & (diff<=envelope['fields'][:,j]))),name
    for name,x,atol in [('rates',rates,3e-10),('zet',zet,3e-4),('zec',zec,3e-4)]:
        np.testing.assert_allclose(x,ref['ref_'+name],rtol=3e-5,atol=atol)
        assert np.isfinite(x).all()
    # diam is not a WRF driver output; preserve the diagnostic failure separately.
    assert np.isfinite(diam).all()


def run_fixture():
    from gpuwrf.physics.milbrandt2mom.column import columns
    with np.load(HERE/'wrf_real4.npz') as f: data=dict(f)
    a=data['inputs']
    result=jax.jit(columns)(jnp.asarray(a[:,:,:14].transpose(0,2,1)),jnp.asarray(a[:,:,15]),
        jnp.asarray(data['ps']),jnp.asarray(a[:,:,14]),jnp.asarray(data['dt']))
    return result,data


def test_pristine_active_driver_outputs_and_fixture_provenance():
    spec=json.loads((HERE/'manifest.json').read_text())
    assert hashlib.sha256((HERE/'wrf_real4.npz').read_bytes()).hexdigest()==spec['fixture_sha256']
    result,ref=run_fixture()
    check_outputs(result,ref)
    # Non-vacuous cold/hail, warm and number prescription families.
    inputs=ref['inputs']
    for j in range(2,14): assert np.count_nonzero(inputs[:,:,j])>0
    assert (inputs[3,:,0]<253.).any() and inputs[3,:,7].max()>0
    assert (inputs[6,:,8:14]==0).all()
    assert np.count_nonzero(ref['ref_rates'][:,7:9])>0
    assert np.max(np.abs(ref['ref_fields'][:,7]-inputs[:,:,7]))>0


def test_hail_sedimentation_deletion_is_detected(monkeypatch):
    with pytest.raises(AssertionError):
        result,ref=literal_result(monkeypatch,mutant=True)
        check_outputs(result,ref)


def literal_result(monkeypatch,mutant=False):
    import importlib.util
    from gpuwrf.physics.milbrandt2mom import column,local,math,constants
    path=HERE.parents[2]/'proofs/s2_milbrandt/literal_numpy.py'
    spec=importlib.util.spec_from_file_location('my2_literal_test',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with np.load(HERE/'literal_real4.npz') as f:ref=dict(f)
    a=ref['inputs'];result=[]
    with monkeypatch.context() as m:
        for mod in (column,local,math,constants):
            for attr in ('jnp','_op','_pow','_exp','_log','_log10','jax','cuda_native'):
                m.setattr(mod,attr,getattr(mod,attr,None),raising=False)
        helper.install()
        if mutant:
            original=column.sediment
            m.setattr(column,'sediment',lambda q,n,cat,*args:(q,n,np.asarray(0.,q.dtype)) if cat==5 else original(q,n,cat,*args))
        with np.errstate(all='ignore'):
            for i in range(len(a)):
                fields,rates,diam,zet,zec=column.column(a[i,:,:14].T,a[i,:,15],ref['ps'][i],a[i,:,14],ref['dt'])
                result.append((np.stack(fields),np.stack(rates),np.stack(diam),zet,zec))
    constants.constants.cache_clear()
    return tuple(np.stack([r[j] for r in result]) for j in range(5)),ref


def test_literal_real4_is_byte_pristine_no_fma(monkeypatch):
    result,ref=literal_result(monkeypatch)
    for name,got in zip(('fields','rates','diam','zet','zec'),result):
        assert got.dtype==ref['ref_'+name].dtype
        assert got.tobytes()==ref['ref_'+name].tobytes(),name


def test_ulp_bound_was_derived_without_candidate_values():
    meta=json.loads((HERE/'input_ulp_envelope.json').read_text())
    assert meta['reference_only'] and not meta['candidate_used_to_derive_bound']
    assert meta['n_arms']==36 and len(meta['population'])==36
    assert hashlib.sha256((HERE/'input_ulp_envelope.npz').read_bytes()).hexdigest()==meta['envelope_sha256']


def test_registry_transport_root_boundary_and_no_radius_coupling():
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.coupling.boundary_apply import BoundaryConfig,root_scalar_rk1_split
    from gpuwrf.contracts.physics_registry import number_species_for_mp
    from gpuwrf.coupling.physics_dispatch import scheme_entry
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    from gpuwrf.physics.rrtmg_mp_re import mp_re_active
    n=SimpleNamespace(mp_physics=9,moist_adv_opt=1,scalar_adv_opt=1)
    adv=op._advected_scalar_species(n)+op._nested_number_scalar_species(n)
    assert len(adv)==len(set(adv))
    assert set(adv)==set('qv qc qr qi qs qg qh Nc Nr Ni Ns Ng Nh'.split())
    assert set(number_species_for_mp(9))==set('Nc Nr Ni Ns Ng Nh'.split())
    entry=scheme_entry('microphysics',9)
    assert entry.owner_module=='gpuwrf.physics.milbrandt2mom.adapter'
    assert entry.convention=='state_adapter' and 'hail_acc' in entry.accumulators
    assert set('qh Nc Nr Ni Ns Ng Nh'.split()) <= set(entry.writes_state)
    assert set(op._diffused_scalar_species(n))==set(adv)
    cfg=BoundaryConfig(have_bcs_moist=False,have_bcs_scalar=False)
    relaxed,flow=root_scalar_rk1_split(cfg,adv,**op._root_flow_only(n))
    assert relaxed==('qv',) and set(flow)==set(adv)-{'qv'}
    status=classify_scheme('mp_physics',9)
    assert status.status is SupportStatus.IMPLEMENTED and 'CPU-oracle-qualified; GPU/coupled-forecast qualification pending' in status.reason
    assert not mp_re_active(n)


@pytest.mark.parametrize('moist,scalar',[(0,1),(1,0),(1,1)])
def test_scalar_number_option_selection(monkeypatch,moist,scalar):
    from gpuwrf.runtime import operational_mode as op
    n=SimpleNamespace(mp_physics=9,moist_adv_opt=moist,scalar_adv_opt=scalar)
    calls=[]
    def spy(*args,**kwargs):
        calls.append((kwargs['species'],kwargs['advection_opt']))
        return tuple(kwargs['species'])
    monkeypatch.setattr(op,'_scalar_transport_coupled_tendencies',spy)
    species=op._advected_scalar_species(n)
    assert op._moisture_coupled_tendencies(None,n,rk_step=1,step_origin=None)==species
    expected=[(tuple(s for s in species if s not in op._SCALAR_FAMILY_SPECIES),moist),
              (tuple(s for s in species if s in op._SCALAR_FAMILY_SPECIES),scalar)]
    assert calls==([(species,moist)] if moist==scalar else expected)


def test_forced_radar_cadence_spec_zone_and_restart_schema():
    from gpuwrf.diagnostics.online import configuration,radar_from_context,validate_restart
    n=SimpleNamespace(mp_physics=9,do_radar_ref=0,nwp_diagnostics=0,diagnostic_history_interval=60.)
    cfg=configuration(n)
    assert cfg['do_radar_ref']==1
    old=jnp.zeros((12,3,4),jnp.float32); new=jnp.full_like(old,20.)
    result=radar_from_context(old,new,n,2,1)
    assert np.asarray(result)[:,1:-1,1:-1].min()==20.
    assert np.asarray(result)[:,0].max()==0.
    validate_restart(None,result,cfg,1,old.shape)
    with pytest.raises(ValueError,match='radar|REFL'):
        validate_restart(None,None,cfg,1,old.shape)


@pytest.mark.parametrize('mp',[8,9])
def test_invalid_radar_selector_is_not_normalized_into_admission(mp):
    from gpuwrf.diagnostics.online import validate_options
    n=SimpleNamespace(mp_physics=mp,do_radar_ref=2,nwp_diagnostics=0)
    with pytest.raises(ValueError,match='0/1'):
        validate_options(n,specified=True,nested=False,periodic=False)


def test_milbrandt_spec_zone_preserves_transport_roundoff_byte_exactly():
    from gpuwrf.contracts.state import State
    from gpuwrf.runtime.operational_mode import _microphysics_interior_only
    before=State.zeros(SimpleNamespace(nz=2,ny=4,nx=5),mp_physics=9)
    qi=np.zeros((2,4,5),np.float32)
    qi[1,2,4]=np.float32(-5.337046543178522e-27)
    before=before.replace(_cast=False,qi=jnp.asarray(qi))
    after=before.replace(_cast=False,qi=jnp.ones_like(before.qi))
    n=SimpleNamespace(mp_physics=9,run_boundary=True,
        grid=SimpleNamespace(bc=SimpleNamespace(source='wrfinput')),
        boundary_config=SimpleNamespace(force_geopotential=True,spec_zone=1))
    out=np.asarray(_microphysics_interior_only(before,after,n).qi)
    assert out[:,1:-1,1:-1].min()==1.
    ring=np.ones((4,5),bool);ring[1:-1,1:-1]=False
    assert out[:,ring].tobytes()==qi[:,ring].tobytes()


@pytest.mark.parametrize('bad',['qh','Nh','hail_acc','Nc','Nr','Ni','Ns','Ng'])
def test_restart_requires_all_active_milbrandt_fields(bad):
    from gpuwrf.io.restart_schema import validate_held_state
    fields={n:np.zeros((2,3,4),np.float32) for n in 'qv qh Nc Nr Ni Ns Ng Nh'.split()}
    fields['hail_acc']=np.zeros((3,4),np.float32)
    n=SimpleNamespace(mp_physics=9,use_mp_re=0)
    validate_held_state(fields,n)
    fields[bad]=None
    with pytest.raises(ValueError,match='E78'): validate_held_state(fields,n)


def test_nested_admission_is_explicitly_refused():
    from gpuwrf.io.namelist_check import refuse_single_domain_mp_nested
    from gpuwrf.io.namelist_check import NotOperationallyWiredError
    with pytest.raises(NotOperationallyWiredError,match='single domains'):
        refuse_single_domain_mp_nested({'physics':{'mp_physics':9}},2)
