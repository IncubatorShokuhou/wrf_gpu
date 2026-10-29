"""WRF mp8 retains entry L_qg for its column sedimentation gate.

The density controls are explicitly derived from a real h72 column. They
straddle WRF's unchanged R1 floor: immediate fall used to erase 21 traces.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import thompson_column as tc

HERE=Path(__file__).parent
FIXTURE=HERE/'fixtures/graupel_entry_sedimentation.npz'
spec=importlib.util.spec_from_file_location('graupel_mixed_helper',HERE/'test_mixed_phase_wrf_oracle.py')
mixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(mixed)


@pytest.fixture(scope='module')
def replay():
    z=dict(np.load(FIXTURE))
    manifest=json.loads(FIXTURE.with_suffix('.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==manifest['files'][FIXTURE.name]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('GPUWRF_THOMPSON_WRF_CONSTANTS','1')
        result=mixed._run(z,np.arange(len(z['names'])))
    return z,manifest,result


def test_graupel_birth_matches_pristine_zero_pattern_and_frozen_increment_gate(replay):
    z,manifest,port=replay
    assert np.count_nonzero(z['wrf_qg'])>0
    assert np.count_nonzero((z['wrf_qg']>0)!=(z['old_qg']>0))==21
    np.testing.assert_array_equal(port['qg']>0,z['wrf_qg']>0)
    errors=mixed._excess(port,z,np.arange(len(z['names'])),mixed._scales(z))
    assert not [(key,value) for key,value in errors.items() if value>mixed.TOL[key[1]]]


def test_existing_graupel_columns_keep_pre_fix_outputs(replay):
    z,manifest,port=replay
    idx=manifest['entry_active_control_indices']
    for field in mixed.FIELDS:
        # Regression/identity evidence; WRF fidelity is tested separately.
        np.testing.assert_array_equal(np.asarray(port[field][idx],np.float32),z['old_'+field][idx])


def test_deleting_entry_gate_restores_trace_erasure(replay,monkeypatch):
    z,_,_=replay
    source=textwrap.dedent(inspect.getsource(tc._thompson_source_sink_body))
    old='graupel_entry = state.qg > R1 if _mixed_phase_wrf_enabled() else None'
    assert source.count(old)==1
    namespace=dict(vars(tc))
    exec(compile(source.replace(old,'graupel_entry = None'),tc.__file__,'exec'),namespace)
    monkeypatch.setattr(tc,'_thompson_source_sink_body',namespace['_thompson_source_sink_body'])
    monkeypatch.setenv('GPUWRF_THOMPSON_WRF_CONSTANTS','1')
    jax.clear_caches()
    try:
        port=mixed._run(z,np.arange(len(z['names'])))
        assert np.count_nonzero((port['qg']>0)!=(z['wrf_qg']>0))==21
    finally:jax.clear_caches()


def test_fused_sedimentation_receives_the_column_gate(replay):
    from gpuwrf.kernels.phys_thompson_prep import sediment_and_finish
    z,_,_=replay
    idx=int(np.flatnonzero(np.any((z['wrf_qg']>0)&(z['old_qg']==0),axis=1))[0])
    f=lambda name:jnp.asarray(z['in_'+name][idx:idx+1],jnp.float32)
    p,qv,T=f('p'),f('qv'),f('th')*f('pii');zero=jnp.zeros_like(qv)
    qg=jnp.asarray(z['wrf_qg'][idx:idx+1],jnp.float32)
    state=tc.ThompsonColumnState(qv=qv,qc=f('qc'),qr=f('qr'),qi=f('qi'),qs=f('qs'),qg=qg,
        Ni=f('ni'),Nr=f('nr'),Ns=zero,Ng=zero,T=T,p=p,
        rho=tc.density_from_pressure_temperature(p,T,qv),dz=f('dz8w'),w=f('w'))
    with mixed._env():
        result,ppt=sediment_and_finish(state,6.,interpret=True,graupel_sed_on=jnp.array([[False]]))
        np.testing.assert_array_equal(result.qg,qg)
        np.testing.assert_array_equal(ppt['graupel'],0.)
