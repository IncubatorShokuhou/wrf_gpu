"""Pristine MSKF component/calm-reporting regression; full port gate is separate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.cumulus_mskf_support import mskf_time_scale, wrf_nint_i4


ROOT=Path(__file__).resolve().parents[3]
FIXTURE=ROOT/'proofs/v040/cuplus/mskf_timescale'
GATE=dict(relative=16*np.finfo(np.float32).eps,absolute=1e-7)


def _call(wind):
    return mskf_time_scale(wind,0.,wind,0.,dx=9000.,dt=54.,chmin=4000.,cloud_height=6000.,
                           scale_fac=2.,surface_pressure=100000.,surface_temperature=300.,
                           surface_vapor=.015,hfx=150.,pblh=1000.,source_layer_height=800.,
                           zlcl=500.,ust=.3,zol=-.1,abe=1000.)


def test_original_nint_intrinsic_bits():
    manifest=json.loads((FIXTURE/'manifest.json').read_text())
    raw=(FIXTURE/'nint_fixture.npz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['fixture_sha256']
    with np.load(FIXTURE/'nint_fixture.npz',allow_pickle=False) as fixture:
        original=fixture['expected']
        inputs=fixture['bits'].view(np.float32)
    actual=np.asarray(jax.jit(wrf_nint_i4)(jnp.asarray(inputs)))
    np.testing.assert_array_equal(actual,original)
    assert actual.dtype==np.int32


def test_original_supported_weak_and_nonzero_timescales():
    manifest=json.loads((FIXTURE/'manifest.json').read_text())
    rows=manifest['rows'][1:]
    actual=jax.jit(jax.vmap(_call))(jnp.asarray([row['wind'] for row in rows],jnp.float32))
    for key in ('vconv','tadvec','timec','sclvel'):
        value=np.asarray(actual[key]);reference=np.asarray([row[key] for row in rows],np.float32)
        assert np.isfinite(value).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(value,reference,rtol=GATE['relative'],atol=GATE['absolute'])
        assert value.dtype==np.float32
    np.testing.assert_array_equal(np.asarray(actual['nic']),[row['nic'] for row in rows])


def test_calm_timescale_retains_pristine_singularity():
    # No wind floor and no finite timescale substitute. This is deliberately
    # separate from the finite supported-regime primary gate.
    original=json.loads((FIXTURE/'manifest.json').read_text())['rows'][0]
    assert original['tadvec']=='Infinity' and original['nic']==0 and original['timec']==0
    actual=jax.jit(_call)(jnp.float32(0.0))
    assert np.isinf(actual['tadvec'])
    assert int(actual['nic'])==0 and float(actual['timec'])==0


@pytest.mark.parametrize('case',[1,2])
def test_original_calm_cps_nonfinite_is_reported_without_repair(case):
    """Replay actual original-CPS NaNs through the retained census and guard.

    This authenticates reporting of the preserved upstream failure. The full
    MSKF native-CPS comparison is still required before operational admission.
    """
    from gpuwrf.diagnostics.census import DeviceCensus,GUARDS,EVENTS,WORK,count_guard
    from gpuwrf.runtime.finite_state_guard import NonFiniteStateError,assert_state_finite_at_boundary
    reference=json.loads((FIXTURE/'calm_cps.json').read_text())['cases'][str(case)]
    vapor=np.asarray(reference['qv'],np.float32)
    tendency=np.asarray(reference['rqvcuten'],np.float32)
    assert np.count_nonzero(~np.isfinite(tendency))==reference['vapor_nonfinite']>0
    qv=jnp.asarray((vapor+np.float32(54.0)*tendency).reshape(-1,1,1))
    before=np.asarray(qv).view(np.uint32).copy()
    with jax.enable_x64(True):
        census=DeviceCensus(jnp.zeros((len(GUARDS),len(EVENTS)),jnp.uint64),jnp.zeros(len(WORK),jnp.uint64))
        census=count_guard(census,'boundary.qv',qv,rejected=jnp.zeros_like(qv,dtype=bool))
    count=int(census.guards[GUARDS.index('boundary.qv'),EVENTS.index('nonfinite')])
    assert count==reference['vapor_nonfinite']
    assert int(census.guards[GUARDS.index('boundary.qv'),EVENTS.index('repaired')])==0
    with pytest.raises(NonFiniteStateError,match='field=qv'):
        assert_state_finite_at_boundary(SimpleNamespace(qv=qv),domain='mskf-upstream-calm',step=1,enabled=True)
    np.testing.assert_array_equal(np.asarray(qv).view(np.uint32),before)
