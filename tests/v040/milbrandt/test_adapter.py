"""Actual WRF wrapper outputs through the production State adapter."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np

HERE=Path(__file__).parent
SPECIES='qv qc qr qi qs qg qh Nc Nr Ni Ns Ng Nh'.split()


def test_production_adapter_matches_actual_pristine_wrapper():
    from gpuwrf.contracts.state import State
    from gpuwrf.physics.milbrandt2mom.adapter import milbrandt_adapter
    with np.load(HERE/'wrapper_real4.npz') as f:ref=dict(f)
    with np.load(HERE/'wrapper_ulp_envelope.npz') as f:envelope=dict(f)
    spec=json.loads((HERE/'wrapper_manifest.json').read_text())
    assert hashlib.sha256((HERE/'wrapper_real4.npz').read_bytes()).hexdigest()==spec['fixture_sha256']
    a=ref['inputs'];n,nz,_=a.shape;ny,nx=3,4
    assert n==ny*nx
    def volume(v):return jnp.asarray(np.moveaxis(v.reshape(ny,nx,nz),-1,0),jnp.float32)
    state=State.zeros(SimpleNamespace(nz=nz,ny=ny,nx=nx),mp_physics=9)
    changes={'theta':volume(a[:,:,0]),'p_total':volume(a[:,:,15]),
             'p_perturbation':jnp.zeros_like(state.p_perturbation,dtype=jnp.float32)}
    changes.update({name:volume(a[:,:,i+1]) for i,name in enumerate(SPECIES)})
    z=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None,None]*100.,(nz+1,ny,nx))
    changes['ph_total']=jnp.asarray(z*np.float32(9.81))
    changes['ph_perturbation']=jnp.zeros_like(changes['ph_total'])
    changes['w']=jnp.concatenate((volume(a[:,:,14]),jnp.zeros((1,ny,nx),jnp.float32)))
    changes.update({name:jnp.zeros((ny,nx),jnp.float32) for name in ('rain_acc','snow_acc','graupel_acc','hail_acc','ice_acc')})
    state=state.replace(_cast=False,**changes)
    grid=SimpleNamespace(metrics=SimpleNamespace(fnm=jnp.full(nz,.5,jnp.float32),fnp=jnp.full(nz,.5,jnp.float32)))
    fn=jax.jit(lambda s:milbrandt_adapter(s,float(ref['dt']),grid,return_radar=True))
    out,precip,radar=fn(state)
    for i,name in enumerate(('theta',*SPECIES)):
        got=np.asarray(getattr(out,name)).transpose(1,2,0).reshape(n,nz)
        want=ref['ref_fields'][:,i]
        atol=3e-5 if i==0 else 3e-10 if i<8 else 3e-3
        assert np.isfinite(got).all()
        diff=np.abs(got-want)
        valid=(diff<=atol+3e-5*np.abs(want))|(diff<=envelope['fields'][:,i])
        failures=[(tuple(map(int,ix)),float(got[tuple(ix)]),float(want[tuple(ix)]),
                   float(envelope['fields'][:,i][tuple(ix)])) for ix in np.argwhere(~valid)]
        assert np.all(valid),(name,failures[:8])
        assert getattr(out,name).dtype==jnp.float32
    for name,ix in (('rainncv',1),('sr',8)):
        np.testing.assert_allclose(np.asarray(precip[name]).ravel(),ref['ref_rain'][:,ix],rtol=3e-5,atol=3e-7)
    for name,ix in (('snow_acc',2),('graupel_acc',4),('hail_acc',6)):
        np.testing.assert_allclose(np.asarray(getattr(out,name)).ravel(),ref['ref_rain'][:,ix],rtol=3e-5,atol=3e-7)
    total=sum(getattr(out,n) for n in ('rain_acc','snow_acc','graupel_acc','hail_acc','ice_acc'))
    np.testing.assert_allclose(np.asarray(total).ravel(),ref['ref_rain'][:,0],rtol=3e-5,atol=3e-7)
    np.testing.assert_allclose(np.asarray(radar).transpose(1,2,0).reshape(n,nz),ref['ref_zet'],rtol=3e-5,atol=3e-4)
