"""Whole pristine FDDAGD + caller source lifetime, Swiss active-regime crops."""
from pathlib import Path
import json
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import fdda_grid as fg
from gpuwrf.assimilation.fdda_surface import (
    SurfaceAnalysis, SurfaceFddaOptions, SurfaceGridFddaInputs, SURFACE_FIELDS,
)

jax.config.update('jax_enable_x64', True)
ROOT=Path(__file__).resolve().parents[3]/'proofs/s2_sfdda/savepoints'
MANIFEST=json.loads((ROOT/'MANIFEST.json').read_text()) if (ROOT/'MANIFEST.json').exists() else {'cases':{}}


def load(name):
    meta=MANIFEST['cases'][name]
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256']
    d=dict(np.load(ROOT/name))
    o=meta['opts']
    opt=SurfaceFddaOptions(**{k:v for k,v in o.items() if k not in ('dt','dx','specified')},dx_m=o['dx'])
    arrays={n+'_'+s:jnp.stack([d[f'rec{i}_{n}_{s}'] for i in (1,2)])
            for n in fg.FDDA_FIELDS for s in ('old','new')}
    surface=SurfaceAnalysis(**{n+'_'+s:jnp.stack([d[f'rec{i}_{n}_{s}'] for i in (1,2)])
            for n in SURFACE_FIELDS for s in ('old','new')})
    inp=SurfaceGridFddaInputs(**arrays,options=opt,record_steps=(0,200),surface=surface,surface_steps=(0,200))
    return d,inp


@pytest.mark.parametrize('name',sorted(MANIFEST['cases']))
def test_full_pristine_surface_chain(name):
    d,inp=load(name)
    dtype=jnp.float32 if name.endswith('r4.npz') else jnp.float64
    c=fg.initial_fdda_carry(inp,dtype)
    # Bound BEFORE comparison: at most 16 rounded operations (including exp/log/
    # power and FMA contraction) from the un-subtracted operand magnitude.
    # WRF theta subtraction rounds at 300K. The held coupled rates inherit this
    # error; coupling+decoupling adds at most four eps of their own magnitude.
    eps=np.finfo(dtype).eps
    hfx_path=ROOT/"wrf_surface_hfx.npz"
    hfx_meta=json.loads((ROOT/"HFX_MANIFEST.json").read_text())
    assert hashlib.sha256(hfx_path.read_bytes()).hexdigest()==hfx_meta["sha256"]
    with np.load(hfx_path) as ds:hfx_expected=ds[name]
    multiplier=max(float(np.max(np.abs(d[f"frame{fi}_p_phy"]/(287*d[f"frame{fi}_th_phy"])*1004*np.diff(d[f"frame{fi}_z_at_w"],axis=0)))) for fi in (1,2))
    active=0
    for ci,(it,secs,rec,frame) in enumerate(d['calls']):
        f=lambda n:jnp.asarray(d[f'frame{frame}_{n}'],dtype)
        c=fg.fdda_driver_tendencies(c,inp,f('u'),f('v'),f('th_phy'),f('qv'),f('pblh'),d['ht'],
            f('z_at_w'),float(secs),int(it),dtype,
            surface_context=(f('t_phy'),f('p_phy'),f('regime'),f('znt')))
        assert np.isfinite(np.asarray(c.hfx_fdda)).all()
        expected=hfx_expected[ci]
        assert np.all(np.abs(np.asarray(c.hfx_fdda)-expected)<=16*eps*(np.abs(expected)+multiplier*.3)), (name,ci,"HFX_FDDA")
        stages={'unc':c.tendencies}
        c=fg.couple_fdda_rates(c,f('mub')+f('mu_2'),f('mu_2'),f('mub'),d['c1h'],d['c2h'],dtype)
        stages['cpl']=c.tendencies
        stages['own']=fg.owned_fdda_tendf(c,specified_or_nested=True)
        c=fg.finish_fdda_rates(c,f('muts'),d['c1h'],d['c2h'],dtype)
        stages['dec']=c.tendencies
        for stage,values in stages.items():
            for field,a in zip(('run','rvn','rth','rqv'),values):
                b=d[f'call{ci}_{stage}_{field}'];a=np.asarray(a)
                assert a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
                scale={'run':.02,'rvn':.02,'rth':.3,'rqv':3e-5}[field]
                mass=1 if stage in ('unc','dec') else 1.2e5
                tol=16*eps*(np.abs(b)+mass*scale)
                assert np.all(np.abs(a-b)<=tol),(name,ci,stage,field,float(np.max(np.abs(a-b))),float(tol.max()))
                active+=np.count_nonzero(b)
        if inp.options.surface_code==2:
            for n,scale in (('sda_hfx',.3),('sda_qfx',3e-5)):
                a=np.asarray(getattr(c,n));b=d[f'call{ci}_{n}']
                assert np.all(np.abs(a-b)<=16*eps*(np.abs(b)+scale)),(name,ci,n,np.max(np.abs(a-b)))
    assert active>0


@pytest.mark.parametrize('name',[n for n in sorted(MANIFEST['cases']) if 'wind_census' in n])
def test_wind_census_has_both_faces_and_roughness_branches(name):
    d,inp=load(name)
    z=.5*(d['frame1_z_at_w'][:-1]+d['frame1_z_at_w'][1:])
    ht=d['ht'];reg=d['frame1_regime'];z0=d['frame1_znt']
    for axis in ('u','v'):
        if axis=='u':
            rr=.5*(reg[:,:-1]+reg[:,1:]);zz=.5*(z[0,:,:-1]-ht[:,:-1]+z[0,:,1:]-ht[:,1:]);rough=.5*(z0[:,:-1]+z0[:,1:])
        else:
            rr=.5*(reg[:-1]+reg[1:]);zz=.5*(z[0,:-1]-ht[:-1]+z[0,1:]-ht[1:]);rough=.5*(z0[:-1]+z0[1:])
        live=(rr>3.5)&(zz>10)
        assert np.count_nonzero(live&(rough<=.2))>0
        assert np.count_nonzero(live&(rough>.2))>0
