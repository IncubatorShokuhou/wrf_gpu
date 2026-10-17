"""Component qualification against unmodified pristine Fortran, not a forecast gate."""
from functools import partial
from pathlib import Path
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import urban_bep_components as k
from gpuwrf.physics.urban_bep_bem import bep_step, bep_bem_step

jax.config.update('jax_enable_x64', True)
SAVE = Path(__file__).resolve().parents[1]/'proofs/v040/urban_bep_bem/savepoints'


@pytest.fixture(params=['fp64', 'fp32'])
def oracle(request):
    dtype = np.float64 if request.param == 'fp64' else np.float32
    data = np.load(SAVE/f'components_{request.param}.npz')['values']
    assert np.isfinite(data).all()
    assert len({tuple(row[:4]) for row in data}) == len(data)

    def get(kind, step=0):
        rows = data[(data[:, 0] == kind) & (data[:, 3] == step)]
        rows = rows[np.lexsort((rows[:, 2], rows[:, 1]))]
        nfield = int(rows[:, 2].max())
        assert len(rows) == 48*nfield
        return rows[:, 4].reshape(48, nfield).astype(dtype)

    return request.param, get


def check(actual, expected, precision, *, material=False):
    atol, rtol = (2e-11, 2e-12) if precision == 'fp64' else (2e-5, 4e-6)
    if material and precision == 'fp32':
        atol = 8e-5
    actual = np.asarray(actual)
    assert np.isfinite(actual).all()
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize('kind', [1, 2, 3, 4])
def test_surface_fluxes(oracle, kind):
    precision, get = oracle
    dz,z0,u,v,theta,theta0,surface,rho,angle,dt,pressure,qv = map(jnp.asarray, get(0).T)
    if kind == 1:
        result = jax.jit(k.bep_flux_flat)(dz,z0,u,v,theta,theta0,surface)
    elif kind == 2:
        result = jax.jit(k.bep_flux_wall)(u,v,theta,rho,surface,angle,dt)
    elif kind == 3:
        result = jax.jit(k.bem_flux_flat)(dz,z0,u,v,theta,theta0,surface,qv,rho,pressure)
    else:
        result = jax.jit(k.bem_flux_wall)(u,v,theta,rho,surface,surface+2.,angle,dt,0.4)
    check(jnp.stack(result, axis=-1), get(kind), precision)


def test_material_temperature_every_step(oracle):
    precision, get = oracle
    inputs = jnp.asarray(get(10));temp,depth,ala,cs = [inputs[:, i:i+10] for i in (0,10,20,30)]
    sw,lw,em,alb,heat = inputs[:,40:].T
    pressure,dt = jnp.asarray(get(0))[:,[10,9]].T
    step = jax.jit(k.bep_soil_temp)
    for s in range(1,7):
        temp,pt,rt,gf = step(depth,temp,ala,cs,sw,lw,pressure,dt,em,alb,heat)
        actual = jnp.concatenate((temp,pt[:,None],rt[:,None],gf[:,None]), axis=-1)
        check(actual[:,:11], get(5,s)[:,:11], precision, material=True)
        check(actual[:,11:], get(5,s)[:,11:], precision)


def map_inputs(get, index=slice(None)):
    inputs = jnp.asarray(get(11)[index])
    z,c,zu,pb,ss = (inputs[:,a:b] for a,b in [(0,6),(6,12),(12,30),(30,48),(48,66)])
    wall = jnp.asarray(get(12)[index]).reshape(-1,4,18,7)
    flat = jnp.asarray(get(13)[index]).reshape(-1,2,18,4)
    return z,c,zu,pb,ss,k.WallFlux(*(wall[...,j] for j in range(7))),k.FlatFlux(*(flat[...,j] for j in range(4)))


def test_meso_to_urban_interpolation(oracle):
    precision,get = oracle
    z,c,zu,*_ = map_inputs(get)
    result = jax.jit(k.bep_interpolate)(z,zu[:,:7],c)
    check(result,get(6),precision)


@pytest.mark.parametrize('nd', [1,2])
def test_multilayer_sources_and_lengths(oracle,nd):
    precision,get = oracle
    index = np.arange(48)[(np.arange(1,49)%2+1) == nd]
    z,c,zu,pb,ss,wall,flat = map_inputs(get,index)
    bs = jnp.array([10.,20.],dtype=z.dtype)[None,:].repeat(len(index),axis=0)
    ws = jnp.array([15.,12.],dtype=z.dtype)[None,:].repeat(len(index),axis=0)
    dz = z[:,1:]-z[:,:-1]
    result = jax.jit(partial(k.bep_urban_meso,nd=nd))(z,dz,zu[:,:7],pb,ss,bs,ws,wall,flat)
    fields = jnp.stack((*result[:8],result.surface[:,:-1],result.volume),axis=-1).reshape(len(index),50)
    actual = jnp.concatenate((fields,result.surface[:,-1:]),axis=-1)
    check(actual,get(7)[index],precision)
    length = jax.jit(partial(k.bep_interp_length,nd=nd))(z,zu[:,:7],ss,ws,bs)
    check(jnp.stack(length,axis=-1).reshape(len(index),10),get(8)[index],precision)


def test_bem_hvac_including_latent_inout(oracle):
    precision,get = oracle
    hour,qb,hs,volume,tlev,shum,hl = jnp.asarray(get(14)).T
    rho,dt = jnp.asarray(get(0))[:,[7,9]].T
    switch = jnp.arange(1,49)%2
    sensible = jax.jit(k.bem_regtemp)(switch,hour,dt,qb,hs,tlev,8.,20.,298.,0.5)
    latent = jax.jit(k.bem_reghum)(switch,hour,dt,volume,rho,2450000.,hl,shum,8.,20.,0.005,0.0002)
    exhaust = jax.jit(k.bem_air_cond)(sensible,latent,3.5)
    check(jnp.stack((sensible,latent,*exhaust),axis=-1),get(9),precision)
    expected = get(9)
    assert np.any(expected[:,0]>0.) and np.any(expected[:,0]<0.)
    assert np.any(expected[:,4]>0.) and np.any(expected[:,4]==0.)
    assert np.any(expected[:,1]!=expected[:,5])  # heating suppresses latent demand


def test_full_canopy_requires_explicit_state_forcing_and_geometry():
    for function in (bep_step,bep_bem_step):
        with pytest.raises(TypeError):
            function()


def test_oracle_parser_rejects_early_stop_missing_step_and_duplicate(tmp_path):
    script = SAVE.parent/'build_oracle.py'
    spec = importlib.util.spec_from_file_location('bep_oracle_build', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = 'NREG=1\nREGIME_NAME[1]=day\nNSTEPS[1]=1\nX[1][0]=1.0\nX[1][1]=2.0\n'
    src,dst = tmp_path/'dump.txt',tmp_path/'test.npz'
    src.write_text(valid)
    module.parse_dump(src,dst)
    for bad in (valid.replace('NREG=1','NREG=2'),
                valid.replace('X[1][1]=2.0\n',''),
                valid+'X[1][1]=2.0\n'):
        src.write_text(bad)
        with pytest.raises(AssertionError):
            module.parse_dump(src,dst)


@pytest.mark.parametrize('scheme,table', [(2,'ac_on'),(3,'ac_on'),(3,'ac_off')])
@pytest.mark.parametrize('precision',['fp64','fp32'])
def test_whole_wrf_oracle_is_active_and_complete(scheme,table,precision):
    with np.load(SAVE/f'urban{scheme}_{precision}_{table}.npz') as data:
        for field in data.files:
            if field != 'regime_names':
                assert np.isfinite(data[field]).all(),field
        assert data['A_U_BEP'].shape == (6,7,9)
        assert np.any(data['A_U_BEP'][:,:6,:8]!=0.)
        assert np.any(data['B_T_BEP'][:,:6,:8]!=0.)
        assert np.count_nonzero(data['A_U_BEP'][5,1,:8]) >= 2
        assert np.count_nonzero(data['B_T_BEP'][5,1,:8]) >= 2
        assert np.any(data['TRB_URB4D'][:,1:,:]!=data['TRB_URB4D'][:,:-1,:])
        if scheme == 3:
            assert np.any(data['TLEV_URB3D'][:,1:,:]!=data['TLEV_URB3D'][:,:-1,:])
            if table == 'ac_on':
                assert np.any(data['CM_AC_URB3D'][:,1:]>0.)
            else:
                assert np.all(data['CM_AC_URB3D']==0.)
