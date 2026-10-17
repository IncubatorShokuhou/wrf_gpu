"""Typed constants and trace-time locals for literal WRF Goddard 4-ice."""
import numpy as np
import jax.numpy as jnp
from .constants import CONSTANTS

F = jnp.float32
D = jnp.float64
C = {name: np.asarray(value, np.float32) for name, value in CONSTANTS.items()}
C['as_'] = C['as']
C.update(itaobraun=0, improve=8, new_ice_sat=9, id=0, diagflag=False, do_radar_ref=0)

class _Work:
    """SSA locals; no Python objects or dictionaries reach the executable."""
    def __init__(self, **fields):
        self.values = fields
    def __getattr__(self, name):
        return self.values.get(name, F(0))
    def __setattr__(self, name, value):
        if name == 'values':
            object.__setattr__(self,name,value)
        else:
            self.values[name]=value
    def put(self, mask, **fields):
        for name,value in fields.items():
            if mask is True:
                self.values[name] = value
                continue
            self.values[name] = jnp.where(mask, value, self.values.get(name,jnp.zeros_like(value)))

def eff_rad(slope):
    # WRF module_mp_gsfcgce_4ice_nuwrf.F:4986, REAL arithmetic.
    return jnp.where(slope <= F(0), F(0), F(1.5) / (slope * F(100.)) * F(1.e6))

def gamma_toshi(x):
    """Exact precomputed original DOUBLE-island results for WRF land/water CCN.

    No aerosol coupling is enabled. Its only callers receive mu+1/3/4,
    where mu = min(15,1000/CCN+2), CCN=1500 or150. WRF getter computed
    these six REAL-return values with the unmodified gamma_toshi body.
    """
    land_mu = F(1000.) / F(1500.) + F(2.)
    result = jnp.zeros_like(x)
    for offset in (1,3,4):
        land = C[f'gamma_land_{offset}']
        water = C[f'gamma_water_{offset}']
        # The two families are widely separated; exact offset ownership.
        result = jnp.where(x == land_mu + F(offset), land, result)
        water_mu = F(1000.) / F(150.) + F(2.)
        result = jnp.where(x == water_mu + F(offset), water, result)
    return result

def gammagce(x):
    raise RuntimeError('gammagce is proof-time constant initialization only')


def ipow(x, exponent):
    """Fortran integer exponentiation uses products, not REAL powf."""
    if exponent == 0:
        return jnp.ones_like(x)
    if exponent == 1:
        return x
    if exponent == 5:
        # gfortran REAL**5: x2=x*x; (x*x2)*x2 (measured -O2 assembly).
        square = x * x
        return (x * square) * square
    half = ipow(x, exponent // 2)
    square = half * half
    return square if exponent % 2 == 0 else x * square


def table_lookup(name,index):
    return jnp.asarray(C[name])[jnp.asarray(index,jnp.int32)-1]

def shift_up(value,name):
    return jnp.roll(value,-1,axis=0)


def first_density(value):
    return value[0]
