"""WRF SSiB canopy/soil/multilayer snow model (sf_surface_physics=8).

Literal column arithmetic lives in _ssib_generated; this module owns the array
boundary and complete resident INOUT/history carry. Native WRF REAL is fp32;
local D-exponent expressions retain WRF's DOUBLE evaluation before assignment.
The UPDAT1 uninitialized SNOWW(2) read uses the manager-approved zero policy.
GPU and coupled-forecast qualification remain pending; see the support matrix.
"""
from __future__ import annotations
import inspect
from typing import Any, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from gpuwrf.physics._ssib_generated import _ssib, _ssib_seaice

SSIB_ORACLE_DIR = 'proofs/s2_ssib'
SSIB_SAVEPOINT_GLOB = 'proofs/v017/savepoints/ssib/fp64/ssib_case_*.json'
SSIB_NUM_SOIL_LAYERS = 3
SSIB_SCHEMA_VERSION = 1
ARGUMENTS = tuple(inspect.signature(_ssib).parameters)
SEAICE_ARGUMENTS = tuple(inspect.signature(_ssib_seaice).parameters)
SNOW_LEAVES = tuple(f'{name}{layer}' for layer in range(1,5)
                    for name in ('dzo','wo','tssn','tssno','bwo','bto','cto','fio','flo','bio','blo','ho'))
INTEGER_ARGUMENTS = frozenset(('ix','jx','itime','ivgtyp','isnow','sw_physics','mminlu'))
FORCING_FIELDS = frozenset(('ix','jx','ddtt','itime','zlat','sunangle','ppl','ppc','rlwdown','zwind2',
    'umm','vmm','qm','tm','pm','psur','ivgtyp','swdown1','radfrac11','radfrac12','radfrac21','radfrac22',
    'day','cloud','sw_physics','mminlu'))
STATE_FIELDS = tuple(name for name in ARGUMENTS if name not in FORCING_FIELDS) + ('sfcevp',)
SsibLandState = NamedTuple('SsibLandState', [(name,Any) for name in STATE_FIELDS])
SsibLandState.__module__ = __name__
SsibLandState.replace = SsibLandState._replace


def initial_ssib_land(*, tslb, smois, snow, canwat, albedo, sfcrunoff=None, sfcevp=None):
    """Cold REAL carry from real.exe's three SSiB soil levels, no interpolation."""
    t=jnp.asarray(tslb,jnp.float32); w=jnp.asarray(smois,jnp.float32)
    if t.shape!=w.shape or t.ndim<2 or t.shape[0]!=3:
        raise ValueError('SSiB requires real.exe-prepared three-layer TSLB/SMOIS')
    shape=t.shape[1:]; z=jnp.zeros(shape,jnp.float32)
    values={name:z for name in STATE_FIELDS}
    values.update(tc=t[0],tgs=t[1],td=t[2],www1=w[0],www2=w[1],www3=w[2],
        snoa=jnp.asarray(snow,jnp.float32),snob=jnp.asarray(canwat,jnp.float32),
        bedo=jnp.asarray(albedo,jnp.float32),ta=t[0],isnow=jnp.ones(shape,jnp.int32),
        snowden=jnp.full(shape,3.75,jnp.float32),tkair=z,tgeff=t[0],
        roff=z if sfcrunoff is None else jnp.asarray(sfcrunoff,jnp.float32),
        sfcevp=z if sfcevp is None else jnp.asarray(sfcevp,jnp.float32))
    return SsibLandState(**values)


def ssib_column(*, land, forcing, is_land=True, is_ice=False):
    """One literal WRF column; returns every evolving state and diagnostic leaf."""
    values={name:getattr(land,name) for name in STATE_FIELDS}
    values.update(forcing)
    args=tuple(jnp.asarray(values[name],jnp.int32 if name in INTEGER_ARGUMENTS else jnp.float32)
               for name in ARGUMENTS)
    def run_land(_):
        out=dict(zip(ARGUMENTS,_ssib(*args)))
        return tuple(out[name] if name!='sfcevp' else land.sfcevp+out['xhlflx']*args[2]
                     for name in STATE_FIELDS)
    def run_ice(_):
        iceargs=tuple(jnp.asarray(values[name],jnp.int32 if name in INTEGER_ARGUMENTS else jnp.float32)
                      for name in SEAICE_ARGUMENTS)
        out=dict(zip(SEAICE_ARGUMENTS,_ssib_seaice(*iceargs)))
        return tuple(out.get(name,getattr(land,name)) if name!='sfcevp'
                     else land.sfcevp+out['xhlflx']*args[2] for name in STATE_FIELDS)
    # Including sea ice in the trace is opt-in at the bundle boundary; the two
    # routines return different WRF diagnostic inventories, with untouched snow
    # fields retained for sea ice just as the caller's grid arrays are retained.
    ice_enabled='yice' in forcing
    def active(_):
        return lax.cond(jnp.asarray(is_ice),run_ice,run_land,None) if ice_enabled else run_land(None)
    return SsibLandState(*lax.cond(jnp.asarray(is_land),active,lambda _:tuple(land),None))


def ssib_columns(*, land, forcing, is_land=None, is_ice=None):
    """Batched resident columns; arrays have the land carry's common (ny,nx) shape."""
    shape=land.tc.shape; size=int(np.prod(shape))
    flatten=lambda x:jnp.broadcast_to(x,shape).reshape(size)
    flat_land=jax.tree.map(flatten,land)
    flat_forcing={name:flatten(value) for name,value in forcing.items()}
    active=jnp.ones(size,bool) if is_land is None else flatten(is_land)
    ice=jnp.zeros(size,bool) if is_ice is None else flatten(is_ice)
    out=jax.vmap(lambda l,f,a,i:ssib_column(land=l,forcing=f,is_land=a,is_ice=i))(
        flat_land,flat_forcing,active,ice)
    return jax.tree.map(lambda x:x.reshape(shape),out)


def validate_ssib_restart(land, schema, shape, *, active=False):
    """Host-only E78 admission, including active missing groups and all 48 snow leaves."""
    if land is None:
        if active: raise ValueError('active SSiB restart missing land carry (E78)')
        return
    if schema!=SSIB_SCHEMA_VERSION or not isinstance(land,SsibLandState):
        raise ValueError('SSiB restart schema/type mismatch (E78)')
    for name in STATE_FIELDS:
        value=getattr(land,name); dtype=np.int32 if name=='isnow' else np.float32
        if value is None or np.shape(value)!=tuple(shape) or np.asarray(value).dtype!=dtype:
            raise ValueError(f'SSiB restart {name} shape/dtype mismatch (E78)')
        if not np.all(np.isfinite(value)):
            raise ValueError(f'SSiB restart {name} nonfinite (E78)')
