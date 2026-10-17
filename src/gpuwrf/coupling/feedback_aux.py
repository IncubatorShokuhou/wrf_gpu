"""Resident WRF up-feedback diagnostics with distinct forcing/history aliases."""
import jax
import jax.numpy as jnp

from gpuwrf.coupling.feedback_registry import FEEDBACK_FIELDS
from gpuwrf.runtime.history_accumulators import PackedFields,RADIATION_SOURCES

FLUXES={name[2:]:attr for name,attr in RADIATION_SOURCES.items()}


@jax.jit
def feedback_group(parent,child,weights):
    from gpuwrf.coupling.boundary_feedback import _feedback_leaf,point_feedback_weights
    point=weights.point if weights.point is not None else point_feedback_weights(weights.mass)
    out={}
    for name,p in parent.items():
        method,smooth,_,_=FEEDBACK_FIELDS.get(name,(None,False,None,None))
        if method is None or name not in child:
            out[name]=p
            continue
        plan=point if method in ('copy_fcnm','copy_fcni') else weights.mass
        out[name]=_feedback_leaf(p,child[name],plan,weights.mass_smooth if smooth else None,
                                 weights.smooth_option if smooth else 0)
    return PackedFields.pack(out)


def seed_feedback_history(case,domain,bundle,carry):
    """Initialization only; static DVEG=4 pools remain resident after feedback."""
    import numpy as np
    from gpuwrf.io.land_history import load_land_history_inputs,LAND_LEAVES
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.netcdf_lock import Dataset
    fields=load_land_history_inputs(case/f'wrfinput_{domain}',table_dir=wrf_run_dir(),
        parameters=bundle.namelist.noahmp_static.parameters,
        field_overrides=bundle.writer_static_overrides)
    fields={name:jnp.asarray(value,jnp.float32) for name,value in fields.items()
            if name in FEEDBACK_FIELDS and name not in LAND_LEAVES and name!='TSK'}
    zero=jnp.zeros_like(carry.state.t_skin,dtype=jnp.float32)
    with Dataset(case/f'wrfinput_{domain}') as ds:
        for name in (*FLUXES,'TH2'):
            overrides=bundle.writer_static_overrides or {}
            fields[name]=(jnp.asarray(overrides[name],jnp.float32) if name in overrides else
                jnp.asarray(np.asarray(ds[name][0]),jnp.float32) if name in ds.variables else zero)
    return PackedFields.pack(fields)


def refresh_fluxes(carry,run_radiation):
    """Keep SWDOWN/GLW forcing separate from the up-fed SWDNB/LWDNB history."""
    old=carry.feedback_history
    if old is None:return None
    def refresh(_):
        rad=carry.radiation_diagnostics
        return PackedFields.pack({name:jnp.asarray(getattr(rad,FLUXES[name]),old.data.dtype)
            if name in FLUXES and getattr(rad,FLUXES[name],None) is not None else value
            for name,value in old.items()})
    if isinstance(run_radiation,bool):return refresh(None) if run_radiation else old
    return jax.lax.cond(run_radiation,refresh,lambda _:old,None)


def radiation_for_sums(carry):
    """WRF sums consume each up-fed instantaneous flux until its next refresh."""
    rad=carry.radiation_diagnostics
    if carry.feedback_history is None or rad is None:return rad
    return rad._replace(**{attr:carry.feedback_history[name] for name,attr in FLUXES.items()
                          if getattr(rad,attr,None) is not None})


def refresh_surface(carry):
    old=carry.feedback_history
    if old is None:return None
    history=carry.history_diagnostics
    if history is None:raise ValueError('two-way feedback requires resident surface diagnostics')
    # SFCLAY TH2=T2/pi2; use WRF REAL constants/operation order.
    th2=history.t2*(jnp.float32(100000.)/history.psfc)**(jnp.float32(287.)/jnp.float32(1004.5))
    return PackedFields.pack({name:th2 if name=='TH2' else value for name,value in old.items()})
