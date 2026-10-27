"""Held MYNN histories: optional groups, separate from prognostic State."""
from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp

from gpuwrf.physics.mynn_budget import TKE_NAMES, EDMF_NAMES, PBL_NAMES
from gpuwrf.io.mynn_budget_options import configuration

SCHEMA_VERSION = 1


class MynnBudgetHistory(NamedTuple):
    tke: tuple | None = None
    output: tuple | None = None


def seed(shape,config):
    """Allocate only the requested REAL history groups, after precision setup."""
    if not (config['tke_budget'] or config['bl_mynn_output']):return None
    zero=jnp.zeros(shape,jnp.float32)
    return MynnBudgetHistory((zero,)*5 if config['tke_budget'] else None,
                            (zero,)*16 if config['bl_mynn_output'] else None)


def from_columns(values,ny,nx,config):
    """Native (column,level) rates become resident mass-grid history fields."""
    assert len(values)==21
    def mass(value):
        return jnp.moveaxis(value.reshape(ny,nx,value.shape[-1]),-1,0).astype(jnp.float32)
    return MynnBudgetHistory(tuple(mass(v) for v in values[:5]) if config['tke_budget'] else None,
                            tuple(mass(v) for v in values[5:]) if config['bl_mynn_output'] else None)


def history_fields(history):
    if history is None:return {}
    result={}
    for names,values in ((TKE_NAMES,history.tke),(EDMF_NAMES+PBL_NAMES,history.output)):
        if values is not None:
            if len(names)!=len(values):raise ValueError('incomplete MYNN budget history group')
            result.update(zip(names,values,strict=True))
    return result


def validate_restart(history,config,marker,shape):
    """An ON restart must contain the exact selected history schema and shape."""
    import numpy as np
    active=bool(config['tke_budget'] or config['bl_mynn_output'])
    if not active:
        if history is not None:raise ValueError('checkpoint MYNN budget history was enabled; preserve its switches')
        return
    if marker!=SCHEMA_VERSION or not isinstance(history,MynnBudgetHistory):
        raise ValueError('checkpoint lacks MYNN budget history schema 1; cannot enable budgets on this restart')
    for values,selected,count in ((history.tke,config['tke_budget'],5),(history.output,config['bl_mynn_output'],16)):
        if bool(values is not None)!=bool(selected) or (values is not None and len(values)!=count):
            raise ValueError('checkpoint MYNN budget switches differ from the requested configuration')
        for value in values or ():
            a=np.asarray(value)
            if a.shape!=tuple(shape) or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError('checkpoint MYNN budget history has invalid shape/dtype/values')


def validate_payload(payload,requested=None):
    """Validate host checkpoint groups before device restoration (E78)."""
    saved=payload.get('mynn_budget_config')
    history=payload.get('extra_fields',{}).get('mynn_budget')
    if saved is None:
        if history is not None:raise ValueError('checkpoint lacks MYNN budget selection binding (E78)')
        saved=configuration({})
    else:
        if not isinstance(saved,dict) or set(saved)!=set(configuration({})):
            raise ValueError('checkpoint MYNN budget selection is malformed (E78)')
        saved=configuration(saved)
    shape=payload['state_fields']['qv'].shape
    validate_restart(history,saved,payload.get('mynn_budget_schema_version'),shape)
    if requested is not None:
        target=configuration(requested)
        validate_restart(history,target,payload.get('mynn_budget_schema_version'),shape)
        if target!=saved:raise ValueError('checkpoint MYNN budget/EDMF selection differs (E78)')
