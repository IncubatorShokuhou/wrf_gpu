"""SIMGM opt_run=1 initial state and crash-safe restart validation."""
from dataclasses import replace
from collections.abc import Mapping
import numpy as np
import jax.numpy as jnp


def bind_groundwater(namelist,*,state):
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    # NOAHMP_INIT:2144-2147: the run!=5 cold state. All three are REAL.
    initial=GroundwaterState(*(jnp.full(state.t_skin.shape,value,jnp.float32) for value in (2.5,4900.,4900.)))
    validate_restart(initial,namelist,1,state.t_skin.shape)
    return replace(namelist,noahmp_groundwater_initial=initial)


def validate_restart(group,namelist,version,shape):
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    opts=namelist.get("noahmp_options") if isinstance(namelist,Mapping) else getattr(namelist,"noahmp_options",None)
    active=opts is not None and opts.opt_run==1
    if active!=(group is not None):raise ValueError("Noah-MP SIMGM groundwater restart is missing or disagrees with selection (E78)")
    if not active:return
    if version!=1 or not isinstance(group,GroundwaterState):raise ValueError("Noah-MP SIMGM groundwater restart schema is missing/incompatible (E78)")
    for value in group:
        array=np.asarray(value)
        if array.shape!=tuple(shape) or array.dtype!=np.dtype(np.float32) or not np.isfinite(array).all():
            raise ValueError("Noah-MP SIMGM groundwater restart requires finite REAL32 ZWT/WA/WT surface arrays (E78)")
