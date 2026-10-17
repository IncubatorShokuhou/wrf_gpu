"""Retain actual shared WRF PBL operands for post-RK CAM-MG.

module_microphysics_driver.F:3030 passes KVH3D=EXCH_H, TKE_PBL=tke_pbl.
Only actual producer writes are retained: MYNN/YSU do not write TKE_PBL;
TEMF/GFS/MRF do not write EXCH_H. No PBL is recomputed here.
"""
import jax.numpy as jnp


def retain_mass(state, *, exch_h=None, tke=None):
    if state.cam_kvh3d is None:
        return state
    updates={}
    for name,value in (('cam_kvh3d',exch_h),('cam_tke_pbl',tke)):
        if value is not None:
            value=jnp.asarray(value,jnp.float32)
            held=getattr(state,name)
            # kts:kte writes retain the unassigned kte+1 memory element.
            # CAM-UW is the separate full-interface producer, already wired.
            updates[name]=jnp.concatenate((value,held[value.shape[0]:]),axis=0)
    return state.replace(**updates) if updates else state


def retain_columns(state, *, exch_h=None, tke=None):
    if state.cam_kvh3d is None:
        return state
    _,ny,nx=state.theta.shape
    def mass(value):
        return None if value is None else jnp.moveaxis(value.reshape(ny,nx,-1),-1,0)
    return retain_mass(state,exch_h=mass(exch_h),tke=mass(tke))
