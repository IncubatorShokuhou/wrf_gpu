"""Fail-closed E78 carry checks for experimental EEPS/KEPS."""
from collections.abc import Mapping

import numpy as np

from gpuwrf.io.restart_schema import physics_option
from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars
from gpuwrf.contracts.precision import EPSILON_PBL_DIAGNOSTICS,EPSILON_PBL_LEAVES

EPSILON_PBL_SCHEMA_VERSION=1


def epsilon_pbl_active(namelist):
    return physics_option(namelist,'bl_pbl_physics',5) in (16,17)


def validate_epsilon_state(state,namelist,version,*,check_values=True):
    bl=physics_option(namelist,'bl_pbl_physics',5)
    get=state.get if isinstance(state,Mapping) else lambda name:getattr(state,name,None)
    if bl not in (16,17):
        if any(get(name) is not None for name in EPSILON_PBL_LEAVES):
            raise ValueError('populated epsilon PBL carry requires bl_pbl_physics=16/17')
        return
    if version != EPSILON_PBL_SCHEMA_VERSION:
        raise ValueError('EEPS/KEPS restart predates epsilon PBL carry schema (E78)')
    required=('qke','epsilon_pbl','epsilon_pbl_step',*EPSILON_PBL_DIAGNOSTICS)+epsilon_pbl_scalars(bl)+(('tpe_pbl','pr_pbl') if bl==17 else ())
    if any(get(name) is None for name in required):
        raise ValueError(f'bl_pbl_physics={bl}: missing required epsilon PBL carry (E78)')
    shape=get('qv').shape
    for name in required:
        value=get(name)
        if name=='epsilon_pbl_step':
            if value.shape!=() or np.dtype(value.dtype)!=np.dtype('int32'):
                raise ValueError('epsilon PBL call counter must be scalar int32 (E78)')
            if check_values and int(np.asarray(value))<0:
                raise ValueError('epsilon PBL call counter is negative')
            continue
        if value.shape!=shape or np.dtype(value.dtype)!=np.dtype('float32'):
            raise ValueError(f'bl_pbl_physics={bl}: {name} must be REAL32 on the mass grid (E78)')
        if check_values and not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'bl_pbl_physics={bl}: nonfinite {name} in restart')
    boundary=(namelist.get('boundary_config') if isinstance(namelist,Mapping) else getattr(namelist,'boundary_config',None))
    nested=(boundary.get('nested_frozen_wrf_boundary_bundle',False) if isinstance(boundary,Mapping)
            else getattr(boundary,'nested_frozen_wrf_boundary_bundle',False))
    have=(boundary.get('have_bcs_scalar',False) if isinstance(boundary,Mapping)
          else getattr(boundary,'have_bcs_scalar',False))
    if (nested or have is True) and any(get(f'{name}_bdy') is None for name in epsilon_pbl_scalars(bl)):
        raise ValueError(f'bl_pbl_physics={bl}: missing required PBL scalar boundary records (E78)')


def epsilon_pbl_config(namelist):
    bl=physics_option(namelist,'bl_pbl_physics',5)
    boundary=(namelist.get('boundary_config') if isinstance(namelist,Mapping) else getattr(namelist,'boundary_config',None))
    nested=(boundary.get('nested_frozen_wrf_boundary_bundle',False) if isinstance(boundary,Mapping)
            else getattr(boundary,'nested_frozen_wrf_boundary_bundle',False))
    have=(boundary.get('have_bcs_scalar',None) if isinstance(boundary,Mapping)
          else getattr(boundary,'have_bcs_scalar',None))
    return {'bl_pbl_physics':bl,'boundary_config':{'nested_frozen_wrf_boundary_bundle':bool(nested),
                                                  'have_bcs_scalar':None if have is None else bool(have)}}


def validate_epsilon_history(history,namelist,surface_shape):
    """Full EPS carry needs its resident history group for the scan ABI."""
    if not epsilon_pbl_active(namelist): return
    for name in ('t2','q2','u10','v10','lh','psfc'):
        value=getattr(history,name,None)
        if value is None:
            raise ValueError('EEPS/KEPS full restart missing surface history carry (E78)')
        if value.shape!=surface_shape or np.dtype(value.dtype)!=np.dtype('float32'):
            raise ValueError('EEPS/KEPS surface history must be REAL32 on the surface grid (E78)')
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError('EEPS/KEPS nonfinite surface history (E78)')
