"""Fail-closed resident cuplus restart schema (both checkpoint formats)."""
from __future__ import annotations
import numpy as np
from gpuwrf.io.restart_schema import physics_option

SCHEMA_VERSION=1


def validate(value,namelist,shape,version=None):
    scheme=physics_option(namelist,'cu_physics',0)
    required=scheme in (7,10,11)
    if value is None:
        if required:raise ValueError('cuplus restart missing cloud/rate carry (E78)')
        if version is not None:raise ValueError('inactive cuplus restart has schema metadata (E78)')
        return
    from gpuwrf.coupling.cuplus import CuPlusCarry
    from gpuwrf.kernels.phys_cuplus_driver import fields,INTEGER_FIELDS
    if not required:raise ValueError('cuplus carry exists with an inactive cumulus selector (E78)')
    if version!=SCHEMA_VERSION:raise ValueError('cuplus restart schema version mismatch (E78)')
    if not isinstance(value,CuPlusCarry):raise ValueError('cuplus restart carry type mismatch (E78)')
    nz,ny,nx=map(int,shape);ncol=ny*nx
    profiles,scalars=fields(scheme)
    if set(value.held)!=set(profiles+scalars):raise ValueError('cuplus restart field set mismatch (E78)')
    def check(name,array,expected,dtype):
        a=np.asarray(array)
        if a.shape!=expected or a.dtype!=np.dtype(dtype):raise ValueError(f'cuplus restart {name} shape/dtype mismatch (E78)')
        if not np.isfinite(a).all():raise ValueError(f'cuplus restart {name} contains nonfinite values (E78)')
    check('W0AVG',value.w0avg,(ncol,nz),np.float32)
    for key,array in value.held.items():
        expected=(ncol,nz+1 if key=='CMFMC' else nz) if key in profiles else (ncol,)
        check(key,array,expected,np.int32 if key in INTEGER_FIELDS else np.float32)
    if scheme==7 and np.any(np.asarray(value.held['ENTROPY_FAILED'])!=0):
        raise ValueError('CAM-ZM original entropy inversion failed; refusing checkpoint (E78)')
