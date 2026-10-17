"""Fail-closed restart admission for CAM-MG's held REAL histories."""
from collections.abc import Mapping
import numpy as np
from .registry import STATE_FIELDS, INTERFACE_FIELDS, SURFACE_FIELDS

SCHEMA_VERSION=1


def validate(fields,namelist,version):
    from gpuwrf.io.restart_schema import physics_option
    get=fields.get if isinstance(fields,Mapping) else lambda n:getattr(fields,n,None)
    active=physics_option(namelist,'mp_physics',8)==11
    present=any(get(n) is not None for n in STATE_FIELDS)
    if not active and not present and version is None:
        return
    if not active or version!=SCHEMA_VERSION:
        raise ValueError('CAM-MG restart selection/schema differs (E78)')
    shape=np.shape(get('qv'))
    if len(shape)!=3:
        raise ValueError('CAM-MG restart needs a mass-grid domain (E78)')
    for name in (*STATE_FIELDS,'Nc','Ni','Nr','Ns'):
        value=get(name)
        expected=((shape[0]+1,*shape[1:]) if name in INTERFACE_FIELDS else
                  shape[1:] if name in SURFACE_FIELDS else shape)
        dtype=np.int32 if name=='cam_fatal' else np.float32
        if value is None or np.shape(value)!=expected or (np.dtype(value.dtype)!=np.dtype(dtype) and not (name in ("Nc","Ni","Nr","Ns") and np.dtype(value.dtype)==np.dtype("float64"))):
            raise ValueError(f'CAM-MG restart missing/wrong shape/dtype: {name} (E78)')
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'CAM-MG restart nonfinite: {name} (E78)')
    if np.any(np.asarray(get('cam_fatal'))!=0):
        raise ValueError('CAM-MG restart contains a WRF endrun event')
