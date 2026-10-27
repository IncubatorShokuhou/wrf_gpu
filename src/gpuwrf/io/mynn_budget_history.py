"""WRF Registry.EM_COMMON metadata and host-only MYNN history packing.

Registry SHA256 6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a.
The four Z-staggered budgets copy the mass prefix and retain an unused top.
No field is recomputed by the writer.
"""
from gpuwrf.io.mynn_budget_options import configuration

METADATA = {
    'qSHEAR': ('TKE Production - shear','m2 s-2','Z'),
    'qBUOY': ('TKE Production - buoyancy','m2 s-2','Z'),
    'qDISS': ('TKE dissipation','m2 s-2','Z'),
    'qWT': ('TKE vertical transport','m2 s-2','Z'),
    'Dtke': ('TKE change','m2 s-2',''),
    'edmf_a': ('EDMF relative updraft  area - moist updrafts','-',''),
    'edmf_w': ('EDMF vertical velocity  - mean moist updrafts','m s-1',''),
    'edmf_qt': ('EDMF qt  - mean moist updrafts','kg kg-1',''),
    'edmf_thl': ('EDMF thetaL  - mean moist updrafts','K',''),
    'edmf_ent': ('EDMF entrainment  - mean moist updrafts','m-1',''),
    'edmf_qc': ('EDMF qc - mean moist updrafts','kg kg-1',''),
    'sub_thl3D': ('thetaL subsidence tendency from EDMF','K s-1',''),
    'sub_sqv3D': ('qv subsidence tendency from EDMF','kg kg-1 s-1',''),
    'det_thl3D': ('thetaL detrainment tendency from EDMF','K s-1',''),
    'det_sqv3D': ('qv detrainment tendency from EDMF','kg kg-1 s-1',''),
    'RTHBLTEN': ('THETA TENDENCY DUE TO PBL PARAMETERIZATION','K s-1',''),
    'RQVBLTEN': ('Q_V TENDENCY DUE TO PBL PARAMETERIZATION','kg kg-1 s-1',''),
    'RQCBLTEN': ('Q_C TENDENCY DUE TO PBL PARAMETERIZATION','kg kg-1 s-1',''),
    'RQIBLTEN': ('Q_I TENDENCY DUE TO PBL PARAMETERIZATION','kg kg-1 s-1',''),
    'RUBLTEN': ('X WIND TENDENCY DUE TO PBL PARAMETERIZATION','m s-2',''),
    'RVBLTEN': ('Y WIND TENDENCY DUE TO PBL PARAMETERIZATION','m s-2',''),
}

# WRF Registry.EM_COMMON:3190 package mynn_dmp_edmf (bl_mynn_edmf==1):
# ztop_plume, maxmf, maxwidth exist only with EDMF on; history-only (io 'h').
DMP_EDMF_HISTORY = ('MAXMF','MAXWIDTH','ZTOP_PLUME')


def inactive_package_history(namelist):
    """History names whose WRF package is inactive on this resolved domain."""
    return DMP_EDMF_HISTORY if not configuration(namelist)['bl_mynn_edmf'] else ()


def pack_history(diagnostics,namelist,shape,requested=None,*,host_value):
    """Consume already-batched host planes; pad only the original unused top."""
    import numpy as np
    config=configuration(namelist)
    names=tuple(METADATA)
    enabled=(names[:5] if config['tke_budget'] else ())+(names[5:] if config['bl_mynn_output'] else ())
    if requested is not None:
        forbidden=set(requested).intersection(METADATA)-set(enabled)
        if forbidden:raise ValueError(f'MYNN budget fields require their WRF history switch: {sorted(forbidden)}')
        enabled=tuple(name for name in enabled if name in requested)
    result={}
    for name in enabled:
        if diagnostics is None or diagnostics.get(name) is None:
            raise ValueError(f'MYNN budget history missing held {name}')
        value=np.asarray(host_value(diagnostics[name]))
        if value.shape!=tuple(shape) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError(f'MYNN budget {name} needs finite REAL mass-grid history')
        if name in names[:5] and np.count_nonzero(value[-1]):
            raise ValueError(f'MYNN budget {name} unused last mass level must be zero')
        if METADATA[name][2]=='Z':value=np.concatenate((value,np.zeros_like(value[:1])),axis=0)
        result[name]=value
    return result
