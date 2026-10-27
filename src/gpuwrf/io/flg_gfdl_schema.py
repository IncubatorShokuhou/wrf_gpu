"""E78: CPU-only legacy radiation retains its native rates/fluxes/clouds."""
import numpy as np


def marker(diagnostics,gfdl):
    if type(diagnostics).__name__!='LegacyRadiationDiagnostics':return None
    return {'version':1,'scheme':99 if gfdl is not None else 7}


def validate(diagnostics,gfdl,namelist=None,schema=None,*,state_shape=None):
    from gpuwrf.io.restart_schema import physics_option
    lw=physics_option(namelist,'ra_lw_physics',4);sw=physics_option(namelist,'ra_sw_physics',4)
    own=type(diagnostics).__name__=='LegacyRadiationDiagnostics'
    if not own and not ({lw,sw}&{7,99}) and schema is None and gfdl is None:return
    if schema not in ({'version':1,'scheme':7},{'version':1,'scheme':99}):
        raise ValueError('FLG/GFDL restart schema/selection missing (E78)')
    scheme=schema['scheme']
    if namelist is not None and (lw,sw)!=(scheme,scheme):
        raise ValueError('FLG/GFDL restart radiation selection differs (E78)')
    from gpuwrf.coupling.flg_gfdl_radiation import LegacyRadiationDiagnostics,GfdlCarry
    if not isinstance(diagnostics,LegacyRadiationDiagnostics):
        raise ValueError('FLG/GFDL restart missing held native radiation diagnostics (E78)')
    shape=np.shape(diagnostics.cloud_fraction);surface=shape[1:]
    if len(shape)!=3 or (state_shape is not None and shape!=tuple(state_shape)):
        raise ValueError('FLG/GFDL restart cloud grid differs (E78)')
    grid=getattr(namelist,'grid',None)
    if grid is not None and shape!=(grid.nz,grid.ny,grid.nx):
        raise ValueError('FLG/GFDL restart domain differs (E78)')
    def array(name,value,expected,dtype):
        if value is None or np.shape(value)!=expected or np.asarray(value).dtype!=dtype or not np.isfinite(value).all():
            raise ValueError(f'FLG/GFDL restart {name} missing/shape/dtype/nonfinite differs (E78)')
    from gpuwrf.coupling.physics_couplers import unproduced_optional_radiation_fields
    absent=unproduced_optional_radiation_fields(('cloud_fraction',))
    for name in diagnostics._fields:
        value=getattr(diagnostics,name)
        if name in absent:
            if value is not None:raise ValueError(f'FLG/GFDL restart has foreign optional {name} (E78)')
        else:array(name,value,shape if name=='cloud_fraction' else surface,np.int32 if name=='shadow_mask' else np.float32)
    if scheme==7:
        if gfdl is not None:raise ValueError('FLG restart contains GFDL carry (E78)')
    else:
        if not isinstance(gfdl,GfdlCarry):raise ValueError('GFDL restart missing cloud counters/held rates (E78)')
        for name in gfdl._fields:
            array(name,getattr(gfdl,name),shape if name in ('lw_rate','sw_rate') else surface,
                  np.int32 if name in ('ncfrcv','ncfrst') else np.float32)
