"""Backend-dark admission of the qualified Goddard caller configuration."""
from collections.abc import Mapping


def _value(options,key,default):
    if isinstance(options,Mapping):
        value=options.get('physics',options).get(key,default)
    else:value=getattr(options,key,default)
    # OperationalNamelist's legacy land toggle leaves this selector unset.
    return default if key=='sf_surface_physics' and value is None else value


def selection_errors(options):
    values={key:_value(options,key,default) for key,default in {
        'ra_lw_physics':4,'ra_sw_physics':4,'mp_physics':8,'use_mp_re':1,
        'gsfcrad_gocart_coupling':0,'aer_ra_feedback':0,'topo_shading':0,'slope_rad':0,'swint_opt':0,
        'sf_surface_physics':4,'sf_urban_physics':0,'sf_lake_physics':0}.items()}
    n=max([len(v) for v in values.values() if isinstance(v,(list,tuple))] or [1])
    errors=[]
    for d in range(n):
        c={k:int(v[min(d,len(v)-1)] if isinstance(v,(list,tuple)) else v) for k,v in values.items()}
        if 5 not in (c['ra_lw_physics'],c['ra_sw_physics']):continue
        expected=dict(ra_lw_physics=5,ra_sw_physics=5,mp_physics=8,use_mp_re=1,
                      gsfcrad_gocart_coupling=0,aer_ra_feedback=0,topo_shading=0,slope_rad=0,swint_opt=0,
                      sf_lake_physics=0)
        bad=[f'{k}={v}' for k,v in expected.items() if c[k]!=v]
        if c['sf_surface_physics'] not in (0,3,4):
            bad.append('sf_surface_physics=0,3,4 (other land models lack a qualified held ALBEDO/EMISS view)')
        if c['sf_urban_physics'] not in (0,1):
            bad.append('sf_urban_physics=0,1 (BEP/BEM requires RRTMG4/4)')
        if bad:errors.append(f'Goddard domain {d+1} requires '+', '.join(bad)+
            '; other MP effective-radius mediation and mixed radiation pairs are not qualified')
    return errors


def validate_selection(options):
    errors=selection_errors(options)
    if errors:raise ValueError('; '.join(errors))


def validate_geometry(options):
    """Loader preflight only; no device transfer from the timestep loop."""
    import numpy as np
    static=getattr(options,'radiation_static',None)
    if static is None:raise ValueError('Goddard radiation requires supplied WRF XLAT/XLONG radiation_static')
    lat=np.asarray(static.xlat_deg,dtype=np.float32)
    if not np.isfinite(lat).all() or np.any(np.abs(lat)>90):
        raise ValueError('Goddard radiation requires finite WRF latitude within [-90,90]')
    if np.any(lat==0):
        raise ValueError('Goddard WRF4.7.1 sounding_interp at exact latitude 0 reads mcdat index11 outside1:10; equator is unsupported')
