"""Instantaneous synthetic radar products on WRF mass levels (K0)."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .radar_rip import rip_reflectivity
from .radar_thompson import thompson_reflectivity

PROFILE = ("Time", "bottom_top", "south_north", "west_east")
SURFACE = ("Time", "south_north", "west_east")
REQUIRED = ("PH", "PHB", "HGT")
OPTIONAL = ("REFL_10CM", "P", "PB", "T", "QVAPOR", "QRAIN", "QNRAIN", "QSNOW", "QGRAUP", "QNGRAUPEL", "QHAIL")


def _field(ds, name, dims=PROFILE):
    if name not in ds:
        raise ValueError(f"radar requires {name}")
    da = ds[name]
    if set(da.dims) != set(dims):
        raise ValueError(f"{name}: expected {dims}, got {da.dims}")
    a = np.asarray(da.transpose(*dims).values)
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: NaN/Inf input")
    return a


def column_products(dbz, height, *, echo_threshold=18., floor=-35.):
    """Composite (dBZ), echo top (m), flags and VIL (kg/m²).

Arrays (..., z), increasing height in m AGL. Echo top interpolates the highest
threshold crossing linearly in dBZ. 0=no echo; 1=resolved; 2=top-censored.
VIL follows Greene & Clark: 3.44e-6 sum[((Zi+Zi+1)/2)**(4/7)*dz],
linear Z capped at 56 dBZ, floor treated as zero. No extrapolation above/below
mass-level observations, so the reported integral excludes the surface layer.
"""
    z, h = np.asarray(dbz, float), np.asarray(height, float)
    if z.shape != h.shape or z.ndim < 1 or z.shape[-1] < 2:
        raise ValueError("radar columns need identical shapes and >=2 vertical levels")
    if not np.isfinite(z).all() or not np.isfinite(h).all() or not np.isfinite(echo_threshold) or not np.isfinite(floor):
        raise ValueError("radar columns/threshold/floor must be finite")
    if echo_threshold <= floor:
        raise ValueError("echo threshold must exceed reflectivity floor")
    if np.any(h < 0) or np.any(np.diff(h, axis=-1) <= 0):
        raise ValueError("radar heights must be nonnegative and strictly increase")
    hits = z >= echo_threshold
    highest = np.max(np.where(hits, np.arange(z.shape[-1]), -1), axis=-1)
    lo = np.maximum(highest, 0)
    hi = np.minimum(lo+1, z.shape[-1]-1)
    at = lambda a,k: np.take_along_axis(a,k[...,None],axis=-1)[...,0]
    zl, zh, hl, hh = at(z,lo), at(z,hi), at(h,lo), at(h,hi)
    fraction = np.divide(echo_threshold-zl, zh-zl, out=np.zeros_like(zl), where=zh != zl)
    top = np.where(highest < 0, 0., hl+fraction*(hh-hl))
    flags = np.where(highest < 0, 0, np.where(highest == z.shape[-1]-1, 2, 1)).astype('i1')
    linear = np.where(z <= floor, 0., 10**(np.minimum(z,56.)/10))
    vil = 3.44e-6*np.sum(((linear[...,1:]+linear[...,:-1])/2)**(4/7)*np.diff(h,axis=-1),axis=-1)
    return np.max(z,axis=-1), top, flags, vil


def radar(ds, options=None):
    """Registry runner: saved S reflectivity, explicit Thompson or RIP fallback.

Options: radar_band S/C/X; radar_method auto/saved/thompson/rip;
radar_echo_dbz (default18); radar_variable_intercept, radar_liquid_skin (RIP).
"""
    options = dict(options or {})
    band = options.get('radar_band', 'S')
    method = options.get('radar_method', 'auto')
    if band not in ('S','C','X') or method not in ('auto','saved','thompson','rip'):
        raise ValueError('invalid radar_band or radar_method')
    if method == 'auto':
        if band == 'S' and 'REFL_10CM' in ds:
            method = 'saved'
        elif int(ds.attrs.get('MP_PHYSICS', -1)) == 8:
            method = 'thompson'
        else:
            raise ValueError('radar auto needs saved REFL_10CM or MP_PHYSICS=8; select radar_method="rip" explicitly for the documented fallback')
    if method == 'saved':
        if band != 'S':
            raise ValueError('saved REFL_10CM is S-band only; C/X require hydrometeor scattering')
        dbz = _field(ds,'REFL_10CM').copy()
        floor = -35.
        definition = 'WRF-written REFL_10CM, preserved exactly; scheme/diagnostic-call sampling inherited from input'
    else:
        p = _field(ds,'P').astype(float)+_field(ds,'PB')
        t = (_field(ds,'T').astype(float)+300)*(p/100000)**(287/1004.5)
        vapor, rain = _field(ds,'QVAPOR'), _field(ds,'QRAIN')
        snow = _field(ds,'QSNOW') if 'QSNOW' in ds else np.zeros_like(rain)
        graupel = _field(ds,'QGRAUP') if 'QGRAUP' in ds else np.zeros_like(rain)
        if 'QHAIL' in ds and np.any(_field(ds,'QHAIL') > 1e-12):
            raise ValueError('active hail requires a scheme-specific scattering operator; unsupported')
        if method == 'thompson':
            if int(ds.attrs.get('MP_PHYSICS', -1)) != 8:
                raise ValueError('Thompson reconstruction currently requires MP_PHYSICS=8')
            for name in ('QSNOW','QGRAUP'):
                if name not in ds:
                    raise ValueError(f'Thompson reconstruction requires {name}')
            if 'QNGRAUPEL' not in ds and np.any(graupel > 1e-6):
                raise ValueError('active Thompson graupel needs QNGRAUPEL (#/kg) from the diagnostic-call context; standard mp8 history omits it. Use saved REFL_10CM for S-band.')
            ng = _field(ds,'QNGRAUPEL') if 'QNGRAUPEL' in ds else np.zeros_like(rain)
            nr = _field(ds,'QNRAIN')
            a = [np.moveaxis(x,1,-1) for x in (vapor,rain,nr,snow,graupel,ng,t,p)]
            dbz = np.moveaxis(thompson_reflectivity(*a, band=band),-1,1)
            floor = -35.
            definition = ('Thompson mp8 distributions; history-field reconstruction, not diagnostic-call-time identity; '
                          +('WRF Rayleigh + wet snow' if band == 'S' else 'full Mie effective homogeneous spheres, WRF nested Maxwell-Garnett wet snow; no attenuation/beam geometry/polarimetry; fixed 0C dielectric'))
        else:
            if band != 'S':
                raise ValueError('RIP fallback is S-band only; no wavelength scaling allowed')
            flags = []
            for key in ('radar_variable_intercept','radar_liquid_skin'):
                val = options.get(key,False)
                if not isinstance(val,bool):
                    raise ValueError(f'{key} must be a JSON boolean')
                flags.append(val)
            dbz = rip_reflectivity(p,t,vapor,rain,snow,graupel,variable_intercept=flags[0],liquid_skin=flags[1],snow_present='QSNOW' in ds)
            floor = -30.
            definition = 'NCAR wrf-python RIP CALCDBZ fallback; exponential PSD; NOT scheme-specific REFL_10CM parity'
    ph = _field(ds,'PH',('Time','bottom_top_stag','south_north','west_east')).astype(float)
    ph += _field(ds,'PHB',('Time','bottom_top_stag','south_north','west_east'))
    terrain = _field(ds,'HGT',SURFACE)
    height = (ph[:,:-1]+ph[:,1:])/(2*9.81)-terrain[:,None]
    composite, top, flags, vil = column_products(np.moveaxis(dbz,1,-1),np.moveaxis(height,1,-1),
                                               echo_threshold=options.get('radar_echo_dbz',18.),floor=floor)
    suffix = band.lower()
    attrs = {'radar_band':band,'wavelength_m':{'S':.1,'C':.056,'X':.032}[band],
             'radar_method':method,'definition':definition,'cell_methods':'time: point'}
    def da(values,dims,units,name,**extra):
        return xr.DataArray(values,dims=dims,attrs={**attrs,'units':units,'long_name':name,**extra})
    result = xr.Dataset({
        f'radar_dbz_{suffix}':da(dbz,PROFILE,'dBZ','Equivalent synthetic radar reflectivity'),
        f'radar_composite_{suffix}':da(composite,SURFACE,'dBZ','Maximum sampled column reflectivity'),
        f'radar_echo_top_{suffix}':da(top,SURFACE,'m','Echo top above ground',threshold_dbz=float(options.get('radar_echo_dbz',18.)),no_echo_value=0.),
        f'radar_echo_top_status_{suffix}':da(flags,SURFACE,'1','Echo top status',flag_values=np.array([0,1,2],dtype='i1'),flag_meanings='no_echo resolved model_top_censored'),
        f'radar_vil_{suffix}':da(vil,SURFACE,'kg m-2','Radar equivalent vertically integrated liquid',definition='Greene-Clark linear-Z layer mean; 56dBZ cap; floor is zero; integral from lowest to highest mass level only; liquid-equivalent radar proxy, not actual model condensate'),
    })
    if 'Time' in ds.coords:
        result = result.assign_coords(Time=ds.Time)
    return result


def register_products():
    """Optional post-registry extension, idempotent like apps.register_products."""
    from .registry import PRODUCTS, register
    if 'radar' not in PRODUCTS:
        register('radar',radar,REQUIRED,OPTIONAL)
