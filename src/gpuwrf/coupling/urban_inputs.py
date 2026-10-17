"""WRF LCZ, NUDAPT and anthropogenic-heat input preparation (CPU only at load).

NUDAPT uses deterministic experimental LF(1): parity is with a one-cell
compiler-zero-initialized caller only. WRF reuses its undefined U10/V10 locals
between tile cells; compiler initialization does not reset them per cell.
The sealed two-cell diagnostic differs by HFX 6.273636 W/m2, U10 0.992088 m/s
and TSK 0.301880 K. See docs/UPSTREAM_WRF_BUGS.md; coupled CPU-WRF/D6
qualification is pending. Geometry is computed once in WRF REAL order and
carried as device arrays. No IO in steps.
"""
from __future__ import annotations

from datetime import timedelta
from typing import NamedTuple, Any

import numpy as np


class UrbanMorphology(NamedTuple):
    zr: Any
    r: Any
    rw: Any
    hgt: Any
    zdc: Any
    z0c: Any
    z0hc: Any
    z0r: Any
    svf: Any


class UrbanInputs(NamedTuple):
    morphology: Any = None
    longitude: Any = None
    ahe: Any = None
    hourly: Any = None
    clock: Any = None  # REAL anchors: one-based DOY, GMT hours, UTC minutes, start month


def file_sha256(path):
    """Bounded-memory input seal, outside the timestep loop."""
    import hashlib
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_layered(run,domain,name,shape,count=None):
    """Read WRF's named dimensions into (category,y,x), including XZY inputs."""
    raw=np.ma.asarray(run.load_wrfinput(domain,name,lazy=False),np.float32).filled(np.nan)
    if raw.ndim==4 and raw.shape[0]==1:
        raw=raw[0]
    if hasattr(run,'_read_dataset') and hasattr(run,'wrfinput_file'):
        with run._read_dataset(run.wrfinput_file(domain)) as dataset:
            dims=tuple(d for d in dataset.variables[name].dimensions if d!='Time')
        if len(dims)!=3 or 'south_north' not in dims or 'west_east' not in dims:
            raise ValueError(f'{name}: expected one category axis and WRF mass-grid dimensions')
        y,x=dims.index('south_north'),dims.index('west_east')
        layer=next(i for i in range(3) if i not in (y,x))
        raw=np.transpose(raw,(layer,y,x))
    if raw.ndim!=3 or raw.shape[1:]!=tuple(shape) or (count is not None and raw.shape[0]!=count):
        raise ValueError(f'{name}: expected ({count if count is not None else "categories"},y,x)')
    if not np.isfinite(raw).all():
        raise ValueError(f'{name} contains nonfinite/missing values')
    return raw


def prepare_morphology(table, ivgtyp, isurban, frc, fields):
    """urban_var_init's HGT>0 gate, then urban:680-774 (LF1 zero-init policy)."""
    v = np.asarray(ivgtyp,np.int32)
    shape = v.shape
    T = table.dtype.type
    zero = np.zeros(shape, table.dtype)
    def field(name):
        a = np.asarray(fields.get(name, zero), table.dtype)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'{name}: expected finite urban field of shape {shape}')
        return a
    mh, hgt, lp, lb, std = [field(k) for k in
        ('MH_URB2D','HGT_URB2D','LP_URB2D','LB_URB2D','STDH_URB2D')]
    active = ((v == isurban) | ((v >= 51) & (v <= 61))) & (hgt > 0) & (mh > 0)
    if not active.any():
        return None
    lf = np.asarray(fields.get('LF_URB2D', np.zeros(shape+(4,))), table.dtype)
    if lf.shape != shape+(4,) or not np.isfinite(lf).all():
        raise ValueError('LF_URB2D must have four finite frontal-area directions')
    # Refuse invalid geometries instead of masking their divisions/NaNs.
    if np.any(active & ((lp <= 0) | (lp >= 1) | ((lb > lp) & (frc <= lp)) |
                        (std <= 0) | (lf[...,0] <= 0))):
        raise ValueError('NUDAPT requires 0<LP<1, FRC>LP when LB>LP, STDH>0, LF(1)>0')
    idx = np.where(v == isurban, 4 if table.icate == 11 else 1,
                   np.clip(v-51, 0, table.icate-1))
    # Match literal operation association, rounding each host REAL operation.
    with np.errstate(all='ignore'):
        bw = T(2)*hgt*lp/(lb-lp)
        sw = T(2)*hgt*lp*((frc/lp)-T(1))/(lb-lp)
        bw = np.where(lb>lp,bw,table.building_width[idx])
        sw = np.where(lb>lp,sw,table.street_width[idx])
        nh = mh/(bw+sw)
        rw = T(1)-lp
        zdc = mh*(T(1)+(T(4.43)**(-lp))*(lp-T(1)))
        fac = T(1)-zdc/mh
        z0c = mh*fac*np.exp(-(T(.5)*T(1)*T(1.2)/(T(.4)**T(2))*fac*lf[...,0])**T(-.5))
        z0r = mh*fac*np.exp(-(T(.5)*T(1)*T(1.2)/(T(.4)**T(2))*fac*(std/(sw+bw)))**T(-.5))
        dh = nh/T(100); height = nh-dh/T(2); vf = zero.copy()
        for _ in range(99):
            height = height-dh
            vf = vf+T(.25)*(T(1)-height/np.sqrt(height**T(2)+rw**T(2)))
        vf = vf/T(99); vf = vf*T(2)
        svf = T(1)-T(2)*vf*nh/rw
    values = (mh,lp,rw,nh,zdc,z0c,T(.1)*z0c,z0r,svf)
    names = UrbanMorphology._fields
    out = UrbanMorphology(*(np.where(active,a,np.asarray(getattr(table,k))[idx])
        for k,a in zip(names,values)))
    if not all(np.isfinite(a).all() for a in out):
        raise ValueError('NUDAPT produced nonfinite canyon geometry')
    return out


def urban_clock(bundle, inputs, seconds):
    """Held radiation-driver calc_coszen hour angle, plus current calendar month."""
    import jax.numpy as jnp
    from gpuwrf.coupling.physics_couplers import _wrf_solar_geometry_real
    s = jnp.asarray(seconds,jnp.float32)
    interval = np.float32(bundle.rad_interval_s)
    # WRF HRANG is updated at radiation calls, at the half-radt solar midpoint.
    cadence_s = np.float32(max(1,int(float(interval)/bundle.dt_s+.5))*bundle.dt_s)
    rad_s = s if bundle.solar_every_step else jnp.floor(s/cadence_s)*cadence_s
    if inputs.clock is not None:
        julian,minute,month=inputs.clock[0],inputs.clock[2],inputs.clock[3].astype(jnp.int32)
    else:  # retained convenience API; production carries the anchors as traced inputs.
        start=bundle.start
        minute=np.float32(start.hour*60+start.minute+start.second/60)
        julian=np.float32(start.timetuple().tm_yday)
        month=jnp.int32(start.month)
    geometry = _wrf_solar_geometry_real(jnp.zeros_like(inputs.longitude),inputs.longitude,
        julian, minute, rad_s, rad_s+interval*np.float32(.5))
    # Caller passes grid%julday (simulation START day), not grid%julian.
    return geometry.hour_angle_rad, month


def ahe_at_lead(bundle, inputs, seconds):
    """WRF 288-bin AHE month/hour selection; extension uses hourly UTC hold."""
    import jax.numpy as jnp
    s = jnp.asarray(seconds,jnp.float32)
    if inputs.hourly is not None:
        index=jnp.floor(s/np.float32(3600)).astype(jnp.int32)
        return jnp.take(inputs.hourly,index,axis=0,mode='fill',fill_value=np.float32(np.nan))
    # Both surface/PBL drivers use JULDAY/JULYR fixed at simulation start.
    if inputs.clock is not None:
        month=inputs.clock[3].astype(jnp.int32)
        gmt=inputs.clock[1]
    else:
        month=bundle.start.month
        gmt=np.float32(bundle.start.hour+bundle.start.minute/60+bundle.start.second/3600)
    hour = jnp.mod((gmt+s/np.float32(3600)).astype(jnp.int32),24)
    return inputs.ahe[(month-1)*24+hour]


def pbl_heat_tendency(rthblten, ahe, dz1):
    """Verbatim PBL-driver bottom tendency addition (WRF REAL association)."""
    import jax.numpy as jnp
    dtype = rthblten.dtype
    T = np.dtype(dtype).type
    return rthblten.at[0].add(jnp.asarray(ahe,dtype)/T(1200)/jnp.asarray(dz1,dtype))


def load_hourly(path, *, start, hours, shape):
    """Beyond-WRF extension: Times-aligned hourly AHE(Time,y,x), W m-2, no interpolation."""
    from pathlib import Path
    from datetime import datetime, timezone
    from gpuwrf.io.netcdf_lock import Dataset
    with Dataset(Path(path)) as ds:
        if 'AHE' not in ds.variables or 'Times' not in ds.variables:
            raise ValueError('urban_ahe_file requires AHE and hourly WRF Times')
        var = ds.variables['AHE']
        if str(getattr(var,'units','')).strip() not in ('W m-2','W m^-2','W/m2'):
            raise ValueError('hourly AHE units must be W m-2')
        values = np.ma.asarray(var[:],np.float32).filled(np.nan)
        times = [datetime.strptime(b''.join(row).decode(),'%Y-%m-%d_%H:%M:%S')
                 for row in np.asarray(ds.variables['Times'][:])]
    required = int(np.ceil(hours))
    if values.shape != (len(times),)+tuple(shape) or not np.isfinite(values).all():
        raise ValueError('hourly AHE has invalid shape or nonfinite/missing values')
    # WRF Times are UTC without a zone suffix; the operational pipeline uses
    # aware UTC datetimes, while the standalone API also accepts naive UTC.
    utc_start=start.astimezone(timezone.utc).replace(tzinfo=None) if start.tzinfo is not None else start
    wanted = [utc_start+timedelta(hours=h) for h in range(len(times))]
    if times != wanted or len(times) < required:
        raise ValueError('hourly AHE must cover every forecast hour from the exact run start')
    # Keep the declared stream so --extend-run can consume its later hours safely.
    return values
