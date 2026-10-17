"""Host-only Fitch turbine input validation and WRF map/nearest-grid placement."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class WindFarm:
    # (zero-based x,y,(hub,diameter,standing,rated_kW,((speed,Ct,power_kW),...)))
    turbines: tuple
    tke_factor: float = .25
    fingerprint: str = ""

    def __hash__(self):
        return hash(self.fingerprint) if self.fingerprint else hash((self.turbines,self.tke_factor))


def _rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split('!')[0].strip()
        if not line:
            raise ValueError(f'{path}: blank turbine records are not supported')
        try:
            values = tuple(float(x.replace('D','e').replace('d','e')) for x in line.split())
        except ValueError as exc:
            raise ValueError(f'{path}: invalid numeric record') from exc
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f'{path}: NaN/Inf turbine input')
        with np.errstate(over='ignore'):
            if not np.isfinite(np.asarray(values,np.float32)).all():
                raise ValueError(f'{path}: turbine input overflows WRF REAL32')
        rows.append(values)
    return rows


def read_turbine_table(path):
    rows = _rows(path)
    if len(rows) < 4 or len(rows[0]) != 1 or rows[0][0] != int(rows[0][0]):
        raise ValueError(f'{path}: expected nval, hub/diameter/Ct/rated, then speed/Ct/kW rows')
    count = int(rows[0][0])
    if not 2 <= count <= 100 or len(rows) != count+2 or len(rows[1]) != 4 or any(len(r) != 3 for r in rows[2:]):
        raise ValueError(f'{path}: invalid table count or dimensions (WRF maximum 100)')
    hub, diameter, standing, rated = rows[1]
    curves = rows[2:]
    if hub <= 0 or diameter <= 0 or hub < diameter/2 or standing < 0 or rated < 0:
        raise ValueError(f'{path}: invalid rotor geometry/standing thrust/rated power')
    real_hub,real_diameter=np.float32(hub),np.float32(diameter)
    if real_hub<=0 or real_diameter<=0 or real_hub<real_diameter/np.float32(2.):
        raise ValueError(f'{path}: rotor geometry underflows or becomes invalid in WRF REAL32')
    if curves[0][0] <= 0 or any(b[0] <= a[0] for a,b in zip(curves,curves[1:])) or any(min(r[1:]) < 0 for r in curves):
        raise ValueError(f'{path}: speeds must be positive and strictly increasing; Ct/kW nonnegative')
    real_speeds=np.asarray([r[0] for r in curves],np.float32)
    if real_speeds[0]<=0 or not (np.diff(real_speeds)>0).all():
        raise ValueError(f'{path}: curve speeds collapse or underflow in WRF REAL32')
    return (hub, diameter, standing, rated, tuple(curves))


def read_windfarm(case_dir, *, ij, nx, ny, projection=None, tke_factor=.25):
    root = Path(case_dir)
    if ij not in (0,1) or not math.isfinite(tke_factor) or not 0 <= tke_factor <= np.finfo(np.float32).max:
        raise ValueError('windfarm_ij must be 0/1 and windfarm_tke_factor finite/nonnegative')
    location = root/('windturbines-ij.txt' if ij else 'windturbines.txt')
    records = _rows(location)
    turbines, tables = [], {}
    digest = hashlib.sha256(location.read_bytes())
    for row in records:
        if len(row) != 3 or row[2] != int(row[2]) or not 1 <= row[2] <= 100:
            raise ValueError(f'{location}: expected two coordinates and integer type 1..100')
        kind = int(row[2])
        if kind not in tables:
            path = root/f'wind-turbine-{kind}.tbl'
            tables[kind] = read_turbine_table(path)
            digest.update(path.read_bytes())
        if ij:
            if any(v != int(v) for v in row[:2]):
                raise ValueError(f'{location}: i,j must be WRF one-based integers')
            i,j = map(int,row[:2])
        else:
            if projection is None or projection.code not in (1,2,3):
                raise ValueError('Fitch lat/lon placement requires WRF Lambert, polar stereographic or Mercator projection')
            if not -90 <= row[0] <= 90 or not -180 <= row[1] <= 180:
                raise ValueError(f'{location}: invalid latitude/longitude')
            # Shared map algebra returns mass i+.5,j+.5 for OBS; Fitch uses
            # latlon_to_ij directly, then Fortran NINT (ties away from zero).
            rx,ry = projection.station_coordinates(*row[:2],mass_offset=False)
            nint = lambda a: int(math.copysign(math.floor(abs(float(a))+.5),a))
            i,j = nint(rx),nint(ry)
        if 1 <= i <= nx and 1 <= j <= ny:
            turbines.append((i-1,j-1,tables[kind]))
    digest.update(repr((tuple(turbines),float(tke_factor))).encode())
    return WindFarm(tuple(turbines),float(tke_factor),digest.hexdigest())


def windfarm_option(nl,domain='d01'):
    """WRF array defaults: omitted domain entries remain zero, not the last value."""
    raw=(nl.get('physics') or {}).get('windfarm_opt')
    index=int(domain[1:])-1
    value=(raw[index] if index<len(raw) else None) if isinstance(raw,(list,tuple)) else (raw if index==0 else None)
    return 0 if value is None else value


def windfarm_errors(nl, domain='d01'):
    from gpuwrf.io.namelist_binding import namelist_value
    get = lambda key,default: default if namelist_value(nl,key,domain) is None else namelist_value(nl,key,domain)
    opt = windfarm_option(nl,domain)
    if opt == 0:
        return []
    errors = []
    for key in ('windfarm_ij','windfarm_tke_factor'):
        raw=(nl.get('physics') or {}).get(key)
        if isinstance(raw,(list,tuple)) and len(raw)>1:
            errors.append(f'{key} is a WRF scalar; provide one value for all domains')
    if opt != 1:
        errors.append('windfarm_opt: only Fitch=1 is ported; MAV=2 is not qualified')
    for key,default in (('bl_pbl_physics',5),('sf_sfclay_physics',5)):
        if get(key,default) != 5:
            errors.append(f'windfarm_opt=1 requires {key}=5 (MYNN)')
    if get('windfarm_ij',0) not in (0,1):
        errors.append('windfarm_ij must be 0 or 1')
    factor = get('windfarm_tke_factor',.25)
    if not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 0 <= factor <= np.finfo(np.float32).max:
        errors.append('windfarm_tke_factor must be finite/nonnegative')
    if get('bldt',0) != 0:
        errors.append('windfarm_opt=1 requires bldt=0 (every-step PBL)')
    return errors


def bind_windfarm(namelist, case_dir, domain):
    if int(namelist.windfarm_opt) == 0:
        return namelist
    if int(namelist.windfarm_opt) != 1 or int(namelist.bl_pbl_physics) != 5 or int(namelist.sf_sfclay_physics) != 5:
        raise ValueError('Fitch requires windfarm_opt=1, MYNN PBL=5 and surface layer=5')
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.obs_projection import ObservationProjection
    with Dataset(Path(case_dir)/f'wrfinput_{domain}') as ds:
        nx,ny = len(ds.dimensions['west_east']),len(ds.dimensions['south_north'])
        projection = None
        if namelist.windfarm_ij == 0:
            projection = ObservationProjection(int(ds.MAP_PROJ),float(ds['XLAT'][0,0,0]),float(ds['XLONG'][0,0,0]),
                         float(ds.TRUELAT1),float(ds.TRUELAT2),float(ds.STAND_LON),float(ds.DX),float(ds.DY))
        farm = read_windfarm(case_dir,ij=namelist.windfarm_ij,nx=nx,ny=ny,projection=projection,
                             tke_factor=namelist.windfarm_tke_factor)
        z = (np.asarray(ds['PH'][0],np.float32)+np.asarray(ds['PHB'][0],np.float32))/np.float32(9.81)
        for x,y,spec in farm.turbines:
            profile = z[:,y,x]-z[0,y,x]
            if not np.isfinite(profile).all() or not (np.diff(profile)>0).all():
                raise ValueError('Fitch requires finite strictly increasing WRF interface heights')
            mid = .5*(profile[:-1]+profile[1:])
            hub,diameter=np.float32(spec[0]),np.float32(spec[1])
            if not mid[0] <= hub < mid[-1] or hub+diameter/np.float32(2.) > profile[-1]:
                raise ValueError('Fitch hub must be bracketed by mass levels and full rotor inside model column; WRF has undefined hub indices otherwise')
    return replace(namelist,windfarm=farm)
