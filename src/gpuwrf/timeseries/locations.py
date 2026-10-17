"""Host-only WRF spherical map projection/NINT station placement.

Transcription of share/module_llxy.F set_lc/llij_lc, set_ps/llij_ps and
set_merc/llij_merc, using the WRF sphere (6,370,000 m). No land/water search:
calc_ts_locations in WRF does not perform one.
"""
from __future__ import annotations

import numpy as np

from .files import StationLocation


def latlon_to_ij(latitude, longitude, attrs, lat1, lon1):
    f = np.float32
    lat, lon, lat1, lon1 = map(f, (latitude, longitude, lat1, lon1))
    pi = f(3.141592653589793)
    rad = pi / f(180)
    dx, tl1 = f(attrs['DX']), f(attrs['TRUELAT1'])
    rebydx = f(6370000) / dx
    hemi = f(1 if tl1 >= 0 else -1)
    stdlon = f(attrs['STAND_LON'])

    def delta(a, b):
        d = a - b
        if d > f(180):
            d -= f(360)
        if d < f(-180):
            d += f(360)
        return d

    projection = int(attrs['MAP_PROJ'])
    if projection == 1:
        tl2 = f(attrs.get('TRUELAT2', tl1))
        if abs(tl1 - tl2) > f(.1):
            cone = (np.log10(np.cos(tl1 * rad)) - np.log10(np.cos(tl2 * rad)))
            cone /= (np.log10(np.tan((f(45) - abs(tl1)/f(2)) * rad))
                     - np.log10(np.tan((f(45) - abs(tl2)/f(2)) * rad)))
        else:
            cone = np.sin(abs(tl1) * rad)
        ctl1r = np.cos(tl1 * rad)

        def radius(la):
            return (rebydx * ctl1r / cone
                    * (np.tan((f(90)*hemi-la)*rad/f(2))
                       / np.tan((f(90)*hemi-tl1)*rad/f(2)))**cone)

        rsw = radius(lat1)
        arg = cone * (delta(lon1, stdlon) * rad)
        polei = hemi - hemi * rsw * np.sin(arg)
        polej = hemi + rsw * np.cos(arg)
        rm = radius(lat)
        arg = cone * (delta(lon, stdlon) * rad)
        return hemi * (polei + hemi*rm*np.sin(arg)), hemi * (polej - rm*np.cos(arg))
    if projection == 2:
        reflon = stdlon + f(90)
        scale = f(1) + hemi * np.sin(tl1 * rad)

        def radius(la):
            return rebydx*np.cos(la*rad)*scale/(f(1)+hemi*np.sin(la*rad))

        rsw = radius(lat1)
        alo1 = (lon1-reflon)*rad
        polei = f(1)-rsw*np.cos(alo1)
        polej = f(1)-hemi*rsw*np.sin(alo1)
        rm, alo = radius(lat), (lon-reflon)*rad
        return polei + rm*np.cos(alo), polej + hemi*rm*np.sin(alo)
    if projection == 3:
        dlon = dx/(f(6370000)*np.cos(rad*tl1))
        rsw = f(0) if lat1 == 0 else np.log(np.tan(f(.5)*((lat1+f(90))*rad)))/dlon
        return (f(1)+delta(lon, lon1)/(dlon*(f(180)/pi)),
                f(1)+np.log(np.tan(f(.5)*((lat+f(90))*rad)))/dlon-rsw)
    raise ValueError(f"tslist LAT/LON supports WRF MAP_PROJ=1,2,3; got {projection}; use I/J")


def locate_stations(stations, domain_id, lat, lon, terrain, attrs):
    """Locate independently on each domain; preserve original tslist IDs.

    WRF allows an extra high-edge halo (ide/jde). gpuwrf has no output halo, so
    an otherwise in-range high-edge halo station is refused explicitly rather
    than silently clamped or sampled from another cell.
    """
    lat, lon, terrain = map(np.asarray, (lat, lon, terrain))
    if lat.ndim != 2 or lon.shape != lat.shape or terrain.shape != lat.shape:
        raise ValueError("time series coordinates/terrain must have identical 2-D shapes")
    ny, nx = lat.shape
    locations = []
    for station in stations:
        if station.ij or int(attrs['MAP_PROJ']) == 0:
            rx, ry = station.first, station.second
        else:
            rx, ry = latlon_to_ij(station.first, station.second, attrs, lat[0, 0], lon[0, 0])
        # Fortran NINT rounds ties away from zero (Python round uses ties-to-even).
        i, j = (int(np.copysign(np.floor(abs(v) + .5), v)) for v in (rx, ry))
        if not (1 <= i <= nx + 1 and 1 <= j <= ny + 1):
            continue
        if i > nx or j > ny:
            raise ValueError(f"tslist {station.prefix}: WRF high-edge halo ({i},{j}) is unavailable")
        values = lat[j-1, i-1], lon[j-1, i-1], terrain[j-1, i-1]
        if not np.isfinite(values).all():
            raise ValueError("time series location has non-finite metadata")
        locations.append(StationLocation(station, int(domain_id), i, j, *map(float, values)))
    return tuple(locations)
