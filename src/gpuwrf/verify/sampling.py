"""Reusable station stencils on a static WRF mass grid."""
from __future__ import annotations

import numpy as np

from .observations import finite


def _longitude_delta(a, b):
    return (a - b + 180) % 360 - 180


def _cell_coordinates(points):
    """Invert the bilinear quadrilateral map at the origin with Newton steps."""
    p00, p10, p01, p11 = points
    a, b, c = p10 - p00, p01 - p00, p11 - p10 - p01 + p00
    x = y = 0.5
    for _ in range(15):
        residual = p00 + a*x + b*y + c*x*y
        if np.max(np.abs(residual)) < 1e-10:
            break
        jac = np.column_stack((a + c*y, b + c*x))
        if abs(np.linalg.det(jac)) < 1e-14:
            return None
        dx, dy = np.linalg.solve(jac, residual)
        x, y = x-dx, y-dy
    residual = p00 + a*x + b*y + c*x*y
    if np.max(np.abs(residual)) > 1e-8 or not (-1e-8 <= x <= 1+1e-8 and -1e-8 <= y <= 1+1e-8):
        return None
    x, y = np.clip([x, y], 0, 1)
    return np.array([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y])


class StationSampler:
    """Nearest or bilinear; coastal weights retain only the station's surface class.

    Bilinear here means a four-node isoparametric lat/lon quadrilateral,
    locally unwrapped about each station. Out-of-domain stations are rejected.
    """

    def __init__(self, lat, lon, stations, *, landmask=None, method="bilinear", land_aware=True):
        self.lat, self.lon = finite(lat, "XLAT"), finite(lon, "XLONG")
        self.shape = self.lat.shape
        if self.lat.ndim != 2 or min(self.shape) < 2 or self.lon.shape != self.shape:
            raise ValueError("latitude/longitude must share a >=2x2 mass grid")
        if method not in ("nearest", "bilinear"):
            raise ValueError("method must be nearest or bilinear")
        if land_aware:
            if landmask is None:
                raise ValueError("land-aware sampling requires LANDMASK")
            landmask = finite(landmask, "LANDMASK")
            if landmask.shape != self.shape or not np.isin(landmask, [0, 1]).all():
                raise ValueError("LANDMASK must be a binary mass-grid field")
        self.nodes, self.weights = [], []
        ny, nx = self.shape
        for station in stations.itertuples():
            dx = _longitude_delta(self.lon, station.lon) * np.cos(np.deg2rad(station.lat))
            dy = self.lat - station.lat
            dist = dx*dx + dy*dy
            j, i = np.unravel_index(np.argmin(dist), self.shape)
            stencil = None
            for y in (j-1, j):
                for x in (i-1, i):
                    if not (0 <= y < ny-1 and 0 <= x < nx-1):
                        continue
                    nodes = np.array([y*nx+x, y*nx+x+1, (y+1)*nx+x, (y+1)*nx+x+1])
                    points = np.column_stack((dx.ravel()[nodes], dy.ravel()[nodes]))
                    weights = _cell_coordinates(points)
                    if weights is not None:
                        stencil = nodes, weights
                        break
                if stencil is not None:
                    break
            if stencil is None:
                raise ValueError(f"station {station.station_uid} outside supported grid cells")
            nodes, weights = stencil
            if method == "nearest":
                # Nearest compatible mass point, bounded by the containing cell.
                eligible = np.ones(4, dtype=bool) if not land_aware else landmask.ravel()[nodes] == station.landmask
                if not eligible.any():
                    raise ValueError(f"station {station.station_uid}: no matching land/sea point")
                pick = np.argmin(np.where(eligible, dist.ravel()[nodes], np.inf))
                weights = np.eye(4)[pick]
            elif land_aware:
                weights *= landmask.ravel()[nodes] == station.landmask
                if weights.sum() <= 1e-12:
                    raise ValueError(f"station {station.station_uid}: no positive matching land/sea weight")
                weights /= weights.sum()
            self.nodes.append(nodes)
            self.weights.append(weights)
        self.nodes, self.weights = np.array(self.nodes), np.array(self.weights)

    def sample(self, field):
        values = finite(field, "sampled field")
        if values.shape != self.shape:
            raise ValueError("field does not match mass grid")
        return np.sum(values.ravel()[self.nodes] * self.weights, axis=1)


def earth_wind(u, v, lat, lon, attrs, *, cosalpha=None, sinalpha=None, frame="grid"):
    """NCAR wrf-python DCOMPUTEUVMET equations (unstaggered U10/V10)."""
    if frame == "earth":
        return u, v
    if frame != "grid":
        raise ValueError("wind_frame must be grid or earth")
    if cosalpha is not None and sinalpha is not None:
        c, s = finite(cosalpha, "COSALPHA"), finite(sinalpha, "SINALPHA")
        if c.shape != u.shape or s.shape != u.shape or not np.allclose(c*c+s*s, 1, atol=1e-5):
            raise ValueError("invalid wind rotation arrays")
        # WRF COS/SINALPHA convention is opposite to DCOMPUTEUVMET longitude angle.
        return u*c - v*s, v*c + u*s
    projection = int(attrs.get("MAP_PROJ", -1))
    if projection == 3:  # Mercator
        return u, v
    if projection == 6 and float(attrs.get("POLE_LAT", 90)) == 90:
        return u, v
    if projection not in (1, 2):
        raise ValueError("grid wind needs COSALPHA/SINALPHA or supported MAP_PROJ metadata; use wind_frame=earth only for earth-relative inputs")
    try:
        stand = float(attrs["STAND_LON"])
        t1 = abs(float(attrs["TRUELAT1"]))
        t2 = abs(float(attrs.get("TRUELAT2", t1)))
    except KeyError as exc:
        raise ValueError(f"missing projection metadata {exc}") from exc
    if projection == 2:
        cone = 1.0
    elif abs(t1-t2) > 0.1 and abs(t2-90) > 0.1:
        cone = np.log(np.cos(np.deg2rad(t1))/np.cos(np.deg2rad(t2))) / np.log(
            np.tan(np.deg2rad(45-t1/2))/np.tan(np.deg2rad(45-t2/2)))
    else:
        cone = np.sin(np.deg2rad(t1))
    angle = np.deg2rad(_longitude_delta(lon, stand))*cone*np.where(lat < 0, -1, 1)
    c, s = np.cos(angle), np.sin(angle)
    return finite(u*c + v*s, "earth u"), finite(v*c-u*s, "earth v")
