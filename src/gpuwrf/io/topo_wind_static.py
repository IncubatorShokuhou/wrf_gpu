"""WRF ``topo_wind`` surface-drag correction fields CTOPO/CTOPO2 (init-only, static).

WRF 4.7.1 computes two REAL mass-point fields once per domain in
``dyn_em/start_em.F`` (``IF (.NOT. config_flags%restart)``, :1535-1633):

* ``lap_hgt`` = 4-point terrain Laplacian (:1557, edge-clamped indices :1540-1554)
* ``topo_wind=1`` (Jimenez & Dudhia 2012, :1584-1611): over land
  ``CTOPO = ln(sqrt(VAR_SSO))`` when ``sqrt(VAR_SSO) > 2.718`` (else 1); in valleys
  (``lap_hgt <= -10``) CTOPO is blended toward 1 / ramped to 0 and CTOPO2 ramped to 0
* ``topo_wind=2`` (Mass & Ovens, :1612-1625): over land
  ``CTOPO = min(1.575, VAR*0.4/200 + 1.175)**2`` (``VAR`` = geogrid std-dev of
  sub-grid orography, ``grid%var2d``); CTOPO2 stays 1

and ``start_em.F:222`` refuses ``topo_wind=1`` without ``GOT_VAR_SSO``.  The fields
reach only YSU (``module_pbl_driver.F:1232``) and Shin-Hong (:1317): the momentum
surface diagonal and the 10 m wind blend.  MYNN and every other PBL never read them.

Registry ``ctopo/ctopo2`` are ``rdu``: restart (static, recomputed here at every
init/resume) and two-way feedback ``u`` (``copy_fcn``) — with feedback the parent
CTOPO over the nest becomes the child average after the first parent step; see
:func:`fed_back_topo_wind_static`.

All arithmetic is literal REAL4 in the WRF operation order (the pristine build is
``-O2`` without FMA); ``alog`` is the host libm ``logf`` gfortran calls.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

TOPO_WIND_OPTIONS = (0, 1, 2)
F4 = np.float32


class TopoWindStatic(NamedTuple):
    """Per-domain CTOPO/CTOPO2 (``(ny, nx)`` REAL) consumed by YSU/Shin-Hong.

    ``ctopo``/``ctopo2`` hold the value every own step uses.  With two-way
    feedback onto this domain, they hold the fed-back value (from the second own
    step on) and ``ctopo_first``/``ctopo2_first`` the domain's own start_domain
    value that WRF uses on its first step only (``itimestep == 1``)."""

    ctopo: Any
    ctopo2: Any
    ctopo_first: Any = None
    ctopo2_first: Any = None


@lru_cache(maxsize=1)
def _libm_logf():
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    fn = lib.logf
    fn.argtypes, fn.restype = [ctypes.c_float], ctypes.c_float
    return fn


def _alog(values: np.ndarray) -> np.ndarray:
    """Elementwise REAL4 ``alog`` through libm ``logf`` (gfortran's scalar call)."""

    logf = _libm_logf()
    flat = np.ascontiguousarray(values, dtype=F4).ravel()
    return np.fromiter((logf(float(x)) for x in flat), dtype=F4, count=flat.size).reshape(values.shape)


def _real4(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    out = array.astype(F4)
    if array.dtype != F4 and not np.array_equal(out.astype(array.dtype), array, equal_nan=True):
        raise ValueError(f"topo_wind: {name} is not REAL4-representable (WRF grid%{name} is REAL)")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"topo_wind: {name} contains non-finite values")
    return out


def lap_hgt(ht, *, periodic_x: bool = False, periodic_y: bool = False) -> np.ndarray:
    """``start_em.F:1540-1557`` 4-point terrain Laplacian, REAL4, (ny, nx) layout."""

    h = _real4("ht", ht)
    ny, nx = h.shape
    i, j = np.arange(nx), np.arange(ny)
    im1, ip1 = ((i - 1) % nx, (i + 1) % nx) if periodic_x else (np.maximum(i - 1, 0), np.minimum(i + 1, nx - 1))
    jm1, jp1 = ((j - 1) % ny, (j + 1) % ny) if periodic_y else (np.maximum(j - 1, 0), np.minimum(j + 1, ny - 1))
    # (ht(ip1,j)+ht(im1,j)+ht(i,jp1)+ht(i,jm1)-ht(i,j)*4.)/4., left to right.
    total = h[:, ip1] + h[:, im1]
    total = total + h[jp1, :]
    total = total + h[jm1, :]
    return (total - h * F4(4.0)) / F4(4.0)


def start_em_ctopo(topo_wind: int, ht, xland, *, var_sso=None, var2d=None,
                   periodic_x: bool = False, periodic_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return WRF ``(ctopo, ctopo2)`` for one domain (``start_em.F:1581-1626``)."""

    topo_wind = int(topo_wind)
    if topo_wind not in TOPO_WIND_OPTIONS:
        raise ValueError(f"topo_wind={topo_wind} is not a WRF 4.7.1 option (0, 1, 2)")
    x = _real4("xland", xland)
    ctopo = np.ones(x.shape, F4)
    ctopo2 = np.ones(x.shape, F4)
    if topo_wind == 0:
        return ctopo, ctopo2
    land = x < F4(1.5)
    if topo_wind == 1:
        vs = _real4("var_sso", var_sso)
        if np.any(vs[land] < 0):
            raise ValueError("topo_wind=1: negative VAR_SSO over land (sqrt would be NaN in WRF)")
        ctopo = np.where(land, np.sqrt(np.where(land, vs, F4(1.0))), ctopo)
        small = ctopo <= F4(2.718)
        ctopo = np.where(small, F4(1.0), _alog(np.where(small, F4(1.0), ctopo)))
        lap = lap_hgt(ht, periodic_x=periodic_x, periodic_y=periodic_y)
        keep = lap > F4(-10.0)
        blend = ~keep & (lap >= F4(-20.0))
        ramp = ~keep & ~blend & (lap >= F4(-30.0))
        alpha = (lap + F4(20.0)) / F4(10.0)
        blended = alpha * ctopo + (F4(1.0) - alpha)
        ramped = (lap + F4(30.0)) / F4(10.0)
        ctopo = np.where(keep, ctopo, np.where(blend, blended, np.where(ramp, ramped, F4(0.0))))
        ctopo2 = np.where(keep | blend, ctopo2, np.where(ramp, ramped, F4(0.0)))
    else:
        v = _real4("var2d", var2d)
        vfac = np.minimum(F4(1.575), (v * F4(0.4)) / F4(200.0) + F4(1.175))
        vfac = vfac * vfac
        ctopo = np.where(land, ctopo * vfac, ctopo)
    return ctopo.astype(F4), ctopo2.astype(F4)


def _wrfinput_field(run, domain: str, name: str):
    try:
        return np.asarray(run.load_wrfinput(domain, name, lazy=False))
    except KeyError:
        return None


def topo_wind_inputs(run, domain: str, topo_wind: int) -> dict[str, Any]:
    """Read and admit the wrfinput fields ``topo_wind`` needs; refuse loudly otherwise."""

    topo_wind = int(topo_wind)
    if topo_wind not in TOPO_WIND_OPTIONS:
        raise ValueError(f"{domain}: topo_wind={topo_wind} is not a WRF 4.7.1 option (0, 1, 2)")
    inputs: dict[str, Any] = {}
    if topo_wind == 1:
        var_sso = _wrfinput_field(run, domain, "VAR_SSO")
        got = _wrfinput_field(run, domain, "GOT_VAR_SSO")
        # real.exe sets GOT_VAR_SSO = any(VAR_SSO /= 0) (module_initialize_real.F:4894-4904);
        # start_em.F:222 reads the flag from wrfinput.
        got_var_sso = (bool(np.any(np.asarray(var_sso) != 0)) if got is None and var_sso is not None
                       else bool(np.asarray(got).ravel()[0]) if got is not None else False)
        if var_sso is None or not got_var_sso:
            raise ValueError(
                f"{domain}: topo_wind=1 requires VAR_SSO data (WRF start_em.F:222 'topo_wind requires "
                f"VAR_SSO data'); wrfinput has {'no VAR_SSO' if var_sso is None else 'GOT_VAR_SSO=0'}. "
                "Add the WPS varsso dataset (GEOGRID.TBL VAR_SSO rel_path) and rerun geogrid/metgrid/real.")
        inputs["var_sso"] = var_sso
    if topo_wind == 2:
        var2d = _wrfinput_field(run, domain, "VAR")
        if var2d is None:
            raise ValueError(f"{domain}: topo_wind=2 requires the geogrid VAR field (grid%var2d) in wrfinput")
        inputs["var2d"] = var2d
    return inputs


def build_topo_wind_static(topo_wind: int, *, ht, xland, inputs: dict[str, Any],
                           periodic_x: bool = False, periodic_y: bool = False) -> TopoWindStatic | None:
    """Device-ready static for ``topo_wind`` 1/2; ``None`` for 0 (default program untouched)."""

    if int(topo_wind) == 0:
        return None
    import jax.numpy as jnp

    ctopo, ctopo2 = start_em_ctopo(topo_wind, ht, xland, periodic_x=periodic_x, periodic_y=periodic_y,
                                   **inputs)
    # CTOPO2 == 1 everywhere (always for topo_wind=2) makes WRF's 10 m blend the
    # identity: keep it None so YSU/Shin-Hong and history stay on the default path.
    return TopoWindStatic(jnp.asarray(ctopo, jnp.float32),
                          None if bool(np.all(ctopo2 == F4(1.0))) else jnp.asarray(ctopo2, jnp.float32))


# module_pbl_driver.F passes CTOPO/CTOPO2 only to these bl_pbl_physics options.
TOPO_WIND_PBL = (1, 11)


def load_topo_wind_static(run, domain: str, *, topo_wind: int, bl_pbl_physics: int, grid, xland,
                          periodic_x: bool = False, periodic_y: bool = False) -> TopoWindStatic | None:
    """Admit ``topo_wind`` like WRF and build the domain's start_domain CTOPO/CTOPO2.

    ``topo_wind=1`` without VAR_SSO is refused for every PBL (start_em.F:222 is
    PBL-independent). With a PBL other than YSU/Shin-Hong WRF computes the fields
    but nothing reads them: return None (program unchanged) and warn."""

    if int(topo_wind) == 0:
        return None
    inputs = topo_wind_inputs(run, domain, topo_wind)
    if int(bl_pbl_physics) not in TOPO_WIND_PBL:
        import warnings
        warnings.warn(f"{domain}: topo_wind={int(topo_wind)} has no effect with bl_pbl_physics="
                      f"{int(bl_pbl_physics)}: WRF 4.7.1 passes CTOPO/CTOPO2 only to YSU (1) and "
                      "Shin-Hong (11) (module_pbl_driver.F:1232/1317)", stacklevel=2)
        return None
    static = build_topo_wind_static(topo_wind, ht=np.asarray(grid.terrain_height), xland=np.asarray(xland),
                                    inputs=inputs, periodic_x=periodic_x, periodic_y=periodic_y)
    return static


def domain_ctopo_fields(run, domain: str, *, topo_wind: int, ht, xland) -> tuple[np.ndarray, np.ndarray]:
    """start_domain CTOPO/CTOPO2 of any domain (all ones for topo_wind=0), whatever its PBL."""

    if int(topo_wind) == 0:
        ones = np.ones(np.shape(xland), F4)
        return ones, ones.copy()
    return start_em_ctopo(topo_wind, np.asarray(ht), np.asarray(xland), **topo_wind_inputs(run, domain, topo_wind))


def feedback_topo_wind_statics(own: dict, pbl: dict, edges, mass_weights: dict) -> dict:
    """Two-way feedback of Registry ``ctopo``/``ctopo2`` (``rdu``, copy_fcn, no smoothing).

    ``own[d]`` = start_domain ``(ctopo, ctopo2)`` (after the initial feedback of HGT/XLAND),
    ``edges`` = ``(parent, child)`` ordered deepest child first, ``mass_weights`` the
    copy_fcn mass-point feedback plan per edge.  Every parent step ends with the nest
    feedback, so from its second own step a parent sees ``fb(own, child-as-fed)``; its
    first step (itimestep 1) still sees ``own``.  Returns ``{domain: TopoWindStatic | None}``
    for the YSU/Shin-Hong domains whose fields are not identically one."""

    import jax.numpy as jnp

    from gpuwrf.coupling.boundary_feedback import _feedback_leaf

    fed = {d: tuple(np.asarray(a, F4) for a in fields) for d, fields in own.items()}
    for parent, child in edges:
        fed[parent] = tuple(np.asarray(_feedback_leaf(jnp.asarray(fed[parent][k]), jnp.asarray(fed[child][k]),
                                                      mass_weights[(parent, child)], None, 0), F4) for k in (0, 1))
    statics = {}
    for d, (ctopo, ctopo2) in own.items():
        rest = fed[d]
        if int(pbl[d]) not in TOPO_WIND_PBL or all(np.all(a == F4(1.0)) for a in (ctopo, ctopo2, *rest)):
            statics[d] = None
            continue
        need2 = not (np.all(ctopo2 == F4(1.0)) and np.all(rest[1] == F4(1.0)))
        arr = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
        changed = not all(np.array_equal(a, b) for a, b in zip((ctopo, ctopo2), rest))
        statics[d] = TopoWindStatic(arr(rest[0]), arr(rest[1]) if need2 else None,
                                    arr(ctopo) if changed else None,
                                    arr(ctopo2) if changed and need2 else None)
    return statics


__all__ = ["TOPO_WIND_OPTIONS", "TOPO_WIND_PBL", "TopoWindStatic", "domain_ctopo_fields", "feedback_topo_wind_statics", "build_topo_wind_static", "lap_hgt",
           "load_topo_wind_static", "start_em_ctopo", "topo_wind_inputs"]
