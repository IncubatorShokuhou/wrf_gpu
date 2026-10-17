"""Operational ``State -> State`` adapter for NSSL 2-moment microphysics (``mp_physics=18``).

Mirrors WRF ``microphysics_driver`` CASE (NSSL_2MOM) (module_microphysics_driver.F:2237-2303) on the
port's State, exactly like the Thompson adapter does for mp=8:

* inputs: dry ``th_phy`` from theta_m (``phy_prep`` use_theta_m=1), ``pi_phy = (p/p0)**rcp``,
  ``rho``/``dz8w`` exactly as the post-RK caller ``moist_physics_prep_em`` forms them (solve_em.F:3750,
  module_big_step_utilities_em.F:5480/5488/5542): DRY ``rho = 1./(al+alb)`` and ``dz8w`` from
  ``(ph+phb)/g`` (g=9.81, no clamp), ``w`` at the
  bottom w-face (WRF ``w(i,k,j)``), hydrometeors/numbers/volumes in WRF NSSL Registry order:
  qg -> NSSL graupel, qh -> NSSL hail, Nc -> qndrop, Nn -> qnn (activated CCN), Ni/Nr/Ns/Ng/Nh,
  qvolg/qvolh;
* outputs: theta_m recoupled with ``moist_physics_finish_em`` (``_theta_m_from_microphysics``),
  every prognostic replaced, precipitation added to the port's DISJOINT accumulators
  (rain_acc = liquid ``dtp*dn*xfall(lr)``, snow_acc = SNOWNCV, graupel_acc = GRPLNCV,
  hail_acc = HAILNCV; the wrfout writer folds them into WRF's all-phase RAINNC).

``first_step`` (traced bool) reproduces WRF's ``itimestep == 1`` cold start (CN := 0 and calcnfromq).
REAL precision follows the carry: fp32 State -> WRF REAL build constants, fp64 State -> the
``-fdefault-real-8`` constants (both oracle-qualified).
"""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.state import State

from .constants import get_constants,get_one_moment_constants
from .driver import nssl2mom_column
from .indices import FP32, FP64

# State leaf -> WRF NSSL field name used by driver.nssl2mom_column
_STATE_TO_WRF = {
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg", "qh": "qh",
    "Nc": "qndrop", "Nr": "qnr", "Ni": "qni", "Ns": "qns", "Ng": "qng", "Nh": "qnh", "Nn": "qnn",
    "qvolg": "qvolg", "qvolh": "qvolh",
}


def nssl2mom_adapter(state: State, dt: float, grid=None, *, first_step=False, return_precipitation=False,
                     cumulus_tendencies=None, hail_enabled=True, variant=None, radar_depth=None):
    """mp=18 NSSL 2-moment scan adapter (one WRF microphysics call)."""
    from gpuwrf.coupling.physics_couplers import (
        P0_PA, R_D_OVER_CP, _dry_theta_view, _from_columns, _output_dtype,
        _theta_m_from_microphysics, _to_columns)

    one_moment=variant in (19,21)
    if one_moment:
        hail_enabled=variant==19
    # The operational entry already resolves use_mp_re and the LW/SW pair.
    # Preserve its static selection when filling the other NSSL leaves.
    radius_active = state.re_cloud is not None and not one_moment
    selected_mp=variant if one_moment else (18 if hail_enabled else 22)
    state = state.ensure_conditional_leaves(mp_physics=selected_mp, use_mp_re=int(radius_active))
    R = jnp.asarray(state.theta).dtype
    prec = FP32 if R == jnp.float32 else FP64
    C = get_one_moment_constants(prec.name,variant) if one_moment else get_constants(prec.name)
    theta_dry = _dry_theta_view(state)
    # NSSL DN/DZ: microphysics_driver gets grid%rho and dz8w from moist_physics_prep_em, i.e. the DRY
    # 1./(al+alb) (:5542) -- not phy_prep's (1+qv)/alt (:4856), which that call overwrites first.
    # Original CPU-WRF c14/c15 REFL_10CM is reproduced to 7.6e-6 dBZ only with the dry density.
    from gpuwrf.coupling.scan_adapters import _wrf_moist_prep_den_delz
    rho, delz = _wrf_moist_prep_den_delz(state, grid, R)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    represented = {leaf: wrf for leaf, wrf in _STATE_TO_WRF.items()
                   if hail_enabled or leaf not in ("qh", "Nh", "qvolh")}
    if one_moment:
        represented={leaf:wrf for leaf,wrf in represented.items()
                     if leaf in ('qv','qc','qr','qi','qs','qg') or (variant==19 and leaf in ('qh','qvolg','Nn'))}
    fields = {wrf: _to_columns(getattr(state, leaf)) for leaf, wrf in represented.items()}
    fields.update(th=_to_columns(theta_dry), pii=_to_columns(pii), p=_to_columns(state.p),
                  rho=_to_columns(rho), w=_to_columns(state.w[:-1]), dz=_to_columns(delz))
    cu_columns = (None if cumulus_tendencies is None else
                  {name: _to_columns(rate) for name, rate in cumulus_tendencies.items()})
    radar_kwargs={} if radar_depth is None else {"radar_depth":radar_depth}
    out, precip, _diag = nssl2mom_column(fields, dt, C, prec, first_step=first_step,
                                      radii=radius_active,
                                      cumulus_tendencies=cu_columns, hail_enabled=hail_enabled, **radar_kwargs)

    qv_new = _from_columns(out["qv"]).astype(_output_dtype(state, "qv"))
    theta_new_dry = _from_columns(out["th"]).astype(_output_dtype(state, "theta"))
    updates = {"theta": _theta_m_from_microphysics(theta_dry, state.qv, theta_new_dry, qv_new,
                                                   _output_dtype(state, "theta")),
               "qv": qv_new}
    for leaf, wrf in represented.items():
        if leaf != "qv":
            updates[leaf] = _from_columns(out[wrf]).astype(_output_dtype(state, leaf))
    if radius_active:
        for name in ("re_cloud", "re_ice", "re_snow"):
            updates[name] = _from_columns(_diag[name]).astype(jnp.float32)
    for acc, key in (("rain_acc", "rain_liquid"), ("snow_acc", "snowncv"), ("graupel_acc", "grplncv"),
                     ("hail_acc", "hailncv")):
        prev = getattr(state, acc)
        if prev is not None:
            updates[acc] = (jnp.asarray(prev, jnp.float64) + jnp.asarray(precip[key], jnp.float64)).astype(
                _output_dtype(state, acc))
    new_state = state.replace(**updates)
    if radar_depth is not None:
        return new_state, precip, _from_columns(_diag["dbz"]).astype(jnp.float32)
    if return_precipitation:
        return new_state, precip
    return new_state


def nssl1mom19_adapter(state,dt,grid=None,**kwargs):
    return nssl2mom_adapter(state,dt,grid,variant=19,**kwargs)


def nssl1mom21_adapter(state,dt,grid=None,**kwargs):
    return nssl2mom_adapter(state,dt,grid,variant=21,**kwargs)
