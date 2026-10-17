"""Operational State<->kernel adapters for the MYJ PBL + Janjic Eta pair.

These ``State -> State`` adapters thread the v0.13 operational MYJ pair into the
device scan (``runtime.operational_mode`` dispatches them in the surface-layer
and PBL slots, in WRF call order). They are the MYJ analogue of the YSU/ACM2
PBL adapters and the revised-MM5 surface-layer adapter in
``coupling.scan_adapters``, reusing the SAME public ``coupling.physics_couplers``
helpers (mass-point winds, A2C momentum coupling, density, dtype-safe writes).

WRF call order (``module_surface_driver.F`` then ``module_pbl_driver.F``):

1. ``janjic_sfclay_adapter`` (sf=2) runs the Janjic Eta surface layer
   (``physics.sf_myj.myjsfc_columns``) and writes the frozen B2 surface-flux
   handles (ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv), exactly like the
   revised-MM5 adapter -- so the rest of the operational pipeline sees the MYJ
   surface fluxes.
2. ``myj_pbl_adapter`` (bl=2) RE-derives the full surface coupling the MYJ PBL
   consumes (USTAR/AKHS/AKMS/THZ0/QZ0/QSFC/CHKLOWQ/ELFLX) by re-running the same
   traceable Janjic surface layer (the YSU/ACM2 adapters use the identical
   "re-run the surface layer inside the PBL adapter" pattern, because the frozen
   State carries kinematic fluxes only, not the scheme-specific exchange coeffs),
   then runs ``physics.bl_myj.myj_columns`` and applies the u/v (A2C) and
   theta/qv (mass-grid) increments. The MYJ TKE state is carried in the State's
   ``qke`` leaf, which uses the MYNN/MYJ q^2 convention (``qke = 2*TKE = q**2``,
   identical to the WRF MYJ ``Q2`` array). MYJ's per-column ``tke`` argument is
   ``TKE_MYJ = 0.5*q**2 = 0.5*qke``; the surface layer's ``q2`` argument is the
   q^2 array (= qke). The PBL writes the updated ``q**2`` back to ``qke``.

The TKE coupling is faithful: the surface layer's TKE-derived PBL height feeds
the surface Beljaars term, and the PBL's updated ``TKE_MYJ`` is written back to
``qke`` so it persists across steps (the 1.5-order closure's prognostic).

fp64-safe, no host transfer, ``jax.jit``/``jax.vmap``-traceable end to end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State
from gpuwrf.coupling.physics_couplers import (
    _add_a2c_u_increment,
    _add_a2c_v_increment,
    _output_dtype,
    _rho_from_state,
    _theta_m_tendency_from_dry,
    _u_mass,
    _v_mass,
)
from gpuwrf.coupling.wrf_phy_prep import wrf_phy_prep
from gpuwrf.physics import myj_constants as C
from gpuwrf.physics.bl_myj import myj_columns
from gpuwrf.physics.sf_myj import myjsfc_columns


# MYJ TKE floor in the qke (TKE) leaf: TKE_MYJ >= EPSQ2/2 (q2 floor is EPSQ2).
_MYJ_TKE_FLOOR = 0.5 * C.EPSQ2


def _myj_geometry(state: State, metrics=None):
    """Common column geometry the MYJ pair consumes (pure ``jnp``, no transfer).

    Returns the ``(ncol, nz)`` bottom-up profile views + the per-column surface
    scalars, staged like WRF ``phy_prep`` (``coupling.wrf_phy_prep``): MYJSFC
    (module_surface_driver.F:2210-2224) and MYJPBL (module_pbl_driver.F:1439-1465)
    both get DRY ``TH=th_phy``/``T=t_phy`` (first_rk_step_part1.F:657/662, 1138/1139),
    ``PMID=p_hyd``/``PINT=p_hyd_w`` (:633/637, 1130/1132), ``PSFC=PINT(LOWLYR)`` and
    ``dz8w``/``HT`` from ``(ph+phb)/g`` with g=9.81. Without dycore metrics
    (isolated tests) the pressures fall back to the non-hydrostatic assembler.
    """

    nz, ny, nx = state.theta.shape
    ncol = ny * nx

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    def _flat2d(field2d):  # (ny, nx) -> (ncol,)
        return jnp.asarray(field2d, jnp.float64).reshape(ncol)

    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    prep = wrf_phy_prep(state, metrics)
    T = prep["t_phy"]
    pii = prep["pi_phy"]
    rho = _rho_from_state(state) if prep["rho"] is None else prep["rho"]
    interface_z = prep["z_at_w"]  # (nz+1, ny, nx), g=9.81
    dz = jnp.maximum(prep["dz8w"], 1.0)  # (nz, ny, nx); clamp inert for real layers
    if prep["p_hyd"] is not None:
        p, p_int, psfc = prep["p_hyd"], prep["p_hyd_w"], prep["psfc"]
    else:
        # No metrics: interior faces = mean of adjacent levels; edges by
        # zero-gradient extrapolation; lowest-level pressure as PSFC proxy.
        p = state.p.astype(jnp.float64)
        p_int_interior = 0.5 * (p[:-1] + p[1:])
        p_int = jnp.concatenate([p[:1], p_int_interior, p[-1:]], axis=0)  # (nz+1, ny, nx)
        psfc = p[0]
    ht = _flat2d(interface_z[0])  # terrain height = lowest interface

    return {
        "ncol": ncol, "ny": ny, "nx": nx, "nz": nz,
        "u_cols": _cols(u_mass), "v_cols": _cols(v_mass),
        "theta_cols": _cols(prep["th_phy"]), "T_cols": _cols(T),
        "qv_cols": _cols(jnp.maximum(state.qv, 0.0)),
        "qc_cols": _cols(jnp.maximum(state.qc, 0.0)),
        "p_cols": _cols(p), "pii_cols": _cols(pii), "rho_cols": _cols(rho),
        "dz_cols": _cols(dz),
        "p_int_cols": jnp.moveaxis(p_int, 0, -1).reshape(ncol, nz + 1),
        "qke_cols": _cols(jnp.maximum(state.qke, 2.0 * _MYJ_TKE_FLOOR)),
        "tsk": _flat2d(state.t_skin),
        "xland": _flat2d(state.xland),
        "ust": jnp.maximum(_flat2d(state.ustar), 1.0e-3),
        "znt": jnp.maximum(_flat2d(jnp.maximum(state.roughness_m, 1.0e-4)), 1.0e-7),
        "mavail": _flat2d(state.mavail),
        "psfc": _flat2d(psfc),
        "ht": ht,
    }


def _run_janjic(g: dict) -> dict:
    """Run the batched Janjic Eta surface layer on the geometry views.

    Warm-start surface seeds match the savepoint-validated column convention:
    THZ0 = lowest-layer theta, QSFC = lowest-layer mixing ratio, QZ0 = lowest
    specific humidity. ``q2`` passed to MYJSFC is the MYJ TKE (qke/2 -> TKE_MYJ).
    """

    qv0 = g["qv_cols"][:, 0]
    th0 = g["theta_cols"][:, 0]
    # State.qke uses the MYNN/MYJ q^2 convention (qke = 2*TKE = q^2), identical to
    # the WRF MYJ ``Q2`` array. ``myjsfc_columns`` takes that ``Q2`` directly (it is
    # used for the TKE-derived PBL-height floor), so pass qke straight through.
    q2_cols = g["qke_cols"]
    return myjsfc_columns(
        g["u_cols"], g["v_cols"], g["T_cols"], g["theta_cols"], g["qv_cols"],
        g["qc_cols"], g["p_cols"], g["dz_cols"], q2_cols,
        tsk=g["tsk"], xland=g["xland"], z0base=g["znt"], psfc=g["psfc"],
        znt=g["znt"], ustar=g["ust"], mavail=g["mavail"],
        qsfc=qv0, thz0=th0, qz0=qv0 / (1.0 + qv0), uz0=0.0, vz0=0.0,
        pblh=jnp.full((g["ncol"],), 1000.0, dtype=jnp.float64),
    )


def _back3d(field2d, ny, nx, nz):  # (ncol, nz) -> (nz, ny, nx)
    return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)


def janjic_sfclay_adapter(state: State, dt: float, grid=None, *, return_cumulus_inputs=False,
                          history=False) -> State | tuple[State, dict[str, jax.Array]]:
    """sf_sfclay=2 Janjic Eta surface-layer scan adapter (writes B2 flux handles).

    Mirrors the revised-MM5 surface adapter contract: writes only the frozen B2
    kinematic-flux handles. Kinematic theta/qv fluxes are derived from the Janjic
    HFX/QFX (HFX = -rho*cp*akhs*(thlow-thz0); theta_flux = HFX/(rho*cp)).
    ``history=True`` also returns this solve's equivalent-roughness 10 m winds.
    """

    g = _myj_geometry(state, getattr(grid, "metrics", None))
    out = _run_janjic(g)

    rhosfc = out["flhc"] / (C.CP * jnp.maximum(out["akhs"], 1.0e-30))
    theta_flux = out["hfx"] / jnp.maximum(rhosfc * C.CP, 1.0e-12)
    qv_flux = out["qfx"] / jnp.maximum(rhosfc, 1.0e-12)
    tau_u = out["akms"] * g["u_cols"][:, 0]
    tau_v = out["akms"] * g["v_cols"][:, 0]
    fltv = theta_flux

    ny, nx = g["ny"], g["nx"]
    r2 = lambda a: jnp.asarray(a, jnp.float64).reshape(ny, nx)
    cumulus_fields={}
    if return_cumulus_inputs:
        # Subsequent CAM-ZM consumes physical HFX/QFX from this actual call.
        cumulus_fields={name:r2(out[key]).astype(_output_dtype(state,name)) for name,key in (('hfx','hfx'),('qfx','qfx'))}
    result = state.replace(
        ustar=r2(out["ustar"]).astype(_output_dtype(state, "ustar")),
        theta_flux=r2(theta_flux).astype(_output_dtype(state, "theta_flux")),
        qv_flux=r2(qv_flux).astype(_output_dtype(state, "qv_flux")),
        tau_u=r2(tau_u).astype(_output_dtype(state, "tau_u")),
        tau_v=r2(tau_v).astype(_output_dtype(state, "tau_v")),
        rhosfc=r2(rhosfc).astype(_output_dtype(state, "rhosfc")),
        fltv=r2(fltv).astype(_output_dtype(state, "fltv")),
        roughness_m=r2(out["znt"]).astype(_output_dtype(state, "roughness_m")),
        **cumulus_fields,
    )
    if history:
        # MYJSFC:985-1018 special winds use equivalent roughness over land.
        # Return the actual solve's values; never reconstruct/rotate at output.
        return result, {name: r2(out[name]).astype(jnp.float32) for name in ('u10e', 'v10e')}
    return result


def myj_pbl_adapter(state: State, dt: float, grid=None, *, return_tendencies=False,
                    return_cumulus_inputs=False, f_qi=True, surface=None, chklowq=None,
                    snow=None) -> State:
    """bl_pbl=2 MYJ PBL scan adapter (re-derives the Janjic coupling, runs MYJ).

    Re-runs the Janjic Eta surface layer to obtain the full per-cell coupling the
    MYJ PBL consumes (the frozen State carries kinematic fluxes only), then runs
    the traceable MYJ PBL kernel and applies the WRF-faithful increments:
    u/v via the A2C mass->C-grid average, theta/qv directly on the mass grid. The
    updated TKE is written back to the State ``qke`` leaf (the closure's
    prognostic carry).

    ``surface`` (the Janjic-family memory of THIS step's MYJSFC call, QnseCarry)
    replaces the private re-solve: WRF MYJPBL reads grid AKHS/AKMS/THZ0/QZ0/UZ0/VZ0/
    CT/UST/QSFC, ELFLX=grid%lh (post-LSM) and CHKLOWQ (pbl_driver.F:1439-1465), and
    returns THZ0/QZ0/QSFC (module_bl_myjpbl.F:540-563); the updated THZ0/QZ0 are then
    appended to the result as a last ``dict`` and QSFC is written to State.
    """

    g = _myj_geometry(state, getattr(grid, "metrics", None))
    if surface is None:
        sfc = _run_janjic(g)
    else:
        flat = lambda a: jnp.asarray(a, jnp.float64).reshape(g["ncol"])
        sfc = dict(ustar=flat(state.ustar), qsfc=flat(state.qsfc), flx_lh=flat(surface.lh),
                   **{k: flat(getattr(surface, k)) for k in ('akhs', 'akms', 'thz0', 'qz0', 'uz0', 'vz0', 'ct')})

    # State.qke is q^2 (= 2*TKE). MYJ's ``tke`` argument is TKE_MYJ = 0.5*q^2,
    # so TKE_MYJ = 0.5*qke.
    tke_cols = 0.5 * g["qke_cols"]

    # CHKLOWQ / ELFLX: the Janjic surface layer exposes the moisture availability
    # coupling via FLQC/QFX. WRF's MYJSFC sets CHKLOWQ (wet fraction, 0..1) and the
    # latent-heat flux ELFLX (=LH) consumed by the MYJ QSFC update. CHKLOWQ follows
    # MAVAIL over land (the wet-soil fraction); ELFLX = LH (W m^-2).
    chklowq = jnp.clip(g["mavail"], 0.0, 1.0) if surface is None else jnp.broadcast_to(
        jnp.asarray(chklowq, jnp.float64), (g["ncol"],))
    elflx = sfc["flx_lh"]

    # module_pbl_driver.F:1448 passes QCW=qc_curr, QCI=qi_curr unclipped; MYJPBL
    # mixes both (CWM=QCW+QCI) and returns RQCBLTEN/RQIBLTEN from those inputs.
    raw_cols = lambda field3d: jnp.moveaxis(jnp.asarray(field3d, jnp.float64), 0, -1).reshape(g["ncol"], g["nz"])  # noqa: E731
    out = myj_columns(
        g["u_cols"], g["v_cols"], g["T_cols"], g["theta_cols"], g["qv_cols"],
        raw_cols(state.qc), g["p_cols"], g["p_int_cols"], g["pii_cols"], g["dz_cols"],
        tke_cols,
        tsk=g["tsk"], xland=g["xland"], ustar=sfc["ustar"],
        akhs=sfc["akhs"], akms=sfc["akms"], chklowq=chklowq, elflx=elflx,
        thz0=sfc["thz0"], qz0=sfc["qz0"], uz0=sfc["uz0"], vz0=sfc["vz0"],
        qsfc=sfc["qsfc"], ct=sfc["ct"], dt=float(dt), stepbl=1, ht=g["ht"],
        # QCI=qi_curr is always PRESENT (flQI=.TRUE.); without a Registry QI slot
        # (mp 0/1/3) WRF hands MYJ the zero dummy moist slot.
        qi=raw_cols(state.qi) if f_qi else None,
        **({} if snow is None else {"snow": jnp.asarray(snow, jnp.float64).reshape(g["ncol"])}),
    )

    ny, nx, nz = g["ny"], g["nx"], g["nz"]
    dt_f = float(dt)
    du_mass = dt_f * _back3d(out["RUBLTEN"], ny, nx, nz)
    dv_mass = dt_f * _back3d(out["RVBLTEN"], ny, nx, nz)
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))
    # WRF couples the DRY RTHBLTEN into theta_m via conv_t_tendf_to_moist
    # (module_big_step_utilities_em.F:6699-6703) at the PBL-entry state.
    theta_new = (state.theta + dt_f * _theta_m_tendency_from_dry(
        _back3d(out["RTHBLTEN"], ny, nx, nz), _back3d(out["RQVBLTEN"], ny, nx, nz),
        state.theta, state.qv, state.theta.dtype)).astype(_output_dtype(state, "theta"))
    qv_new = (state.qv + dt_f * _back3d(out["RQVBLTEN"], ny, nx, nz)).astype(
        _output_dtype(state, "qv")
    )
    condensate = {name: (getattr(state, name) + dt_f * _back3d(out[key], ny, nx, nz)).astype(
        _output_dtype(state, name)) for name, key in (("qc", "RQCBLTEN"), ("qi", "RQIBLTEN"))
        if name == "qc" or f_qi}
    # Persist the updated TKE in qke (q^2 convention = 2*TKE_MYJ). Floor at EPSQ2.
    qke_new = _back3d(2.0 * jnp.maximum(out["TKE_MYJ"], _MYJ_TKE_FLOOR), ny, nx, nz)
    qke_new = jnp.maximum(qke_new, 2.0 * _MYJ_TKE_FLOOR).astype(_output_dtype(state, "qke"))

    diagnostics = {}
    if surface is not None:
        diagnostics['qsfc'] = out['QSFC'].reshape(ny, nx).astype(state.qsfc.dtype)
    if return_cumulus_inputs:
        diagnostics['pblh'] = out['PBLH'].reshape(ny,nx).astype(state.pblh.dtype)
    if return_tendencies:
        rates = {name:_back3d(out[key],ny,nx,nz).astype(jnp.float32) for name,key in
                 (('rublten','RUBLTEN'),('rvblten','RVBLTEN'),('rthblten','RTHBLTEN'),('rqvblten','RQVBLTEN'),
                  ('rqcblten','RQCBLTEN'),('rqiblten','RQIBLTEN'))}
        result = (state.replace(qke=qke_new,**diagnostics),rates)
    else:
        result = (state.replace(u=u_new,v=v_new,theta=theta_new,qv=qv_new,qke=qke_new,**condensate,
                                **diagnostics),)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=(retain_columns(result[0],exch_h=out['EXCH_H'],tke=out['TKE_MYJ']),*result[1:])
    if return_cumulus_inputs:
        result += ((jnp.zeros((ny,nx),jnp.float32),out['KPBL'].reshape(ny,nx).astype(jnp.int32)),)
    if surface is not None:
        result += ({k: out[k.upper()].reshape(ny, nx).astype(jnp.float32) for k in ('thz0', 'qz0')},)
    return result if len(result)>1 else result[0]


__all__ = ["janjic_sfclay_adapter", "myj_pbl_adapter"]
