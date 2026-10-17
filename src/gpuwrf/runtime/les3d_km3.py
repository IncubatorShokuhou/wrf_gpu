"""Operational adapter for WRF ``diff_opt=2`` / ``km_opt=3`` (3-D Smagorinsky) and ``km_opt=2`` (3-D TKE).

Builds the WRF RK1-frozen diffusion bundle from a time-t :class:`State` with the
literal operator in :mod:`gpuwrf.dynamics.les3d_smagorinsky`, mirroring the
caller wiring of ``module_first_rk_step_part2.F`` (diff_opt=2 block):

* ``thp`` = ``grid%t_2`` (theta_m - T0 with use_theta_m=1), ``th_phy``/``t_phy``/
  ``p_phy``/``p8w``/``t8w``/``rho`` from the port's ``phy_prep`` transcriptions,
  ``ph``/``phb`` split as WRF stores them, the full moist + other-scalar set.
* ``vertical_diffusion_2`` only when ``bl_pbl_physics == 0`` (WRF gate), with the
  surface stress/heat/moisture fluxes of ``isfflx`` from ``ustar``/``hfx``/``qfx``.
* ``km_opt=2`` additionally returns WRF's RK1-frozen ``tke_tend`` (``tke_rhs`` + doubled TKE diffusion,
  coupled, sc_tend units) for the prognostic ``State.tke`` (WRF ``tke_2``); :func:`les_tke_stage_update`
  advances it every RK stage as solve_em's ``TKE_advance`` does.
* Dry outputs are returned already folded the way ``rk_addtend_dry`` consumes
  them (``ru_tendf/msfuy``, ``rv_tendf*(1/msfvx)``, ``rw_tendf/msfty``,
  ``t_tendf/msfty``); moist/scalar increments are the unscaled ``moist_tend``/
  ``scalar_tend`` (``sc_tend``) contributions reused by every ``rk_update_scalar``.

CPU-oracle-qualified against pristine WRF REAL4 (proofs/o1_smag3d); GPU and
coupled-forecast qualification pending.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.dynamics.les3d_smagorinsky import Les3dConfig, Les3dInputs, les3d_smagorinsky_tendencies

# WRF moist array order (qv, qc, qr, qi, qs, qg) followed by the other scalars.
LES3D_MOIST_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
LES3D_SCALAR_SPECIES = ("Ni", "Nr")
_T0 = 300.0


class Les3dKm3Bundle(NamedTuple):
    """RK1-frozen km_opt=3 bundle in the operational large-step space."""

    u_t: jax.Array
    v_t: jax.Array
    w_t: jax.Array
    theta_t: jax.Array
    scalar_sc: dict
    tke_t: jax.Array | None = None
    sms_updates: dict | None = None


def les3d_km3_active(namelist) -> bool:
    return int(namelist.diff_opt) == 2 and int(namelist.km_opt) in (2, 3, 5)


def _boundary(namelist) -> str:
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags

    periodic, specified, nested = _acoustic_lateral_bc_flags(namelist)
    if specified or nested:
        return "specified"
    if periodic:
        return "periodic"
    raise ValueError("km_opt=3 supports specified/nested or doubly periodic lateral boundaries")


def les3d_km3_forward_tendencies(reference: State, namelist, *, base_state=None) -> Les3dKm3Bundle:
    """WRF diff_opt=2/km_opt=3 RK1 bundle from the time-t ``reference`` state."""

    from gpuwrf.io.namelist_binding import KM2_UNWIRED_MP

    if int(namelist.km_opt) in (2,5) and int(namelist.mp_physics) in KM2_UNWIRED_MP:
        raise NotImplementedError(f"km_opt={namelist.km_opt} extra microphysics scalar diffusion is not wired")
    if int(namelist.km_opt) == 5:
        if int(namelist.diff_opt) != 2 or int(namelist.bl_pbl_physics) != 0:
            raise ValueError("km_opt=5 requires diff_opt=2 and bl_pbl_physics=0 (WRF)")
        if int(namelist.sf_sfclay_physics) not in (0, 5):
            raise NotImplementedError("km_opt=5 SMS surface caller is qualified for sf_sfclay_physics=0/5 only")
        land = getattr(namelist,'sf_surface_physics',None)
        land = int(land) if land is not None else (4 if bool(getattr(namelist,'use_noahmp',False)) else 0)
        if bool(getattr(namelist,'use_noahmp',False)) and land!=4:
            raise ValueError('km_opt=5 Noah-MP caller requires sf_surface_physics=4')
        if land not in (0,4) or (int(namelist.sf_sfclay_physics)==0 and land!=0):
            raise NotImplementedError("km_opt=5 SMS land caller is qualified for sf_surface_physics=0/4; surface layer 0 requires land 0")
        if int(namelist.cu_physics) == 11:
            raise ValueError("km_opt=5 cannot work with cu_physics=11 (WRF)")

    from gpuwrf.coupling.physics_couplers import (
        _dry_temperature_from_state,
        _dry_theta_view,
        _wrf_phy_prep_p8w,
        _wrf_phy_prep_rho_from_state,
        _wrf_phy_prep_temperature_interfaces,
    )

    metrics = namelist.metrics
    dtype = jnp.float32 if int(namelist.km_opt)==5 else jnp.asarray(reference.theta).dtype
    a = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
    dx = np.float32(namelist.grid.projection.dx_m)
    dy = np.float32(namelist.grid.projection.dy_m)
    ph_total = a(reference.ph_total)
    ph = a(reference.ph_perturbation)
    phb = a(base_state.phb) if base_state is not None and getattr(base_state, "phb", None) is not None else ph_total - ph
    p_phy = a(reference.p_total)
    th_phy = a(_dry_theta_view(reference))
    t_phy = a(_dry_temperature_from_state(reference))
    z_at_w = ph_total / np.float32(9.81)
    p8w = a(_wrf_phy_prep_p8w(p_phy, z_at_w, metrics.fnm, metrics.fnp))
    t8w = a(_wrf_phy_prep_temperature_interfaces(t_phy, reference, metrics))
    rho = a(_wrf_phy_prep_rho_from_state(reference, metrics, output_dtype=dtype))
    names = [n for n in LES3D_MOIST_SPECIES + LES3D_SCALAR_SPECIES if getattr(reference, n, None) is not None]
    moist = tuple(a(getattr(reference, n)) for n in names)
    pos = {n: i for i, n in enumerate(names)}
    vertical = int(namelist.bl_pbl_physics) == 0
    ustar = getattr(reference, "ustar", None)
    hfx = getattr(reference, "hfx", None)
    qfx = getattr(reference, "qfx", None)
    if vertical and int(getattr(namelist, "isfflx", 1)) in (1, 2) and (ustar is None or hfx is None or qfx is None):
        raise ValueError("km_opt=3 with bl_pbl_physics=0 and isfflx=1/2 needs State ustar/hfx/qfx")
    scalar = lambda x: jnp.asarray(x, dtype=dtype).reshape(-1)[0]  # noqa: E731  (trace-safe)
    km_opt = int(namelist.km_opt)
    if km_opt in (2,5) and getattr(reference, "tke", None) is None:
        raise ValueError("km_opt=2 needs the prognostic State.tke leaf (seeded by the initial carry)")
    inp = Les3dInputs(
        u=a(reference.u), v=a(reference.v), w=a(reference.w),
        thp=a(reference.theta) - np.float32(_T0), th_phy=th_phy, t_phy=t_phy, p_phy=p_phy, p8w=p8w, t8w=t8w,
        ph=ph, phb=phb, rho=rho, moist=moist,
        msftx=a(metrics.msftx), msfty=a(metrics.msfty), msfux=a(metrics.msfux), msfuy=a(metrics.msfuy),
        msfvx=a(metrics.msfvx), msfvy=a(metrics.msfvy),
        fnm=a(metrics.fnm), fnp=a(metrics.fnp), dn=a(metrics.dn), dnw=a(metrics.dnw),
        cf1=scalar(metrics.cf1), cf2=scalar(metrics.cf2), cf3=scalar(metrics.cf3),
        rdx=np.float32(1.0) / dx, rdy=np.float32(1.0) / dy, dx=dx, dy=dy, dt=np.float32(namelist.dt_s),
        ust=None if ustar is None else a(ustar), hfx=None if hfx is None else a(hfx),
        qfx=None if qfx is None else a(qfx),
        tke=a(reference.tke) if km_opt in (2,5) else None,
        mut=a(reference.mu_total) if km_opt in (2,5) else None,
        c1h=a(metrics.c1h) if km_opt in (2,5) else None,
        c2h=a(metrics.c2h) if km_opt in (2,5) else None,
        **(dict(c1f=a(metrics.c1f),c2f=a(metrics.c2f),sms_pblh=a(reference.pblh),
                **{n:a(getattr(reference,n)) for n in ('sms_br','sms_rmol','sms_wspd','sms_gamu','sms_gamv','sms_ustm')}) if km_opt==5 else {}),
    )
    cfg = Les3dConfig(
        boundary=_boundary(namelist),
        isotropic=int(getattr(namelist, "mix_isotropic", 0)),
        c_s=float(namelist.c_s),
        mix_upper_bound=float(getattr(namelist, "mix_upper_bound", 0.1)),
        vertical=vertical,
        isfflx=int(getattr(namelist, "isfflx", 1)),
        tke_drag_coefficient=float(getattr(namelist, "tke_drag_coefficient", 0.0)),
        tke_heat_flux=float(getattr(namelist, "tke_heat_flux", 0.0)),
        use_theta_m=1,
        mix_full_fields=False,
        iqv=pos["qv"],
        iqc=pos.get("qc"),
        iqi=pos.get("qi"),
        km_opt=km_opt,
        c_k=float(getattr(namelist, "c_k", 0.15)),
        tke_mix2_off=bool(getattr(namelist, "tke_mix2_off", False)),
        sms_precip_scalar_indices=tuple(pos[n] for n in ('Nr',) if n in pos) if km_opt==5 else (),
    )
    out = les3d_smagorinsky_tendencies(inp, cfg)
    msfuy = a(metrics.msfuy)[None]
    msfvx_inv = (np.float32(1.0) / a(metrics.msfvx))[None]
    msfty = a(metrics.msfty)[None]
    return Les3dKm3Bundle(
        u_t=out.ru_tendf / msfuy,
        v_t=out.rv_tendf * msfvx_inv,
        w_t=out.rw_tendf / msfty,
        theta_t=out.t_tendf / msfty,
        scalar_sc={n: out.moist_tendf[i] for n, i in pos.items()},
        tke_t=out.tke_tendf,
        sms_updates=out.sms_updates,
    )


def les_tke_sixth_order_sc(reference: State, muts_rk1, namelist):
    """rk_scalar_tend's RK1 sixth-order TKE diffusion (diff_6th_opt != 0, tke_mix6_off=.false.)."""

    from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_scalar_tendf
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags

    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    metrics = namelist.metrics
    return wrf_sixth_order_scalar_tendf(
        reference.tke, muts_rk1, c1=metrics.c1h, c2=metrics.c2h, msftx=metrics.msftx, msfty=metrics.msfty,
        dt=float(namelist.dt_s) / float(namelist.rk_order), diff_6th_factor=float(namelist.diff_6th_factor),
        monotonic=(int(namelist.diff_6th_opt) == 2), specified_or_nested=bool(specified or nested),
    )


def les_tke_stage_update(state: State, step_origin: State, *, tke_tend, rk_step: int, dt_rk: float, namelist,
                         transport_velocities) -> State:
    """solve_em ``TKE_advance`` for one RK stage (km_opt=2).

    ``rk_update_scalar_pd`` at the final stage (tke_adv_opt not ORIGINAL/WENO), ``rk_scalar_tend``
    flux-form advection with the acoustic-averaged ru_m/rv_m/ww_m (the moisture loop's transport),
    ``rk_update_scalar`` (advection*msfty inside the spec-zone rectangle on specified/nested domains,
    RK1-frozen ``tke_tend`` everywhere, step-start mass/field), ``bound_tke`` and ``flow_dep_bdy``.
    """

    from gpuwrf.coupling.boundary_apply import flow_dep_bdy
    from gpuwrf.runtime.operational_mode import (
        _acoustic_lateral_bc_flags,
        _rk_update_scalar_pd,
        _scalar_transport_coupled_tendencies,
    )

    metrics = namelist.metrics
    adv_opt = int(getattr(namelist, "tke_adv_opt", 1))
    sms = int(getattr(namelist, 'km_opt', 2)) == 5
    origin, sc = step_origin, tke_tend
    if int(rk_step) == int(namelist.rk_order) and adv_opt not in (0, 3):
        if sms:
            real = lambda x: jnp.asarray(x, jnp.float32)
            mass = real(metrics.c1h)[:, None, None] * real(step_origin.mu_total)[None] + real(metrics.c2h)[:, None, None]
            origin = step_origin.replace(tke=(mass*real(step_origin.tke)+real(dt_rk)*real(tke_tend))/mass)
        else:
            origin = _rk_update_scalar_pd(step_origin, ("tke",), ({"tke": tke_tend},), float(dt_rk), metrics)
        sc = jnp.zeros_like(tke_tend)
    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    bounded = bool(specified or nested)
    from gpuwrf.dynamics.flux_advection import _native_pd_enabled

    if (bounded and adv_opt == 1 and int(rk_step) == int(namelist.rk_order)
            and not _native_pd_enabled(adv_opt, transport_velocities)):
        # WRF limits donor cells only in [ids+1:ide-2,jds+1:jde-2]. The
        # retained legacy limiter also scales the outer ring. Use the existing
        # qualified REAL kernel for this new REAL prognostic on that path.
        # Release defaults already select this kernel in the shared helper.
        from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32

        if transport_velocities is None or transport_velocities.ru_full is None:
            raise NotImplementedError("km_opt=2 PD transport needs acoustic-averaged full-face fluxes")
        real = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        vel = transport_velocities
        adv = advect_scalar_pd_fp32(
            real(state.tke)[None], real(origin.tke)[None], real(vel.ru_full), real(vel.rv_full),
            real(vel.rom), real(state.mu_total), real(origin.mu_total), real(metrics.c1h), real(metrics.c2h),
            real(metrics.msftx), real(metrics.msfty), real(metrics.rdnw), real(metrics.fnm), real(metrics.fnp),
            rdx=1.0 / float(namelist.grid.projection.dx_m),
            rdy=1.0 / float(namelist.grid.projection.dy_m), dt=float(namelist.dt_s),
            interpret=jax.default_backend() == "cpu",
        )[0]
    else:
        adv = _scalar_transport_coupled_tendencies(
            state, namelist, rk_step=int(rk_step), step_origin=origin, species=("tke",),
            advection_opt=adv_opt, transport_velocities=transport_velocities,
        )[0]
    adv = adv * jnp.asarray(metrics.msfty, dtype=adv.dtype)[None, :, :]
    if bounded:
        sz = int(namelist.boundary_config.spec_zone)
        ny, nx = (int(n) for n in adv.shape[-2:])
        adv = jnp.zeros_like(adv).at[..., sz:ny - sz, sz:nx - sz].set(adv[..., sz:ny - sz, sz:nx - sz])
    tend = adv + jnp.asarray(sc, dtype=adv.dtype)
    if sms:
        # WRF TKE_advance operands are REAL even when retained metric storage is DOUBLE.
        real = lambda x: jnp.asarray(x, jnp.float32)
        c1, c2 = real(metrics.c1h)[:, None, None], real(metrics.c2h)[:, None, None]
        mass_old = c1 * real(origin.mu_total)[None] + c2
        mass_new = c1 * real(state.mu_total)[None] + c2
    else:
        mass_old = metrics.c1h[:, None, None] * origin.mu_total[None, :, :] + metrics.c2h[:, None, None]
        mass_new = metrics.c1h[:, None, None] * state.mu_total[None, :, :] + metrics.c2h[:, None, None]
    tke_old = jnp.asarray(origin.tke, dtype=tend.dtype)
    tke_new = (mass_old.astype(tend.dtype) * tke_old + float(dt_rk) * tend) / mass_new.astype(tend.dtype)
    tke_new = jnp.minimum(float(getattr(namelist, "tke_upper_bound", 1000.0)), jnp.maximum(tke_new, 0.0))
    if bounded:
        if transport_velocities is None or getattr(transport_velocities, "ru_full", None) is None:
            raise NotImplementedError("km_opt=2 flow_dep_bdy needs the acoustic-averaged ru_m/rv_m")
        tke_new = flow_dep_bdy(tke_new, transport_velocities.ru_full, transport_velocities.rv_full,
                               namelist.boundary_config)
    return state.replace(tke=tke_new.astype(state.tke.dtype))


__all__ = ["LES3D_MOIST_SPECIES", "LES3D_SCALAR_SPECIES", "Les3dKm3Bundle", "les3d_km3_active",
           "les3d_km3_forward_tendencies", "les_tke_sixth_order_sc", "les_tke_stage_update"]
