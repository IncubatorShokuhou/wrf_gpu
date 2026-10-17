"""WRF RK coupling of cumulus tendencies (F9): sources inside every RK stage, not an Euler add.

WRF never applies RTHCUTEN/RQ*CUTEN/RU/RVCUTEN to the state before the dycore.
``calculate_phy_tend`` (module_em.F:2252-2305) couples each uncoupled rate with the
time-n dry mass ``c1h*mut + c2h`` and ``phy_cu_ten`` (module_physics_addtendc.F) adds
it to t_tendf / moist_tendf with ``add_a2a`` (scalars) and to ru/rv_tendf with
``add_a2c`` (momentum); ``conv_t_tendf_to_moist`` then turns the summed dry theta
forcing into the theta_m forcing, and rk_addtend_dry / rk_update_scalar integrate it
at every RK stage and acoustic substep.  The port's KF (KF_TEND_RK_WRF) and cuplus
schemes already use this seam (``kf_tendf`` / ``cuplus_momentum`` in
operational_mode); this module gives every other cumulus scheme the same path.
"""
from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.coupling.kf_rk import kf_add_a2a

SCALARS = ("theta", "qv", "qc", "qr", "qi", "qs")


def _mass(mu_total, metrics, dtype):
    return (jnp.asarray(metrics.c1h, dtype)[:, None, None] * jnp.asarray(mu_total, dtype)[None]
            + jnp.asarray(metrics.c2h, dtype)[:, None, None])


def sources_from_coupled(coupled, *, specified_or_nested: bool, periodic_x: bool = False,
                         periodic_y: bool = False):
    """phy_cu_ten: add_a2a of already-coupled scalar rates, add_a2c of coupled u/v."""

    scalars = tuple(
        None if coupled.get(name) is None else
        kf_add_a2a(coupled[name], specified_or_nested=specified_or_nested, periodic_x=periodic_x)
        for name in SCALARS)
    momentum = None
    if coupled.get("u") is not None:
        from gpuwrf.coupling.cuplus import momentum_sources

        momentum = momentum_sources(coupled["u"], coupled["v"], specified_or_nested=specified_or_nested,
                                    periodic_x=periodic_x, periodic_y=periodic_y)
    return scalars, momentum


# Held KF-layout tuples (RTH, RQV, RQC, RQR, RQI, RQS, PRATEC[, RU, RV]) and the fields
# each held-rate scheme's phy_cu_ten case adds (module_physics_addtendc.F:1125-1514).
HELD_LAYOUT = ("theta", "qv", "qc", "qr", "qi", "qs", "pratec", "u", "v")
HELD_APPLIED = {
    4: ("theta", "qv", "qc", "qi"), 94: ("theta", "qv", "qc", "qi"), 95: ("theta", "qv", "qc", "qi"),
    14: ("theta", "qv", "qc", "qi", "u", "v"), 96: ("theta", "qv", "qc", "qi", "u", "v"),
    99: ("theta", "qv", "qc", "qr", "qi", "qs"),
}


def couple_held(rates, mu_total, metrics):
    """calculate_phy_tend (module_em.F:2249-2300): every held R*CUTEN profile x (c1h*mut + c2h)."""

    dtype = jnp.asarray(rates[0]).dtype
    mass = _mass(mu_total, metrics, dtype)
    return tuple(rate if name == "pratec" else mass * jnp.asarray(rate, dtype)
                 for name, rate in zip(HELD_LAYOUT, rates))


def held_rk_sources(coupled, scheme, *, specified_or_nested, periodic_x=False, periodic_y=False):
    named = dict(zip(HELD_LAYOUT, coupled))
    return sources_from_coupled({n: named[n] for n in HELD_APPLIED[int(scheme)]},
                                specified_or_nested=specified_or_nested,
                                periodic_x=periodic_x, periodic_y=periodic_y)


def decouple_held(coupled, final_mu_total, metrics):
    """phy_prep_part2 (solve_em.F:3669): held profiles / (c1h*muts + c2h); PRATEC is never coupled."""

    dtype = jnp.asarray(coupled[0]).dtype
    mass = _mass(final_mu_total, metrics, dtype)
    return tuple(rate if name == "pratec" else jnp.asarray(rate, dtype) / mass
                 for name, rate in zip(HELD_LAYOUT, coupled))


def cumulus_rk_sources(rates, mu_total, metrics, *, specified_or_nested: bool,
                       periodic_x: bool = False, periodic_y: bool = False):
    """calculate_phy_tend + phy_cu_ten for one cumulus call.

    ``rates`` maps ``theta`` (DRY RTHCUTEN), ``qv`` ... ``qs`` and optionally ``u``/``v``
    (mass-point RUCUTEN/RVCUTEN) to uncoupled WRF rates; species the scheme does
    not produce are simply absent.  Returns ``(kf_tendf, momentum)``: the six-slot
    coupled scalar tuple (``None`` for absent species) consumed like KF's, and the
    add_a2c face sources ``(ru, rv)`` or ``None``.
    """

    dtype = jnp.asarray(rates["theta"]).dtype
    mass = _mass(mu_total, metrics, dtype)
    coupled = {name: None if rate is None else mass * jnp.asarray(rate, dtype) for name, rate in rates.items()}
    return sources_from_coupled(coupled, specified_or_nested=specified_or_nested,
                                periodic_x=periodic_x, periodic_y=periodic_y)


def accumulate_rainc(state, increment):
    """advance_ppt RAINC accumulation (module_physics_addtendc.F:2286-2293) of this step's
    convective amount (mm): ``PRATEC*DT`` or the scheme's RAINCV, as its Euler path added."""

    from gpuwrf.coupling.physics_couplers import _output_dtype

    return state.replace(rainc_acc=(state.rainc_acc + jnp.asarray(increment, state.rainc_acc.dtype)
                                    ).astype(_output_dtype(state, "rainc_acc")))
