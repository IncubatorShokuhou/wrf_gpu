"""JIT/vmap-batched Janjic Eta surface layer (``sf_sfclay_physics=2``).

This is the OPERATIONAL batched entrypoint for the Janjic Eta surface layer. The
single-column kernel ``physics.sfclay_janjic.myjsfc_column`` is ALREADY fully
``jax.jit``/``jax.vmap``-traceable (the SFCDIF Monin-Obukhov iteration is a
``jax.lax.scan``; the PSIM/PSIH integral functions are static device-table
lookups; everything else is pure ``jnp``). So unlike the MYJ PBL (whose host-
NumPy reference needed a traceable rewrite in ``physics.bl_myj``), the surface
layer only needs a ``jax.vmap`` wrapper over the ``(ncol, nz)`` grid columns --
no re-derivation. ``myjsfc_columns`` is the batched twin of ``myjsfc_column``.

Pairing: ``sf_sfclay_physics=2`` MUST run with ``bl_pbl_physics=2`` (MYJ PBL).
The surface layer runs FIRST in the WRF call chain and produces the exchange
coefficients / fluxes (USTAR, AKHS, AKMS, THZ0, QZ0, QSFC, CHKLOWQ, ELFLX) that
the paired MYJ PBL consumes; the operational scan + dispatcher enforce the pair.

fp64 throughout for savepoint parity; allocation-free; no host transfer.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config

configure_jax_x64()

import jax.numpy as jnp

from gpuwrf.physics.sfclay_janjic import myjsfc_column


def myjsfc_columns(
    u, v, temperature, theta, qv, qc, p_mid, dz, q2,
    *, tsk, xland, z0base, psfc, znt, ustar, mavail,
    qsfc, thz0, qz0, uz0=0.0, vz0=0.0, pblh=1000.0, dt=60.0,
    itimestep=None, akhs=None, akms=None, wrf_real=False, ht=None,
) -> dict:
    """``jax.vmap``-batched Janjic Eta surface layer over ``(ncol, nz)`` columns.

    Profile args are ``(ncol, nz)`` bottom-up (index 0 = lowest model layer).
    Surface scalars are ``(ncol,)`` (or broadcastable scalars). ``q2`` is TKE
    (m^2/s^2); MYJSFC converts to ``2*q2`` internally. Returns the batched SFCDIF
    output dict (each leaf ``(ncol,)``), identical keys to ``myjsfc_column``.
    ``itimestep``/``akhs``/``akms`` (WRF ITIMESTEP and INOUT AKHS/AKMS memory) are
    optional; omitted they keep the legacy cold NTSD>1 call. ``wrf_real`` evaluates the
    WRF REAL routine in REAL with the PSIM/PSIH tables and ZTMAX/DZETA exactly as the REAL
    MYJSFCINIT builds them (operational sf=2 caller); default keeps the frozen fp64 path.
    """

    ncol = u.shape[0]
    dt = jnp.float32 if wrf_real else jnp.float64
    f1 = lambda a: jnp.broadcast_to(jnp.asarray(a, dt).reshape(-1), (ncol,))

    memory = {}
    if itimestep is not None:
        memory = dict(itimestep=f1(itimestep), akhs=f1(0.0 if akhs is None else akhs),
                      akms=f1(0.0 if akms is None else akms))
        if ht is not None:
            memory['ht'] = f1(ht)

    def _one(u_c, v_c, t_c, th_c, qv_c, qc_c, pmid_c, dz_c, q2_c,
             tsk_c, xland_c, z0base_c, psfc_c, znt_c, ust_c, mav_c,
             qsfc_c, thz0_c, qz0_c, uz0_c, vz0_c, pblh_c, mem):
        return myjsfc_column(
            u=u_c, v=v_c, temperature=t_c, theta=th_c, qv=qv_c, qc=qc_c,
            p_mid=pmid_c, dz=dz_c, q2=q2_c, tsk=tsk_c, xland=xland_c,
            z0base=z0base_c, psfc=psfc_c, znt=znt_c, ustar=ust_c, mavail=mav_c,
            dt=dt, qsfc=qsfc_c, thz0=thz0_c, qz0=qz0_c, uz0=uz0_c, vz0=vz0_c,
            pblh=pblh_c, wrf_real=wrf_real, **mem,
        )

    return jax.vmap(_one)(
        jnp.asarray(u, dt), jnp.asarray(v, dt),
        jnp.asarray(temperature, dt), jnp.asarray(theta, dt),
        jnp.asarray(qv, dt), jnp.asarray(qc, dt),
        jnp.asarray(p_mid, dt), jnp.asarray(dz, dt),
        jnp.asarray(q2, dt),
        f1(tsk), f1(xland), f1(z0base), f1(psfc), f1(znt), f1(ustar), f1(mavail),
        f1(qsfc), f1(thz0), f1(qz0), f1(uz0), f1(vz0), f1(pblh), memory,
    )


__all__ = ["myjsfc_columns"]
