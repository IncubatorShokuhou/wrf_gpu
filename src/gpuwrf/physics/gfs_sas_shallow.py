"""WRF GFS OLD_ARW_SHALCV, shared by SAS94 and OSAS95 (DOUBLE islands).

Lookup tables come from pristine GFUNCPHYS. Only host arrays are cached, so a
first call inside a JAX trace cannot leak a tracer to the next compiled program.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.cumulus_scalesas import _consts


@lru_cache(maxsize=2)
def _tables(mode):
    if mode not in ("r4", "r8"):
        raise ValueError("literal mode must be r4 or r8")
    with np.load(Path(__file__).with_name(f"gfs_sas_tables_{mode}.npz")) as data:
        arrays = {k: np.array(v) for k, v in data.items()}
    for a in arrays.values():
        a.setflags(write=False)
    return arrays


def _interp2(tab, coeff, x, y, *, zero_below=False):
    nx, ny = tab.shape
    c1x, c2x, c1y, c2y = coeff
    xj = jnp.minimum(c1x+c2x*x, float(nx))
    yj = jnp.minimum(c1y+c2y*y, float(ny))
    valid = (xj >= 1.) & (yj >= 1.)
    xj, yj = jnp.maximum(xj, 1.), jnp.maximum(yj, 1.)
    jx = jnp.minimum(xj, nx-1.).astype(jnp.int32)
    jy = jnp.minimum(yj, ny-1.).astype(jnp.int32)
    ix, iy = jx-1, jy-1
    ftx1 = tab[ix,iy] + (xj-jx)*(tab[ix+1,iy]-tab[ix,iy])
    ftx2 = tab[ix,iy+1] + (xj-jx)*(tab[ix+1,iy+1]-tab[ix,iy+1])
    value = ftx1 + (yj-jy)*(ftx2-ftx1)
    return jnp.where(valid, value, 0.) if zero_below else value


def _funcs(mode):
    host = _tables(mode)
    data = {k: jnp.asarray(v, jnp.float64) for k, v in host.items()}
    coeff = data["coefficients"]

    def ftdp(pv):
        tab = data["tdp"]
        xj = jnp.clip(coeff[0]+coeff[1]*pv, 1., float(tab.size))
        jx = jnp.minimum(xj, tab.size-1.).astype(jnp.int32)
        return tab[jx-1]+(xj-jx)*(tab[jx]-tab[jx-1])

    def fthe(t, pk):
        return _interp2(data["the"], coeff[2:6], t, pk, zero_below=True)

    def ftlcl(t, tdpd):
        return _interp2(data["tlcl"], coeff[6:10], t, tdpd)

    def stma(the, pk):
        return (_interp2(data["tma"], coeff[10:14], the, pk),
                _interp2(data["qma"], coeff[10:14], the, pk))
    return ftdp, fthe, ftlcl, stma


def old_arw_shalcv(t, q, dele, prsi, prsl, prslk, kuo, dt, *, mode="r4"):
    """One padded 1-based WRF column; pressure arrays in kPa, DT in seconds.

    WRF compresses inactive columns and solves a shared [K1,K2] range. Independent
    columns outside their own cloud have identity rows, so a full-range solve
    preserves those values and avoids cross-column dependence in the port.
    """
    c = _consts(mode)
    ftdp, fthe, ftlcl, stma = _funcs(mode)
    km = t.size-1
    K = jnp.arange(km+1)
    g, cp, hvap, rd = c.grav, c.cp, c.hvap, c.rd
    eldq = hvap*(q[:-1]-q[1:])
    cpdt = cp*(t[:-1]-t[1:])
    rtdls = (prsl[:-1]-prsl[1:])/prsi[1:km+1]*rd*.5*(t[:-1]+t[1:])
    unstable = (kuo == 0) & jnp.any((K[:-1] >= 1) & (eldq+cpdt-rtdls > 0.))
    # MSTADBT3 called with KM-1, K1=K2=2.
    pv = 1000.*prsl[2]*q[2]/(c.eps-c.epsm1*q[2])
    tdpd = t[2]-ftdp(pv)
    tlcl = jnp.where(tdpd > 0., ftlcl(t[2],tdpd), t[2])
    slklcl = jnp.where(tdpd > 0., prslk[2]*tlcl/t[2], prslk[2])
    thema = fthe(tlcl, slklcl)
    slkma = jnp.where(thema > 0., slklcl, 0.)
    thema = jnp.maximum(thema, 0.)
    lifted = (K >= 2) & (K <= km-1) & (prslk <= slkma)
    klcl = jnp.min(jnp.where(lifted, K, km))
    tma, qma = stma(thema, prslk)
    buoyant = lifted & (tma*(1.+c.fv*qma) > t*(1.+c.fv*q))
    ktop = jnp.max(jnp.where(buoyant, K, 0))
    ktopm = jnp.max(jnp.where((K >= 2) & (K <= km) & (prsi[1]-prsi[:km+1] <= 30.), K, 0))
    ktopm = jnp.where(ktopm > 0, ktopm, km)
    kbot = jnp.minimum(klcl-1, ktopm-1)
    ktop = jnp.minimum(ktop+1, ktopm)
    edges = (K[:-1] >= kbot) & (K[:-1] < ktop) & (K[:-1] >= 1)
    active = unstable & jnp.any(edges & (eldq+cpdt-rtdls > 0.))
    edges = edges & active
    # SHALCV coefficients and explicit dry-static-energy correction.
    ck = jnp.where(K[:-1] == kbot, 1.5,
         jnp.where(K[:-1] == ktop-1, 1.,
         jnp.where(K[:-1] == ktop-2, 3.,
         jnp.where((K[:-1] > kbot) & (K[:-1] < ktop-2), 5., 0.))))
    ck = jnp.where(edges, ck, 0.)
    au = jnp.where(edges, g/rtdls, 0.)
    dsig = prsl[:-1]-prsl[1:]
    dsdz1 = ck*dsig*au*(g/cp)
    dsdz2 = ck*dsig*au*au
    # Row 0 is padding; reciprocal denominators on inactive edges must be safe.
    dtodsl = dt/jnp.where(K[:-1] >= 1, dele[:-1], 1.)
    dtodsu = dt/dele[1:]
    upper = -dtodsl*dsdz2
    lower = -dtodsu*dsdz2
    # WRF visits interfaces from bottom to top. Each interior row first gets
    # the lower-interface subtraction and only then its upper-interface add.
    # Reversing those two rounded DOUBLE operations changes REAL outputs on
    # the rare work-rejection/shallow witnesses.
    diag = jnp.ones(km+1, jnp.float64).at[1:].add(-lower).at[:-1].add(-upper)
    rhs_t = t.at[1:].add(-dtodsu*dsdz1).at[:-1].add(dtodsl*dsdz1)
    # TRIDI2T3: literal reciprocal then multiply, two right-hand sides.
    def forward(carry, k):
        au_prev, q_prev, t_prev = carry
        lo = lower[k-1]
        fk = 1./(diag[k]-lo*au_prev)
        a = fk*jnp.where(k < km, upper[k], 0.)
        qk = fk*(q[k]-lo*q_prev)
        tk = fk*(rhs_t[k]-lo*t_prev)
        return (a,qk,tk), (a,qk,tk)
    zero = jnp.asarray(0., jnp.float64)
    _, (upper_s, q_s, t_s) = jax.lax.scan(forward, (zero,zero,zero), jnp.arange(1,km+1))
    def backward(carry, k):
        q_next, t_next = carry
        qk = q_s[k-1]-upper_s[k-1]*q_next
        tk = t_s[k-1]-upper_s[k-1]*t_next
        return (qk,tk), (qk,tk)
    _, (qo,to) = jax.lax.scan(backward, (q_s[-1],t_s[-1]), jnp.arange(1,km), reverse=True)
    qo = q.at[1:km].set(qo).at[km].set(q_s[-1])
    to = t.at[1:km].set(to).at[km].set(t_s[-1])
    return jnp.where(active,to,t), jnp.where(active,qo,q)
