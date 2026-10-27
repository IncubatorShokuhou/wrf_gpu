"""Independent NumPy oracle of WRF's specified-ROOT ring-0 walk (o5-coast, no gpuwrf imports).

Pristine WRF 4.7.1 references:
  * real.exe stores the lateral boundary records mass-COUPLED (dyn_em/module_big_step_utilities_em.F:524 ``couple``):
      u: u*(c1h*muu+c2h)/msfuy, v: v*(c1h*muv+c2h)/msfvx, t: t*(c1h*(mu+mub)+c2h), h(ph): ph*(c1f*(mu+mub)+c2f),
      w: w*(c1f*(mu+mub)+c2f)/msfty; mu is NOT coupled.  muu/muv from calc_mu_staggered (:403): interior faces
      average the two cells, the outermost faces take the edge cell.
  * the tendency record is (rec[n+1] - rec[n]) / interval (real.exe stuff_bdytend).
  * solve_em.F:378 advances dtbc by dt BEFORE the step; the relaxation target is bdy + dtbc*btend (end of step).
  * small_step_prep (module_small_step_em.F:113-199) restarts every RK stage from time level n (muts = mub + mu_1),
    and spec_bdyupdate (solve_em.F:1346/1356/1462/1472/1482, spec_bdyupdate_ph :1587) adds dts_rk*btend to the
    ring-0 work arrays after every acoustic substep.  Hence after substep m of stage k the ring holds the
    boundary trajectory at  t_n + m*dts_k  (stage 1: 1 substep of dt/3; stage 2: ns/2 of dt/ns; stage 3: ns of dt/ns).
"""
from __future__ import annotations

import numpy as np


def wrf_rk_substeps(dt: float, ns: int) -> list[tuple[float, int, float]]:
    """WRF RK3 (dt_rk, number_of_small_timesteps, dts_rk) per stage (solve_em.F rk_order 3 branch)."""
    dts = dt / ns
    return [(dt / 3.0, 1, dt / 3.0), (dt / 2.0, max(ns // 2, 1), dts), (dt, ns, dts)]


def wrf_ring_times(step_end: float, dt: float, ns: int) -> list[list[float]]:
    """Valid time of the WRF ring-0 work arrays after every acoustic substep, per RK stage."""
    t_n = step_end - dt
    return [[t_n + m * dts for m in range(1, n + 1)] for _dt_rk, n, dts in wrf_rk_substeps(dt, ns)]


def couple_mass(field, mu, mub, c1, c2, msf=None):
    """WRF couple for t/ph/w-type fields: field*(c1*(mu+mub)+c2)[/msf]; c1,c2 broadcast over levels."""
    out = field * (c1 * (mu + mub) + c2)
    return out if msf is None else out / msf


def mu_staggered_u(mu_total_row):
    """calc_mu_staggered along x for one row of total mass (mu+mub): length nx -> nx+1 faces."""
    m = np.asarray(mu_total_row, dtype=np.float64)
    faces = np.empty(m.shape[:-1] + (m.shape[-1] + 1,))
    faces[..., 1:-1] = 0.5 * (m[..., 1:] + m[..., :-1])
    faces[..., 0] = m[..., 0]
    faces[..., -1] = m[..., -1]
    return faces


def btend(rec_a, rec_b, interval: float):
    """real.exe stuff_bdytend: per-second tendency between two coupled records."""
    return (np.asarray(rec_b, np.float64) - np.asarray(rec_a, np.float64)) / float(interval)


def walk(start, rate, dts: float, nsub: int):
    """spec_bdyupdate: start + m*dts*rate after substep m = 1..nsub (returns list)."""
    out, x = [], np.asarray(start, np.float64)
    for _ in range(nsub):
        x = x + dts * np.asarray(rate, np.float64)
        out.append(x.copy())
    return out
