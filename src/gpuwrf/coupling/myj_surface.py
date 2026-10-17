"""WRF MYJSFC -> land -> MYJPBL coupling (sf_sfclay_physics=2 with bl_pbl_physics=2).

WRF runs MYJSFC ONCE per step in the surface driver (module_surface_driver.F:2210-2224)
on the phy_prep operands (dry th_phy/t_phy, hydrostatic p_phy/p8w, dz8w with g=9.81,
TKE_PBL) and its grid INOUT memory: UST, ZNT, QSFC on the grid and THZ0, QZ0, UZ0,
VZ0, AKHS, AKMS carried across steps (module_sf_myjsfc.F:300-310, NTSD==1 cold start
:186-200).  Noah-MP consumes no MYJSFC output except ZNT; MYJPBL (pbl_driver.F:
1439-1465) reads THIS solve's AKHS/AKMS/THZ0/QZ0/UZ0/VZ0/CT/UST, the post-LSM grid LH
as ELFLX and CHKLOWQ (=1 after an LSM, surface_driver.F:3429), and returns updated
THZ0/QZ0/QSFC (module_bl_myjpbl.F:540-563).

QNSESFC is the MYJSFC derivative with the identical INOUT memory, so the Janjic-family
memory rides the existing ``qnse`` carry leaf (QnseCarry, restart schema E78): the
surface fields akhs/akms/thz0/qz0/uz0/vz0/ct/z0base/lh are used, the three QNSE
mass-flux profiles stay zero for MYJ.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_R_D, _R_V, _CP = 287.0, 461.6, 1004.5


def stage_profiles(state, grid):
    """phy_prep REAL operands MYJSFC/MYJPBL read, as (ncol, nz) bottom-up columns."""
    from gpuwrf.coupling.physics_couplers import (_u_mass, _v_mass, _dry_theta_view,
        _wrf_hydrostatic_pressure_profiles_from_state)
    f32 = jnp.float32
    nz, ny, nx = state.theta.shape
    real = lambda x: jnp.asarray(x, f32)
    cols = lambda a: jnp.moveaxis(real(a), 0, -1).reshape(ny * nx, a.shape[0])
    with jax.enable_x64(False):
        theta = real(_dry_theta_view(state))
        p = real(state.p)
        pi = (p / f32(1.e5)) ** (f32(_R_D) / f32(_CP))
        p_hyd, p_hyd_w, psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, grid.metrics, output_dtype=f32)
        z = real(state.ph) / f32(9.81)
        profiles = dict(u=cols(_u_mass(state)), v=cols(_v_mass(state)), th=cols(theta), t=cols(theta * pi),
                        qv=cols(state.qv), qc=cols(state.qc), p_mid=cols(p_hyd), p_int=cols(real(p_hyd_w)),
                        dz=cols(z[1:] - z[:-1]), tke=cols(f32(.5) * real(state.qke)))
        profiles['ht'] = z[0].reshape(-1)  # WRF HT: terrain height of the lowest face
    return profiles, real(psfc).reshape(-1), (nz, ny, nx)


def myj_surface_step(state, held, grid, *, first_timestep=False, land=None):
    """One MYJSFC call: ``(state', held', noah_surface, outputs)``.

    ``held`` is the Janjic-family memory (QnseCarry).  ``state'`` carries the WRF
    grid outputs (UST, ZNT, QSFC, HFX, QFX, PBLH) and the kinematic handles; the
    Noah-MP ``surface_diagnostics`` follow the EpsilonNoahSurface ABI.
    """
    from gpuwrf.coupling.epsilon_pbl import EpsilonNoahSurface
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    from gpuwrf.physics.sf_myj import myjsfc_columns
    if held is None:
        raise ValueError('MYJSFC surface memory is missing (E78)')
    f32 = jnp.float32
    prof, psfc, (nz, ny, nx) = stage_profiles(state, grid)
    flat = lambda a: jnp.asarray(a, f32).reshape(-1)
    znt = flat(state.roughness_m)
    out = myjsfc_columns(prof['u'], prof['v'], prof['t'], prof['th'], prof['qv'], prof['qc'], prof['p_mid'],
                         prof['dz'], prof['tke'], tsk=flat(state.t_skin), xland=flat(state.xland), z0base=znt,
                         psfc=psfc, znt=znt, ustar=flat(state.ustar), mavail=flat(state.mavail),
                         qsfc=flat(state.qsfc), thz0=flat(held.thz0), qz0=flat(held.qz0), uz0=flat(held.uz0),
                         vz0=flat(held.vz0), itimestep=jnp.where(jnp.asarray(first_timestep), 1, 2),
                         akhs=flat(held.akhs), akms=flat(held.akms), wrf_real=True, ht=prof['ht'])
    r = lambda a: jnp.asarray(a, f32).reshape(ny, nx)
    out = {k: r(v) for k, v in out.items()}
    with jax.enable_x64(False):
        u0, v0 = prof['u'][:, 0].reshape(ny, nx), prof['v'][:, 0].reshape(ny, nx)
        rho = out['flhc'] / (f32(_CP) * jnp.maximum(out['akhs'], f32(1.e-30)))
        flt = out['hfx'] / jnp.maximum(rho * f32(_CP), f32(1.e-12))
        flq = out['qfx'] / jnp.maximum(rho, f32(1.e-12))
        th0 = prof['th'][:, 0].reshape(ny, nx); qv0 = prof['qv'][:, 0].reshape(ny, nx)
        ep1 = f32(_R_V) / f32(_R_D) - f32(1.)
        fluxes = SurfaceFluxes(ustar=out['ustar'], theta_flux=flt, qv_flux=flq,
            tau_u=out['akms'] * (u0 - out['uz0']), tau_v=out['akms'] * (v0 - out['vz0']), rhosfc=rho,
            fltv=(f32(1.) + ep1 * qv0) * flt + ep1 * th0 * flq, xland=jnp.asarray(state.xland, f32), wspd=None)
        noah = EpsilonNoahSurface(fluxes, out['hfx'], out['flx_lh'], out['u10'], out['v10'], out['t02'],
            out['th02'], out['q02'], out['qsfc'], jnp.asarray(state.mol, f32), out['znt'], out['chs'],
            out['chs2'], out['cqs2'], None if land is None else land.ch, None if land is None else land.cm,
            psfc.reshape(ny, nx))
    updates = dict(ustar=out['ustar'], theta_flux=flt, qv_flux=flq, tau_u=fluxes.tau_u, tau_v=fluxes.tau_v,
                   rhosfc=rho, fltv=fluxes.fltv, roughness_m=out['znt'], qsfc=out['qsfc'], hfx=out['hfx'],
                   qfx=out['qfx'], pblh=out['pblh'])
    if getattr(state, 'sfc_wspd', None) is not None:
        # surface_driver.F:2232 WSPD=MAX(SQRT(u**2+v**2),0.001) after MYJSFC.
        updates['sfc_wspd'] = jnp.maximum(jnp.sqrt(u0 * u0 + v0 * v0), f32(.001))
    next_state = state.replace(_cast=False, **{k: jnp.asarray(val, getattr(state, k).dtype)
                                               for k, val in updates.items()})
    next_held = held._replace(**{k: out[k] for k in ('akhs', 'akms', 'thz0', 'qz0', 'uz0', 'vz0', 'ct')},
                              lh=out['flx_lh'])
    return next_state, next_held, noah, out


def initial_myj_memory(state):
    """Registry-zero MYJSFC memory (module_sf_myjsfc.F MYJSFCINIT sets only LOWLYR in ARW)."""
    from gpuwrf.coupling.qnse_adapter import initial_qnse_carry
    return initial_qnse_carry(state)
