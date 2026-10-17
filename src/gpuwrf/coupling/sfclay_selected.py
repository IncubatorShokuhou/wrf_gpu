"""Selected surface layer sf_sfclay_physics=3/7/91: ONE WRF surface_driver call per step.

WRF (module_surface_driver.F) runs exactly the selected scheme -- SF_GFS (3,
:2344-2353), PXSFCLAY (7, :2165-2176) or SFCLAY (91, :2064-2077) -- on the
phy_prep operands (dry t_phy/th_phy, hydrostatic p_phy, PSFC=p8w(kts), dz8w with
g=9.81) and its grid INOUT memory (UST, ZNT, MOL, QSFC, HFX, QFX, PBLH).  Its
UST/stress, water HFX/QFX/LH/QSFC/ZNT, U10/V10 and CHS2/CQS2 reach Noah-MP,
the PBL and history; noahmplsm takes no other sfclay output (own CHXY/CMXY).

This module stages those operands exactly like the qualified SF1 producer
(coupling.epsilon_pbl.epsilon_surface_step), calls the selected kernel once,
writes the WRF INOUT fields back to State, and returns the Noah-MP diagnostics
and the PBL handoff.  No MYNN-SL re-solve and no substitution happens here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

SELECTED_SFCLAY_OPTIONS = (3, 7, 91)

# WRF module_model_constants values the surface driver passes (E83).
_R_D, _R_V, _CP = 287.0, 461.6, 1004.5


def stage_inputs(state, grid):
    """WRF phy_prep lowest-level REAL operands + carried grid memory (2-D)."""
    from gpuwrf.coupling.physics_couplers import (_u_mass, _v_mass, _dry_theta_view,
        _wrf_hydrostatic_pressure_from_state, _wrf_phy_prep_rho_from_state)
    f32 = jnp.float32
    real = lambda x: jnp.asarray(x, f32)
    with jax.enable_x64(False):
        # Same operand assembly as the qualified SF1 call (epsilon_surface_step).
        theta = real(_dry_theta_view(state))
        p = real(state.p)
        pi = (p / 100000.) ** (f32(287.) / f32(1004.5))
        phyd, psfc = _wrf_hydrostatic_pressure_from_state(state, grid.metrics, output_dtype=f32)
        rho = _wrf_phy_prep_rho_from_state(state, grid.metrics, output_dtype=f32)
        dz = (real(state.ph[1]) - real(state.ph[0])) / f32(9.81)
        shape = jnp.shape(state.t_skin)
        inputs = dict(
            u=real(_u_mass(state)[0]), v=real(_v_mass(state)[0]), t=theta[0] * pi[0], th=theta[0],
            qv=real(state.qv[0]), p=real(phyd[0]), dz=dz, psfc=real(psfc), rho=real(rho[0]),
            tsk=real(state.t_skin), xland=real(state.xland), lakemask=real(state.lakemask),
            mavail=real(state.mavail), pblh=real(state.pblh), znt=real(state.roughness_m),
            ust=real(state.ustar), mol=real(state.mol), qsfc=real(state.qsfc), hfx=real(state.hfx),
            qfx=real(state.qfx), dx=jnp.full(shape, f32(grid.projection.dx_m)))
    return inputs


def column_call(sf, inputs, itimestep):
    """Call the selected original-WRF surface layer once on 2-D REAL operands.

    Returns REAL WRF-named outputs.  Keys absent from the WRF routine are absent.
    ``itimestep`` is the WRF step index (only PXSFCLAY reads it: ==1 forces QSFC).
    """
    f32 = jnp.float32
    shape = jnp.shape(inputs['u'])
    flat = {k: jnp.asarray(v, f32).reshape(-1) for k, v in inputs.items()}
    if sf == 91:
        from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns
        with jax.enable_x64(False):
            out = sfclay_old_mm5_columns(*(flat[k] for k in (
                'u', 'v', 't', 'qv', 'p', 'dz', 'psfc', 'tsk', 'xland', 'znt', 'ust', 'mol', 'qsfc',
                'pblh', 'mavail', 'lakemask', 'dx', 'hfx', 'qfx')), wrf_constants=True)
        keys = ('ust', 'znt', 'mol', 'zol', 'rmol', 'regime', 'psim', 'psih', 'fm', 'fh', 'qsfc', 'qgh',
                'cpm', 'hfx', 'qfx', 'lh', 'flhc', 'flqc', 'chs', 'chs2', 'cqs2', 'u10', 'v10', 'th2',
                't2', 'q2', 'gz1oz0', 'wspd', 'br')
    elif sf == 7:
        from gpuwrf.physics.sfclay_pleim_xiu import PleimXiuSfclayColumnState, pxsfclay_run
        with jax.enable_x64(False):
            column = PleimXiuSfclayColumnState(*(flat[k] for k in (
                'u', 'v', 't', 'th', 'qv', 'p', 'dz', 'psfc', 'tsk', 'xland', 'mavail', 'znt', 'ust',
                'mol', 'hfx', 'qfx', 'qsfc', 'pblh', 'dx')),
                jnp.broadcast_to(jnp.asarray(itimestep, f32), flat['u'].shape))
            out = pxsfclay_run(column, dtype=f32)._asdict()
        # PXSFCLAY has no T2/Q2/TH2/FM/FH arguments (module_sf_pxsfclay.F:21-31).
        keys = ('ust', 'znt', 'mol', 'zol', 'rmol', 'regime', 'psim', 'psih', 'qsfc', 'qgh', 'cpm',
                'hfx', 'qfx', 'lh', 'flhc', 'flqc', 'chs', 'chs2', 'cqs2', 'u10', 'v10', 'gz1oz0',
                'wspd', 'br')
    elif sf == 3:
        from gpuwrf.physics.sfclay_gfs import sf_gfs_columns
        # PROGTM is kind_phys DOUBLE in WRF; the kernel stores REAL grid outputs.
        out = sf_gfs_columns(*(flat[k] for k in ('u', 'v', 't', 'qv', 'p', 'psfc', 'tsk', 'xland',
                                                 'znt', 'ust')), wrf_constants=True)
        keys = ('ust', 'znt', 'psim', 'psih', 'qsfc', 'qgh', 'cpm', 'hfx', 'qfx', 'lh', 'flhc', 'flqc',
                'chs', 'chs2', 'cqs2', 'u10', 'v10', 'gz1oz0', 'wspd', 'br')
    else:
        raise ValueError(f'selected surface layer supports sf_sfclay_physics {SELECTED_SFCLAY_OPTIONS}, got {sf}')
    return {k: jnp.asarray(out[k], f32).reshape(shape) for k in keys}


def _water_2m(out, inputs):
    """surface_driver ISWATER bulk form (:3446-3458) for schemes without T2/Q2."""
    f32 = jnp.float32
    rho = inputs['psfc'] / (f32(_R_D) * inputs['tsk'])
    q2 = jnp.where(out['cqs2'] < f32(1.e-5), out['qsfc'],
                   out['qsfc'] - out['qfx'] / (rho * jnp.maximum(out['cqs2'], f32(1.e-5))))
    t2 = jnp.where(out['chs2'] < f32(1.e-5), inputs['tsk'],
                   inputs['tsk'] - out['hfx'] / (rho * f32(_CP) * jnp.maximum(out['chs2'], f32(1.e-5))))
    th2 = t2 * (f32(1.e5) / inputs['psfc']) ** (f32(_R_D) / f32(_CP))
    return t2, q2, th2


def selected_surface_step(state, grid, sf, *, land=None, first_timestep=False):
    """One selected sf 3/7/91 call: ``(state', pbl_handoff, noah_surface, outputs)``.

    ``state'`` carries the WRF INOUT grid fields (UST, ZNT, MOL, QSFC, HFX, QFX,
    WSPD) and the kinematic handles MYNN-type consumers read.  ``noah_surface`` is
    the Noah-MP ``surface_diagnostics`` (EpsilonNoahSurface ABI); ``pbl_handoff``
    the selected PSIM/PSIH/BR/WSPD/GZ1OZ0/U10/V10 (WRF BL_GFS reads TSKOLD =
    post-LSM TSK, so no pre-LSM skin is handed over).
    """
    from gpuwrf.coupling.epsilon_pbl import EpsilonNoahSurface
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    f32 = jnp.float32
    inputs = stage_inputs(state, grid)
    itimestep = jnp.where(jnp.asarray(first_timestep), f32(1.), f32(2.))
    out = column_call(int(sf), inputs, itimestep)
    with jax.enable_x64(False):
        u, v, qv, rho = inputs['u'], inputs['v'], inputs['qv'], inputs['rho']
        cpm = f32(_CP) * (f32(1.) + f32(.8) * qv)
        wind = jnp.maximum(jnp.sqrt(u * u + v * v), f32(.1))
        flt = out['hfx'] / (rho * cpm)
        flq = out['qfx'] / rho
        ep1 = f32(_R_V) / f32(_R_D) - f32(1.)
        fluxes = SurfaceFluxes(ustar=out['ust'], theta_flux=flt, qv_flux=flq,
            tau_u=-out['ust'] ** 2 * u / wind, tau_v=-out['ust'] ** 2 * v / wind, rhosfc=rho,
            fltv=(f32(1.) + ep1 * qv) * flt + ep1 * inputs['th'] * flq, xland=inputs['xland'], wspd=None)
        mol = out.get('mol', inputs['mol'])
        if 't2' in out:
            t2, q2, th2 = out['t2'], out['q2'], out['th2']
        else:
            t2, q2, th2 = _water_2m(out, inputs)
        noah = EpsilonNoahSurface(fluxes, out['hfx'], out['lh'], out['u10'], out['v10'], t2, th2, q2,
            out['qsfc'], mol, out['znt'], out['chs'], out['chs2'], out['cqs2'],
            None if land is None else land.ch, None if land is None else land.cm, inputs['psfc'])
    updates = dict(ustar=fluxes.ustar, theta_flux=fluxes.theta_flux, qv_flux=fluxes.qv_flux,
                   tau_u=fluxes.tau_u, tau_v=fluxes.tau_v, rhosfc=fluxes.rhosfc, fltv=fluxes.fltv,
                   roughness_m=out['znt'], mol=mol, qsfc=out['qsfc'], hfx=out['hfx'], qfx=out['qfx'])
    if getattr(state, 'sfc_wspd', None) is not None:
        updates['sfc_wspd'] = out['wspd']
    next_state = state.replace(_cast=False, **{k: jnp.asarray(val, getattr(state, k).dtype)
                                               for k, val in updates.items()})
    handoff = {k: out[k] for k in ('psim', 'psih', 'br', 'wspd', 'u10', 'v10', 'gz1oz0', 'fm', 'fh', 'rmol')
               if k in out}
    return next_state, handoff, noah, out


def history_values(sf, noah, history):
    """No-LSM history fields this surface call produces (others stay held)."""
    values = {k: getattr(noah, k) for k in ('hfx', 'lh', 'u10', 'v10', 'psfc')}
    if int(sf) == 91:
        values.update(t2=noah.t2, q2=noah.q2)
    else:
        # SF_GFS/PXSFCLAY return no T2/Q2: without an LSM WRF history keeps them.
        values.update(t2=history.t2, q2=history.q2)
    return values
