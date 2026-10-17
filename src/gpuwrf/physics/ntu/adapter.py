"""MP_NTU caller: dry theta, actual Exner, staggered W and six precip phases."""
import jax.numpy as jnp
from .registry import output_mapping, SURFACE_LEAVES

OUTPUT_NAMES = (*output_mapping(), 'th', 'rainnc', 'snownc', 'grapnc', 'hailnc', 'sr',
                'clodncv', 'rainncv', 'icencv', 'snowncv', 'grapncv', 'hailncv')


def ntu_adapter(state, dt, grid=None, *, ccnty=2, return_precipitation=False):
    from gpuwrf.coupling.physics_couplers import (
        _to_columns, _from_columns, _dry_theta_view,
        _column_dz_from_state, _theta_m_from_microphysics)
    mapping = output_mapping()
    missing = [n for n in (*mapping.values(), *SURFACE_LEAVES)
               if getattr(state, n, None) is None]
    if missing:
        raise ValueError(f'NTU requires initialized prognostics/totals: {missing}')
    f = lambda a: jnp.asarray(a, jnp.float32)
    real = state.replace(theta=f(state.theta), qv=f(state.qv), _cast=False)
    old_th, old_qv = _dry_theta_view(real), f(state.qv)
    p = f(state.p)
    # module_big_step_utilities_em.F:4854, WRF model constants rcp=2/7.
    pii = (p / jnp.float32(100000)) ** jnp.float32(2 / 7)
    fields = {n: _to_columns(f(getattr(state, target))) for n, target in mapping.items()}
    fields.update(th=_to_columns(old_th), p=_to_columns(p), pii=_to_columns(pii),
                  dz=_column_dz_from_state(real, grid, dtype=jnp.float32),
                  w=_to_columns(f(state.w)), rainnc=f(state.ntu_rainnc),
                  snownc=f(state.ntu_snownc), grapnc=f(state.graupel_acc), hailnc=f(state.hail_acc))
    from gpuwrf.backends import pallas as backend
    names = OUTPUT_NAMES
    if backend.cuda_native():
        from gpuwrf.kernels.phys_ntu import full_column
        out = full_column(fields, f(dt), ccnty=ccnty, output_names=names)
    else:
        from .column import step
        out = step(fields, f(dt), ccnty=ccnty, xp=jnp, output_names=names)
    updates = {target: f(_from_columns(out[n])) for n, target in mapping.items()}
    updates['theta'] = _theta_m_from_microphysics(
        old_th, old_qv, _from_columns(out['th']), updates['qv'], jnp.float32)
    surface = lambda n: out[n].reshape(state.rain_acc.shape)
    # Original RAINNC includes cloud, rain, ice, snow, graupel and hail.
    # Preserve its REAL association in the authoritative overlapping totals;
    # independent model buckets consume instantaneous fluxes, never differences
    # of already-rounded accumulated totals.
    liquid = surface('clodncv') + surface('rainncv')
    ice, snow = surface('icencv'), surface('snowncv')
    graupel, hail = surface('grapncv'), surface('hailncv')
    updates.update(rain_acc=f(state.rain_acc) + liquid,
                   ice_acc=f(state.ice_acc) + ice, snow_acc=f(state.snow_acc) + snow,
                   graupel_acc=surface('grapnc'), hail_acc=surface('hailnc'),
                   ntu_rainnc=surface('rainnc'), ntu_snownc=surface('snownc'), ntu_sr=surface('sr'))
    updated = state.replace(**updates)
    # Preserve the actual WRF surface-driver interface, including NTU's
    # upstream peculiarity: public RAINNCV is rain-only and SNOWNCV snow-only.
    # Private cloud/ice fluxes contribute to RAINNC/SNOWNC but are not passed
    # in those public instantaneous slots (module_mp_ntu.F:1720,2573–2578;
    # module_surface_driver.F:1569,3283; module_sf_noahmpdrv.F:779–783).
    # The generic holder accepts original total/hail overrides; no differencing
    # of rounded totals or repair of the upstream partition is needed.
    precip = dict(rain=surface('rainncv'), snow=snow, graupel=graupel,
                  ice=jnp.zeros_like(ice), hail=hail, total=surface('rainncv'), sr=surface('sr'))
    return (updated, precip) if return_precipitation else updated
