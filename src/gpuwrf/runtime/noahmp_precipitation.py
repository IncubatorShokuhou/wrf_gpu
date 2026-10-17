"""WRF previous-step precipitation passed to the next Noah-MP surface call."""
from typing import NamedTuple
import jax.numpy as jnp

class NoahMPPrecipitation(NamedTuple):
    prcpconv: object
    prcpnonc: object
    prcpsnow: object
    prcpgrpl: object
    prcphail: object
    # WRF SR (the microphysics' own frozen fraction of RAINNCV, 0..1), held only for the
    # LSM that reads it: Noah-classic FFROZP=SR (module_sf_noahdrv.F:859-866; FRPCPN is
    # .true. because SR is present, module_surface_driver.F:2708). None on the Noah-MP/
    # RUC/SSiB path (release carry unchanged). A pre-F24 checkpoint restores None and
    # Noah-classic refuses it (E78).
    sr: object = None

def seed_precipitation(state, *, with_sr=False):
    zero = jnp.zeros(jnp.shape(state.t_skin), dtype=jnp.float32)
    return NoahMPPrecipitation(zero, zero, zero, zero, zero, zero if with_sr else None)

def precipitation_from_step(precip, convective_rate, dt, *, spec_zone=0, with_sr=False, thompson_sr=False):
    """Use instantaneous MP amounts (mm), never differences of rounded totals.

    WRF RAINNCV includes all phases; SNOWNCV includes snow+ice. Both the
    precipitation producer and its specified/nested exclusion match the MP
    driver's ownership. PRCPCONV is the actual held KF rain rate (mm/s).
    """
    p = {name: jnp.asarray(value, jnp.float32) for name, value in precip.items()}
    total = p['rain'] + p['snow'] + p['graupel'] + p['ice']
    sr = None
    if with_sr:
        if 'sr' in p:  # the scheme's own SR (WRF microphysics SR=SR argument)
            sr = p['sr']
        elif thompson_sr:  # mp_gt_driver :1308 SR=(pptsnow+pptgraul+pptice)/(RAINNCV+1.e-12)
            sr = (p['snow'] + p['graupel'] + p['ice']) / (total + jnp.float32(1.e-12))
        else:
            raise ValueError("Noah-classic FFROZP needs the microphysics' own SR; this producer supplies none")
    if 'hail' in p:
        total = total + p['hail']
    if 'total' in p:
        total = p['total']  # preserve a scheme's original REAL association
    snow = p['snow'] + p['ice']
    if spec_zone:
        ny, nx = total.shape
        mask = ((jnp.arange(ny) >= spec_zone) & (jnp.arange(ny) < ny - spec_zone))[:, None] & (
                (jnp.arange(nx) >= spec_zone) & (jnp.arange(nx) < nx - spec_zone))[None, :]
        total = jnp.where(mask, total, 0)
        snow = jnp.where(mask, snow, 0)
        p['graupel'] = jnp.where(mask, p['graupel'], 0)
        if 'hail' in p:
            p['hail'] = jnp.where(mask, p['hail'], 0)
        if sr is not None:  # the MP driver never sets SR on the specified/nested ring
            sr = jnp.where(mask, sr, 0)
    seconds = jnp.asarray(dt, jnp.float32)
    return NoahMPPrecipitation(jnp.asarray(convective_rate, jnp.float32), total / seconds,
                              snow / seconds, p['graupel'] / seconds,
                              p['hail'] / seconds if 'hail' in p else jnp.zeros_like(total),
                              None if sr is None else jnp.broadcast_to(sr, total.shape))
