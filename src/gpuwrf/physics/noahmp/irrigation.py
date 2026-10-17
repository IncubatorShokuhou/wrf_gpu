"""WRF Noah-MP irrigation subroutines, literal REAL column arithmetic.

The driver calls the trigger at soil cadence, sprinkler before PRECIP_HEAT,
and flood followed by drip immediately before SOILWATER. No extra work or
arrays are created when the option is disabled.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class IrrigationParameters(NamedTuple):
    irr_frac: object
    irr_har: object
    irr_lai: object
    irr_mad: object
    filoss: object
    sprir_rate: object
    micir_rate: object
    firtfac: object
    ir_rain: object
    pltday: object
    hsday: object
    nroot: object
    smcwlt: object
    smcref: object
    smcmax: object
    bexp: object
    dksat: object
    dwsat: object


class IrrigationState(NamedTuple):
    irnumsi: object
    irnummi: object
    irnumfi: object
    irwatsi: object
    irwatmi: object
    irwatfi: object
    irsivol: object
    irmivol: object
    irfivol: object
    ireloss: object
    irrsplh: object
    loss_fallback: object  # on-device count of WRF's explicit IRRLOSS=4 fallback


class IrrigationStatic(NamedTuple):
    parameters: IrrigationParameters
    irfract: object
    sifract: object
    mifract: object
    fifract: object
    cropland: object
    initial: object = None


def _r(value):
    return jnp.asarray(value, jnp.float32)


def seed_irrigation(shape):
    zero = jnp.zeros(shape, jnp.float32)
    count = jnp.zeros(shape, jnp.int32)
    return IrrigationState(count, count, count, *(zero for _ in range(8)), count)


def method_fractions(bundle, opt_irrm):
    """NOAHMP_SFLX:884-907, including unknown-method sprinkler fallback."""
    p = bundle.parameters
    si, mi, fi = map(_r, (bundle.sifract, bundle.mifract, bundle.fifract))
    if opt_irrm == 0:
        unknown = (si == 0.) & (mi == 0.) & (fi == 0.) & (_r(bundle.irfract) >= _r(p.irr_frac))
        si = jnp.where(unknown, 1., si)
    elif opt_irrm in (1, 2, 3):
        si, mi, fi = (jnp.full_like(si, float(opt_irrm == method)) for method in (1,2,3))
    else:
        raise ValueError("Noah-MP opt_irrm supports 0, 1, 2 or 3")
    return si, mi, fi


def trigger_irrigation(p, zsoil, sh2o, fveg, julian, irrfra, lai, fractions, state, *, opt_irr, opt_irrm):
    """TRIGGER_IRRIGATION:9235-9349; caller eligibility is applied separately."""
    zsoil, sh2o = _r(zsoil), _r(sh2o)
    wlt, ref = _r(p.smcwlt), _r(p.smcref)
    # zsoil is (4,) or already broadcast over the column grid.
    zsoil = jnp.reshape(zsoil, (4,) + (1,) * (sh2o.ndim - 1)) if zsoil.ndim == 1 else zsoil
    available = (sh2o[0] - wlt[0]) * (-1.0) * zsoil[0]
    limit = (ref[0] - wlt[0]) * (-1.0) * zsoil[0]
    for k in range(1, 4):
        active = k < jnp.asarray(p.nroot, jnp.int32)
        available = available + jnp.where(active, (sh2o[k] - wlt[k]) * (zsoil[k-1] - zsoil[k]), 0.)
        limit = limit + jnp.where(active, (ref[k] - wlt[k]) * (zsoil[k-1] - zsoil[k]), 0.)
    active = jnp.ones_like(_r(fveg), dtype=bool)
    if opt_irr == 2:
        active &= ~((_r(julian) < _r(p.pltday)) | (_r(julian) > _r(p.hsday) - _r(p.irr_har)))
    elif opt_irr == 3:
        active &= _r(lai) >= _r(p.irr_lai)
    elif opt_irr != 1:
        active &= False
    dry = available / limit <= _r(p.irr_mad)
    amount = (limit - available) * _r(irrfra) * _r(fveg)
    changes = {}
    for method, (count_name, amount_name, fraction) in enumerate(zip(
            ("irnumsi", "irnummi", "irnumfi"), ("irwatsi", "irwatmi", "irwatfi"), fractions), 1):
        old = _r(getattr(state, amount_name))
        enter = (old == 0.) & ((fraction > 0.) if opt_irrm == 0 else (opt_irrm == method))
        water = fraction * amount if opt_irrm == 0 else amount
        if method == 3:
            # WRF multiplies the reciprocal after the amount/fraction product.
            water = water * (1. / (1. - _r(p.filoss)))
        changes[amount_name] = jnp.where(active & dry & enter, water,
                                       jnp.where(active & ~dry, 0., old))
        changes[count_name] = jnp.asarray(getattr(state, count_name), jnp.int32) + (active & dry & enter).astype(jnp.int32)
    return state._replace(**changes)


def philip_infiltration(p, smc, sh2o, sice, dt):
    """IRR_PHILIP_INFIL:9525-9580 and its original WDFCND2 caller mapping."""
    from gpuwrf.physics.noahmp.water_hydro import _wdfcnd2
    smc, sh2o, sice = map(_r, (smc, sh2o, sice))
    maxice = jnp.maximum(jnp.maximum(sice[0], sice[1]), jnp.maximum(sice[2], sice[3]))
    wdf, wcnd = _wdfcnd2(sh2o[0], maxice, _r(p.bexp)[0], _r(p.smcmax)[0], _r(p.dksat)[0], _r(p.dwsat)[0])
    sp = jnp.sqrt(2. * jnp.maximum(0., _r(p.smcmax)[0] - smc[0]) * (_r(p.dwsat)[0] - wdf))
    ap = jnp.minimum(wcnd, (jnp.float32(2.) / jnp.float32(3.)) * _r(p.dksat)[0])
    ap = jnp.maximum(ap, (jnp.float32(1.) / jnp.float32(3.)) * _r(p.dksat)[0])
    return jnp.maximum(0., .5 * sp * _r(dt)**(-.5) + ap)


def sprinkler_irrigation(p, dt, sh2o, smc, t2, windu, windv, eair, fraction, amount):
    """SPRINKLER_IRRIGATION:9353-9425; return remainder, delivered water, evaporation, fallback predicate."""
    sh2o, smc, t2, windu, windv, eair, fraction, amount, dt = map(_r,
        (sh2o, smc, t2, windu, windv, eair, fraction, amount, dt))
    sice = jnp.maximum(0., smc - sh2o)
    fsur = philip_infiltration(p, smc, sh2o, sice, dt)
    temp_rate = _r(p.sprir_rate) * (jnp.float32(1.) / jnp.float32(1000.)) * dt / 3600.
    rate = jnp.minimum(jnp.minimum(fsur * dt, amount), temp_rate)
    windspeed = jnp.sqrt(windu**2. + windv**2.)
    esat = 610.8 * jnp.exp((17.27 * (t2 - 273.15)) / (237.3 + (t2 - 273.15)))
    warm = 4.375 * jnp.exp(.106 * windspeed) * ((esat - eair) * .01)**(-.092) * (t2 - 273.15)**(-.102)
    cold = 4.337 * jnp.exp(.077 * windspeed) * ((esat - eair) * .01)**(-.098)
    loss = jnp.where(t2 > 273.15, warm, cold)
    bad = jnp.isnan(loss) | (loss > 100.) | (loss < 0.)
    loss = jnp.where(bad, 4., loss)  # WRF explicit fallback, counted by the caller
    rate = rate * fraction
    exhausted = rate >= amount
    rate = jnp.where(exhausted, amount, rate)
    remaining = jnp.where(exhausted, 0., amount - rate)
    evaporated = rate * loss * (jnp.float32(1.) / jnp.float32(100.))
    return remaining, rate - evaporated, evaporated, bad


def micro_irrigation(p, dt, sh2o, smc, sice, fraction, amount):
    """MICRO_IRRIGATION:9435-9476, applied to SH2O(1) by the WATER caller."""
    dt, fraction, amount = map(_r, (dt, fraction, amount))
    fsur = philip_infiltration(p, smc, sh2o, sice, dt)
    temp = _r(p.micir_rate) * (jnp.float32(1.) / jnp.float32(1000.)) * dt / 3600.
    rate = jnp.minimum(jnp.minimum(.5 * fsur * dt, amount), temp) * fraction
    return jnp.where(rate >= amount, 0., amount - rate), jnp.where(rate >= amount, amount, rate)


def flood_irrigation(p, dt, sh2o, smc, sice, fraction, amount):
    """FLOOD_IRRIGATION:9479-9522, applied to ACC_QINSUR by WATER."""
    dt, fraction, amount = map(_r, (dt, fraction, amount))
    fsur = philip_infiltration(p, smc, sh2o, sice, dt)
    rate = fsur * dt * _r(p.firtfac)
    rate = rate * fraction
    return jnp.where(rate >= amount, 0., amount - rate), jnp.where(rate >= amount, amount, rate)


def prepare_irrigation(bundle, state, land, forcing, phen, zsoil, rain, dt, *, opt_irr, opt_irrm):
    """NOAHMP_SFLX:910-941, calculate_soil=True / soil_update_steps=1."""
    p = bundle.parameters
    fractions = method_fractions(bundle, opt_irrm)
    crop = jnp.asarray(bundle.cropland, bool)
    irrfra = _r(bundle.irfract)
    rainy = rain >= _r(p.ir_rain) / 3600.
    irrigated = irrfra >= _r(p.irr_frac)
    empty = (state.irwatsi + state.irwatmi + state.irwatfi) == 0.
    eligible = crop & irrigated & ~rainy & empty
    triggered = trigger_irrigation(p, zsoil, land.sh2o, phen.fveg, forcing.julian,
                                  irrfra, phen.lai, fractions, state, opt_irr=opt_irr, opt_irrm=opt_irrm)
    state = jax.tree.map(lambda new, old: jnp.where(eligible, new, old), triggered, state)
    clear = rainy | ~irrigated
    state = state._replace(**{name:jnp.where(clear, 0., getattr(state,name))
                              for name in ("irwatsi","irwatmi","irwatfi")})
    active = crop & (state.irwatsi > 0.)
    ea = _r(forcing.qair) * _r(forcing.sfcprs) / (.622 + .378 * _r(forcing.qair))
    amount, rate, evaporation, bad = sprinkler_irrigation(p, dt, land.sh2o, land.smois,
        forcing.sfctmp, forcing.uu, forcing.vv, ea, fractions[0], state.irwatsi)
    rate, evaporation = jnp.where(active, rate, 0.), jnp.where(active, evaporation, 0.)
    eirr = evaporation * 1000. / _r(dt)
    firr = evaporation * 1000. * jnp.float32(2.5104e6) / _r(dt)
    state = state._replace(irwatsi=jnp.where(active,amount,state.irwatsi),
        irsivol=state.irsivol + rate * 1000., ireloss=state.ireloss + eirr * _r(dt),
        irrsplh=state.irrsplh + firr * _r(dt),
        loss_fallback=state.loss_fallback + (active & bad).astype(jnp.int32))
    return state, rate * 1000. / _r(dt), firr, eirr


def soil_irrigation(bundle, state, smc, sh2o, sice, zsoil, qinsur, dt, *, opt_irrm):
    """WATER:6187-6206: flood to QINSUR, then drip to liquid layer 1."""
    fractions = method_fractions(bundle,opt_irrm)
    crop = jnp.asarray(bundle.cropland,bool)
    active = crop & (state.irwatfi > 0.)
    remaining, rate = flood_irrigation(bundle.parameters,dt,sh2o,smc,sice,fractions[2],state.irwatfi)
    rate = jnp.where(active,rate,0.)
    qinsur = qinsur + rate / _r(dt)
    state = state._replace(irwatfi=jnp.where(active,remaining,state.irwatfi),irfivol=state.irfivol+rate*1000.)
    active = crop & (state.irwatmi > 0.)
    remaining, rate = micro_irrigation(bundle.parameters,dt,sh2o,smc,sice,fractions[1],state.irwatmi)
    rate = jnp.where(active,rate,0.)
    sh2o = sh2o.at[0].set(sh2o[0] + rate / (-1. * _r(zsoil)[0]))
    state = state._replace(irwatmi=jnp.where(active,remaining,state.irwatmi),irmivol=state.irmivol+rate*1000.)
    return sh2o,qinsur,state
