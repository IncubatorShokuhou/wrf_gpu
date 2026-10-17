"""WRF MMF groundwater operators (pristine module_sf_noahmp_groundwater.F).

REAL4 operators; spatial arrays are (y,x), soil profiles (soil,y,x).
This module alone does not admit opt_run=5: cold initialization, the Noah-MP
column coupling and the surface-driver cadence must also be qualified.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp


class GroundwaterStatic(NamedTuple):
    fdepth: jax.Array
    area: jax.Array
    topo: jax.Array
    eqzwt: jax.Array
    riverbed: jax.Array
    rivercond: jax.Array
    pexp: jax.Array
    smoiseq: jax.Array
    rechclim: jax.Array | None = None


class GroundwaterCarry(NamedTuple):
    zwt: jax.Array
    deeprech: jax.Array
    rech: jax.Array
    qlat: jax.Array
    qslat: jax.Array
    qrf: jax.Array
    qrfs: jax.Array
    qspring: jax.Array
    qsprings: jax.Array


@jax.tree_util.register_pytree_node_class
class MMFBundle(NamedTuple):
    """Complete restartable MMF data; readonly geometry remains on device."""
    static: GroundwaterStatic
    carry: GroundwaterCarry
    eligibility: jax.Array
    fsatmx: jax.Array
    seconds: jax.Array
    stepwtd: jax.Array
    opt_inf: int = 1
    # Registry ACRECH/ACQSPRING are invariant allocator zero with
    # NOAHMP_ACC_DT=0 (the only admitted mode). Persist as static scalars;
    # two redundant device planes would add 8 B/cell without any evolution.
    acrech: int = 0
    acqspring: int = 0

    def tree_flatten(self):
        return self[:6], self[6:]

    @classmethod
    def tree_unflatten(cls, aux, children):
        if isinstance(aux, int):  # Initial schema-1 files predate invariant accumulator metadata.
            aux = (aux, 0, 0)
        return cls(*children, *aux)


def lateral_flow(wtd, fdepth, topo, area, soil_type, landmask, dksat_table, seconds):
    """Pristine LATERALFLOW: eight neighbors, literal order, zero outer ring.

    ``landmask`` is the caller's eligibility predicate; the neighbor head and
    conductivity are computed on all cells just as in WRF. No periodic wrap.
    Soil types must be validated as 1..19 by the input initializer.
    """
    dtype = wtd.dtype
    factors = jnp.asarray((2., 3., 4., 10., 10., 12., 14., 20., 24., 28.,
                           40., 48., 2., 0., 10., 0., 20., 2., 2.), dtype=dtype)
    klat = dksat_table[soil_type - 1] * factors[soil_type - 1]
    # WRF branches before division. This inactive-branch operand prevents an
    # artificial divide by zero while preserving every FDEPTH>0 operation.
    depth = jnp.where(fdepth > 0, fdepth, jnp.ones_like(fdepth))
    kcell = jnp.where(wtd < -1.5,
                      fdepth * klat * jnp.exp((wtd + 1.5) / depth),
                      klat * (wtd + 1.5 + fdepth))
    kcell = jnp.where(fdepth > 0, kcell, jnp.zeros_like(kcell))
    head = topo + wtd
    center_k, center_h = kcell[1:-1, 1:-1], head[1:-1, 1:-1]
    q = jnp.zeros_like(center_h)
    # WRF I-1,J+1; I-1,J; I-1,J-1; I,J+1; I,J-1; I+1,J+1; I+1,J; I+1,J-1.
    for ys, xs, diagonal in ((slice(2, None), slice(None, -2), True),
                             (slice(1, -1), slice(None, -2), False),
                             (slice(None, -2), slice(None, -2), True),
                             (slice(2, None), slice(1, -1), False),
                             (slice(None, -2), slice(1, -1), False),
                             (slice(2, None), slice(2, None), True),
                             (slice(1, -1), slice(2, None), False),
                             (slice(None, -2), slice(2, None), True)):
        term = (kcell[ys, xs] + center_k) * (head[ys, xs] - center_h)
        if diagonal:
            term = term / jnp.sqrt(jnp.asarray(2., dtype=dtype))
        q = q + term
    flux = jnp.asarray(0.22754493, dtype=dtype) * q * seconds / area[1:-1, 1:-1]
    flux = jnp.where(landmask[1:-1, 1:-1], flux, jnp.zeros_like(flux))
    return jnp.pad(flux, ((1, 1), (1, 1)))


def update_wtd(dzs, zsoil, smceq, smcmax, psisat, bexp,
               totwater, wtd, smc, sh2o, smcwtd):
    """Pristine UPDATEWTD, including its literal downward SMC update sign.

    ``zsoil`` includes the surface at index zero. Fixed four-layer loops remain
    column-local when lowered inside a column kernel; no host decisions.
    """
    nsoil = len(dzs)
    sice = smc - sh2o
    layers = list(smc)
    zero = jnp.zeros_like(wtd)
    up, down = totwater > 0, totwater < 0
    resolved = wtd >= zsoil[nsoil]
    shallow = wtd >= zsoil[nsoil] - dzs[-1]
    deep = ~shallow
    water = totwater
    old_wtd = wtd
    # The descending Fortran search returns K=0 when every comparison fails.
    iwtd = jnp.zeros_like(wtd, dtype=jnp.int32)
    found = jnp.zeros_like(resolved)
    for k in range(nsoil - 1, 0, -1):
        match = ~found & (wtd < zsoil[k])
        iwtd = jnp.where(match, k, iwtd)
        found = found | match
    start_layer = iwtd  # zero-based KWTD=IWTD+1
    eqdeep = jnp.maximum(smcmax * (psisat / (psisat - dzs[-1])) ** (1. / bexp), 1.e-4)

    # Deep rising table first fills the interval below the fictitious layer.
    capacity = (smcmax - smcwtd) * (zsoil[nsoil] - dzs[-1] - wtd)
    partial = up & deep & (water <= capacity)
    wtd = jnp.where(partial, wtd + water / jnp.where(partial, smcmax - smcwtd, 1.), wtd)
    water = jnp.where(partial, zero, water)
    enters_deep_layer = up & deep & ~partial
    water = jnp.where(enters_deep_layer, water - capacity, water)
    wtd = jnp.where(enters_deep_layer, zsoil[nsoil] - dzs[-1], wtd)

    # Rising table in the fictitious layer below resolved soil.
    fictitious = up & ~resolved & ((~deep) | enters_deep_layer)
    capacity = (smcmax - smcwtd) * dzs[-1]
    partial = fictitious & (water <= capacity)
    next_smcwtd = jnp.minimum(smcwtd + water / dzs[-1], smcmax)
    diagnosed = (next_smcwtd * dzs[-1] - eqdeep * zsoil[nsoil]
                 + smcmax * (zsoil[nsoil] - dzs[-1])) / jnp.where(
                     partial & (deep | (next_smcwtd > eqdeep)), smcmax - eqdeep, 1.)
    diagnosed = jnp.where(deep, diagnosed, jnp.minimum(diagnosed, zsoil[nsoil]))
    wtd = jnp.where(partial & (deep | (next_smcwtd > eqdeep)), diagnosed, wtd)
    smcwtd = jnp.where(partial, next_smcwtd, smcwtd)
    water = jnp.where(partial, zero, water)
    fills_soil = fictitious & ~partial
    smcwtd = jnp.where(fills_soil, smcmax, smcwtd)
    water = jnp.where(fills_soil, water - capacity, water)
    start_layer = jnp.where(fills_soil, nsoil - 1, start_layer)

    for k in range(nsoil - 1, -1, -1):
        active = up & (resolved | fills_soil) & (start_layer >= k) & (water > 0)
        entering = active & (fills_soil | (start_layer > k))
        wtd = jnp.where(entering, zsoil[k + 1], wtd)
        capacity = dzs[k] * (smcmax - layers[k])
        partial = active & (water <= capacity)
        changed = jnp.minimum(layers[k] + water / dzs[k], smcmax)
        diagnosed = (changed * dzs[k] - smceq[k] * zsoil[k]
                     + smcmax * zsoil[k + 1]) / jnp.where(
                         partial & (changed > smceq[k]), smcmax - smceq[k], 1.)
        diagnosed = jnp.where(deep, diagnosed, jnp.minimum(diagnosed, zsoil[k]))
        wtd = jnp.where(partial & (changed > smceq[k]), diagnosed, wtd)
        layers[k] = jnp.where(partial, changed, jnp.where(active, smcmax, layers[k]))
        water = jnp.where(partial, zero, jnp.where(active, water - capacity, water))
    # The last K=0 iteration in WRF sets the water table to the surface.
    wtd = jnp.where(up & (resolved | fills_soil) & (water > 0), zero, wtd)
    spring = jnp.where(up, water, zero)

    # Falling table through the resolved layers.
    active_down = down & resolved
    for k in range(nsoil):
        active = active_down & (k >= start_layer)
        capacity = dzs[k] * (layers[k] - jnp.maximum(smceq[k], sice[k]))
        partial = active & (-water <= capacity)
        changed = layers[k] + water / dzs[k]
        diagnosed = (changed * dzs[k] - smceq[k] * zsoil[k]
                     + smcmax * zsoil[k + 1]) / jnp.where(
                         partial & (changed > smceq[k]), smcmax - smceq[k], 1.)
        wtd = jnp.where(partial, jnp.where(changed > smceq[k], diagnosed, zsoil[k + 1]),
                        jnp.where(active, zsoil[k + 1], wtd))
        # Pristine :522 uses PLUS maxwatdw, despite a falling water table.
        layers[k] = jnp.where(partial, changed,
                             jnp.where(active & (capacity >= 0), layers[k] + capacity / dzs[k], layers[k]))
        water = jnp.where(partial, zero,
                          jnp.where(active & (capacity >= 0), water + capacity, water))
        active_down = active_down & ~partial

    fictitious_down = down & ~deep & ((~resolved) | (active_down & (water < 0)))
    capacity = dzs[-1] * (smcwtd - eqdeep)
    partial = fictitious_down & (-water <= capacity)
    changed = smcwtd + water / dzs[-1]
    diagnosed = (changed * dzs[-1] - eqdeep * zsoil[nsoil]
                 + smcmax * (zsoil[nsoil] - dzs[-1])) / jnp.where(partial, smcmax - eqdeep, 1.)
    wtd = jnp.where(partial, jnp.maximum(diagnosed, zsoil[nsoil] - dzs[-1]), wtd)
    deeper = fictitious_down & ~partial
    dzup = (eqdeep - changed) * dzs[-1] / jnp.where(deeper, smcmax - eqdeep, 1.)
    wtd = jnp.where(deeper, zsoil[nsoil] - dzs[-1] - dzup, wtd)
    smcwtd = jnp.where(partial, changed, jnp.where(deeper, eqdeep, smcwtd))

    falling_deep = down & deep
    distance = jnp.where(falling_deep, zsoil[nsoil] - old_wtd, 1.)
    wgpmid = jnp.maximum(smcmax * (psisat / (psisat - distance)) ** (1. / bexp), 1.e-4)
    lowered = old_wtd + totwater / jnp.where(falling_deep, smcmax - wgpmid, 1.)
    changed = (smcwtd * (zsoil[nsoil] - old_wtd) + wgpmid * (old_wtd - lowered)) / jnp.where(
        falling_deep, zsoil[nsoil] - lowered, 1.)
    wtd = jnp.where(down & deep, lowered, wtd)
    smcwtd = jnp.where(down & deep, changed, smcwtd)
    smc = jnp.stack(layers)
    return wtd, smc, smc - sice, smcwtd, spring


def wtable(static, carry, smc, sh2o, smcwtd, dzs, soil_type,
           landmask, urbanmask, parameters, seconds):
    """Pristine WTABLE_mmf_noahmp; ``parameters`` uses WRF's junk slot zero.

    ``seconds`` is WTDDT*60, independent of DT*STEPWTD (WRF rounds STEPWTD).
    The eligibility predicate must match XLAND/XICE/IVGTYP in surface_driver.
    """
    dtype = smc.dtype
    soil = lambda name: jnp.asarray(getattr(parameters, name), dtype=dtype)[soil_type]
    bexp, dksat, psisat = soil('bexp'), soil('dksat'), -soil('psisat')
    smcmax = jnp.where(urbanmask, jnp.asarray(.45, dtype=dtype), soil('smcmax'))
    # WRF never enters these column expressions off land. Supply finite
    # inactive operands; every executed original land operand stays unchanged.
    bexp = jnp.where(landmask, bexp, 1.)
    psisat = jnp.where(landmask, psisat, -1.)
    smcmax = jnp.where(landmask, smcmax, 1.)
    zsoil = jnp.concatenate((jnp.zeros((1,), dtype=dtype), -jnp.cumsum(dzs)))
    qlat = lateral_flow(carry.zwt, static.fdepth, static.topo, static.area,
                        soil_type, landmask, jnp.asarray(parameters.dksat[1:], dtype=dtype), seconds)
    exponential = landmask & (carry.zwt > static.riverbed) & (static.eqzwt > static.riverbed)
    rcond = jnp.where(exponential,
                      static.rivercond * jnp.exp(jnp.where(exponential,
                          static.pexp * (carry.zwt - static.eqzwt), 0.)),
                      static.rivercond)
    qrf = rcond * (carry.zwt - static.riverbed) * seconds / static.area
    qrf = jnp.where(landmask, jnp.maximum(qrf, 0.), jnp.zeros_like(qrf))
    deep = landmask & (carry.zwt < zsoil[-1] - dzs[-1])
    ddz = jnp.where(deep, zsoil[-1] - carry.zwt, jnp.ones_like(carry.zwt))
    work_smcwtd = jnp.where(deep, smcwtd, jnp.ones_like(smcwtd))
    mid = .5 * (work_smcwtd + smcmax)
    psi = psisat * (smcmax / work_smcwtd) ** bexp
    wcnd = dksat * (mid / smcmax) ** (2. * bexp + 3.)
    flux = -seconds * wcnd * ((psisat - psi) / ddz - 1.)
    changed = work_smcwtd + (carry.deeprech - flux) / ddz
    wplus = jnp.maximum(changed - smcmax, 0.) * ddz
    wminus = jnp.maximum(1.e-4 - changed, 0.) * ddz
    changed = jnp.maximum(jnp.minimum(changed, smcmax), 1.e-4)
    deeprech = jnp.where(deep, flux + wplus - wminus, carry.deeprech)
    smcwtd = jnp.where(deep, changed, smcwtd)
    zwt, next_smc, next_sh2o, next_smcwtd, spring = update_wtd(
        dzs, zsoil, jnp.where(landmask[None], static.smoiseq, .5), smcmax, psisat, bexp,
        qlat - qrf + deeprech, carry.zwt, smc, sh2o, smcwtd)
    # The original caller's non-land slots survive the IF LANDMASK block.
    zwt = jnp.where(landmask, zwt, carry.zwt)
    smc = jnp.where(landmask[None], next_smc, smc)
    sh2o = jnp.where(landmask[None], next_sh2o, sh2o)
    smcwtd = jnp.where(landmask, next_smcwtd, smcwtd)
    spring = jnp.where(landmask, spring, carry.qspring)
    scale = jnp.asarray(1.e3, dtype=dtype)
    carry = GroundwaterCarry(zwt, jnp.zeros_like(deeprech), carry.rech + deeprech * scale,
                              qlat, carry.qslat + qlat * scale, qrf, carry.qrfs + qrf * scale,
                              spring, carry.qsprings + spring * scale)
    return carry, smc, sh2o, smcwtd


def _newton(initial, equation):
    """Original per-column 100-iteration Newton loop, exit ABS(DX)<1.e-6."""
    def body(_, state):
        value, done = state
        func, dfunc = equation(value)
        dx = func / dfunc
        return jnp.where(done, value, value - dx), done | (jnp.abs(dx) < 1.e-6)
    return lax.fori_loop(0, 100, body, (initial, jnp.zeros_like(initial, dtype=bool)))[0]


def equilibrium_moisture(zsoil, smcmax, bexp, dwsat, dksat):
    """Verbatim-equation EQSMOISTURE (Noah-MP driver :2758-2807)."""
    profiles = []
    for k in range(len(zsoil)):
        if k == 0:
            ddz = -zsoil[k + 1] * .5
        elif k < len(zsoil) - 1:
            ddz = (zsoil[k - 1] - zsoil[k + 1]) * .5
        else:
            ddz = zsoil[k - 1] - zsoil[k]
        expon = bexp + 1.
        aa = dwsat / ddz
        bb = dksat / smcmax ** expon
        def equation(smc):
            return ((smc - smcmax) * aa + bb * smc ** expon,
                    aa + bb * expon * smc ** bexp)
        value = _newton(.5 * smcmax, equation)
        profiles.append(jnp.minimum(jnp.maximum(value, 1.e-4), smcmax * .99))
    return jnp.stack(profiles)


def initialize(fdepth, area, topo, eqzwt, rechclim, riverbed, smc, sh2o,
               dzs, soil_type, veg_type, parameters, seconds):
    """Pristine serial GROUNDWATER_INIT (:2442-2755), after NOAHMP_INIT.

    Initialization's land mask is IVGTYP (not the runtime XLAND/XICE mask).
    The 500 global adjustment iterations are genuine spatial dependencies;
    no host round trips or per-cell Python decisions are required.
    """
    dtype = smc.dtype
    soil = lambda name: jnp.asarray(getattr(parameters, name), dtype=dtype)[soil_type]
    bexp, dksat, dwsat = soil('bexp'), soil('dksat'), soil('dwsat')
    psisat, smcmax = -soil('psisat'), soil('smcmax')
    urban = (veg_type == parameters.isurban) | ((veg_type >= 51) & (veg_type <= 61))
    smcmax = jnp.where(urban, jnp.asarray(.45, dtype=dtype), smcmax)
    land = (veg_type != parameters.iswater) & (veg_type != parameters.isice)
    zsoil = [-dzs[0]]
    for k in range(1, len(dzs)):
        zsoil.append(-dzs[k] + zsoil[-1])
    zsoil = jnp.stack(zsoil)
    year = jnp.asarray(365. * 24. * 3600., dtype=dtype)
    table = jnp.asarray(parameters.dksat[1:], dtype=dtype)
    def condition(state):
        iteration, _, _, changed = state
        return (iteration < 500) & ((iteration == 0) | changed)
    def adjust(state):
        iteration, wtd, _, _ = state
        qlat = lateral_flow(wtd, fdepth, topo, area, soil_type, land, table, year)
        active = land & (qlat > 1.e-2)
        wtd = jnp.where(active, jnp.minimum(wtd + .25, 0.), wtd)
        return iteration + 1, wtd, qlat, jnp.any(active)
    _, wtd, qlat_year, _ = lax.while_loop(condition, adjust,
        (jnp.asarray(0, dtype=jnp.int32), eqzwt, jnp.zeros_like(eqzwt), jnp.asarray(True)))
    eqzwt = wtd
    ddz = eqzwt - (riverbed - topo)
    riverbed = jnp.where(ddz < 0, topo + eqzwt, riverbed)
    ddz = jnp.where(ddz < 0, jnp.zeros_like(ddz), ddz)
    total = area * (qlat_year + rechclim * .001) / year
    rivercond = jnp.where(total > 0, total / jnp.maximum(ddz, .05), jnp.asarray(.01, dtype=dtype))
    riverbed = jnp.where(total > 0, riverbed, topo + eqzwt)
    riverbed = jnp.minimum(riverbed - topo, 0.)
    qlat = lateral_flow(wtd, fdepth, topo, area, soil_type, land, table, seconds)
    qrf = jnp.where(land, jnp.maximum(rivercond * (wtd - riverbed) * seconds / area, 0.), 0.)
    valid = (bexp > 0) & (smcmax > 0) & (-psisat > 0)
    # These are inactive-branch operands, never replacements for valid soil.
    work_b = jnp.where(valid, bexp, jnp.ones_like(bexp))
    work_max = jnp.where(valid, smcmax, jnp.ones_like(smcmax))
    work_psi = jnp.where(valid, psisat, -jnp.ones_like(psisat))
    work_dwsat = jnp.where(valid, dwsat, jnp.ones_like(dwsat))
    work_dksat = jnp.where(valid, dksat, jnp.ones_like(dksat))
    eq = equilibrium_moisture(zsoil, work_max, work_b, work_dwsat, work_dksat)
    deep = valid & (wtd < zsoil[-1] - dzs[-1])
    ddz = jnp.where(deep, zsoil[-1] - wtd, jnp.ones_like(wtd))
    cc = work_psi / ddz
    flux = (qlat - qrf) / seconds
    expon = 2. * work_b + 3.
    def equation(value):
        dd = (value + work_max) / (2. * work_max)
        aa = -work_dksat * dd ** expon
        bb = cc * ((work_max / value) ** work_b - 1.) + 1.
        func = aa * bb - flux
        dfunc = (-work_dksat * (expon / (2. * work_max)) * dd ** (expon - 1.) * bb
                 + aa * cc * (-work_b) * work_max ** work_b * value ** (-work_b - 1.))
        return func, dfunc
    deep_smc = jnp.maximum(_newton(.5 * work_max, equation), 1.e-4)
    eqdeep = jnp.maximum(work_max * (work_psi / (work_psi - dzs[-1])) ** (1. / work_b), 1.e-4)
    fictitious_smc = work_max * (wtd - (zsoil[-1] - dzs[-1])) + eqdeep * (zsoil[-1] - wtd)
    smcwtd = jnp.where(deep, deep_smc, jnp.where(wtd < zsoil[-1], fictitious_smc, work_max))
    layers, liquids = list(smc), list(sh2o)
    active = valid & (wtd >= zsoil[-1])
    for k in range(len(dzs) - 1, 0, -1):
        saturated = active & (wtd >= zsoil[k - 1])
        liquid_fraction = liquids[k] / layers[k]
        liquids[k] = jnp.where(saturated, work_max * liquid_fraction, liquids[k])
        layers[k] = jnp.where(saturated, work_max, layers[k])
        ending = active & ~saturated
        diagnosed = ((layers[k] * dzs[k] - eq[k] * zsoil[k - 1]
                      + work_max * zsoil[k]) / (work_max - eq[k]))
        wtd = jnp.where(ending, jnp.where(layers[k] < eq[k], zsoil[k], diagnosed), wtd)
        active = active & ~ending
    eq = jnp.where(valid[None], eq, smcmax[None])
    smcwtd = jnp.where(valid, smcwtd, smcmax)
    wtd = jnp.where(valid, wtd, jnp.zeros_like(wtd))
    zero = jnp.zeros_like(wtd)
    static = GroundwaterStatic(fdepth, area, topo, eqzwt, riverbed, rivercond, jnp.ones_like(wtd), eq, rechclim)
    carry = GroundwaterCarry(wtd, zero, zero, zero, zero, zero, zero, zero, zero)
    return static, carry, jnp.stack(layers), jnp.stack(liquids), smcwtd


def _pick(layers, index):
    """Column-local selection; inactive out-of-range operands are never used."""
    value = layers[0]
    for k in range(len(layers)):
        value = jnp.where(index == k, layers[k], value)
    return value


def shallow_wtd(smc, wtd, smcwtd, smceq, zsoil, dzs, smcmax, psisat, bexp):
    """Original SHALLOWWATERTABLE :9774-9909; return local recharge in m.

    The driver initializes its RECH scalar to zero before SFLX (:842). This
    also supplies zero when the original routine's deep-table branch writes
    no RECH despite its INTENT(OUT) declaration (E24).
    """
    nsoil = len(dzs)
    zero = jnp.zeros_like(wtd)
    zs = [zero] + list(zsoil)
    iwtd = jnp.zeros_like(wtd, dtype=jnp.int32)
    found = jnp.zeros_like(wtd, dtype=bool)
    for k in range(nsoil, 0, -1):
        match = ~found & (wtd + 1.e-6 < zs[k])
        iwtd = jnp.where(match, k, iwtd)
        found = found | match
    initial_index = iwtd  # zero-based soil layer, Fortran KWTD=IWTD+1
    inside = initial_index < nsoil
    level_smc, eq, maximum = (_pick(v, initial_index) for v in (smc, smceq, smcmax))
    excess = level_smc > eq
    saturated = level_smc == maximum
    old = wtd
    recharge = zero
    def diagnosed(index):
        eq0, max0 = _pick(smceq, index), _pick(smcmax, index)
        return jnp.minimum((_pick(smc, index) * _pick(dzs, index)
                            - eq0 * _pick(zs, index) + max0 * _pick(zs, index + 1))
                           / (max0 - eq0), _pick(zs, index))

    partial = inside & excess & ~saturated
    change = diagnosed(initial_index)
    wtd = jnp.where(partial, change, wtd)
    recharge = jnp.where(partial, -(old - change) * (maximum - eq), recharge)

    rising = inside & excess & saturated
    interface = _pick(zs, initial_index)
    wtd = jnp.where(rising, interface, wtd)
    recharge = jnp.where(rising, -(old - interface) * (maximum - eq), recharge)
    next_index = initial_index - 1
    iwtd = jnp.where(rising, next_index, iwtd)
    next_excess = _pick(smc, next_index) > _pick(smceq, next_index)
    adjusts = rising & (next_index >= 0) & next_excess
    change = diagnosed(next_index)
    recharge = jnp.where(adjusts, recharge - (interface - change)
                         * (_pick(smcmax, next_index) - _pick(smceq, next_index)), recharge)
    wtd = jnp.where(adjusts, change, wtd)

    falling = inside & ~excess
    interface = _pick(zs, initial_index + 1)
    wtd = jnp.where(falling, interface, wtd)
    recharge = jnp.where(falling, -(old - interface) * (maximum - eq), recharge)
    next_index = initial_index + 1
    iwtd = jnp.where(falling, next_index, iwtd)
    adjusts = falling & (next_index < nsoil)
    change = jnp.where(_pick(smc, next_index) > _pick(smceq, next_index),
                        diagnosed(next_index), _pick(zs, next_index + 1))
    recharge = jnp.where(adjusts, recharge - (interface - change)
                         * (_pick(smcmax, next_index) - _pick(smceq, next_index)), recharge)
    wtd = jnp.where(adjusts, change, wtd)

    eqdeep = smcmax[-1] * (-psisat[-1] / (-psisat[-1] - dzs[-1])) ** (1. / bexp[-1])
    change = jnp.minimum((smcwtd * dzs[-1] - eqdeep * zs[-1]
                          + smcmax[-1] * (zs[-1] - dzs[-1])) / (smcmax[-1] - eqdeep), zs[-1])
    crosses_bottom = falling & (next_index >= nsoil)
    recharge = jnp.where(crosses_bottom, recharge - (interface - change)
                         * (smcmax[-1] - eqdeep), recharge)
    wtd = jnp.where(crosses_bottom, change, wtd)

    below = ~inside & (old >= zs[-1] - dzs[-1])
    high = below & (smcwtd > eqdeep)
    recharge = jnp.where(high, -(old - change) * (smcmax[-1] - eqdeep), recharge)
    wtd = jnp.where(high, change, wtd)
    low = below & ~high
    base = zs[-1] - dzs[-1]
    dzup = (eqdeep - smcwtd) * dzs[-1] / (smcmax[-1] - eqdeep)
    local_recharge = -(old - base) * (smcmax[-1] - eqdeep)
    local_recharge = local_recharge - (smcmax[-1] - eqdeep) * dzup
    recharge = jnp.where(low, local_recharge, recharge)
    wtd = jnp.where(low, base - dzup, wtd)
    smcwtd = jnp.where(low, eqdeep, smcwtd)
    smcwtd = jnp.where((iwtd < nsoil) & (iwtd > 0), _pick(smcmax, iwtd - 1),
                       jnp.where((iwtd < nsoil) & (iwtd <= 0), smcmax[0], smcwtd))
    return wtd, smcwtd, recharge


def _srt5(zsoil, pddum, etrani, qseva, sh2o, smc, wtd, smcwtd, fcr, sicemax,
          bexp, smcmax, dksat, dwsat, *, opt_inf):
    from gpuwrf.physics.noahmp.water_hydro import _wdfcnd1, _wdfcnd2
    if opt_inf == 1:
        pairs = [_wdfcnd1(smc[k], fcr[k], bexp[k], smcmax[k], dksat[k], dwsat[k]) for k in range(4)]
        smx, smxwtd = smc, smcwtd
    elif opt_inf == 2:
        pairs = [_wdfcnd2(sh2o[k], sicemax, bexp[k], smcmax[k], dksat[k], dwsat[k]) for k in range(4)]
        smx, smxwtd = sh2o, smcwtd * sh2o[-1] / smc[-1]
    else:
        raise ValueError('MMF requires opt_inf=1 or 2')
    wdf, wcnd = [p[0] for p in pairs], [p[1] for p in pairs]
    denom, ddz, dsmdz, flux = [], [], [], []
    for k in range(4):
        denom.append(-zsoil[k] if k == 0 else zsoil[k - 1] - zsoil[k])
        if k < 3:
            temp = -zsoil[1] if k == 0 else zsoil[k - 1] - zsoil[k + 1]
            ddz.append(2. / temp)
            dsmdz.append(2. * (smx[k] - smx[k + 1]) / temp)
            if k == 0:
                flux.append(wdf[k] * dsmdz[k] + wcnd[k] - pddum + etrani[k] + qseva)
            else:
                flux.append(wdf[k] * dsmdz[k] + wcnd[k] - wdf[k - 1] * dsmdz[k - 1]
                            - wcnd[k - 1] + etrani[k])
        else:
            temp = 2. * denom[k]
            below = smx[k] - (smx[k] - smxwtd) * denom[k] * 2. / (denom[k] + zsoil[k] - wtd)
            below = jnp.where(wtd < zsoil[k] - denom[k], below, smxwtd)
            dsmdz.append(2. * (smx[k] - below) / temp)
            qdrain = wdf[k] * dsmdz[k] + wcnd[k]
            flux.append(-(wdf[k - 1] * dsmdz[k - 1]) - wcnd[k - 1] + etrani[k] + qdrain)
    ai, bi, ci, rhs = [], [], [], []
    for k in range(4):
        a = jnp.zeros_like(pddum) if k == 0 else -wdf[k - 1] * ddz[k - 1] / denom[k]
        if k == 0:
            b = wdf[k] * ddz[k] / denom[k]
            c = -b
        else:
            c = -wdf[k] * ddz[k] / denom[k] if k < 3 else jnp.zeros_like(pddum)
            b = -(a + c)
        ai.append(a); bi.append(b); ci.append(c); rhs.append(flux[k] / (-denom[k]))
    return rhs, ai, bi, ci, qdrain


def _rosr12_4(a, b, c, d):
    """Original ROSR12 :5565-5587, including separately rounded reciprocals."""
    p = [-c[0] / b[0]]
    delta = [d[0] / b[0]]
    for k in range(1, 4):
        denominator = b[k] + a[k] * p[k - 1]
        p.append(-c[k] * (1. / denominator))
        delta.append((d[k] - a[k] * delta[k - 1]) * (1. / denominator))
    solution = [None, None, None, delta[3]]
    for k in range(2, -1, -1):
        solution[k] = p[k] * solution[k + 1] + delta[k]
    return solution


def _sstep5(dt, zsoil, dzs, sice, sh2o, smcwtd, wtd, deeprech, ai, bi, ci, rhs, smcmax, qdrain):
    a = [v * dt for v in ai]
    b = [1. + v * dt for v in bi]
    c = [v * dt for v in ci]
    d = [v * dt for v in rhs]
    solution = _rosr12_4(a, b, c, d)
    sh2o = [sh2o[k] + solution[k] for k in range(4)]
    deep = wtd < zsoil[-1] - dzs[-1]
    deeprech = jnp.where(deep, deeprech + dt * qdrain, deeprech)
    changed = smcwtd + dt * qdrain / dzs[-1]
    wplus = jnp.maximum(changed - smcmax[-1], 0.) * dzs[-1]
    wminus = jnp.maximum(1.e-4 - changed, 0.) * dzs[-1]
    smcwtd = jnp.where(deep, smcwtd, jnp.maximum(jnp.minimum(changed, smcmax[-1]), 1.e-4))
    sh2o[-1] = jnp.where(deep, sh2o[-1], sh2o[-1] + wplus / dzs[-1])
    qdrain = jnp.where(deep, qdrain, qdrain - wplus / dt)
    deeprech = jnp.where(deep, deeprech, deeprech - wminus)
    for k in range(3, 0, -1):
        epore = jnp.maximum(1.e-4, smcmax[k] - sice[k])
        overflow = jnp.maximum(sh2o[k] - epore, 0.) * dzs[k]
        sh2o[k] = jnp.minimum(epore, sh2o[k])
        sh2o[k - 1] = sh2o[k - 1] + overflow / dzs[k - 1]
    epore = jnp.maximum(1.e-4, smcmax[0] - sice[0])
    wplus = jnp.maximum(sh2o[0] - epore, 0.) * dzs[0]
    sh2o[0] = jnp.minimum(epore, sh2o[0])
    down = list(sh2o)
    down[1] = down[1] + wplus / dzs[1]
    overflow = wplus
    for k in range(1, 4):
        epore = jnp.maximum(1.e-4, smcmax[k] - sice[k])
        overflow = jnp.maximum(down[k] - epore, 0.) * dzs[k]
        down[k] = jnp.minimum(epore, down[k])
        if k < 3:
            down[k + 1] = down[k + 1] + overflow / dzs[k + 1]
    active = wplus > 0
    sh2o = [jnp.where(active, down[k], sh2o[k]) for k in range(4)]
    wplus = jnp.where(active, overflow, wplus)
    return sh2o, [sh2o[k] + sice[k] for k in range(4)], smcwtd, deeprech, qdrain, wplus


def _soilwater5_core(qinsur, qseva, etrani, sh2o, smc, sice, zsoil, dzs, smcwtd,
                     wtd, smceq, bexp, smcmax, dksat, dwsat, psisat, fsatmx, dt, urban,
                     *, opt_inf=1):
    """SOILWATER + WATER's SHALLOWWATERTABLE seam, one column-local kernel."""
    zero = jnp.zeros_like(qinsur)
    rsat = zero
    sh2o = list(sh2o)
    for k in range(4):
        epore = jnp.maximum(1.e-4, smcmax[k] - sice[k])
        rsat = rsat + jnp.maximum(0., sh2o[k] - epore) * dzs[k]
        sh2o[k] = jnp.minimum(epore, sh2o[k])
    exp4 = jnp.exp(jnp.asarray(-4., dtype=qinsur.dtype))
    fcr, sicemax = [], zero
    for k in range(4):
        fice = jnp.minimum(1., sice[k] / smcmax[k])
        fcr.append(jnp.maximum(0., jnp.exp(-4. * (1. - fice)) - exp4) / (1. - exp4))
        sicemax = jnp.maximum(sicemax, sice[k])
    fcr[0] = jnp.where(urban, jnp.asarray(.95, dtype=qinsur.dtype), fcr[0])
    fsat = fsatmx * jnp.exp(-.5 * 6. * jnp.maximum(-2. - wtd, 0.))
    runsrf = jnp.where(qinsur > 0., qinsur * ((1. - fcr[0]) * fsat + fcr[0]), zero)
    pddum = jnp.where(qinsur > 0., qinsur - runsrf, zero)
    niter = jnp.where(pddum * dt > dzs[0] * smcmax[0],
                       jnp.asarray(6., dtype=qinsur.dtype), jnp.asarray(3., dtype=qinsur.dtype))
    fine = dt / niter
    qdrain_save, runsrf_save, deeprech = zero, zero, zero
    for iteration in range(6):
        rhs, ai, bi, ci, qdrain = _srt5(zsoil, pddum, etrani, qseva, sh2o, smc, wtd,
            smcwtd, fcr, sicemax, bexp, smcmax, dksat, dwsat, opt_inf=opt_inf)
        newsh, newsm, newaux, newdeep, qdrain, wplus = _sstep5(fine, zsoil, dzs, sice, sh2o,
            smcwtd, wtd, deeprech, ai, bi, ci, rhs, smcmax, qdrain)
        active = iteration < niter
        sh2o = [jnp.where(active, newsh[k], sh2o[k]) for k in range(4)]
        smc = [jnp.where(active, newsm[k], smc[k]) for k in range(4)]
        smcwtd = jnp.where(active, newaux, smcwtd)
        deeprech = jnp.where(active, newdeep, deeprech)
        rsat = rsat + jnp.where(active, wplus, zero)
        qdrain_save = qdrain_save + jnp.where(active, qdrain, zero)
        runsrf_save = runsrf_save + jnp.where(active, runsrf, zero)
    qdrain = qdrain_save / niter
    runsrf = runsrf_save / niter
    runsrf = runsrf * 1000. + rsat * 1000. / dt
    qdrain = qdrain * 1000.
    mliq = [sh2o[k] * dzs[k] * 1000. for k in range(4)]
    for k in range(3):
        xs = jnp.where(mliq[k] < 0., .01 - mliq[k], zero)
        mliq[k] = mliq[k] + xs
        mliq[k + 1] = mliq[k + 1] - xs
    xs = jnp.where(mliq[-1] < .01, .01 - mliq[-1], zero)
    mliq[-1] = mliq[-1] + xs
    runsub = zero - xs / dt
    deeprech = deeprech - xs * .001
    sh2o = [mliq[k] / (dzs[k] * 1000.) for k in range(4)]
    smc = [sh2o[k] + sice[k] for k in range(4)]
    wtd, smcwtd, recharge = shallow_wtd(smc, wtd, smcwtd, smceq, zsoil, dzs, smcmax, psisat, bexp)
    sh2o[-1] = smc[-1] - sice[-1]
    return sh2o, smc, smcwtd, wtd, deeprech, recharge, runsrf, qdrain, runsub


def soilwater5(qinsur, qseva, etrani, sh2o, smc, sice, zsoil, dzs, smcwtd,
               carry, smceq, bexp, smcmax, dksat, dwsat, psisat, fsatmx, dt, urban,
               *, opt_inf=1):
    from functools import partial
    from gpuwrf.kernels.phys_noahmp_columns import column_call, columns_enabled
    fields = (list(etrani), list(sh2o), list(smc), list(sice), list(zsoil), list(dzs))
    args = (qinsur, qseva, *fields, smcwtd, carry.zwt, list(smceq), list(bexp), list(smcmax),
            list(dksat), list(dwsat), list(psisat), fsatmx, dt, urban)
    core = partial(_soilwater5_core, opt_inf=opt_inf)
    if columns_enabled():
        result = column_call(core, args, grid_shape=jnp.shape(qinsur), name='b_core_noah_mmf_soilwater')
    else:
        result = core(*args)
    sh2o, smc, smcwtd, wtd, deep, recharge, runsrf, qdrain, runsub = result
    # Driver :842-843 zeros LOCAL RECH/DEEPRECH; :1366-1367 adds their deltas
    # to the persistent grid arrays only after NOAHMP_SFLX returns.
    updated = carry._replace(zwt=wtd, deeprech=carry.deeprech + deep,
                              rech=carry.rech + recharge * 1000.)
    return jnp.stack(sh2o), jnp.stack(smc), runsrf, qdrain, runsub, smcwtd, updated


def advance_lateral(bundle, land_state, static, itimestep):
    """surface_driver:3345 cadence, after the column solve, no host transfer."""
    def call(data):
        current, land = data
        state, smc, sh2o, aux = wtable(current.static, current.carry, land.smois,
            land.sh2o, land.smcwtd, static.dzs, static.isltyp, current.eligibility,
            static.ivgtyp == static.parameters.isurban, static.parameters, current.seconds)
        return current._replace(carry=state), land.replace(smois=smc, sh2o=sh2o, smcwtd=aux)
    active = jnp.mod(jnp.asarray(itimestep, jnp.int32), bundle.stepwtd) == 0
    return lax.cond(active, call, lambda data: data, (bundle, land_state))
