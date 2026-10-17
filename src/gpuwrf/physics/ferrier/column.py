"""Literal masked Ferrier-Aligo EGCP01COLUMN calculations.

This module is a column implementation in progress, not an admitted scheme.
Variable names follow phys/module_mp_fer_hires.F. Masks preserve Fortran
branch ownership while allowing adjacent independent columns in one tile.
"""
from jax import lax
import jax.numpy as jnp
import numpy as np

from .saturation import F, EPS, EPSQ, RD, RV, XLV, XLF, XLS, XLV1, XLS1, condense, deposit, fpvs, host_tables


CP = F(1004.6)
T0C = F(273.15)
EPS1 = F(F(RV / RD) - F(1))
XLF1 = F(XLF / CP)
XLV2, XLS2 = F(F(XLV * XLV) / RV), F(F(XLS * XLS) / RV)
RCPRV = F(F(F(1) / CP) / RV)
XLS3 = F(F(XLS * XLS) * RCPRV)
C1, C2 = F(1 / 3), F(1 / 6)
PI = F(3.141592653589793)
NCW, N0R0 = F(250e6), F(8e6)
CLIMIT, TOLER = F(F(10) * EPSQ), F(5e-7)
MY600 = np.array([5.5e-8, 1.4e-7, 2.8e-7, 6e-7, 3.3e-6, 2e-6, 9e-7,
                 8.8e-7, 8.2e-7, 9.4e-7, 1.2e-6, 1.85e-6, 5.5e-6, 1.5e-5,
                 1.7e-5, 1.5e-5, 1e-5, 3.4e-6, 1.85e-6, 1.35e-6, 1.05e-6,
                 1e-6, 9.5e-7, 9e-7, 9.5e-7, 9.5e-7, 9e-7, 9e-7, 9e-7,
                 9e-7, 9e-7, 9e-7, 9e-7, 9e-7, 9e-7], dtype=np.float32)


def _table(name, index):
    return jnp.asarray(host_tables()[name], jnp.float32)[jnp.asarray(index, jnp.int32) - 50]


def _vel_rime(ixs, ixrf):
    return jnp.asarray(host_tables()["vel_rf"], jnp.float32)[ixs - 2, ixrf]


def _growth(index, dt):
    return (dt / F(600)) ** F(1.5) * jnp.asarray(MY600)[index - 1] * F(1e-3)


def _rain_dsd(rqr, drzl, strat, minimum):
    small = rqr <= N0R0 * _table("massr", 50)
    large = rqr >= N0R0 * _table("massr", 1000)
    dum = jnp.maximum(F(1), F(.5e-3) / rqr)
    drizzle_n0 = jnp.minimum(F(1e9), N0R0 * dum * jnp.sqrt(dum))
    drizzle_size = F(1e6) * jnp.sqrt(jnp.sqrt(rqr / (PI * F(1000) * drizzle_n0)))
    cn0r0 = F(F(1e6) / F(np.sqrt(F(np.sqrt(F(F(PI * F(1000)) * N0R0))))))
    ordinary_size = cn0r0 * jnp.sqrt(jnp.sqrt(rqr))
    size = jnp.clip(jnp.where(drzl, drizzle_size, ordinary_size), F(50), F(1000)).astype(jnp.int32)
    forced = ~drzl & ~large & strat & (size < minimum)
    size = jnp.where(forced, minimum, size)
    size = jnp.where(small, 50, jnp.where(~drzl & large, 1000, size))
    variable_n0 = small | drzl | large | forced
    return size, jnp.where(variable_n0, rqr / _table("massr", size), N0R0)


class _Work:
    """Trace-time SSA locals; this object never exists on the device."""
    def __init__(self, **fields):
        self.values = fields

    def __getattr__(self, name):
        return self.values.get(name, F(0))

    def put(self, mask, **fields):
        for name, value in fields.items():
            self.values[name] = jnp.where(mask, value, self.values.get(name, jnp.zeros_like(value)))


def level(p, q, wc, qw, qi, qr, rime, temp, thick, rh, dt, flux):
    """One top-down level; flux is the incoming level's column-local carry.

    q is specific humidity; WC and species are condensate mixing ratios;
    thick is density times layer height. flux = ARAIN, ASNOW, VRAIN1,
    VSNOW1, INDEXS0C, RHOX0C, TIMLT, STRAT, DRZL, INDEXRmin, previous RimeF.
    """
    arain, asnow, vrain1, vsnow1, indexs0c, rhox0c, timlt, strat, drzl, indexrmin, prev_rime = flux
    has_rain, has_snow = arain > CLIMIT, asnow > CLIMIT
    arain = jnp.where(has_rain, arain, F(0))
    asnow = jnp.where(has_snow, asnow, F(0))
    # Literal source resets the incoming speeds when there is incoming flux.
    vrain1, vsnow1 = jnp.where(has_rain, F(0), vrain1), jnp.where(has_snow, F(0), vsnow1)
    tc = temp - T0C
    wv = q / (F(1) - q)
    esw = jnp.minimum(F(1000) * fpvs(temp, water=True), F(.99) * p)
    qsw = EPS * esw / (p - esw)
    esi = jnp.minimum(F(1000) * fpvs(temp, water=False), F(.99) * p)
    qsi = jnp.where(tc < F(0), EPS * esi / (p - esi), qsw)
    active = has_rain | has_snow | (wv > rh * qsi) | (wc > EPSQ)
    rho = p / (RD * temp * (F(1) + EPS1 * q))
    rrho = F(1) / rho
    dtrho = dt * rho
    tk2 = F(1) / (temp * temp)
    tfactor = jnp.sqrt(temp * temp * temp) / (temp + F(120))
    dynvis = F(1.496e-6) * tfactor
    therm = F(2.116e-3) * tfactor
    diffus = F(8.794e-5) * temp ** F(1.81) / p
    gammas = jnp.minimum(F(1.5), (F(1e5) / p) ** C1)
    gammar = (F(1.194) / rho) ** F(.4)
    ice = active & ((tc < F(0)) | (qi > EPSQ) | has_snow)
    rain = active & (has_rain | (qr > EPSQ))
    s = _Work(indexs=jnp.clip((F(1000) * jnp.exp(F(.0536) * jnp.minimum(F(0), tc))).astype(jnp.int32), 50, 1000),
              indexr=jnp.full_like(p, 50, dtype=jnp.int32), rimef1=jnp.ones_like(p),
              flimass=jnp.ones_like(p), strat=strat, drzl=drzl, timlt=timlt,
              indexs0c=indexs0c, rhox0c=rhox0c, indexrmin=indexrmin)
    cold = ice & (tc < F(0))
    nsimax = jnp.maximum(F(250e3), F(.1) * rho * qi / _table("massi", 50))
    s.put(cold, ninuclei=jnp.minimum(F(5) * jnp.exp(F(-.304) * tc),
                                   jnp.minimum(F(.01) * jnp.exp(F(-.6) * tc), nsimax)))
    s.put(cold & (qi > EPSQ), nsmice=jnp.minimum(s.ninuclei, qi / (rrho * _table("massi", 50))))
    s.put(cold & (qi > EPSQ), qsmice=s.nsmice * (rrho * _table("massi", 50)))
    existing = ice & ((qi > EPSQ) | has_snow)
    s.put(existing, tot_ice=thick * qi + asnow)
    s.put(existing, piloss=-s.tot_ice / thick, qlgice=jnp.maximum(F(0), qi - s.qsmice),
          vci=gammas * _table("vsnowi", 50))
    s.put(existing, rimef1=(rime * thick * s.qlgice + prev_rime * asnow) / s.tot_ice)
    # Convective / transition / stratiform particle sizes, source 1518–1542.
    rh_ngc = F(F(PI * F(500)) * F(1e3))
    rh_ngt = F(F(PI * F(300)) * F(5e3))
    conv_size = F(1e6) * F(F(1) / rh_ngc) ** C1 * (rho * s.qlgice) ** C1
    trans_size = F(1e6) * F(F(1) / rh_ngt) ** C1 * (rho * s.qlgice) ** C1
    blend5 = F(.2) * (s.rimef1 - F(5))
    blend2 = F(.33333) * (s.rimef1 - F(2))
    size = jnp.where(s.rimef1 >= F(10), conv_size,
                     jnp.where(s.rimef1 >= F(5), conv_size * blend5 + trans_size * (F(1) - blend5),
                               trans_size * blend2 + s.indexs.astype(jnp.float32) * (F(1) - blend2)))
    s.put(existing & (s.rimef1 > F(2)), indexs=jnp.clip(size.astype(jnp.int32), 50, 1000))
    s.put(existing, qlice=(thick * s.qlgice + asnow) / (thick + dtrho * vsnow1))
    s.put(existing, rqlice=rho * s.qlice, qtice=s.qlice + s.qsmice)
    s.put(existing, flimass=s.qlice / s.qtice)
    running = existing
    for ipass in range(1, 3):
        diameter = F(1e-6) * s.indexs.astype(jnp.float32)
        rfmx = F(PI * F(900)) * diameter * diameter * diameter / _table("massi", s.indexs)
        s.put(running, rimef1=jnp.minimum(s.rimef1, rfmx))
        s.put(running, rimef1=jnp.clip(s.rimef1, F(1), F(F(1.1) ** 40)))
        ixs = jnp.clip(s.indexs // 100, 2, 9)
        xrf = F(10.492) * jnp.log(s.rimef1)
        ixrf = jnp.clip(xrf.astype(jnp.int32), 0, 40)
        vr = _vel_rime(ixs, ixrf) + (xrf - ixrf.astype(jnp.float32)) * (_vel_rime(ixs, jnp.minimum(ixrf + 1, 40)) - _vel_rime(ixs, ixrf))
        s.put(running, vel_inc=gammas * jnp.maximum(F(1), vr))
        s.put(running, vsnow=s.vel_inc * _table("vsnowi", s.indexs))
        bvhail = F(F(.82) * C1)
        avhail = F(F(1353) * F(F(1) / rh_ngc) ** bvhail)
        hail_speed = gammas * avhail * s.rqlice ** bvhail
        h = running & (s.rimef1 >= F(5)) & (s.indexs == 1000) & (s.rqlice > F(rh_ngc * F(F(1e-3) ** 3))) & (hail_speed > s.vsnow)
        s.put(h, vsnow=hail_speed)
        s.put(h, vel_inc=s.vsnow / _table("vsnowi", s.indexs))
        s.put(running, nlice=s.rqlice / (s.rimef1 * _table("massi", s.indexs)))
        running &= ~((s.nlice >= F(1e3)) & (s.nlice <= F(250e3))) & (ipass < 2)
        s.put(running, nlice=jnp.clip(s.nlice, F(1e3), F(250e3)))
        xli = s.rqlice / (s.nlice * s.rimef1)
        size = jnp.where(xli <= _table("massi", 50), F(50),
                         jnp.where(xli <= _table("massi", 450), F(9.5885e5) * xli ** F(.42066),
                                   jnp.where(xli < _table("massi", 1000), F(3.9751e6) * xli ** F(.49870), F(1000))))
        s.put(running, indexs=jnp.clip(size.astype(jnp.int32), 50, 1000))
    # The no-existing-ice branch still initializes the ventilation multiplier.
    s.put(ice & ~existing, vel_inc=gammas)
    s.put(active & (tc <= F(0)), strat=False, indexrmin=50, timlt=F(0), indexs0c=s.indexs,
          rhox0c=F(22.5) * jnp.clip(s.rimef1, F(1), F(40)))
    warm = active & (tc > F(0))
    s.put(warm & ~rain, strat=False, indexrmin=50)
    s.put(warm & ~rain & ~ice, timlt=F(0))
    s.put(warm & rain, strat=(s.timlt > EPSQ) & (s.rhox0c <= F(225)))
    s.put(warm & rain & ~s.strat, indexrmin=50)
    s.put(warm & rain & s.strat & (s.indexrmin <= 50),
          indexrmin=jnp.clip((s.indexs0c * (F(.001) * s.rhox0c) ** C1).astype(jnp.int32), 50, 500))
    s.put(active, drzl=~(s.strat | (s.timlt > EPSQ)))
    # Source 1664–1715: cloud autoconversion and collection by ice.
    rdis = F(.5)
    beta6 = F(F(F(F(1) + F(3) * rdis * rdis) * F(F(1) + F(4) * rdis * rdis)) * F(F(1) + F(5) * rdis * rdis) /
              F(F(F(1) + rdis * rdis) * F(F(1) + F(2) * rdis * rdis)))
    araut = F(F(1.03e19) / F(NCW * F(np.sqrt(NCW))))
    braut = dt * F(1.1e10) * beta6 / NCW
    qaut0 = F(F(F(F(PI * F(1000)) * NCW) * F(F(20e-6) ** 3)) / F(6))
    dum2 = rho * qw
    auto = active & (qw > EPSQ) & (tc >= F(-40)) & (dum2 > qaut0)
    dum2sq = dum2 * dum2
    dum1 = araut * dum2sq
    s.put(auto, praut=jnp.minimum(qw, braut * dum2sq * qw * (F(1) - jnp.exp(-dum1 * dum1))))
    cloud_collect = active & (qw > EPSQ) & (tc >= F(-40)) & (s.qlice > EPSQ)
    ciacw = dt * F(.25) * PI * F(.5) * F(F(1e5) ** C1)
    s.put(cloud_collect, piacw=jnp.minimum(F(1), ciacw * s.vel_inc * s.nlice * _table("accri", s.indexs)) * qw)
    s.put(cloud_collect & (tc < F(0)), piacwi=s.piacw)
    # In WRF DUM2 was squared by the autoconversion branch before this use.
    arcw = F(F(1e6) * F(F(.75) / F(F(PI * NCW) * F(1000))) ** C1)
    s.put(cloud_collect & (tc < F(0)), rcw=arcw * jnp.where(auto, dum2sq, dum2) ** C1)
    homogeneous = active & (tc < F(-40)) & ((wv > rh * qsw) | (qw > EPSQ))
    s.put(homogeneous, piacw=qw, piacwi=qw, rcw=F(0))
    dummy_t = temp + XLF1 * s.piacw
    dummy_es = jnp.minimum(F(1000) * fpvs(dummy_t, water=False), F(.99) * p)
    dummy_ws = rh * EPS * dummy_es / (p - dummy_es)
    deposit_active = homogeneous & (wv > dummy_ws)
    s.put(deposit_active, pidep=deposit(p, dummy_t, wv, rh, active=deposit_active)[0])
    deposition = active & ~homogeneous & (tc < F(0))
    denomi = F(1) + XLS3 * qsi * tk2
    dwvi = wv - rh * qsi
    pidep_max = jnp.maximum(s.piloss, dwvi / denomi)
    abi = F(1) / (rho * XLS2 * qsi * tk2 / therm + F(1) / diffus)
    factor = (rho / (diffus * diffus * dynvis)) ** C2
    ventis = (_table("venti1", 50) + jnp.sqrt(gammas) * factor * _table("venti2", 50)) * s.nsmice
    ventil = (_table("venti1", s.indexs) + jnp.sqrt(s.vel_inc) * factor * _table("venti2", s.indexs)) * s.nlice
    s.put(deposition & (s.qtice > F(0)), didep=abi * (ventil + ventis) * dt)
    im = jnp.clip((F(.5) - tc).astype(jnp.int32), 1, 35)
    growth = _growth(im, dt)
    s.put(deposition & (wv > rh * qsw) & (tc < F(-12)) & (s.nsmice < s.ninuclei),
          pinit=jnp.maximum(F(0), jnp.maximum(F(0), s.ninuclei - s.nsmice) * growth * rrho))
    s.put(deposition & (dwvi > F(0)), pidep=jnp.minimum(dwvi * s.didep + s.pinit, pidep_max))
    s.put(deposition & (dwvi < F(0)), pidep=jnp.maximum(dwvi * s.didep, pidep_max))
    condense_active = active & (tc >= F(-40)) & ((qw > EPSQ) | (wv > rh * qsw))
    s.put(condense_active,
          pcond=condense(p, qw - s.piacwi, temp + XLS1 * s.pidep + XLF1 * s.piacwi, wv - s.pidep, rh,
                         active=condense_active))
    tcc = tc + XLV1 * s.pcond + XLS1 * s.pidep + XLF1 * s.piacwi
    s.put(active & (tcc > F(0)), piacwi=F(0), rcw=F(0), pidep=F(0))
    tcc = jnp.where(tcc > F(0), tc + XLV1 * s.pcond, tcc)
    # Source 1842–1894: melting and evaporation of wet precipitation ice.
    melting = active & (tc > F(0)) & (tcc > F(0)) & ice
    ventil = s.nlice * (_table("venti1", s.indexs) + jnp.sqrt(s.vel_inc) * factor * _table("venti2", s.indexs))
    aievp = ventil * diffus * dt
    dievp = jnp.where(aievp < F(.025), aievp, F(1) - jnp.exp(-aievp))
    esw0 = F(1000) * fpvs(jnp.float32(T0C), water=True)
    qsw0 = EPS * esw0 / (p - esw0)
    dwv0 = jnp.minimum(wv, qsw) - qsw0
    evaporate = melting & (wv < qsw) & (qw + s.pcond <= EPSQ)
    s.put(evaporate, pievp=jnp.maximum(jnp.minimum(F(0), dwv0 * dievp), s.piloss),
          picnd=jnp.maximum(F(0), dwv0 * dievp))
    s.put(melting, pimlt=therm * tcc * ventil * rrho * dt / XLF)
    s.put(melting, pimlt=jnp.minimum(s.pimlt, jnp.maximum(F(0), (tcc + XLV1 * s.pievp) / XLF1)))
    scale = s.piloss / (s.pievp - s.pimlt)
    s.put(melting & (s.pievp - s.pimlt < s.piloss), pimlt=s.pimlt * scale, pievp=s.pievp * scale)
    # Source 1926–2083: rain, evaporation, accretion and freezing.
    s.put(rain, tot_rain=thick * qr + arain)
    s.put(rain, prloss=-s.tot_rain / thick, qtrain=s.tot_rain / (thick + dtrho * vrain1))
    s.put(rain, rqr=rho * s.qtrain)
    s.put(rain & s.strat & (s.rqr > F(1e-3)), strat=False)
    s.put(rain & s.drzl & (s.pimlt > EPSQ), drzl=False)
    size, n0 = _rain_dsd(s.rqr, s.drzl, s.strat, s.indexrmin)
    s.put(rain, indexr=size, n0r=n0)
    s.put(rain, vrain2=gammar * _table("vrain", s.indexr))
    s.put(rain & (tc < F(-40)), piacr=-s.prloss)
    not_frozen = rain & (tc >= F(-40))
    dwvr = wv - s.pcond - qsw
    evaporation = not_frozen & (dwvr < F(0)) & (qw + s.pcond <= EPSQ)
    rfactor = jnp.sqrt(gammar) * factor
    abw = F(1) / (rho * XLV2 * qsw * tk2 / therm + F(1) / diffus)
    ventr = s.n0r * (_table("ventr1", s.indexr) + rfactor * _table("ventr2", s.indexr))
    s.put(evaporation, prevp=jnp.maximum(dwvr * (abw * ventr * dt), s.prloss))
    cracw = dt * F(.25) * PI * F(1)
    s.put(not_frozen & ~evaporation & (qw > EPSQ),
          pracw=jnp.minimum(F(1), cracw * gammar * s.n0r * _table("accrr", s.indexr)) * qw)
    freezing = not_frozen & ice & (tc < F(0)) & (tcc < F(0))
    dum = F(.001) * s.indexr.astype(jnp.float32)
    bigg = (jnp.exp(F(-.66) * tc) - F(1)) * dum * dum * dum * dum * dum * dum * dum
    cbfr = F(20) * PI * PI * F(100) * F(1000) * F(1e-21) * dt
    s.put(freezing, piacr=jnp.minimum(cbfr * s.n0r * rrho * bigg, s.qtrain))
    velocity = gammar * _table("vrain", s.indexr)
    delta_velocity = velocity - s.vsnow
    collision_speed = jnp.sqrt(delta_velocity * delta_velocity + F(.04) * velocity * s.vsnow)
    area = F(5e-12) * s.indexr * s.indexr + F(2e-12) * s.indexr * s.indexs + F(.5e-12) * s.indexs * s.indexs
    fir = jnp.minimum(F(1), (PI * dt) * s.nlice * area * collision_speed)
    s.put(freezing & (s.qlice > EPSQ), piacr=jnp.minimum(s.piacr + fir * s.qtrain, s.qtrain))
    scale = s.prloss / (s.prevp - s.piacr)
    s.put(freezing & (s.prevp - s.piacr < s.prloss), prevp=scale * s.prevp, piacr=scale * s.piacr)
    indexr1 = s.indexr
    # Source 2094 onward: all process rates are applied together.
    loss = s.piacw + s.praut + s.pracw - jnp.minimum(F(0), s.pcond)
    scale = qw / loss
    limit_cloud = active & (loss > qw)
    s.put(limit_cloud, piacw=scale * s.piacw, piacwi=scale * s.piacwi,
          praut=scale * s.praut, pracw=scale * s.pracw)
    s.put(limit_cloud & (s.pcond < F(0)), pcond=scale * s.pcond)
    piacwr = s.piacw - s.piacwi
    qwnew = qw + (s.pcond - s.piacw - s.praut - s.pracw)
    qwnew = jnp.where((qwnew <= EPSQ) | ((qw > F(0)) & (qwnew != F(0)) & (qwnew / qw < TOLER)), F(0), qwnew)
    delt = XLV1 * (s.pcond + s.pievp + s.picnd + s.prevp) + XLS1 * s.pidep + XLF1 * (s.piacwi + s.piacr - s.pimlt)
    tnew = temp + delt
    wvnew = wv + (-s.pcond - s.pidep - s.pievp - s.picnd - s.prevp)
    deli = s.pidep + s.pievp + s.piacwi + s.piacr - s.pimlt
    s.put(ice, tot_icenew=s.tot_ice + thick * deli)
    s.put(ice & (s.tot_ice > F(0)) & (s.tot_icenew != F(0)) & (s.tot_icenew / s.tot_ice < TOLER), tot_icenew=F(0))
    s.put(active, rimef=F(1))
    remaining_ice = ice & (s.tot_icenew > CLIMIT)
    psdep = jnp.where(s.pinit <= EPSQ, jnp.maximum(F(0), s.pidep), F(0))
    dels = s.piacwi + s.piacr + psdep
    s.put(remaining_ice & ((dels <= EPSQ) | (tnew >= T0C)), rimef=s.rimef1)
    grow_rime = remaining_ice & (dels > EPSQ) & (tnew < T0C)
    diameter = F(1e-6) * s.indexs.astype(jnp.float32)
    volume_mass = diameter * diameter * diameter / _table("massi", s.indexs)
    rfmx = F(PI * F(900)) * volume_mass
    density_parameter = -s.rcw * s.vsnow / jnp.minimum(F(-.5), tc)
    density = jnp.clip(F(300) * jnp.clip(density_parameter, F(.275), F(12.14)) ** F(.44), F(170), F(900))
    rfrime = jnp.where((s.piacwi > F(0)) & (s.rcw > F(0)), PI * density * volume_mass, F(1))
    rfnew = (psdep + rfrime * s.piacwi + rfmx * s.piacr) / dels
    dum2 = s.tot_ice + thick * dels
    dum3 = s.rimef1 * s.tot_ice + rfnew * thick * dels
    s.put(grow_rime, rimef=jnp.where(dum2 <= CLIMIT, s.rimef1, jnp.minimum(rfmx, jnp.maximum(F(1), dum3 / dum2))))
    outflow_ice = dtrho * (s.flimass * s.vsnow + (F(1) - s.flimass) * s.vci)
    s.put(remaining_ice, qinew=s.tot_icenew / (thick + outflow_ice))
    s.put(remaining_ice & ((s.qinew <= EPSQ) | ((qi > F(0)) & (s.qinew != F(0)) & (s.qinew / qi < TOLER))), qinew=F(0))
    s.put(remaining_ice, asnownew=s.qinew * outflow_ice)
    s.put(remaining_ice & (asnow > F(0)) & (s.asnownew != F(0)) & (s.asnownew / asnow < TOLER), asnownew=F(0))
    s.put(active, timlt=s.timlt + s.pimlt * thick)
    delr = s.praut + s.pracw + piacwr - s.piacr + s.pimlt + s.prevp + s.picnd
    s.put(active, tot_rainnew=s.tot_rain + thick * delr)
    s.put(active & (s.tot_rain > F(0)) & (s.tot_rainnew != F(0)) & (s.tot_rainnew / s.tot_rain < TOLER), tot_rainnew=F(0))
    remaining_rain = active & (s.tot_rainnew > CLIMIT)
    s.put(active & ~remaining_rain, vrain2=F(0))
    s.put(remaining_rain, qrnew=s.tot_rainnew / (thick + dtrho * s.vrain2))
    s.put(remaining_rain, rqrnew=rho * s.qrnew, indexr2=indexr1)
    s.put(remaining_rain & s.strat & (s.rqrnew > F(1e-3)), strat=False)
    s.put(remaining_rain & s.drzl & (s.timlt > EPSQ), drzl=False)
    running = remaining_rain
    for _ in range(3):
        size, n0 = _rain_dsd(s.rqrnew, s.drzl, s.strat, s.indexrmin)
        s.put(running, indexr=size, n0r=n0)
        s.put(running, vrain2=gammar * _table("vrain", s.indexr))
        s.put(running, qrnew=s.tot_rainnew / (thick + dtrho * s.vrain2))
        s.put(running, rqrnew=rho * s.qrnew)
        idr = jnp.abs(s.indexr - s.indexr2)
        s.put(running, indexr2=s.indexr)
        running &= idr > 5
    s.put(remaining_rain & ((s.qrnew <= EPSQ) | ((qr > F(0)) & (s.qrnew != F(0)) & (s.qrnew / qr < TOLER))), qrnew=F(0))
    s.put(remaining_rain, arainnew=dtrho * s.vrain2 * s.qrnew)
    s.put(remaining_rain & (arain > F(0)) & (s.arainnew != F(0)) & (s.arainnew / arain < TOLER), arainnew=F(0))
    wcnew = qwnew + s.qrnew + s.qinew
    result = (tnew, jnp.maximum(EPSQ, wvnew / (F(1) + wvnew)), jnp.maximum(EPSQ, wcnew),
              jnp.maximum(EPSQ, qwnew), jnp.maximum(EPSQ, s.qinew), jnp.maximum(EPSQ, s.qrnew), s.rimef)
    original = (temp, q, wc, qw, qi, qr, rime)
    result = tuple(jnp.where(active, new, old) for new, old in zip(result, original))
    newflux = (s.arainnew, s.asnownew, s.vrain2, s.vsnow, s.indexs0c, s.rhox0c,
               s.timlt, s.strat, s.drzl, s.indexrmin, result[-1])
    # CLEAR resets flags and TIMLT but preserves the incoming speed locals.
    clearflux = (arain, asnow, vrain1, vsnow1, indexs0c, rhox0c,
                 jnp.zeros_like(timlt), jnp.zeros_like(strat), jnp.zeros_like(drzl),
                 jnp.full_like(indexrmin, 50), rime)
    newflux = tuple(jnp.where(active, new, old) for new, old in zip(newflux, clearflux))
    rates = jnp.stack([s.pcond, s.pidep, s.piacwi, s.piacr, s.pimlt, s.praut, s.pracw, s.prevp])
    return result, newflux, rates


def _partition(wc, fice, frain):
    qi = jnp.where(fice >= F(1), wc, jnp.where(fice <= F(0), F(0), fice * wc))
    qw = jnp.where(fice >= F(1), F(0), jnp.where(fice <= F(0), wc, wc - qi))
    rain = (qw > F(0)) & (frain > F(0))
    qr = jnp.where(rain, jnp.where(frain >= F(1), qw, frain * qw), F(0))
    qw = jnp.where(rain, jnp.where(frain >= F(1), F(0), qw - qr), qw)
    return qw, qi, qr


def prepare_level(theta, vapor, cloud, rain, ice, p, pii, rho, dz, *, top, rimef=None, qrimef=None, gid=1):
    """Literal driver wiring for one level or a broadcast array of levels."""
    theta, vapor, cloud, rain, ice, p, pii, rho, dz = (jnp.asarray(x, jnp.float32)
        for x in (theta, vapor, cloud, rain, ice, p, pii, rho, dz))
    temp = theta * pii
    q = vapor / (F(1) + vapor)
    qt = cloud + rain + ice
    if rimef is None:
        rimef = jnp.ones_like(ice)
    if qrimef is not None:
        rimef = jnp.where(ice > EPSQ, jnp.asarray(qrimef, jnp.float32) / ice, F(1))
    fice = jnp.where(ice <= EPSQ, jnp.where(temp < F(T0C + F(-40)), F(1), F(0)), jnp.clip(ice / qt, F(0), F(1)))
    rimef = jnp.where(ice <= EPSQ, F(1), rimef)
    frain = jnp.where(rain <= EPSQ, F(0), rain / (rain + cloud))
    qt = jnp.where(top & (qt <= EPSQ), EPSQ, qt)
    fice, frain, rimef = jnp.where(top, F(1), fice), jnp.where(top, F(0), frain), jnp.where(top, F(1), rimef)
    negligible = qt <= F(F(1.001) * EPSQ)
    wc = jnp.where(negligible, F(0), qt)
    fice = jnp.where(negligible, jnp.where(temp - T0C < F(-40), F(1), F(0)), fice)
    frain, rimef = jnp.where(negligible, F(0), frain), jnp.where(negligible, F(1), rimef)
    qw, qi, qr = _partition(wc, fice, frain)
    rimef = jnp.where(qi <= F(0), F(1), rimef)
    thick = ((rho * F(9.806)) * dz) * F(F(1) / F(9.806))
    rh = jnp.where((gid == 1) & (p < F(85000)), F(.975), F(1))
    return (p, jnp.maximum(EPSQ, q), wc, qw, qi, qr, rimef, temp, thick, rh)


def prepare(theta, vapor, cloud, rain, ice, p, pii, rho, dz, *, rimef=None, qrimef=None, gid=1):
    """FER_HIRES + EGCP01DRV input wiring; vertical axis 0, bottom to top."""
    top = (jnp.arange(theta.shape[0]) == theta.shape[0] - 1).reshape((-1,) + (1,) * (theta.ndim - 1))
    fields = prepare_level(theta, vapor, cloud, rain, ice, p, pii, rho, dz,
                           top=top, rimef=rimef, qrimef=qrimef, gid=gid)
    return tuple(x[::-1] for x in fields), pii


def finish_level(result, pii):
    """EGCP01DRV storage and FER_HIRES species reconstruction at one level."""
    temp, q, wc, qw, qi, qr, rimef = result
    fice = jnp.where(qi <= EPSQ, jnp.where(temp < F(T0C + F(-40)), F(1), F(0)), jnp.clip(qi / wc, F(0), F(1)))
    frime = jnp.where(qi <= EPSQ, F(1), jnp.maximum(F(1), rimef))
    frain = jnp.where(qr <= EPSQ, F(0), qr / (qr + qw))
    qw, qi, qr = _partition(wc, fice, frain)
    return (temp / pii, q / (F(1) - q), wc, qw, qi, qr, fice, frain, frime, qi * frime)


def finish(result, flux, pii):
    output = finish_level(tuple(x[::-1] for x in result), pii)
    amounts = (flux[0] + flux[1]) * F(F(1) / F(1000))
    snow_ratio = jnp.where(amounts < F(1e-8), F(0), F(F(1) / F(1000)) * flux[1] / amounts)
    names = ("TH_OUT", "QV_OUT", "QT_OUT", "QC_OUT", "QI_OUT", "QR_OUT", "F_ICE_PHY_OUT", "F_RAIN_PHY_OUT", "F_RIMEF_PHY_OUT", "QRIMEF_OUT")
    return {**dict(zip(names, output)), "RAINNCV": amounts * F(1000), "SR": snow_ratio}


def initial_flux(sample):
    zero = jnp.zeros_like(sample)
    fifty = jnp.full_like(sample, 50, dtype=jnp.int32)
    false = jnp.zeros_like(sample, dtype=bool)
    return (zero, zero, zero, zero, fifty, jnp.full_like(sample, F(22.5)),
            zero, false, false, fifty, jnp.ones_like(sample))


def reference_column(theta, vapor, cloud, rain, ice, p, pii, rho, dz, dt, *, rimef=None, qrimef=None, gid=1):
    """CPU qualification route; operational GPU route will use column-local loops."""
    fields, pii = prepare(theta, vapor, cloud, rain, ice, p, pii, rho, dz, rimef=rimef, qrimef=qrimef, gid=gid)
    dt = jnp.asarray(dt, jnp.float32)

    def body(flux, row):
        result, flux, rates = level(*row, dt, flux)
        return flux, (result, rates)

    flux, (result, rates) = lax.scan(body, initial_flux(fields[0][0]), fields)
    return finish(result, flux, pii), rates
