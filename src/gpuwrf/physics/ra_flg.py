"""WRF Fu-Liou-Gu REAL radiation primitives (not operationally admitted yet).

Pressure is hPa, interface profiles run top to surface; water content is g/m3.
Tables are unchanged WRF DATA values, emitted by proofs/s2_flggfdl/tables.F90.
The column solver and operational admission remain separate gates.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax
from gpuwrf.physics import flg_transfer as transfer

G_POINTS = (10, 8, 12, 7, 12, 5, 2, 3, 4, 4, 3, 5, 2, 10, 12, 7, 7, 8)


@lru_cache(maxsize=1)
def _host_tables():
    # Cache NumPy only; never cache traced device arrays (E206).
    with np.load(Path(__file__).with_name("flg_tables_v1.npz"), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _table(key, dtype):
    return jnp.asarray(_host_tables()[key], dtype=dtype)


def layer_thickness(p, t):
    """WRF thicks, km per layer."""
    return .0146337 * (t[..., :-1] + t[..., 1:]) * jnp.log(p[..., 1:] / p[..., :-1])


def rayleigh_pressure(p):
    return 14.6337 * (p[..., :-1] + p[..., 1:]) * jnp.log(p[..., 1:] / p[..., :-1])


def ozone_profile(p, profile):
    """WRF o3prof, prescribed profile 1..5; pressure may run either direction."""
    tabp = jnp.log(_table("ozone_pres", p.dtype)[:, profile - 1])
    tabo = _table("ozone", p.dtype)[:, profile - 1]
    lp = jnp.log(p)
    # WRF's descending search stops at the matching knot and uses that knot
    # as the lower endpoint of the following interval.
    k = jnp.clip(jnp.searchsorted(tabp, lp, side="right") - 1, 0, tabp.size - 2)
    x1, x2, y1, y2 = tabp[k], tabp[k + 1], tabo[k], tabo[k + 1]
    value = (y1 * (x2 - lp) + y2 * (lp - x1)) / (x2 - x1)
    high = (tabo[-1] * (tabp[-2] - lp) + tabo[-2] * (lp - tabp[-1])) / (tabp[-2] - tabp[-1])
    value = jnp.where(lp >= tabp[-1], high, value)
    return jnp.where((lp <= tabp[0]) | (lp >= tabp[-1]), jnp.maximum(value, 0.), value)


def absorption_coefficient(p, t, coefficients, *, infrared=False, ozone=False):
    """WRF qks/qki/qkio3; returns coefficients and the live temperature profile.

    qki modifies T in place in WRF; callers must retain the returned profile.
    """
    if infrared:
        sp = jnp.asarray([.251, .398, .631, 1., 1.58, 2.51, 3.98, 6.31, 10.,
                          15.8, 25.1, 39.8, 63.1, 100., 158., 251., 398., 631., 1000.], p.dtype)
        if not ozone:
            t = jnp.clip(t, 180., 345.)
    else:
        sp = jnp.asarray([10., 15.8, 25.1, 39.8, 63.1, 100., 158., 251., 398., 631., 1000.], p.dtype)
    k = jnp.clip(jnp.searchsorted(sp, p, side="right"), 1, sp.size - 1)
    delta = t - (250. if ozone else 245.)
    square = delta * delta
    c = coefficients

    def polynomial(index):
        # WRF's REAL intermediates round before the next operation. Exp magnifies
        # contracted polynomial errors; preserve these boundaries explicitly.
        # optimization_barrier alone does not prevent CPU FMA (E107).
        # nextafter(x,x) is value-identical and enforces REAL rounding (E100).
        linear = c[1, index] * delta
        linear = jnp.nextafter(linear, linear)
        quadratic = c[2, index] * square
        quadratic = jnp.nextafter(quadratic, quadratic)
        return (c[0, index] + linear) + quadratic

    x1 = jnp.exp(polynomial(k - 1))
    x2 = jnp.exp(polynomial(k))
    interp = x1 + (x2 - x1) / (sp[k] - sp[k - 1]) * (p - sp[k - 1])
    # k is clamped to 1 below sp[0], so x1 already is the exact first-table
    # exponential. Reuse it instead of a third polynomial/exp per interface.
    low = x1 * p / sp[0]
    return jnp.where(p < sp[0], low, interp), t


def continuum(p, t, q, band):
    """WRF gascon/qopcon; outside bands 11..17 the continuum is zero."""
    if not 11 <= band <= 17:
        return jnp.zeros_like(p[..., :-1])
    vv = (1175., 1040., 890., 735., 605., 470., 340.)[band - 11]
    s = (jnp.asarray(4.18, p.dtype) + 5577.8 * jnp.exp(jnp.asarray(-.00787, p.dtype) * vv)) / 1013.25
    pe = p * q / (.622 + .378 * q)
    w = jnp.exp(1800. / t - 6.08108)
    ff = s * (pe + .002 * p) * w
    return (ff[..., :-1] * q[..., :-1] + ff[..., 1:] * q[..., 1:]) * (p[..., 1:] - p[..., :-1]) * .5098835


def _mass_path(fkg, mixing_ratio, p, scale):
    return (fkg[..., :-1] * mixing_ratio[..., :-1] + fkg[..., 1:] * mixing_ratio[..., 1:]) * (p[..., 1:] - p[..., :-1]) * scale


def _gas_path(fkg, p, scale):
    return (fkg[..., :-1] + fkg[..., 1:]) * (p[..., 1:] - p[..., :-1]) * scale


def gas_optics(p, t, q, o3, band, gpoint, *, co2=345., ch4=1.6, n2o=.28):
    """WRF gases with NGAS=0; band 1..18 and gpoint 0..KG-1 are static."""
    weight = _table(f"hk_{band}", p.dtype)[gpoint]
    if band == 1:
        fq = 238.08 * _table("fko3_1", p.dtype)[gpoint]
        depth = (o3[..., :-1] + o3[..., 1:]) * (p[..., 1:] - p[..., :-1]) * fq
        return depth, 619.618 * weight, t
    if band in (14, 15):
        fa, t = absorption_coefficient(p, t, _table(f"coehca_{band}", p.dtype)[..., gpoint], infrared=True)
        fb, t = absorption_coefficient(p, t, _table(f"coehcb_{band}", p.dtype)[..., gpoint], infrared=True)
        pq = jnp.where(p >= 63.1, q, 0.)
        fkg = fa / 330. * co2 + pq * fb
        return _gas_path(fkg, p, .5), weight, t
    if band == 12:
        fk_o3, t = absorption_coefficient(p, t, _table("coeo3_12", p.dtype)[..., gpoint], infrared=True, ozone=True)
        tg1 = _mass_path(fk_o3, o3, p, 238.08)
        fk_h2o, t = absorption_coefficient(p, t, _table("coeh2o_12", p.dtype), infrared=True)
        return tg1 + _mass_path(fk_h2o, q, p, 634.9205), weight, t
    fk, t = absorption_coefficient(p, t, _table(f"coeh2o_{band}", p.dtype)[..., gpoint], infrared=band >= 7)
    depth = _mass_path(fk, q, p, 634.9205)
    if band in (10, 11):
        fm, t = absorption_coefficient(p, t, _table(f"coech4_{band}", p.dtype), infrared=True)
        fn, t = absorption_coefficient(p, t, _table(f"coen2o_{band}", p.dtype), infrared=True)
        depth = depth + _gas_path(fm, p, 6.3119e-4) / 1.6 * ch4 + _gas_path(fn, p, 1.10459e-4) / .28 * n2o
    if band <= 6:
        weight = weight * (484.295, 149.845, 48.7302, 31.6576, 5.79927)[band - 2]
    return depth, weight, t


def _phase_moments(g):
    g2 = g * g
    g3 = g2 * g
    g4 = g3 * g
    return jnp.stack((3. * g, 5. * g2, 7. * g3, 9. * g4), axis=-1)


def precipitation_optics(content, dz, band, *, graupel=False):
    """WRF rain/graup; the threshold is WRF's own process branch."""
    prefix = "graup" if graupel else "rain"
    extinction = _table(f"{prefix}_{'bg' if graupel else 'brn'}", content.dtype)[band - 1]
    albedo = _table(f"{prefix}_{'wgf' if graupel else 'wrnf'}", content.dtype)[band - 1]
    g = _table(f"{prefix}_{'gg' if graupel else 'grn'}", content.dtype)[band - 1]
    density = _table(f"{prefix}_{'gwc' if graupel else 'rwc'}", content.dtype)
    active = content >= 1.e-5
    tau = jnp.where(active, dz * content * extinction / density, 0.)
    omega = jnp.where(active, albedo, 0.)
    moments = jnp.where(active[..., None], _phase_moments(g), 0.)
    return tau, omega, moments


def rayleigh_optics(trp, band, coszen):
    if band > 6:
        z = jnp.zeros_like(trp)
        return z, z, jnp.zeros(trp.shape + (4,), trp.dtype)
    x = (-3.902860e-6 * coszen * coszen + 6.120070e-6 * coszen + 4.177440e-6) if band == 1 else _table("rayle_ri", trp.dtype)[band - 1]
    tau = trp * x
    return tau, jnp.ones_like(tau), jnp.broadcast_to(jnp.asarray([0., .5, 0., 0.], trp.dtype), trp.shape + (4,))


def ice_effective_size(iwc, temperature, latitude):
    """Default NPDE=2/PDERANDOM=false mean size, with WRF REAL coefficients."""
    lat = jnp.abs(latitude)
    region = jnp.where(lat < 30., jnp.int32(0), jnp.where(lat > 60., jnp.int32(3), jnp.where(temperature < 233., jnp.int32(1), jnp.int32(2))))
    a = jnp.asarray([5.4199, 4.3257, 5.2375, 4.851], iwc.dtype)[region]
    b = jnp.asarray([.35211, .26535, .13142, .33159], iwc.dtype)[region]
    c = jnp.asarray([.01268, .021864, 0., .026189], iwc.dtype)[region]
    critical = jnp.asarray([9.2125e-7, .0024, 0., 0.], iwc.dtype)[region]
    minimum = jnp.asarray([19.6, 33.81, 0., 0.], iwc.dtype)[region]
    log_iwc = jnp.log(jnp.where(iwc > 1.e-7, iwc, 1.))
    mean = jnp.exp(a + b * log_iwc + c * log_iwc ** 2)
    mean = jnp.where(iwc <= critical, minimum, mean)
    fallback=jnp.where(iwc > 0.,jnp.asarray(85.,iwc.dtype),jnp.zeros_like(iwc))
    return jnp.where(iwc > 1.e-7, jnp.clip(mean, 10., 150.), fallback)


def ice_optics(iwc, diameter, dz, band):
    """WRF ice_new_comb (NICE=2); tuple is optical depth/albedo/4 moments."""
    active = iwc >= 1.e-5
    d = jnp.where(active, diameter, 1.)
    d2, d3 = d * d, (d * d) * d
    ap = _table("ice_ap", iwc.dtype)[:, band - 1]
    bp = _table("ice_bp", iwc.dtype)[:, band - 1]
    tau = dz * 1000. * iwc * (ap[0] + ap[1] / d + ap[2] / d2)
    omega = 1. - (bp[0] + bp[1] * d + bp[2] * d2 + bp[3] * d3)
    if band <= 6:
        cps = _table("ice_cps", iwc.dtype)[..., band - 1]
        moments = cps[0] + cps[1] * d[..., None] + cps[2] * d2[..., None] + cps[3] * d3[..., None]
    else:
        cpir = _table("ice_cpir", iwc.dtype)[:, band - 7]
        g = cpir[0] + cpir[1] * d + cpir[2] * d2 + cpir[3] * d3
        moments = _phase_moments(g)
    return jnp.where(active, tau, 0.), jnp.where(active, omega, 0.), jnp.where(active[..., None], moments, 0.)


def water_optics(lwc, radius, dz, band):
    """WRF water_fl, interpolating extinction in inverse effective radius."""
    re, fl = _table("water_re", lwc.dtype), _table("water_fl", lwc.dtype)
    bz = _table("water_bz", lwc.dtype)[:, band - 1]
    wz, gz = _table("water_wz", lwc.dtype)[:, band - 1], _table("water_gz", lwc.dtype)[:, band - 1]
    active = lwc >= 1.e-5
    radius = jnp.clip(radius, re[0], re[-1])
    k = jnp.clip(jnp.searchsorted(re, radius, side="right") - 1, 0, re.size - 2)
    inverse = 1. / radius
    extinction = bz[k] / fl[k] + (bz[k + 1] / fl[k + 1] - bz[k] / fl[k]) / (1. / re[k + 1] - 1. / re[k]) * (inverse - 1. / re[k])
    omega = wz[k] + (wz[k + 1] - wz[k]) / (re[k + 1] - re[k]) * (radius - re[k])
    g = gz[k] + (gz[k + 1] - gz[k]) / (re[k + 1] - re[k]) * (radius - re[k])
    # Source endpoint branches use the tabulated value directly, without
    # interpolating to an endpoint (which would introduce cancellation).
    extinction = jnp.where(radius <= re[0], bz[0] / fl[0], jnp.where(radius >= re[-1], bz[-1] / fl[-1], extinction))
    omega = jnp.where(radius <= re[0], wz[0], jnp.where(radius >= re[-1], wz[-1], omega))
    g = jnp.where(radius <= re[0], gz[0], jnp.where(radius >= re[-1], gz[-1], g))
    tau = jnp.where(radius <= re[0], dz * lwc * bz[0] / fl[0],
                    jnp.where(radius >= re[-1], dz * lwc * bz[-1] / fl[-1], dz * lwc * extinction))
    return jnp.where(active, tau, 0.), jnp.where(active, omega, 0.), jnp.where(active[..., None], _phase_moments(g), 0.)


def planck_band(t, surface_t, band):
    """WRF planck; sequential 10 cm^-1 bins, W/m2/sr at interfaces/surface."""
    if not 7 <= band <= 18:
        raise ValueError("FLG Planck band must be 7..18")
    start = (2200.,1900.,1700.,1400.,1250.,1100.,980.,800.,670.,540.,400.,280.001)[band - 7]
    count = (30,20,30,15,15,12,18,13,13,14,12,28)[band - 7]
    # Fold only the source's loop-invariant REAL products, rounding after each
    # multiplication; do not fold a Python-double prefix (E83/E133).
    f = np.float32
    v1 = f(start)
    fq1, fq2 = [], []
    for _ in range(count):
        v2 = f(v1 - f(10.))
        w = f(f(v1 + v2) * f(.5))
        fq1.append(f(f(f(f(1.19107e-8) * w) * w) * w))
        fq2.append(f(f(1.43884) * w))
        v1 = v2
    a, b = jnp.asarray(np.asarray(fq1), t.dtype), jnp.asarray(np.asarray(fq2), t.dtype)

    def body(i, carry):
        radiance, surface = carry
        radiance = radiance + a[i] / (jnp.exp(b[i] / t) - 1.)
        surface = surface + a[i] / (jnp.exp(b[i] / surface_t) - 1.)
        return radiance, surface

    radiance, surface = lax.fori_loop(0, count, body, (jnp.zeros_like(t), jnp.zeros_like(surface_t)))
    return radiance * 10., surface * 10.


def preprocess_column(p8w, interface_t, mass_t, rho, qv, qc, qr, qi, qs, qg,
                      albedo, emissivity, surface_t, latitude, longitude,
                      julday, gmt, xtime, radt, declination, solar_constant,
                      *, f_qi=True, warm_rain=False, cloud_fraction=None):
    """Literal RAD_FLG input mapping, defaults NPDE2/NICE2/NOZONE1/NAERO0.

    Fields include the top interface/padded mass slot. Default NFRACT=0 is the
    original binary-cloud mode; cloud_fraction selects WRF's NFRACT=1 API path.
    """
    p = p8w[..., ::-1] / 100.
    t = interface_t[..., ::-1]
    vapor = jnp.maximum(qv[..., ::-1], 1.e-20)
    lat = jnp.abs(latitude)
    summer = (julday > 80) & (julday < 265)
    summer = jnp.where(latitude >= 0., summer, (julday < 80) | (julday > 265))
    profile = jnp.where(lat <= 30., 5, jnp.where(lat > 60., jnp.where(summer,3,4), jnp.where(summer,1,2)))
    # One column: dynamic profile is legal in table indexing; no host conversion.
    ozone = ozone_profile(p, profile)
    density = rho[..., -3::-1]
    lwc = (qc[..., -3::-1] * density) * 1.e3
    rain = (qr[..., -3::-1] * density) * 1.e3
    graupel = (qg[..., -3::-1] * density) * 1.e3
    above_t = mass_t[..., -2:0:-1]
    if f_qi:
        iwc = ((qi[..., -3::-1] + qs[..., -3::-1]) * density) * 1.e3
    elif not warm_rain:
        iwc = jnp.where(t[..., 1:-1] < 273.15, lwc + rain, 0.)
        lwc = jnp.where(t[..., 1:-1] < 273.15, 0., lwc)
        rain = jnp.where(t[..., 1:-1] < 273.15, 0., rain)
    else:
        iwc = jnp.zeros_like(lwc)
    diameter = ice_effective_size(iwc, above_t, latitude)
    radius = jnp.where(lwc > 0., jnp.asarray(10.,lwc.dtype), jnp.zeros_like(lwc))
    cloud = jnp.where((lwc > 0.) | (iwc > 0.), jnp.ones_like(lwc), jnp.zeros_like(lwc))
    if cloud_fraction is not None:
        cloud = cloud * cloud_fraction[..., -2:0:-1]
    prepend = lambda a: jnp.concatenate((jnp.zeros_like(a[..., :1]), a), axis=-1)
    iwc, diameter, lwc, radius, rain, graupel, cloud = map(prepend, (iwc, diameter, lwc, radius, rain, graupel, cloud))
    # Source midnight modulo includes RADT/2; day/declination is supplied by WRF.
    xt24 = jnp.fmod(xtime + radt * .5, 1440.)
    local = gmt + xt24 / 60. + longitude / 15.
    degrad = jnp.asarray(np.float32(np.float32(3.141592653589793) / np.float32(180.)), p.dtype)
    hour_angle = 15. * (local - 12.) * degrad
    lat_rad = latitude * degrad
    coszen = jnp.sin(lat_rad) * jnp.sin(declination) + jnp.cos(lat_rad) * jnp.cos(declination) * jnp.cos(hour_angle)
    return dict(p=p,t=t,q=vapor,o3=ozone,iwc=iwc,diameter=diameter,lwc=lwc,radius=radius,
                rain=rain,graupel=graupel,cloud=cloud,inhomogeneity=jnp.ones_like(cloud),
                albedo=albedo,emissivity=emissivity,surface_t=jnp.clip(surface_t,180.,320.),
                coszen=coszen,solar_constant=solar_constant)


def _combined_default_optics(ice,water,rain,graupel,rayleigh,continuum_depth,gas_depth):
    """RAD_ALL/COMSCP default NFRACT0/NINHO0/NAERO0 configuration.

    With binary clouds each cloud group has one unit-area configuration. The
    source always chooses CLEAR optics above its three groups, even if those
    highest layers contain condensate. Keep that exact source convention.
    """
    ti,wi,mi=ice;tw,ww,mw=water;trn,wrn,mr=rain;tgr,wgr,mg=graupel;tr,_,mray=rayleigh
    nz=ti.shape[0]
    head=nz-((nz-1)//3)*3
    cloudy=jnp.arange(nz)>=head
    clear_depth=tr+continuum_depth+gas_depth+trn+tgr
    cloudy_depth=ti+tw+tr+continuum_depth+gas_depth+trn+tgr
    depth=jnp.where(cloudy,cloudy_depth,clear_depth)
    tis=ti*wi;tws=tw*ww;trns=trn*wrn;tgrs=tgr*wgr
    clear_scatter=tr+trns+tgrs
    cloudy_scatter=tis+tws+tr+trns+tgrs
    scatter=jnp.where(cloudy,cloudy_scatter,clear_scatter)
    omega=scatter/depth
    clear_moments=(tr[:,None]*mray+trns[:,None]*mr+tgrs[:,None]*mg)
    cloudy_moments=(tis[:,None]*mi+tws[:,None]*mw+tr[:,None]*mray+trns[:,None]*mr+tgrs[:,None]*mg)
    numerator=jnp.where(cloudy[:,None],cloudy_moments,clear_moments)
    small=scatter<1.e-20
    moments=jnp.where(small[:,None],0.,numerator/jnp.where(small,1.,scatter)[:,None])
    return moments,omega,transfer.cumulative_depth(depth)


def _column_from_preprocessed(pre,pi,*,gravity=9.81,cp=1004.5):
    """Complete default RAD_FLG physics on one column (reference composition).

    This is the CPU/qualification composition. Native operational GPU dispatch
    must use a Ref-based Pallas column kernel; do not domain-vmap these loops.
    Inputs are exactly the preprocessed boundary captured by the native oracle.
    """
    p,t,q,o3=(pre[key] for key in ('p','t','q','o3'))
    nz=p.shape[0]-1
    dz=layer_thickness(p,t);trp=rayleigh_pressure(p)
    zeros=jnp.zeros_like(p)
    f0=jnp.asarray(np.float32(np.float32(1.)/np.float32(3.14159)),p.dtype)

    def optics(band):
        ice=ice_optics(pre['iwc'],pre['diameter'],dz,band)
        water=water_optics(pre['lwc'],pre['radius'],dz,band)
        rain=precipitation_optics(pre['rain'],dz,band)
        graupel=precipitation_optics(pre['graupel'],dz,band,graupel=True)
        rayleigh=rayleigh_optics(trp,band,pre['coszen'])
        return ice,water,rain,graupel,rayleigh

    def shortwave(_):
        up,down,direct,diffuse=zeros,zeros,zeros,zeros
        for band in range(1,7):
            components=optics(band)
            con=continuum(p,t,q,band)
            def gpoint(ig,carry):
                up,down,direct,diffuse=carry
                gas,hk,_=gas_optics(p,t,q,o3,band,ig)
                moments,omega,total=_combined_default_optics(*components,con,gas)
                fu,fd,fdirect,fdiffuse=transfer.solar_transfer(pre['albedo'],pre['coszen'],f0,moments,omega,total)
                return up+fu*hk,down+fd*hk,direct+fdirect*hk,diffuse+fdiffuse*hk
            up,down,direct,diffuse=lax.fori_loop(0,G_POINTS[band-1],gpoint,(up,down,direct,diffuse))
        scale=pre['solar_constant']/1340.
        return up*scale,down*scale,direct*scale,diffuse*scale

    sw_up,sw_down,direct,diffuse=lax.cond(pre['coszen']>1.e-4,shortwave,lambda _:(zeros,zeros,zeros,zeros),None)
    lw_up,lw_down=zeros,zeros
    clipped_t=jnp.clip(t,180.,345.)
    for band in range(7,19):
        components=optics(band)
        # qki in the first IR g-point updates PT. Band7 Planck precedes it;
        # all later Planck/continuum calls see the clipped working temperature.
        bt=t if band==7 else clipped_t
        bf,bs=planck_band(bt,pre['surface_t'],band)
        con=continuum(p,clipped_t,q,band)
        def gpoint(ig,carry):
            up,down=carry
            gas,hk,_=gas_optics(p,clipped_t,q,o3,band,ig)
            moments,omega,total=_combined_default_optics(*components,con,gas)
            fu,fd=transfer.infrared_transfer(pre['emissivity'],bf,bs,moments,omega,total)
            return up+fu*hk,down+fd*hk
        lw_up,lw_down=lax.fori_loop(0,G_POINTS[band-1],gpoint,(lw_up,lw_down))
    # WRF's final band18 surface-emission correction is applied at every level.
    correction=bs*.03*3.14159*pre['emissivity']
    lw_up=lw_up+correction
    # Preserve separate REAL differences; do not difference rounded net totals.
    dts=(sw_down[:-1]-sw_up[:-1]-sw_down[1:]+sw_up[1:])/(p[1:]-p[:-1])
    dtir=(lw_down[:-1]-lw_up[:-1]-lw_down[1:]+lw_up[1:])/(p[1:]-p[:-1])
    g=jnp.asarray(gravity,p.dtype);heat_capacity=jnp.asarray(cp,p.dtype)
    fac=.01*g/heat_capacity
    sw_rate=dts[::-1]*fac/pi[:nz]
    lw_rate=dtir[::-1]*fac/pi[:nz]
    total_rate=jnp.zeros_like(pi[:nz])+(dts+dtir)[::-1]*fac/pi[:nz]
    dni=jnp.where(pre['coszen']>1.e-4,direct[-1]/jnp.where(pre['coszen']>1.e-4,pre['coszen'],1.),0.)
    return dict(total_rate=total_rate,lw_rate=lw_rate,sw_rate=sw_rate,
                glw=lw_down[-1],gsw=sw_down[-1]-sw_up[-1],swdn=sw_down[-1],
                sw_up_top=sw_up[0],lw_up_top=lw_up[0],direct=direct[-1],diffuse=diffuse[-1],dni=dni)
