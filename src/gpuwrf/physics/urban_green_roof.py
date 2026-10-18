"""WRF BEP+BEM green roofs (GR_FLAG=1, GR_FRAC_ROOF>0), module_sf_bep_bem.F / module_sf_bem.F.

Literal REAL operation order of roof_temp_veg (:2419-2592), soil_temp_veg (:3866-3959),
soil_moist (:3779-3857), flux_flat_roof (:4404-4589) and BEM wall_gr (module_sf_bem.F:909-1034).
Every array here carries the green-roof soil column of one roof level and street direction:
``(..., 10)`` layers, deepest (roof side) first, surface last, as in WRF.

gfortran -O2 keeps IEEE division by constants and calls libm powf for real exponents, so the
constant divisors/exponents ride through :func:`urban_bem_building.real_literal` (XLA would
otherwise rewrite x/c -> x*(1/c) and x**.5 -> sqrt).
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.urban_bem_building import real_literal
from gpuwrf.physics.urban_bep_pbl import _invert

NGR = 10
# roof_temp_veg / flux_flat_roof DATA statements.
DZGR = (0.1, 0.003, 0.06, 0.003, 0.05, 0.04, 0.02, 0.0125, 0.005, 0.0025)
QRMAX = (0.0, 0.0, 0.0, 0.0, 0.439, 0.37, 0.37, 0.37, 0.37, 0.37)
SMAX = (0, 0, 0, 0, -0.01, -0.1, -0.1, -0.1, -0.1, -0.1)
KMAX = (0, 0, 0, 0, 3.32e-3, 2.162e-3, 2.162e-3, 2.162e-3, 2.162e-3, 2.162e-3)
BEXP = (0, 0, 0, 0, 2.7, 3.9, 3.9, 3.9, 3.9, 3.9)
CD = (0, 0, 0, 0, 331500, 1.342e6, 1.342e6, 1.342e6, 1.342e6, 1.342e6)
CSA = (7.5e4, 2.1e6, 4.48e4, 2.1e6)
KA = (0.035, 0.7, 0.024, 0.7)
CW = 4.295e6
SIGMA, LATENT, CP, G, VK, P0 = 5.67e-08, 2.45e+06, 1004., 9.81, 0.40, 1.e+5


def _rcp(T):
    return T(287.)/T(1004.)  # parameter rcp_u=r/cp_u, folded in REAL
H_WATER = 0.0009722  # module_sf_bep_bem.F:54, mm of irrigation per hour
Z0V = 0.01  # buildings(): vegetation roughness (:2717)
# gr_type -> (emissivity, albedo) (roof_temp_veg) and (rsmin, rsmax, lai) (flux_flat_roof).
SURFACE = {1: (0.95, 0.3), 2: (0.83, 0.154)}
STOMATA = {1: (40., 5000., 2.), 2: (150., 5000., 3.)}


class GreenRoofConfig(NamedTuple):
    fraction: float   # gr_frac_roof_u(iurb)
    gr_type: int      # GR_TYPE (URBPARM.TBL)
    irho: tuple       # IRHO(24) irrigation hours (URBPARM.TBL)


def config_from_params(params):
    fraction = float(params.get('gr_frac_roof', 0.))
    if int(params.get('gr_flag', 0)) != 1 or fraction <= 0.:
        return None
    gr_type = int(params['gr_type'])
    if gr_type not in SURFACE:
        # roof_temp_veg leaves em_gr/alb_gr undefined for other GR_TYPE values.
        raise NotImplementedError(f'BEP+BEM green roof GR_TYPE={gr_type}: WRF defines only 1 and 2')
    return GreenRoofConfig(fraction, gr_type, tuple(int(x) for x in params['irho']))


def irrigation(config, hour, dtype):
    """BEP1D :1494-1504: irri_now=h_water/sum(irho) in irrigation hours (nhourday local)."""
    T = np.dtype(dtype).type
    total = sum(config.irho)
    if total <= 0:
        return jnp.zeros_like(hour)
    per_step = T(H_WATER)/T(total)
    table = jnp.asarray(np.asarray(config.irho, np.int32) != 0)
    return jnp.where(table[hour.astype(jnp.int32)], jnp.asarray(per_step, dtype), jnp.asarray(0, dtype))


def roof_average(roof_outer, green_outer, fraction):
    """BEP1D :1528: tr_av=((1-gr)*tr**4.+gr*trv**4.)**(1./4.) (powf)."""
    dtype = roof_outer.dtype;R = lambda x: real_literal(x, dtype)
    gr = R(fraction)
    return ((1-gr)*roof_outer**R(4.)+gr*green_outer**R(4.))**R(1./4.)


def _hydraulics(qr, dtype):
    """roof_temp_veg layer loop: clamped moisture, cs, alagr, k, d for all 10 layers."""
    T = np.dtype(dtype).type;R = lambda x: real_literal(x, dtype)
    dz = R(np.asarray(DZGR, dtype))
    cs = [];alagr = [];k = [None]*NGR;d = [None]*NGR
    for ig in range(NGR):
        if ig < 4:
            cs.append(jnp.full(qr.shape[:-1], T(CSA[ig]), dtype))
            alagr.append(jnp.full(qr.shape[:-1], T(KA[ig])/T(CSA[ig]), dtype))
            continue
        if ig > 4:
            qr_m = (qr[..., ig]*dz[ig-1]+qr[..., ig-1]*dz[ig])/(dz[ig]+dz[ig-1])
        else:
            qr_m = qr[..., ig]
        b = T(BEXP[ig]);qrmax = R(T(QRMAX[ig]))
        c = (1-qr_m)*T(CD[ig])+qr_m*T(CW)
        s = T(SMAX[ig])*(qrmax/qr_m)**R(b)
        k[ig] = T(KMAX[ig])*(qr_m/qrmax)**R(T(2)*b+T(3))
        d[ig] = -b*T(KMAX[ig])*T(SMAX[ig])*((qr_m/qrmax)**R(b+T(3)))/qr_m
        logs = jnp.log10(jnp.abs(s))
        # Two independent IFs (:2551-2556): NaN keeps neither, i.e. WRF's prior value.
        ala = jnp.where(logs <= T(5.1), jnp.exp(-(logs+T(2.7)))*T(4.186e2)/c, jnp.nan)
        ala = jnp.where(logs > T(5.1), T(0.00041)*T(4.186e2)/c, ala)
        cs.append(c);alagr.append(ala)
    return jnp.stack(cs, axis=-1), jnp.stack(alagr, axis=-1), k, d


def soil_temp_veg(heflro, temp, ala, cs, rs, rl, press, dt, *, em, alb, sf, lf, pv, tpv):
    """module_sf_bep_bem.F:soil_temp_veg -> (temp, pt)."""
    dtype = temp.dtype;T = np.dtype(dtype).type;R = lambda x: real_literal(x, dtype)
    dz = R(np.asarray(DZGR, dtype));em = T(em);alb = T(alb);sigma = T(SIGMA)
    if pv > 0.:
        P = R(pv)
        rl_eff = (1-P)*em*rl+em*sigma*tpv**4*P
        rs_eff = (T(1)-P)*rs
    else:
        rl_eff = em*rl
        rs_eff = rs
    tsig = temp[..., -1]
    alpha = (T(1)-alb)*rs_eff+rl_eff-em*sigma*(tsig**4)+sf+lf
    cddz = [ala[..., 0]/dz[0]]+[T(2)*ala[..., i]/(dz[i]+dz[i-1]) for i in range(1, NGR)]
    zero = jnp.zeros_like(tsig);one = jnp.ones_like(tsig)
    lower = [zero];diag = [one];upper = [zero]
    rhs = [temp[..., 0]-heflro*dt/dz[0]]
    for i in range(1, NGR-1):
        lower.append(-cddz[i]*dt/dz[i])
        diag.append(T(1)+dt*(cddz[i]+cddz[i+1])/dz[i])
        upper.append(-cddz[i+1]*dt/dz[i])
        rhs.append(temp[..., i])
    lower.append(-dt*cddz[-1]/dz[-1]);diag.append(T(1)+dt*cddz[-1]/dz[-1]);upper.append(zero)
    rhs.append(temp[..., -1]+dt*alpha/cs[..., -1]/dz[-1])
    stack = lambda xs: jnp.stack(xs, axis=-1)
    new = _invert(stack(lower), stack(diag), stack(upper), stack(rhs))
    pt = new[..., -1]*(press/R(P0))**R(-_rcp(T))
    return new, pt


def soil_moist(qv, dt, lf, d, k, rain, drain, irri):
    """module_sf_bep_bem.F:soil_moist (layers 1-4 are identity rows)."""
    dtype = qv.dtype;T = np.dtype(dtype).type;R = lambda x: real_literal(x, dtype)
    dz = R(np.asarray(DZGR, dtype));dw = T(1000.)
    alpha = rain/(dw*dt)+lf/R(LATENT)/R(dw)+irri/R(dw)
    cddz = [None]*NGR
    for i in range(4, NGR):  # cddz(2:4) use undefined d(2:4) but only feed identity rows
        cddz[i] = T(2.)*d[i]/(dz[i]+dz[i-1])
    zero = jnp.zeros_like(alpha);one = jnp.ones_like(alpha)
    lower = [zero]*4;diag = [one]*4;upper = [zero]*4;rhs = [qv[..., i] for i in range(4)]
    lower.append(zero);diag.append(T(1)+dt*(cddz[5])/dz[4]);upper.append(-cddz[5]*dt/dz[4])
    rhs.append(qv[..., 4]+dt*(k[5]-drain)/dz[4])
    for i in range(5, NGR-1):
        lower.append(-cddz[i]*dt/dz[i])
        diag.append(T(1)+dt*(cddz[i]+cddz[i+1])/dz[i])
        upper.append(-cddz[i+1]*dt/dz[i])
        rhs.append(qv[..., i]+dt*(k[i+1]-k[i])/dz[i])
    lower.append(-dt*cddz[-1]/dz[-1]);diag.append(T(1)+dt*cddz[-1]/dz[-1]);upper.append(zero)
    rhs.append(qv[..., -1]+dt*alpha/dz[-1]-dt*k[NGR-2]/dz[-1])
    stack = lambda xs: jnp.stack(xs, axis=-1)
    return _invert(stack(lower), stack(diag), stack(upper), stack(rhs))


class GreenRoofStep(NamedTuple):
    temperature: object   # trv (..., 10)
    moisture: object      # qr (..., 10)
    potential: object     # ptrv
    roof_flux: object     # hfgr (heat flux green roof -> roof layer 5)
    drain: object         # drain (before soil_moist, from the clamped layer-5 moisture)


def roof_temp_veg(trv, qr, sfrv, lfrv, roof5, rs, rl, press, dt, rain, irri, *, config,
                  roof_conductivity5, roof_capacity5, roof_depth5, pv, tpv):
    """module_sf_bep_bem.F:roof_temp_veg for one level/direction batch (broadcast ``...``)."""
    dtype = trv.dtype;T = np.dtype(dtype).type;R = lambda x: real_literal(x, dtype)
    qr = jnp.maximum(qr, T(1e-6))  # cenlin 11/4/2020
    cs, alagr, k, d = _hydraulics(qr, dtype)
    alar = T(roof_conductivity5);csr = T(roof_capacity5);dzr = T(roof_depth5)
    hfgroof = (alar/csr+alagr[..., 0])*(trv[..., 0]-roof5)/R(dzr+T(DZGR[0]))
    em, alb = SURFACE[config.gr_type]
    temp, pt = soil_temp_veg(hfgroof, trv, alagr, cs, rs, rl, press, dt, em=em, alb=alb, sf=sfrv, lf=lfrv, pv=pv, tpv=tpv)
    b5 = T(BEXP[4])
    drain = T(KMAX[4])*(qr[..., 4]/R(T(QRMAX[4])))**R(T(2)*b5+T(3))
    moist = soil_moist(qr, dt, lfrv, d, k, rain, drain, irri)
    qrmax = R(np.asarray(QRMAX, dtype))
    moist = jnp.maximum(jnp.minimum(moist, qrmax), T(1e-6))
    return GreenRoofStep(temp, moist, pt, hfgroof, drain)


def flux_flat_roof(dz, ua, va, pt, pt0, ptg, da, qv, pr, rsg, qr, *, gr_type, pv):
    """module_sf_bep_bem.F:flux_flat_roof -> (uhb, vhb, sf, lf, ehb)."""
    dtype = pt.dtype;T = np.dtype(dtype).type;R = lambda x: real_literal(x, dtype)
    b, cm, ch, rr = T(9.4), T(7.4), T(5.3), T(0.74)
    z0 = T(Z0V);z0t = z0/T(10.)
    rsmin, rsmax, lai = (T(x) for x in STOMATA[gr_type])
    rs_eff = (1-R(pv))*rsg
    utot = (ua**2+va**2)**R(.5)
    zz = T(dz)/T(2.)
    utot = jnp.maximum(utot, T(0.01))
    ric = T(2.)*T(G)*zz*(pt-ptg)/((pt+ptg)*(utot**2))
    # log of the static zz/z0: correctly rounded (gfortran folds or calls logf).
    aa = T(VK)/T(np.log(np.float64(zz/z0)));ah = T(VK)/T(np.log(np.float64(zz/z0t)))
    stable = T(1.)/(T(1.)+T(0.5)*b*ric)**2
    c = b*cm*aa*aa*R(zz/z0)**R(.5)
    unstable = jnp.maximum(-ric, T(0.))**R(.5)
    fm = jnp.where(ric > T(0.), stable, T(1.)-b*ric/(T(1.)+c*unstable))
    c = b*cm*aa*ah*R(zz/z0t)**R(.5)
    c = c*ch/R(cm)
    fh = jnp.where(ric > T(0.), stable, T(1.)-b*ric/(1+c*unstable))
    fbuw = -aa*aa*utot*utot*fm
    fbpt = -aa*ah*utot*(pt-ptg)*fh/R(rr)
    tmp = ptg*(pr/R(P0))**R(_rcp(T))-T(273.15)
    es = T(6.11)*(R(10.)**(tmp*T(7.5)/(T(237.7)+tmp)))
    qvsg = T(0.62197)*es/(T(0.01)*pr-T(0.378)*es)
    f = T(0.55)*rs_eff/R(T(100.))*T(2.)/R(lai)
    f1 = (f+rsmin/rsmax)/(T(1.)+f)
    ta = pt*(pr/R(P0))**R(_rcp(T))-T(273.15)
    esa = T(6.11)*(R(10.)**(ta*T(7.5)/(T(237.7)+ta)))
    qvs = T(0.62197)*esa/(T(0.01)*pr-T(0.378)*esa)
    f2 = T(1.)/(T(1.)+T(36.35)*(qvs-qv))
    f3 = T(1.)-T(0.0016)*(T(25.)-ta)**2
    qw, qref = T(0.06), T(0.37)
    f4 = jnp.zeros_like(pt);total = T(0.)
    for iz in range(4, NGR):
        gx = (qr[..., iz]-qw)/R(qref-qw)
        gx = jnp.where(gx > T(1), T(1.), gx)
        gx = jnp.where(gx < T(0), T(0.), gx)
        f4 = f4+gx*T(DZGR[iz]);total = total+T(DZGR[iz])
    f4 = f4/R(total)
    rsveg = jnp.minimum(rsmin/jnp.maximum(lai*f1*f2*f3*f4, T(1e-9)), rsmax)
    resg = rr/(aa*aa*utot*fh)
    fbqq = -(qv-qvsg)/(resg+rsveg)
    ustar = (-fbuw)**R(.5)
    tstar = -fbpt/ustar
    qstar = -fbqq/ustar
    buu = -T(G)/pt0*ustar*tstar
    return (-ustar*ustar*ua/utot, -ustar*ustar*va/utot, ustar*tstar*da*T(CP), ustar*qstar*da*T(LATENT), buu)


def roof_wall_gr(temp, depth, conductivity, capacity, flux_inner, flux_outer, dt, *, fraction, hfgr):
    """module_sf_bem.F:wall_gr: wall() with the layer-5 green-roof coupling (:1005-1011)."""
    dtype = temp.dtype;R = lambda x: real_literal(x, dtype)
    n = temp.shape[-1]
    dz = jnp.asarray(depth, dtype);k = jnp.asarray(conductivity, dtype);cs = jnp.asarray(capacity, dtype)
    kc = dt[..., None]/(dz*cs) if hasattr(dt, 'ndim') and dt.ndim else dt/(dz*cs)
    kf = jnp.concatenate(((2*k[0]/dz[0])[None], 2*k[1:n]/(dz[1:]+dz[:-1])))
    k1 = kc*kf;k2 = kc[..., :-1]*kf[1:]*cs[:-1]/cs[1:]
    gr = R(fraction)
    lower = jnp.concatenate((jnp.zeros_like(k1[..., :1]), -k1[..., 1:]), axis=-1)
    upper = jnp.concatenate((-k2, jnp.zeros_like(k1[..., :1])), axis=-1)
    diag = jnp.concatenate(((1+k2[..., 0])[..., None], 1.+k1[..., 1:-1]+k2[..., 1:], (1.+k1[..., -1])[..., None]), axis=-1)
    i = 4
    diag = diag.at[..., i].set(1+k1[..., i]+(1-gr)*k2[..., i])
    upper = upper.at[..., i].set(-k2[..., i]*(1-gr))
    rhs = temp.at[..., 0].set(temp[..., 0]+flux_inner*kc[..., 0])
    rhs = rhs.at[..., i].set(temp[..., i]+(gr*hfgr*dt)/R(jnp.asarray(depth, dtype)[i]))
    rhs = rhs.at[..., -1].set(temp[..., -1]+flux_outer*kc[..., -1])
    return _invert(lower, diag, upper, rhs)
