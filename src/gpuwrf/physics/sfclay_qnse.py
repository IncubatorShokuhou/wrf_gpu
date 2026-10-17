"""Standalone REAL32/64 QNSE surface column from WRF module_sf_qnsesfc.F.

Selected by the mandatory operational bl=4/sf=4 pair. Host-only cached tables follow WRF's sequential
REAL zeta recurrence; no cached JAX arrays or import-time device allocation.
Scalar iterations are unrolled within a column; GPU costs unmeasured.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.pbl_qnse import _constants
from gpuwrf.physics._qnse_levels import ColumnLevels


@lru_cache(maxsize=2)
def _tables(dtype):
    # WRF's sequential REAL initializer and scalar libm differ from vector
    # NumPy by up to several ULP. This small, source-hashed immutable asset is
    # exported by proofs/s2_qnse/build_surface_tables.py from unmodified WRF.
    mode = 'fp32' if np.dtype(dtype) == np.float32 else 'fp64'
    with np.load(Path(__file__).with_name('qnse_surface_tables_v1.npz')) as data:
        pm, ph = data[mode+'_psim'], data[mode+'_psih']
        dzeta, maximum = data[mode+'_meta']
    pm.flags.writeable = ph.flags.writeable = False
    return pm, ph, dzeta, maximum


@lru_cache(maxsize=2)
def _surface_constants(dtype):
    f = np.dtype(dtype).type
    p = dict(_constants(dtype))
    for k,v in dict(VKARMAN=.4,EXCML=.0001,EXCMS=.0001,USTC=.7,USTR=.225,
                    EPSU2=1.e-6,EPSUST=1.e-9,EPSZT=1.e-28,SQVISC=258.2,
                    A2S=17.2693882,A3S=273.16,A4S=35.86,A2=17.2693882,A3=273.16,A4=35.86,
                    XLV=2.5e6,PQ0=379.90516,P1000MB=1.e5).items():
        p[k] = f(v)
    rv = f(1.)/f(1.5e-5); rt = f(1.)/f(2.1e-5)
    cziv = f(.35)*f(30.)
    p.update(USTFC=f(.018)/p['G'],RVISC=rv,RTVISC=rt,RQVISC=rt,
             FZU1=cziv*f(1.5e-5), FZT1=rv*f(2.1e-5)*f(.84),
             FZT2=cziv*(f(10.)/f(30.))*f(2.1e-5)*f(.84),
             FZQ1=rt*f(2.1e-5)*(f(.84)/f(.84)),FZQ2=rt*f(2.1e-5)*(f(.84)/f(.84)),
             WWST2=f(1.2)*f(1.2),RCAP=f(1.)/p['CAPA'],
             GOCP02=p['G']/p['CP']*f(2.),GOCP10=p['G']/p['CP']*f(10.))
    _, _, dzeta, maximum = _tables(dtype)
    p.update(SFC_ZTMIN1=f(-5.),SFC_ZTMIN2=f(-5.),SFC_ZTMAX1=maximum,SFC_ZTMAX2=maximum,
             SFC_DZETA1=dzeta,SFC_DZETA2=dzeta,KZTM2=9999)
    return SimpleNamespace(**p)


def _interp_psi(table,zeta,ztmin,dzeta):
    rz=(zeta-ztmin)/dzeta
    k=rz.astype(jnp.int32)
    rdzt=rz-k.astype(zeta.dtype)
    k=jnp.clip(k,0,9999)
    return (table[k+1]-table[k])*rdzt+table[k]


def _qnse_pblh(q2,dz,ht):
    z=[ht]
    for k in range(dz.shape[0]):
        z.append(z[-1]+dz[k])
    height=z[-1]-ht
    threshold=np.dtype(q2.dtype).type(.01)*np.dtype(q2.dtype).type(1.01)
    found=jnp.asarray(False)
    for k in range(1,dz.shape[0]):
        use=(2.*q2[k] <= threshold) & (found == False)
        height=jnp.where(use,z[k+1]-ht,height)
        found=found|use
    return height


def _sfcdif_column(
    *,
    seamask, ths, qs_in, psfc, tz0, tsk, thz0_in, qz0_in, uz0_in, vz0_in,
    ustar_in, z0_in, z0base, akhs_in, akms_in, ulow, vlow, tlow, thlow,
    thelow, qlow, cwmlow, zsl, plow, pblh, wetm, ht, itimestep, hfx_in, qfx_in, scm_force_flux,
    _tables_input=None,
):
    """Transcription of WRF ``SFCDIF`` for first and ordinary steps.

    Returns a dict of every SFCDIF output: exchange coeffs (AKHS/AKMS),
    surface roughness Z0, fluxes (HFX/QFX/LH/FLHC/FLQC), updated INOUT surface
    state (USTAR/THZ0/QZ0/UZ0/VZ0/QSFC), RMOL/RIB, and 2m/10m diagnostics.
    The land/sea diagnostic block reads the last iteration's geometry directly.
    """

    C = _surface_constants(str(thlow.dtype))
    if _tables_input is None:
        _PSIM, _PSIH, _, _ = _tables(str(thlow.dtype))
        _PSIM = jnp.asarray(_PSIM, dtype=thlow.dtype)
        _PSIH = jnp.asarray(_PSIH, dtype=thlow.dtype)
    else:
        _PSIM, _PSIH = _tables_input
    rdz = 1.0 / zsl
    cxchl = C.EXCML * rdz
    cxchs = C.EXCMS * rdz
    btgx = C.G / thlow
    elfc = C.VKARMAN * btgx
    btgh = jnp.where(pblh > 1000.0, btgx * pblh, btgx * 1000.0)
    is_sea = seamask > 0.5
    cxch = jnp.where(is_sea, cxchs, cxchl)
    ztmin = jnp.where(is_sea, C.SFC_ZTMIN1, C.SFC_ZTMIN2)
    ztmax = jnp.where(is_sea, C.SFC_ZTMAX1, C.SFC_ZTMAX2)
    fhh = jnp.asarray(.72, thlow.dtype)

    thz0_land = ths
    qs_in = jnp.where((~is_sea) & (itimestep == 1), qlow, qs_in)
    qz0_land = qs_in
    tem_l = (tlow + tz0) * 0.5
    thm_l = (thelow + thz0_land) * 0.5
    a_l = thm_l * C.P608
    b_l = (C.ELOCP / tem_l - 1.0 - C.P608) * thm_l
    dthv_l = ((thelow - thz0_land) * ((qlow + qz0_land + cwmlow) * (0.5 * C.P608) + 1.0)
              + (qlow - qz0_land + cwmlow) * a_l + cwmlow * b_l)
    du2_l = jnp.maximum((ulow - 0.0) ** 2 + (vlow - 0.0) ** 2, C.EPSU2)
    rib_l = btgx * dthv_l * zsl / du2_l
    zu_l = z0_in
    zslu_l = zsl + zu_l
    rzsu_l = zslu_l / zu_l
    rlogu_l = jnp.log(rzsu_l)
    zslt_l = zsl + zu_l
    zilfc = -0.1 * C.VKARMAN * C.SQVISC
    zzil = jnp.where(dthv_l > 0., zilfc*jnp.maximum(9.e-6*ht**2., 3.), zilfc)

    def body(carry, _):
        (ustar, uz0, vz0, thz0, qz0, qs, akhs, akms, z0) = carry

        first_smooth = is_sea & (ustar < C.USTR) & (itimestep == 1)
        akhs = jnp.where(first_smooth, cxchs, akhs)
        akms = jnp.where(first_smooth, cxchs, akms)
        qs = jnp.where(first_smooth, qlow, qs)
        z0_sea = jnp.maximum(C.USTFC * ustar * ustar, 1.59e-5)
        zu_a = C.FZU1 * jnp.sqrt(jnp.sqrt(z0_sea * ustar * C.RVISC)) / ustar
        wght = akms * zu_a * C.RVISC
        rwgh = wght / (wght + 1.0)
        uz0_a = (ulow * rwgh + uz0) * 0.5
        vz0_a = (vlow * rwgh + vz0) * 0.5
        zt_a = C.FZT1 * zu_a
        zq_a = C.FZQ1 * zt_a
        wghtt_a = akhs * zt_a * C.RTVISC
        wghtq_a = akhs * zq_a * C.RQVISC
        thz0_a = ((wghtt_a * thlow + ths) / (wghtt_a + 1.0) + thz0) * 0.5
        qz0_a = ((wghtq_a * qlow + qs) / (wghtq_a + 1.0) + qz0) * 0.5
        thz0_a = jnp.where(itimestep > 1, thz0_a, (wghtt_a*thlow+ths)/(wghtt_a+1.))
        qz0_a = jnp.where(itimestep > 1, qz0_a, (wghtq_a*qlow+qs)/(wghtq_a+1.))
        zu_b = z0_sea
        zt_b = C.FZT2 * jnp.sqrt(jnp.sqrt(z0_sea * ustar * C.RVISC)) / ustar
        zq_b = C.FZQ2 * zt_b
        wghtt_b = akhs * zt_b * C.RTVISC
        wghtq_b = akhs * zq_b * C.RQVISC
        thz0_b = ((wghtt_b * thlow + ths) / (wghtt_b + 1.0) + thz0) * 0.5
        qz0_b = ((wghtq_b * qlow + qs) / (wghtq_b + 1.0) + qz0) * 0.5
        thz0_b = jnp.where(itimestep > 1, thz0_b, (wghtt_b*thlow+ths)/(wghtt_b+1.))
        qz0_b = jnp.where(itimestep > 1, qz0_b, (wghtq_b*qlow+qs)/(wghtq_b+1.))
        zu_c = z0_sea
        zt_c = z0_sea
        thz0_c = ths
        qz0_c = qs

        below_ustr = ustar < C.USTR
        below_ustc = ustar < C.USTC
        sel_a = below_ustc & below_ustr
        sel_b = below_ustc & (~below_ustr)

        zu_sea = jnp.where(sel_a, zu_a, jnp.where(sel_b, zu_b, zu_c))
        zt_sea = jnp.where(sel_a, zt_a, jnp.where(sel_b, zt_b, zt_c))
        uz0_sea = jnp.where(sel_a, uz0_a, jnp.zeros_like(uz0))
        vz0_sea = jnp.where(sel_a, vz0_a, jnp.zeros_like(vz0))
        thz0_sea = jnp.where(sel_a, thz0_a, jnp.where(sel_b, thz0_b, thz0_c))
        qz0_sea = jnp.where(sel_a, qz0_a, jnp.where(sel_b, qz0_b, qz0_c))

        tem_s = (tlow + tz0) * 0.5
        thm_s = (thelow + thz0_sea) * 0.5
        a_s = thm_s * C.P608
        b_s = (C.ELOCP / tem_s - 1.0 - C.P608) * thm_s
        dthv_s = ((thelow - thz0_sea) * ((qlow + qz0_sea + cwmlow) * (0.5 * C.P608) + 1.0)
                  + (qlow - qz0_sea + cwmlow) * a_s + cwmlow * b_s)
        du2_s = jnp.maximum((ulow - uz0_sea) ** 2 + (vlow - vz0_sea) ** 2, C.EPSU2)
        rib_s = btgx * dthv_s * zsl / du2_s
        rzsu_s = (zsl + zu_sea) / zu_sea
        rzst_s = (zsl + zt_sea) / zt_sea
        rlogu_s = jnp.log(rzsu_s)
        rlogt_s = jnp.log(rzst_s)

        zt_land = jnp.maximum(jnp.exp(zzil * jnp.sqrt(ustar * z0base)) * z0base, C.EPSZT)
        zt_land = jnp.maximum(zt_land, zu_l*.1)
        rzst_l = zslt_l / zt_land
        rlogt_l = jnp.log(rzst_l)

        zu = jnp.where(is_sea, zu_sea, zu_l)
        zt = jnp.where(is_sea, zt_sea, zt_land)
        uz0_new = jnp.where(is_sea, uz0_sea, jnp.zeros_like(uz0))
        vz0_new = jnp.where(is_sea, vz0_sea, jnp.zeros_like(vz0))
        thz0_new = jnp.where(is_sea, thz0_sea, thz0_land)
        qz0_new = jnp.where(is_sea, qz0_sea, qz0_land)
        dthv = jnp.where(is_sea, dthv_s, dthv_l)
        du2 = jnp.where(is_sea, du2_s, du2_l)
        rib = jnp.where(is_sea, rib_s, rib_l)
        rzsu = jnp.where(is_sea, rzsu_s, rzsu_l)
        rzst = jnp.where(is_sea, rzst_s, rzst_l)
        rlogu = jnp.where(is_sea, rlogu_s, rlogu_l)
        rlogt = jnp.where(is_sea, rlogt_s, rlogt_l)
        zslu = zsl + zu
        zslt = jnp.where(is_sea, zsl + zt, zslt_l)

        rlmo = elfc * akhs * dthv / ustar ** 3
        zetalu = jnp.clip(zslu * rlmo, ztmin, ztmax)
        zetalt = jnp.clip(zslt * rlmo, ztmin, ztmax)
        zetau = jnp.clip(zu * rlmo, ztmin / rzsu, ztmax / rzsu)
        zetat = jnp.clip(zt * rlmo, ztmin / rzst, ztmax / rzst)

        psmz = _interp_psi(_PSIM, zetau, C.SFC_ZTMIN1, C.SFC_DZETA1)
        psmzl = _interp_psi(_PSIM, zetalu, C.SFC_ZTMIN1, C.SFC_DZETA1)
        simm = psmzl - psmz + rlogu
        pshz = _interp_psi(_PSIH, zetat, C.SFC_ZTMIN1, C.SFC_DZETA1)
        pshzl = _interp_psi(_PSIH, zetalt, C.SFC_ZTMIN1, C.SFC_DZETA1)
        simh = (pshzl - pshz + rlogt) * fhh

        ustark = ustar * C.VKARMAN
        akms_new = jnp.maximum(ustark / simm, cxch)
        akhs_new = jnp.maximum(ustark / simh, cxch)
        wstar2 = jnp.where(dthv <= 0.0, C.WWST2 * jnp.abs(btgh * akhs_new * dthv) ** (2.0 / 3.0), jnp.zeros_like(dthv))
        ustar_new = jnp.maximum(jnp.sqrt(akms_new * jnp.sqrt(du2 + wstar2)), C.EPSUST)

        thz0_new = jnp.where(is_sea, thz0_new, hfx_in/(akhs_new*(plow/(C.R_D*tlow))*C.CP)+thlow)
        carry_out = (ustar_new, uz0_new, vz0_new, thz0_new, qz0_new, qs,
                     akhs_new, akms_new, z0_sea)
        diag = (zu, zt, rlmo, psmz, pshz, rib, du2, wstar2, ustark)
        return carry_out, diag

    carry = (ustar_in, uz0_in, vz0_in, thz0_in, qz0_in, qs_in,
             akhs_in, akms_in, z0_in)
    for _ in range(5):
        carry, diag = body(carry, None)
    (ustar, uz0, vz0, thz0, qz0, qs, akhs, akms, z0_sea_final) = carry
    z0 = jnp.where(is_sea, z0_sea_final, z0_in)
    zu, zt, rlmo, psmz, pshz, rib, _, _, ustark = diag

    umflx = akms * (ulow - uz0)
    vmflx = akms * (vlow - vz0)
    hsflx = akhs * (thlow - thz0)
    hlflx = akhs * (qlow - qz0)

    zu10 = zu + 10.0
    zt02 = zt + 2.0
    zt10 = zt + 10.0
    rlnu10 = jnp.log(zu10 / zu)
    rlnt02 = jnp.log(zt02 / zt)
    rlnt10 = jnp.log(zt10 / zt)
    ztau10 = jnp.clip(zu10 * rlmo, ztmin, ztmax)
    ztat02 = jnp.clip(zt02 * rlmo, ztmin, ztmax)
    ztat10 = jnp.clip(zt10 * rlmo, ztmin, ztmax)

    psm10 = _interp_psi(_PSIM, ztau10, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simm10 = psm10 - psmz + rlnu10
    psh02 = _interp_psi(_PSIH, ztat02, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simh02 = (psh02 - pshz + rlnt02) * fhh
    psh10 = _interp_psi(_PSIH, ztat10, C.SFC_ZTMIN1, C.SFC_DZETA1)
    simh10 = (psh10 - pshz + rlnt10) * fhh

    akms10 = jnp.maximum(ustark / simm10, cxch)
    akhs02 = jnp.maximum(ustark / simh02, cxch)
    akhs10 = jnp.maximum(ustark / simh10, cxch)

    u10 = umflx / akms10 + uz0
    v10 = vmflx / akms10 + vz0
    th02 = hsflx / akhs02 + thz0
    th10 = hsflx / akhs10 + thz0
    q02 = hlflx / akhs02 + qz0
    q10 = hlflx / akhs10 + qz0
    pshltr = psfc * jnp.exp(-0.068283 / tlow)

    zuuz = jnp.minimum(zu * 0.50, 0.18)
    zu_e = jnp.maximum(zu * 0.35, zuuz)
    zu10_e = zu_e + 10.0
    rlnu10_e = jnp.log(zu10_e / zu_e)
    ztau10_e = jnp.clip(zu10_e * rlmo, C.SFC_ZTMIN2, C.SFC_ZTMAX2)
    psm10_e = _interp_psi(_PSIM, ztau10_e, C.SFC_ZTMIN2, C.SFC_DZETA2)
    simm10_e = psm10_e - psmz + rlnu10_e
    ekms10 = jnp.maximum(ustark / simm10_e, cxchl)
    u10e = jnp.where(is_sea, u10, umflx / ekms10 + uz0)
    v10e = jnp.where(is_sea, v10, vmflx / ekms10 + vz0)

    rlow = plow / (C.R_D * tlow)
    chs = akhs
    chs2 = akhs02
    cqs2 = akhs02
    hfx = jnp.where(scm_force_flux == 0, -rlow*C.CP*hsflx, hfx_in)
    qfx = jnp.where(scm_force_flux == 0, -rlow*hlflx*wetm, qfx_in)
    flx_lh = C.XLV * qfx
    flhc = rlow * C.CP * akhs
    flqc = rlow * akhs * wetm
    qgh = ((1.0 - seamask) * C.PQ0 + seamask * C.PQ0SEA) / plow * jnp.exp(C.A2S * (tlow - C.A3S) / (tlow - C.A4S))
    qgh = qgh / (1.0 - qgh)
    cpm = C.CP * (1.0 + 0.8 * qlow)

    qs_sea = qlow+qfx/(rlow*akhs)
    qs_sea = qs_sea / (1.0 - qs_sea)
    qsfc = jnp.where(is_sea, qs_sea, qs)

    return {
        "ustar": ustar, "znt": z0, "akhs": akhs, "akms": akms, "rmol": rlmo,
        "rib": rib, "chs": chs, "chs2": chs2, "cqs2": cqs2, "hfx": hfx,
        "qfx": qfx, "flx_lh": flx_lh, "flhc": flhc, "flqc": flqc, "qgh": qgh,
        "cpm": cpm, "ct": jnp.zeros_like(ustar), "qsfc": qsfc, "thz0": thz0,
        "qz0": qz0, "uz0": uz0, "vz0": vz0, "u10": u10, "v10": v10, "th02": th02,
        "th10": th10, "tshltr": th02, "q02": q02, "qshltr": q02, "q10": q10,
        "pshltr": pshltr, "u10e": u10e, "v10e": v10e,
        "akhs02": akhs02, "akhs10": akhs10, "akms10": akms10,
    }


def qnsesfc_column(
    *, u, v, temperature, theta, qv, qc, p_mid, dz, q2, tsk, xland, z0base,
    psfc, znt, ustar, mavail, dt=60.0, qsfc=None, thz0=None, qz0=None,
    uz0=0.0, vz0=0.0, pblh=1000.0, ht=0., itimestep=2, akhs=0., akms=0., hfx=0., qfx=0., scm_force_flux=0,
    _column_kernel=False, _tables_input=None,
):
    """Run QNSESFC for LOWLYR=1, preserving optional forced-flux semantics.

    ``u/v/temperature/theta/qv/qc/p_mid/dz/q2`` are length-``nz`` columns ordered
    bottom-up (index 0 = lowest model layer). ``q2`` is TKE (m^2/s^2); QNSESFC
    converts to ``2*q2``. Surface scalars are floats/0-d arrays. ``thz0/qz0/qsfc``
    default to lowest-layer values (warm-start seed) when not supplied.
    """

    dtype = u.dtype if _column_kernel else jnp.asarray(u).dtype
    if dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
        raise ValueError('QNSE surface profiles must be float32 or float64')
    C = _surface_constants(str(dtype))
    f = lambda a: a if isinstance(a,ColumnLevels) else jnp.asarray(a, dtype)
    u = f(u); v = f(v); temperature = f(temperature); theta = f(theta)
    qv = f(qv); qc = f(qc); p_mid = f(p_mid); dz = f(dz); q2 = f(q2)
    if u.ndim != (2 if _column_kernel else 1) or u.shape[0] < 3:
        raise ValueError('QNSE surface requires a column with n >= 3')
    if any(a.shape != u.shape for a in (v,temperature,theta,qv,qc,p_mid,dz,q2)):
        raise ValueError('QNSE surface profile shapes must match')

    pblh = _qnse_pblh(q2, dz, f(ht))

    tlow = temperature[0]
    thlow = theta[0]
    ratiomx = qv[0]
    qlow = ratiomx / (1.0 + ratiomx)
    cwmlow = qc[0]
    thelow = (cwmlow * (-C.ELOCP / tlow) + 1.0) * thlow
    ulow = u[0]
    vlow = v[0]
    plow = p_mid[0]
    zsl = dz[0] * 0.5
    psfc_v = f(psfc)
    apesfc = (psfc_v * 1.e-5) ** C.CAPA
    thsk = f(tsk) / apesfc
    seamask = f(xland) - 1.0
    tz0 = (f(thz0) if thz0 is not None else thlow) * apesfc

    thz0_in = f(thz0) if thz0 is not None else thlow
    qz0_in = f(qz0) if qz0 is not None else qlow
    qsfc_in = f(qsfc) if qsfc is not None else qv[0]

    ustar = jnp.where(itimestep == 1, f(.1), f(ustar))
    if scm_force_flux is None:
        scm_force_flux = 1
    out = _sfcdif_column(
        seamask=seamask, ths=thsk, qs_in=qsfc_in, psfc=psfc_v, tz0=tz0,
        tsk=f(tsk), thz0_in=thz0_in, qz0_in=qz0_in, ustar_in=f(ustar),
        z0_in=f(znt), z0base=f(z0base), akhs_in=f(akhs), akms_in=f(akms),
        uz0_in=f(uz0), vz0_in=f(vz0),
        ulow=ulow, vlow=vlow, tlow=tlow, thlow=thlow, thelow=thelow,
        qlow=qlow, cwmlow=cwmlow, zsl=zsl, plow=plow, pblh=pblh,
        wetm=f(mavail), ht=f(ht), itimestep=itimestep, hfx_in=f(hfx), qfx_in=f(qfx), scm_force_flux=scm_force_flux,
        _tables_input=_tables_input,
    )

    rapa = apesfc
    th02p = out["tshltr"]
    th10p = out["th10"]
    rapa02 = rapa - C.GOCP02 / th02p
    rapa10 = rapa - C.GOCP10 / th10p
    t02p = th02p * rapa02
    t10p = th10p * rapa10
    p02p = (rapa02 ** C.RCAP) * C.P1000MB
    p10p = (rapa10 ** C.RCAP) * C.P1000MB
    qs02 = C.PQ0 / p02p * jnp.exp(C.A2 * (t02p - C.A3) / (t02p - C.A4))
    qs10 = C.PQ0 / p10p * jnp.exp(C.A2 * (t10p - C.A3) / (t10p - C.A4))
    qshltr = jnp.minimum(out["qshltr"], qs02)
    q10 = jnp.minimum(out["q10"], qs10)
    out["qshltr"] = qshltr
    out["q10"] = q10
    out["q02"] = qshltr / (1.0 - qshltr)        # specific humidity -> mixing ratio
    out["t02"] = out["th02"] * apesfc
    out["pblh"] = pblh
    return out
