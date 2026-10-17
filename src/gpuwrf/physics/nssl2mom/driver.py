"""Column driver of the NSSL 2-moment port: ``nssl_2mom_driver`` (module lines 2361-3569), mp=18.

WRF field names (``module_microphysics_driver.F`` CASE (NSSL_2MOM), lines 2237-2303) map onto the
scheme's internal stack ``an`` (Fortran species indices, see ``indices.py``):

    th qv qc qr qi qs | qg -> QH (lh, graupel) | qh -> QHL (lhl, hail)
    qndrop -> CCW (lnc) | qnr -> CRW | qni -> CCI | qns -> CSW | qng -> CHW | qnh -> CHL
    qnn -> CN (activated CCN, lccna; background CCN lccn = constant qccn) | qvolg -> VHW | qvolh -> VHL

All columns are ``(..., nz)`` arrays, k = 0 at the surface; ``rho`` is the density WRF passes as ``dn``:
the DRY ``1./(al+alb)`` of moist_physics_prep_em (module_big_step_utilities_em.F:5542), formed by the
adapter.  Order of operations per call (default WRF config,
loopmax = 1, optional cumulus rates, no chem/elec, ``isedonly = 0``):

  1. itimestep == 1: CN := 0 (cold start of activated CCN, lines 2717-2738)
  2. pack, t0 = th*pii, t00 = 380/p, t77 = pii, Meyers/Ferrier nucleation rate t7 (lines 2988-3078)
  3. density-scale indices lnb..na (#/kg -> #/m3, m3/kg -> m3/m3; lines 3094-3104)
  4. itimestep == 1: calcnfromq                                  -> cleanup.calcnfromq
     cu_used == 1: calcnfromcuten                                -> cleanup.calcnfromcuten
  5. sediment1d + surface precipitation binding (lines 3153-3206)  -> sediment.sediment1d
  6. nssl_2mom_gs (parts A, B, C)                                   -> gs_a / gs_b / gs_c
  7. NUCOND, smallvalues                                             -> nucond / cleanup
  8. diagnostics when ``diag`` (dbz, effective radii)               -> diagnostics
  9. de-scale, unpack (CN = max(0, an(lccna)))
"""

from __future__ import annotations

import jax.numpy as jnp

from .indices import (LT, LV, LC, LR, LI, LS, LH, LHL, LCCN, LNC, LNR, LNI, LNS, LNH, LNHL,
                      LVH, LVHL, LCCNA, NA, Prec)
from . import satfun

# WRF field name -> Fortran species index of the NSSL stack
WRF_TO_AN = {
    "th": LT, "qv": LV, "qc": LC, "qr": LR, "qi": LI, "qs": LS, "qg": LH, "qh": LHL,
    "qndrop": LNC, "qnr": LNR, "qni": LNI, "qns": LNS, "qng": LNH, "qnh": LNHL,
    "qvolg": LVH, "qvolh": LVHL, "qnn": LCCNA,
}
LNB = LCCN  # Max(lh,lhl)+1 (driver line 2812)


def _denscale_indices(C):
    return [il for il in range(LNB, NA + 1) if int(C.denscale[il]) == 1]


def pack(fields: dict, itimestep: int, C, prec: Prec):
    """Steps 1-3: returns (an, aux) with aux = dict(t0, t7, t00, t77, pn, wn, dn, dz)."""
    R = prec.R
    th = jnp.asarray(fields["th"], R)
    shape = th.shape
    cn = jnp.asarray(fields["qnn"], R)
    if itimestep == 1:  # lccna > 1 .and. .not. present(cna): CN zeroed at cold start
        cn = jnp.zeros(shape, R)
    rows = [jnp.zeros(shape, R)] * (NA + 1)
    for name, il in WRF_TO_AN.items():
        if C.ipconc==0 and name=='qnn':
            continue  # Original2900 CCN packing is inside ipconc>=2.
        rows[il] = cn if name == "qnn" else jnp.asarray(fields[name], R)
    if C.ipconc >= 2:
        rows[LCCN] = jnp.full(shape, C.qccn, R)
    pii = jnp.asarray(fields["pii"], R)
    pn = jnp.asarray(fields["p"], R)
    dn = jnp.asarray(fields["rho"], R)
    t0 = th * pii
    t00 = 380.0 / pn
    t77 = pii
    # Meyers/Ferrier primary nucleation rate (icenucopt=1, imeyers5=.false.), lines 2988-3017
    l = satfun.ltemq_index(t0, C, R)
    tq = satfun.tabqvs(l, C, R)
    esb = jnp.asarray(C.esbolton, R)
    t8s = jnp.asarray(C.rdorv, R) * esb * tq / (pn - esb * tq)
    t9s = t00 * satfun.tabqis(l, C, R)
    ssival = jnp.minimum(t8s, jnp.maximum(rows[LV], 0.0)) / t9s
    rho00 = jnp.asarray(1.225, R)
    expo = jnp.minimum(jnp.asarray(57.0, R), 12.96 * (ssival - 1.0) - 0.639)
    dp1 = (dn / rho00 * 1.0e3 * jnp.exp(expo)).astype(jnp.float64)  # DOUBLE dp1 = REAL expression
    t7_val = jnp.minimum(dp1, 1.0e30).astype(R)
    t7 = jnp.where((ssival > 1.0) & (t0 <= 268.15), t7_val, jnp.zeros(shape, R))
    an = jnp.stack(rows)
    for il in _denscale_indices(C):
        an = an.at[il].set(an[il] * dn)
    aux = dict(t0=t0, t7=t7, t00=t00, t77=t77, pn=pn, wn=jnp.asarray(fields["w"], R), dn=dn,
               dz=jnp.asarray(fields["dz"], R))
    return an, aux


def bind_precip(xfall, dn_sfc, dtp, C, prec: Prec):
    """Surface accumulations of one call (driver lines 3153-3206), mm liquid equivalent."""
    R = prec.R
    dtp = jnp.asarray(dtp, R)
    xdr = jnp.asarray(C.xdn0[LR], R)
    a = dtp * dn_sfc
    rainncv = a * (xfall[LR] + xfall[LS] * 1000.0 / xdr + xfall[LH] * 1000.0 / xdr + xfall[LHL] * 1000.0 / xdr)
    snowncv = a * xfall[LS] * 1000.0 / xdr
    grplncv = a * xfall[LH] * 1000.0 / xdr
    hailncv = a * xfall[LHL] * 1000.0 / xdr
    sr = (snowncv + hailncv + grplncv) / (rainncv + 1.0e-12)
    return dict(rainncv=rainncv, snowncv=snowncv, grplncv=grplncv, hailncv=hailncv, sr=sr)


def unpack(an, dn, C, prec: Prec):
    """Steps 9: de-scale and map back to WRF names."""
    an = jnp.asarray(an, prec.R)
    dn = jnp.asarray(dn, prec.R)
    for il in _denscale_indices(C):
        an = an.at[il].set(an[il] / dn)
    out = {name: an[il] for name, il in WRF_TO_AN.items()}
    out["qnn"] = jnp.maximum(0.0, an[LCCNA])
    return out


def liquid_rain(xfall, dn_sfc, dtp, prec: Prec):
    """Liquid-only part of RAINNCV (``dtp*dn(1)*xfall(lr)``) for the port's disjoint rain_acc channel."""
    return jnp.asarray(dtp, prec.R) * dn_sfc * xfall[LR]


def nssl2mom_column(fields: dict, dtp, C, prec: Prec, *, itimestep: int | None = None,
                    first_step=None, diag: bool = False, radii: bool = False, cumulus_tendencies=None,
                    hail_enabled: bool = True, radar_depth=None):
    """One WRF ``nssl_2mom_driver`` call (default mp=18) on columns ``(..., nz)``.

    ``itimestep`` (static) selects the cold-start branch exactly like WRF (==1: CN := 0 and
    calcnfromq); alternatively ``first_step`` (traced bool, scalar) selects it at run time.
    Returns ``(out, precip, diagnostics)``: ``out`` = updated WRF fields (th, q*, numbers, volumes),
    ``precip`` = per-call accumulations (bind_precip + ``rain_liquid``), ``diagnostics`` = dbz and
    effective radii when ``diag``; ``radii`` selects radii alone, independently
    of WRF's radar ``makediag`` flag (has_req block at module lines3356-3386).
    ``cumulus_tendencies`` optionally supplies WRF Q*CUTEN rates (kg/kg/s),
    keyed by qc/qr/qi/qs, after phy_prep_part2; None means cu_used=0.
    """
    from . import cleanup, gs_a, gs_b, gs_c, nucond as nucond_mod, sediment

    R = prec.R
    one_moment=C.ipconc==0
    if one_moment:
        if radii:
            raise NotImplementedError('One-moment NSSL has no effective radii; original has_req flags are0')
        z=jnp.zeros_like(fields['qg'],dtype=R)
        fields={**fields,**{k:z for k in ('qndrop','qnr','qni','qns','qng','qnh','qvolh')}}
        if C.variant==21:
            fields={**fields,'qh':z,'qvolg':z,'qnn':z}
    if not hail_enabled:
        # WRF legacy22 has no QHL/CHL/VHL Registry members. Retain the
        # canonical internal indexing, with compiler-known zero dummy slots.
        # All active REAL constants are identical to default18 (oracle census).
        zero_hail = jnp.zeros_like(fields["qg"], dtype=R)
        fields = {**fields, "qh": zero_hail, "qnh": zero_hail, "qvolh": zero_hail}
    if first_step is None:
        first_step = jnp.asarray(itimestep == 1)
    first_step = jnp.asarray(first_step)
    an_cold, aux = pack(fields, 1, C, prec)
    an_warm, _ = pack(fields, 2, C, prec)
    an = jnp.where(first_step, an_cold, an_warm)  # only CN differs (zeroed at cold start)
    dn, dz = aux["dn"], aux["dz"]
    if not one_moment:
        an = jnp.where(first_step, cleanup.calcnfromq(an, dn, C, prec), an)
    if cumulus_tendencies is not None and not one_moment:
        mass = {name: jnp.asarray(dtp, R) * jnp.asarray(rate, R)
                for name, rate in cumulus_tendencies.items()}
        an = cleanup.calcnfromcuten(an, dn, mass, C, prec)
    an, xfall = sediment.sediment1d(an, aux["t0"], aux["t7"], dn, dz, dtp, C, prec)
    dn_sfc = dn[..., 0]
    precip = bind_precip(xfall, dn_sfc, dtp, C, prec)
    precip["rain_liquid"] = liquid_rain(xfall, dn_sfc, dtp, prec)
    zero = jnp.zeros_like(aux["t0"])
    G = gs_a.gs_part_a(an, aux["t0"], aux["t7"], aux["t00"], aux["t77"], aux["pn"], aux["wn"], dn, dz,
                       dtp, C, prec)
    G = gs_b.gs_part_b(G, C, prec, dtp)
    an, tt = gs_c.gs_part_c(G, an, aux["t0"], zero, zero, zero, zero, zero, zero, aux["t7"], zero, zero,
                            C, prec, dtp, hail_enabled=hail_enabled)
    if one_moment:
        t0=tt[0]  # Original NUCOND10020 jumps to2200 for ipconc<=1.
    else:
        an, t0, _ssat = nucond_mod.nucond(an, dn, aux["t77"], aux["pn"], aux["wn"], dtp, C, prec)
    an, t0 = cleanup.smallvalues(an, t0, dn, aux["wn"], aux["t77"], dtp, C, prec)
    diagnostics = {}
    if diag:
        from . import diagnostics as diag_mod
        radar = diag_mod.radardd02_one_moment if one_moment else diag_mod.radardd02
        diagnostics["dbz"] = radar(an, dn, C, prec)
    elif radar_depth is not None:
        # WRF makediag/ke_diag: first-step bottom level, full history
        # alarms, and no radar work on ordinary radar-only steps. Reuse S5
        # before unpacking; reconstructing it from output loses REAL rounding.
        import jax
        from . import diagnostics as diag_mod
        # ipconc = 0 (legacy mp=19/21) selects radardd02's one-moment intercept branches.
        radardd02 = diag_mod.radardd02_one_moment if one_moment else diag_mod.radardd02
        def active():
            def bottom():
                value=radardd02(an[..., :1],dn[..., :1],C,prec)
                return jnp.concatenate((value,jnp.zeros_like(dn[..., 1:])),axis=-1)
            return jax.lax.cond(radar_depth==dn.shape[-1],
                                lambda:radardd02(an,dn,C,prec),bottom)
        diagnostics["dbz"]=jax.lax.cond(radar_depth>0,active,lambda:jnp.zeros_like(dn))
    if (diag and not one_moment) or radii:
        from . import diagnostics as diag_mod
        re_c, re_i, re_s = diag_mod.eff_radius(an, dn, C, prec)
        diagnostics.update(re_cloud=re_c, re_ice=re_i, re_snow=re_s)
    out = unpack(an, dn, C, prec)
    if one_moment:
        # Original one-moment CN is an external held Registry scalar; it is
        # not packed/scattered through the microphysics density-scaled stack.
        out['qnn']=jnp.where(first_step,jnp.zeros_like(fields['qnn'],dtype=R),jnp.asarray(fields['qnn'],R))
    out = {k: jnp.asarray(v, R) for k, v in out.items()}
    return out, precip, diagnostics
