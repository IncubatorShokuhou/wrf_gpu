"""Aerosol and hail context for the existing single-pass Thompson equations."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import FunctionType, SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gpuwrf.assets import asset_path, ensure_asset

from .native import TC as tc, AERO as aero, real_tables
from . import _coefficients as constants
from . import entry_distribution, volume_tendency, OGG2, OGG3


ASSET_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def collision_tables():
    names = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")
    # > 10 MB each: repo copy in a private checkout, else fetched on first use (gpuwrf.assets, fail-closed).
    paths = [ensure_asset(asset_path(f"src/gpuwrf/physics/thompson_hail/collision_tables_{density}.npz"))
             for density in range(9)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("mp38 requires all original nine-density hail collision assets; no mp8 fallback")
    result = {name: np.empty((37, 37, 9, 37, 37), np.float64) for name in names}
    for density, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            for name in names:
                piece = data[name]
                if piece.shape != (37, 37, 37, 37) or piece.dtype != np.float64:
                    raise ValueError(f"invalid original hail collision asset {path}:{name}")
                result[name][:, :, density, :, :] = piece
                del piece
    for value in result.values():
        value.setflags(write=False)
    return result


def clone(function, **replacements):
    env = dict(function.__globals__)
    env.update(replacements)
    copied = FunctionType(function.__code__, env, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


class HailContext:
    """Per-call arrays; shared Thompson mass/thermal equations remain in tc."""

    def __init__(self, state, tables, cold_tables, *, raw_qv):
        self.state = state
        # Original qv1d is distinct from the private floored working qv;
        # every cumulative REAL reconstruction retains this incoming array.
        self.raw_qv = raw_qv
        self.tables = tables
        self.aero = real_tables()
        # The distribution coefficients are REAL in original WRF, while the
        # freezing/collision table values are explicit DOUBLE.
        coeff = self.aero._replace(**{name: getattr(self.aero, name).astype(jnp.float32)
                                     for name in ("ccg", "cce", "ocg1", "ocg2")})
        self.nc = aero._entry_cloud_number(state.qc, state.Nc, state.rho, coeff, wrf_hail=True).astype(jnp.float32)
        rc, self.nu, lamc, xdc, mvd, active = aero._cloud_distribution_aero(state.qc, self.nc, state.rho, coeff)
        # WRF :1792 initializes zero, then :2169 resets every warm-rate
        # level to D0c before computing the active-cloud distribution.
        # Inactive values must not pass the strict mvd_c>D0c collectors.
        lamc = lamc.astype(jnp.float64)
        factor = np.float32(3.) + (self.nu + 1).astype(jnp.float32) + np.float32(.672)
        mvd = jnp.minimum(np.float32(tc.D0R), jnp.maximum(np.float32(tc.D0C), (factor / lamc).astype(jnp.float32)))
        mvd = jnp.where(active, mvd, np.float32(tc.D0C))
        self.cloud = (rc, lamc.astype(jnp.float64), xdc.astype(jnp.float32), mvd, active)
        self.rain = tc._rain_distribution(state.qr, state.Nr, state.rho, hail_mode="entry")
        self.nwfa = jnp.maximum(np.float32(11.1e6), jnp.minimum(np.float32(9999.e6), state.nwfa * state.rho))
        self.nifa = jnp.maximum(np.float32(5000.), jnp.minimum(np.float32(9999.e6), state.nifa * state.rho))
        self.g = entry_distribution(state.qg, state.Ng, state.qvolg, state.rho)
        hail_tables = {name: jnp.asarray(value, jnp.float64) for name, value in collision_tables().items()}
        self.cold_tables = SimpleNamespace(**{**{name: getattr(cold_tables, name) for name in cold_tables._fields}, **hail_tables})
        self.hail_table_ids = {id(a) for a in hail_tables.values()}
        av = jnp.asarray((45.9173813,67.0867386,98.0158463,122.353378,143.204224,161.794724,178.762115,194.488785,209.225876), jnp.float32)
        self.av = av[self.g.index]
        self.bv = np.float32(.640961647)
        cgg, cge = jnp.asarray(constants.CGG, jnp.float32), jnp.asarray(constants.CGE, jnp.float32)
        self.cgg6 = cgg[self.g.index, 5]
        self.cge9, self.cge11 = cge[self.g.index, 8], cge[self.g.index, 10]
        self.t1_qg_qc = np.float32(tc.PI) * np.float32(.25) * self.av * cgg[self.g.index, 8]
        sc3 = np.float32(.632) ** np.float32(np.float32(1.) / np.float32(3.))
        self.t2_subl = np.float32(.28) * sc3 * jnp.sqrt(self.av) * cgg[self.g.index, 10]
        self.t2_melt = np.float32(tc.PI) * np.float32(4.) * np.float32(tc.C_CUBE) * np.float32(tc._OLFUS32) * self.t2_subl
        self.amg = jnp.asarray(constants.AM_G, jnp.float32)[self.g.index]
        self.density = jnp.asarray(constants.RHO_G, jnp.float32)[self.g.index]
        self.coeff = coeff
        # Original :2083-2088 explicitly integrates the two-gamma number
        # distribution. Its ns is deliberately not the Field smo0 moment.
        snow_active = state.qs > np.float32(tc.R1)
        rs = jnp.where(snow_active, state.qs*state.rho, np.float32(tc.R1))
        smob = rs * (np.float32(1.) / np.float32(tc.AM_S))
        smoc = tc._snow_moment(tables.cse[0], smob, jnp.minimum(np.float32(-.1), state.T-np.float32(273.15)), tables, wrf_hail=True)
        m0 = smob/smoc
        mrat = smob*m0*m0*m0
        slam1, slam2 = m0*np.float32(20.78), m0*np.float32(3.29)
        ns = mrat*np.float32(490.6)/slam1 + mrat*np.float32(17.46)*m0**np.float32(.6357)*np.float32(.8980315327644348)/slam2**np.float32(1.635699987411499)
        self.ns = jnp.where(snow_active, ns, np.float32(0.)).astype(jnp.float32)

    def graupel_distribution(self):
        g = self.g
        return g.rg, g.ng, g.lamg, g.ilamg, g.n0g, g.qg > tc.R1

    def balance_cloud_number(self, state, rc, rho, ncten, dt):
        # Original :2997-3019 balances the REAL tendency itself; deriving it
        # back from an already rounded Nc would lose the original chronology.
        f = np.float32
        xrc = jnp.maximum(f(tc.R1), state.qc * rho)
        xnc = jnp.maximum(f(2.), state.Nc * rho)
        nu_c = jnp.minimum(15, jnp.floor(f(1000.e6) / xnc + f(.5)).astype(jnp.int32) + 2)
        nu = nu_c - 1
        ccg1, ccg2 = self.coeff.ccg[0, nu], self.coeff.ccg[1, nu]
        ocg1, ocg2 = self.coeff.ocg1[nu], self.coeff.ocg2[nu]
        lamc = (xnc * f(tc._am_r()) * ccg2 * ocg1 / rc) ** f(tc.OBMR)
        xdc = (f(3.) + nu_c.astype(jnp.float32) + f(1.)) / lamc.astype(jnp.float64)
        xdc = xdc.astype(jnp.float32)
        low, high = xdc < f(tc.D0C), xdc > f(tc.D0R) * f(2.)
        lamc_edge = jnp.where(low, self.coeff.cce[1, nu] / f(tc.D0C), self.coeff.cce[1, nu] / (f(tc.D0R) * f(2.))).astype(jnp.float64)
        edge_nc = (ccg1 * ocg2 * xrc / f(tc._am_r()) * lamc_edge ** f(3.)).astype(jnp.float32)
        odts, orho = f(1.) / f(dt), f(1.) / rho
        ncten = jnp.where(low | high, (edge_nc - self.state.Nc * rho) * odts * orho, ncten)
        ncten = jnp.where(xrc > f(tc.R1), ncten, -self.state.Nc * odts)
        cap = jnp.maximum(f(0.), (self.state.Nc + ncten * f(dt)) * rho) > f(aero.NT_C_MAX)
        ncten = jnp.where(cap, (f(aero.NT_C_MAX) - self.state.Nc * rho) * odts * orho, ncten).astype(jnp.float32)
        return state.replace(Nc=self.state.Nc + ncten * f(dt)), ncten

    def warm_rates(self, state, dt, tables, odts):
        nu = (self.nu + 1).astype(jnp.float32)
        dc_prefix = (self.coeff.ccg[2, self.nu] * self.coeff.ocg2[self.nu]) ** np.float32(tc.OBMR)
        denominator = np.float32(tc._am_r()) * nu * np.float32(10.) * np.float32(tc.D0R) * np.float32(tc.D0R) * np.float32(tc.D0R)
        wrf = lambda name, old: dc_prefix if name == "DC_G_PREFIX" else denominator if name == "PNR_WAU_DEN" else tc._wrf(name, old)
        fn = clone(tc._wrf_warm_rain_rates, _cloud_distribution=lambda *_: self.cloud,
                   _rain_distribution=lambda *_: self.rain, _wrf=wrf)
        return fn(state, dt, tables, odts, wrf_hail=True)

    def rain_collection(self, state, dt, twet, odts):
        def take4(table, i, j, k, m):
            if id(table) in self.hail_table_ids:
                return table[i, j, self.g.index, k, m]
            return tc._take4(table, i, j, k, m)
        fn = clone(tc._wrf_rain_snow_graupel_rates, _graupel_distribution=lambda *_: self.graupel_distribution(),
                   AM_G_MP8=self.amg, _take4=take4,
                   _rain_distribution=lambda *_: self.rain,
                   _snow_moments=lambda *args: tc._snow_moments(*args, wrf_hail=True))
        return fn(state, dt, self.cold_tables, twet, odts)

    def freezing(self, state, dt, rc, rr, nr, ni, smo0, qvs, qvsi, ssatw, ssati, rate_max, idx_r, idx_r1, idx_tc):
        odts = np.float32(1.) / np.float32(dt)
        tempc = state.T - np.float32(273.15)
        cold = state.T < np.float32(tc.T_0)
        demott = aero._ice_demott(jnp.minimum(tempc, np.float32(-1.e-6)), state.rho, self.nifa).astype(jnp.float32)
        idx_in = aero._idx_in_from_xni(demott)
        qrfz = aero._take_qrfz4(self.aero, idx_r, idx_r1, idx_tc, idx_in)
        rain_table = cold & (rr > tc.R_R_FIRST)
        rain_hom = cold & ~rain_table & (rr > tc.R1) & (state.T < tc.HGFR)
        pri_rfz = jnp.where(rain_table, qrfz[..., 0] * odts, jnp.where(rain_hom, rr * odts, 0.)).astype(jnp.float64)
        prg_rfz = jnp.where(rain_table, qrfz[..., 1] * odts, 0.)
        pni_rfz = jnp.where(rain_table, qrfz[..., 2] * odts, jnp.where(rain_hom, nr * odts, 0.)).astype(jnp.float64)
        pnr_rfz = jnp.where(rain_table, jnp.minimum((nr * odts).astype(jnp.float64), qrfz[..., 3] * odts), 0.)
        idx_c = aero._idx_c_from_rc(rc)
        idx_n = aero._idx_n_from_nc(self.nc, self.aero)
        qcfz = aero._take_qcfz(self.aero, idx_c, idx_n, idx_tc, idx_in)
        cloud_table = cold & (rc > aero.R_C_FIRST)
        cloud_hom = cold & ~cloud_table & (rc > tc.R1) & (state.T < tc.HGFR)
        pri_wfz = jnp.where(cloud_table, jnp.minimum((rc * odts).astype(jnp.float64), qcfz[..., 0] * odts), jnp.where(cloud_hom, rc * odts, 0.)).astype(jnp.float64)
        pni_wfz = jnp.where(cloud_table, jnp.minimum(jnp.minimum((self.nc * odts).astype(jnp.float64), pri_wfz / np.float32(2. * tc.XM0I)), qcfz[..., 1] * odts), jnp.where(cloud_hom, self.nc * odts, 0.)).astype(jnp.float64)
        nuc = cold & ((ssati >= .25) | ((ssatw > tc.EPS) & (state.T < 253.15)))
        xni = (ni + (pni_rfz + pni_wfz) * np.float32(dt)).astype(jnp.float32)
        pni_inu = (np.float32(.5) * (demott - xni + jnp.abs(demott - xni)) * odts).astype(jnp.float64)
        pri_inu = jnp.where(nuc, jnp.minimum(rate_max.astype(jnp.float64), np.float32(tc.XM0I) * pni_inu), 0.)
        pni_inu = jnp.where(nuc, pri_inu / np.float32(tc.XM0I), 0.)
        total_ni = (self.ns + ni + (pni_rfz + pni_wfz + pni_inu) * np.float32(dt)).astype(jnp.float32)
        koop_active = cold & (total_ni <= 999.e3) & (state.T < 238.) & (ssati >= .4)
        koop = aero._ice_koop(state.T, state.qv, qvs, self.nwfa, np.float32(dt), wrf_hail=True).astype(jnp.float32)
        pni_iha = jnp.where(koop_active, koop * odts, 0.).astype(jnp.float64)
        pri_iha = jnp.minimum(rate_max.astype(jnp.float64), np.float32(tc.XM0I * .1) * pni_iha)
        pri_iha = jnp.where(koop_active, pri_iha, 0.)
        pni_iha = pri_iha / np.float32(tc.XM0I * .1)
        return pri_rfz, prg_rfz, pni_rfz, pnr_rfz, pri_wfz, pni_wfz, pri_inu, pni_inu, pri_iha, pni_iha

    def aerosol_tendencies(self, state, dt, *, rc, rr, lamr, mvd_r, n0_r, mvd_c, rg, ilamg, n0_g,
                           rs, xds, smoe, rhof, visco, prr_wau, prr_rcw, prs_scw, prg_gcw,
                           pni_wfz, pni_inu, pni_iha, pnc_scw, pnc_gcw):
        odts = np.float32(1.) / np.float32(dt)
        rain_geom = rhof * np.float32(tc._wrf("T1_QR_QC", tc.T1_QR_QC)) * n0_r * (lamr + np.float32(tc.FV_R)) ** np.float32(-tc.CRE9)
        rain_active = (rr > tc.R1) & (mvd_r > tc.D0R)
        autoconverts = (rc > np.float32(.01e-3)) & (state.qc > tc.R1)
        mvd_auto = jnp.where(autoconverts, mvd_c, np.float32(1.))
        pnc_wau = jnp.where(autoconverts, jnp.minimum((self.nc * odts).astype(jnp.float64),
            prr_wau / (np.float32(tc._am_r()) * mvd_auto * mvd_auto * mvd_auto)), 0.)
        ef_idx_r = jnp.clip(jnp.floor(tc.N_EFRW_R * jnp.log(jnp.maximum(mvd_r, tc.DR_FIRST) / tc.DR_FIRST) / jnp.log(tc.DR_LAST / tc.DR_FIRST)), 0, tc.N_EFRW_R - 1).astype(jnp.int32)
        ef_idx_c = jnp.clip(jnp.floor(mvd_c * np.float32(1.e6)).astype(jnp.int32) - 1, 0, tc.N_EFRW_C - 1)
        ef_rw = tc._take2(self.tables.t_Efrw, ef_idx_r, ef_idx_c).astype(jnp.float32)
        pnc_rcw = jnp.where(rain_active & (mvd_c > tc.D0C), jnp.minimum((self.nc * odts).astype(jnp.float64),
            rhof * np.float32(tc._wrf("T1_QR_QC", tc.T1_QR_QC)) * ef_rw * self.nc * n0_r * (lamr + np.float32(tc.FV_R)) ** np.float32(-tc.CRE9)), 0.)
        scavenge_r = rain_active
        # Original r_s(1)/r_g(1) are both 1e-6 (:293/:301), not the old
        # Aero subset's 1e-5 constants. GDB H38-28 records the omitted sinks.
        scavenge_s = rs > np.float32(1.e-6)
        scavenge_g = rg > np.float32(1.e-6)
        snow_geom = rhof * np.float32(tc._wrf("T1_QS_QC", tc.T1_QS_QC)) * smoe
        hail_geom = rhof * self.t1_qg_qc * n0_g * ilamg ** self.cge9
        def sinks(aerosol, radius):
            er = aero._eff_aero(mvd_r, radius, visco, state.rho, state.T, "r", wrf_hail=True).astype(jnp.float32)
            es = aero._eff_aero(xds, radius, visco, state.rho, state.T, "s", wrf_hail=True).astype(jnp.float32)
            # xDg is REAL and Eff_aero uses the hail-initialized idx_bg1.
            xdg = (np.float32(4.) * ilamg).astype(jnp.float32)
            eg = aero._eff_aero(xdg, radius, visco, state.rho, state.T, "g", wrf_hail=True).astype(jnp.float32)
            # Each original REAL prefix rounds before multiplication by a
            # DOUBLE intercept/slope. Snow's all-REAL expression widens on
            # assignment to its DOUBLE process rate (:2214/:2445/:2464).
            rrate = rhof * np.float32(tc._wrf("T1_QR_QC", tc.T1_QR_QC)) * er * aerosol * n0_r * (lamr + np.float32(tc.FV_R)) ** np.float32(-tc.CRE9)
            srate = (rhof * np.float32(tc._wrf("T1_QS_QC", tc.T1_QS_QC)) * es * aerosol * smoe).astype(jnp.float64)
            grate = rhof * self.t1_qg_qc * eg * aerosol * n0_g * ilamg ** self.cge9
            cap = (aerosol * odts).astype(jnp.float64)
            return (jnp.where(scavenge_r, jnp.minimum(cap, rrate), 0.),
                    jnp.where(scavenge_s, jnp.minimum(cap, srate), 0.),
                    jnp.where(scavenge_g, jnp.minimum(cap, grate), 0.))
        water = sinks(self.nwfa, .04e-6)
        ice = sinks(self.nifa, .8e-6)
        orho = np.float32(1.) / state.rho
        nwfaten = (-(water[0] + water[1] + water[2] + pni_iha) * orho).astype(jnp.float32)
        nifaten = (-(ice[0] + ice[1] + ice[2]) * orho).astype(jnp.float32)
        nifaten = (nifaten.astype(jnp.float64) - pni_inu * orho).astype(jnp.float32)
        ncten = ((-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw) * orho).astype(jnp.float32)
        return ncten, nwfaten, nifaten
