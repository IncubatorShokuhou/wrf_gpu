"""w41-swddir (b): slope_rad / topo_shading surface-flux error caused by the released direct/diffuse mis-split [I].

WRF (module_radiation_driver.F :3042-3051, module_surface_driver.F TOPO_RAD_ADJ :7148-7215):
    diffuse_frac = min(SWDDIF / SWDOWN, 1)  (SWDOWN > 0.001),   SWNORM = SWDOWN * corr_fac,
    corr_fac = diffuse_frac + (1 - diffuse_frac) * csza_slp / csza   (shadow: csza_slp = 0 -> corr_fac = diffuse_frac)
=>  SWNORM = SWDDIF + r * SWDDIR, r = csza_slp / csza, and at fixed SWDOWN a direct-beam error dDIR moves SWNORM by
    dSWNORM = dDIR * (r - 1)   (shaded cells: -dDIR).
Operands: the 186 real WN3 0227 d02/d03 RE01 columns, WRF truth = pristine RRTMG_SWRAD (AE01 arms 0 and 6 = aer_opt 0 at
the column's own date and at julian 196.6), released port = w41-aerosol AE03 port_release.npz (REAL entry + fused band
kernel, CPU interpret, pre-fix direct product).  Slope geometry is a sweep (slope 10/20/30 deg x 24 azimuths, the
column's own latitude / declination / hour angle at the radiation call) because the columns' own terrain is not in the
fixture: the result is an [I] estimate of what slope_rad/topo_shading do with the mis-split, not a paired run.

usage: python slope_estimate.py OUT_JSON [PORT_NPZ [REF [PREFIX]]]
  PREFIX: npz key prefix, e.g. kernel_scaled_direct_ for tests/v025/w41_swddir/swddir_oracle_runner.py output.
  PORT_NPZ keys surface_direct_<arm> / surface_down_<arm>; REF = AE01 (default: WRF truth arms 0 and 6, RE01 arm-A
  wiring) or E (RE01 arm E = Thompson radii + random overlap, own date only; matches the archived AE03 npz, whose
  solve ran with maximum-random overlap OFF: its SWDOWN equals E to 1.7 W/m2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AE01 = ROOT / "tests/v025/w41_aerosol/fixtures/tegen_sw_0227_v1.npz"
RE01 = ROOT / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
PORT = Path("<USER_HOME>/wrf_gpu2_lanes/w41-aerosol/AE01/port_release.npz")
DEG = np.pi / 180.0


def hour_angle(cz, lat, dec, tloctm):
    """|hrang| from the call's coszen (calc_coszen inverted), sign from local solar time."""
    c = np.clip((cz - np.sin(lat) * np.sin(dec)) / (np.cos(lat) * np.cos(dec)), -1.0, 1.0)
    return np.arccos(c) * np.where(tloctm >= 12.0, 1.0, -1.0)


def csza_slope(lat, dec, h, slope, azi):
    """TOPO_RAD_ADJ :7196-7201 (literal formula), floored at 1e-4 -> 0."""
    c = ((np.sin(lat) * np.cos(h)) * (-np.cos(azi) * np.sin(slope)) - np.sin(h) * (np.sin(azi) * np.sin(slope))
         + (np.cos(lat) * np.cos(h)) * np.cos(slope)) * np.cos(dec) \
        + (np.cos(lat) * (np.cos(azi) * np.sin(slope)) + np.sin(lat) * np.cos(slope)) * np.sin(dec)
    return np.where(c <= 1e-4, 0.0, c)


def stats(x):
    x = np.abs(np.asarray(x, np.float64))
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()), "n": int(x.size)}


def main(out, port_path=PORT, ref="AE01", prefix=""):
    ae, re = np.load(AE01), np.load(RE01)
    port = np.load(port_path)
    cat = re["category"]
    cloudy = cat != "clear"
    lat = re["in_xlat"].astype(np.float64) * DEG
    result = {"inputs": {"ae01": str(AE01), "re01": str(RE01), "port_prefix": str(port_path), "reference": ref,
                         "key_prefix": prefix},
              "arms": {}}
    for arm in ((0, 6) if ref == "AE01" else (0,)):
        cz = ae["coszen"][:, arm].astype(np.float64)
        dec = ae["declin"][:, arm].astype(np.float64)
        day = cz > 0.05
        if ref == "AE01":
            wrf_dir, wrf_dn = ae["swddir"][:, arm].astype(np.float64), ae["swdnb"][:, arm].astype(np.float64)
        else:
            assert np.array_equal(ae["coszen"][:, 0], re["col_coszen"])  # AE01 arm 0 = the RE01 radiation call
            wrf_dir, wrf_dn = re[f"{ref}_swddir"].astype(np.float64), re[f"{ref}_swdnb"].astype(np.float64)
        p_dir = port[f"{prefix}surface_direct_{arm}"].astype(np.float64)
        p_dn = port[f"{prefix}surface_down_{arm}"].astype(np.float64)
        d_dir = p_dir - wrf_dir
        # local solar time of the call (sign of the hour angle only): gmt + tau + lon/15
        tloctm = (re["in_gmt"] + re["in_tau"] + re["in_xlong"] / 15.0) % 24.0
        h = hour_angle(cz, lat, dec, tloctm)
        sel = day & cloudy
        row = {"day_columns": int(day.sum()), "day_cloudy_columns": int(sel.sum()),
               "dSWDDIR_day_cloudy_Wm2": stats(d_dir[sel]), "dSWDDIR_day_clear_Wm2": stats(d_dir[day & ~cloudy]),
               "dSWDDIR_signed_mean_day_cloudy": float(d_dir[sel].mean()),
               "wrf_diffuse_frac_day_cloudy_median": float(np.median(1 - wrf_dir[sel] / wrf_dn[sel])),
               "port_diffuse_frac_day_cloudy_median": float(np.median(1 - p_dir[sel] / p_dn[sel])),
               "dSWDOWN_day_Wm2": stats((p_dn - wrf_dn)[day]),
               "shaded_dSWNORM_Wm2 (topo_shading, = -dSWDDIR)": stats(d_dir[sel]),
               "by_category_max_dSWDDIR": {c: float(np.abs(d_dir[day & (cat == c)]).max()) for c in np.unique(cat[day])},
               "slopes": {}}
        for slope_deg in (10, 20, 30):
            azis = np.arange(0, 360, 15) * DEG
            r = np.stack([csza_slope(lat, dec, h, slope_deg * DEG, a) / cz for a in azis])  # (azi, col)
            d_norm = d_dir[None, :] * (r - 1.0)
            worst = np.abs(d_norm[:, sel]).max(axis=0)  # worst orientation per column
            equator = d_norm[np.argmin(np.abs(azis - np.pi)), sel]  # azi 180: equator-facing (WRF convention)
            pole = d_norm[0, sel]
            row["slopes"][f"{slope_deg}deg"] = {"worst_orientation_Wm2": stats(worst),
                                                "equator_facing_Wm2": stats(equator), "pole_facing_Wm2": stats(pole)}
        result["arms"][str(arm)] = {"julian": float(ae["arm_julian"][arm]), **row}
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    a0 = result["arms"]["0"]
    print(json.dumps({k: a0[k] for k in ("day_cloudy_columns", "dSWDDIR_day_cloudy_Wm2", "dSWDDIR_signed_mean_day_cloudy")}))
    print(json.dumps({s: v["worst_orientation_Wm2"] for s, v in a0["slopes"].items()}))


if __name__ == "__main__":
    main(sys.argv[1], *(Path(p) for p in sys.argv[2:3]), *sys.argv[3:5])
