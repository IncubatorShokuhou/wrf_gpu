"""Synthetic IGRA2 v2 sounding for the CPU loop proof (no real observation is read).

Profile = the case wrfinput column at the domain centre, TEMP + OFFSET K, written in the IGRA2 v2 fixed
columns that ``letkf.obs.read_igra2`` parses, with the nominal time VALID (YYYY-MM-DD_HH:MM:SS).
usage: JAX_PLATFORMS=cpu python make_sonde.py WRFINPUT VALID OFFSET_K OUT
"""
import sys
from datetime import datetime

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

wrfinput, valid, offset, out = sys.argv[1], datetime.strptime(sys.argv[2], "%Y-%m-%d_%H:%M:%S"), float(sys.argv[3]), sys.argv[4]
with Dataset(wrfinput) as ds:
    ds.set_auto_mask(False)
    j, i = ds["T"].shape[2] // 2, ds["T"].shape[3] // 2
    theta = ds["T"][0, :, j, i] + 300.0
    p = ds["P"][0, :, j, i] + ds["PB"][0, :, j, i]
    qv = ds["QVAPOR"][0, :, j, i]
    u = 0.5 * (ds["U"][0, :, j, i] + ds["U"][0, :, j, i + 1])
    v = 0.5 * (ds["V"][0, :, j, i] + ds["V"][0, :, j + 1, i])
    phf = (ds["PH"][0, :, j, i] + ds["PHB"][0, :, j, i]) / 9.81
    z = 0.5 * (phf[1:] + phf[:-1])
    lat, lon = float(ds["XLAT"][0, j, i]), float(ds["XLONG"][0, j, i])
temp = theta * (p / 1.0e5) ** (287.0 / 1004.5) + offset
e = qv * p / (0.622 + qv)
td = 243.5 * np.log(e / 611.2) / (17.67 - np.log(e / 611.2)) + 273.15
keep = (p > 25000.0) & (p < 95000.0)
rows = []
for k in np.flatnonzero(keep)[::2]:
    wspd = float(np.hypot(u[k], v[k]))
    wdir = float((np.degrees(np.arctan2(-u[k], -v[k])) + 360.0) % 360.0)
    # IGRA2 v2 fixed columns (obs.read_igra2): ETIME 4-8, PRESS 10-15, GPH 17-21, TEMP 23-27, RH 29-33,
    # DPDP 35-39, WDIR 41-45, WSPD 47-51 (1-based); -9999 missing.
    rows.append(f"20 {-9999:5d} {int(round(p[k])):6d} {int(round(z[k])):5d} {int(round((temp[k] - 273.15) * 10)):5d} "
                f"{-9999:5d} {int(round((temp[k] - offset - td[k]) * 10)):5d} {int(round(wdir)):5d} {int(round(wspd * 10)):5d}")
fixed = rows
header = "#%-11s %4d %02d %02d %02d %4s %4d %-8s %-8s %7d %8d" % (
    "SWZ00000001", valid.year, valid.month, valid.day, valid.hour, "9999", len(fixed), "synth", "synth",
    int(round(lat * 1e4)), int(round(lon * 1e4)))
with open(out, "w") as handle:
    handle.write(header + "\n" + "\n".join(fixed) + "\n")
print(out, len(fixed), "levels", lat, lon)
