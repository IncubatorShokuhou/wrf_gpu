"""Build real-structured SLUCM oracle regimes (sf_urban_physics=1).

Census first (E154): the WN3 0227 d03 (1 km Tenerife) wrfinput has 116 ISURBAN=13
cells (MODIFIED_IGBP_MODIS_NOAH), FRC_URB2D == 0 everywhere (-> WRF falls back to
FRC_URB_TBL(UTYPE=2)=0.9), no NUDAPT morphology.  Forcing columns are read from the
original CPU-WRF wrfout_d03 of that case at urban cells: day (02-28 13Z, SWDOWN
311-903 W/m2), night (03-01 03Z, HFX -83..29, some rain) and pre-dawn (02-28 06Z).

Fields WRF history does not carry (CHS/CHS2/CQS2/QSFC/ZNT and the Noah-MP
T2MV/T2MB/Q2MV/Q2MB tiles) are synthesised from the same column (declared below);
stress regimes force the remaining branches: rain > 1 mm/h (BETR/BETG=0.7),
calm wind (UA<1 clamp), CHS/CQS2 below the 0.01 limit, strong wind (UC>5 Jurges),
and a wrfinput FRC_URB2D in (0,1].

Output: regimes.txt (one name line + one list-directed value line per regime).
Usage: python make_urban_oracle_inputs.py <wrf_run_dir> <out_file>
"""

from __future__ import annotations

import sys

import netCDF4
import numpy as np

FIELDS = (
    "dt nsteps julian julyr frc_in xlat cosz omg declin t3d1 qv1 u1 v1 swdown glw p8w1 p8w2 "
    "rainbl dz8w1 znt tsk hfx qfx lh grdflx albedo emiss qsfc chs chs2 cqs2 ust fveg t2mv t2mb "
    "q2mv q2mb psfc tslb1 tslb2 tslb3 tslb4 tmn vegfra"
).split()

DT = 6.0          # WN3 d03 time step [s]
NSTEPS = 30
CELLS_PER_TIME = 6


def _col(ds, name, j, i):
    return float(np.asarray(ds[name][0])[..., j, i]) if np.asarray(ds[name][0]).ndim == 2 else None


def regime_from_cell(ds, prev, j, i, hour_utc, julian):
    a = lambda n: np.asarray(ds[n][0])
    theta1 = a("T")[0, j, i] + 300.0
    p1 = a("P")[0, j, i] + a("PB")[0, j, i]
    t1 = theta1 * (p1 / 1.0e5) ** (287.0 / 1004.5)
    qv1 = a("QVAPOR")[0, j, i]
    u1 = 0.5 * (a("U")[0, j, i] + a("U")[0, j, i + 1])
    v1 = 0.5 * (a("V")[0, j, i] + a("V")[0, j + 1, i])
    ph = a("PH")[:, j, i] + a("PHB")[:, j, i]
    dz1 = (ph[1] - ph[0]) / 9.81
    psfc = a("PSFC")[j, i]
    p8w2 = 2.0 * p1 - psfc
    rain_h = (a("RAINNC")[j, i] + a("RAINC")[j, i]) - (np.asarray(prev["RAINNC"][0])[j, i]
                                                      + np.asarray(prev["RAINC"][0])[j, i])
    rainbl = max(rain_h, 0.0) / (3600.0 / DT)
    ust = max(a("UST")[j, i], 0.05)
    chs = 0.4 * ust / 7.0
    xlong = a("XLONG")[j, i]
    omg = np.deg2rad((hour_utc + xlong / 15.0 - 12.0) * 15.0)
    q2 = a("Q2")[j, i]
    t2 = a("T2")[j, i]
    return {
        "dt": DT, "nsteps": NSTEPS, "julian": julian, "julyr": 2026, "frc_in": 0.0,
        "xlat": a("XLAT")[j, i], "cosz": a("COSZEN")[j, i], "omg": omg, "declin": -0.15,
        "t3d1": t1, "qv1": qv1, "u1": u1, "v1": v1, "swdown": a("SWDOWN")[j, i], "glw": a("GLW")[j, i],
        "p8w1": psfc, "p8w2": p8w2, "rainbl": rainbl, "dz8w1": dz1, "znt": 0.8,
        "tsk": a("TSK")[j, i], "hfx": a("HFX")[j, i], "qfx": a("QFX")[j, i], "lh": a("LH")[j, i],
        "grdflx": a("GRDFLX")[j, i], "albedo": a("ALBEDO")[j, i], "emiss": a("EMISS")[j, i],
        "qsfc": 1.05 * q2, "chs": chs, "chs2": 0.6 * chs, "cqs2": 0.55 * chs, "ust": ust,
        "fveg": a("FVEG")[j, i], "t2mv": t2 + 0.1, "t2mb": t2 - 0.1, "q2mv": q2, "q2mb": 0.98 * q2,
        "psfc": psfc, "tslb1": a("TSLB")[0, j, i], "tslb2": a("TSLB")[1, j, i],
        "tslb3": a("TSLB")[2, j, i], "tslb4": a("TSLB")[3, j, i], "tmn": a("TMN")[j, i],
        "vegfra": a("VEGFRA")[j, i],
    }


def main(run_dir, out):
    inp = netCDF4.Dataset(f"{run_dir}/wrfinput_d03")
    iv = np.asarray(inp["IVGTYP"][0])
    jj, ii = np.nonzero(iv == 13)
    order = np.argsort(jj * 1000 + ii)
    jj, ii = jj[order], ii[order]
    pick = np.linspace(0, len(jj) - 1, CELLS_PER_TIME).round().astype(int)
    regimes = []
    for stamp, prevstamp, hour, julian, label in (
        ("2026-02-28_13:00:00", "2026-02-28_12:00:00", 13.0, 59, "day"),
        ("2026-03-01_03:00:00", "2026-03-01_02:00:00", 3.0, 60, "night"),
        ("2026-02-28_06:00:00", "2026-02-28_05:00:00", 6.0, 59, "predawn"),
    ):
        ds = netCDF4.Dataset(f"{run_dir}/wrfout_d03_{stamp}")
        prev = netCDF4.Dataset(f"{run_dir}/wrfout_d03_{prevstamp}")
        for k in pick:
            j, i = int(jj[k]), int(ii[k])
            r = regime_from_cell(ds, prev, j, i, hour, julian)
            regimes.append((f"{label}_j{j}_i{i}", r))
    # stress regimes derived from the first day / night cells
    day = dict(regimes[0][1])
    night = dict(regimes[CELLS_PER_TIME][1])
    s = dict(day); s["rainbl"] = 2.5 / (3600.0 / DT); regimes.append(("stress_rain_day", s))
    s = dict(night); s["rainbl"] = 4.0 / (3600.0 / DT); regimes.append(("stress_rain_night", s))
    s = dict(night); s["u1"] = 0.3; s["v1"] = -0.2; regimes.append(("stress_calm_night", s))
    s = dict(day); s["chs"] = 0.004; s["chs2"] = 0.003; s["cqs2"] = 0.002; regimes.append(("stress_chs_limit", s))
    s = dict(day); s["u1"] = 14.0; s["v1"] = 9.0; regimes.append(("stress_wind", s))
    s = dict(night); s["frc_in"] = 0.62; regimes.append(("stress_frc_input", s))
    s = dict(night); s["tsk"] = night["tsk"] - 8.0; s["tslb1"] = night["tslb1"] - 6.0; regimes.append(("stress_cold_stable", s))
    with open(out, "w") as fh:
        fh.write(f"{len(regimes)} {len(FIELDS)}\n")
        fh.write(" ".join(FIELDS) + "\n")
        for name, r in regimes:
            fh.write(name + "\n")
            fh.write(" ".join(repr(float(r[k])) for k in FIELDS) + "\n")
    print(f"wrote {len(regimes)} regimes to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
