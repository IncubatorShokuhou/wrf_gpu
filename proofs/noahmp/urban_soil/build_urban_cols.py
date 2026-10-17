"""GPUWRF_NOAH_URBAN_SOIL_PARAMS gate inputs: real CPU-WRF R d03 land columns (state at H0, lowest-level forcing at H1 held,
d03 dt 6 s), all urban (IVGTYP 13, <=36) + 24 non-urban cells of b-diff's 531 mask (LANDMASK>.5, HGT<300, ring>=5).
Writes <out>/columns.json (test _build format) + <out>/noahmp_columns.in (wired pristine driver).
usage: build_urban_cols.py <src_tree> <outdir>"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("bns", src + "/proofs/noahmp/build_noahmp_savepoints.py")
bns = importlib.util.module_from_spec(spec); spec.loader.exec_module(bns)
R = "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"
H0, H1, JULIAN, DT = "2026-02-28_00", "2026-02-28_01", 58.0, 6.0
d0 = Dataset([str(p) for p in Path(R).glob(f"wrfout_d03_{H0}*")][0]); d1 = Dataset([str(p) for p in Path(R).glob(f"wrfout_d03_{H1}*")][0])
v0 = lambda n: np.asarray(d0[n][0], float)  # noqa: E731
v1 = lambda n: np.asarray(d1[n][0], float)  # noqa: E731
hgt, lm, iv, il = v0("HGT"), v0("LANDMASK"), v0("IVGTYP").astype(int), v0("ISLTYP").astype(int)
ny, nx = hgt.shape
jj, ii = np.mgrid[0:ny, 0:nx]; ring = np.minimum(np.minimum(jj, ii), np.minimum(ny - 1 - jj, nx - 1 - ii))
m = (lm > 0.5) & (hgt < 300) & (ring >= 5) & (v0("ISNOW") == 0)
rng = np.random.default_rng(20261007)
urb = [tuple(x) for x in np.argwhere(m & (iv == 13))]
urb = [urb[k] for k in sorted(rng.choice(len(urb), size=min(36, len(urb)), replace=False))]
non = [tuple(x) for x in np.argwhere(m & (iv != 13))]
non = [non[k] for k in sorted(rng.choice(len(non), size=24, replace=False))]
acc = lambda n: (v1(n) - v0(n)) / 3600.0  # noqa: E731
dzs = [float(x) for x in d0["DZS"][0]]; zsoil = [float(z) for z in -np.cumsum(dzs)]
cols = []
for kind, cells in (("urban", urb), ("nonurban", non)):
    for (i, j) in cells:
        g = lambda n: float(v0(n)[i, j])  # noqa: E731
        lay = lambda n: [float(x) for x in v0(n)[:, i, j]]  # noqa: E731
        f = bns._column_forcing(d1, i, j)
        f.update(prcpconv=float(acc("RAINC")[i, j]), prcpnonc=float(acc("RAINNC")[i, j]), prcpsnow=float(acc("SNOWNC")[i, j]),
                 prcpgrpl=float(acc("GRAUPELNC")[i, j]), prcphail=0.0)
        state = {"stc": lay("TSNO") + lay("TSLB"), "smc": lay("SMOIS"), "sh2o": lay("SH2O"), "tv": g("TV"), "tg": g("TG"),
                 "tah": g("TAH"), "eah": g("EAH"), "canliq": g("CANLIQ"), "canice": g("CANICE"), "fwet": g("FWET"),
                 "qsfc": f["q2"], "snowh": g("SNOWH"), "sneqv": g("SNOW"), "sneqvo": g("SNEQVO"), "albold": g("ALBOLD"),
                 "tauss": g("TAUSS"), "isnow": int(g("ISNOW")), "zsnso": lay("ZSNSO"), "snice": lay("SNICE"),
                 "snliq": lay("SNLIQ"), "cm": g("CM"), "ch": g("CH"), "smcwtd": g("SMCWTD")}
        cols.append({"name": f"d03_{kind}_{i}_{j}", "case": "R0227", "vegtyp": int(iv[i, j]), "isltyp": int(il[i, j]),
                     "lat_rad": float(np.deg2rad(v0("XLAT")[i, j])), "julian": JULIAN, "yearlen": 365, "dt": DT,
                     "dx": float(d0.DX), "zsoil": zsoil, "dzs": dzs, "shdfac": g("VEGFRA") / 100.0,
                     "shdmax": g("SHDMAX") / 100.0, "tbot": g("TMN"), "lai0": g("LAI"), "sai0": g("XSAI"),
                     "forcing": f, "state_in": state})
# WET urban variants (water-side coverage): the first 8 urban columns with a moist profile and steady rain, real structure
# otherwise (the real night columns are dry, SMOIS ~.07-.11, so SOILWATER/INFIL barely act there).
for c0 in [c for c in cols if "_urban_" in c["name"]][:8]:
    c = json.loads(json.dumps(c0)); c["name"] = c0["name"].replace("_urban_", "_urban_wet_")
    c["state_in"]["smc"] = c["state_in"]["sh2o"] = [0.40, 0.38, 0.36, 0.34]
    c["forcing"]["prcpnonc"] = 2.0e-4      # mm/s = 0.72 mm/h
    cols.append(c)
# FROZEN urban variants (FRZX-order coverage; review-s2small 04:49Z): the next 8 real urban columns with a partly
# frozen, wet soil (SICE > 0 in layers 1-2) under warm rain (sfctmp 275.5 K -> liquid, OPT_SNF=1) -> INFIL's frozen-soil
# term ACRT = 3*FRZX/DICE (water_hydro._infil, WRF INFIL) is active; real structure otherwise.
for c0 in [c for c in cols if "_urban_" in c["name"] and "_wet_" not in c["name"]][8:16]:
    c = json.loads(json.dumps(c0)); c["name"] = c0["name"].replace("_urban_", "_urban_frozen_")
    st, fo = c["state_in"], c["forcing"]
    st["smc"] = [0.40, 0.38, 0.36, 0.34]; st["sh2o"] = [0.18, 0.22, 0.36, 0.34]
    st["stc"] = st["stc"][:3] + [271.5, 271.8, 272.5, 273.5]
    st["tv"] = st["tg"] = st["tah"] = 273.0
    fo["sfctmp"] = 275.5; fo["prcpnonc"] = 2.0e-4
    cols.append(c)
json.dump({"dataset": d0.MMINLU, "source": f"{R} d03 {H0}->{H1} (+8 wet + 8 frozen urban variants)", "columns": cols}, open(out / "columns.json", "w"), indent=1)
bns.HERE, bns.DATASET, bns.JULIAN, bns.YEARLEN, bns.DX, bns.DT, bns.ZSOIL = out, d0.MMINLU, JULIAN, 365, float(d0.DX), DT, zsoil
bns._write_columns([dict(c, lat=c["lat_rad"], state=c["state_in"]) for c in cols])
print(len(urb), "urban +", len(non), "non-urban columns ->", out, "| MMINLU", d0.MMINLU, "DX", d0.DX)
