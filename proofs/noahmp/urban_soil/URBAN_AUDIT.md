# Noah-MP URBAN-branch audit (V033-W2, manager 2026-10-07T04:39Z) — pristine WRF vs port at 69d8a9a7d

`lsm` = `<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F`, `drv` = `.../module_sf_noahmpdrv.F`; port paths are
relative to `src/gpuwrf/`. Our options: dveg 4, opt_run 3, opt_inf 1, opt_rsf 1, opt_soil 1, opt_crop 0, sf_urban_physics 0,
MODIS land use (ISURBAN 13), no LCZ categories present. Line references were read; spot-checked by hand: lsm:2185-2186,
lsm:7362, port water_hydro.py:270-274 / :779-783, energy.py:993-1002.

| # | WRF | WRF does for urban | active | port | status | note |
|---|---|---|---|---|---|---|
| 1 | lsm:874 | FVEG = 0 (urban or barren) | yes | physics/noahmp/phenology.py:220 | MATCH | literal 13; CANWATER local fveg not zeroed but LAI+SAI = 0 there |
| 2 | lsm:1061-1065 | QSFC = QFX/(RHOAIR*CH)+QAIR; Q2B = QSFC | yes | physics/noahmp/energy.py:1080-1082, :1089 | MATCH | port floors CH at 1e-6 (WRF none) — not urban-specific, disclosed |
| 3 | lsm:1327-1331 | PHENOLOGY LAI = SAI = 0 | yes | phenology.py:191-198 | MATCH | literal ISURBAN_MODIS |
| 4 | lsm:2101-2106 | Z0MG = Z0MVT, ZPDG = .65*HVT, Z0M/ZPD | yes | energy.py:945-948 | MATCH | |
| 5 | lsm:2204-2206 | RSURF = 1e6 if SNOWH == 0 | yes | energy.py:999 | MATCH | snowy urban RSURF uses #16/#18 |
| 6 | lsm:2468-2472 | THERMOPROP soil DF = 3.24 | yes | energy.py:248-249 (urban from noahmp_driver.py:241) | MATCH | 13 only; HCPCT uses #16/#20 |
| 7 | lsm:4473 | BARE_FLUX Q2B = QSFC | superseded by #2 | (#2) | MATCH | |
| 8 | lsm:7362 | SOILWATER FCR(1) = 0.95 | yes | physics/noahmp/water_hydro.py:586-587, mask passed :781-782 only `if native_real_enabled()` | PARTIAL | ON in the release (GPUWRF_NOAHMP_NATIVE_REAL=1 is a fast default, _fast_defaults.py:50); legacy f64 path lacks it (pre-existing, C28) |
| 9 | lsm:7701 | INFMAX*0.05 (commented out in WRF) | — | — | N/A | |
| 10 | lsm:9983 | CARBON zeroes LAI/pools | no (DVEG 2/5/6 only) | — | INACTIVE | |
| 11 | lsm:10369 | CARBON_CROP zeroes | no (opt_crop 0) | — | INACTIVE | |
| 12 | drv:916-928 | urban/LCZ -> VEGTYP = ISURBAN_TABLE if sf_urban 0 | no-op for 13 | raw ivgtyp | MATCH / INACTIVE (LCZ) | |
| 13 | drv:976 | skip pedotransfer for urban | no (opt_soil 1) | — | INACTIVE | |
| 14 | drv:1008-1022 | urban irrigation | no (sf_urban > 0 only) | — | INACTIVE | |
| 15 | drv:1462-1467 | URBAN_FLAG = ISURBAN or LCZ_1..11 (51..61) | yes | energy.py:937, noahmp_driver.py:241 (isurban); phenology.py:195,220 (13); water_hydro.py:781 (13+51..61); io/land_history.py:72-73 | MATCH here | four mask definitions agree only because only 13 occurs (latent, disclosed) |
| 16 | drv:1695 | SMCMAX = 0.45 (all layers) | yes | noahmp_driver.py:135; water_hydro.py:705 | **MISSING -> FIXED (key)** | HCPCT, TDFCND, PSI/RHSUR, L/D_RSURF, PHASECHANGE SUPERCOOL, WATER EPORE/FCR/INFIL/SRT/SSTEP |
| 17 | drv:1696 | SMCREF = 0.42 | yes | noahmp_driver.py:135; water_hydro.py:706 | **MISSING -> FIXED (key)** | via FRZX (#21); BTRAN inert for urban (no veg) |
| 18 | drv:1697 | SMCWLT = 0.40 | yes | noahmp_driver.py:135; water_hydro.py:707 | **MISSING -> FIXED (key)** | INFIL SMCAV = 0.05 (urban nearly impermeable), D_RSURF |
| 19 | drv:1698 | SMCDRY = 0.40 | — | — | N/A | no consumer in lsm under our options |
| 20 | drv:1699 | CSOIL = 3.E6 | yes | noahmp_driver.py:137; energy.py:184 | **MISSING -> FIXED (key)** | HCPCT -> TSNOSOI, PHASECHANGE (the soil-lead root cause) |
| 21 | drv:1702-1706 | FRZX from the urban SMCMAX/SMCREF | yes | water_hydro.py:765-767 | **MISSING -> FIXED (key)** | formed from the overridden smcmax/smcref under the key; frozen soil only |
| 22 | drv:2155-2186 | NOAHMP_INIT urban LAI/XSAI/pools = 0 | t = 0 | io/land_history.py:72-82 | MATCH | |
| 23 | drv:2655-2667 | GROUNDWATER_INIT urban SMCMAX/SMCWLT | no (opt_run 5) | — | INACTIVE | |
| 24 | drv:2994-3783 | noahmp_urban UCM/BEP/BEM | no (sf_urban 0) | runtime/operational_mode.py:6517-6520 fails closed | INACTIVE | |

Albedo, emissivity (EG(IST)), IST and sfc_diff have no further urban branch in the driver.
Remaining gaps after the key: #8 legacy-path only (release path matches), #15 mask unification (latent; would matter for
USGS/LCZ land use) -> v0.3.4 items, not active in any release case.
