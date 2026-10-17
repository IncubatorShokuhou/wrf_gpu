# CALLER_AUDIT — E11 caller-wiring audit of every admitted PBL / cumulus / surface-layer / LSM option vs pristine WRF 4.7.1

Lane t-calleraudit · 2026-10-15 · port = main ea6c83d25 · WRF = <USER_HOME>/src/wrf_pristine/WRF (4.7.1)
Method: static source audit (no runs). Seven read-only part audits, each with item-by-item tables quoting
file:line on BOTH sides, in `parts/` (pbl_a.md MYJ/GFS/MRF/ACM2/BouLac · pbl_b.md MYNN/QNSE/TEMF/GBM/EEPS-KEPS ·
cu_a.md KF/BMJ/GF/G3/GD · cu_b.md Tiedtke/NTiedtke/KF99/KF-CuP/MSKF · cu_c.md KSAS/NSAS/SAS94/OSAS95/scaleSAS/ZM ·
sfclay.md sf 0/1/2/3/4/5/7/10/91 · lsm.md sf_surface 0/1/2/3/4/7/8). Out of scope (other lanes): YSU bl1 (t-ysuqc),
Shin-Hong bl11, CAM-UW bl9 / CAM-MG / CAM radiation, WSM/WDM, all microphysics and radiation internals.
Release defaults assumed (src/gpuwrf/_fast_defaults.py): rad_rk_tendf=1 (source_leaf_mode), PHYS_TEND_RK_WRF=1,
KF_TEND_RK_WRF=1. Label: V = re-verified by the lane owner on both sides; H = helper HIGH (both sides read); M = medium.
Composite IDs = v-prep matrix (`composites.txt`; same Swiss Jan case for c02–c25: MYNN baseline T2 rmse/limit ≈ .47,
c03 MYJ .73, c04 GFS .81, c05 FAIL RAINNC 1.11, c06 RAINNC .97 — `composite_d6_margins.txt`).

## 1. Ranked findings (forecast impact first)

| # | cls | finding | schemes | composites | port | WRF | lbl |
|---|---|---|---|---|---|---|---|
| F1 | A | **Noah-MP (and RUC/SSiB) get ZERO precipitation for every MP except 8/5/7/15/38**: `noahmp_precipitation` carry seeded only for those MPs; without it `prc2d` returns zeros (no rain, snow, graupel, hail, convective) — land runs dry, no snowpack from snowfall | all mp ∉ {5,7,8,15,38} with lsm 4 (+3,8) | c02–c25, c61, c87, c88, c89 (28) | OM:7793-7795, OM:8579-8582 (mp28 + generic MP_SCAN_ADAPTERS return no precip), noahmp_coupler.py:151-185, noahmp_surface_hook.py:263 | surface_driver.F:1568 RAINBL, :3283-3284; noahmpdrv.F:764,778-788 | V |
| F2 | A | **Convective rain never reaches the LSM for cu 2/3/5/6/16/93** (`cumulus_tendencies` None → `convective_rate` = 0); they only add to rainc_acc | cu 2,3,5,6,16,93 | c03 c04 c05 c06 c07 c13 c17 (masked by F1 today) | OM:10398-10399; SA:1024,1155,1313,1412 | surface_driver.F:1568,3283 (RAINCV→RAINBL/MP_RAINC) | V |
| F3 | A | **Selected surface layer silently replaced by MYNN-SL under Noah-MP for sf 2/3/7/91**: no `surface_diagnostics` handed to Noah-MP → noahmp_coupler re-solves MYNN-SL and its blended UST/fluxes/roughness/MOL/QSFC/U10/V10 overwrite the selected scheme (sf1 already hands its solve in: `sf1_noah_surface`) | sf 2,3,7,91 | c03 c04 c25 c43 c61 | OM:9053-9063, OM:9113-9131; noahmp_coupler.py:283; noahmp_surface_hook.py:321-339 | surface_driver.F:2145-2359 (selected scheme only); noahmplsm :3227-3290 | V |
| F4 | A | **Legacy phy_prep staging = moist θ as TH/T** (θ_m passed where WRF passes dry th_phy/t_phy ≈ +2…7 K in T) | MYJ PBL+sf2, GFS PBL (non-sf1 branch), sf 3/7/91, BMJ cu2, GF cu3, NTiedtke cu16, slab/Noah-classic/PX LSM | c03 c04 c07 c25 c43 c61 (+API-only LSMs) | myj_adapters.py:83,98; SA:603,663,1243,1442,1074,1568; slab/noahclassic/px hooks | big_step_utilities_em.F:4834,4855; first_rk_step_part1.F:657,1138,1379 | V (MYJ, GFS) H (rest) |
| F5 | A | **Dry-θ tendency added straight onto θ_m** (no conv_t_tendf_to_moist; loses Rv/Rd·θ·Δqv) | MYJ (cu≠7), GFS-PBL legacy, BMJ, GF, NTiedtke | c03 c04 c07 c25 | myj_adapters.py:231; SA:1309,1489,1151,1834 | big_step_utilities_em.F:6674-6707; first_rk_step_part2.F:1005-1016 | H |
| F6 | A | **PBL cloud-water/ice mixing missing** (qc/qi not passed, RQC/RQIBLTEN not returned/applied) | ACM2 bl7 (qc+qi), MRF bl99 (qc; zeros passed), GFS bl3 (qc zeros, qi absent), MYJ bl2 (qi not in CWM, no RQC/RQI) | c05 (FAIL), c09, c04 c25, c03 | SA:1704 (+pbl_acm2.py:1190-1207), SA:1788,1797, SA:1821,1830, bl_myj.py:495,537,576 | pbl_driver.F:1592-1606 ACMPBL QC3D/QI3D/RQC/RQIBLTEN; acm.F:554-628; addtendc.F:419-426,474-481,567-580,674-681 | V (ACM2) H |
| F7 | A | **G3/GD closure forcing ≈ 0**: RTHFTEN/RQVFTEN = state change since physics entry (≈0 on the default path), RTHRATEN = 0 | cu 5, 93 | c05 (FAIL) c13 c17 | SA:1376-1387; OM:9694-9707 | cumulus_driver.F:875-883; module_em.F:661,1372; big_step_utilities_em.F:5297-5331 | V |
| F8 | A | **MYNN never mixes/applies Nc and aerosol numbers** (RQNCBLTEN; RQNWFA/RQNIFA/RQNBCA for mp28/38) — and WRF disables the generic scalar mixer under MYNN, so these get NO PBL mixing in the port | bl5 with mp 9/11/14/16/26/28/38/40/51/52/53/56 | c18 c19 c20 c22 c23 c24 c65 c88 | physics_couplers.py:2276; OM:9573-9590 | bl_mynnedmf.F:4549-4740; pbl_driver.F:1879-1917; check_a_mundo.F:2516-2518 | H |
| F9 | A | **Cumulus/PBL rates applied as one Euler increment outside the RK** (dynamics never see heating/moistening/momentum inside the step; no μ coupling/decoupling) | cu 2,3,5,6,16,93,99,14,94,95,96,4; PBL MYJ (cu≠7), GFS legacy | c03–c11, c13, c14, c17, c25 | OM:8375-8392,10326; sas_consumers.py:31-44; SA:1006-1027,1141-1160; kf99_adapter.py:66-79; myj_adapters.py:231 | module_em.F:2252-2305; addtendc.F:1125-1618; big_step_utilities_em.F:5112 | H (KF/cu7/10/11 already RK) |
| F10 | A | **GF runs shallow convection by default** (port ishallow_g3=1; WRF Registry default 0 = off) | cu3 | c04 | SA:1194; OM:9763 | first_rk_step_part1.F:1410; cu_gf_wrfdrv.F:331,504 | H |
| F11 | A | **Condensate tendencies applied to species the MP lacks** (KF never gets warm_rain/F_QI/F_QS flags; G3/GD/GF/BMJ/Tiedtke always split ice at T<258 and add qi) | KF cu1 (Kessler/WSM3), G3/GD/GF/BMJ/Tiedtke with no-ice MP | c02 (mp1+KF), c04 (mp3+GF) | cumulus_kf.py:774-776, SA:823-826, OM:10067-10069; SA:1024,1158 | cu_kfeta.F:211-216,493-525; addtendc.F:1222-1258,1390-1395,1613 | H |
| F12 | A/B | **MYJ PBL surface inputs from a private 2nd Janjic solve**: memory (THZ0/QZ0/UZ0/VZ0/AKHS/AKMS) reset each call, ELFLX = re-solve LH (~1 % of physical over land), CHKLOWQ = mavail; Janjic fed qke (=2·TKE) and doubles it again | bl2/sf2 | c03 | myj_adapters.py:128-135,202-213; bl_myj.py:548-550; sfclay_janjic.py:90,413 | pbl_driver.F:1455-1457 (ELFLX=grid%lh); surface_driver.F:2214,3429 (CHKLOWQ=1); sf_myjsfc.F:246,297-310 | H |
| F13 | A | **GFS PBL forcing from a third, grid-less MYNN-SL solve** (bulk HFX over land, MYNN psim/psih/br/u10/v10, own gz1oz0) instead of SF_GFS outputs; UST/WSPD writeback dropped | bl3 (sf3/91) | c04 c25 | OM:9540 → SA:1561,1584,1818 | pbl_driver.F:1400-1424; bl_gfs.F:464-466 | V (branch) H |
| F14 | B | Tiedtke/NTiedtke forcing staging: THFTEN = θ_m change since entry (≈0, no ×π, no advection); RQVFTEN recomputed stage-1 unlimited on current state, never zeroed at itimestep 1; cudt ignored; ZNU proxy; HFX/QFX rebuilt from kinematic handles | cu6, cu16 | c06 c07 | OM:9685-9688, OM:5487-5500; SA:945-957,1091-1092,1130 | cumulus_driver.F:831-883; module_em.F:1365-1373; tiedtke.F:364-370,2343 | H |
| F15 | B | Hydrostatic pressure staging: full non-hydrostatic P, p8w = neighbour means / extrapolated PSFC (≈150–300 Pa at sfc) | sf 2/3/7/91, QNSE, TEMF, EEPS/KEPS, G3/GD, GF, BMJ, Tiedtke/NTiedtke, MYJ | many non-default | SA:614-616,747,940-944,1366-1373; qnse_adapter.py:63-85; temf.py:41-54; epsilon_pbl.py:133-153 | first_rk_step_part1.F:633,637,1130-1133,1380,1394,1400; surface_driver.F:1988 | H |
| F16 | B | g = 9.80665 for dz8w and dry ρ where WRF uses 9.81 / total ρ (Tiedtke/NTiedtke/GF/BMJ dz; NTiedtke/GF/BMJ ρ) | cu 2,3,6,16 | c03 c04 c06 c07 | SA:938,1073,1076,1242,1245,1441,1443 | big_step_utilities_em.F:4856,4869 | H |
| F17 | B | GBM gets RTHRATEN = 0 (cloud-top TKE production / PBLH search) | bl12 | c08 | bl_gbm.py:462-467 | bl_gbmpbl.F:243,757-760,972-985 | H |
| F18 | B | GF: PBL top from a home-made Ri diagnostic (G3 too), w averaged to mass, PSFC = lowest level, no 4-cell lateral band | cu3 (cu5 kpbl) | c04 (c05) | SA:1249,1262-1265,1394; _gf_jax.py:2129,2149 | first_rk_step_part1.F:1397; cu_gf_wrfdrv.F:264-277,385,471 | H |
| F19 | B | PX/old-MM5/GFS/Janjic sfclay carry no WRF memory (PX prev HFX/QFX=0, PBLH 1000, MOL/ZOL 0, water znt not written back) | sf 2,3,7,91 | c03 c04 c25 c43 c61 | SA:666-678,716-727; myj_adapters.py:134 | sf_pxsfclay.F:370,408; sf_sfclay.F:460,532; sf_myjsfc.F:297-310 | H |
| F20 | B | **MYNN default path (PROD/Swiss!)**: (a) EDMF resolved-w taper fed mass-averaged w instead of face w (incl. surface face); (b) no WRF HFX/QFX clamps [-600,1200] W/m², [-3e-4,9e-4] | bl5 (DEFAULT) | ALL MYNN incl. PROD/WN3 | physics_couplers.py:444-449,2261 → mynn_edmf.py:291-298; noahmp_coupler.py:434-461 | mynnedmf_driver.F:348,381-397; bl_mynnedmf.F:5855-5879 | V (a) H (b) |
| F21 | B | sf=0 still runs MYNN-SL (+Noah-MP) where WRF returns before any surface/LSM | sf0 | c01 c39 history; forecast if PBL/LSM selected | OM:9233,9113 | surface_driver.F:1490 | H |
| F22 | B | bldt>0 / cudt>0 (BMJ/GF/Tiedtke) only warn, port runs every step | PBL all; cu 2,3,6,16 | none (composites use 0) | namelist_check.py:675-690; SA:1467 | pbl_driver DTBL; cu_bmj.F:171,257 | M |
| F23 | B | QNSE SICE always 0 (WRF XICE); Noah-MP opt_sfc=2 + QNSE overwrites Noah CH; slope_rad SW not applied for RUC/SSiB | bl4; lsm 3/8 | c26 (only sea ice) | qnse_adapter.py:117,136; noahmp_coupler.py:322-324; OM:9361 | pbl_driver.F:1545; noahmpdrv.F:817-818; surface_driver.F:1895-1913 | H/M |
| F24 | A/B | API-only LSMs (CLI refuses): Noah-classic/PX precipitation hard-wired 0; slab/PX use SWDOWN not net GSW; Noah-classic mixing ratio for specific humidity | lsm 1,2,7 | none | noahclassic_surface_hook.py:191,212; pleim_xiu_surface_hook.py:188-201; lsm_slab.py:148 | sf_noahdrv.F:797,835; sf_pxlsm.F:482,596; sf_slab.F:404 | H |

MATCH (no class A/B found): sf1 revised MM5, sf5 MYNN-SL (B39/B39b present), sf10 TEMF, BouLac bl8 (only M seed
item), KF cu1 release path (except F11), KF-CuP cu10, MSKF cu11 (admitted profile), KSAS/NSAS/SAS94/OSAS95 inputs
(only F9), RUC c33 / SSiB c40 inputs (except F2/F23), Noah-MP inputs with Thompson. Fixes verified present:
Tiedtke e213b65c4 (T_PHY dry, total RHO, moist-form θ tendency), ACM2 XTIME, SF1→PBL FM/FHH vs raw PSI, GBM UST.
Class C/D (history only, cosmetic) and all UNVERIFIED items: see the part reports.

## 2. Systemic root causes (fix once, many schemes)
- **S-A Precipitation plumbing to the land model** (F1, F2, F24): one producer contract is needed — every MP adapter
  returns its per-step amounts (RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV) and every cumulus scheme its RAINCV/PRATEC, then
  `precipitation_from_step` for all admitted MP/cu (RUC already refuses this gap at OM:7386-7391; Noah-MP never got the refusal).
- **S-B Pre-phy_prep "legacy assembler"** (F4, F5, F15, F16): the WRF phy_prep assembly already exists
  (`coupling/urban_pbl_hook.urban_pbl_forcing`: dry θ/T_PHY, hydrostatic P/P8W, total ρ, g 9.81) and is used by the
  SF1 PBL branch, KF and Tiedtke; MYJ, GFS-legacy, sf 2/3/7/91, BMJ, GF, G3/GD, NTiedtke still roll their own.
- **S-C Euler increment instead of RK-coupled rates** (F9, MYJ/GFS PBL): the RK source-leaf seam exists for PBL
  (`pbl_moist_raw` + rthblten…) and cumulus (`kf_rk.couple_kf_rates`, cuplus); the older adapters predate it.
- **S-D Surface-layer → LSM/PBL handoff** (F3, F12, F13): only sf1 and sf5 (and TEMF/QNSE) hand the selected solve on.

## 3. Fix plan (this lane, one scheme/seam per commit series; pristine oracle + caller-path test + mutant each)
1. **F1 (+F2)** Noah-MP precipitation for every admitted MP/cu — largest blast radius (28 composites), WRF semantics
   unambiguous; oracle = pristine surface_driver RAINBL/noahmpdrv PRCP* statements on real per-step MP amounts + an
   operational 2-step caller test (forcing == RAINNCV/RAINCV per scheme); mutant = drop one phase/scheme.
2. **F6 ACM2 qc/qi** (c05 FAIL) — pristine public ACMPBL oracle (t-shinhong template) on real cloudy c05 columns.
3. **F3** selected surface layer → Noah-MP for sf 2/3/7/91 (sf1 `sf1_noah_surface` template).
4. **F7** G3/GD forcing (advective RTHFTEN/RQVFTEN of the previous step + held RTHRATEN + PBL rates).
Then F4/F5 per scheme (MYJ, GFS, BMJ, GF, NTiedtke), F6 rest (MRF, GFS, MYJ), F8, F10, F11.
**Needs a manager decision (default program changes): F20 MYNN w-face / flux clamps** — would change Swiss f672a573 /
PROD 6fc715a2 and need Swiss/PROD/WN3 GPU re-qualification; not touched by this lane without an explicit go.
GPU reruns after each fix: listed per finding in the table (composites column).
