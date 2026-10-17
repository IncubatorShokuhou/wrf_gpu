# Caller-wiring audit — surface layer (sf_sfclay_physics 0/1/2/3/4/5/7/10/91)

Static read-only audit, port worktree = main (src identical to bedb0629e), pristine WRF 4.7.1.
Abbrev: OM = src/gpuwrf/runtime/operational_mode.py, SA = coupling/scan_adapters.py, NC = physics/noahmp_coupler.py,
NH = coupling/noahmp_surface_hook.py, MYJA = physics/myj_adapters.py, SD = phys/module_surface_driver.F,
P1 = dyn_em/module_first_rk_step_part1.F, BSU = dyn_em/module_big_step_utilities_em.F, PD = phys/module_pbl_driver.F.

WRF caller facts used throughout (read): surface_driver gets P_PHY=grid%p_hyd, P8W=grid%p_hyd_w (P1:633,637),
PSFC(i,j)=p8w(i,kts,j)=p_hyd_w(kts) (SD:1988), T_PHY=th_phy*pi_phy with DRY th_phy=(t+t0)/(1+Rv/Rd qv) and
pi_phy from nonhydrostatic p+pb (BSU:4834,4853-4855), TH_PHY=th_phy (P1:657), dz8w from (ph+phb)/g, g=9.81
(BSU:4869,4877), RHO=grid%rho=(1+qv)/alt (BSU:4856), u/v A-grid (SD:1992). Order: radiation -> SFCLAY* -> LSM ->
PBL. Noah-MP (noahmplsm SD:3227-3290) takes NO sfclay output (own CHXY/CMXY); sfclay UST, U10/V10, water HFX/QFX/
LH/QSFC/ZNT, CHS2/CQS2 (water T2/Q2, SD:3446-3458) survive to PBL/history; land T2/Q2 = Noah blend (SD:3459-3471).
Port state.theta = theta_m (use_theta_m=1), state.p = p+pb (nonhydrostatic), state.ph = total geopotential.
Release defaults: Noah-MP (lsm=4), GPUWRF_SFCLAY_NATIVE_REAL=1. Every composite with sf in {2,3,7,91} uses lsm=4
(c03 sf2/bl2, c04 sf3/bl3, c25 sf91/bl3, c61 sf91/bl0, c43 sf7/bl0).

## GLOBAL FINDING (drives c03/c04)  — A1, HIGH
With Noah-MP the caller runs the selected scheme (OM:9053-9063: `janjic_sfclay_adapter` / `SFCLAY_SCAN_ADAPTERS[sf]`
for 3/7/91), then calls `noahmp_surface_step` (OM:9113) passing `surface_diagnostics` ONLY for EEPS/KEPS, SF1, QNSE,
TEMF (OM:9118-9131). For sf=2/3/7/91 none is passed, so NC:283-284 runs
`diag = ... else surface_layer_with_diagnostics(state, ...)` = a full **MYNN-SL** solve over all columns, and NH:321-339
OVERWRITES ustar, theta_flux, qv_flux, tau_u/v, rhosfc, fltv, roughness_m (MYNN Charnock water z0), mol, hfx, qfx
(water = MYNN), qsfc (water), sfc_wspd. History U10/V10 = MYNN (NC:510), water T2/Q2 from MYNN CHS2/CQS2 (NC:503-507).
The selected scheme only seeds `ustar` (MYNN-SL `ust_in`) — its fluxes/stress/z0/U10 are discarded.
WRF: SD:2180-2238 (MYJSFC), 2323-2359 (SF_GFS), 2145-2178 (PXSFCLAY), 2007-2087 (SFCLAY 91) are the ONLY surface
layer; their UST/stress (via UST,ZNT), water HFX/QFX/LH, U10/V10, CHS2/CQS2 reach PBL and history.
Affects: every sf in {2,3,7,91} with sf_surface_physics=4 (c03,c04,c25,c43,c61). c43 (PX, bl=0) D6 worst V10@1h —
consistent with MYNN U10/V10 replacing PX from the first hour. Fix sketch: build a scheme-specific diagnostics
NamedTuple from the actual call (fluxes,hfx,lh,u10,v10,t2,q2,qsfc,mol,znt,chs,chs2,cqs2, ch/cm = land carry, as
`EpsilonNoahSurface`/`QnseSurfaceDiagnostics` already do) and pass it as `surface_diagnostics`; never re-solve MYNN.
Oracle: pristine one-step SFCLAY/MYJSFC/SF_GFS/PXSFCLAY + noahmplsm on a real Swiss column set; gate UST, water
HFX/LH, U10/V10, ZNT(water); deletion mutant = restore the MYNN fallback.

---------------------------------------------------------------------------------------------------------------
## sf=3 NCEP-GFS (priority, c04)
(a) Default branch with Noah-MP: OM:9062-9063 `gfs_sfclay_adapter` (SA:735-758) then A1 MYNN overwrite; no-LSM/other
LSM: OM:9230-9231. WRF call: SD:2344-2353 `SF_GFS(u_phytmp,v_phytmp,t_phy,qv_curr,p_phy,CP,RCP,R_d,XLV,PSFC,...)`.
(b)
| item | WRF | port | verdict |
|---|---|---|---|
| T3D | t_phy dry (BSU:4855) | θ_m·π, SA:603 `_temperature_from_theta(state.theta, state.p)` | MISMATCH (A2) |
| P3D | p_hyd (P1:637) | state.p[0] nonhydrostatic, SA:614 | MISMATCH (B1, small) |
| PSFC | p_hyd_w(kts) (SD:1988) | p[0]+ρ g dz/2, SA:745-747 | MISMATCH (B1) |
| QV, U/V | qv_curr, A-grid | state.qv[0], `_u_mass/_v_mass` | MATCH |
| ZNT, UST in | grid znt/ust raw | max(znt,1e-4), max(ust,1e-3) SA:751 | MISMATCH (D) |
| ISFFLX/itimestep | namelist / grid | isfflx=1 hard; itimestep only GFUNCPHYS init | MATCH (honour check) |
| ZNT out (water z0) | INOUT written (module_sf_gfs.F `ZNT=Z0RL*.01`) | not written back (SA:757 handles only) | MISMATCH (B2) |
| HFX/QFX/LH/QSFC/CHS*/FLHC/FLQC/PSIM/PSIH/GZ1OZ0/BR/WSPD/U10/V10 out | to LSM-unused, PBL BL_GFS (PD:1405-1420), history | only kinematic handles written; all discarded under Noah-MP (A1); history = MYNN | MISMATCH (A1/A4/C1) |
(c) A1, A2, A4, B1, B2, C1 below. No T2/Q2 in SF_GFS: WRF no-LSM history T2/Q2 are never updated (port: output MYNN).

## sf=2 Janjic Eta MYJSFC (priority, c03)
(a) OM:9053-9059 (Noah) / OM:9221-9228 (non-Noah) `janjic_sfclay_adapter` (MYJA:143-186) then A1. MYJ PBL (OM:9433)
re-runs Janjic (MYJA:202). WRF: SD:2210-2224 `MYJSFC(itimestep,ht,dz8w,p_phy,p8w,th_phy,t_phy,qv,qc,u_phy,v_phy,
tke_pbl,tsk,qsfc,thz0,qz0,uz0,vz0,lowlyr,...,ust,znt,z0,pblh,mavail,rmol,akhs,akms,...)`; PBL PD:1446-1460.
(b)
| item | WRF | port | verdict |
|---|---|---|---|
| TH, T | th_phy dry, t_phy | state.theta (θ_m), θ_m·π (MYJA:83,98) | MISMATCH (A2) |
| PMID / PINT, PSFC | p_hyd / p_hyd_w, PSFC=PINT(LOWLYR) (myjsfc.F:224) | state.p nonhydro / midpoint assembler, psfc=p[0] (MYJA:90-110) | MISMATCH (B1: THSK +~0.1-0.26 K) |
| DZ | dz8w, g=9.81 | ph/9.80665 (MYJA:86) | MISMATCH (D) |
| Q2 | tke_pbl = TKE; kernel Q2K=2·Q2 (myjsfc.F:246) | qke (=2·TKE) passed, kernel doubles again (MYJA:128, sfclay_janjic.py:90) | MISMATCH (B3) |
| THZ0,QZ0,UZ0,VZ0,QSFC INOUT | carried grid%thz0.. (SD:2214) | re-seeded every call: thz0=θ_m low, qsfc=qv0, qz0=q_spec, uz0=vz0=0 (MYJA:134) | MISMATCH (B2; A3 via PBL) |
| AKHS/AKMS in (sea viscous iter.) | carried | 0 (sfclay_janjic.py:413) | MISMATCH (B2) |
| ITIMESTEP | NTSD==1 branch (myjsfc.F:189,297) | always NTSD>1 path | MISMATCH (C, step 1) |
| PBLH | diagnosed from Q2 | diagnosed from 2× Q2 | see B3 |
| ZNT out | INOUT | written roughness_m (MYJA:178) then MYNN-overwritten (A1) | A1 |
| U10E/V10E | MYJSFC (Registry 1069) | from this solve (MYJA:181-184) | MATCH |
| outputs to MYJPBL | UST,AKHS,AKMS,THZ0,QZ0,UZ0,VZ0,QSFC,CT this step; ELFLX=grid%lh post-LSM, CHKLOWQ=1 (SD:3429) | second Janjic solve on post-LSM state; elflx=re-solve FLX_LH, chklowq=mavail (MYJA:202-213) | MISMATCH (A3) |
(c) A1, A2, A3, B1, B2, B3, C1.

## sf=7 Pleim-Xiu (c43)
(a) OM:9062-9063 (Noah: then A1) / OM:9230-9231 + PX LSM OM:9304-9330. WRF SD:2165-2176 `PXSFCLAY(u,v,t_phy,th_phy,qv,
p_phy,dz8w,...,psfc,...,znt,ust,pblh,mavail,zol,mol,...,hfx,qfx,lh,tsk,...,qsfc,...,dx,...,itimestep)`.
| item | WRF | port (SA:657-680) | verdict |
|---|---|---|---|
| T3D/TH3D | t_phy / th_phy dry | θ_m·π / state.theta[0]=θ_m (SA:603,663) | MISMATCH (A2) |
| P3D / PSFC | p_hyd / p_hyd_w(kts) | state.p[0] / state.p[0] (lowest mass level, SA:616) | MISMATCH (B1, ~ρg dz/2 ≈150-300 Pa: RHOX, QSFC, CPOT) |
| HFX,QFX (WSTAR fluxc), PBLH | grid previous (pxsfclay.F:408-410) | defaults 0,0,1000 | MISMATCH (B2) |
| QSFC / ITIMESTEP | carried; recompute only water/≤0/itimestep==1 (:370) | qsfc=-1, itimestep=1 always -> recomputed every step on land | MISMATCH (B2) |
| ZNT out (water) | INOUT | not written | MISMATCH (B2) |
| dz | 9.81 | 9.80665 (SA:604) | D |
| outputs | CHS..,RMOL,U10/V10 to PX LSM/history | handles only; Noah: A1 | A1/C1 |

## sf=91 old MM5 SFCLAY (c25, c61)
(a) OM:9062-9063 (Noah: then A1) / OM:9230-9231. WRF SD:2064-2077 `SFCLAY(...,mol,regime,...,qsfc,...,pblh,...)`.
| item | WRF | port (SA:710-732) | verdict |
|---|---|---|---|
| T3D, P3D, PSFC | t_phy dry, p_hyd, p_hyd_w(kts) | θ_m·π, p_nh[0], p_nh[0] | MISMATCH (A2, B1) |
| MOL in | carried; `IF(MOL<0) BR=MIN(BR,0)` (module_sf_sfclay.F:532), ZOL seed | mol=0, zol=0 (SA:716,724) | MISMATCH (B2) |
| QSFC in | carried (:460 recompute water/≤0 only) | -1 always | MISMATCH (B2) |
| PBLH, HFX/QFX (fluxc gust :513-515) | grid previous | 1000, 0, 0 | MISMATCH (B2) |
| ZNT/MOL/QSFC/HFX out | INOUT carried | only handles (+ust) written | MISMATCH (B2) |
| bl=3 consumer (c25) | BL_GFS gets SFCLAY PSIM/PSIH/GZ1OZ0/BR/WSPD | MYNN re-solve (A4) | A4 |

## sf=4 QNSE (c26)
(a) OM:8978-8995 `qnse_surface_step` (qnse_adapter.py:76-113) before LSM; Noah gets `surface_diagnostics=qnse_surface`
(OM:9121), so no MYNN re-solve. WRF SD:2292-2306 `QNSESFC(itimestep,ht,dz8w,p_phy,p8w,th_phy,t_phy,...)`.
| item | WRF | port | verdict |
|---|---|---|---|
| TH/T dry, Q2=TKE, dz 9.81, RHO, u/v | th_phy, t_phy, tke_pbl | dry θ, θ·π, 0.5·qke, ph/9.81, phy_prep rho (qnse_adapter.py:61-73) | MATCH |
| PMID / P8W / PSFC | p_hyd / p_hyd_w / p_hyd_w(kts) | state.p nonhydro / phy_prep p8w of p_nh / its k=0 (qnse_adapter.py:63,68,85) | MISMATCH (B1-small, ~10-20 Pa; FINDINGS psfc note) |
| THZ0/QZ0/UZ0/VZ0/AKHS/AKMS/QSFC/HFX/QFX/ITIMESTEP | carried INOUT | carried (QnseCarry, OM restart) | MATCH |
| Noah seed | CHXY own | diag.ch=CHS seeded into land_state.ch (NC:321) | MISMATCH (C3; inert for opt_sfc=1) |

## sf=10 TEMF (c27)
(a) OM:9036-9043 `coupling.temf.surface_step` (temf.py:62-93); Noah gets TEMF diagnostics (OM:9131, temf.py:115-129).
WRF SD:2521-2537 `TEMFSFCLAY(th3d=th_phy,p3d=p_phy,pi3d=pi_phy,rho,z,ht,psfc,...,hd_temf,te_temf,wm_temf)`.
th dry, p=p_hyd and psfc=p_hyd_w (temf.py:66-69), pi from p_nh, rho phy_prep, z mass heights (ph/9.81), ht, znt IN
(not written: WRF znt INTENT(IN), sf_temfsfclay.F:100), hfx prev, te/hd/wm carried, Noah ch/cm = land carry: MATCH.
No surface-side finding (PBL side uses p_nh for P3D — PBL lane).

## sf=1 revised MM5 SFCLAYREV (c02, c05-c11)
(a) OM:9013-9022 `revised_mm5_surface_step` -> `epsilon_surface_step` (epsilon_pbl.py:51-89); Noah gets sf1_noah_surface
(OM:9119); PBL handoff `surface_handoff` (surface_handoff.py:3-11). WRF SD:2118-2132.
T=dry θ·π(p_nh) rcp 287/1004.5 ✓, P=p_hyd ✓, PSFC=p_hyd_w ✓, dz=(ph1-ph0)/9.81 ✓, u/v A ✓, ZNT/UST/MOL/HFX/QFX/QSFC/
PBLH carried ✓, dx ✓; outputs UST/handles/HFX/QFX/QSFC/MOL/ZNT carried ✓, FM/FH/PSIM/PSIH/BR/WSPD/U10/V10/GZ1OZ0/
RMOL + pre-LSM TSK to PBL ✓, CHS/CHS2/CQS2 to Noah water T2/Q2 ✓, no-LSM history T2/Q2/U10/V10/HFX/LH/PSFC ✓: MATCH.
Notes: water_depth=50 placeholder (only used if shalwater_z0, not admitted); kernel floors PBLH≥1 m
(sfclay_revised_mm5.py:373) vs raw WRF (sf_sfclayrev.F90:339) — inert (fluxc=0 when PBLH=0 at cold start) (D).
sf=1 with sf_urban_physics=2/3 (c34/c56/c57) uses urban_surface_layer.revised_surface_layer (NC:279-281): UNVERIFIED.

## sf=5 MYNN-SL (release default) — B39/B39b verify only
(a) Noah: inside NC:283-286 on `_build_column_view` (NH:116-191); non-Noah: `surface_adapter` (physics_couplers.py:
2826-2873) on `_surface_column_view` (:2769-2823). WRF SD:2432-2444 `SFCLAY_mynn(u,v,t_phy,qv,p_phy,dz8w,th_phy,rho,
...,psfc,...,znt,ust,pblh,mavail,...,mol,...,hfx,qfx,...,qsfc,...,snowh,...,dx,...,itimestep,ch,qcg,spp,...)`.
T3D dry t_air ✓, P3D=p_hyd ✓, PSFC=p_hyd_w ✓, dz 9.81 ✓, RHO phy_prep ✓, SNOWH pre-Noah (land_state.snowh) ✓,
first-step (itimestep==1) UST/MOL/QSFC seeds ✓, B39 MOL/HFX/QFX/QSFC/PBLH/dx in view (NH:177-189) and written back
blended (NC:540-552, NH:336-339) ✓, B39b constants CP=1004.5, EP1=0.60836, EP2=0.62175, R_D 287 (surface_constants.py:
25-37) ✓, QCG unused by WRF (module_sf_mynn.F only declares) ✓, Noah ch/cm not seeded (diag has no ch/cm) ✓.
History: U10/V10 MYNN ✓, ISWATER bulk T2/Q2 ✓, land Noah blend ✓. MATCH. Minor: qv0=max(qv,0) (surface_layer.py:636)
vs raw QV1D (module_sf_mynn.F:319) — negative-qv cells only (D). Non-ISWATER water points (lakes without lake model):
port T2/Q2 = MYNN diag (NC:506-509) vs WRF ELSE branch Noah FVEG blend at a skipped point (SD:3466-3470) (C2).

## sf=0 (none; c01, c39)
WRF: `if (sf_sfclay_physics .eq. 0) return` (SD:1490) before PSFC, sfclay AND LSM: UST/HFX/QFX/LH/ZNT/QSFC/T2/Q2/
U10/V10/TH2/PSFC keep initial values; no LSM.
Port: sf=0 is admitted (OM:7134, scheme_catalog.py:200) and not skipped except km_opt=5 (OM:9219): non-Noah falls
through to `surface_adapter` = MYNN-SL (OM:9233) writing ustar/handles/mol/hfx/qfx/qsfc; with use_noahmp the whole
Noah-MP + MYNN-SL step runs (OM:9113). History PSFC/T2/Q2/U10/V10/TH2 are held correctly
(io/disabled_surface_history.py, daily_pipeline.py:402-408) but UST/HFX/LH/QFX/(QSFC) are not. -> B4.

---------------------------------------------------------------------------------------------------------------
## RANKED FINDINGS
**A1 (HIGH, read both sides)** sf=2/3/7/91 + Noah-MP: selected surface layer discarded, MYNN-SL re-solved and
written (see GLOBAL). Composites c03,c04,c25,c43,c61. Owner: surface caller (OM/NC).

**A2 (HIGH)** Moist temperature input to GFS/PX/old-MM5/Janjic: port T3D=θ_m·π (SA:603; MYJA:83), PX/Janjic TH=θ_m
(SA:663; MYJA:98) vs WRF dry t_phy/th_phy (BSU:4834,4855; P1:657). θ_m/θ = 1+1.608 qv -> THX/TH1D/THLOW +3.7 K at
qv=8 g/kg: HFX biased ≈ -FLHC·3.7 K (tens of W/m²), Ri/regime shifted, rhox/QGH off. Live for sf=3/7/91 without
Noah-MP (and every flux into bl=5 MYNN in that case), for the Janjic solve inside the MYJ PBL (c03, see A3), and as
the UST seed under Noah-MP. Same moist T is reused by `_flux_handles_from_hfx_qfx` (SA:692-700) for rhox/thx and by
`sfclay_revised_mm5_adapter` default inputs (SA:633; SSiB passes its own inputs). Same defect class as the fixed
Tiedtke T_PHY (e213b65c4). Fix: build `_surface_layer_bottom_inputs` from `_dry_theta_view` (+ p_hyd/psfc/dz8w as in
epsilon_surface_step) and pass dry θ to PX/Janjic. Oracle: pristine SF_GFS/PXSFCLAY/SFCLAY/MYJSFC on dumped
phy_prep inputs (the MYNN oracle dumps t_phy/th_phy/p_phy/psfc, SD:2370-2410) + mutant feeding θ_m.

**A3 (HIGH, cross-lane MYJ PBL; origin = sf2 handoff)** MYJ PBL re-solves Janjic on the post-LSM state with
re-seeded INOUT (QSFC=qv0 mixing ratio, THZ0=θ_m, QZ0=q_spec, UZ0=VZ0=0, AKHS/AKMS=0: MYJA:128-135, sfclay_janjic.py:
413) and passes `elflx=sfc["flx_lh"]`, `chklowq=clip(mavail)` (MYJA:212-213). Re-solve land QZ0=QS=qv0 gives
QFX=ρ·AKHS·mavail·(qv0-qv0/(1+qv0)) ≈ 1 % of physical LH; then MYJPBL land QSFC=QLOW+ELFLX/(XLV·CHKLOWQ·AKHS·ρ)
(bl_myj.py:548-550 = module_bl_myjpbl.F:547-557) ≈ qv0 -> land moisture flux into the PBL ≈ 0. WRF: ELFLX=grid%lh
= Noah-MP LH (PD:1457), CHKLOWQ=1 (SD:3429), THZ0/QZ0/AKHS/AKMS from THIS step's MYJSFC (SD:2214-2219). Plus TH/T
moist (A2) and PSFC=p[0] (B1: THSK=THZ0_land +0.1-0.26 K, bl_myj.py:527). Composite c03 (T2 margin .73 vs MYNN .47).
Fix: carry MYJSFC INOUT/outputs (QnseCarry pattern), pass post-LSM LH and CHKLOWQ=1. Oracle: pristine MYJSFC ->
noahmplsm -> MYJPBL one step; mutant = elflx from re-solve.

**A4 (HIGH, cross-lane GFS PBL; origin = no sf3/sf91 handoff)** bl=3 with sf≠1 goes to OM:9540 without
`surface_forcing`; `_pbl_surface_forcing(state, None)` (SA:1561) re-solves MYNN-SL on the grid-less legacy view
(θ_m, p_nh, psfc=p[0], dx 3000) and feeds its BULK hfx/lh (SA:1584-1589; land NOT Noah-MP), psim/psih/br/u10/v10,
wspd=|V1|, gz1oz0=log(za/znt) (SA:1818), tsk=post-LSM. WRF BL_GFS (PD:1405-1420): PSIM/PSIH/GZ1OZ0/BR/WSPD/UST
from SF_GFS (or SFCLAY), HFX/QFX post-LSM (Noah land), TSK=tskold. Composites c04, c25. Fix: extend
`surface_handoff` to sf=3/91 (GFS: PSIM=PM, PSIH=PH, GZ1OZ0=log(Z1/z0), BR=RB, WSPD=WIND) and hfx/qfx from the
post-LSM state. Oracle: pristine SF_GFS -> noahmplsm -> BL_GFS.

**B1 (MED)** Pressure/PSFC staging for sf=2/3/7/91 (+QNSE small): PSFC = lowest-level p (SA:616, MYJA:110) or
p+ρg dz/2 (SA:747) or nonhydro p8w (qnse_adapter.py:68,85) vs p_hyd_w(kts) (SD:1988); P3D/PMID = p_nh vs p_hyd (P1:637).
~150-300 Pa for 2/7/91 -> THGB/THSK +0.1-0.26 K, QSFC/RHOX/T2 conversions; QNSE ~10-20 Pa. Fix: reuse
`_wrf_hydrostatic_pressure_from_state` like SF1/TEMF.

**B2 (MED)** Carried INOUT not threaded: PX hfx/qfx=0 (no WSTAR), pblh=1000, qsfc=-1 + itimestep=1 (land QSFC
recomputed each step; pxsfclay.F:370) (SA:666-678); old-MM5 mol=0 (BR clamp :532 and z/L seed lost), zol=0,
qsfc=-1, pblh=1000, hfx/qfx=0 (SA:716-727); GFS/PX/91 do not write back ZNT (water z0), MOL, QSFC, HFX/QFX;
Janjic THZ0/QZ0/UZ0/VZ0/AKHS/AKMS/QSFC re-seeded each call and NTSD==1 branch never taken. Fix: thread these
leaves like SF1 (state.mol/hfx/qfx/qsfc/pblh/roughness_m) and a Janjic carry.

**B3 (MED-LOW)** Janjic Q2 doubled: q2=qke=2·TKE (MYJA:128) but kernel applies WRF's Q2K=2·Q2 again
(sfclay_janjic.py:90) -> PBLH from 4·TKE vs 2·TKE (myjsfc.F:246; PD:1455 TKE_MYJ=tke_pbl). Changes BTGH convective
gust when PBLH>1000 m (sea and land). Fix: pass 0.5·qke (as qnse_adapter.py:73 does).

**B4 (MED for sf0 with LSM/PBL; LOW for c01/c39)** sf=0 runs MYNN-SL (and Noah-MP if use_noahmp) where WRF runs
nothing (SD:1490). Prognostic difference if bl≠0 (PBL gets fluxes instead of held grid values) or lsm≠0 (TSK/soil
evolve); c01/c39 (bl0/lsm0) only UST/HFX/LH/QFX history. Fix: skip the surface slot for sf=0 (and refuse/skip LSM),
carry held UST/HFX/QFX/LH.

**C1** History for sf=2/3/7/91: U10/V10/water T2,Q2/HFX/LH from MYNN (Noah, NC:503-510) or the output-time MYNN
re-solve (non-Noah, OM:10880-10886); sf=2 non-Noah seeds zero history (OM:7886-7892) that the step never fills
except U10E/V10E — T2/Q2/U10/V10/LH may stay 0 (UNVERIFIED writer path). Fixed with A1 + per-scheme history write.
**C2** Non-ISWATER water points (lakes, no lake model) T2/Q2 (see sf=5). **C3** QNSE CHS seeded into Noah
land_state.ch (NC:321) — WRF CHXY is Noah's own; inert for opt_sfc=1 (recomputed), wrong seed for opt_sfc=2.
**D** g=9.80665 dz in SA:604/MYJA:86; znt/ust clamps SA:671-672/720/751; MYNN qv0≥0; SF1 PBLH≥1 floor.

Flags: isfflx, iz0tlnd, isftcflx, scm_force_flux, shalwater_z0 are hard-coded at WRF defaults in these callers and
guarded by the namelist honour check (namelist_binding.py:396-583) — not re-verified per scheme (UNVERIFIED).
