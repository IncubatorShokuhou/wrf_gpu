# t-calleraudit — PBL group B caller-wiring audit (MYNN 5, QNSE 4, TEMF 10, GBM 12, EEPS 16 / KEPS 17)

Static read-only audit, 2026-10-15. Port = worktree `lane/t-calleraudit` (= main ea6c83d25 + notes).
WRF = <USER_HOME>/src/wrf_pristine/WRF (4.7.1). Abbrev: OM = src/gpuwrf/runtime/operational_mode.py,
PC = src/gpuwrf/coupling/physics_couplers.py, PBLD = phys/module_pbl_driver.F, FRK1 = dyn_em/module_first_rk_step_part1.F,
MDRV = phys/module_bl_mynnedmf_driver.F, MCORE = phys/module_bl_mynnedmf.F, ADDT = phys/module_physics_addtendc.F.

WRF caller facts used by every scheme below (quoted once):
- FRK1:1131 `,LH=grid%lh ,LOWLYR=grid%lowlyr ,P8W=grid%p_hyd_w` and FRK1:1133 `,PSIM=psim ,P_PHY=grid%p_hyd`
  -> every PBL scheme gets HYDROSTATIC p_phy/p8w; PI_PHY (FRK1:1132) is the nonhydrostatic phy_prep Exner.
- PBLD:1033 `PSFC(I,J)=p8w(I,kms,J)` (hydrostatic surface face); PBLD:1021 `TSKOLD(i,j)=TSK(i,j)`.
- Release defaults (_fast_defaults.py): rad_rk_tendf=1 -> `source_leaf_mode`; GPUWRF_PHYS_TEND_RK_WRF=1; MYNN flags
  CLOUDMIX, QNI_MIXING, PHY_EXNER, FLTV_WRF, SFC_WSPD, RHOSFC_WRF, TKEPROD_UP, FP32_COLUMNS all =1.

---------------------------------------------------------------------------------------------------------------------
## 1. bl_pbl_physics=5 MYNN-EDMF (PROD / release default)

(a) Default branch: OM:9545-9594 (`elif bl_opt == DEFAULT_BL_PBL_PHYSICS:` -> `source_leaf_mode` ->
`mynn_adapter_with_source_leaves` PC:2663 -> `_phys_tend_rk_wrf_enabled()` raw-rate RK fold, entry state restored).
WRF call: PBLD:1628-1918 (`CASE (MYNNPBLSCHEME)`, `CALL mynnedmf_driver` PBLD:1715) -> MDRV:340-660 -> MCORE `mynnedmf`.
Tendency application: ADDT:704-760 (phy_bl_ten MYNN case) + PBLD:1866-1918 (scalar_tend for QNI/QNC/QNWFA/QNIFA/QNBCA).

(b) Wiring table

| item | WRF | port | verdict |
|---|---|---|---|
| th | PBLD:1719 `th=th_phy` (dry theta) | PC:2240 theta/(1+rv/rd*qv) | MATCH |
| p | FRK1:1133 p_hyd | PC:2243 `_wrf_hydrostatic_pressure_from_state` | MATCH |
| exner / T3D | pi_phy (P+PB), t_phy=th*pi | PC:2251 PHY_EXNER from state.p; kernel tk=th*exner (mynn_pbl.py:365) | MATCH |
| rho | phy_prep rho=(1+qv)/alt | PC:2244 `_wrf_phy_prep_rho_from_state` | MATCH (ALT rebuilt) |
| dz | dz8w, g=9.81 | PC:2245 `_surface_dz_from_state` (WRF_PHYSICS_G) | MATCH |
| u,v | u_phy/v_phy A-grid | PC:2259-2260 `_u_mass/_v_mass` | MATCH |
| **w** | PBLD:1719 `w=w` = FRK1 `W=grid%w_2`; MDRV `w1(k)=w(i,k,j)` k=kts..kte = BOTTOM FACE of each layer (incl. surface face) | PC:2261 `column(_w_mass(...))` = 0.5*(w_k+w_k+1) mass average | **MISMATCH (B1)** |
| qv,qc,qi | qv_curr/qc_curr/qi_curr, FLAG_QC/QI=F_QC/F_QI | PC:2264,2271-2272 | MATCH |
| qs | qs passed, used only in thlv (MCORE:1007); not mixed (MCORE:1063 `kzero1 replaces sqs`, :4451 `.AND. .false.`); RQSBLTEN not passed by FRK1 | qs in column (PC:2273), used in thlv (mynn_pbl.py:430), never mixed/written | MATCH |
| qni | QNI_CURR + F_QNI; RQNIBLTEN -> scalar_tend(P_QNI) PBLD:1869-1876 (mixscalars=1) | ni=column(state.Ni) PC:2276; OM:9579-9590 `_pbl_scalar_rate` -> coupled scalar_tend (OM:8790) gated by `_pbl_qni_active` | MATCH |
| **qnc, qnwfa, qnifa, qnbca** | FLAG_QNC/QNWFA/QNIFA/QNBCA from F_*; mixed in MCORE:4549/4610/4675/4740 and DMP_mf; PBLD:1879-1917 `scalar_tend(i,k,j,P_QNC)=RQNCBLTEN(i,k,j)` etc. | not in MynnPBLColumnState, no rate, no mixing anywhere (grep qnc/nwfa in mynn*/phys_mynn*: none) | **MISMATCH (A1)** for mp with those scalars |
| ozone | commented out in PBLD:1724 (`!ozone=ozone`) | not mixed | MATCH |
| hfx/qfx | grid%hfx/qfx post-LSM; **MDRV:381-397 clamps hfx to [-600,1200], qfx to [-3e-4,9e-4]** before MYNN | FLTV_WRF: noahmp_coupler.py:434-461 / surface_layer.py:1056 blended HFX/QFX, no clamp anywhere (grep 1200/9e-4: none) | **MISMATCH (B2, rare)** |
| flt/flqv/fltv | MCORE:859-876 from hfx1/qfx1, rho1(kts), cpm, th_sfc=ts/ex1 | surface_layer.py:514-535 same statements | MATCH (minus clamp) |
| ust, wspd | grid%ust, grid%wspd (SFCLAY_mynn) | state.ustar; SFC_WSPD state.sfc_wspd | MATCH |
| ts, ps | tsk, psfc=p_hyd_w(kms) | flux.t_skin; RHOSFC_WRF psfc = hydrostatic face (PC:2195, 2780) | MATCH |
| rhosfc | MCORE:3960 psfc/(R_d*(tk+p608*qv)) (qv mixing ratio) | mynn_pbl.py:461-467 | MATCH |
| ch, qsfc, znt, uoce/voce, rthraten | passed but unused (ch only in comments, znt print; topdown=0, edmf_dd=0 MCORE:335-337) | not passed | MATCH (dead in WRF) |
| prev-step el_pbl | MDRV:427 el1=el_pbl (initflag=0) -> TKEprod_up | PC:2269 el_pbl (TKEPROD_UP=1) | MATCH |
| prev-step qc_bl/qi_bl/cldfra_bl, sh3d | read MDRV:411-416, but CASE(2) recomputes before use (MCORE:3601-3760); *_old unused in DMP_mf | zeros in, recomputed (mynn_pbl.py:1868-1898) | MATCH (cloudpdf=2 only; others refused) |
| initflag | PBLD:1642 itimestep==1 & !restart | `_mynn_state_with_first_call_qke` PC:2626 | MATCH (nest start UNVERIFIED) |
| switches | Registry defaults EDMF=1, MOM=1, TKE=0, mixscalars=1, cloudmix=1, mixqt=0, cloudpdf=2, closure=2.6, tkeadvect=F | scheme_catalog.py:857-915 refuse non-default; others refused by namelist_binding.py:577-588 | MATCH |
| post_run conversion | MDRV:878-955 rq=sqv/(1-sqv), qc=sqc*(1+sqv_new) | phys_mynn_cloudmix.py:110-112 identical | MATCH |
| RTH/RU/RV/RQV/RQC/RQI application | ADDT:704-760 add_a2a/add_a2c into rt/ru/rv/moist_tendf (adv_moist_cond=T) | OM:8645-8740 coupled RK fold, theta_m conversion, add_a2c edges; OM:9573-9578 moist raw | MATCH |
| PBLH | grid%pblh | PC:2572 `_with_mynn_pblh` | MATCH |
| KPBL | MDRV kpbl output -> grid%kpbl (GF/G3 read it) | not published; G3 uses `_kpbl_bulk_richardson` (scan_adapters.py:1391-1394) | cross-lane note (cu) |

(c) FINDINGS

**A1 — MYNN drops WRF's QNC/QNWFA/QNIFA/QNBCA PBL mixing (class A, non-PROD, HIGH).**
WRF: FRK1:1298 `,QNC_CURR=scalar(ims,kms,jms,P_QNC), F_QNC=f_qnc`, FRK1:1203-1205 QNIFA/QNWFA/QNBCA_CURR+F_*;
PBLD:1720-1722 `&qnc=qnc_curr,qni=qni_curr, &QNWFA=qnwfa_curr,QNIFA=qnifa_curr,QNBCA=qnbca_curr`;
MCORE:4549 `IF (bl_mynn_cloudmix > 0 .AND. FLAG_QNC .AND. bl_mynn_mixscalars > 0) THEN` (same :4610/:4675/:4740);
PBLD:1879-1886 `IF (PRESENT( qnc_curr ) .AND. Flag_qnc) THEN ... scalar_tend(i,k,j,P_QNC)=RQNCBLTEN(i,k,j)`
(and QNWFA :1889, QNIFA :1899, QNBCA :1909); coupled with mut in module_em.F:2479 ("4d couple scalar tendencies").
check_a_mundo.F:2516-2518 sets scalar_pblmix=0 under MYNN mixscalars=1, so MYNN is the ONLY PBL mixer of these scalars.
Port: MynnPBLColumnState has only `ni` (PC:2276); OM:9573-9590 couples qv/qc/qi/Ni only; no Nc/nwfa/nifa leaf is mixed.
Affects every admitted bl=5 + mp with Registry scalar qnc: 9, 11, 14, 16, 26, 28, 38, 40, 51, 52, 53, 56 (scheme_catalog.py:171),
plus qnwfa/qnifa/qnbca for 28/38. PROD (mp=8: scalar qni,qnr only) unaffected.
Fix sketch: add Nc (and nwfa/nifa when present) to the column view exactly like `ni` (specific? no: WRF passes qnc1 raw,
MDRV:456-475), mix them in the cloudmix kernel with s_awqnc/s_awqnwfa/s_awqnifa MF terms (DMP_mf already outputs
s_awqnc1.. in WRF), return `(post-entry)/dt` rates and add `pbl_moist_raw["Nc"/"nwfa"/"nifa"]` gated by an `F_QNC` table
like `_pbl_qni_active`. Oracle: pristine MDRV single-column driver with FLAG_QNC/QNWFA/QNIFA=.true. on a real mp=28
column (RQNCBLTEN, RQNWFABLTEN, RQNIFABLTEN dumps); mutant = drop the rate -> gate fails.

**B1 — MYNN DMP_mf receives mass-averaged w instead of WRF's bottom-face w_2 (class B, PROD default, HIGH).**
WRF: FRK1:1142 `,W=grid%w_2`; PBLD:1719 `&u=u_phy,v=v_phy,w=w,th=th_phy`; MDRV:348 `w1(k) = w(i,k,j)` for k=kts..kte
(w-staggered array -> bottom face of each mass layer, k=1 is the terrain surface face); MCORE:5858-5864
`zagl = zw1(k) + half*dz1(k) ... wpbl = w1(k); if (w1(k) < zero)wpbl = 2.*w1(k); maxw = max(maxw,abs(wpbl))` ->
MCORE:5877-5879 `maxw = max(zero, maxw - one); Psig_w = max(zero, one - maxw)` tapers/turns off the whole EDMF plume.
Port: PC:2261 `column(_w_mass(state, dtype=wind_dtype))`, PC:444-449 `return 0.5 * (w[:-1, :, :] + w[1:, :, :])`;
consumed at mynn_edmf.py:294-297 (`wpbl = jnp.where(w < 0.0, 2.0 * w, w)`) — identical formula, different operand.
Effect: the surface-face kinematic w (u·grad h, nonzero over slopes; downslope doubled) is halved/averaged away, so over
steep terrain (Canary/Tenerife, Swiss) the port tapers MF less often than WRF -> stronger/more frequent EDMF plumes on
slopes; flat terrain nearly unaffected. Changes the default PROD/Swiss program (release identity) -> needs GPU reruns.
Fix sketch: PC:2261 pass `column(jnp.asarray(state.w[:-1], ...))` (faces 0..nz-1 = WRF kts..kte); `w` is used only by
`_dmp_setup` (mynn_edmf.py:276; mynn_pbl.py:1820, phys_mynn_columns.py:53), so nothing else changes. Oracle: the existing
WRFGPU2_ORACLE_PBL_IN dump already writes `w` with stagger 'w' (PBLD:1667-1668); unit gate with w_sfc=-0.6, w_face2=+0.4
(WRF maxw=1.2 -> Psig_w=0.8 vs port Psig_w=1.0); mutant = revert to `_w_mass`.

**B2 — MYNN driver hfx/qfx sanity clamps not reproduced (class B, PROD, rare, HIGH on absence).**
WRF MDRV:381-397 `hfx1 = hfx(i,j); if (hfx1 > 1200.) hfx1 = 1200.; if (hfx1 < -600.) hfx1 = -600.; qfx1 = qfx(i,j);
if (qfx1 > 9e-4) qfx1 = 9e-4; if (qfx1 < -3e-4) qfx1 = -3e-4` (only the PBL's copy). Port forms flt/flqv/fltv from the
unclamped blended HFX/QFX (noahmp_coupler.py:434-461, surface_layer.py:1050-1060; only `_finite_clip` ±5 K m/s in
coupling/driver.py:999). Fires only for |HFX|>600-1200 W/m2 or LH outside [-750,2250] W/m2 (strong cold-air outbreaks
over warm water, hurricanes, fire). Fix: clamp hfx/qfx inside `mynn_driver_surface_fluxes` before forming flt/flqv
(both producers call it). Oracle: column with hfx=1500, qfx=1.2e-3 vs pristine mynnedmf_driver; mutant removes clamp.

**Notes (not findings):** (i) KPBL from MYNN is not published; GF (scan_adapters.py:1265) and G3 cu=5 (:1391-1394)
substitute a bulk-Richardson KPBL where WRF passes the PBL's grid%kpbl -> forward to the cumulus auditors.
(ii) mp=5 Ferrier (adv_moist_cond=.false., module_physics_init.F:4495): WRF ADDT:720-735 puts RQCBLTEN+RQIBLTEN into
scalar_tendf(P_QT), not moist qc/qi; port always uses moist qc/qi — UNVERIFIED how the port's mp=5 path treats QT
(microphysics lane). Same applies to EEPS/KEPS/QNSE/BouLac under mp=5.

---------------------------------------------------------------------------------------------------------------------
## 2. bl_pbl_physics=4 QNSE (+ mandatory sf=4 QNSESFC, optional mfshconv=1 MFSHCONVPBL)

(a) Default branch: surface OM:8978-8983 `qnse_surface_step`; PBL OM:9462-9486 `qnse_pbl_step(..., apply_tendencies=not raw_only)`,
raw_only = source_leaf_mode and PHYS_TEND_RK -> rates RTH/RU/RV/RQV/RQC to RK. WRF: surface_driver.F:2292 `CALL QNSESFC`,
PBLD:1514-1565 (`MFSHCONVPBL` :1520, `qnsepbl` :1539), ADDT:605-647.

| item | WRF | port | verdict |
|---|---|---|---|
| PMID / PINT (PBL + MF) | PBLD:1521,1541 `PMID=p_phy,PINT=p8w` = p_hyd / p_hyd_w | qnse_adapter.py:67,73,75 `pressure=state.p` (P+PB), `pint=_wrf_phy_prep_p8w(pressure,...)` | **MISMATCH (B3)** |
| PMID / PINT (QNSESFC) | surface_driver.F:2293 `p_phy,p8w` = FRK1 surface call `P_PHY=grid%p_hyd`, `P8W=grid%p_hyd_w` | same `_geometry` (qnse_adapter.py:84-89) | **MISMATCH (B3)** |
| TH, T, EXNER | th_phy, t_phy, pi_phy (P+PB) | qnse_adapter.py:65-70 | MATCH |
| DZ, HT, CORF | dz8w (g 9.81), grid%ht, grid%f | z=ph_total/9.81, ht=z[0], metrics.f | MATCH |
| QV, CWM | qv_curr, CWM=qc_curr (no qi) | qv, qc | MATCH |
| TKE | tke_pbl | 0.5*state.qke (port stores 2*TKE) | MATCH |
| HFX, ELFLX=LH, UST, QSFC | grid%hfx, grid%lh (post Noah-MP), ust, qsfc | state.hfx; carry.lh from Noah-MP history (OM:9186-9188) / land.lh (OM:9469-9471) | MATCH |
| CHKLOWQ | surface_driver.F:3429 Noah-MP sets 1.0; undefined for no-LSM | `chklowq=1.` (qnse_adapter.py:136) | MATCH (Noah-MP); UNVERIFIED (sf_surface_physics=0) |
| SNOW | grid%snow | `land.sneqv` (OM:9468) | MATCH (Noah-MP) |
| **SICE** | PBLD:1545 `SICE=xice`; qnsepbl.F:497 `IF(SNOW(I,J)>0..OR.SICE(I,J)>0.5) QFC1=QFC1*RLIVWV` | `sice=0.` default, caller never passes it (qnse_adapter.py:117,136; OM:9475-9477) | **MISMATCH (B5)** |
| RHO, TSK, ZNT, WTHV_MF, LM_BL89 | passed; RHOK recomputed from PMID (qnsepbl.F:474); TSK/ZNT unused; LM_BL89 dead in MIXLEN, WTHV_MF commented (:948) | not passed | MATCH (dead in WRF) |
| tendencies | ADDT:605-647: RTH, RU, RV, RQV, RQC (no RQI) | OM:9480-9483 qv, qc + th/u/v | MATCH |

**B3 — QNSE/MFSHCONV/QNSESFC get nonhydrostatic P+PB and a phy_prep-extrapolated p8w instead of grid%p_hyd /
grid%p_hyd_w (class B, HIGH).** WRF evidence above (FRK1:1131/1133, PBLD:1521/1541, surface_driver.F:2293).
Port qnse_adapter.py:67 `pressure=jnp.asarray(state.p,f)`, :73 `pint=_wrf_phy_prep_p8w(pressure,z,metrics.fnm,metrics.fnp)`,
:75 `p_mid=pressure,p_int=pint`. Used for RHOK=PMID/(R_D*T*(1+..)) (qnsepbl.F:474, akh/akm density weighting), PSFC=PINT(LOWLYR)
(:444, :507, :557: THSK from HFX, sea QSFC), QSAT, MF plume pressure. Magnitude = nonhydrostatic perturbation + surface
extrapolation error (Pa to tens of Pa) -> small but systematic. Same WRF contract the port already honours for MYNN
(PC:2243), GBM (urban_pbl_hook.py:14) and CAM-UW (scan_adapters.py "WRF passes p_hyd/p_hyd_w here").
Fix: in `_geometry` use `_wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=f32)` for p_mid/p_int,
keep exner from state.p. Oracle: existing QNSE column oracle fed with FRK1 operands (p_hyd/p_hyd_w) from a pristine
d01 dump; mutant = nonhydro pressure. Affects bl=4/sf=4 composites only.

**B5 — QNSE SICE hard-wired to 0 (class B, low, HIGH).** WRF PBLD:1545 `SICE=xice`; port never threads XICE. Only sea-ice
land points (XLAND=1, XICE>0.5) without snow: QFC1 misses RLIVWV, so the diagnosed land QSFC/QZ0 there differs.
Fix: pass `sice=next_state.xice` (or the sea-ice fraction leaf) in OM:9475.

---------------------------------------------------------------------------------------------------------------------
## 3. bl_pbl_physics=10 TEMF (+ mandatory sf=10)

(a) Default branch: OM:9447-9461 `temf.pbl_step` -> source_leaf_mode+PHYS_TEND_RK raw rates (u,v,theta,qv,qc), entry restored.
WRF: PBLD `CASE (TEMFPBLSCHEME)` -> `temfpbl(...)` (P3D=p_phy, P3DI=p8w, QI3D, RQIBLTEN, FLAG_QI); ADDT:974-1009.

| item | WRF | port | verdict |
|---|---|---|---|
| **P3D / P3DI** | `P3D=p_phy,P3DI=p8w` = p_hyd/p_hyd_w; used in rsat (module_bl_temf.F:695,726,924) and beta2 (:1022-1029) | temf.py:41 `p=state.p`, :48 `p8w=_wrf_phy_prep_p8w(p,...)`, :54; pbl_step :134 `kw=phy_inputs(...)` | **MISMATCH (B3, same class)** |
| surface step pressure | p_hyd/p_hyd_w | temf.py:67-70 overrides with hydrostatic | MATCH |
| TH3D, T3D, PI3D, RHO, Z, HT, fCor | th_phy, t_phy, pi_phy, phy_prep rho, grid%z, ht, f | temf.py:39-58 | MATCH |
| QI3D / RQIBLTEN | passed but unused in temf2d (qix/rqibltenx only declared; init :1451 `rqiblten=0.`) | qi not passed, no qi rate | MATCH |
| TSK, ZNT, HFX, QFX | TSKOLD=TSK, znt post-LSM, grid%hfx/qfx | state.t_skin, roughness_m, hfx, qfx | MATCH |
| tendencies | RTH, RU, RV, RQV, RQC, RQI(=0) | OM:9456-9460 qv, qc | MATCH |

Note temf.py:66 comment "PBL uses p_phy" is right, but p_phy at the PBL call is ALSO grid%p_hyd (FRK1:1133) -> fix
pbl_step's `phy_inputs` to the hydrostatic pair exactly as `surface_step` already does (temf.py:67-70).

---------------------------------------------------------------------------------------------------------------------
## 4. bl_pbl_physics=12 GBM (requires sf=1)

(a) Default branch: OM:9522-9542 (`sf1_surface_forcing is not None and bl_opt in PBL_SCAN_ADAPTERS`) -> scan_adapters.py:1890
`gbm_pbl_adapter(_with_sources=True)` -> raw rates theta/qv/u/v/qc to RK. WRF: PBLD `CASE (GBMPBLSCHEME)` `CALL gbmpbl(...,
RTHRATEN=rthraten, ...)`, ADDT:1011-1049.

| item | WRF | port | verdict |
|---|---|---|---|
| P3D, PSFC | p_hyd, p_hyd_w(kms) | urban_pbl_hook.py:14 hydrostatic profiles | MATCH |
| T3D, PI3D, DZ8W, Z, QV, QC | phy_prep | urban_pbl_forcing | MATCH |
| QI3D / RQIBLTEN | passed, unused in GBM2D | not passed | MATCH |
| PSIM/PSIH/GZ1OZ0/UST writeback | raw SF1 psi + GZ1OZ0, UST INOUT | already fixed (t-c02nan) | MATCH |
| 2-pass driver | module_bl_gbmpbl.F:251-279 (TKE pass1, U/V avg, scalars pass2) | bl_gbm.py:533-547 | MATCH |
| **RTHRATEN** | PBLD `RTHRATEN=rthraten` (FRK1 `RTHRATEN=grid%rthraten`); gbmpbl.F:243 `t2dten_ra(i,k) = RTHRATEN(i,k,j)*PI3D(I,K,J)`; used in cloud-top buoyancy :757-760 and PBL-height radnnll :972-985 | gbm_columns has no rthraten operand (bl_gbm.py:550); bl_gbm.py:462-467 `rad = jnp.where(..., 0.0,  # RTHRATEN is zero in the staged/current operational path. 0.0)`; `_pblhgt_simple` has no radnnll | **MISMATCH (B4)** |
| tendencies | RTH, RU, RV, RQV, RQC, RQI(=0) | theta, qv, u, v, qc | MATCH |

**B4 — GBM ignores the held radiative heating RTHRATEN (class B, HIGH).** WRF passes grid%rthraten and uses it in the
cloud-top TKE source `bouyan(k) = bouyan(k) - rttenx(k)*(presfl(k+1)-presfl(k)) * rrhoxfl(k) * rexnerfl(k) / thvx(k)`
(gbmpbl.F:757-760) and in the cloud-layer depth `radnnll = rttenx(k)*(presfl(k+1)-presfl(k))/(rhoxfl(k)*thvx(k)*exnerfl(k))`
(:972-974, :985). The port's operational RTHRATEN is NOT zero (carry.rthraten), so the comment is false in operation.
Affects bl=12 composites with stratocumulus/cloud tops (radiatively driven entrainment/TKE) — the GBM design target.
Fix: thread `rthraten=next_carry.rthraten` (RK-held, same as EEPS/CAM-UW, OM:9439/9493) into gbm_pbl_adapter ->
gbm_columns, multiply by pii, port the two rttenx terms and the radnnll branch of the PBL-height search.
Oracle: the v018 standalone Fortran GBM driver (proofs/v018/run_gbm_pbl12_parity.py) with nonzero RTHRATEN on a capped
cloudy column; mutant = zero rthraten.

---------------------------------------------------------------------------------------------------------------------
## 5. bl_pbl_physics=16 EEPS / 17 KEPS (require sf=1)

(a) Default branch: OM:9436-9446 `epsilon_pbl_adapter` (epsilon_pbl.py:120) -> rates always to RK, entry u,v,theta,qv,qc,qi
restored. Surface: `epsilon_surface_step` (hydrostatic, MATCH). WRF: PBLD:1920-1948 `eeps(...)`, KEPS call PBLD (`CALL KEPS(...)`);
ADDT:762-815 (EEPS: RQV,RQC,RQI), ADDT:816-858 (KEPS: RQV,RQC only).

| item | WRF | port | verdict |
|---|---|---|---|
| **EEPS P3D** | `P3D=p_phy` (p_hyd) -> eepsilon.F:277 `prsl(i,zz) = p3d(i,k,j)` | epsilon_pbl.py:133 `p=state.p`, :146 `pressure=cols(p)` | **MISMATCH (B3)** |
| EEPS PSFC | PBLD:1033 p_hyd_w(kms) | :145 hydrostatic psfc | MATCH |
| **KEPS P8W** | `P8W=p8w` (p_hyd_w) -> keps.F:223 radiative flux `flux*1004.5/9.81*(p8w(k)-p8w(k+1))` | :153 `p8w=_wrf_phy_prep_p8w(p,...)` from P+PB | **MISMATCH (B3)** |
| T3D/TH, PI3D, RHO, DZ8W | t_phy, pi_phy, rho, dz8w | theta*exner, exner(P+PB), phy_prep rho, ph/9.81 | MATCH |
| QR/QS/QG (EEPS) | qr/qs/qg_curr | passed | MATCH |
| RMOL, WSPD, BR, FM/FHH (KEPS PSIM=fm,PSIH=fhh) | SF1 outputs | EpsilonSurfaceForcing (rmol, wspd, br, fm, fh) | MATCH |
| XLAND/TSK/MOL/XTIME | EEPS xland unused; KEPS tsk/mol only in unused c_nsquare args; XTIME=itimestep (≤2 init) | step counter `epsilon_pbl_step` | MATCH |
| RTHRATEN | grid%rthraten (fresh on radiation steps: radiation precedes PBL) | `next_carry.rthraten`; RRTMG driver refresh precedes PBL (OM:8909) | MATCH for RRTMG; UNVERIFIED for non-RRTMG ra (legacy refresh at OM:9983-10003 runs after the PBL slot -> one-step lag on radiation steps) |
| tendencies | EEPS RQV,RQC,RQI; KEPS RQV,RQC | qv,qc,qi both; KEPS rqiblten=zero (pbl_keps.py:365) | MATCH |

---------------------------------------------------------------------------------------------------------------------
## Ranked summary

| rank | id | class | scheme / composites | conf |
|---|---|---|---|---|
| 1 | A1 | A | MYNN + mp 9/11/14/16/26/28/38/40/51/52/53/56: RQNCBLTEN (+RQNWFA/RQNIFA/RQNBCA for 28/38) never mixed/applied | HIGH |
| 2 | B1 | B | MYNN (PROD/Swiss default): DMP_mf Psig_w uses mass-averaged w instead of WRF bottom-face w_2 (incl. surface w) | HIGH |
| 3 | B4 | B | GBM bl=12: RTHRATEN dropped (hard-coded 0) in cloud-top TKE source + PBL-height radnnll | HIGH |
| 4 | B3 | B | QNSE(+MF+QNSESFC) bl/sf=4, TEMF PBL bl=10, EEPS P3D, KEPS P8W: nonhydrostatic P+PB / extrapolated p8w instead of p_hyd/p_hyd_w | HIGH |
| 5 | B2 | B | MYNN: driver hfx/qfx clamps [-600,1200] W/m2, [-3e-4,9e-4] kg/m2/s missing (extreme fluxes only) | HIGH |
| 6 | B5 | B | QNSE: SICE=0 instead of XICE (sea-ice land points) | HIGH |

PROD-identity impact: only B1 and B2 touch the default program (B2 only numerically when a clamp would fire; B1 changes
the compiled default program and outputs over sloped terrain). Everything else is non-default composites.
GPU reruns needed after fixes: B1/B2 -> Swiss f672a573 + PROD 6fc715a2 default IDs, WN3 release cases; A1 -> any bl=5 +
mp 28/16/40/P3 composite; B3/B4/B5 -> the bl=4/10/12/16/17 composites of V040-VALIDATION-PLAN.
