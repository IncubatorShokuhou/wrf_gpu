# Caller-wiring audit — cumulus group C (SAS family: cu=14 KSAS, 96 NSAS, 94 SAS, 95 OSAS, 4 scale-SAS; +cu=7 glance)

Static read-only audit, tree `lane/t-calleraudit` = main ea6c83d25 (+notes), pristine WRF 4.7.1. No code run.
Port root `src/gpuwrf/`, WRF root `<USER_HOME>/src/wrf_pristine/WRF/`. "om" = `runtime/operational_mode.py`.

## 0. Shared WRF caller (all SAS options) and shared port path

WRF call: `dyn_em/module_first_rk_step_part1.F:1377-1530` `cumulus_driver(U=grid%u_phy, V=grid%v_phy, T=grid%t_phy,
W=grid%w_2, P=grid%p_hyd, PI=pi_phy, RHO=grid%rho, DZ8W=dz8w, P8W=grid%p_hyd_w, PBLH=grid%pblh, HPBL_HOLD=hpbl_hold,
HFX=grid%hfx, QFX=grid%qfx, XLAND, DX=grid%dx, STEPCU, ZNU=grid%znu, MP_PHYSICS, pgcon=sas_pgcon, NSAS_DX_FACTOR,
HTOP=grid%cutop, HBOT=grid%cubot, QV/QC/QI_CURR=moist(...))` (lines 1379-1380, 1397-1400, 1416, 1482, 1491, 1519).
Not passed by ARW: hpbl2d/evap2d/heat2d, MOMMIX, store_rand, shalconv, shal_pgcon, sas_mass_flux, pert_sas, DYNMM.
Gate `phys/module_cumulus_driver.F:828-864` (itimestep==1 | cudt==0 | stepcu==1 | MOD(itimestep,stepcu)==0); STEPCU =
max(1,NINT(cudt*60/dt)) (`phys/module_physics_init.F:4027-4028`). Calls: SAS 1131-1160, SCALESAS 1161-1191, OSAS 1192-1217,
KSAS 1444-1478, NSAS 1480-1513. Application chain: `calculate_phy_tend` couples every R*CUTEN incl. RU/RVCUTEN with
(c1*mut+c2) (`dyn_em/module_em.F:2252-2305`, e.g. 2259) → `update_phy_ten`/`phy_cu_ten` (`phys/module_physics_addtendc.F`)
CASE(KSAS,NSAS) 1414-1467: add_a2a(rt_tendf,RTH), **add_a2c_u/v(ru/rv_tendf,RU/RV)**, add_a2a(moist_tendf QV/QC/QI);
CASE(SAS,OSAS,SCALESAS) 1469-1512: RTH/QV/QC/QI only, **no momentum** → `conv_t_tendf_to_moist(t_1, moist_old)`
(`module_first_rk_step_part2.F:1009`) → integrated in every RK stage/acoustic substep; RAINC += PRATEC*DT every step
(`advance_ppt`, addtendc.F:2291); held rates decoupled with end-of-step muts (`phy_prep_part2`,
`module_big_step_utilities_em.F:5108-5175`, called `solve_em.F:3669`) so the COUPLED rate is what is held between calls.

Port: cumulus slot om:9709 (cu4), 9719 (14/96), 9740 (95), 9751 (94); each calls `<x>_cadence_step(before, next_state,
held, ...)` → rates from `before` (= carry.state at step entry, after WRF-order MP of previous step = WRF phy_prep state),
then `x += dt*R` on the slot state (`coupling/sas_consumers.py:15-58`; cu4 `scalesas_adapter.py:107-125`). That slot
state is `physics_forcing.state`; RK starts from the unmodified carry and the delta is added AFTER RK by
`_apply_physics_non_dry_updates` (om:8375-8392, called om:10326; u/v/theta/qv/qc/qi in `_PHYSICS_NON_DRY_INCREMENT_FIELDS`
om:515-530). Only cu=1 (`kf_rk`, om:8890) and cu=7/10/11 (om:9766-9786) feed `kf_tendf` into the RK source leaves
(om:10052-10070). Default flags (rad_rk_tendf=1, PHYS_TEND_RK_WRF=1, KF_TEND_RK_WRF=1) do NOT change the SAS branch.

Shared staging helpers (all MATCH, read both sides): t = `_dry_theta_view`·pi (physics_couplers.py:522-533 = phy_prep
th_phy=theta_m/(1+Rv/Rd qv), big_step_utilities_em.F:4834, 4853-4855); pi=(p_total/1e5)^(2/7) (= 4854, rcp=2/7); p_hyd/p_hyd_w
from `_wrf_hydrostatic_pressure_profiles_from_state` (physics_couplers.py:1643-1680 = 4948-4966, qtot over all moist,
p_top, c1/c2·MUT·dnw); rho `_wrf_phy_prep_rho_from_state` (=4856 1/alt·(1+qv)); dz8w = Δ(ph_total)/9.81 (ph is TOTAL
geopotential, contracts/state.py:551; = 4864-4876); u/v `_u_mass/_v_mass` (=4857-4858); w = staggered state.w (= w_2).

## 1. cu=14 KSAS — port om:9719 → `coupling/native_sas_adapter.py:31-105` → `physics/cumulus_ksas14.py:742-780`; WRF cumulus_driver 1444-1478 → `phys/module_cu_ksas.F:8-298`

| item | WRF | port | verdict |
|---|---|---|---|
| T3D/QV/QC/QI/U/V/W | t_phy, moist(QV/QC/QI) at phy_prep, u_phy/v_phy, w_2 | `before` via helpers (adapter :45-60) | MATCH |
| P3D / PI3D / RHO / DZ8W | p_hyd / pi_phy / rho / dz8w | p_hyd, pi(p_total), rho, Δz (adapter :45-55) | MATCH |
| P3DI (p8w) | passed, prsi built (ksas.F:187) but unused in nsas2d | not passed | MATCH (unused) |
| HPBL / HPBL_HOLD | grid%pblh post-PBL / PBLH saved at pbl_driver entry (pbl_driver.F:1024) | `diag.pblh` (=post-PBL next_state) / `before.pblh` (adapter :63,67) | MATCH |
| HFX/QFX | passed, unused in cu_ksas | not passed | MATCH (unused) |
| ZNU → kbmax/kbm/kmax | grid%znu (ksas.F:170-176) | 0.5(eta_k+eta_k+1) from grid.eta_levels (adapter :65-66) | MATCH |
| DX, XLAND, STEPCU, delt | dx, xland, dt*stepcu | projection.dx_m, before.xland, stepcu (kernel :753-756) | MATCH |
| ncloud / F_QC,F_QI | mp 0→0, 1/3→1 else 2; RQC/RQI only if F_Q* (ksas.F:143-150, 269-287) | same; has_qc/has_qi gate (adapter :56, 73-76; kernel :767) | MATCH (admitted mp set) |
| pgcon, dx_factor_nsas | pgcon_use computed but not passed to nsas2d; dx_factor unused | not used (kernel docstring :748-749) | MATCH |
| cadence/held | gate as §0; held R*CUTEN coupled/decoupled each step | cond(step==1 or mod==0) else held raw (adapter :94-105) | MATCH gate; held form see A1 |
| RTH/RQV/RQC/RQI apply | coupled RK t_tendf/moist_tendf, add_a2a ring | post-RK `x+=dt*R`, add_a2a ring (sas_consumers :17-37) | **MISMATCH form (A1)** |
| RU/RV apply | coupled add_a2c_u/v into ru/rv_tendf | post-RK add_a2c increment (sas_consumers :40-58), ring bounds = addtendc 2531-2636 | ring MATCH, **form A1** |
| RAINC / PRATEC | RAINC += PRATEC*DT each step (2291) | rainc_acc += dt*pratec each step (sas_consumers :36) | MATCH |
| HTOP/HBOT → CUTOP/CUBOT | restart-only state (Registry.EM_COMMON:1249-1250 `r`) | discarded (adapter :73-76) | D |

## 2. cu=96 NSAS — om:9719 → native_sas_adapter (scheme=96) → `physics/cumulus_nsas96.py:674-732` (+`_shallow`); WRF 1480-1513 → `phys/module_cu_nsas.F:8-308`

Same rows as KSAS (MATCH) plus:

| item | WRF | port | verdict |
|---|---|---|---|
| P3DI | p_hyd_w → prsii (nsas.F:174), used | cols(p_hyd_w) (adapter :69) | MATCH |
| HPBL/HFX/QFX into nscv2d | grid%pblh, grid%hfx, grid%qfx (nsas.F:249) post-surface/PBL | diag.pblh/hfx/qfx from slot next_state (adapter :69-70) | MATCH source; land HFX post-LSM UNVERIFIED for sf=1/16/17 paths (see U1) |
| pgcon, dx_factor | pgcon_use to nsas2d and nscv2d; nsas_dx_factor | sas_pgcon (default .55 = Registry 2655), nsas_dx_factor | MATCH |
| kbm/kbmax/kmax limits | computed per tile i-row in nsas2d | per full row (row_width=nx, kernel :700-712) | MATCH for serial 1-tile WRF; WRF is decomposition-dependent (D) |
| rain sum | raincv1+raincv2 / pratec1+pratec2 (nsas.F:261-262) | rn+rn2 (kernel :729-730) | MATCH |
| apply | as KSAS (with momentum) | as KSAS | **A1** |

## 3. cu=94 SAS (2011 GFS) — om:9751 → `coupling/sas94_adapter.py:33-110` → `physics/cumulus_sas94.py:666`; WRF 1131-1160 → `phys/module_cu_sas.F:7-500`

| item | WRF | port | verdict |
|---|---|---|---|
| T3D/QV/QC/QI/U/V/W/PCPS/PI3D/RHO/DZ8W | as §0 | adapter :41-74 | MATCH |
| PSFC | P8W(kms)=p_hyd_w(1) (sas.F:357-361) | p_hyd_w[0] (adapter :50) | MATCH |
| shalconv | absent → EM default 1 → OLD_ARW_SHALCV (sas.F:240, 442) | shallow=True default (kernel :668, :711-720) | MATCH |
| pgcon / massf / pert_sas | sas_pgcon / absent→9e9 / absent | pgcon / mass_flux=None / none | MATCH |
| QC/QI absent species | P_QC/P_QI gated outputs (sas.F:482-497), dummy input | has_qc/has_qi zero (adapter :67-69) | MATCH |
| RU/RVCUTEN | computed only under NMM; ARW phy_cu_ten 1469-1512 does NOT apply | not computed/applied (adapter :5, 81-84) | MATCH |
| RTH/RQV/RQC/RQI apply | coupled RK (§0) | post-RK increment | **A1** |
| HTOP/HBOT | restart-only | dropped | D |

## 4. cu=95 OSAS (old GFS) — om:9740 → `coupling/osas95_adapter.py:34-113` → `physics/cumulus_osas95.py:394`; WRF 1192-1217 → `phys/module_cu_osas.F:7-394`

Inputs/PSFC/QC-QI gating/no-momentum as cu=94: MATCH. Shallow `SHALCV` always under EM (osas.F:342) = port shallow=True
(kernel :438-447): MATCH. Random XKT2: WRF `init_random_seed()` from SYSTEM_CLOCK, reseeded per j-row every call
(osas.F:2493-2505, 281-282) → not reproducible and rows sharing a clock tick get identical draw sequences; port
Threefry counter (step, global column) (`physics/osas_rng.py`, adapter :76) — independent per-column draws. Documented
deterministic deviation; statistical only (D). Apply: **A1**. HTOP/HBOT dropped: D.

## 5. cu=4 scale-aware SAS (API only; CLI refused like check_a_mundo:671) — om:9709-9718 → `coupling/scalesas_adapter.py`; WRF 1161-1191 → `phys/module_cu_scalesas.F:13-556`

Inputs as cu=94 (MATCH); DX2D = dx (MATCH); DY: ARW passes the ABSENT optional DYNMM to non-optional DY (UB in WRF, reason
for the ARW refusal) — port uses grid dy (documented, adapter :10-11, 72-73; D). pgcon: mfdeepcnv uses internal 0.55
(scalesas.F:966) → ignoring sas_pgcon is MATCH. Differences (B2): apply has NO add_a2a ring restriction and no
periodic handling (`apply_scalesas_rates` :107-125; om:9712-9716 passes no BC flags) vs add_a2a (addtendc.F:2435-2481);
qc/qi passed/applied without the P_QC/P_QI gate (adapter :79, 88-89, 121-122) unlike 94/95. Apply form: **A1**.

## 6. cu=7 Zhang-McFarlane (quick glance only)

Routed through `coupling/cuplus` with coupled RK sources (om:9766-9786): theta/qv/qc/qi + Nc/Ni scalars, qr/qs skipped
(om:10067 `cu_opt==7 ... continue`), momentum via `cuplus.momentum_sources` → consistent with phy_cu_ten CAMZM
(addtendc.F:1514-1582: add_a2c_u/v, RTH, QV, QC, QI, QNC, QNI). Nothing obvious; CAM callers belong to another lane.

## FINDINGS (ranked)

### A1 — SAS-family cumulus tendencies bypass WRF's coupled RK route (post-RK lumped Euler add) [form; HIGH confidence]
- WRF: R*CUTEN are mass-coupled `(c1*mut+c2)*R` (module_em.F:2259 ff.), added via phy_cu_ten to rt_tendf / ru_tendf,rv_tendf
  (KSAS/NSAS only, addtendc.F:1414-1467) / moist_tendf (1414-1512), theta→theta_m by conv_t_tendf_to_moist (part2.F:1009),
  and integrated in every RK3 stage and acoustic substep; between cumulus calls the held rate is decoupled with the
  end-of-step muts (big_step_utilities_em.F:5112) and re-coupled next step, i.e. the coupled rate is held.
- Port: `sas_consumers.py:31-37` `theta=add('theta', dt*thermal)`, `qv=add('qv', dt*rqv)` … and `:43-44`
  `u = _add_a2c_u_increment(u, du)`; increments ride `physics_forcing.state` and are added after the dycore by
  om:10326 `next_state = _apply_physics_non_dry_updates(next_state, physical_origin, physics_forcing.state)` (X_RK + ΔX).
  Same for cu=4 (`scalesas_adapter.py:114-125`). Contrast: cu=1 default `kf_rk` (om:8890, 9787-9799) and cu=7/10/11
  (om:9766-9786) already feed coupled `kf_tendf` into `_source_leaf_dry_tendencies` (om:10052-10070) and finish post-RK
  with `finish_kf_rates` (om:10330-10339). Code admits it: native_sas_adapter.py:6-7, sas94_adapter.py:5-6, osas95 :5-6,
  scalesas_adapter.py:19-21 ("existing step-entry coupling seam … qualification pending").
- Effect: convective heating/drying and (14/96) cumulus momentum transport are invisible to the same step's RK/acoustic
  stages (buoyancy/pressure/w response lags one step), not part of WRF's per-stage scalar update, and lack the mu
  weighting (WRF effective increment dt·R·mut_n/mu_new; held over stepcu steps with the coupled rate constant; momentum
  averages coupled mass-point rates and decouples with muu). Amount per step is equal to first order, so this is a
  form/time-placement error, systematic wherever SAS convects (same class that KF_TEND_RK_WRF fixed for cu=1).
- Affects every run with cu=14/94/95/96 (composites c08 cu14/mp10/bl12, c09 cu94/mp13/bl99, c10 cu95/mp14/bl16,
  c11 cu96/mp16/bl17 in proofs/v040/final_validation/MATRIX.json) and cu=4 API runs.
- Fix sketch: reuse `coupling/kf_rk`: at the slot, `coupled = couple_kf_rates(rates[:7], before.mu_total, metrics)`
  (+ couple RU/RV for 14/96), held carry stores the coupled rates; `kf_tendf = tuple(kf_add_a2a(r, ...) for r in
  coupled[:6])` (RQR/RQS zero); 14/96 momentum through `cuplus.momentum_sources` (add_a2c on coupled rates) into
  `cuplus_momentum`; drop the Euler apply; post-RK decouple with `next_state.mu_total` (finish_kf_rates minus NCA) and
  RAINC += dt*PRATEC. Keep raw rates only for history. Gate as for KF_TEND_RK_WRF (default-off flag first).
- Oracle: pristine serial WRF 1-domain (Swiss c11/c08 namelists), history every step for 2-3 steps from the same IC:
  compare per-step theta/qv/u/v increments in convecting columns port vs WRF; mutant = current Euler seam must fail
  the per-step increment bound while the 6 h D6 may still pass (cheap: reuse the KF RK A/B harness).

### B1 — none found for 14/94/95/96 inputs (all staged values match phy_prep, see tables) [HIGH]

### B2 — cu=4 (API-only) apply ignores add_a2a ring + species gating [MEDIUM-LOW impact, HIGH confidence]
- WRF add_a2a skips ids/ide-1 and jds/jde-1 for specified/nested (addtendc.F:2461-2466); phy_cu_ten adds QC/QI only if
  P_QC/P_QI allocated (1482-1490). Port `apply_scalesas_rates` (scalesas_adapter.py:107-125) adds everywhere; om:9712-9716
  passes no specified/periodic flags; `scalesas_rates` feeds raw state.qc/qi (:79) for mp 0/1/3.
- Fix: call `apply_sas_scalar_rates(..., specified_or_nested, periodic_x)` like 94/95 and add has_qc/has_qi gating
  (copy sas94_adapter.py:67-69, 81-84). Oracle: CPU 1-step nested/specified ring check (ring-0 values unchanged).

### U1 — NSAS shallow HFX/QFX on land for sf=1 / bl=16,17 surface paths [UNVERIFIED]
- NSAS nscv2d needs grid%hfx/qfx = post-LSM merged fluxes. Port reads `next_state.hfx/qfx` at the cumulus slot
  (native_sas_adapter.py:69-70). Noah-MP hook writes them (noahmp_surface_hook.py:336-338), but for c11 (sf=1, bl=17,
  Noah-MP) the surface goes through `epsilon_surface_step` (om:9005-9011, epsilon_pbl.py:70-76) — whether state.hfx/qfx
  on land carry the Noah-MP values (WRF) or the sfclay bulk values was not traced. Ask the surface-layer helper to
  confirm; impact limited to NSAS shallow trigger/closure (and KSAS ignores HFX/QFX).

### D (cosmetic / documented)
- D1 HTOP/HBOT (→CUTOP/CUBOT) dropped for all SAS options; Registry marks them restart-only (`r`), no SAS consumer.
- D2 OSAS RNG: deterministic Threefry vs WRF clock-seeded per-row RANDOM_NUMBER (documented, osas_rng.py:1-8).
- D3 NSAS row-shared level limits match a single-tile serial WRF only (WRF result depends on tiling).
- D4 add_sas_momentum `periodic_y` is passed `periodic_x` (om:9726); WRF add_a2c_v has no periodic_y branch and uses
  y-halo rows on non-specified domains — idealized-only edge rows differ.
- D5 cu=4 DY: WRF ARW passes absent DYNMM into non-optional DY (undefined); port uses grid dy (documented).
