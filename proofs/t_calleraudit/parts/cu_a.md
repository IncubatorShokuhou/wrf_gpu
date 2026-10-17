# Caller-wiring audit (E11) — cumulus group A: KF(1), BMJ(2), GF(3), G3(5), GD(93)

Auditor: helper of lane t-calleraudit, read-only static audit, tree `lane/t-calleraudit` @ b51dd0805.
Port root `src/gpuwrf/` (OM = runtime/operational_mode.py, SA = coupling/scan_adapters.py). WRF root
`<USER_HOME>/src/wrf_pristine/WRF` (CD = phys/module_cumulus_driver.F, P1 = dyn_em/module_first_rk_step_part1.F,
BSU = dyn_em/module_big_step_utilities_em.F, ADD = phys/module_physics_addtendc.F).
Release defaults assumed: rad_rk_tendf=1 (source_leaf_mode), GPUWRF_PHYS_TEND_RK_WRF=1, GPUWRF_KF_TEND_RK_WRF=1,
GPUWRF_KF_COLUMN_FP32=1, GPUWRF_KF_PHYD_WRF=1, WRF-order post-RK microphysics.
Composites touched: c02 (mp1 Kessler + KF), c03 (mp2 Lin + BMJ + MYJ), c04 (mp3 WSM3 + GF + GFS-PBL, sf3),
c05 (mp4 WSM5 + G3 + ACM2/sf1), c13 (mp18 + G3 + MYNN), c17 (mp24 + GD + MYNN), c36/c46/c71..c90 (mp8 + KF + MYNN).

## WRF reference wiring (common to all five)
cumulus_driver call P1:1377-1541 reads the phy_prep step-entry state: U/V=u_phy/v_phy (A-grid), TH=th_phy,
T=t_phy=th_phy*pi_phy (DRY theta, BSU:4834,4859), W=w_2 (staggered), P=p_hyd (BSU:4946-4968), PI=pi_phy (from
p_phy=p+pb, BSU:4858), RHO=(1+qv)/alt (TOTAL, BSU:4860), DZ8W from z_at_w=(ph+phb)/9.81, P8W=p_hyd_w,
QV_CURR=moist(P_QV); KPBL=grid%kpbl (PBL scheme), HFX/QFX/XLAND/HT/GSW; RTHFTEN/RQVFTEN = previous step's
last-RK-stage advective tendency (module_em.F:654-665 set_tend t_tend; :1366-1376 advect_tend qv) decoupled by
mut and converted to dry theta in phy_prep_part2 (BSU:5293-5331); RTHRATEN = held radiation rate; RTHBLTEN/
RQVBLTEN = this step's PBL rates. Gate CD:832-856 (itimestep==1 | cudt==0 | stepcu==1 | mod(itimestep,stepcu)==0).
Outputs are rates, coupled with mut in calculate_phy_tend (module_em.F:2252-2305, each RQ* only if P_Q* exists),
summed by update_phy_ten/phy_cu_ten with add_a2a (ADD:1213-1283 KFETA; :1197-1210 BMJ; :1369-1412 GD/G3/GF:
theta, qv, qc, qi only, qc/qi only if P_QC/P_QI), dry->theta_m by conv_t_tendf_to_moist (part2:1005-1016,
BSU:6699-6703), integrated in every RK stage; RAINC += PRATEC*DT and KF NCA/R*CUTEN clearing in advance_ppt
(ADD:2286-2293, :2347-2380).

---------------------------------------------------------------------------------------------------------------
## 1. cu_physics=5 Grell-3D (G3) and cu_physics=93 Grell-Devenyi (GD) — PRIORITY (c05 RAINNC FAIL)
(a) Default branch: OM:9694-9708 `CU_STATELESS_SCAN_ADAPTERS[cu_opt](next_state, ..., forcing_entry_state=before,
rthblten/rqvblten only if pbl_restored, gsw=swdown-swup)` -> SA:1319 grell_cumulus_adapter -> _grell_cup_jax
g3drv_tile:1288 / grelldrv_tile:1919. WRF: CD:1218-1272 (G3DRV) + spread3d CD:1627-1645; CD:1095-1135 (GRELLDRV).
cudt>0 refused (OM:7333-7338), ishallow/cugd_avedx/clos_choice bound at Registry defaults 0/1/0.

| item | WRF | port | verdict |
|---|---|---|---|
| T | t_phy dry (P1:1379) | theta_dry*pi(state.p) SA:1353-1355 | MATCH |
| state time level | phy_prep step entry | next_state after PBL; = entry when PBL restored (sf1 PBLs OM:9517-9537, MYNN OM:9562-9590); post-PBL Euler state for unrestored PBLs (MYJ, sf!=1 PBLs, OM:9433, 9540-9544) | MATCH c05/c13/c17; MISMATCH other PBLs |
| RHO | (1+qv)/alt | _wrf_phy_prep_rho_from_state SA:1356-1358 | MATCH |
| W / omega | w_2 face k (module_cu_g3.F:405, gd.F:253) | state.w[:kx] (_grell_cup_jax.py:1337) | MATCH |
| P | p_hyd (P1:1380) | state.p total non-hydrostatic, SA:1366-1373 `p_hyd = p_mass` | MISMATCH (B) |
| P8W(1)=PSUR | p_hyd_w(1) (P1:1400; g3.F:342) | p_mass[0]*exp(g dz/2/(287 Tv)), SA:1369-1372 | MISMATCH (B) |
| RTHFTEN | prev-step RK advection, dry (BSU:5317-5331) | (theta_dry(next)-theta_dry(before))/dt SA:1374-1376 (~0 in source-leaf mode) | MISMATCH (A) |
| RQVFTEN | prev-step RK qv advection (module_em.F:1366-1376) | (qv_next-qv_before)/dt SA:1377 (~0) | MISMATCH (A) |
| RTHRATEN | held radiation rate (CD:879; gd.F:221) | `rthraten=zero3` SA:1387 | MISMATCH (A) |
| RTHBLTEN/RQVBLTEN | this step PBL | passed when PBL restored (OM:9705-9706) | MATCH (restored paths) |
| prescale (F+RA+BL)*pi | CD:877-883 (G3), gd.F:221-222 | g3drv_tile:1309-1310, grelldrv_tile:1935 | MATCH |
| KPBL | grid%kpbl | bulk-Ri diag SA:1390-1395 (only ishallow=1 consumer, g3.F:1238-1241) | MISMATCH, inert (D) |
| GSW | grid%gsw (net sfc SW) | rad.swdown-rad.swup OM:9700-9702 | MATCH [UNVERIFIED surface level] |
| HT/XLAND/DX | grid | ph[0]/9.81, xland, dx_m | MATCH |
| periodic_y | config periodic_y (P1:1455) | periodic_x reused OM:9707 | MISMATCH idealized only (C) |
| F_QI partition | if .NOT.F_QI all outqc->RQCCUTEN (g3.F:541-556; gd.F:351) | always T<258 split (_grell_cup_jax.py:1376-1378) | MISMATCH for mp without qi (A, minor) |
| theta form | dry RTHCUTEN coupled + conv_t_tendf_to_moist in RK | dry->theta_m at old state, Euler add SA:1403-1408 | MATCH form, MISMATCH integration (A-) |
| RQC/RQI | moist_tend through RK, P_QC/P_QI gated | Euler add to qc/qi SA:1410-1411 | MISMATCH integration |
| RAINC | += PRATEC*DT | rainc_acc += dt*PRATEC SA:1412-1414 | MATCH |
| GDC/GDC2/HTOP/HBOT | radiation only with cu_rad_feedback (refused) | not stored | MATCH (C history) |

## 2. cu_physics=1 Kain-Fritsch eta (release KF)
(a) Default branch: OM:9783-9800 (kf_rk): `_kf_cadence_step(before, ...)` OM:8591-8622 (gate step==1|mod(step,stepcu)==0,
lax.cond refresh/held) -> SA:761 kf_adapter(return_tendencies) -> cumulus_kf.step_kf_column:757; couple_kf_rates with
before.mu_total (kf_rk.py:22-28), kf_add_a2a, theta/qv folded into t_tendf before conv_t_tendf_to_moist
(OM:8680-8709), qv..qs into moist_tendf (OM:10063-10070); finish_kf_rates after RK (OM:10331-10338, kf_rk.py:46-60).
WRF: CD:1015-1046 KF_ETA_CPS; module_cu_kfeta.F:200-525.

| item | WRF | port | verdict |
|---|---|---|---|
| T | t_phy dry | theta_dry*pi_phy(state.p) REAL SA:794-799 | MATCH |
| P (PCPS) | p_hyd | _wrf_hydrostatic_pressure_profiles SA:796 | MATCH (fix present) |
| PI | pi_phy, RTHCUTEN=DTDT/pi (kfeta.F:487) | cumulus_kf.py:804-806 | MATCH |
| RHO | total | SA:782-785 | MATCH |
| DZ8W | g=9.81 | ph/9.81 SA:799 (max 1 m clamp inert) | MATCH |
| W/W0AVG | 0.5(w(k)+w(k+1)), TST=2*stepcu, only on call steps (kfeta.F:236-248) | _w_mass + update_w0avg cumulus_kf.py:732-754 inside cond | MATCH |
| NCA gate | NCA>=0.5*DT keep (kfeta.F:410) | SA:843-848, cumulus_kf.py:803 | MATCH |
| shallow NCA | CUDT*60 (kfeta.F:2573) | cumulus_kf.py:811 | MATCH |
| state time level | step entry | `before` OM:9789 | MATCH |
| warm_rain/F_QI/F_QS/F_QR | from mp (physics_init.F:4463-4484; kfeta.F:211-216, 493-525, 2599-2635) | never passed: step_kf_column defaults warm_rain=False,f_qi=f_qs=True (cumulus_kf.py:774-776; call SA:823-826) | MISMATCH (A) |
| RQC/RQR/RQI/RQS application | only if P_Q* (module_em.F:2265-2303; ADD:1222-1258) | always all five OM:10067-10069 | MISMATCH (A) |
| coupling | mut-coupled, add_a2a, conv at moist_old/t_1 | before.mu_total, kf_add_a2a, before.qv/theta OM:10052-10054 | MATCH |
| advance_ppt clear / NCA-=DT / decouple final mu | ADD:2347-2380; BSU phy_prep_part2 | finish_kf_rates kf_rk.py:46-60 | MATCH |
| RAINC | += PRATEC*DT (PRATEC not cleared) | dt*rates[6] | MATCH |
| kfeta_trigger=2 (RQVFTEN) | optional | refused namelist_binding.py:420-421 | MATCH |
| CUTOP/CUBOT->HTOP/HBOT | advance_ppt (GFDL rad only) | dropped | C |

## 3. cu_physics=3 Grell-Freitas (GF)
(a) Default branch: falls to the generic stateless call OM:9762-9766 `CU_STATELESS_SCAN_ADAPTERS[3](next_state, dt, grid)`
-> SA:1194 gf_adapter (ishallow_g3=1, ichoice=0 defaults) -> _gf_jax.gfdrv_batched/gfdrv_column:2109.
WRF: CD:1274-1330 GFDRV; phys/module_cu_gf_wrfdrv.F.

| item | WRF | port | verdict |
|---|---|---|---|
| ishallow_g3 | config ishallow, Registry default 0 (P1:1410, CD:1305, Registry 2550); cutens=0 (gf_wrfdrv:331), no CUP_gf_sh (:504) | 1 (SA:1194, no kwarg at OM:9763) | MISMATCH (A) |
| T | t_phy dry | theta_m*pi `_temperature_from_theta(state.theta, ...)` SA:1243 | MISMATCH (A) |
| forcing TN/QO/TSHALL/DHDT | (RTHFTEN+RTHRATEN+RTHBLTEN)*pi*dt, RQVFTEN+RQVBLTEN (gf_wrfdrv:412-417) | zero_c for BL, no FTEN/RATEN args (SA:1281, 1292-1293; gf_jax:2139-2143) | MISMATCH (A) |
| theta form | dry RTHCUTEN (gf_wrfdrv:745 /pi) -> conv_t_tendf_to_moist in RK | `theta + dt*rth` on theta_m, SA:1309 | MISMATCH (A) |
| RQICUTEN | phy_cu_ten adds only if P_QI (ADD:1390-1395) | always added SA:1312 (c04 WSM3 has no qi) | MISMATCH (A) |
| RHO | total | dry `_rho_from_state` SA:1242 (thompson_column.py:450: 0.622p/(Rd T(qv+.622))) | MISMATCH (B) |
| P / PSUR | p_hyd; PSUR=p8w(1) (gf_wrfdrv:385,404) | state.p SA:1274; psur=p_col[1] lowest mass level gf_jax:2129 | MISMATCH (B) |
| omega | -g*rho*w(i,k,j) face (gf_wrfdrv:471) | w averaged to mass `_w_mass` SA:1249, gf_jax:2149 | MISMATCH (B) |
| KPBL | grid%kpbl from PBL (P1:1397; used gf_deep.F:585,1703,3848) | bulk-Ri thermal diag SA:1262-1265 | MISMATCH (B) |
| active region | ids+4..ide-5 unless periodic (gf_wrfdrv:264-277) | all columns, no mask SA:1292-1317 | MISMATCH (B) |
| DZ8W/HT | g=9.81 | GRAVITY 9.80665 SA:1245 | MISMATCH (B, 0.035 %) |
| HFX/QFX | grid%hfx/qfx (LSM) | rhosfc*1004*theta_flux, rhosfc*qv_flux SA:1252-1257 | UNVERIFIED (shallow-only consumer) |
| state time level | entry | next_state; c04 GFS-PBL (sf3) Euler-applied, not restored (OM:9540-9544) | MISMATCH (B) |
| integration | RK coupled | Euler + post-RK increment | MISMATCH (A-) |
| RUCUTEN/RVCUTEN | produced, NOT applied for GFSCHEME (ADD:1369-1412) | none | MATCH |
| cudt>0 | stepcu gate, held rates | every step, not refused | MISMATCH (B, non-default) |

## 4. cu_physics=2 Betts-Miller-Janjic (BMJ)
(a) Default branch: OM:9814-9822 `bmj_adapter(next_state, dt, cldefi, grid)` SA:1430 -> cumulus_bmj.step_bmj_column:1157.
WRF: CD:992-1013 BMJDRV, phys/module_cu_bmj.F:165-265. (KPBL/LPBL and PT are dummies inside BMJ: no effect.)

| item | WRF | port | verdict |
|---|---|---|---|
| T (TCOL) | t_phy dry (bmj.F:209) | theta_m*pi SA:1442 | MISMATCH (A) |
| theta form | RTHCUTEN=DTDT/PI dry -> conv_t_tendf_to_moist | `theta + dt*rth` SA:1489 | MISMATCH (A) |
| RHO in DPCOL=RHO*G*DZ8W | total, g=9.81 (bmj.F:213) | dry rho SA:1441, dz with 9.80665 SA:1443 | MISMATCH (B, ~1 % column mass) |
| PSFC | PINT(LOWLYR)=p_hyd_w(1) (bmj.F:193) | p0+0.5(p0-p1) cumulus_bmj.py:1177 (no pint passed) | MISMATCH (B) |
| PMID | p_hyd | state.p SA:1452 | MISMATCH (B) |
| PI | pi_phy | (state.p/1e5)^(2/7) SA:1445 | MATCH |
| STEPCU/DTCNVC | dt*stepcu, held rates (bmj.F:171,257-258) | stepcu=1 hard-coded SA:1467, every step, cudt>0 not refused | MISMATCH (B, non-default) |
| RQVCUTEN | DQDT/(1-q)^2 | cumulus_bmj.py:1145 | MATCH |
| CLDEFI carry / BMJINIT | AVGEFI | initial_bmj_cldefi | MATCH |
| state time level | entry | post-MYJ Euler state (c03, OM:9433-9434) | MISMATCH (B) |
| integration / add_a2a | RK coupled, ring 0 excluded | Euler + post-RK increment, all columns | MISMATCH (A-) |
| bmj_rad_feedback | default .false. | not bound (default) | MATCH |

---------------------------------------------------------------------------------------------------------------
## FINDINGS (ranked by forecast impact)

A1 [G3 cu=5, GD cu=93] Closure forcing lacks advection and radiation. HIGH.
 WRF: G3 `RTHFTEN=(RTHFTEN+RTHRATEN+RTHBLTEN)*pi; RQVFTEN=RQVFTEN+RQVBLTEN` (CD:877-883; GD gd.F:221-222), with grid%rthften/
 rqvften = previous step's last-stage RK advective tendencies (module_em.F:661 `set_tend(RTHFTEN, t_tend, msfty)`, :1372
 `set_tend(RQVFTEN, advect_tend, msfty)`, decoupled + made dry in BSU:5297-5331); `TN=t2d+RTHFTEN*dt`, `QO=q2d+RQVFTEN*dt`
 (g3.F:372-373), Tx/Qx ensemble members (g3.F:407-412). Port: `thften = (theta_dry - _dry_theta_view(forcing_entry_state))/dt`,
 `qvften = (qv - forcing_entry_state.qv)/dt` (SA:1374-1377) — in source-leaf mode radiation and restored PBL are not in the
 state, so both are ~0 — and `rthraten=zero3` (SA:1387). The "large-scale forcing" closures (Grell dA/dt, AS) see no
 destabilisation; only stability/omega/mconv members act -> systematically different G3/GD mass flux and RAINC/RAINNC split.
 Combos: every cu=5/93 run (c05, c13, c17); bites from itimestep 2 (WRF arrays are 0 at step 1).
 Fix: carry two REAL held leaves RTHFTEN/RQVFTEN = last-RK-stage theta/qv advection (theta-only advection like rk_tendency,
 msfty-scaled), decouple by final mut and dry-convert exactly as phy_prep_part2 (BSU:5317-5331); pass the uncoupled held
 RTHRATEN of this step (needs radiation rate computed before the cumulus slot) and keep RTHBLTEN/RQVBLTEN. Restart schema +1.
 Oracle: pristine WRF run c05 with a one-step dump of RTHFTEN/RQVFTEN/RTHRATEN entering G3DRV at itimestep 2..3; feed the
 port g3drv_tile those vs current zeros and compare RTHCUTEN/PRATEC (deletion-sensitive: zero-forcing must FAIL).

A2 [GF cu=3] ishallow forced on. HIGH. WRF passes `ishallow_g3=ishallow` (CD:1305, P1:1410, Registry default 0) ->
 `if(ishallow_g3.eq.0)cutens(i)=0.` (gf_wrfdrv:331), CUP_gf_sh not called (:504). Port `gf_adapter(..., ishallow_g3: int = 1`
 (SA:1194) and OM:9763 passes no kwarg -> shallow GF heating/moistening/rain WRF never applies (c04 and every cu=3 run).
 Fix: thread namelist ishallow (default 0). Oracle: pristine GF column with ishallow=0 vs port output (shallow terms must vanish).

A3 [GF cu=3, BMJ cu=2] Moist temperature input. HIGH. WRF T=grid%t_phy = th_phy*pi_phy with th_phy=theta_m/(1+Rv/Rd qv)
 (BSU:4834,4859; P1:1379). Port `T = _temperature_from_theta(state.theta, state.p)` on theta_m (GF SA:1243, BMJ SA:1442) ->
 T too warm by ~(1.608 qv)*T (~+4.6 K at qv=10 g/kg); also shifts GF's T<258 ice split. Same defect class as the fixed
 Tiedtke cu6 (e213b65c4). Combos c03, c04, all cu=2/3. Fix: use `_dry_temperature_from_state`/t_phy as G3 does (SA:1353-1355).
 Oracle: pristine BMJDRV/GFDRV on a real moist column with t_phy vs the port caller's T.

A4 [GF cu=3, BMJ cu=2] Dry-theta RTHCUTEN added to theta_m without conv_t_tendf_to_moist. HIGH. WRF couples RTHCUTEN into
 t_tendf then `t_tendf=(1+Rv/Rd qv)t_tendf + Rv/Rd (t_1+T0)/(1+Rv/Rd qv)*moist_tend(P_QV)` (part2:1005-1016, BSU:6699-6703).
 Port `theta=(state.theta + dt_f * rth)` (GF SA:1309, BMJ SA:1489); G3/GD/KF use `_theta_m_tendency_from_dry` correctly. The
 missing Rv/Rd*theta*RQVCUTEN term is of the same order as RTHCUTEN for BMJ/GF drying. Fix: reuse _theta_m_tendency_from_dry
 (old state). Oracle: one-step pristine theta_m increment from a frozen BMJ/GF tendency pair.

A5 [KF cu=1] Microphysics-species flags never wired. HIGH. WRF KF branches on warm_rain (Kessler, physics_init.F:4480-4481),
 F_QS/F_QI/F_QR (kfeta.F:211-216, 493-525, 2599-2635: warm_rain melts all ice into qc/qr with -RLF heating; .NOT.F_QS
 melts/freezes at the ML and puts ice into qc; .NOT.F_QR puts rain into RQCCUTEN), and calculate_phy_tend/phy_cu_ten couple and
 apply RQC/RQR/RQI/RQS only if P_Q* exists (module_em.F:2265-2303, ADD:1222-1258). Port kf_adapter calls step_kf_column without
 flags (SA:823-826) -> defaults `warm_rain=False, f_qi=True, f_qs=True` (cumulus_kf.py:774-776) and OM:10067-10069 adds all
 five species. For c02 (Kessler+KF) KF keeps freezing heat and writes RQICUTEN/RQSCUTEN into the inactive qi/qs slots
 (advected, loaded into p_hyd, never consumed by Kessler); same for mp 0/3 (no qi/qs) and P3/JI 50-53/55 (no qs). The
 legacy-KF adapter already does it right (kf99_adapter.py:45-46, 57-58: moist_species_for_mp, warm_rain=mp==1).
 Fix: pass warm_rain/f_qi/f_qs(/f_qr) from moist_species_for_mp(mp) and zero RQ* of absent species (as kf99).
 Oracle: pristine KF_eta_PARA column with warm_rain=.true. (c02 namelist) vs port; mutant = drop the flag.

A6 [G3/GD/GF/BMJ] Species gating for mp without qi (WSM3 c04, Kessler, mp0). HIGH (minor magnitude). WRF G3DRV/GRELLDRV put all
 detrained condensate in RQCCUTEN when .NOT.F_QI (g3.F:541-556, gd.F:351); GF/G3 RQICUTEN is dropped by phy_cu_ten when P_QI is
 absent (ADD:1390-1395). Port always splits at T<258 and adds qi (SA:1411, 1312). Fix: gate by moist_species_for_mp.

A7 [G3/GD/GF/BMJ] Euler pre-add + post-RK increment instead of RK-coupled sources. MEDIUM-HIGH (port path read; magnitude
 UNVERIFIED). Stateless adapters update next_state; the RK starts from `carry.state` (OM:8857,10274) and the cumulus deltas
 are added after RK by `_apply_physics_non_dry_updates` (OM:8383-8389: `dynamics_value + (physics_value - reference_value)`),
 so the acoustic/buoyancy response, the RK-stage PD/mono limiter on qc/qi, mu coupling and add_a2a ring exclusion
 (ADD:2435-2481) are bypassed; WRF integrates RTHCUTEN/RQ*CUTEN inside every RK stage. KF already moved to the WRF form
 (KF_TEND_RK_WRF). Fix: route G3/GD/GF/BMJ rates through couple_kf_rates/kf_add_a2a/kf_tendf like KF (rates only, no state
 write). Oracle: one-step pristine solve_em with a frozen RTHCUTEN vs port both forms (w/p after step 1).

B1 [G3/GD] P=state.p (full non-hydrostatic) instead of p_hyd, and PSUR from a hydrostatic extrapolation of p_mass[0] instead of
 p_hyd_w(1) (SA:1364-1373 vs P1:1380,1400; g3.F:342, gd.F:233). HIGH (port docstring names it a caveat). Fix: reuse
 `_wrf_hydrostatic_pressure_profiles_from_state` as KF does (SA:796). Small (non-hydrostatic p' and condensate loading).
B2 [GF] PSUR = lowest mass-level pressure `psur = p_col[1] * 0.01` (gf_jax:2129) vs `PSUR(I)=p8w(I,1,J)*.01`
 (gf_wrfdrv:385); P=state.p vs p_hyd (SA:1274, gf_wrfdrv:404). HIGH. Half-layer (~1-5 hPa) PSUR error -> cup_env heights.
B3 [GF] omega from mass-averaged w (SA:1249 `_w_mass`, gf_jax:2149) vs face `omeg=-g*rho(i,k,j)*w(i,k,j)` (gf_wrfdrv:471);
 feeds mconv and omega closures. HIGH. Fix: pass state.w[:kx].
B4 [GF] KPBL from a thermal bulk-Ri diagnostic (SA:1262-1265, `_kpbl_bulk_richardson` SA:1162) instead of grid%kpbl of the
 active PBL (P1:1397); GF deep uses kpbl for cap_max, k22 and zu/zd PDFs (gf_deep.F:585,1703,3848). HIGH. Fix: thread the
 PBL scheme's KPBL (held leaf) like PBLH. (G3's own bulk-Ri KPBL SA:1394 is inert: only ishallow=1 reads it, g3.F:1238.)
B5 [GF] No 4-point lateral exclusion: WRF ibegc=ids+4..ide-5, jbegc=jds+4..jde-5 unless periodic (gf_wrfdrv:264-277; rows
 skip at `go to 100`); port computes/applies GF on every column (SA:1292-1317; no mask in gfdrv_column). HIGH. Fix: reuse
 `_grell_cup_jax._interior_mask` (as G3/GD do).
B6 [GF/BMJ] RHO dry (`_rho_from_state` = 0.622p/(Rd T (qv+0.622)), thompson_column.py:450) vs total (1+qv)/alt; DZ8W/HT with
 g=9.80665 (SA:1245, 1443) vs 9.81. BMJ DPCOL=RHO*G*DZ8W (bmj.F:213) -> ~1 % low column mass weights. HIGH.
B7 [BMJ] PSFC = p0+0.5(p0-p1) (cumulus_bmj.py:1177; bmj_adapter passes no pint) vs PINT(LOWLYR)=p_hyd_w(1) (bmj.F:193);
 PMID=state.p vs p_hyd (SA:1452). PSFC scales DEPMIN/deep-shallow choice. HIGH. Fix: pass pint from the hydrostatic profiles.
B8 [GF/BMJ/G3 input time level] For PBLs that are not restored after the PBL call (MYJ OM:9433-9434 -> c03; any bl in
 PBL_SCAN_ADAPTERS when sf_sfclay!=1, OM:9540-9544 -> c04 GFS-PBL), the cumulus adapters read the post-PBL Euler-updated
 T/qv/u/v; WRF reads the phy_prep step-entry state. KF is immune (reads `before`, OM:9789). HIGH. Fix: pass `before`
 (as KF) plus the PBL rates as forcing.
B9 [BMJ/GF cudt>0] Port runs every step (BMJ `stepcu=1` SA:1467; GF stateless) and admission does not refuse cudt>0 (only G3/GD,
 OM:7333); WRF calls every stepcu steps with DTCNVC=dt*stepcu (bmj.F:171, RAINCV/PRATEC /STEPCU :257-258) and holds R*CUTEN.
 Default cudt=0 (Registry) -> not in composites. HIGH. Fix: refuse cudt>0 or add held-rate cadence like KF/SAS.

C1 [G3] periodic_y receives periodic_x (OM:9707) vs config periodic_y (P1:1455): idealized channel domains only.
C2 [all] CUTOP/CUBOT/HTOP/HBOT, RAINCV, GD_CLOUD(2), KF CLDFRA_DP/SH not kept/written (advance_ppt ADD:2341-2357 HTOP/HBOT
 only feed GFDL radiation; cu_rad_feedback refused for cu!=11) — history only.
D1 [KF legacy] Non-default fp64 KF path (GPUWRF_KF_COLUMN_FP32=0) still uses state.p for PCPS and g=9.80665 (SA:798-801);
 release path is correct.

Verified-present fixes: KF P_HYD/PI_PHY/total RHO/dry T (SA:782-799), KF RK coupling + advance_ppt (kf_rk.py), G3/GD dry T +
total RHO + theta_m conversion (SA:1353-1358, 1403-1408).
