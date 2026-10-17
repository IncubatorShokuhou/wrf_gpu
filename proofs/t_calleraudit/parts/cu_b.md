# Caller-wiring audit: cumulus group B (cu=6, 16, 99, 10, 11). Read-only static audit, 2026-10-15

Port: `<USER_HOME>/src/wrf_gpu2_wt/t-calleraudit` @ b51dd0805. WRF: `<USER_HOME>/src/wrf_pristine/WRF` (4.7.1).
Abbreviations: OM = `src/gpuwrf/runtime/operational_mode.py`, SA = `src/gpuwrf/coupling/scan_adapters.py`, CP = `src/gpuwrf/coupling/cuplus.py`,
CPD = `src/gpuwrf/kernels/phys_cuplus_driver.py`, KF99 = `src/gpuwrf/coupling/kf99_adapter.py`, CD = `phys/module_cumulus_driver.F`,
P1 = `dyn_em/module_first_rk_step_part1.F`, BSU = `dyn_em/module_big_step_utilities_em.F`, ADD = `phys/module_physics_addtendc.F`.

## Default runtime context (release defaults: rad_rk_tendf=1 so source_leaf_mode, PHYS_TEND_RK_WRF=1, KF_TEND_RK_WRF=1)
- WRF order (P1): phy_prep :208, then radiation :264, surface :594, pbl :1113, cumulus_driver :1377. Every driver reads the time-n phy_prep state.
  Cumulus args (P1:1379-1394): `U=u_phy V=v_phy TH=th_phy T=grid%t_phy W=w_2 P=grid%p_hyd PI=pi_phy RHO=grid%rho ... DZ8W=dz8w P8W=grid%p_hyd_w ... HFX=grid%hfx QFX=grid%qfx`.
  CD gate :831-858: run if itimestep==1, cudt==0, stepcu==1 or MOD(itimestep,stepcu)==0. Otherwise the R*CUTEN stay held in grid.
  Coupling: part2 calculate_phy_tend :392, update_phy_ten :801 (ADD phy_cu_ten), conv_t_tendf_to_moist :1009, advance_ppt (solve_em :3649),
  phy_prep_part2 decouple (solve_em :3669).
- Port: `_physics_step_forcing` has `before = carry.state` (OM:8861). With MYNN plus PHYS_TEND_RK_WRF the PBL slot restores theta/u/v/qv/qc/qi
  to `pbl_entry_state` (OM:9568-9593). So the cumulus slot (OM:9664-9823) sees the time-n post-surface state.
  Exception: non-restoring PBLs (e.g. the legacy `PBL_SCAN_ADAPTERS` path OM:9540, QNSE `pbl_after_state`, and every PBL when rad_rk_tendf=0)
  leave the post-PBL Euler state in `next_state`. Tiedtke/NTiedtke read `next_state`; KF99/cuplus read `before`.

---------------------------------------------------------------------------------------------------------------------------
## 1. cu_physics=6 modified Tiedtke
(a) Branch OM:9666-9674 `tiedtke_adapter(next_state, dt, grid, qvften=_tiedtke_qvften_from_flux_advection(next_state), qvpblten=qvpblten)`.
The tendencies are applied as an Euler increment to `next_state` (SA:1006-1027). WRF: CD:1389-1416 `CU_TIEDTKE`; scheme `phys/module_cu_tiedtke.F:115-442`.
Admission: requires use_flux_advection (OM:7313/7339). The e213b65c4 fixes are present: total RHO SA:934 (`_wrf_phy_prep_rho_from_state`),
dry T_PHY SA:936 (`_dry_theta_view`), moist-form theta SA:1016 (`_theta_m_tendency_from_dry`). PASS.

| item | WRF | port | verdict |
|---|---|---|---|
| T3D | P1:1379 `T=grid%t_phy` (dry th_phy*pi, BSU:4834/4855) | SA:936 dry theta * (p/p0)^(2/7) | MATCH (fixed) |
| RHO3D | P1:1380 `grid%rho`=(1+qv)/alt (BSU:4856) | SA:934 same reconstruction | MATCH (fixed) |
| PI3D | pi_phy from full p | SA:937 `_exner_columns(state.p)` | MATCH |
| QV/QC/QI | QV_CURR etc., raw | SA:967-969 clipped at >=0 | MATCH except where values are negative (D) |
| U/V/W | u_phy/v_phy (A-grid), w_2 (faces) | `_u_mass/_v_mass`, state.w faces | MATCH |
| PCPS | P1:1380 `P=grid%p_hyd` (BSU:4968) → CD:1404 `PCPS=p` | SA:940 `p=state.p` (full non-hydrostatic) | MISMATCH (B1) |
| P8W | P1:1394 `P8W=grid%p_hyd_w` (BSU:4946/4957: p_top at the top, hydrostatic psfc at the bottom) | SA:941-944 mean of adjacent p; bottom face = p(k=1), top face = p(kte) | MISMATCH (B1) |
| DZ8W | (ph+phb)/g, g=9.81 (BSU:4869) | SA:938 ph/9.80665 | MISMATCH tiny (B8) |
| ZNU | P1:1519 `ZNU=grid%znu` → sig1 (tiedtke.F:388, used :2343 cevapcu) | SA:945-948 proxy (p-p(kte))/(p(1)-p(kte)) | MISMATCH (B6) |
| QVFTEN | prev. step final RK stage advect_tend*msfty / (c1*muts+c2) (module_em.F:1365-1373, solve_em:2299-2301, BSU:5291-5302) | OM:5473-5500 RK-stage-1 unlimited flux tendency on the current state | MISMATCH (B3) |
| QVPBLTEN | RQVBLTEN of this step | `pbl_moist_raw["qv"]` (OM:9636-9640) | MATCH |
| itimestep==1 | tiedtke.F:364-370 zeroes q1b/q1bl at step 1 | kernel has no itimestep, never zeroes | MISMATCH, first step only (B3) |
| QFX | grid%qfx (post-LSM) | SA:957 `qv_flux*rhosfc` | MISMATCH small (B7) |
| XLAND | grid%xland | state.xland | MATCH |
| stepcu/cudt | delt=dt*stepcu, called every stepcu steps, rates held between calls | stepcu=1 every step whatever cudt is | MISMATCH when cudt>0 (B5) |
| outputs | ADD:1583-1618 phy_cu_ten TIEDTKE case: RTHCUTEN, RUCUTEN/RVCUTEN (a2c), RQV/RQC/RQI (QI only if P_QI) through RK | Euler increment of theta_m/qv/qc/qi/u/v (qr/qs zero) | FORM MISMATCH (A3); P_QI guard missing (B9) |
| RAINCV→LSM | surface_driver:3283 `MP_RAINC=RAINCV` → noahmpdrv:778 PRCPCONV | OM:10399 convective_rate=0 (cumulus_tendencies is None) | DROPPED (A2) |
| RAINC | advance_ppt RAINC+=PRATEC*DT | rainc_acc+=RAINCV | MATCH (stepcu=1) |

## 2. cu_physics=16 New Tiedtke
(a) Branch OM:9675-9693: `qvften16 = flux-adv + qvpblten`, `thften16 = (next_state.theta - before.theta)/dt`, `ntiedtke_adapter` (SA:1031-1161),
Euler increment. WRF: CD:1418-1442 `CU_NTIEDTKE_DRIVER`, pre-processing CD:864-881, `phys/module_cu_ntiedtke.F:19-529`.

| item | WRF | port | verdict |
|---|---|---|---|
| T3D | `t_phy` (dry) | SA:1074 `_temperature_from_theta(state.theta,...)`, i.e. theta_m (moist) | MISMATCH (A1) |
| RHO3D | total (1+qv)/alt | SA:1073 `_rho_from_state`, dry density 0.622p/(Rd T(qv+0.622)) | MISMATCH (B4) |
| PCPS/P8W | p_hyd / p_hyd_w | SA:1078-1081 same proxy as cu=6 | MISMATCH (B1) |
| THFTEN | CD:875-876 `(RTHFTEN+RTHRATEN+RTHBLTEN)*pi`. RTHFTEN = theta advection (module_em.F:654-665) decoupled and made dry (BSU:5309-5331) | OM:9685-9688 theta_m difference ≈ 0 by default (PBL restored, radiation via t_tendf), no *pi, no advection | MISMATCH (B2) |
| QVFTEN | CD:877 RQVFTEN+RQVBLTEN | flux-adv(RK1, current state)+qvpblten | MISMATCH (B3) |
| itimestep | pre_run :451 zeroes forcing at step 1 | SA:1130 hardcodes itimestep=2 | MISMATCH, first step only (B3) |
| HFX/QFX | grid%hfx / grid%qfx | SA:1091-1092 rhosfc*1004*theta_flux, rhosfc*qv_flux | MISMATCH small (B7) |
| DX | CD:1426 `DX=dx2d` = dx (physics_init :5686) | grid.projection.dx_m | MATCH |
| DZ8W, stepcu, outputs, RAINCV→LSM | as cu=6 | as cu=6 | B8, B5, A3/B9, A2 |
| theta output | RTHCUTEN (dry) → conv_t_tendf_to_moist (part2:1009) | SA:1151 `theta += dt*rth`, no moist conversion | MISMATCH (A1) |

## 3. cu_physics=99 old Kain-Fritsch (KFSCHEME)
(a) Branch OM:9730-9739 `kf99_cadence_step(before, next_state, ...)` (KF99:59-88). Inputs come from the time-n `before`.
Held rates are applied as an Euler increment with the moist-theta conversion at the old state. WRF: CD:958-990 `KFCPS` (`phys/module_cu_kf.F`),
advance_ppt ADD:2291 (RAINC+=PRATEC*DT), :2304-2322 (KFSCHEME NCA clear).

| item | WRF | port | verdict |
|---|---|---|---|
| T / PI | t_phy, pi_phy | KF99:34-36 dry theta * pi(full p) | MATCH |
| PCPS | p_hyd | KF99:39 `_wrf_hydrostatic_pressure_profiles_from_state` | MATCH |
| DZ8W / RHO | g=9.81, total rho | KF99:37-40 /9.81, phy_prep rho | MATCH |
| W0AVG | KFCPS: TST=2*stepcu running mean, inside the run gate | KF99:41 + cumulus_kf99.py:736-750, inside `lax.cond(run)` | MATCH |
| gate | CD:831-858 | KF99:64 `(step==1)|(stepcu==1)|(mod==0)|(cudt==0)` | MATCH |
| NCA hold | KFCPS `NCA>0.5*DT` keeps old rates | KF99:51-52 | MATCH |
| advance_ppt | NINT(NCA/DT)<=0 clears 6 rates; NCA-=DT; RAINC+=PRATEC*DT | KF99:84-87, :82 | MATCH |
| warm_rain / F_Q* | warm_rain only for Kessler (physics_init:4463-4484); F_QI/F_QS/F_QR | KF99:45-46, :54-55 species mask | MATCH |
| outputs | ADD:1125-1197 KFSCHEME: theta,qv,qc,qr,qi,qs via add_a2a through RK | kf_add_a2a + Euler (KF99:66-79) | FORM MISMATCH (A3) |
| RAINCV→LSM | PRCPCONV=RAINCV/DT | OM:10399 rates[6]=held PRATEC | MATCH |
| prec_acc buckets | RAINCV | OM:10409 zero for cu=99 | MISMATCH (C1) |

## 4. cu_physics=10 KF-CuP and 5. cu_physics=11 MSKF (shared `cuplus`)
(a) Branch OM:9766-9783: `cuplus.driver(before, next_state, ...)` (CP:33-104, CPD:41-147), `couple(before.mu_total)` (CP:152), rates go to
`kf_add_a2a` → `kf_tendf` → `_source_leaf_dry_tendencies` t_tendf/qv (OM:10052) + moist_tendf (OM:10063-10071). MSKF DUDT/DVDT go through
`momentum_sources` (add_a2c, CP:167) → ru/rv_tendf. Post-RK `finish_coupled` (advance_ppt + decouple by next mu, OM:10341, CP:159; CPD:151-163).
Both require rad_rk_tendf=1 (OM:8893), Thompson mp=8, MYNN 5/5, RRTMG 4/4, trigger=1, aercu_opt=0, shcu_aerosols_opt=0 (cuplus_config.py:41-76).
WRF: MSKF CD:1051-1093 (`module_cu_mskf.F` MSKF_CPS :3420ff), KF-CuP CD:1515-1584 (`module_cu_kfcup.F`), phy_cu_ten ADD (MSKF :1285-1366, KFCUP :1212-1283).

| item | WRF | port | verdict |
|---|---|---|---|
| T / TH / PI | t_phy, th_phy (dry), pi_phy | CP:52-53 dry th, pi with 287/1004.5 | MATCH |
| PCPS | p_hyd | CP:51 phyd | MATCH |
| RHO / DZ8W / Z / Z_AT_W / HT | phy_prep total rho, g=9.81, grid%z, z_at_w, ht | CP:56-58 | MATCH |
| U/V/W/W0AVG | A-grid; W0AVG TST=2*stepcu updated before the NCA test | CP:54-55, CPD:94-98 | MATCH |
| gates / NCA | CD:831-858; `NCA .ge. 0.5*DT` (mskf.F:3843, kfcup.F:516) | CPD:82, :100 | MATCH |
| PBLH/HFX/QFX/UST | grid fields after surface+PBL | CP:59 from `next_state` (B39 hfx/qfx carry) | MATCH |
| ZOL (11) | grid%zol | `cuplus_zol` from MYNN-SL (OM:9130/9239) | MATCH |
| KPBL (10) | MYNN kpbl (module_bl_mynnedmf.F:5568-5574: first zw>=pblh, k-1) | CP:21-31 identical rule | MATCH |
| PSFC/TSK/MAVAIL/BR/REGIME/T2/Q2 (10) | passed | not passed | MATCH in effect: with sf=5 only `UnstableOrNeutral` uses them and it is dead (kfcup.F:4679-4743) |
| RQVFTEN (11) | passed, used only for trigger=2 (mskf.F:3687-3831) | not passed, trigger 2 refused | MATCH under admission |
| XLAND (11) / aerosol | declared, unused / aercu_opt>0 | refused | MATCH |
| RTHCUTEN | DTDT/pi | CPD:113 | MATCH |
| outputs | phy_cu_ten via calculate_phy_tend coupling at time-n mut, add_a2a/a2c, conv_t_tendf_to_moist, decouple by muts | CP:152-165, OM:9779-9783, OM:10052-10071 | MATCH |
| radiation clouds | MSKF icloud_cu=2 always (physics_init:1105-1117): radiation_driver:1336-1347 merge; CuP CLDFRA_CUP :1510-1540 | CP:188-212 (radiation scope: structure only) | MATCH (UNVERIFIED detail) |
| RAINCV→LSM | PRCPCONV | OM:10401 held PRATEC | MATCH |

---------------------------------------------------------------------------------------------------------------------------
## FINDINGS (ranked by forecast impact)

### Class A
**A1 (cu=16): wrong prognostic temperature input and wrong output form. These are the e213b65c4 Tiedtke defects, still present in New Tiedtke. HIGH confidence.**
- Port SA:1074 `T = _temperature_from_theta(state.theta, state.p)`: State.theta is theta_m (use_theta_m=1), so T is the moist virtual-like
  temperature, about +1.6·qv·T warmer (≈ +7 K at qv=15 g/kg). WRF P1:1379 `T=grid%t_phy` = dry th_phy*pi (BSU:4834/4855).
- Port SA:1151 `theta=(state.theta + dt_f*_back(rth))`: dry RTHCUTEN is added directly to theta_m. The (1+Rv/Rd·qv) factor and the
  Rv/Rd·θm/(1+Rv/Rd·qv)·RQVCUTEN term of conv_t_tendf_to_moist (part2:1009, BSU:6674-6707) are missing. In convecting columns the qv term
  is of the same order as RTHCUTEN.
- Affects every cu=16 run. Fix: copy cu=6 lines SA:934-936 and SA:1016-1018 into `ntiedtke_adapter`, which also fixes RHO (B4).
  Oracle: pristine 1-step dump of RTHCUTEN/RQVCUTEN and t_2 for cu=16 on a real wrfinput, compared with the adapter on the same state.
  The existing frozen ntiedtke column oracle cannot see this because it feeds T directly.

**A2 (cu=6, cu=16; same pattern for every stateless cumulus without `cumulus_tendencies`): convective rain never reaches the LSM. HIGH.**
- Port OM:10398-10399: `rates = carry.cumulus_tendencies; convective_rate = zeros if rates is None else rates[6]`. cu=6/16 never seed
  `cumulus_tendencies` (OM:7555-7724), so `precipitation_from_step(...)` gives Noah-MP PRCPCONV=0. The `total` term contains MP precipitation only
  (noahmp_precipitation.py:23-41).
- WRF: surface_driver:3283 `MP_RAINC = RAINCV` → noahmpdrv:778 `PRCPCONV = MP_RAINC/DT`. RAINBL also includes RAINCV (:1568).
- Affects cu=6/16 with Noah-MP/RUC/SSiB and mp 8/7/38 (OM:7793). Soil moisture, canopy water, runoff and the next step's LH/HFX all miss the
  Tiedtke rain. Fix: return the Tiedtke RAINCV/PRATEC from the adapter, hold it in the carry, and use it as `convective_rate`. Oracle: 3 h
  cu=6+Noah-MP pristine run; compare SMOIS(1)/CANLIQ/SFROFF in convective cells (RAINC>0).

**A3 (cu=6, 16, 99): cumulus tendencies applied as a step-entry Euler increment, not through the WRF RK physics sources. HIGH confidence, MEDIUM impact.**
- Port SA:1006-1027 / SA:1141-1160 / KF99:66-79 do `state += dt*rate` before the dycore. Under the default MYNN restore nothing removes this.
  So the RK3/acoustic loop starts from a cumulus-modified state, and theta/u/v never enter t_tendf/ru_tendf.
- WRF: ADD phy_cu_ten TIEDTKE :1583-1618 and KFSCHEME :1125-1197 put mass-coupled sources (calculate_phy_tend part2:392, time-n mut) into
  rt_tendf/moist_tendf/ru/rv_tendf (add_a2c). They are integrated at every RK stage and acoustic substep, and the KF held rate does a
  mut/muts round trip. The port already does this for cu=1 (KF_TEND_RK_WRF) and cu=7/10/11 (OM:9779-9783 + :10052-10071).
- Fix: route the six/eight rates into `kf_tendf`/`cuplus_momentum` exactly as the cuplus branch does (couple with before.mu_total, kf_add_a2a,
  momentum_sources), and remove the Euler adds. Oracle: compare one root step against a pristine 1-step dump of t_2/qv/u after RK
  (single-stage difference), as in the KF_TEND_RK_WRF qualification.

### Class B
**B1 (cu=6, cu=16): pressure inputs are non-hydrostatic p plus a crude interface proxy, not WRF p_hyd/p_hyd_w. HIGH confidence, MEDIUM-HIGH impact.**
- Port SA:940-944 and SA:1078-1081: `p8w = concat([p[:1], 0.5*(p[:-1]+p[1:]), p[-1:]])`. Bottom face = lowest mass-level p (not psfc),
  top face = p(kte) (not p_top), and PCPS = full p.
- WRF P1:1380/1394 `P=grid%p_hyd, P8W=grid%p_hyd_w` (BSU:4946-4968), then CD:1404/1428 `PCPS=p, P8W=p8w`, tiedtke.F:375/382.
- The lowest and top layer dp=paph(k+1)-paph(k) are halved. That doubles the lowest-layer flux-divergence tendencies and shrinks the column
  moisture-convergence integrals that close the scheme. The fix is a few lines because the helper exists (used by KF99:39 and CP:51):
  `phyd, faces, _ = _wrf_hydrostatic_pressure_profiles_from_state(state, grid.metrics)`. Oracle: the cu=6/16 driver-argument dump
  (P8W(1)=PSFC_hyd, P8W(kte+1)=p_top), then a column oracle fed WRF-staged paph.

**B2 (cu=16): THFTEN is essentially zero and in the wrong units. HIGH.**
- Port OM:9685-9688 `thften16 = (next_state.theta - before.theta)/dt`. By default this is about 0: the PBL theta is restored (OM:9590) and
  radiation goes via t_tendf. It is theta_m, has no ×π, and has no advective part.
- WRF CD:875-876 `RTHFTEN=(RTHFTEN+RTHRATEN+RTHBLTEN)*pi`, with RTHFTEN = last-stage theta advection (module_em.F:654-665), decoupled by muts
  and converted to dry (BSU:5309-5331). NTiedtke's closure loses all large-scale temperature forcing.
- Fix: thften = (dry advective tendency + held RTHRATEN + rthblten) * pi_phy. Oracle: driver-argument dump of THFTEN at step 2.

**B3 (cu=6, cu=16): RQVFTEN time level and stage. MEDIUM.**
- Port OM:5487-5500 recomputes `_moisture_coupled_tendencies(..., rk_step=1)` on the current time-n state (no limiter, instantaneous coupled
  winds) and divides by c1h·mu+c2h.
- WRF: set_tend every RK stage, so the stored value is the previous step's final stage (module_em.F:1365-1373; for moist_adv_opt≥1 the PD/mono
  limited advect_tend of :1261-1295), with acoustic-averaged ru_m/rv_m/ww_m (solve_em:2299-2301), ×msfty (BSU:5827), decoupled by end-of-step
  muts (BSU:5291-5302).
- WRF also zeroes QVFTEN/QVPBLTEN (cu=6, tiedtke.F:364-370) and QVFTEN/THFTEN (cu=16, ntiedtke.F:451-458) at itimestep==1. The port never
  does (the cu=6 kernel has no itimestep argument; cu=16 uses SA:1130 `itimestep=2`).
- Fix: carry RQVFTEN from the previous step's final-stage tendency (the scalar loop already computes it) and pass the real itimestep.
  The msfty factor is UNVERIFIED in the port (PROD msf≈1).

**B4 (cu=16): dry density instead of total.** SA:1073 `_rho_from_state` vs P1:1380 grid%rho=(1+qv)/alt (BSU:4856). It enters the omega used by
the scheme (ntiedtke.F pre_run `dot=-0.5*grav*rho*(w+w)`), about 1–2% low. Fixed together with A1.

**B5 (cu=6, cu=16): cudt ignored.** OM:9668/9687 call the adapters with the default `stepcu=1` every step. WRF stepcu=nint(cudt*60/dt)
(physics_init:4027), delt=dt*stepcu, scheme called every stepcu steps, R*CUTEN held in between. No admission refusal exists, although Grell
has one (OM:7333). Affects only cudt>0 namelists. Fix: refuse cu=6/16 with cumulus_cadence_steps>1, or hold the rates as cu=99 does.

**B6 (cu=6): ZNU proxy.** SA:945-948 `(p-p8w[-1])/(p8w[0]-p8w[-1])` (1.0 at the lowest level, 0 at the top) vs P1:1519 `ZNU=grid%znu`
(tiedtke.F:388 sig1, used only at :2343 `cevapcu=cevapcu1*sqrt(cevapcu2*sqrt(sig1))`). This changes the sub-cloud rain-evaporation
coefficient by up to a few %. Fix: znu = c3h + c4h/(p0 - p_top) from the hybrid metrics (or read ZNU from wrfinput).

**B7 (cu=6 QFX; cu=16 QFX+HFX): fluxes rebuilt from kinematic handles.** SA:957, SA:1091-1092. The Noah-MP producer stores
`qv_flux = qfx/ρ_sfclay` and `theta_flux = hfx/(ρ_sfclay·cp(1+0.84qv))`, but `rhosfc = ρ_MYNN-PBL` (noahmp_coupler.py:297/318/449-469).
So QFX is scaled by ρ_pbl/ρ_sfc, and HFX is about 1–2% low over moist columns (1004 vs cpm). WRF passes grid%hfx/grid%qfx. The port carries
them as State.hfx/qfx (B39, written by noahmp_surface_hook.py:336-338) and the cuplus driver already uses them (CP:59).
Fix: use `state.hfx/state.qfx`.

**B8 (cu=6, cu=16): DZ8W uses g=9.80665** (SA:938/1076, `GRAVITY_M_S2` physics_couplers.py:90) vs WRF g=9.81 (BSU:4869). 0.03%, the same class
as the WSM DELZ fix. Use `WRF_PHYSICS_G_M_S2`.

**B9 (cu=6, cu=16): RQICUTEN applied without the P_QI/F_QI guard.** SA:1024/1158 always add `dt*rqi` to state.qi. WRF writes rqicuten only
`if (f_qi)` (tiedtke.F after :410) and ADD:1613 adds it only `if (P_QI >= PARAM_FIRST_SCALAR)`. With a microphysics that has no QI (Kessler 1,
WSM3 3) the port creates non-WRF cloud ice that radiation and advection then see. Conditional; admission for these pairs is UNVERIFIED.
Fix: mask qi (and qc) by the mp species set as KF99:54-55 does.

### Class C
- **C1** OM:10409 `raincv = rain_increment if cu in (1,7,10,11) else 0`: prec_acc buckets (PREC_ACC_C) miss cu=6/16/99 convective rain.
  (rainc_acc itself is correct.)
- **C2** cu=6/16: RAINCV/PRATEC are not kept as held/history fields (WRF RAINCV/PRATEC history). CUTOP/CUBOT are not produced by WRF Tiedtke
  either (no HTOP/HBOT args), so that part matches.

### Class D / notes
- SA:967-969 / SA:1103-1105 clip qv/qc/qi≥0 before the kernel. WRF passes raw values; this differs only where values are negative.
- The cu=6/16 Euler A2C helper (physics_couplers.py:2280) excludes the edge faces but not WRF's specified row range. Moot once A3 is fixed.
- KF99 held rates are not round-tripped through mut/muts (WRF couples with the step's mut and decouples with muts). Negligible; it disappears with A3.
- cuplus (10/11): every input and output item checked matches WRF under the admitted profile. The only open items are radiation-merge details
  (outside this lane's scope).
