# t-calleraudit — PBL group A (MYJ 2, GFS 3, MRF 99, ACM2 7, BouLac 8) — E11 caller-wiring audit

Read-only static audit, worktree `lane/t-calleraudit` @ b51dd0805 (port paths relative to `src/gpuwrf/`), pristine WRF 4.7.1
`<USER_HOME>/src/wrf_pristine/WRF`. Release defaults assumed: `rad_rk_tendf=1` ⇒ `source_leaf_mode=True`,
`GPUWRF_PHYS_TEND_RK_WRF=1`, `use_theta_m=1` (⇒ `State.theta` = theta_m, cf. `coupling/physics_couplers.py:522-533`
`_dry_theta_view` and `io/wrfout_theta_convention.py:4-5`), `bldt=0`. No code run.

## 0. Branch map (which pairing reaches which PBL branch)

| port branch (runtime/operational_mode.py) | reached by | coupling form |
|---|---|---|
| `elif bl_opt == 2:` / `if cu_opt==7` rates (:9425-9431) | MYJ2+Janjic2 with cu=7 only | raw rates → `_source_leaf_dry_tendencies` (RK, conv_t_tendf_to_moist) |
| `elif bl_opt == 2:` else `next_state = myj_pbl_adapter(...)` (:9432-9433) | **MYJ2+Janjic2 with every cu≠7 (the default MYJ composite)** | **Euler state increment**, applied post-RK by `_apply_physics_non_dry_updates` (:8375-8396, call :10326; u/v/theta/qv/qc/qke ∈ `_PHYSICS_NON_DRY_INCREMENT_FIELDS` :515-547) |
| `elif sf1_surface_forcing is not None and bl_opt in PBL_SCAN_ADAPTERS` (:9517-9539) | ACM2/BouLac/MRF (admission forces sf=1: `io/namelist_binding.py:693`, `coupling/physics_dispatch.py:106,688`), and GFS3 only if user picks sf=1 | raw dry rates (`pbl_rates`) → RK; only keys present in `pbl_rates` among qc/qi are coupled (:9533-9536) |
| `elif bl_opt in PBL_SCAN_ADAPTERS:` legacy (:9540-9544) | **GFS3 with sf≠1, i.e. every WRF-legal GFS pairing (sf=3 GFS / 88)**; nothing else in my group | Euler state increment post-RK, legacy forcing re-derived by MYNN-SL |
| urban `sf_urban_physics in (2,3)` (:9403-9424) | BouLac8+BEP/BEM | out of scope |

WRF admission (`phys/module_physics_init.F:3698-3800`): MYJ needs isfc=2 (sf 2/3/88); **GFS needs isfc=2 (sf 2/3/88, fatal
with sf=1)**; MRF isfc=1 (sf 1/91); ACM2 isfc 1/7; BouLac isfc 1/2. Port: MYJ↔sf2 forced pair (physics_dispatch.py:679);
ACM2/BouLac/MRF forced to sf=1 (refusals only → fine); **GFS has no constraint** (physics_dispatch.py:106 set excludes 3;
`io/scheme_catalog.py:187-189` documents GFS as "consumes the revised-MM5 surface layer (sf_sfclay=1)", a WRF-fatal pairing).

WRF call side common to all (`dyn_em/module_first_rk_step_part1.F:1113-1316` → `phys/module_pbl_driver.F`):
TH_PHY=dry th_phy (`module_big_step_utilities_em.F:4834-4848`), T_PHY=th_phy·pi_phy, PI_PHY from full p+pb,
**P_PHY=grid%p_hyd, P8W=grid%p_hyd_w** (:1130-1132; built in phy_prep :4940-4970), RHO=grid%rho=(1/alt)(1+qv),
DZ8W/Z from (ph+phb)/g, g=9.81; PSFC=p8w(kms) (pbl_driver.F:1011); TSKOLD=TSK at pbl_driver entry (:1004, i.e. post-LSM);
HFX/QFX/LH/UST/TSK/QSFC = grid fields after surface_driver+LSM; DTBL=dt (bldt=0). Outputs are rates only; `phy_bl_ten`
(`phys/module_physics_addtendc.F:237-1070`) couples them into the RK (rt/ru/rv_tendf, moist_tendf) at RK cadence, and
`conv_t_tendf_to_moist` converts the dry RTHBLTEN (+RQVBLTEN) to theta_m.

## 1. MYJ (bl=2, with sf=2 Janjic)

(a) Default branch: `runtime/operational_mode.py:9432-9433` (cu≠7) → `physics/myj_adapters.py:188-257` → `physics/bl_myj.py:463-610`.
WRF: `phys/module_pbl_driver.F:1439-1465` `CALL myjpbl(...)`; coupling `module_physics_addtendc.F:505-603`.

(b)
| item | WRF | port | verdict |
|---|---|---|---|
| coupling form | rates → RK each stage via phy_bl_ten add_a2a/add_a2c (addtendc.F:507-523) | `theta_new=state.theta+dt*RTHBLTEN`, `u/v` A2C increment, `qv_new=state.qv+dt*RQVBLTEN` (myj_adapters.py:228-235), applied **after** RK as Euler increment (operational_mode.py:10326, :8383-8389) | MISMATCH (A) |
| theta form | dry RTHBLTEN → conv_t_tendf_to_moist: (1+Rv/Rd qv)·t_tendf + Rv/Rd·θ·qv_tend | dry-type rate added straight to theta_m (myj_adapters.py:231), no factor, no Rv/Rd·θ·ΔQV term | MISMATCH (A) |
| TH, T inputs | TH=th_phy dry, T=t_phy dry (part1.F:1138-1139) | `theta_cols=state.theta` (theta_m), `T=theta_m·Π` (myj_adapters.py:83,98) | MISMATCH (A, wrong prognostic input) |
| QCI / CWM | QCI=qi_curr always present ⇒ flQI=.TRUE., CWM=QCW+QCI, NSPEC=4 (myjpbl.F:270-273, 331-332, 505-509) | `cwm=qc_top` only; species row 4 (qci) "stays zero" (bl_myj.py:495, 536-537) | MISMATCH (A) |
| RQCBLTEN | computed (myjpbl.F:646) and applied to moist_tendf(P_QC) when adv_moist_cond (addtendc.F:567-573; adv_moist_cond=.true. except Ferrier, physics_init.F:4464/4495) | qc mixed inside VDIFH and used in THNEW (bl_myj.py:573-576) but **no RQCBLTEN output**; qc never updated (myj_adapters.py:247, return dict bl_myj.py:593-609) | MISMATCH (A) |
| RQIBLTEN | computed (myjpbl.F:647), applied if P_QI active (addtendc.F:574-580) | absent | MISMATCH (A) |
| PMID / PINT | p_hyd / p_hyd_w (part1.F:1130-1132) | full `state.p`; PINT = midpoint, **PINT(surface)=p[0]**, PINT(top)=p[-1] (myj_adapters.py:89-92) | MISMATCH (B) |
| PSFC for THSK/QSFC-sea | PINT(LOWLYR)=p_hyd_w(1) (myjpbl.F:476-477, 560) | `pint_b[0]`=lowest mass-level p (bl_myj.py:527, 551) | MISMATCH (B) |
| AKHS/AKMS/USTAR/THZ0/QZ0/UZ0/VZ0/CT | grid fields from the surface-driver MYJSFC call (pre-LSM TSK, persistent THZ0/QZ0/UZ0/VZ0/QSFC, sf_myjsfc.F:300-305, 472-516) | **second Janjic solve inside the PBL adapter** with post-LSM `t_skin`, seeds thz0=θ_m(k0), qsfc=qv0, uz0=vz0=0, pblh=1000, psfc=p[0] (myj_adapters.py:110,120-136,205) | MISMATCH (B) |
| ELFLX | ELFLX=grid%lh (LSM latent heat) (pbl_driver.F:1453) | `sfc["flx_lh"]` = re-run Janjic LH (myj_adapters.py:213) | MISMATCH (B) |
| CHKLOWQ | surface_driver: 1.0 after Noah-MP (surface_driver.F:3429) | `clip(state.mavail,0,1)` (myj_adapters.py:212) | MISMATCH (B, value-dependent) |
| QSFC | grid%qsfc (LSM) | re-run Janjic qsfc | MISMATCH (B; only used where QFC1≤0) |
| DT/STEPBL | DT=dt, STEPBL (myjpbl.F:289) | dt, stepbl=1 | MATCH (bldt=0) |
| DZ, HT | dz8w, ht with g=9.81 | ph/9.80665 (myj_adapters.py:86) | MISMATCH (D, 0.03 %) |
| TKE carry | TKE_PBL=grid%tke_pbl; init TKE_MYJ=EPSQ2 (myjpbl.F:1721-1725) | qke=2·TKE leaf ✓; cold start seeds MYNN equilibrium QKE for all non-GBM PBLs (`integration/d02_replay.py:2537-2550`) | MISMATCH (B, first hours) |
| PBLH/KPBL/EXCH_H/EL_PBL | written (myjpbl.F:401-463) → history, consumers | default branch writes neither PBLH nor KPBL/EXCH_H/EL (myj_adapters.py:241-252; pblh only if `return_cumulus_inputs`) | MISMATCH (C) |
| later drivers' input | cumulus reads phy_prep state + RQVBLTEN | non-KF cumulus (cu 6/16/5/93/2/3) reads post-PBL `next_state`; Grell gets no rthblten (pbl_restored False, operational_mode.py:9702-9706) | MISMATCH (B) |

(c) Findings (MYJ)
- **MYJ-A1 (HIGH) Euler post-RK increment instead of RK-integrated rates, theta applied in dry form to theta_m.**
  Port `myj_adapters.py:231` `theta_new = (state.theta + dt_f * _back3d(out["RTHBLTEN"],...))` and `operational_mode.py:9433`
  `next_state = myj_pbl_adapter(next_state, ...)` (rthblten/pbl_moist_raw stay None ⇒ `_source_leaf_dry_tendencies` gets no BL
  source, :10052-10063; the increment is re-applied after RK by `updates[name] = dynamics_value + (physics_value - reference_value)` :8389).
  WRF: `CALL add_a2a(rt_tendf,RTHBLTEN,...)` (addtendc.F:507) + conv_t_tendf_to_moist. Missing term Rv/Rd·θ·ΔQV ≈ 0.48 K per g/kg
  of PBL moistening (dry θ falsely drops wherever PBL adds vapour; [I] ~0.1 K/h in a moist daytime BL), missing (1+1.61qv)
  factor on the heating, momentum drag not in ru/rv_tendf (no acoustic/RK interplay, uncoupled from mu). Affects every MYJ run
  with cu≠7. Fix: call the existing `return_tendencies=True` path for all cu (as the cu=7 branch already does) and set
  `pbl_moist_raw={'qv':..,'qc':..,'qi':..}`; the RK fold then does conv_t_tendf_to_moist. Oracle: pristine single-step WRF
  bl=2/sf=2 dump of RTHBLTEN/RQVBLTEN/RQCBLTEN + post-step T/QVAPOR vs GPU step; deletion test = revert to Euler branch.
- **MYJ-A2 (HIGH) wrong prognostic input: TH and T are moist (theta_m).** Port `myj_adapters.py:83,98`
  `T = _temperature_from_theta(state.theta, state.p)`, `"theta_cols": _cols(state.theta)`; WRF `TH=th_phy,T=t_phy`
  (part1.F:1138-1139; th_phy = theta_m/(1+Rv/Rd qv), big_step_utilities_em.F:4834). θ_m exceeds θ by ≈1.61·qv·θ (≈4.8 K at
  10 g/kg), while the MYJ lower boundary THZ0=THSK is dry (bl_myj.py:527,546) ⇒ the implicit surface heat flux
  AKHS·(THZ0−θ_low) is biased strongly negative and the stability/mixing length (MIXLEN uses THE/T) is shifted. Also feeds the
  in-PBL Janjic re-solve (thlow, THZ0 seed) and the surface-slot `janjic_sfclay_adapter` (same `_myj_geometry`). Same defect
  in the cu=7 rates branch. Fix: build TH/T with `_dry_theta_view`/`_dry_temperature_from_state` (or `urban_pbl_forcing`).
  Oracle: MYJPBL column driver fed phy_prep operands from one real Swiss d01 state vs port adapter on the same state.
- **MYJ-A3 (HIGH) cloud ice not mixed and RQCBLTEN/RQIBLTEN never applied.** Port `bl_myj.py:495` `cwm = qc_top`,
  `:537 # row 4 (qci) stays zero`, return dict has only RUBLTEN/RVBLTEN/RTHBLTEN/RQVBLTEN; WRF myjpbl.F:331-332
  `IF (flQI) CWM=CWM+QCI`, :646-647 RQCBLTEN/RQIBLTEN, applied addtendc.F:567-580. Worse than a missing species: port THNEW =
  THE_new + CWM_new·ELOCP/Π (bl_myj.py:576) contains the latent-heat consequence of the qc redistribution while qc itself is
  not changed ⇒ energy/water inconsistency in every cloud-topped BL; ice clouds get a θ_l that ignores qi. Fix: species row 4 =
  qi, cwm = qc+qi, return RQCBLTEN/RQIBLTEN, couple via pbl_moist_raw. Oracle as A1 (cloudy column fixture with qc,qi>0).
- **MYJ-B1 (HIGH) PBL surface coupling comes from a private second Janjic solve, not from the surface driver + LSM.**
  Port `myj_adapters.py:205` `sfc = _run_janjic(g)` with seeds `qsfc=qv0, thz0=th0, ..., uz0=0.0, vz0=0.0`, `pblh=1000` (:134-135),
  post-LSM `t_skin`; `elflx = sfc["flx_lh"]` (:213); `chklowq = jnp.clip(g["mavail"],0,1)` (:212). WRF passes
  `AKHS=grid%akhs, THZ0=grid%thz0, QZ0=grid%qz0, UZ0/VZ0, QSFC=grid%qsfc, CHKLOWQ=chklowq, ELFLX=grid%lh, USTAR=grid%ust`
  (part1.F:1114,1121,1130,1133,1138-1141; pbl_driver.F:1447-1453) — MYJSFC values with time memory (sf_myjsfc.F:472-516
  `THZ0=((WGHTT*THLOW+THS)/(WGHTT+1.)+THZ0)*0.5`, `UZ0=(ULOW*RWGH+UZ0)*0.5`) and the LSM's LH with CHKLOWQ=1 under Noah-MP
  (surface_driver.F:3429), which makes MYJ's land moisture flux equal the LSM's (myjpbl.F:549-555). Port moisture flux =
  Janjic's own LH ⇒ PBL Q/T budget not the LSM's. Affects all MYJ runs (land; sea via UZ0/VZ0/THZ0/QZ0 memory). Fix: carry
  THZ0/QZ0/UZ0/VZ0/AKHS/AKMS/QSFC/CT from the surface slot (needs carry leaves), pass grid LH and CHKLOWQ=1 for Noah-MP.
  Oracle: pristine 2-step run dumping AKHS/THZ0/QZ0/QSFC/LH at PBL entry vs port values.
- **MYJ-B2 (HIGH) pressure staging.** Port PINT = midpoints with PINT(sfc)=p[0] and PMID=full p (myj_adapters.py:89-92);
  WRF PMID=p_hyd, PINT=p_hyd_w (part1.F:1130-1132). THSK=TSK·(1e5/PSFC)^κ uses p[0] instead of surface p ⇒ [I] THZ0
  ≈ +0.2-0.3 K too warm over land for a 25-50 m first layer (extra upward heat flux); sea QSFC uses wrong PSFC. Fix: use
  `_wrf_hydrostatic_pressure_profiles_from_state` as `urban_pbl_forcing` does.
- **MYJ-B3 (MED) cold-start TKE**: MYNN equilibrium QKE seeded (d02_replay.py:2537-2550) vs WRF TKE_MYJ=EPSQ2 (myjpbl.F:1725).
- **MYJ-B4 (MED) later drivers see the post-PBL state** (Tiedtke/NTiedtke/Grell/GF/BMJ read `next_state`; Grell gets
  rthblten=None, operational_mode.py:9702-9706) instead of phy_prep state + RQV/RTHBLTEN. Disappears with the A1 fix.
- **MYJ-C1 (HIGH)** default branch never writes PBLH/KPBL/EXCH_H/EL_PBL (myj_adapters.py:241-252); WRF history PBLH and
  every PBLH consumer (obs nudging `next_state.pblh`, operational_mode.py:10037) see a stale field.
- MYJ-D1: g=9.80665 for z/dz (myj_adapters.py:86) vs WRF 9.81; QV/QC clipped ≥0 before entry (myj_adapters.py:99-100).

## 2. GFS (bl=3)

(a) WRF-legal pairings (sf=2/3/88; sf=2 refused by the port's MYJ pair rule) → legacy branch `operational_mode.py:9540-9544`
→ `coupling/scan_adapters.py:1804-1834` with `surface_forcing=None`. Port-advertised sf=1 (WRF-fatal, physics_init.F:3743-3745)
→ sf1 rates branch :9517-9539. WRF: `pbl_driver.F:1400-1424` `CALL bl_gfs(...)`; coupling addtendc.F:649-702.

(b)
| item | WRF | port (legacy branch unless noted) | verdict |
|---|---|---|---|
| surface forcing | grid UST/HFX/QFX/BR/WSPD/PSIM/PSIH/GZ1OZ0 from the selected sfclay (sf=3) + LSM | `diag = surface_layer_with_diagnostics(_surface_column_view(state))` = **MYNN-SL re-solve** (scan_adapters.py:1561; physics/surface_layer.py:1-10); hfx=diag.hfx, qfx=diag.lh/2.5e6 (:1583-1587); wspd=max(|V1|,0.1) (:1609); gz1oz0=ln(0.5dz/znt) (:1818) | MISMATCH (A/B) |
| TH/T input | dry t_phy (BL_GFS uses T3D, PI3D) | T=theta_m·Π (scan_adapters.py:1568) | MISMATCH (A) |
| P3D/PSFC | p_hyd / p_hyd_w(1) | full p; psfc=p[0] (:1823 fallback) | MISMATCH (B) |
| QC3D, QI3D | mixed: NTRAC=2 (qc), 3 if P_QI (bl_gfs.F:272-274, 366-400) | `jnp.zeros_like(f["qv_cols"])` as qc, no qi (scan_adapters.py:1821) | MISMATCH (A) |
| RQCBLTEN/RQIBLTEN | returned (bl_gfs.F:451-458), applied (addtendc.F:673-685) | kernel returns rqcblten but `tend` keeps u/v/theta/qv only (:1830-1831) — dropped in both branches | MISMATCH (A) |
| theta form | conv_t_tendf_to_moist | legacy: dry rate added to theta_m (`_pbl_state_rates` no-op without theta_m_cols, :1647-1653); sf1 branch: correct | MISMATCH (A, legacy) |
| coupling time | RK | legacy: Euler post-RK | MISMATCH (A, legacy) |
| UST/WSPD writeback | `UST(i,j)=SQRT(STRESS(i))`, WSPD overwritten (bl_gfs.F:464-466) | kernel returns `ust` (bl_gfs.py:520) but adapter discards it | MISMATCH (B/C) |
| TSK | TSKOLD = post-LSM TSK (pbl_driver.F:1004) | sf1: `result['tsk']=skin_before_lsm` (surface_handoff.py:9); legacy: state.t_skin | MISMATCH (B, sf1) |
| PBLH/KPBL | written | legacy: not written (`_pbl_diagnostic_state` needs surface_forcing, :1656-1659) | MISMATCH (C) |

(c) Findings (GFS)
- **GFS-A1 (HIGH) every WRF-legal GFS run takes the legacy branch with MYNN-SL forcing, theta_m input and a dry-form Euler
  increment.** Port `operational_mode.py:9540` `elif bl_opt in PBL_SCAN_ADAPTERS: next_state = PBL_SCAN_ADAPTERS[bl_opt](...)`
  (no surface_forcing) → `scan_adapters.py:1561` MYNN-SL solve, `:1568` `T = _temperature_from_theta(state.theta, state.p)`,
  `:1834` `_apply_pbl_increment(state, dt, _pbl_state_rates(tend,f))`; WRF bl_gfs gets the selected sfclay's grid fields,
  dry T3D, and phy_bl_ten rates. Port admission lets GFS run with sf=3 (no refusal) and recommends sf=1, which WRF rejects
  (`physics_init.F:3744` "use myjsfc scheme for this pbl option"), so **no GFS pairing in the port is WRF-faithful**. Fix:
  either refuse bl=3 until a GFS-sfclay (sf=3) handoff exists, or build the sf=3 handoff (psim/psih/gz1oz0/br/wspd from the
  paid `gfs_sfclay_adapter`) + `urban_pbl_forcing` staging + rates branch; and refuse bl=3 with sf=1 (WRF-fatal). Oracle:
  pristine bl=3/sf=3 one-step dump of RTHBLTEN/RQVBLTEN/RQCBLTEN/UST.
- **GFS-A2 (HIGH) QC/QI not mixed; RQCBLTEN/RQIBLTEN dropped** (scan_adapters.py:1821 zeros, :1830 tend dict) vs
  bl_gfs.F:366-400/451-458 + addtendc.F:673-685; also changes MONINP's moist terms. Both branches. Fix: pass qc/qi, return
  rates, add to `pbl_moist_raw` (the sf1 branch loop :9533 already consumes 'qc'/'qi' keys).
- GFS-B1 (MED) UST/WSPD writeback missing (bl_gfs.F:464-466 vs adapter discarding `out["ust"]`); next-step surface layer
  and history UST differ. GFS-B2 (MED) TSK pre-LSM (surface_handoff.py:9) vs TSKOLD post-LSM (pbl_driver.F:1004); TSEA only
  enters THESV, low impact [I].

## 3. ACM2 (bl=7, sf=1 forced)

(a) `operational_mode.py:9517-9539` → `scan_adapters.py:1699-1716` via `_pbl_surface_island` (:2250-2262) with
`surface_forcing=sf1_surface_forcing`; staging `urban_pbl_forcing` (`coupling/urban_pbl_hook.py:6-26`). WRF
`pbl_driver.F:1584-1606` `CALL ACMPBL(...)`; coupling addtendc.F:449-503. XTIME=itimestep fix present (:9522-9524, :9543).

(b)
| item | WRF | port | verdict |
|---|---|---|---|
| TH/T/RHO/DZ | dry th_phy, t_phy, grid%rho, dz8w | dry θ, θΠ, `_wrf_phy_prep_rho_from_state`, ph/9.81 (urban_pbl_hook.py:13-24) | MATCH |
| UST/HFX/QFX/WSPD/PBLH(in) | grid (post-LSM), sfclay WSPD, previous PBLH | state.ustar/hfx/qfx, sf1 wspd, state.pblh (urban_pbl_hook.py:22-25; scan_adapters.py:1552) | MATCH |
| MUT, XTIME | grid%mut, itimestep | state.mu_total, 1+rint(lead/dt) | MATCH |
| QC3D/QI3D | mixed as ACM species 3/4 (module_bl_acm.F:1003-1004 VCI), cloud test in EDDYX `(QCS+QIS)>0.01e-3` (:766-767) | `acm2_columns` called without qc/qi ⇒ zeros (scan_adapters.py:1704-1708; pbl_acm2.py:1190-1193) | MISMATCH (A+B) |
| RQCBLTEN/RQIBLTEN | returned (module_bl_acm.F:554-555), applied addtendc.F:473-485 | traceable kernel returns only u/v/theta/qv (pbl_acm2.py:1207-1212) ⇒ dropped at operational_mode.py:9533 | MISMATCH (A) |
| PSFC/TSK/PSIM/GZ1OZ0/RMOL | passed, unused by ACM2D except MUT | not passed | MATCH (no effect) |
| EXCH_H/REGIME/KPBL | written | only pblh written; exch_h only for CAM-MG | MISMATCH (C) |

(c) **ACM2-A1 (HIGH)**: port `scan_adapters.py:1704` `out = acm2_columns(f["u_cols"], ..., f["qv_cols"], f["rho_cols"], ...)`
(no `qc=`, `qi=`) vs WRF `ACMPBL(..., QC3D=qc_curr, QI3D=qi_curr, ..., RQCBLTEN=rqcblten, RQIBLTEN=rqiblten)`
(pbl_driver.F:1595,1603) and `add_a2a(moist_tendf(P_QC),RQCBLTEN)` / `(P_QI),RQIBLTEN` (addtendc.F:473-485). Cloud water/ice
are never PBL-mixed and the in-cloud eddy-diffusivity branch of EDDYX never triggers (cloud test sees zeros) ⇒ different Kz in
cloud-topped BLs → plausible contributor to the c05/c06 rain deviation [I]. The numpy reference already carries qc/qi
(pbl_acm2.py:542-543, 623, 667-668). Fix: pass `qc=f['qc_cols'], qi=cols(state.qi)`, return `qc`/`qi` rates from
`_acm2_column_traceable`; the sf1 branch then couples them. Oracle: pristine ACMPBL column (or 1-step WRF) on a cloudy
Swiss column; deletion test drops the qc/qi keys.

## 4. BouLac (bl=8, sf=1 forced; BEP path out of scope)

(a) `operational_mode.py:9517-9539` → `scan_adapters.py:1719-1769` (islanded). WRF `pbl_driver.F:2001-2020`; addtendc.F:860-901.

(b)
| item | WRF | port | verdict |
|---|---|---|---|
| TH_PHY/RHO/DZ8W/U/V | dry, grid%rho, dz8w, u_phytmp | dry θ, prep rho, ph/9.81, mass u/v (urban_pbl_hook.py) | MATCH |
| QV/QC, HFX/QFX/UST | qv_curr, qc_curr, grid post-LSM | state.qv, state.qc, state.hfx/qfx/ustar | MATCH |
| DT | DT=dt (not dtbl) | dt | MATCH |
| outputs | RTH/RQV/RQC/RU/RVBLTEN; RQC applied when adv_moist_cond (addtendc.F:894-900) | `out` has qc ⇒ `pbl_moist_raw['qc']` (:9533-9536) | MATCH |
| TKE | TKE_PBL state, init 1e-4 (module_bl_boulac.F:944) | qke leaf holds BouLac TKE; cold start seeds MYNN equilibrium QKE (d02_replay.py:2537-2550) | MISMATCH (B) |
| PBLH | written | written (`_pbl_diagnostic_state`) | MATCH |
| EXCH_H/EL_PBL/WU..WQ | written | only for CAM-MG | MISMATCH (C) |

(c) **BouLac-B1 (MED, port side HIGH, nested-init path [I])**: initial TKE = MYNN `mym_initialize` equilibrium QKE
(O(0.1-10) m²/s² in unstable layers) instead of WRF 1e-4 ⇒ first-hours mixing too strong; QKE is the q² convention while
BouLac stores TKE in the same leaf (myj/boulac docstrings) — the seed is also off by the factor 2. Fix: exempt bl=8 (and bl=2,
TKE_MYJ=EPSQ2) from the MYNN seed as GBM already is (d02_replay.py:2537-2544). No class A found for BouLac.

## 5. MRF (bl=99, sf=1 forced)

(a) `operational_mode.py:9517-9539` → `scan_adapters.py:1772-1801`. WRF `pbl_driver.F:1351-1382`; addtendc.F:394-447.
(b) Staging/forcing via `urban_pbl_forcing` + sf1 handoff (raw PSIM/PSIH, GZ1OZ0, BR, WSPD, hydrostatic PSFC/P): MATCH.
QC3D/QI3D: WRF mixes both (FLAG_QI, pbl_driver.F:1361-1363, 1368-1369) and returns RQCBLTEN/RQIBLTEN (applied
addtendc.F:418-430); port passes `jnp.zeros_like(f["qv_cols"])` as qc (scan_adapters.py:1788), no qi, `tend` keeps
u/v/theta/qv (:1797-1798) although `mrf_column` returns `rqcblten` (bl_mrf.py:485) — MISMATCH (A). The zero qc also disables
the IMVDIF in-cloud branch (bl_mrf.py:297). TSK: pre-LSM `skin_before_lsm` (surface_handoff.py:9) vs TSKOLD post-LSM
(pbl_driver.F:1004), used in THGB/TGDSA (bl_mrf.F:492-496, 548, 766) — MISMATCH (B).
(c) **MRF-A1 (HIGH)** qc/qi not mixed, RQCBLTEN/RQIBLTEN dropped (fix as ACM2-A1). **MRF-B1 (MED)** TSKOLD timing: pass
post-LSM `state.t_skin` (same for GFS on the sf1 branch); oracle: pristine MRF column with TSK≠TSKOLD.

## 6. Cross-cutting

- X-B1 (MED): `bldt>0` only warns (`io/namelist_check.py:675-690`); port runs PBL every step with DT=dt while WRF uses
  DTBL=dt·STEPBL and holds RxBLTEN between calls (pbl_driver.F:960-990). Affects all five schemes when bldt≠0.
- X-C1: EXCH_H/EL_PBL/KPBL/TKE_PBL history for non-MYNN PBLs only partially produced (see tables).
- Verified present (not re-reported): ACM2 XTIME=itimestep (operational_mode.py:9523); SF1 raw PSI+GZ1OZ0 for MRF/GFS
  (surface_handoff.py:5-8).
- UNVERIFIED: spec-zone ownership of the post-RK Euler increments (MYJ/GFS legacy) vs add_a2a/add_a2c ring handling;
  whether `State.qke` is advected for MYJ/BouLac (bl_mynn_tkeadvect path) — not traced.

## Priority (forecast impact)
1. MYJ-A1/A2/A3 (all MYJ cu≠7 runs; theta_m input + dry-form Euler increment + qc latent heat without qc update).
2. GFS-A1/A2 (no WRF-faithful GFS pairing at all).
3. ACM2-A1, MRF-A1 (cloud species never PBL-mixed; cloud-Kz branch dead) — rain-relevant for ACM2 composites.
4. MYJ-B1/B2 (surface coupling from a private seeded Janjic re-solve, wrong PSFC), GFS-B1/B2, MRF-B1, BouLac-B1, MYJ-B3/B4.
