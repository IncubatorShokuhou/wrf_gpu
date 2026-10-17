# t-cfix-cu qualification packet — caller-audit cumulus group (F7, F9, F10, F11, F14, F18)

Lane t-cfix-cu (Opus 5.5), branch lane/t-cfix-cu, rebased on main a366702b5c (product head abac2ce26 + receipts/notes); rv-t-cfix MERGE 03:07Z on bf2e8c36ae. Audit: <USER_HOME>/wrf_gpu2_lanes/t-calleraudit/CALLER_AUDIT.md
(+ parts/cu_a.md, cu_b.md). WRF = <USER_HOME>/src/wrf_pristine/WRF (4.7.1). Evidence [M CPU] + the F7 c05 pristine oracle on the RTX 5090 release path (+ an earlier wz1080 legacy debug arm);
GPU composites on the RTX 5090 pending (list at the end).

## Commits
| commit | finding | content |
|---|---|---|
| 47f7b683e | F10 (A) | GF cu=3 adapter default `ishallow_g3=0` = WRF Registry default (honour check refuses others) |
| f7bd3e60d | F11 (A) | KF warm_rain/F_QR/F_QI/F_QS from the MP moist array + KF_eta_CPS output gate; legacy fp64 KF feedback branches ported; G3/GD/GF/Tiedtke/NTiedtke qc/qi species gating |
| ed95bfea3 | F7 (A), F9 (A), F14 (B), F18 (B + GF forcing A) | held WRF RTHFTEN/RQVFTEN carry + restart schema; G3/GD/GF/Tiedtke/NTiedtke WRF forcing; cumulus + Euler-PBL rates as RK sources; GF PSUR/omega/KPBL/lateral band; Tiedtke ZNU; cudt>0 refusals |
| c3478e6cc | tests | KF cadence reads `getattr(namelist, 'mp_physics', None)` (harness namelists); MYJ fake takes the F6 kwargs; Grell wiring fixture = mass-consistent deep soundings (old fixture: flat metrics, p_hyd surface 55 kPa) |
| e83d97351 | F18 | GF PSUR read from the shared phy_prep `psfc` (same hydrostatic helper; no second solve) |
| 0aea32eb9 | F7 (r1 closed) | legacy (non-driver) radiation: G3/GD/GF/NTiedtke read THIS step's RTHRATEN — the YSU early refresh/hold on the step-entry state (4bc9325fd) now also runs for cu in (5,93,3,16); the radiation slot reuses it (one SW+LW solve) |
| f64e6137e | F18 (r3, c04) | BL_GFS KPBL2D rides with the GFS RK rates (`kpbl`); the GF slot passes it as grid%kpbl |
| aaee4d0f1 | F10 | GF kernel with ishallow=0 skips CUP_gf_sh like WRF (:504): no outqcs leak into RQCCUTEN, XMB/K22/KBCON/KTOP_SHALLOW = 0 (the v060 tolerance hid both); less work on the default path |
| 054a23a5f | tests | GF operational smoke on 12x12 at 9 km: 4x4 has no WRF interior column, and at 3 km a saturated cloud base gives WRF's own scale-aware ierr=231 (gf_deep.F:462-471, :663) |
| fb088b13c | F7 (r5 closed) | CPU-WRF restart hydration copies the held RTHFTEN/RQVFTEN ('r' stream) for cu 3/5/6/16/93; refuses unseeded/inactive targets |
| 868d6877d | compose | t-cfix-phyprep test: GF staging strip runs periodic (WRF lateral band leaves no interior column on a 1-row grid) |
| cdcd77b8e | compose | t-calleraudit F2 land-RAINCV test runs both rk modes: F9 adapters return (rates, RAINCV); OM `_stateless_cu` feeds RK sources + `convective_amount` |
| 5df2145e9 | F7/F14 (critic FIX 2) | Tiedtke cu=6: RQVFTEN only — no RTHFTEN theta capture, no dry conversion (CU_RTHFTEN_SCHEMES gate; module_em.F:654-665, BSU:5291-5331) |
| 96fc5d357 | critic items | GF/KF oracle build scripts resolve the repo root (32 savepoints rebuilt BYTE); GF test/docs: outputs masked, kernel runs on all columns; packet perf [I]/[U], KPBL fallback, c25 retired |
| cb3bbf05d | compose | t-cfix-mynn F20 GF flux-operand harness runs periodic (WRF lateral band masks the small grid) |
| fd18b5e54 | tests | r3 sentinel KPBL injected into whichever GFS producer runs (plain kernel or t-cfix-sfc BL_GFS REAL wrapper); asserts one ran |
| abac2ce26 | F18 r3 | gfs_pbl_adapter publishes `kpbl` only when the kernel output carries it (both real kernels do; test stubs did not) |

## WRF semantics implemented (file:line on the WRF side)
- set_tend RTHFTEN = t_tend*msfty (theta advection only), module_em.F:654-665; RQVFTEN = advect_tend(qv)*msfty,
  :1365-1376; last RK stage wins. Port: captured in `advance_stage` at `rk_step == rk_order` (theta: the same
  `advect_scalar_flux` call as `_augment_large_step_tendencies`; qv: the stage's root/nested advect_tend), x msfty.
- phy_prep_part2 (solve_em.F:3669, after advance_ppt, before MP, mut = grid%muts): decouple both, then
  use_theta_m dry conversion (BSU:5293-5331). Port: `coupling.cu_forcing.finish_cu_forcing`, called after
  `_apply_physics_non_dry_updates` / root flow_dep and before microphysics.
- cumulus_driver G3/NTiedtke (:866-883) `(RTHFTEN+RTHRATEN+RTHBLTEN)*pi`, `RQVFTEN+RQVBLTEN`; GRELLDRV/GFDRV fold the
  same sum (gd.F:221-222, gf_wrfdrv:412-417); Tiedtke QVFTEN/QVPBLTEN; both Tiedtkes zero forcing at itimestep 1.
- calculate_phy_tend (module_em.F:2249-2300) couples every R*CUTEN with time-n mut; phy_cu_ten field sets
  (addtendc.F:1125-1618); advance_ppt RAINC += PRATEC*DT and KFSCHEME NCA clear (:2286-2322); phy_prep_part2
  decouples held R*CUTEN incl. RU/RVCUTEN (BSU:5105-5170).
- GFDRV caller: PSUR = p8w(1)*.01 (:385), omeg = -g*rho*w on staggered w (:471), grid%kpbl (P1:1397),
  ibegc/jbegc = ids+4..ide-5 unless periodic (:264-277).

## Independent oracles (E98) and caller tests
| test file | what | result |
|---|---|---|
| tests/test_t_cfix_cu_gf_ishallow.py | pristine GFDRV column oracle (v060 driver, argv ISHALLOW and WRF-forcing mode; ishallow=1/mode 0 reproduce committed v060 savepoints byte-for-byte, E83); port at v060 tolerances; old default and dropped forcing caught; exact-zero shallow terms at ishallow=0; operational spy | 7 PASS |
| tests/test_t_cfix_cu_kf_species.py | pristine KF_ETA_CPS column oracle with Kessler / no-ice flag modes (p0_4 driver; mode 0 byte-reproduces p0_4); legacy fp64 + native fp32 kernels; mixed-phase call caught; operational cadence flags | 11 PASS |
| tests/test_t_cfix_cu_g3_forcing.py | REAL4 literal of the pristine phy_prep_part2 statements (source-text guarded) <=1 ulp; capture spies; finish once with final mass; slot passes held pair + RTHRATEN; Euler-PBL increment reaches G3 as rate; E78 refusal; legacy radiation reaches G3/GF/NTiedtke this step from the entry state (r1) | 10 PASS |
| tests/test_t_cfix_cu_cu_rk.py | REAL4 literal of calculate_phy_tend / KFSCHEME advance_ppt / phy_prep_part2 (couple byte, decouple <=1 ulp); every stateless scheme's phy_cu_ten field set incl. momentum, no Euler add; held SAS couple/feed/decouple; KF99 clear after consumption; MYJ/GFS PBL rates published | 16 PASS |
| tests/test_t_cfix_cu_tiedtke_forcing.py | Tiedtke held RQVFTEN + step-1 zero; NTiedtke THFTEN/QVFTEN; cudt refusals; capture x msfty; ZNU half levels; cu6 RK never traces the RTHFTEN capture (cu16 control does); cu6 finish keeps the held RTHFTEN | 14 PASS |
| tests/test_t_cfix_cu_gf_caller.py | operational GF seam: forcing, PSUR=p_hyd_w(1), staggered w, PBLH KPBL; GFS-published KPBL2D reaches GFDRV (r3); outputs masked to WRF's interior band (the batched kernel still runs on all columns); columns | 3 PASS |
| tests/test_cu_forcing_restart_schema.py | validator branches; byte-exact round trip x 4 formats x schemes x dtypes; E78 refusals; 20 wiring mutants; CPU-WRF wrfrst hydration of the held pair (r5) | 109 PASS |
| proofs/t_cfix_cu/f7_c05/ | END-TO-END: private pristine serial wrf.exe (dump-path literals only) on real c05, history every 18 s step with iofields RTHFTEN/RQVFTEN vs the port's held carry after steps 1..3. FINAL = release path on the RTX 5090 (c05 program sha 481eb074 == final tree); earlier wz1080 legacy arm kept as history | release: step-1 interior rel RMS RTHFTEN .00072 / RQVFTEN .00046 (<= .10 PASS); no-dry mutant 59.1x, zero mutants 1387x/2196x (>= 3x PASS) -> pre-registered gate PASS. Legacy arm: .016/.029, no-dry 2.88x (MISS; legacy dynamics, see attribution). E41 for the arm UNKNOWN (tool invocation error, cache deleted) |

## CPU suites (6G RAM slots, one file/group per process; logs <USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/suite{3,4,5}/)
- suite3 (pre-rebase product e0212cbe4): 48 groups — this lane's tests, every cumulus/PBL/SAS/KF/Grell/Tiedtke oracle +
  wiring file, dispatch/catalog/contracts/CLI/namelist/restart, test_v013 smoke all 93 ids (per test where needed),
  phyprep/pblmix/rainnc/F2 bundle, YSU oracle, GFS PBL: all PASS. mp38/mp56 smoke need 12G (pre-existing footprint,
  also seen by t-calleraudit; cu=0): PASS at 12G.
- suite4 (composed with main dca7a4d79) and suite5 (composed with main 9a6faf304): this lane's tests, t-calleraudit
  land-precip/F24 + noahmp precipitation, restart/hydration, phyprep (22), pblmix, Grell wiring (main's mass-consistent
  fixture), cumulus + Tiedtke smoke, PBL smoke 1-5, SAS94/native SAS/KF99/KF RK, NSSL cumulus caller, class-D MP rho,
  GFS PBL, YSU oracle, CAM-MG DLF2/ice diagnostics, dispatch/catalog/contracts: all PASS.

- suite6 (cu6 fix 54cdee223): Tiedtke/NTiedtke oracles + JAX parity, cumulus/Tiedtke smoke, land precip, restart schema,
  cu_rk, GF caller, G3 forcing, Tiedtke forcing (14), phyprep: all PASS.
- suite7 (final, composed with main c360fdf6e incl. F20/F17): this lane's tests, restart/hydration, land precip/F24,
  GBM RTHRATEN + F23 + YSU oracle (shared early-refresh condition), pblmix condensate + MYNN numbers, F20 MYNN caller (29,
  after the periodic harness fix), phyprep + Grell wiring, cumulus/Tiedtke smoke, Tiedtke oracles, dispatch/catalog/
  contracts/RAINNC, PBL smoke 1-3: all PASS.

## Release identity
Official probe `proofs/s1_port/release_identity.py` via the no-assert lane wrapper `proofs/t_cfix_cu/identity/identity_any.py`
(`identity2.json`), CPU traces of the native facade:
- FINAL head abac2ce26 on main a366702b5c: Swiss 7379c7d77e94... / PROD d01 6487f5b039a0... (== main's own post-F20 defaults, unchanged through batch53/54) and c05 release program 481eb074d699... (== the GPU-arm snapshot).
- main c360fdf6e (post-F20 defaults) vs this branch rebased on it: Swiss 7379c7d77e94... == 7379c7d77e94..., PROD d01
  6487f5b039a0... == 6487f5b039a0... -> default programs EXACT vs main's own.
- Before F20 (main 46868408b / 9a6faf304): Swiss f672a573 / PROD 6fc715a2 EXACT on e0212cbe4, beab71d9c and 54cdee223.
- c05 (G3) release program: 54cdee223 (GPU-arm snapshot) == final tree, sha 481eb074d699... (binds the F7 GPU result).

## Mutants (E202)
proofs/t_cfix_cu/mutants.py on a detached snapshot of the final abac2ce26 (rebased on main a366702b5c): 24/24 KILLED (also 24/24 on f3ea0c0f3) (also 24/24 on 08ea2ba92; 22/22 earlier on beab71d9c and 19a585cbb), each by an assertion of its targeted test
(rc 1, no setup/collection errors) -> proofs/t_cfix_cu/mutants_result.json. F10-default and F11 deletion checks live inside
their oracle test files (ishallow=1 kernel fails the ishallow=0 oracle; mixed-phase KF call caught).

## Residuals (disclosed, not hidden)
r1 CLOSED by 77604584f (was: legacy-radiation RTHRATEN one radt stale for G3/GD/GF/NTiedtke on radiation steps).
r2 Tiedtke/NTiedtke/GF HFX/QFX still from kinematic surface handles (audit B7, small).
r3 GF KPBL: the GFS PBL (c04) now publishes its KPBL2D with its RK rates and GF reads it (WRF grid%kpbl);
   other PBLs, and the GFS Euler mode (rad_rk_tendf=0 / PHYS_TEND_RK_WRF off), keep the MYNN GET_PBLH rule on
   that PBL's PBLH as a fallback (exact for MYNN only). No universal exact-KPBL claim. G3's own Ri-derived KPBL
   remains but is consumed only by ishallow=1 (refused; g3.F:1238), so it is inert on the admitted path.
r4 physics_registry CUMULUS_CARRY_MEMBERS does not list cu_forcing (frozen documentation table).
r5 CLOSED by 5fb96db90: CPU-WRF restart hydration copies the held RTHFTEN/RQVFTEN ('r' stream) for cu 3/5/6/16/93 and refuses unseeded/inactive targets.
r6 F4/F15/F16 staging in GF/BMJ/Tiedtke/NTiedtke is t-cfix-phyprep's (cd518dc81, on main 46868408b); this branch is
   rebased on it (Tiedtke takes phyprep's p/p8w + this lane's ZNU; GF PSUR = phyprep psfc).

## Performance / memory (AGENTS §4)
Default programs: unchanged (identity EXACT), no added bytes or work. Opt-in schemes only ([I] = inferred from source,
[U] = unmeasured):
- [I] +2 REAL (nz,ny,nx) carry leaves RTHFTEN/RQVFTEN for cu 3/5/6/16/93 = 8*nz*ny*nx bytes (2,956,800 B on PROD d01
  shape if such a scheme is selected).
- [I] cu 3/5/16/93: one extra final-RK-stage theta flux-advection evaluation per step (same inputs as the dycore's
  call; whether XLA CSEs it on the GPU executable is [U]); cu 6 no longer traces it (54cdee223). Small elementwise
  decouple/dry conversion after the RK.
- [I] F9 replaces the Euler state write by coupled scalar/momentum sources folded into existing t/moist/ru/rv_tendf
  lanes, plus coupled held intermediates for SAS/KF99; launch count and peak/temp memory are [U].
- GF: the batched kernel evaluates every column; the lateral band only masks outputs (no boundary work saved).
- [U] changed-option GPU wall, launch count, peak/temp allocation and readiness: not measured; no speed or
  negligible-cost claim is made. No host transfers added.

## GPU reruns needed (RTX 5090, release defaults, D6 vs original CPU-WRF; v-prep queue)
| case | config touched | findings exercised |
|---|---|---|
| c02 | KF cu1 + Kessler mp1 | F11 (warm_rain / F_QR fold) |
| c03 | BMJ cu2 + MYJ bl2 + Lin mp2 | F9 (BMJ + MYJ rates RK-coupled), F11 (Lin no-ice flags n/a: has qi) |
| c04 | GF cu3 + GFS bl3/sf3 + Dudhia sw1 + mp3 | F7/F18 GF forcing, F10 ishallow=0 + shallow-term fix, F18 PSUR/w/KPBL(r3)/band, F7 r1 legacy RTHRATEN, F9, F11 (WSM3) |
| c05 | G3 cu5 + ACM2 bl7 | F7 (held forcing, end-to-end oracle case), F9 (Euler-PBL increment as G3 rate) |
| c06 | Tiedtke cu6 + BouLac bl8 | F14 (held RQVFTEN, step-1 zero, ZNU, no RTHFTEN capture), F9 |
| c07 | NTiedtke cu16 + Shin-Hong bl11 | F14 (THFTEN/QVFTEN), F9 |
| c08 | KSAS cu14 + GBM bl12 | F9 held SAS RK |
| c09 | cu94 + MRF bl99 | F9 held SAS + generic PBL RK |
| c10/c11 | cu95/96 + EEPS/KEPS | F9 held SAS RK |
| c13 / c17 | G3 cu5 / GD cu93 + MYNN | F7 held forcing (MYNN source-leaf path) |
| c14 | KF99 cu99 | F9 KF99 advance_ppt clear |
| (c25) | GFS PBL bl3, cu0, sf91 | retire: t-cfix-sfc now refuses sf91+bl3 pre-JAX (WRF isfc fatal); bl3 RK path is covered by c04 |
