# w43-conv — convection-active operators vs pristine WRF 4.7.1 (#148 residual, #149 heavy-cell deficit)

Lane w43-conv (Opus 5.5), branch lane/w43-conv. Summary JSON: `analysis/CONV_ORACLES.json`. Per-experiment rows: EXPERIMENTS.tsv
CT02–CW02. Status: **CPU arms complete (critic DELTA2 MERGE 692b0833); GPU arm CG01–08 ran 2026-10-21 11:35:23–11:40:38Z real on the asserted RTX 5090: ALL 8 arms PASS by numeric verdict with full coverage (section GPU CG01–08)**.

## Question and method

Is there an operator that differs from WRF only in ACTIVE convection (heavy rain, strong updrafts, cold pools)? For each
operator I ran a **same-input one-call oracle**: identical REAL inputs go to the pristine WRF 4.7.1 routine (verbatim
source, REAL4 build, caller wiring from solve_em / module_em / first_rk_step_part2 / module_pbl_driver / surface_driver)
and to the **release-default** port (`apply_fast_path_defaults()`, interpret Pallas on CPU). There are no trajectories, so
chaos plays no part. Gates are the frozen ones where they exist. Where none existed, the gate was pre-registered in the
script docstring before the run. Every oracle carries a vacuity census, and mutants are counted only on genuine gate
failures (E202).

**Inputs (real, convective).** These are 11 CPU-WRF reference history frames of #148 (9-nest 9/3/1 km, issue
20261018_12z), read via the sealed GuardedRead facade under A-148 / ADD1 (manager OK; receipts `analysis/receipts/R01_nml`,
`R02_frames`, 16/16 ALLOW, 0 refused):
- d06 (1 km) at τ22, 24, 25, 29, 34;
- d04 (1 km) at τ58, 62, 66;
- d02 (3 km) at τ38, 39, 55.

τ is hours from 2026-10-18_18Z. The frames were picked where column QRAIN exceeds 3 g/kg and |W| is 10–23 m/s. Namelist:
mp 8, MYNN 5/5, Noah-MP, w_damping 1, diff_opt 1 / km_opt 4, diff_6th_opt 2 (0.12), damp_opt 3, epssm 0.5, dt 18/6 s.

## Results (all [M], CPU)

| id | operator (release path) | pristine reference | active-input census | result | mutants |
|---|---|---|---|---|---|
| CT02–03 | Thompson mp8 full call (`phys_thompson_full.full_column`: sources/sinks, sedimentation with adaptive nstep, rain number, riming, melting) | `mp_gt_driver` column driver (TH08/LL01) | 586 cols: core qr ≥ 2 g/kg (max 5.03; 112 cols ≥ 3), graupel (max 4.73; 68 ≥ 2), updraft (W to 23.4), downdraft, snow; 170 cols with WRF precip ≥ 0.1 mm/step | **0/5860 field fails** (frozen mixed-phase increment gate, TOL 1e-2; scorer now rejects non-finite inputs/references first, E200): worst qr 3.1e-3, qs 2.8e-3, qg 2.5e-3, Nr 8e-4. Surface precip sum WRF 52.01 = port 52.01 mm; column-integrated dqr/dqg/dθ equal to 4 digits for every kind | mp_off **2626 fails**, nstep1 **381** (qr/Nr/qg/precip). const_off: 0 fails (WRF REAL constant set below gate resolution, E169) |
| CM01 | MYNN PBL (`step_mynn_pbl_column_with_pblh`, F20 operands) on the WRF caller operands P = p_hyd, PI = pi_phy (dynamic), RHO from the hypso2 ALT | public `mynnedmf_driver` (F20 caller, unmodified objects) | 23–48 cols/frame: cold pool (θ₁ anomaly to −4.9 K, rain at the surface), core, downdraft, env; QKE ≤ 10.2; HFX −77…465 W/m² | **PASS frozen 1e-3** on rθ/rqv/ru/rv/rqc/rqi in every frame (delta re-run). Max \|Δrθ\| = 2.0e-5 K/s (≈ 4 float32 ulp(θ)/dt). Normalized ≤ 1.8 % (θ). PBLH ≤ 0.005 m. Warm-state operands are zero, see OPEN | (F20 mutants already exist; not repeated) |
| CS01–02 | MYNN surface layer (`surface_layer_with_diagnostics`) → U10/V10/T2, caller P = p_hyd, hypso2 RHO | `SFCLAY1D_mynn` (proofs/v090, B39b constants), warm step | 43–80 cols/frame incl. **gust fronts (top-2 % \|V10\|)**, cold pools, cores, downdrafts; all stability regimes | **0 fails** (frozen v090 THR, delta re-run): u10/v10 rel ≤ 4.4e-6, t2 2.1e-7; worst hfx 2.8e-3 (thr 0.02), br 1.5e-3 | nopblh **555 fails**. mol0 survives (MOL only seeds the converging z/L iteration) |
| CD01 | sixth-order filter diff_6th_opt 2 (native `sixth_order_fp32`) | `sixth_order_diffusion` (rk_tendency args) | whole frames; monotonic limiter active in 63–81 % of u/v/w/θ cells, 41–45 % of qv, 2–20 % of qr | **88/88 PASS** on 4 frames, then **242/242 on all 11 frames** (CD01_11), ≤ 2.6e-7 of max tendency | (limiter on vs off differ in the pristine itself) |
| CA01 | PD scalar advection moist_adv_opt 1 (native `advect_scalar_pd_fp32`, species loop) | `advect_scalar_pd` (+ `calc_ww_cp` for rom) | 8 species; limiter-active cells 65–18,124 per species (qr 4,056–10,235) | **32/32 PASS** on 4 frames; **all 11 frames PASS** (CA01_11): errors ≥ 7× under the frozen BC42 bound, limiter cells 7–27,940. d04 τ62 has NO cloud ice (QICE ≡ QNICE ≡ 0): reported as ABSENT, both sides exactly 0, not counted as coverage. Every species is covered on ≥ 10 frames | (limiter-active requirement in the gate) |
| CK01–02 | km_opt 4 Kh chain (`wrf_nonperiodic_diffusion_metrics` → `horizontal_deformation_2d` → `smag2d_horizontal_km`) | **NEW**: `compute_diff_metrics` + `cal_deform_and_div` + `smag2d_km` verbatim (nested config) | Kh up to 3,480 m²/s (3 km) / 1,510 (1 km), steep Canary terrain | **55/55 PASS**, ~1e-6 rel (pre-registered 2e-4/2e-5) | nozx (no terrain-slope term) **killed**, errors O(field) |
| CD02 | diff_opt 1 horizontal diffusion with convective Kh | `horizontal_diffusion` / `_3dmp` | u v w θ qv qr qg Nr, pristine Kh from CK01 | **32/32 PASS** on 4 frames; **all 11 frames PASS** (CD02_11), ≤ 2.5e-7 rel | — |
| CW01–02 | acoustic small step: calc_coef_w, advance_uv, advance_mu_t, advance_w **incl. damp_opt 3**, calc_p_rho (native fp32 kernels, fast defaults) | module_small_step_em routines (b-core oracle), nested, Tier-P with compiler envelope | d06 1 km tile τ24→τ25, \|W\| to 19.5 m/s (coupled w 1.05e5), WRF trips 1/2/4 | **3/3 stage arms PASS**; worst field 16 % of the Tier-P bound, 0 guard events | epssm 0.1 **killed** (up to 1.6e4× bound) |
| — | w_damping 1 (`w_damp_rw_tend_wrf`, BD85 bitwise oracle already exists) | — | vertical CFL ≤ 0.57 in all 11 frames (peak near the surface over terrain, not in updraft cores) | **inactive on the sampled hourly states** (WRF acts only for CFL > 1; critic: pristine calc_ww_cp CFL ≤ 0.524, 0 active faces). Activation at intervening stages is not excluded | — |

## Interpretation

- No convection-active operator mismatch was localized. Every operator in the contract's hypothesis class reproduces
  pristine WRF on the supplied same-input operands (callee check; warm MYNN state OPEN) at float32 rounding level, or far inside the frozen bounds, on real heavy-rain / updraft / cold-pool
  inputs. The class covers Thompson sedimentation, rain evaporation, riming and rain number; MYNN and MYNN-SL in cold pools
  and gust fronts; the 6th-order filter; the km_opt 4 Kh and horizontal diffusion; PD moisture advection; and the acoustic
  step at large W, including damp_opt 3. The Thompson aggregates show no sign of a deficit in surface precipitation or
  column rain (52.01 = 52.01 mm, dqr 15.99 = 15.99).
- Hence the #149 heavy-cell −12…−15 % (48/72 h) and the #148 d06 −15 % / d04 U10 τ59 residual are not explained by a
  one-call deviation of these operators. This is consistent with the #149 CPU-vs-CPU twin: C2 (4×1 decomposition) vs C
  RAINNC RMSE was 4.20 vs 4.00 mm at 72 h for B vs C, and domain means were NOT_SYSTEMATIC. Remaining explanations are
  step-level coupling/order effects or chaotic realization; neither is proved.

## Limitations (stated, not hidden)

1. **One call, same input.** The oracles do not cover caller-order/coupling across the RK step. Exceptions are the caller
   wiring each oracle reproduces (F20 MYNN caller, rk_tendency arguments for sixth/PD/diffusion, the build_savepoints stage
   recipe for acoustic). Not covered: rk_addtend / rk_update_scalar coupling, the h_diabatic pair, KF (d01 only),
   radiation in cloudy convective columns, Noah-MP under heavy rain, nest boundary forcing, and multi-step trajectories.
2. **CPU interpret Pallas** for CT/CM/CS/CD/CA/CK/CW. The Triton-compiled GPU kernels (E17/E84) were re-checked in GPU
   arm CG01–08 on the same inputs and references: all PASS (section GPU CG01–08).
3. **Inputs are history (end-of-step) states.** Derived operands: rom from the pristine calc_ww_cp; ru/rv from
   couple_momentum; the acoustic stage increment is the hourly difference scaled to dt (b-core recipe); MYNN WSPD = |V₁|,
   ZNT 0.1, QSFC = qv₁; MYNN-SL mavail 1 with MOL from a pristine pre-pass. All of these are identical on both sides.
4. **Thompson tables.** The pristine driver computes its tables with thompson_init, because the oracle_run .dat links are
   dangling (E160), while the port uses tables extracted from CPU-WRF's .dat files. The match therefore also cross-checks
   the tables.
5. **Gate resolution.** The WRF REAL Thompson constant set (const_off) and the MYNN-SL MOL seed (mol0) are below gate
   resolution; both are disclosed.

## Delta 1 (critic cr-w43-conv VERDICT FIX, 2026-10-20)

1. **CT03 scorer, E200.** `analysis/thompson/score.py` now validates every input (`in_*`, `dt`, geometry) and every
   reference (`wrf_*`) for finiteness BEFORE any field-wide scale is formed, like the frozen `_validate_fixture`. A
   non-finite value makes the fixture INVALID (FAIL, rc 2), and a non-finite candidate value (any field or any precip
   component) makes that column's excess infinite. Controls: a single NaN/+Inf/−Inf in reference precip, input qr,
   candidate qr or candidate precip gives **12/12 FAIL** (`CT03b_finite_controls.json`). The existing finite outputs were
   re-scored with no numerical rerun and tolerances unchanged: CT03 still 0 fails; CT04 mp_off 2626, nstep1 381,
   const_off 0.
2. **MYNN / MYNN-SL caller operands.** `analysis/mynn/caller_ops.py` provides P = history `P_HYD` (WRF
   `module_first_rk_step_part1.F:637/1380` passes grid%p_hyd), PI = pi_phy from P+PB (dynamic, so t3d is dynamic) and
   RHO = (1+qv)/alt (phy_prep :4856). ALT now uses the hypsometric_opt=2 log form of `module_big_step_utilities_em.F:1045–1048`
   (Registry default, namelist unset); the earlier builder used a hypso1 difference. The changes are 70 Pa in p and
   0.064 % in ρ on d06 τ24, matching the critic's 67.9/78.4 Pa and 0.064 %. CM01/CS01/CS02 were re-run with unchanged
   frozen bounds: all PASS, nopblh still killed (555), mol0 still survives. MYNN/MYNN-SL scorers now also reject
   non-finite references and inputs (E200).
3. **Same-input scope.** These oracles validate the callee on the supplied operands. They do not validate the
   production adapter's operand construction.

## Delta 2 (critic DELTA1 verdict: GPU-artifact items)

- `analysis/gpu/gpu_verdicts.py` requires EXPLICIT coverage per arm: 586 scored Thompson columns; all 11 expected frames
  present and passing with every expected field/species (sixth 22, PD 8, MYNN 6, MYNN-SL 16, Kh 5, hdiff 8); acoustic stage
  steps [1, 2, 4] × 14 fields with coefficients; PD every species covered on ≥ 1 frame. Tests: full CPU outputs give
  8/8 PASS; CG02 = {} gives FAIL; CG02 with 4/11 frames gives FAIL; no outputs gives 8/8 FAIL despite rc 0.
- Running sixth/PD/hdiff on all 11 frames on CPU (for the positive test) exposed the ice-free d04 τ62 frame. The frozen PD
  gate's limiter-active requirement then cannot hold for absent species. Rule added AFTER seeing this frame (E154 census),
  with no numeric bound change: an absent species must be exactly 0 on both sides, is reported, and is not coverage.
- The manifest now includes `cw_fixture_d06.npz.json` (55 files). The launcher runs from the DETACHED snapshot
  `<DATA_ROOT>/wrf_gpu2/scratch/w43-conv/gpu_snapshot` (commit ef7bd20477, data/wrf_pristine link, copied inputs). It refuses
  if HEAD is attached, if it is not running from the snapshot, or if the running file differs from the committed blob
  (rc 13), plus rc 10/11/12 as before.

## GPU CG01–08 (2026-10-21, real 11:35:23–11:40:38Z, one lock, lock rc 0)

Measured process: detached snapshot ef7bd20477 (`<DATA_ROOT>/wrf_gpu2/scratch/w43-conv/gpu_snapshot`), FROZEN_MANIFEST
55/55 sha256 verified by the launcher (manifest 7704262395e6…), src tree b3e9b92a8c (= v0.4.2), release fast defaults,
`GPUWRF_JAX_CACHE=0`, PREALLOCATE=false, private autotune dir. Every arm asserts `jax.devices()[0].platform == "gpu"`
(rc 0 in all 8 = assertion held; CG01 records `cuda:0`). Pallas: interpret=False explicitly in CG01/CG04/CG08; CG02/03/05/06/07
use the product's backend dispatch (Triton on GPU) [I for those five from the dispatch rule]. GPU memory at start 2,721 MiB
(desktop only). Same inputs, same pristine references, same frozen bounds and scorer as the CPU arms; numeric verdicts by
`analysis/gpu/gpu_verdicts.py` with explicit coverage (VERDICTS.json). Results copied to `analysis/gpu/results/`.

| arm | operator | coverage | verdict [M GPU] |
|---|---|---|---|
| CG01 | Thompson mp8 full column (Triton) | 586/586 columns, 5,860 field checks | **PASS** 0 fails; worst excess qr 3.09e-3, qs 2.81e-3, qg 2.50e-3, Nr 7.8e-4 (bound 1e-2; CPU 3.1e-3); precip 52.0068 vs WRF 52.0055 mm |
| CG02 | sixth-order diffusion | 11/11 frames, 22 fields each | **PASS** all |
| CG03 | MYNN PBL (F20 caller operands, p_hyd/hypso2) | 11/11 frames, 6 tendencies | **PASS** frozen 1e-3 |
| CG04 | PD scalar advection | 11/11 frames, 8 species, each species covered ≥1 (QICE/QNICE 10, d04 τ62 ABSENT exact 0) | **PASS** |
| CG05 | MYNN-SL | 11/11 frames, 16 fields | **PASS** v090 THR |
| CG06 | Kh (Smagorinsky 2-D) | 11/11 frames, 5 fields | **PASS** |
| CG07 | 2nd-order horizontal diffusion | 11/11 frames, 8 fields | **PASS** |
| CG08 | acoustic substeps + calc_coef_w (d06 fixture, max\|w\| 3,183 in state) | stages 1/2/4 steps × 14 fields + 3 coefficients | **PASS**, guard events 0, worst ratio 0.222 of bound (muave; CPU 0.16) |

Conclusion: on these real convection-active operands the Triton GPU kernels of the eight operator families match
pristine WRF 4.7.1 inside the frozen bounds, as the CPU interpret runs did. GPU and CPU results differ only at float32
rounding. **No operator-level mismatch was found on CPU or GPU.** The OPEN list below is unchanged: warm MYNN state, RK/step
coupling, cloudy radiation, Noah-MP, KF, boundaries, multi-step. Wall time 5 min 15 s of a 30 min slot.

## OPEN (not covered; narrowed claims)

- **MYNN warm state.** The F20 wrapper zeros QSQ, TSQ, COV, prior SGS cloud (qc_bl/qi_bl/cldfra_bl), exchange
  coefficients and RTHRATEN, which are not all Registry defaults at warm steps (`module_bl_mynnedmf_driver.F:407–430`),
  and the history frames lack them. The release adapter carries QSQ. **Warm-cloud / variance coverage of MYNN is
  therefore OPEN.** It needs caller-real warm operands from an admitted restart/state (A-148 lists τ39 restarts).
- Radiation tendencies in the PBL call (zero here), cloudy radiation, Noah-MP under rain, KF, nest boundaries, RK
  coupling, h_diabatic and multi-step evolution: OPEN, as listed below.

## Remaining / proposed next steps

- **GPU CG01–08**: done, ALL PASS (see below). Nothing left on the operator level for these families.
- **Step-level coupling oracle** (not built): one full model step from an identical active state, pristine-serial WRF vs
  GPU. It needs WRF-restart/state ingestion on both sides (the port has no WRF-restart interchange; restart files at τ39
  are admitted in A-148). This is the only route to the coupling/order class, so it is the manager's call.
- **#148 decisive measurement**: the N18 CPU twin (other decomposition), as w43-148fid and the critic already
  recommended. This is ALISIOS's decision.

## Reproduce

CPU, cores 8,9,12, nice 19, 12 G RAM slot, `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src`:
`analysis/thompson/{build_columns,run_oracle,score,run_mutant}.py`, `analysis/mynn/{mynn_oracle,mynnsl_oracle}.py`,
`analysis/dyn/{sixth_oracle,pd_oracle,kh_oracle,hdiff_oracle,acoustic_oracle}.py`. Inputs `analysis/data/frames/*.npz`
(untracked derived data, 483 MB) come from `analysis/extract_frames.py` (guarded). Pristine builds go to
`<USER_HOME>/wrf_gpu2_lanes/w43-conv/*_pristine`.
