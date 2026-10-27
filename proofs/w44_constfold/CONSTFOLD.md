# w44-constfold — WRF REAL constant folding on the release path

Lane w44-constfold (Opus 5.5, claude_2), 2026-10-24. Origin: o5-n333 FINDING 15:20Z (MYNN CASE2).
CPU only. All numbers [M] unless labelled.

## 1. The defect class

WRF is built with RWORDSIZE=4. gfortran folds a REAL constant expression per operation in REAL:
every leaf is a REAL literal or PARAMETER, and every operation is rounded to REAL.
Examples: `(t0c-6.) - tice`, `VKC*VKC`, `1./273.15`.

The same expression written in Python is folded in float64. It is rounded to float32 only once, when it
meets a float32 field (weak typing). The release (REAL) program can therefore hold a constant 1–3 REAL ulps
away from WRF's. Example: `(T0C-6.0)-TICE` gives 27.15 in the port, but WRF's REAL fold is 27.149994.

A single IEEE operation on float32-exact operands rounds identically both ways. A mismatch needs either an
inexact leaf (273.15, 0.729, 0.278, …) or two or more operations.

An earlier audit (BD94, 2026-10-06) already fixed many sites behind flags that are now on by default:
GPUWRF_{MYNN,KF,RRTMG}_REAL_CONSTANTS, GPUWRF_THOMPSON_WRF_CONSTANTS and GPUWRF_WRITER_RVOVRD_REAL.
This lane audits the residual after BD94.

## 2. Method (tools in `proofs/w44_constfold/tools/`)

1. **`constfold_scan.py` — static scan of the whole source tree.** It parses all 871 modules of
   `src/gpuwrf` and resolves module constants, imports, aliases, function locals and np.float32 casts. Every
   maximal constant subtree, and every use of a constant whose own definition mismatches, is evaluated twice:
   once as the port evaluates it, and once as a REAL per-operation fold.
   - Result: 965 sites over all schemes (scan_all2).
2. **`release_coverage.py` — what the release program actually contains.** It runs the release identity probe
   (`proofs/s1_port/release_identity.py`: real CLI loader, release defaults, one production step traced). It
   records every floating literal, including the bodies of Pallas kernels, together with the innermost gpuwrf
   source line. Dead equations are removed first: DCE at the top level and in every sub-jaxpr, including
   `while` bodies, which have no JAX DCE rule.
   - Cases: Swiss d01 and PROD d01. The identity shas reproduce while tracing (b977712f / 321aa651).
3. **`match_release.py` — join scan and release program.** A site is CONFIRMED when the float32 of its
   Python fold is a live literal on its own lines.
   - Live result (Swiss + PROD): 18 CONFIRMED, 16 already FIXED by BD94, 20 EXECUTED-but-unattributed (all
     reviewed by hand), 911 NOT on the release path.
4. **`unexplained_literals.py` — reverse sweep.** It lists release literals that are not explained by a single
   literal or constant: 800 records, of which 369 remain after removing global-literal and exact-dyadic values.
   All 369 were triaged by hand. This sweep found two sites that the forward scan attributed elsewhere:
   - the SW band-sums `bpade`;
   - the all-constant MYNN `f2` product.
5. **`wrf_residual_oracle.py` — pristine gfortran oracle.** It compiles 31 WRF statements verbatim with the
   pristine build compiler (conda gfortran 14.3). The PARAMETER chain comes from the pristine build's own
   `.mod` files; module-private PARAMETERs are redeclared verbatim. Thompson gamma constants use WRF's
   GAMMLN/WGAMMA copied verbatim. The program prints the IEEE bits of each statement.
   - Output: `wrf_residual_oracle.json`; program source: `wrf_residual_oracle.f90`.

## 3. Findings: release-path sites where the port's constant ≠ WRF's REAL bits (fixed)

| site | expression (WRF ref) | port f32 | WRF REAL (oracle) |
|---|---|---|---|
| mynn_edmf.py `_qsat_blend` chi | `(t0c-6.) - tice` (MYNN :7664) | 27.15 | 27.149994 (3 ulp) |
| mynn_sgs_cloud.py `xl_blend` | `cpv-cice` (:7688) | −259.60001 | −259.59998 |
| mynn_sgs_cloud.py `xl_blend` chi | `t0c - tice` (:7693) | 33.150002 | 33.149994 (2 ulp) |
| mynn_sgs_cloud.py CASE2 q1_rh ×2 | `one-rhcrit` (:3669/:3677) | 0.17 | 0.17000002 |
| mynn_pbl.py `_mym_level2` f1, f2 | `one -c2` (:1555/:1557); f2 = `b1*(g1+g2)-3.0*a1*(one-c2)` | 0.271 / f2 18.21978 | 0.27100003 / 18.219778 |
| surface_layer.py MYNN-SL | `1.-ep_2` (module_sf_mynn.F:81) | 0.37824956 | 0.37824959 |
| surface_layer.py MYNN-SL | `1./273.15` (:525/:543) | 0.003660992 | 0.0036609923 |
| noahmp/energy.py `sfcdif1` ×2 | `VKC*VKC` (noahmplsm :4734/:4735) | 0.16 | 0.16000001 |
| noahmp/energy.py RSURF | `( 2.71828 - 1.0 )` (:2185) | 1.71828 | 1.7182801 |
| noahmp/snow.py SNOWH2O ×2 (+ legacy twin ×2) | `max_liq/(1.0-max_liq)` (:7216/:7217) | 0.6666667 | 0.66666663 |
| kernels/phys_kf_column.py dd_evap (PROD d01) | `0.2/1000.` (kfeta :1743) | 2.0e-4 | 2.0000001e-4 |
| kernels/rad_sw_band_sums.py lookup | `1.0_rb/pade`, pade 0.278 (rrtmg_sw :1229/:4755) | 3.5971222 | 3.5971224 |

The port constant is 1 ulp from WRF's unless noted.

**Same mechanism with a namelist REAL operand (found by the reverse sweep, fixed):** WRF forms these REAL
products at run time; the port folded them in float64 at trace time.

| site | WRF statement | values that differ (port → WRF) |
|---|---|---|
| kernels/dyn_acoustic_fp32.py w-phase `dampmag` | `dampmag = dts*config_flags%dampcoef` (module_small_step_em.F:1284) | dts 4.5: 0.89999998 → 0.90000004 (Swiss stage 2/3, PROD/WN3 d02); dts 18: 3.5999999 → 3.6000001 (PROD d01 stage 1); dts 13.5/6/1.5 equal |
| noahmp/energy_radiation.py CLASS snow albedo | `EXP(-0.01*DT/3600.0)` (noahmplsm :3261) | host-DT (CPU/XLA) path only: DT 54 (PROD d01) −1.50000007e−4 → −1.49999993e−4; DT 18/6 equal. The GPU Noah-MP radiation kernel takes DT as a traced input and already evaluates it per op in REAL: unchanged there (critic F2) |
| noahmp/snow.py SNOWH2O ×2 | `SNOW_RET_FAC*DT*QOUT` (:7211) | DT 54: 0x3b30f27c → 0x3b30f27b; DT 6 (WN3 d03): 0x399d4952 → 0x399d4951; DT 18 equal |

Not covered by this lane (general users): `rdx = 1./dx` with a non-f32-exact dx (e.g. nest dx = parent
dx/3 = 666.67 m). All release cases use integer dx (9000/3000/1000), so the release is unaffected. The class
could matter for user domains.

The MYNN CASE2 sites are o5-n333's original finding. The other rows are new.

## 4. Release-path sites checked and NOT defects

- **Port already equals WRF's bits:**
  - surface_layer `P0_PA*0.001`: WRF writes the literal `100.`, so the port's 100.0 is correct.
  - Thompson `(0.99-0.01)` eva_factor, `(C_cube-C_sqrd)`, `5.*D0s`; MYNN-SL `-KARMAN*CZIL`; MYNN
    `rbm_st-1`; WGAMMA(2.5) = the port's CRG12/CIG7.
  - Thompson size-distribution prefixes (`am_i*cig(2)*oig1`, `am_r*crg(3)*org2`, `am_g(5)*cgg(3,1)*ogg2`,
    `Kap0*csg(4)`, `4218.*olfus`, `oams`).
  - KF `cliq`, `cliq-bliq*dliq`, `rdpr`.
  - MYNN ψ `-0.33333*am_unst`.
  - MYNN level-2 `b1*(g1-c1)`, `2*a1*(3-2*c2)`, `3*a2`, `one-c5`.
  - The BD94 typed values (RVOVRD, P608, GTR, MYNN a1…e5c, KF aliq/gdry, Thompson WRF_REAL_CONSTANTS,
    RRTMG LW bpade, heatfac).
  - dx/dt-derived scalars (rdx, dts, odts, reradius): single operations on exact operands.
  - Exact-dyadic values.
- **Association (evaluation-order) differences, not constant folds.** These are out of this lane's scope and
  reported here only:
  - rrtmg_lw.py:3193: the port scales per band by `delwave*(pi*1e4)`. WRF sums `urad*0.5*delwave` over bands,
    then multiplies by `fluxfac = pi*2.e4`.
  - rrtmg_sw.py:1685: the port computes `(4.35e-4/700)*colo2`. WRF computes `4.35e-4*colo2/700`.
  - noahmp water_hydro.py:843: the port computes `FRZK*(smcmax/smcref)*0.88`. WRF computes
    `FRZK*((smcmax/smcref)*(0.412/0.468))`; the constant itself is equal.
  - `jnp.log10` is lowered as `log(x)*0.4342945`, not REAL `log10f`. This is a libm/implementation class; it
    explains o5-n333's remaining 31 CASE2 ice cells.
- **Thompson graupel number fall speed** (`thompson_column.py:2251`): uses the non-WRF `CRG7`
  (f32(Γ(3.5)) 3.3233509 vs WGAMMA 3.3233511) together with the hail-aware `AV_G_MP8`. mp8 has no prognostic
  graupel number in WRF, so this is not a WRF output path. Noted for b-thompson (B56 context).
- **Owned by another lane (reported via MAILBOX, not touched):**
  - Topo-shadow azimuth thresholds `1.75*pi` and `1.25*pi`: 1 ulp each against WRF's REAL `pi=4.*atan(1.)`.
    Owner: w44-topo.
- **Not on the release path, or host DOUBLE transcriptions (separate class):**
  - d02_replay IC/wrfbdy theta recoupling: this is f64 host arithmetic with f64 rvovrd, a DOUBLE-vs-REAL
    init-transcription class rather than a constant fold.
  - io/wrfrst_netcdf `_dry_theta`: uses Python RVOVRD on f32 arrays. It is an opt-in restart writer and should
    reuse the writer's `RVOVRD_REAL` (FINDING to w44-warmstart).
  - 911 scan sites in non-release schemes, e.g. WDM/WSM/Morrison/Tiedtke/MYJ/Shin-Hong. They are listed in
    `scan_all2.json` and are relevant if those schemes are ever qualified on the REAL path.

## 5. Fix

- **Flag.** New module `src/gpuwrf/contracts/real_constfold.py` defines `GPUWRF_WRF_REAL_CONSTFOLD`, an
  import-time flag registered in `aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS` (E80).
  - **Default OFF (manager 22:12Z).** The v0.4.4 RC switches it on together with the topo and boundary-clock changes
    as ONE identity change; see `SEAM_SUITE_RECIPE.md` for the two edits.
  - The ON identities below are therefore the RC's numbers. On this branch the default program equals main (CF28).
- **Selection rule.** `real_constant(real, legacy, dtype)` returns WRF's REAL fold only for float32 (REAL)
  consumers.
  - float64 (legacy/DOUBLE) consumers keep the Python value.
  - With the flag OFF, every site returns the old Python value: the program is unchanged.
- **Config values that can become runtime parameters** (dt, dts, dampcoef): `real_scalar_product(a, b)` refolds only
  when both operands are host scalars. A traced operand (e.g. w44-params `--runtime-parameters dampcoef`) is never
  converted on the host: it is cast to float32 with jnp and multiplied in REAL, in WRF order (manager rule, 18:19Z).
  Joint test: `test_dampmag_runtime_parameter_real_fold` (traced f32 and f64 dampcoef, flag ON and OFF).
- **Trace order is preserved.** The CLASS albedo argument is computed in place inside `EXP`, so `(ALBOLD-0.55)` is
  traced first exactly as on main (critic F2).
- **How the REAL values are written.** Each REAL value is spelled as `np.float32` arithmetic in WRF's source
  order next to its site. No numeric literals are hand-typed.

## 6. Evidence (all [M] CPU unless labelled)

Source state: commits 85bac645 → de0c8e16 → 99ca2239. src tree `998b2f6d`, identical through HEAD.

| id | what | result |
|---|---|---|
| oracle | `wrf_residual_oracle.{f90,json}`: 39 WRF statements, pristine gfortran 14.3 | 16 fixed release values ≠ port; all fixed values now == oracle bits |
| tests | `tests/test_w44_constfold.py` | 53 PASS. Covers oracle pins, traced-literal ON/OFF/f64 selection per site (incl. `_mym_level2` one-c2 + f2 and MYNN-SL ep_3 + 1/273.15), wiring pins of every REAL-fold call (REAL value + dtype expression), AST site guards, traced-dampcoef joint rule, registry |
| CF23 | own mutants: revert each fix, site, constant, helper or traced cast (`tools/mutants.py`) | 25/25 killed, assertion failures only |
| CF24 | critic's semantic wiring mutants CM1–CM5 (`cr-w44-constfold/tools/critic_mutants.py`, unchanged) | CM1–4 (dtype → f64 / wrong REAL slot) and CM5 all killed, each site by its own pin/traced test |
| CF12 | MYNN CASE2 vs pristine-linked `mym_condensation` (o5-n333 harness), Swiss CPU-WRF icy day 12Z, 1,764 columns | cldfra mismatches 2,898 → 31; qc_bl 2,810 → 30; qi_bl 2,850 → 31. The residual is entirely icy-aloft (`log10` libm class) |
| CF18 | same oracle, night CPU-WRF restarts 2026-05-04 03Z d01/d02/d03 | bitwise 0/0/0 with the flag OFF and ON; non-vacuous (45,703 / 159,845 / 43,223 cloudy cells) |
| CF14–CF17 | release identity (`proofs/s1_port/release_identity.py`) | flag OFF: Swiss **b977712f**, PROD **321aa651** (main, exact). Flag ON (release default): Swiss **91d91b508623e406…**, PROD **5451bf2edd43dbd7…** |
| CF14–CF17 | structure hash: canonical jaxpr with float literal values masked | ON == OFF (Swiss 44f07ff3…, PROD 89b9f72d…). The program has the same equations, primitives, shapes and dtypes; only constants differ, so there are no new kernels or ops and no cost |
| CF19 | `verify_fixed.py` on the live ON/OFF release programs | 16/16 site values: ON contains WRF's bits only; OFF contains the old bits only |
| registry | `tests/test_fast_defaults.py` 6/6; `tests/test_aot_cheap_key.py` import-time-env scanner + registry resolve 2/2 | PASS |

**Release identities in both conventions (E222; CF27, head 1162013b07 = src f3119637, base = branch point 44ec0de450,
same alias path, o5-port `off_identity_probe.py` @3f9cd9f0c4, all arms COMPLETE = 0 interpreted Pallas calls).**
d01 of each case: Swiss RD11, PROD s0_case_20260725, WN3 0227 (staged CPU-WRF inputs, max_dom=1).

| case | arm | canonical (release_identity) | native raw HLO | kernels / lowerings | location-free TTIR vs base |
|---|---|---|---|---|---|
| Swiss | base | b977712f | 09ca2808 | 62 / 157 | – |
| Swiss | flag OFF | **b977712f** | c193a783 | 62 / 157 | all equal |
| Swiss | flag ON (release default) | **91d91b50** | 1cfbd158 | 62 / 157 | 4 kernels differ |
| PROD | base | 321aa651 | c7183eff | 61 / 157 | – |
| PROD | flag OFF | **321aa651** | d8ecefc7 | 61 / 157 | all equal |
| PROD | flag ON | **5451bf2e** | 7c223677 | 61 / 157 | 5 kernels differ |
| WN3 0227 | base | a9b1b7e9 | dea299c3 | 61 / 157 | – |
| WN3 0227 | flag OFF | **a9b1b7e9** | 2584efa2 | 61 / 157 | all equal |
| WN3 0227 | flag ON | **c37b0069** | 9ecc2104 | 61 / 157 | 5 kernels differ |

- **Flag OFF.** The canonical identity is exact, and the location-free TTIR of every kernel is equal to base.
  - The raw HLO differs only through the located TTIR payloads (line shifts, E109) of the kernels defined in edited
    files: acoustic ×7, MYNN plume `_lambda_`, Noah canopy/snowwater/radiation, SW band sums, and KF on PROD/WN3.
  - XLA-part equality from the split HLO is pending (E222h).
- **Flag ON.** The kernels that change are MYNN plume `_lambda_`, `b_core_noah_canopy_flux`, `b_core_noah_snowwater`,
  `rrtmg_sw_band_sums`, and `kf_column_fp32` (PROD/WN3; Swiss has cu=0).
  - `b_noahmp_radiation_real` is unchanged: its CLASS albedo path is already REAL with the in-kernel traced DT.
  - dampmag reaches the program as an XLA constant (acoustic slot vector), not as kernel TTIR.

**Default identity of the merge head (CF28; head 6dc6f1f27f, src 770bb89d, flag unset = OFF).** Measured in the
production convention (o5-port probe, same alias, clean env, all arms COMPLETE) against base 44ec0de450:

| case | canonical | location-free TTIR | env / kernel counts | raw HLO |
|---|---|---|---|---|
| Swiss | b977712f == base | all equal | equal | differs only through located TTIR payloads (line shifts) |
| PROD | 321aa651 == base | all equal | equal | same |
| WN3 0227 | a9b1b7e9 == base | all equal | equal | same |

The critic's HLO split showed that the XLA part is byte-identical (cr-w44-constfold 22:10Z, on the OFF arm of the
same src minus the fast-default line).

**Not yet measured:**
- The touched-seam test subset under release defaults, flag OFF vs ON. Manager 22:12Z moved it into the w44-release
  RC tiers; see `SEAM_SUITE_RECIPE.md` for the recipe.
- D6 relevance on GPU (§7).

## 6b. Where each fixed constant is active on the release cases [I from code paths + namelists]

Namelists: Swiss and PROD use damp_opt 3 (dampcoef 0.2, zdamp 5000 m). cu_physics is 0 on Swiss and 1 on PROD d01.
Noah-MP options are not set, so WRF defaults apply: opt_alb 2 (CLASS), opt_rsf 1, opt_snf 1. bl_mynn_cloudpdf is 2.

| site | active where | expected reach |
|---|---|---|
| MYNN level-2 `(one-c2)` f1/f2 | every PBL column and level, every step (stability functions sm/sh) | everywhere |
| MYNN-SL `ep_3` | QSFC recompute over water, and over land cells where QSFC ≤ 0, every step | all water cells |
| MYNN-SL `1./273.15` | only cells with TSK < 273.15 (ice saturation formula) | Swiss winter land; rare on PROD/WN3 |
| Noah-MP `VKC*VKC` (SFCDIF1) | every land column, every Noah-MP iteration | all land |
| Noah-MP RSURF `(2.71828-1)` | bare-soil surface resistance (opt_rsf 1), all land | all land |
| Noah-MP snow liquid ratio / `SNOW_RET_FAC*DT` | snow layers with liquid excess or outflow | snow and melt only. SNOW_RET_FAC: GPU kernel DT is traced, so DT 54/6 values differ only on the host path |
| acoustic `dampmag` | the w-Rayleigh damping layer (top 5 km), every acoustic step | all columns, top levels |
| RRTMG-SW `bpade` | transmittance lookup in every daylit column (band-sums kernel) | daytime, everywhere |
| MYNN CASE2 qsat_blend / xl_blend / q1_rh | SGS cloud in the 240–267 K band, ice above the PBL | cold or icy cloud (CF12: 1,347 changed cells in 1,764 Swiss columns) |
| KF `0.2/1000.` | KF downdraft evaporation, deep-convective columns only | PROD/WN3 d01 convective columns |
| CLASS albedo | GPU kernel unchanged (traced DT), host path only | none on GPU |

**Consequence.** Several of these constants are touched in every column and every step (level-2, VKC², dampmag,
daytime SW). The change is therefore a global 1-ulp-class perturbation. It grows into round-off-class chaotic
divergence, not a localized bias. The D6 gate (§7) measures exactly that.

## 7. D6 relevance and packaging (proposal)

- **Expected size of the change [I].** Every change is a 1–3 ulp constant in REAL arithmetic.
  - The CASE2 diagnostic changes are ≤ 7e-6 in cldfra (CF12).
  - The expected forecast effect is round-off-class chaotic divergence: the same class as E99 cold-compile
    nondeterminism, far below the D6 bounds (T2 1.5 K, etc.).
- **Swiss is already covered by the release plan.** w44-release `RW_SESSION_PLAN.md` branches on the Swiss
  candidate identity. With this change the identity is 91d91b50, so branch (b) applies: the B9 Swiss 24 h arm is
  scored D6 against the original CPU-WRF RD11/cpu.
- **WN3/PROD-class physics is covered by RW3.** PROD/WN3 numerics already change because of the w44-topo fix,
  and RW3 (WN3 0227 72 h D6 against CPU-WRF) covers MYNN, MYNN-SL, Noah-MP, Thompson and RRTMG together with
  this change (d02/d03).
- **Proposal: no separate GPU window.** Let the composed v0.4.4 RC carry the flag and use B9 + RW3 as the D6
  gate. If the manager wants D6 attributed before composing, a dedicated pair of arms would be needed: Swiss 24 h
  flag ON vs CPU-WRF, plus PROD 24 h.
- **Merge without changing identity is possible.** Flip `GPUWRF_WRF_REAL_CONSTFOLD` to "0" in `_fast_defaults`.
  - OFF is exact in the canonical convention (CF16/CF17) and in the production convention (E222: all location-free
    TTIR equal).
  - The only located-TTIR differences are line shifts in the kernels of edited files.

