# o5-f2neg — F2 post-RK negative moisture: amplitude vs pristine WRF on the same operands

**Verdict: class (1) FAITHFUL on these operands, with the port evaluated on CPU (XLA).** The operands are:

- the Swiss root domain in-run, 0–3.6 h (720 × 18 s);
- history-state probes of Swiss Thompson, PROD d01/d02 interiors and Swiss WDM6.

At the verified sites, the verbatim pristine WRF 4.7.1 REAL4 chain, fed the same operands, reproduces every port
negative within rounding. That holds for Thompson qc/qr/qi/qs/qg/Ni/Nr and for WDM6 + Nn/Nc. Counts under the
pre-registered rule:

- 0 cells more negative in the port;
- 0 cells less negative;
- 0 cells negative only in WRF.

**Above one ulp of the local field (> 1 eps·S), every port negative has WRF's sign and its amplitude within
≤ 1.4e-4 relative.** That is ≤ 2.4e-5 for |q| > 64 eps·S (in-run). Below 1 eps·S, the sign is set by rounding (§3.4).

**No port amplification was found at the verified sites.** The sites in §5 are class (3), unverified. These
include the nested-child path, the KF cumulus tendency, WN3 and GPU evaluation. The critic for this report is
cr-o5-f2neg (verdict FIX → this revision).

## 1. Where the negatives are born (pristine source + measurement)

At the last RK stage WRF pre-adds the whole RK1-frozen `moist_tend`/`scalar_tend` to the step-start field
(`rk_update_scalar_pd`, `solve_em.F:1934-1955`, `module_em.F:1877-1911`). `advect_scalar_pd` then limits only the
outgoing antidiffusive fluxes against that pre-added field, and nothing clips afterwards.

Every large negative measured is **born in the pre-add**. In-run, of 4065 pre-add-born cells:

- the sixth-order diffusion sc_tend is the most negative increment in 3273;
- the diff_opt=1 scalar mixing is the most negative in 792;
- the PBL (MYNN) physics sc_tend never is.

WRF forms the sixth-order term at RK1 with `dt_step = dt/3`, which triples its coefficient, and applies it over
the full dt. Advection then spreads part of the negative to neighbours; those advection-born negatives are
smaller. Rounding-level negatives come from cancellation in the final update.

## 2. Method

**Oracle.** `pristine_chain.py` extracts six WRF routines verbatim at build time:
`horizontal_diffusion('m')`, `sixth_order_diffusion('m')`, `rk_update_scalar_pd`, `advect_scalar_pd`,
`rk_update_scalar` and `flow_dep_bdy`. They sit in one REAL4 driver wired like solve_em, with two modes:

- *chain* mode recomputes everything from the operands;
- *isolated* mode feeds each operator the port's own input.

**Builds.** The oracle is built twice, with `-ffp-contract=off` and with `=fast` (both `-march=native`; the
`fast` build has 232 FMAs). The difference between the two is the per-cell rounding envelope. The critic also
built WRF's own FCOPTIM flags (no FMA); that build is bitwise equal to `off` on 7 steps × 8 species.

**Classification rule** (`replay_lib.py`, pre-registered before any data was seen):

- `tol = max(8·E_FMA, 64·eps32·S)`, where S is the field maximum over the 7×7×3 stencil footprint.
- Every port-negative cell is FAITHFUL, LARGER_IN_PORT or SMALLER_IN_PORT. WRF-only negatives are counted
  separately.
- A band split added afterwards for reporting only (critic F1) does not change the classes. It uses
  R0 = |port| ≤ 1 eps·S (sign set by rounding), R1 = 1–64 eps·S and R2 = > 64 eps·S.

**What the rule can and cannot see (critic F2).**

- It catches amplitude-scale mutants. A sixth-order term formed with `dt` instead of `dt/3` is flagged on 7/7
  species (port min −1.16e-6 vs WRF −3.95e-6), and so is the diff_opt=1 scalar mixing removed.
- It is **blind to E11-class mass-wiring errors**. The critic's mutants all stayed FAITHFUL: hdiff/sixth fed
  `mu_1`, PD fed `mu_1`, pre-add fed `mu_2`. They reach 0.08–2.3 eps·S at port negatives, vs 0.02–0.04 for the
  true wiring.
- The wiring is therefore proven only by the **7 array-equality checks** of §3.3 (54/54 steps each).
- The amplitude evidence is the **measured** max difference (≤ 0.04 eps·S in-run), not the tolerance.
- A relative amplitude error smaller than 64·eps·S/|q| would be invisible to the rule.

**Footprint (critic F6).** 7×7×3 is vertically narrower than the 3rd-order vertical advection stencil. All-cell
(not port-negative) outliers up to 1.3e6 eps·S sit at XLA:CPU-flushed subnormal cells with tiny S. With the FTZ
oracle and a 7×7×5 footprint they drop to ≤ 11 eps·S (§3.4). The pre-registered rule is kept unchanged.

## 3. Results

### 3.1 Port chain functions on history states of our own GPU runs (F2N02–F2N04)

The operands are hourly wrfout of our GPU runs:

- q_old = q_cur = history;
- both sides get the same masses, winds and xkhh;
- the physics sc_tend is set to 0.

The port side is the production function set:

- `horizontal_diffusion_coord_scalar_tendency`;
- `wrf_sixth_order_scalar_tendf`, which calls the native `b_diff_sixth_m`;
- `_rk_update_scalar_pd`;
- the native `advect_scalar_pd_fp32`;
- `_root_scalar_stage_tendencies` and `_apply_moisture_large_step`;
- `flow_dep_bdy`.

"FAITHFUL (WRF ≥ 0)" counts the port negatives that are FAITHFUL while WRF's value is ≥ 0 at rounding level.

| case | species | port neg | FAITHFUL (WRF ≥ 0) | LARGER | SMALLER | WRF-only | WRF neg | port min | WRF min | max diff/(eps·S) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Swiss Thompson, 8 frames 03–23 h | qc | 2347 | 2347 (932) | 0 | 0 | 0 | 2081 | −3.955e-06 | −3.955e-06 | 2.7 |
| | qr | 2007 | 2007 (618) | 0 | 0 | 0 | 1826 | −1.515e-07 | −1.515e-07 | 0.034 |
| | qi | 3271 | 3271 (1155) | 0 | 0 | 0 | 2787 | −3.251e-08 | −3.251e-08 | 0.046 |
| | qs | 4289 | 4289 (1392) | 0 | 0 | 0 | 3619 | −1.219e-06 | −1.219e-06 | 0.022 |
| | qg | 992 | 992 (340) | 0 | 0 | 0 | 873 | −1.165e-07 | −1.165e-07 | 0.13 |
| | Ni | 4173 | 4173 (1229) | 0 | 0 | 0 | 3680 | −4937 | −4937 | 1.3 |
| | Nr | 2384 | 2384 (673) | 0 | 0 | 0 | 2127 | −10.21 | −10.21 | 0.012 |
| PROD d01, 4 windows (ring ≥ 7) | qc | 234 | 234 (91) | 0 | 0 | 0 | 216 | −4.855e-06 | −4.855e-06 | 0.091 |
| | qr | 339 | 339 (85) | 0 | 0 | 0 | 300 | −5.298e-08 | −5.298e-08 | 0.016 |
| | Nr | 393 | 393 (95) | 0 | 0 | 0 | 358 | −0.1811 | −0.1811 | 0.008 |
| PROD d02 interior, 4 windows (ring ≥ 7) | qc | 217 | 217 (72) | 0 | 0 | 0 | 223 | −3.166e-06 | −3.166e-06 | 0.006 |
| | qr | 314 | 314 (97) | 0 | 0 | 0 | 281 | −2.64e-09 | −2.64e-09 | 0.003 |
| | Nr | 372 | 372 (111) | 0 | 0 | 0 | 349 | −0.3948 | −0.3948 | 0.004 |

Notes on this table:

- 24–40 % of the FAITHFUL cells have WRF ≥ 0. These are rounding-level cells; §3.4 has the band split.
- PROD is summer, with no ice in these windows (E88).
- The PROD windows are 48×48 crops (the 1 GB no-slot cap), classified only at ring ≥ 7 inside the window.
- PROD d02 is evaluated with root-style wiring assembled by the probe. That is **not** the production
  nested-child path (§5).

Isolated operators, port vs pristine on the port's own inputs, over all frames:

- hdiff, sixth-order, pre-add and update are ≤ 2.4e-7 relative;
- PD advection is ≤ 1.4e-6 relative (qv; the other species ≤ 9e-7);
- flow_dep is bitwise equal.

The update stage differs on ring 0 for species without a boundary leaf, because the port keeps ring
advection there while WRF masks the spec zone. `flow_dep_bdy` overwrites ring 0 in both, so the post-RK field is
identical. The interior-only update difference is ≤ 2.3e-7.

The critic's band split for the Swiss 12 h frame (`cr_frame_12h.json`) gives R2 cells with the same sign in
every case and max rel diff ≤ 6.9e-6:

- qc 73, qr 158, qi 85, qs 202, qg 76, Ni 124, Nr 209;
- 0 WRF-only cells above 1 eps·S.

### 3.2 WDM6 and the XLA:CPU denormal flush (F2N04, F2N05)

**The discrepancy.** Against pristine with IEEE gradual underflow, WDM6 qs/qg differ at up to 240 cells per frame;
the minima are still equal. The isolated comparison pins this to the sixth-order operator:

- the port returns exactly 0 next to cells holding exactly FLT_MIN (1.175e-38);
- there WRF's limiter test `dflux·(q(i)−q(i−1)) .LE. 0` sees a subnormal product (≈3.6e-44) and keeps the flux;
- XLA:CPU flushes denormals (FTZ+DAZ), so the CPU port zeroes the flux.

**The fix in the oracle.** With the pristine chain run under the same x86 FTZ|DAZ mode (`mxcsr.c`), qs/qg become
FAITHFUL 0/0/0. gfortran's `ieee_set_underflow_mode` is a no-op on this host. CPU-WRF's own WDM6 output also holds
exact-FLT_MIN qs/qg cells (4–6 per frame), so those values come from the scheme itself.

**On the GPU [I].** The shipped v0.4.3 scalar-chain kernels contain no `.FTZ` instruction (SASS census of the
sealed bundle): `b_diff_sixth_m`, `b_diff_horizontal_m`, `b_core_pd_tend/scale_sl_fp32` and
`b_core_scalar_flux_fp32(_stacked)`. The program-wide 0.16 % of FTZ ops are a fixed 8-op pattern inside IEEE
`div.rn.f32` slow paths. So the CPU-only denormal flip cannot occur on the GPU. GPU rounding (`div.full.f32`,
FMA) still differs from REAL4 at the ulp level, and that was **not** evaluated.

WDM6, 6 frames (10/12/14/17/18/19 h, around the GPU 24 h extrema), compared under IEEE gradual underflow and under
the CPU's FTZ|DAZ:

| species | oracle mode | port neg | FAITHFUL (WRF ≥ 0) | LARGER | SMALLER | WRF-only | port min | WRF min | max diff/(eps·S) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qc | IEEE = FTZ | 983 | 983 (390) | 0 | 0 | 0 | −4.032e-08 | −4.032e-08 | 0.033 |
| qr | IEEE = FTZ | 793 | 793 (217) | 0 | 0 | 0 | −1.419e-06 | −1.419e-06 | 0.033 |
| qi | IEEE = FTZ | 4640 | 4640 (1615) | 0 | 0 | 0 | −5.944e-07 | −5.944e-07 | 0.074 |
| qs | IEEE | 8475 | 8311 | 39 | 125 | 301 | −6.8e-06 | −6.8e-06 | 2.4e4 |
| qs | FTZ | 8475 | 8475 (1595) | 0 | 0 | 0 | −6.8e-06 | −6.8e-06 | 0.043 |
| qg | IEEE | 7369 | 7137 | 58 | 174 | 833 | −2.817e-06 | −2.817e-06 | 4.0e4 |
| qg | FTZ | 7369 | 7369 (1462) | 0 | 0 | 0 | −2.817e-06 | −2.817e-06 | 0.105 |
| Nn | IEEE = FTZ | 595 | 595 (45) | 0 | 0 | 0 | −8.849e+07 | −8.849e+07 | 0.054 |
| Nc | IEEE = FTZ | 1280 | 1280 (438) | 0 | 0 | 0 | −1.869e+07 | −1.869e+07 | 0.011 |
| Nr | IEEE = FTZ | 872 | 872 (230) | 0 | 0 | 0 | −277.5 | −277.5 | 0.035 |

### 3.3 In-run capture (F2N01): the port's own operands, real steps

**Capture.** `capture.py` runs the real Swiss release-default case on CPU through the production single-domain
entry `om._advance_chunk`. Its inputs are SHA-identical to the v-run GPU arm. It taps eight scalar-chain call
sites with `jax.debug.callback`, so operands and results come from the same compiled program.

The only program change is disclosed (F5): the RRTMG column tile is 128 instead of 4096, to fit the CPU memory
cap. That is radiation-only, byte-neutral per A43 [I], and the taps change no scalar-chain value.

**Replay.** `replay.py` feeds the captured full-grid operands to the pristine chain:

- the step-start fields;
- the port's PBL physics sc_tend (MYNN; present for qv, qc, qi and Ni);
- its QV boundary tendency;
- xkhh;
- the actual stage masses and the RK3 transport fluxes.

It runs both from scratch (pristine hdiff + sixth-order) and isolated, under IEEE and FTZ semantics.

**Run3.** 720 steps (0–3.6 h), 54 full-grid captures (fixed steps + records), source `c9c7ac0f20`, clean.

**Caller wiring**, verified by array equality on every captured step (54/54 each). These checks are the only
wiring proof (§2):

- hdiff and sixth-order receive the post-RK1-acoustic `muts` (solve_em.F:2303);
- the pre-add uses the step-start `mu_1` (:1946);
- the PD advection uses the RK3 post-acoustic mass and `mu_1`;
- the RK3 scalar field is the RK2 result after its flow_dep_bdy;
- the PD kernel sees exactly the pre-added field.

| species | port neg | FAITHFUL (WRF ≥ 0) | LARGER | SMALLER | WRF-only | port min (step) | WRF at that cell | max diff/(eps·S) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qc | 5674 | 5674 (1857) | 0 | 0 | 0 | −6.53e-13 (600) | 0 (rounding level) | 0.041 |
| qr | 1387 | 1387 (286) | 0 | 0 | 0 | −2.1894e-09 (200) | −2.1894e-09 | 0.015 |
| qi | 1113 | 1113 (301) | 0 | 0 | 0 | −6.92434e-08 (720) | −6.92435e-08 | 0.022 |
| qs | 1011 | 1011 (235) | 0 | 0 | 0 | −6.9699e-08 (720) | −6.9699e-08 | 0.028 |
| qg | 197 | 197 (35) | 0 | 0 | 0 | −5.01577e-09 (720) | −5.01577e-09 | 0.007 |
| Ni | 2013 | 2013 (523) | 0 | 0 | 0 | −7719.27 (123) | −7719.28 | 0.023 |
| Nr | 2399 | 2399 (492) | 0 | 0 | 0 | −0.41821 (720) | −0.41821 | 0.012 |

IEEE and FTZ oracles give identical classes here. Stage by stage, the from-scratch pristine chain matches the port
to ≤ 3.4e-7 relative (interior), and the PD advection to ≤ 7.5e-6 (qv).

[Run4, the full 24 h with captures at the GPU extremum steps, is running; its results will be appended. Run4 is
not a gate for this report.]

### 3.4 Band split: what "same negative" means (F1)

In-run band split over all 54 captured steps (`F2N01_run3_summary.json`, chain mode, IEEE oracle):

| species | R0 (≤ 1 eps·S) n, of which WRF ≥ 0 | R1 n, same sign, max rel diff | R2 n, same sign, max rel diff | WRF neg total, of which R0 | WRF-only > 1 eps·S | all-cell max diff/(eps·S): IEEE 7×7×3 / 7×7×5; FTZ 7×7×3 / 7×7×5 |
|---|---|---|---|---|---:|---|
| qc | 5674 (1857) | 0 | 0 | 7281 (7281) | 0 | 6.8e5 / 6.1e3; 6.2e4 / 11 |
| qr | 748 (286) | 65, 65/65, 3.0e-5 | 574, 574/574, 2.1e-6 | 1512 (873) | 0 | 110 / 2; 110 / 2 |
| qi | 716 (301) | 87, 87/87, 2.7e-5 | 310, 310/310, 5.2e-6 | 1183 (786) | 0 | 1.3e6 / 1.6e4; 16 / 2.2 |
| qs | 619 (235) | 79, 79/79, 2.0e-5 | 313, 313/313, 7.2e-6 | 990 (598) | 0 | 150 / 2.2; 2.2 / 2.2 |
| qg | 84 (35) | 16, 16/16, 2.7e-5 | 97, 97/97, 4.1e-6 | 187 (74) | 0 | 2.6 / 2.6; 2.6 / 2.6 |
| Ni | 1346 (523) | 141, 141/141, 1.4e-4 | 526, 526/526, 2.4e-5 | 2310 (1643) | 0 | 42 / 2.9; 42 / 2.9 |
| Nr | 1714 (492) | 91, 91/91, 5.1e-5 | 594, 594/594, 6.6e-6 | 2476 (1791) | 0 | 3.9e4 / 2.6; 3.9e4 / 2.6 |

Reading of the band split:

- **18–33 % of the FAITHFUL cells have WRF ≥ 0, all in R0.** In-run, WRF also has **more** rounding-level
  negatives than the port (qc 7281 vs 5674, Ni 2310 vs 2013). In R0 the sign is set by rounding, so
  "same negative" means "same value within an ulp", not "same sign".
- **Above 1 eps·S (R1 + R2), every port negative has WRF's sign**, the amplitude agrees to ≤ 1.4e-4 relative
  (≤ 2.4e-5 in R2), and WRF has no extra negatives.
- **In-run qc (0–3.6 h) has no negative above rounding level.** The significant qc class (−3.955e-6 on the 12 h
  history frame; E213's −3.8e-6 class) is verified only on history-state operands (q_cur = q_old, physics
  sc = 0). An in-run qc claim needs run4's R2 qc cells.
- **The all-cell outliers are not port negatives.** They occur at FTZ-flushed subnormal cells (port 0 vs pristine
  ~1e-39, tiny S) and at the vertical edge of the 7×7×3 footprint. They vanish with the FTZ oracle plus 7×7×5:
  ≤ 11 eps·S, max |d|/field-max ≤ 3.2e-7 (critic).

## 4. Consequence for the open t-f2neg residuals [I]

t-f2neg found GPU/WRF extremum ratios of 1.14 (QI) and 1.39 (QG), and WDM6 count differences. Both came from
**independently evolved** 24 h trajectories.

At the verified sites, the operators that create or move those negatives reproduce WRF on identical operands. So
those ratios are most plausibly trajectory differences: different states entering the same faithful operator
chain. This is an inference, conditional on the class-(3) sites of §5.

## 5. Scope: classes per site

| site | class |
|---|---|
| Root/specified final-RK chain: hdiff, sixth, PBL physics sc, QV relax/spec, pre-add, PD advection, update, flow_dep (Swiss in-run 0–3.6 h, CPU) | (1) FAITHFUL |
| Same chain on history states: Swiss Thompson 8 frames, PROD d01 and d02 interiors, WDM6 6 frames (FTZ-matched), physics sc = 0 | (1) FAITHFUL |
| Nested-child production path (`_child_frozen_scalar` / nested_frozen_bundle pre-add with nested relax/spec of all species, `_physics_sc_outside_spec`, nest rings 0–6): PROD d02, WN3 d02/d03 | **(3) unverified** |
| KF cumulus sc_tend in the pre-add (PROD d01 cu=1; Swiss cu=0, probes set physics = 0) | **(3) unverified** |
| RK1/RK2 stage advection and update (q_cur is taken as an operand) | **(3) unverified** |
| WN3 cases | **(3) unverified** (not covered) |
| The port evaluated on GPU | **(3) unverified**; SASS shows no FTZ [I] |
| In-run significant qc (R2) | pending run4; history-state probes only so far |

Other limits:

- The PBL (MYNN) producers of the physics sc_tend are taken as operands, not re-verified here.
- The PROD windows are 48×48 crops classified at ring ≥ 7.
- The capture runs RRTMG with column tile 128 (§3.3, F5).

## 6. Reproduction

- **Oracle:** `pristine_chain.py` (`build(dir, 'off'|'fast')`, needs `mxcsr.c` for FTZ).
- **Probes:** `frame_probe.py` (env `O5_SPECIES`, `O5_CROP=auto48`, `O5_RING_MIN`, `O5_MUTANT`).
- **In-run:**
  - `capture.py` (taps; RAM slot);
  - `replay.py`;
  - `summarize_replay.py`;
  - `capture_smoke.py` (plumbing test, < 1 GB).
- **GPU FTZ census:** `gpu_ftz_scan.py`.
- **Results:** `results/` (manifests of both oracle builds, all probe/replay summaries).
- **Raw captures:** `<DATA_ROOT>/wrf_gpu2/scratch/o5-f2neg/run3` (deleted after final acceptance).
