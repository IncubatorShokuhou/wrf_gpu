# Validation against original WRF — v0.4.0 and earlier

This page holds the detailed evidence behind the short validation summary in
the [README](../../README.md). The reference is always the original Fortran
CPU-WRF (or unmodified WRF source code for single routines). Comparisons
between two `wrf_gpu` versions only show that nothing changed unintentionally;
they are never used as evidence of agreement with WRF.

## How to read this page

The sections below keep the wording of the release records, which uses a few
short labels:

| Label | Meaning |
|---|---|
| D6 | The frozen per-field acceptance limits (for example 1.5 K RMSE for 2 m temperature, 120 Pa for surface pressure, 1 mm for accumulated rain, exact agreement for grid and terrain fields). They were fixed before the comparisons and are never loosened. |
| PASS / FAIL (raw) | The plain result of applying those limits. A raw FAIL stays a FAIL even when it is explained. |
| Composite, c00–c90 | One registered test configuration: a specific combination of physics options run as a 24-hour forecast (c45: six hours) and compared with CPU-WRF. The full list is the [composite ledger](FINAL_COMPOSITES_v0.4.0.md). |
| Inventory, NF0 | Every history field CPU-WRF writes is present, and no output value is NaN or infinite. |
| REPORT_ONLY | A field that is compared and reported but has no frozen limit; it is never counted as a pass. |
| E41 | An automatic check of the compiled GPU code for patterns that tend to be slow or fragile (large stack use, very large kernels). An E41 FAIL is a performance and robustness warning, not a wrong result. |
| F2 | Accounting of intermediate negative or out-of-range values that the model repairs during a step, reported separately from the output comparison. |
| F20, C5′ | F20: the 24-hour checks of the frozen default physics on five real cases (three Tenerife three-domain cases, a two-domain Canary Islands case and the Swiss Alps example). C5′: the repeat on three of them after the last change to the defaults. |
| WN3, PROD | The Tenerife three-domain (9/3/1 km) cases and the two-domain Canary Islands (9/3 km) case. |
| [M], [I] | Measured value, inferred or estimated value. |

Short hexadecimal strings are git commit identifiers of the exact source that
produced a result; they let anyone reproduce it.

## v0.4.0 release status at publication

**v0.4.0 — released 2026-10-17:** default programs are frozen as Swiss
`3356e249` and PROD d01 `196aed9f`. All 66 registered configurations pass
their accepted scientific scope: 65 full 24-hour cases plus Goddard radiation
c45's accepted six-hour case. The [source-bound ledger](FINAL_COMPOSITES_v0.4.0.md)
records every option tuple, clock and raw result. E41 resource failures,
c89's native-scan uncertainty and F2 limits stay disclosed. See the
[release record](../../release_notes/RELEASE_NOTES_v0.4.0.md) for tag targets.

## Features and validation classes

The **[v0.4.0 support matrix](SUPPORT_MATRIX_v0.4.0.md) is the
authoritative feature list**. It enumerates the physics selectors and optional
workflows, their admission rules, validation classes and source receipts.
This table is a summary, not a second per-option inventory.

| Area | Scope | Validation class and detail |
| --- | --- | --- |
| Default forecast suite | ARW; Thompson microphysics, RRTMG, MYNN, Noah-MP; Kain–Fritsch where configured; fixed nests, history and restart | Default-qualified at the frozen sources; [F20/C5′ evidence](SUPPORT_MATRIX_v0.4.0.md#frozen-defaults-and-results-so-far) |
| Microphysics | WSM/WDM, NSSL, Morrison, Milbrandt, Ferrier, Goddard, aerosol-aware Thompson, CAM-MG, P3, Jensen and NTU families | Per-composite D6 PASS, E41-disclosed, or incomplete; c11 now has full 24-hour D6 PASS with E41 disclosed. [Every code](SUPPORT_MATRIX_v0.4.0.md#mp_physics) |
| Cumulus, PBL and surface layer | Expanded Grell/SAS/KF/Tiedtke and PBL/surface families, with WRF pairing restrictions | Source-bound composite D6 and resource classes; [cumulus](SUPPORT_MATRIX_v0.4.0.md#cu_physics), [PBL](SUPPORT_MATRIX_v0.4.0.md#bl_pbl_physics), [surface](SUPPORT_MATRIX_v0.4.0.md#sf_sfclay_physics) |
| Radiation and land | RRTMG/RRTM/CAM/Goddard; admitted Noah-MP/RUC/SSiB, glacier, lake, ocean and urban branches | Exact composite scopes; c45 is six hours. FLG/GFDL are CPU-only; named land modes are API-only or refused. [Radiation](SUPPORT_MATRIX_v0.4.0.md#ra_lw_physics), [land](SUPPORT_MATRIX_v0.4.0.md#sf_surface_physics) |
| Optional online features | DFI, analysis/observation/spectral/surface nudging, two-way feedback, stochastic forcing, turbines, passive tracers, diagnostics, station profiles and regional streams | Individual composite, source/oracle or experimental scopes; [options and workflows](SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows) |
| Forecast tools | Offline weather/application products, station verification, ensemble products and preparation, doctor/provenance, onboarding, export, radar, optimisation and cycling | Tool/source/fixture qualification is separate from a coupled GPU forecast or improved station skill; [workflow classes](SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows) |
| Multi-GPU | Independent-case scheduling and experimental domain decomposition | Real multi-GPU scaling remains unqualified; [scope](SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows) |
| Refused selections | GPU km3, surface mosaic, active unsupported green-roof/PV/aerosol branches, moving nests and named chemistry/fire/hydrology/DA combinations | Refused, CPU-only or API-only as specified per selector; [matrix](SUPPORT_MATRIX_v0.4.0.md), [known issues](../../KNOWN_ISSUES.md) |

An admitted option is not arbitrary-combination approval. Single-domain
qualification does not extend to nests. E41 resource failures remain raw
FAILs even where the opt-in scientific gate passes. Refusals name the reason;
changing the namelist merely to bypass one changes the requested forecast.

## How v0.4.0 was validated

The reference is **original CPU-WRF or pristine WRF code**. GPU-vs-GPU
agreement provides regression evidence only. The release uses several checks
because an oracle-matched column, finite output and a completed forecast
answer different questions.

1. **Audit the real callers.** The caller audit recorded 24 findings and the
   release's 13 class-A repair areas. All 13 code repairs are merged. They restore WRF operands,
   physics cadence, precipitation/number carry and RK coupling. The
   [fix inventory](FIXES_v0.4.0.md) links the audit and integration
   receipts. A repair being merged did not close its affected GPU gate; c11 required
   its changed-source full 24-hour run, now PASS.
2. **Compare composed forecasts.** The [66-configuration ledger](FINAL_COMPOSITES_v0.4.0.md)
   compares the actual selected schemes with original CPU-WRF, using frozen
   strict D6 bounds and complete registered-scope field inventories. 65 cases span 24 hours; c45's accepted six-hour scope is explicit. Unbounded fields remain REPORT_ONLY. Missing fields,
   initial frames and nonfinite values are not relabelled into passes.
3. **Requalify the defaults.** F20 covers WN3 0227/0502/0614, PROD and Swiss:
   300 paired domain-frames over 24 hours. C5′ then requalifies Swiss, PROD and
   WN3-0227 with 150 paired domain-frames, full D6/inventory/NF0 and actual
   E41 PASS 3/3. Host-transfer counts/bytes, synchronization and callbacks
   equal F20. [Exact sources and identities](../../proofs/r_docs040/RELEASE_CHECKPOINT.json)
   keep the F20 and C5′ measurements distinct.
4. **Let the final runs find defects.** c06 exposed Tiedtke ROOT1–3; c40
   exposed the SSiB SWDOWN handoff; native MYNN scalar lowering exposed the F8
   passive-number interface; c11 exposed a pathological WDM6 ptxas fusion.
   [Before/fix/qualification evidence](FIXES_v0.4.0.md) preserves
   the original failed attempts and the affected reruns.
5. **Disclose residuals.** The [E41 inventory](E41_LEDGER_v0.4.0.md)
   lists executable resource failures; [known issues](../../KNOWN_ISSUES.md) retain
   intermediate negative-event limits, precision residuals, incomplete
   scopes and input restrictions. Zero output NaN/Inf does not justify
   intermediate repairs or establish global F2 clearance.

C5′'s accepted matched nsys interval is 58.05 versus 58.89 ms (−1.4% [M]).
It is a profiled root interval, **not whole-run S1 or a valid ABAB benchmark**.
No new v0.4.0 speed, energy, universal station-skill or all-variable PASS claim
is inferred from it.

## What the original-WRF comparisons show

[![c06 Tiedtke RAINNC RMSE before and after the repaired caller/kernel path](v040_figures/c06_tiedtke_rain.png)](v040_figures/c06_tiedtke_rain.svg)

**Validation caught a real bug.** Against the same original CPU-WRF and input
hashes, c06's worst hourly RAINNC RMSE falls from **1.111 to 0.0297 mm: 37.4×
smaller [M]** after Tiedtke ROOT1–3. Both complete 25-frame curves and the
unchanged 1 mm D6 limit are shown; this is a case-specific error metric.

[![66 configurations by62 bounded fields, with separate forecast RMSE and static-grid checks](v040_figures/composite_matrix.png)](v040_figures/composite_matrix.svg)

**The whole registered suite is visible.** Left: 10 forecast-field RMSE checks.
Right: 52 grid/static max-absolute-error checks, including 48 exact-zero rules.
Each cell is the worst gated hourly/domain statistic divided by its actual
frozen limit. c45 is explicitly six hours; the other 65 cases cover 24 hours. Unbounded fields
have no invented tolerance; E41 and F2 are disclosed separately.

[![WN3-0227 hourly differences across all three domains and ten forecast fields](v040_figures/wn3_deviation_bands.png)](v040_figures/wn3_deviation_bands.svg)

**A real three-nest forecast, all 24 hours.** Lines show full-field RMSE against
original CPU-WRF over 75 paired domain-frames; shading shows the spatial 95th
percentile absolute error. These are spatial distributions, not confidence
intervals or pointwise bounds; the remaining 5% can have larger errors.
[Plot data, sources, hashes and reproduction](v040_figures/README.md)
cover all three figures. No model was rerun to make them.

## v0.3.3: seventy-two-hour comparisons

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="evidence/v033/final_w3/stability_six_runs_dark.png">
  <img src="evidence/v033/final_w3/stability_six_runs.png" alt="v0.3.3 GPU versus original CPU-WRF: ten normalized RMSE panels, mean and spread across six Tenerife runs from three cases, all 73 paired hours">
</picture>

Six Tenerife runs come from **three independent cases**, including related IC members and a same-IC fresh-compile replica; they are not six independent samples. Every curve uses the fixed innermost **d03**, all native cells and all 73 paired hours. The thick line is the equal-run mean of stored RMSE divided by the unchanged frozen D6 limit. Light shading is the min–max envelope; dark shading is mean ± one sample SD (ddof=1, lower edge clipped at zero). These are descriptive run-to-run bands, not confidence intervals.

Solid grey is the empirical spread of **two original CPU-WRF 0227 runs with slightly different initial conditions** (intrinsic predictability context). Dotted grey is the **older GPU 0115 IC-pair proxy**, not CPU spread; its T/U/V/QVAPOR curves are unavailable and remain absent. No 0408 pair is available. IC-member comparisons include their IC difference from the admitted original CPU R reference. Monica72 and Swiss24 are separate linked comparisons, excluded from this aggregate. Accumulated RAINNC is included; hourly rain has no separate frozen RMSE bound. [Figure data and source hashes](evidence/v033/final_w3/STABILITY_MANIFEST.json) retain the method and exact captured inputs.

Ratios above 1 remain raw failures. In 0227, accumulated-rain RMSE rises from 1.10 mm at h67 to **2.465 mm at h72** (about 2.47× the 1 mm limit); all eight raw breaches are FLOOR-LIMITED under the frozen A3 annex, with original CPU-pair C/X = 0.81–0.95. The CPU runs differ almost as much; the annex explains these breaches without changing the raw scores or limit. The 0115 strict rain, late wind and cirrus differences remain separately disclosed below.

All eight original-CPU scoring readers are closed. The manager accepts the disclosed deviations under the Owner rule; raw failed checks remain failed.

[All final identity plots](evidence/v033/final_w3/index.html) show per-hour RMSE, bias and spatial bands against original CPU-WRF. Seven 72 h runs include six Tenerife primary/IC/replica arms and Storm Monica; the shipped Swiss case is 24 h.

| Comparison | Forecast | Strict 24 h (raw) | Full-window D6 (raw) | Interpretation |
|---|---:|---|---|---|
| Tenerife 0115 primary | 72 h | FAIL | FAIL | 61 floor-limited rows; one real V10 miss. Rain and cirrus lifetime differences disclosed (L4/L6/L7). |
| Tenerife 0227 primary | 72 h | PASS | FAIL | Rain reaches 2.465 mm at h72; all eight breaches are explained by the frozen CPU-pair floor annex (C/X 0.81–0.95; L5). |
| Tenerife 0408 primary | 72 h | PASS | PASS | D6 passes; trace graupel integrity miss disclosed (L1). |
| 0227 IC member | 72 h | PASS | FAIL | All six breaches pass the frozen CPU-pair spread annex; integrity annex passes. |
| 0115 IC member | 72 h | FAIL | FAIL | 54 floor-limited rows; strict rain FAIL retained. Integrity annex passes; IC difference included. |
| 0115 same-IC replica | 72 h | PASS | FAIL | Eight floor-limited rows, two real V10 misses (h42/h44). Cirrus lifetime integrity FAIL disclosed; same-IC fresh-compile replica. |
| Storm Monica | 72 h | PASS | PASS | D6 and all-frame integrity pass. |
| Shipped Swiss | 24 h | PASS | PASS | D6 and all-frame integrity pass; 25 frames on one domain. |

0227 stations classify CLEAR (raw DAMAGE retained). 0115 coastal wind-speed RMSE is about 1.8% higher; temperature, humidity and wind direction improve. The small persistent excess and late cirrus lifetime differences are disclosed, not renamed PASS. [Release notes](../../release_notes/RELEASE_NOTES_v0.3.3.md) and [technical appendix](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md) give the numbers, unchanged limits, IC/replica qualifications and source bridge.

Whole-run speed/energy figures above remain dated v0.3.2 measurements; no new throughput benchmark is inferred from this science wave.

Earlier W2 results remain available in the [source-dated historical gallery](evidence/v033/w2/index.html); their raw failures are retained.

## v0.3.3: limitations and what is not claimed

All-hour integrity retains its raw reports. A pre-registered trace annex explains
13 new rows; two disclosed manager judgements accept three canopy-ice and two
boundary-graupel rows above their unchanged caps. One canopy cell is reproduced
by the CPU IC twin; the other is a GPU-only placement departure. These are
**recorded exceptions**, not an unqualified clean-output claim.

W2 adds eight reviewed defaults, including urban soil coefficients, MYNN
predictor floors and horizontal SWDOWN history. Their original-WRF source
bindings are in the [fix inventory](V033_W2_FIXES.md).

BD92(c) remains a **known residual**, with binding d01 diagnostic maximum
0.0136 K; the non-binding late 0227 d03 departure is disclosed separately.
Night clear-sky cooling aloft is a v0.3.4 investigation. Dedicated glacier
runtime, universal physics pairing parity, MPI/multi-GPU domain decomposition,
whole-machine energy savings and data-centre GPU measurements are not claimed.

Earlier six-case 72-hour Tenerife and 162-hour Canary studies remain dated
v0.3.0 evidence, including their [raw frame failures](VALIDATION.md).
They are not relabelled as new v0.3.3 runs. The detailed
[floor, integrity and residual disclosures](V0.3.3.md) distinguish
accepted release rules from raw clean results. Original CPU-WRF is the fidelity
reference; comparisons with older GPU releases establish regression only.

## Where the GPU time goes (v0.3.1 profile)

WRF's ARW equations remain recognizable: third-order Runge–Kutta dynamics,
time-split acoustic integration, staggered-grid state and column physics.
JAX stages programs for fixed geometry and XLA compiles them; Pallas/native
GPU kernels handle hot stencils, vertical recurrences and column operations.
The Python interfaces retain the case/grid/state/I/O structure while the
expensive execution is rewritten for the device.

Precision is a performance decision with a fidelity gate. **WRF REAL work and
carry use fp32 fast kernels by default; WRF DOUBLE islands retain double
precision.** On the tested consumer GPU, fp64 arithmetic throughput is about
**1/64 of fp32**, inferred from [NVIDIA’s documented SM core counts](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capability-12-0) [I]. Moving an entire legacy array program onto a card therefore
does not automatically make it fast. The rewrite targets arithmetic precision,
kernel structure and memory traffic together; it does not simply disable
double precision globally or loosen comparison bounds.

The hot path combines larger device operations, keeps forecast carry on the
device and avoids repeatedly materializing intermediates. Column dependencies
need deliberate layouts and recurrences; radiation has different work from an
ordinary dynamics root. Stable executable reuse and buffer lifetime matter
alongside arithmetic. Init, load, history writing and process completion still
cost time, which is why the headline uses a complete forecast clock.

On the historical v0.3.1 Canary profiling program, ordinary-root device time is **30.393 ms** and
radiation-root device time **78.093 ms** [M]. A root represents one outer step
and its nested work; these are profiler diagnostics, not seconds per forecast
hour. Eight conservative kInput flags are bound to their actual PROD runtime:
maximum **0.049 ms**, ordinary-root sum **0.213 ms**, maximum stack **32 B**.
The flags are disclosed rather than called absent. The source/executable/options
binding accompanies the [v0.3.1 runtime evidence](V0.3.1.md); these
profile numbers are not transferred to the optimized v0.3.2 program as new measurements.

That historical ordinary kernel-plus-copy **mean is 33.109 ms/root**; the figures
above are kernel-only medians and should not be mixed. The latest accepted
[endpoint evidence](S4_ENDPOINT_RC3.md) preserves all measured time:

| Largest ordinary-root families | Device ms/root [M] |
|---|---:|
| Acoustics | 9.835 |
| MYNN | 4.028 |
| Advection | 3.304 |
| Carry/stage glue | 2.499 |
| Positive-definite scalar transport | 2.076 |
| Full kernel + copy mean | **33.109** |

Even deleting all d02 advection, glue and scalar-transport work would cap that
scoped device speedup at **1.27× [I]**. A further 2× needs changes across several
families: a common stable layout and cooperative spatial/column tiles keeping
acoustic coefficients, face fluxes and stage intermediates on chip are a
structural route to investigate. Such a gain is unproven. The traffic model
actually overcounts the measured total; it is not a calibrated bandwidth floor
or a claim of “90% hardware utilization.”

A roofline compares arithmetic capacity with memory bandwidth, but neither
number alone predicts a full forecast. Launch latency, dependent vertical
work, cache state, layouts, output and host scheduling can limit progress
before a theoretical roof is reached. The N=4 sweep reported median SM
utilization near 99% [M]; utilization is evidence of activity, not a proof that
every kernel is optimal. Performance changes still need measured profiles and
the same physical work, with correctness riding alongside.

