# Radiation history (V044, numerical qualification passed; counter accounting incomplete)

Measured model source: `a8c2dceb4b` (branch `lane/w44-raddiag`, `src` byte-equal to `fffa38accd`). The later writer-only successor pairs QI_BL in explicitly selected QC_BL streams; its default path/payload is unchanged. Reference: pristine WRF 4.7.1 and its registered RE01/cal_cldfra savepoints. GPU-vs-GPU checks below are regression evidence only.

Use the WRF radiation schemes `ra_lw_physics=4,4,4` and `ra_sw_physics=4,4,4`, then add:

```fortran
&gpuwrf
 rad_diag = 1,
/
```

Omitting this option selects zero additional diagnostic carry arrays or diagnostic arithmetic. A non-RRTMG radiation pair refuses `rad_diag=1`. One scalar selects every nest; a list selects domains individually. Each selected domain holds the fields at its own radiation calls and writes them at its history alarms. The implementation uses the same per-domain writer on deeper nests; a separate 4-nest 333 m forecast has not been measured by this lane.

`QI_BL` is a default history addition next to `QC_BL`, approved by the manager at 17:15:03Z on 2026-10-23. A selected stream requesting QC_BL also includes QI_BL. It transcribes the existing State leaf; no additional prognostic/physics carry is allocated. Metadata is copied from Registry.EM_COMMON:1687. Lead-zero history uses WRF's pre-solve/input convention; no radiation solve occurs in the writer.

| Field | Units | Meaning/source |
|---|---|---|
| QI_BL | kg kg-1 | CLOUD ICE MIXING RATIO IN PBL schemes; existing current MYNN State leaf |
| RTHRATLW | K s-1 | UNCOUPLED THETA TENDENCY DUE TO LONG WAVE RADIATION |
| RTHRATSW | K s-1 | UNCOUPLED THETA TENDENCY DUE TO SHORT WAVE RADIATION |
| RTHRATLWC | K s-1 | UNCOUPLED THETA TEND DUE TO CLEAR SKY LONG WAVE RAD |
| RTHRATSWC | K s-1 | UNCOUPLED THETA TEND DUE TO CLEAR SKY SHORT WAVE RAD |
| QC_RAD, QI_RAD, QS_RAD, QG_RAD | kg kg-1 | gpuwrf diagnostic, no WRF equivalent: the cloud-water/ice/snow/graupel mixing ratios passed to RRTMG after radiation cloud mediation |

The four rate names/descriptions/units match Registry.EM_COMMON:1697–1700. All-sky LW and SW rates use the existing separate solver heating, converted by the same Exner factor used by the tendency coupler. The prognostic combined rate retains its original summation/conversion order. Clear heating reuses the already computed clear interface flux streams and pristine RRTMG's REAL heat-factor/pressure/86400 operation order; there is no additional radiative-transfer solve. The new clear diagnostic uses CP=7*287/2=1004.5 even under retained legacy solver flags. The old generic legacy heat helper used CP=1004, which the stricter export gate exposed.

`CLDFRA_RAD` is unnecessary: the held `radiation_diagnostics.cloud_fraction` is exactly `_from_columns(sw_state.cloud_fraction)`, from the same SW/LW input construction. Existing `CLDFRA` history transcribes that held field. The first history frame precedes radiation and therefore contains initialized zeros, including the new rate/input fields. Rain is not a condensate argument of this RRTMG column interface; QG_RAD completes its four condensate arguments.

Set WRF `&physics icloud_bl=0` to skip the radiation-only CLDFRA_BL/QC_BL/QI_BL merge. MYNN production/carry remains unchanged. This is a static selection in the cheap key. Default `icloud_bl=1` retains the released path. Like pristine WRF, `icloud=0/2/3` bypasses this whole merge regardless of `icloud_bl`. The pre-existing default first-call occupancy fraction is preserved for OFF identity; literal merge tests isolate that merge's input fraction, and existing cal_cldfra3 oracle tests qualify the `icloud=3` branch independently.

Restart schema `rad_diag_schema_version=1` (or NetCDF `GPUWRF_RAD_DIAG_SCHEMA_VERSION=1`) is mandatory when ON. Pickle carry, State/runtime checkpoint, native RestartStore generations and NetCDF carry readers reject missing schema, missing fields, changed selection, wrong REAL dtype/shape and nonfinite values. Native generations bind per-domain `rad_diag_config`; the OFF identity omits that new config entry. Initial optional arrays are seeded after precision enforcement as REAL32.

| Gate | Evidence/status |
|---|---|
| QI_BL writer/init/callback | QI01: [M CPU] 29 passed, 6 existing skips; Registry metadata/readback verified |
| Literal WRF BL merge | [M CPU] pristine source block compiled unchanged; 0/1 × first/later calls pass on real winter columns with active real-structured BL water/ice |
| LW/SW all/clear export | [M CPU] 186 registered real RE01 day/night columns, liquid/ice active; frozen solver heating bound retained. All-sky export byte-exact; clear export additionally bounded by reference REAL flux cancellation; zero-rate deletion sensitivity required |
| icloud=3 | Existing registered pristine cal_cldfra3 tests and real caller seam reused in consumer cohort |
| Restart/readback/nonfinite | Four public format round trips + missing-schema refusal; 24 NaN/Inf field controls; optional-field absence OFF |
| All touched writer consumers | [M CPU] 1787 unique passed, 132 existing skips after source-proven reuse and affected-caller delta. Unrelated full urban numerical module hit180s cap; its actual writer consumers pass. Selected-stream follow-up:52 passed,3 skipped |
| OFF Swiss/PROD identity | [M CPU native-facade trace] exact b977712f / 321aa651; full SHA in CPU_RECEIPT.json |
| Compiled ON/OFF fields | [M GPU] 4536 byte comparisons:378 shared history variables ×12 frames, all exact, including every emitted prognostic/land field. Scope is emitted history, not a claimed hash of every hidden carry leaf |
| GPU fields / icloud_bl=0 | [M GPU] OFF/ON/ON+icloud_bl0 each completes WN3 0227 three-nest3h,12frames, finite new fields and model; RTHRAT/Q*_RAD/CLDFRA/QI_BL readback sealed |
| E41 | [M actual GPU executables] PASS12entries, STACK/callee/HLO/declared-program coverage PASS; rich d01+fused_d02 artifacts for all3 selections |
| ON cost | [M GPU] timing/live-memory table below. [I] 8 REAL32 volumes =32 logical bytes per mass cell; no second solver. [U] extra kernels/copies pair: profiler attempt incomplete, not inferred from partial data |

The frozen GPU launcher uses the retained S0 timing harness with only `--max-dom 3 --emit-initial-history` substituted. Its input manifest hashes every manager-authorized staged WN3 input and every variant namelist. It records asserted GPU backend, cache/compile events, segment times, per-PID VRAM and process RSS. OFF/ON uses the same source and resolved work except the option. Full release RW1–RW4 and integration belong to the manager's composed candidate.

The short arm starts at 2026-02-28_00Z (night). All three options use explicit allocator fraction0.30 and the same source/work. Values below are measured on one sequential pair, with no A/A noise estimate. The third segment is the retained harness's last-call proxy, which omits its post-call tail. No whole-run warm24h S1 or CPU speedup claim is made.

| Selection | Second segment s/fc-h | Last-call proxy s/fc-h | Per-PID NVML peak MiB | JAX live peak MiB | Host peak RSS MiB, incl. compilation |
|---|---:|---:|---:|---:|---:|
| OFF | 5.725 | 5.598 | 10592 | 2746.09 | 12817.25 |
| rad_diag=1 | 6.642 | 6.115 | 10580 | 2979.86 | 13199.35 |
| rad_diag=1, icloud_bl=0 | 5.571 | 5.491 | 10580 | 3053.73 | 12982.76 |

The observed ON second-segment increase is0.917s/fc-h (+16.0%); JAX peak increases245131360bytes (+8.5%). The fixed pool hides this live-memory increase in the NVML reservation. These are cost observations, not an attribution to a kernel family or a day-case bound. The source adds no radiative-transfer solve. Kernel/copy attribution remains open: the 150s nsys cap expired during OFF capture/report cleanup, before ON began. An OFF report exists (482071 raw kernels,43113 raw memcpy rows), but no completed S0 profiling receipt or paired ON result exists; those counts are not promoted to a normalized comparison.

Receipts: CPU_RECEIPT.json, GPU_RECEIPT.json and `<USER_HOME>/wrf_gpu2_lanes/w44-raddiag/` (READBACK_SEAL.json contains every frame/variable byte hash and new-field metadata). The GPU was released before ALISIOS reacquired it at18:44:33Z. No release qualification, completed kernel/copy cost pair or every-hidden-carry byte proof is claimed.
