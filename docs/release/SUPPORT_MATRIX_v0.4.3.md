# Scheme and feature support matrix — v0.4.3

**v0.4.3, released 21 October 2026.** This page is the per-option record of what `wrf_gpu` v0.4.3 runs
and how each option was checked. It is technical on purpose; the
[release notes](../../release_notes/RELEASE_NOTES_v0.4.3.md) and the
[user guide](https://wrf-gpu.github.io/wrf_gpu/) say the same in plain words.

## What changed since v0.4.2

- Forecast model unchanged: the source of every program that can be compiled is identical to v0.4.2 except the version
  line (cheap-key trace closure 519 files, only `__init__.py` differs); release identities Swiss `b977712fe325` / Canary
  outer domain `321aa651067f` exact. The v0.4.3 compile bundle was compiled with the v0.4.2 autotune results and compared
  per program with the v0.4.2 bundle: all 21 programs identical (optimised HLO and native GPU binary), and all 716 release-check
  output files byte-identical.
- New in the run receipt: `host_suspend = {suspended_s, timing_valid}` in `nested_pipeline_run.json` (host clock check,
  outside the compiled model), a warning when the host slept during a run and a resume hint in the failure message.
- New evidence: three-nest 72 h qualification table (Tenerife 9/3/1 km, WN3 and ERA5 forcing, each against a CPU-WRF run
  from the same inputs; 2 of the 8 cases also have a CPU-WRF decomposition twin) and a status per reported item — see
  [Three-nest qualification (v0.4.3)](#three-nest-qualification-v043).
- Unchanged: every v0.4.2, v0.4.1 and v0.4.0 row below, carried forward without a re-run.

## Default program

**Unchanged from v0.4.2 (and v0.4.1).** Same release identities as v0.4.2 on the v0.4.3 pin; compiled programs and outputs identical (see above).
The v0.4.2 real-world checks and D6 results therefore hold for v0.4.3; the v0.4.3 release-day checks are in the
[release notes appendix](../../release_notes/RELEASE_NOTES_v0.4.3.md#appendix-real-world-release-checks).
Compiled release bundle: `d2f4f197b8affdecc0c297faae3593eb9d4df658cc3a58fc24f6523518aeee70` (version 0.4.3).

## Three-nest qualification (v0.4.3)

Eight 72-hour forecasts on the 9/3/1 km Tenerife set-up (six WN3, two ERA5) against the original CPU-WRF 4.7.1
from the same inputs, under a rule fixed in advance: **4 qualified** (27 February, 14 June, 8 June, 20 January 2026),
**4 not qualified** (2 May and 20 February 2026, ERA5 2 March and 18 February 2024). D6 (62 bounded fields only), 3 domains × 73
frames: 23 of 24 case-domains PASS and 1 FLOOR-LIMITED (27 February d03: RAINNC τ72 X/C 1.14, W τ69 X/C 1.20 under
its own valid CPU twin, k = 4/3; strict τ0–24 with 0 breaches; cases without an own twin had no breach). The four
failures are file-integrity items where the GPU field is exactly zero and CPU-WRF holds a trace amount (cause open);
they remain FAIL. D6 says nothing about the variables without a frozen limit. 10 m wind after hour 40: held. Full table:
[release notes](../../release_notes/RELEASE_NOTES_v0.4.3.md#three-nest-qualification-table); per domain and field class:
[`proofs/v043/w43-teacher/TEACHER_QUAL.md`](../../proofs/v043/w43-teacher/TEACHER_QUAL.md).

## How to read this page

| Term | Meaning |
| --- | --- |
| Swiss | Swiss Alps 3 km single-domain case (the bundled example, 15 January 2023) |
| Canary (PROD) | Canary Islands 9/3 km two-domain case, 25 July 2026 |
| Tenerife (WN3) | Tenerife 9/3/1 km three-domain cases (0227 = 27 February 2026 18 UTC start) |
| D6 | The frozen field tolerances fixed before the v0.4 comparisons (for example T2 RMSE ≤ 1.5 K, PSFC ≤ 120 Pa, RAINNC ≤ 1 mm, U/V ≤ 1.8 m/s) applied to every hourly frame of a GPU run against an original CPU-WRF 4.7.1 run from the same inputs |
| E41 | Code-quality scan of the compiled GPU program: per-thread stack ≤ 256 B, no non-leaf device function > 1,000 SASS instructions, no transposing (kInput) fusion > 500 optimized HLO instructions. A FAIL does not mean wrong results; it flags a possible speed/robustness cost that has not been measured |
| NF0 | No non-finite value in any output field |
| F2 | Pre-mask accounting of negative or repaired values inside the step; reported separately, never cleared by a D6 pass |
| Oracle | The verbatim WRF 4.7.1 Fortran routine, built from the pristine source, fed the same operands; "byte" means bit-identical output |
| Experimental | Opt-in, implemented and checked as stated, but outside the validated scope (no passing 24 h comparison, or a single configuration only) |
| Refused | `gpuwrf run` stops before any GPU work with a message naming the setting |

A pass covers the stated configuration only; it does not qualify every combination or
nested use unless the row says so.

## Default program as released with v0.4.2

**Unchanged from v0.4.1.** `proofs/s1_port/release_identity.py` gives Swiss `b977712fe325…` and
Canary outer domain `321aa651067f…` on every v0.4.2 merge (`SWDDIR` follow-up, `icloud=0`, WDM6,
nested-PBL catalog notes, tools) and on the release pin. With `GPUWRF_SWDDIR_V040_SPLIT=1` the
identities are Swiss `e1102c204a79…` / Canary `9b676bec1609…` (= v0.4.1 without the `SWDDIR`
fix). The only intended default-output change is history `COSZEN` (writer, not the step program);
the new compile changes other fields at round-off level (above). The 24-hour D6 requalification of
the v0.4.1 default program therefore carries as program evidence, and D6 is re-checked; the release-day
real-world checks of v0.4.2 are in the [release notes appendix](../../release_notes/RELEASE_NOTES_v0.4.2.md#appendix-real-world-release-checks)
(RW1–RW4 PASS: RW3 72 h D6 d01/d02 PASS 73/73, d03 raw FAIL RAINNC τ69–72 + W τ69, all 5 FLOOR-LIMITED under frozen A3 → binding PASS, 0 REAL FAIL (v0.4.1: 4); output integrity d03 QGRAUP τ72 degenerate flag = REGISTERED EXCEPTION (manager 2026-10-19T16:49:38Z, CPU-WRF twin trips the identical flag; census τ70 154 vs 217 cells); RW3T interleaved 24 h pooled d03 stepping N/O 0.9356 + S 0.0819 = 1.018 ≤ 1.10 (prereg 1305af20); 72 h pair (watch_cover missed, disclosed) whole 6.757 vs 7.148 s/fc-h, VRAM 5,236 vs 5,268 MiB, RSS 5.92 vs 5.91 GiB, compile 612 vs 628 s; icloud=0 QUALIFIED on the bundle; E41 STACK + coverage PASS, gate INCOMPLETE (slim AOT, as v0.4.1); receipts proofs/w42_bundle042/receipts/bundle_b979188128e0/RW_TABLE_FINAL.json). Compiled release bundle: `bundle_sha256 bfa2dc90c034e6e0…16d2` (commit `b979188128`, version 0.4.2; not published, delivered to the operational user).

## New and changed in v0.4.2

### Namelist options and environment switches

| Option | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `icloud=0` (no clouds in radiation) | Experimental; **GPU 24 h D6 PASS** (Tenerife 3 domains); E41 PASS | WRF 4.7.1 `module_radiation_driver.F:1309–1318`: `CLDFRA=0`, no `icloud` branch (no `icloud_bl` merge); RRTMG LW/SW skip every cloud input (`ICLOUD .ne. 0` guards). Unchanged pristine `radiation_driver` on 16 real WN3 0227 columns (`icloud` 0 and 1): `CLDFRA`=`CLDT`=`LWCF`=`SWCF`=0, all-sky == clear-sky; port release path worst 0.0012 of the frozen RRTMG bounds (`icloud=1` control 0.002); 6/6 seam mutants. GPU (RTX 5090) vs original CPU-WRF 4.7.1 `icloud=0` (dmpar, 4 ranks), WN3 0227 3-nest 24 h, release defaults with the `SWDDIR` fix: D6 PASS d01–d03, 25 frames, 0 failures, NF0 (d03 worst-hour T2 0.070 K, PSFC 2.4 Pa, U10/V10 0.10/0.13 m/s, RAINNC 0.003 mm; d01/d02 T2 0.012/0.011 K); `CLDFRA` 0 in every forecast frame of both runs, `SWDNB`==`SWDNBC` and `GLW`==`LWDNBC` exact in both. E41 PASS 4/4 (STACK 32 B). Validation reuse: the merged program traces the same `icloud=0` d01 and fused d02/d03 programs as the measured tree (location-stripped jaxpr + constants identical). Under `icloud=0` the `SWDDIR` fix is inert (pre-fix beam bit-identical on 16 columns). No new state; radiation cost = RRTMG (cloud-fraction step skipped); own compiled program (`icloud` in the static options). Same refusals as `icloud=2/3` (RRTMG 4/4 only; `mp_physics=5/11/50–55/85/95`, `cu_physics=7/10`); `cu_physics=11` keeps `icloud=1`. Bundle arm: QUALIFIED on the sealed v0.4.2 bundle (own programs d01 `ae4b9d39…` / fused `73e33b0a…`, primed 0227 24 h, replayed 0502 3 h with compiled_new 0; D6 3/3 PASS, 25 frames, d03 worst-hour T2 0.070 K, U10/V10 0.097/0.130 m/s, PSFC 2.3 Pa; CLDFRA 0 both runs; E41 STACK + coverage PASS). | [oracle and forecast](../../proofs/w42_icloud0/), [radiation-cloud-fraction.md](../radiation-cloud-fraction.md), [tests](../../tests/v041/test_icloud0_oracle.py) |
| `GPUWRF_SWDDIR_V040_SPLIT=1` (environment, read at import) | Opt-out, continuity only; not WRF | Restores the v0.4.0 delta-scaled direct-beam transmittance in the RRTMG SW band-sum kernel: its own branch replays v0.4.0's transmit pass and equation order; PTX and SASS of the compiled kernel equal the v0.4.0 kernel on 4/4 variants (CPU `triton.compile` for sm_120; the default kernel equals v0.4.1 4/4). Flag on equals the v0.4.0 split bitwise (45/45 + 18/18 arrays); on vs off differ only in `SWDDIR`/`SWDDIF` and what slope radiation derives from them. Part of the AOT cheap key; refused with `aer_opt=1/3`. Identities Swiss `e1102c204a79…` / Canary `9b676bec1609…`. No GPU forecast run with the switch in this release (needs its own cold compile and E41 if used). 10 tests, 8/8 mutants. | [tests](../../tests/v025/w42_swdfid/test_swddir_v040_optout.py), [AI_OPERATOR note](../../AI_OPERATOR.md) |

### Default output

| Item | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| History `COSZEN` | Merged (writer only) | WRF writes the held radiation-call `coszen` (`calc_coszen` at call + radt/2, exact `STEPRA` call time); v0.4.1 wrote the output-instant value. Nested writer (used by every CLI run) now takes `carry.radiation_diagnostics.coszen` on non-initial frames (0 at lead 0; output-instant only as a fallback). Producers RRTMG, Goddard, FLG/GFDL, SSiB hold the same value. CPU-WRF 0420 fixture (d01/d02/d03, 73 hourly frames, every 4th cell): max 1.8e-7 / 2.4e-7 / 2.4e-7 (output instant: up to 0.047/0.057/0.057). 23 tests incl. non-finite regressions; 4/4 mutants. Writer helper 3.0 → 0.015 ms per call (CPU). The replay/daily path keeps the output-instant fallback (not a product path). | [tests](../../tests/v025/w42_swdfid/test_history_coszen_held.py), [fixture receipt](../../proofs/w42_swdfid/coszen_fixture_receipt.json) |

### `SWDDIR` follow-up (#146)

| Item | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| v0.4.1 `SWDDIR` fix, caller path | Audited; fix stays default | WRF 4.7.1 path traced: `RRTMG_SWRAD` `swddif = swdflx − zbbfddir` from `ztdbt_nodel` (unscaled), radiation driver `SWDOWN = GSW/(1−ALBEDO)` and held diffuse fraction, `TOPO_RAD_ADJ_DRVR` every step with `slope_rad=1`, post-LSM swap (`SWDOWN` flat, `SWNORM` slope). Port matches at each step. WN3 0420 72 h, d03 south-slope focus (04-22 12–15 UTC, where F = v0.4.1 looked farther from CPU-WRF than B = v0.4.0): exact flat/slope decomposition, flat-term RMS 169–257 vs slope-term 4–15 W/m²; clear-cell slope factor F − C mean 7e-6, RMS 0.0033 (15,611 cell-h); matched-clearness diffuse fraction F − C +0.0018 vs B − C −0.029. Pre-registered repeat F′ (rule `fprime_rule.py` sha 6e51aaf6) INCONCLUSIVE: focus RMS 140–173 (F 170–269) W/m², land afternoon T2 residual −0.009…−0.023 K in the four cells (fix pair − v0.4.0 pair −0.030 K mean) → tracked (spec R4). ALISIOS closed #146 on 2026-10-19. A regime gate (slope/cloud and precipitation-noise-floor rules) is specified, not implemented; D6 unchanged. | [spec + repeat verdict](../../proofs/w42_swdfid/) |

### Speed (output byte-identical)

| Option | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `mp_physics=16` WDM6 (helpers shared by WDM5 `14`, WDM7 `26`) | Merged; v0.4.0 D6 rows carry (bytes identical) | Rain/Nr fall-out loop `while (n < 64) & (n < mstep)` instead of 64 fixed trips; vectorised first-hit PLM remap search instead of host-synchronised nested while loops. CPU: 210/210 arrays byte-identical (f32 + f64; 6 frozen WDM6 savepoints + real c11 WDM6/WDM7 states), 10 new tests, 3/3 mutants. GPU (RTX 5090, one lock session, interleaved): c11 (Swiss d01, mp16/cu96/bl17) stepping 18.29 → 6.00 s/fc-h (3.05×; A/A 0.02 %, B/B 0.38 %), 3 h warm wall 71 → 34 s, wrfout byte-identical to v0.4.1 cold and warm, VRAM 5,386 → 5,384 MiB; deviceless compile 482 → 256 s; static launches per step 54,839 → 19,217. E41: identical pre-existing hard FAIL set (STACK 512/416/408 B), conservative flags unchanged (one native size 1,192 → 1,224 SASS, disclosed). WDM5/WDM7 GPU speed not measured. Pre-existing 64-sub-step cap (WRF: none) unchanged: binds only if vt·dtcld/dz ≥ 63.5 at the drop-size clamp (dtcld 120 s: dz < 22.4 m); c11 max real `mstep` 5 (CPU-WRF) / 4 (GPU), clamp worst case 6. | [tests](../../tests/test_w42_wdm6_fallout_loops.py) |

### Nested boundary-layer qualification

| `bl_pbl_physics` | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| 1 YSU, 11 Shin-Hong, 7 ACM2, 16 EEPS (with `sf_sfclay_physics=1`) | **Nested 24 h D6 PASS** (one date, one set-up); E41 of the nested programs not measured | ALISIOS G2 GPU runs C2/C3/C4/C7 (v0.4.1 release bundle `24a7b24f`, release defaults), WN3 Tenerife 9/3/1 km issued 2026-07-07 18 UTC, 24 h from 2026-07-08 00 UTC, vs original CPU-WRF 4.7.1 (dmpar, 3 ranks) twins of the same namelist and inputs; `scripts/wn3_score.py` with the frozen D6 manifest, 25 frames per domain: 0/62 bounded fields over the limit on every domain, NF0, full CPU inventory, no GPU-constant field where CPU-WRF varies, all global attributes. Worst ratio 0.110 (EEPS V10 d03). Worst-hour T2 d01/d02/d03: YSU 0.025/0.015/0.036, Shin-Hong 0.021/0.013/0.043, ACM2 0.019/0.022/0.062, EEPS 0.014/0.033/0.061 K. 300 GPU + 300 CPU frame SHAs matched the published manifests. v0.4.2 changes only catalog notes for these schemes; the single-domain E41 FAIL of all four (v0.4.0) remains. KEPS (17) and the other boundary layers: single domains only. | [page](NESTED_PBL_QUALIFICATION_v0.4.2.md), [summary](../../proofs/w41_nestqual/NESTQUAL_SUMMARY.json) |

### Tools

| Command / tool | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `gpuwrf aot-key` | Changed: writes no persistent cache | Case load runs with `JAX_COMPILATION_CACHE_DIR`/`GPUWRF_JAX_CACHE_DIR`/`TRITON_CACHE_DIR`/`CUDA_CACHE_PATH` and the JAX cache config redirected to a private temporary directory, deleted on exit, exception, `SIGTERM` and `SIGINT` (signal re-raised); keyed settings (`GPUWRF_CACHE`, `GPUWRF_XLA_AUTOTUNE_CACHE_DIR`) untouched; a stderr warning per cache setting inside a sealed bundle. Real WN3 0227 under a sealed-bundle environment: keys equal to the v0.4.1 records for 0227/0502/0614/48 h inputs incl. components, bundle 15/15 paths identical + `BUNDLE_OK`; v0.4.1 tool in the incident environment wrote 3,076 paths, the fixed tool 0; Canary keys equal to the v0.4.1 tool. Single-domain cases keep the v0.4.1 refusal (rc 1) and write nothing. 17 tests, 4 mutants. | [receipts](../../proofs/w42_aotkey/), [tests](../../tests/test_aot_key_cli.py) |
| `scripts/wn3_score.py` (D6 + integrity scorer) | Changed: bounded memory, non-finite reference fails | One frame pair open at a time, variable batches by a byte budget, one float32 accumulator per field; `--sequential-domains` default above 24 h or 3 domains. 72 h WN3 0420: peak 2.3–2.4 GB (before 15–17 GB per arm), 10/10 result files byte-identical; 24 h 0227: 7.7 → 4.2 GB (parallel) / 1.7 GB (sequential), byte-identical. Non-finite CPU-WRF values now give `nonfinite_cpu` failures (before: silently dropped; 0 found in 0227 24 h and 0420 72 h). 8 + 3 mutants. | [summary](../../proofs/w42_scorer/SUMMARY.md), [tests](../../tests/test_wn3_score_finite_gate.py) |
| Benchmark harness (`scripts/bench/s0_nested_harness.py`, `scripts/bench_summary.py`, `scripts/wn3_forecast.py`) | Changed (benchmark tooling only) | The v0.4.1 harness initialised the GPU before the CLI's C-auto pool step, so measured processes ran with on-demand allocation. Now backend-dark until that step; receipts attest the pool; `bench_summary` rejects warm arms without C-auto. GPU session (one lock, 10/10 rc 0, 0 compiles, wrfout byte-identical across arms and to the v0.4.1 bundle CLI run): PROD stepping N/O 0.938 (A/A 0.41 %, B/B 0.79 %). Disclosed by the critic: two new-harness arms hit disk contention and were excluded after the session; over all arms whole-run N/O 1.391 (24 h) and proxy N/O 1.203 (3 h); the v0.4.1 PROD 24 h whole-run baseline 4.600 s/fc-h is ONE arm. WN3 par3 24 h (0227/0502/0614): 6.262 s per case-hour = 19.7× CPU-WRF 3×4 cores (123.385). | [session table](../../proofs/w42_harness/receipts/S1H_TABLE.md), [verdict](../../proofs/w42_harness/receipts/S1H_VERDICT.json) |

### Known open items found in v0.4.2

| Item | Status | Evidence |
| --- | --- | --- |
| Heavy-rain cells (≥ 10 mm), WN3 0420 d03 | Open (#149); systematic at 48/72 h by a pre-registered rule, about −12…−15 % vs the CPU-WRF mean; domain totals not systematic; v0.4.0 = v0.4.1; one day | [Known issues](../../KNOWN_ISSUES.md#v042-heavy-rain-cells-on-the-1-km-tenerife-domain) |
| CPU-WRF vs CPU-WRF RAINNC | Two CPU-WRF runs (2×2 vs 4×1 decomposition) differ by RMSE 1.20 mm at 17 h on d03 (> the 1 mm D6 bound); D6 unchanged | [Known issues](../../KNOWN_ISSUES.md#v042-the-1-mm-accumulated-rain-limit-and-cpu-wrf-itself) |
| Pre-existing CPU test failures | `test_v040_urban_slucm_wiring.py::test_catalog_and_validation_accept_slucm` (catalog text pin) and `tests/a1_da/test_member_driver.py::test_process_aot_reuse_shares_only_blob_loaded_executables`; identical on the v0.4.1 tag; 0 new failures on the release candidate | [Known issues](../../KNOWN_ISSUES.md#v042-two-cpu-tests-that-also-fail-on-v041) |

## Carried forward from v0.4.1: new and changed in v0.4.1

### Namelist options

| Option | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `topo_wind=1` (Jiménez–Dudhia, per domain) | Experimental; **GPU 24 h D6 PASS** in three set-ups; E41-disclosed | WRF `start_em` CTOPO/CTOPO2 byte-identical to pristine WRF 4.7.1 on Tenerife d01–d03 and Swiss terrain (every branch); YSU with real CTOPO/CTOPO2 byte-identical (literal) and inside the WRF ULP envelope (compiled) on 147 real columns; Shin-Hong on 80 rows; 12/12 mutants killed. GPU vs original CPU-WRF, 24 h, release defaults, 25/25 frames D6 PASS: Swiss YSU + `sf_sfclay_physics=1` (worst-hour RMSE T2 0.479 K, U10/V10 0.464/0.534 m/s, PSFC 3.34 Pa), Swiss Shin-Hong (T2 0.258 K, U10/V10 0.118/0.123 m/s), Tenerife 0227 d02/d03 YSU one-way nests (d03 T2 0.092 K, V10 0.300 m/s, PSFC 4.04 Pa). WRF's effect is reproduced (Swiss land mean WS10 change −0.283 m/s GPU vs −0.316 m/s CPU-WRF). Acts only with YSU (`bl_pbl_physics=1`) or Shin-Hong (`11`), as in WRF; other PBLs run unchanged with a warning (MYNN with `topo_wind=1/2` traces the release program). Needs `VAR_SSO` in `wrfinput` (refused otherwise, as WRF `start_em.F:222`); the tested cases carried `VAR_SSO` from a geogrid run with the `varsso` data. Two-way feedback of CTOPO/CTOPO2 is wired and CPU-tested only. **E41 FAIL** on every arm: STACK 400 B (Swiss YSU) / 496 B (Tenerife fused d02/d03) in the revised-MM5 surface layer that `topo_wind=1` re-fuses (same-source `topo_wind=0` control 8 B); Shin-Hong 24 B; conservative kInput > 500 flags on all arms. Runtime cost not measured. | [GPU arms](../../proofs/w41_topowind/GPU_ARMS.json), [oracle](../../proofs/w41_topowind/CTOPO_ORACLE.json), [mutants](../../proofs/w41_topowind/MUTANTS.json), [namelist notes](../namelist-compatibility.md) |
| `topo_wind=2` (Mass–Ovens, per domain) | Experimental; **GPU 24 h D6 PASS** on one domain; E41-disclosed | Uses geogrid `VAR`; CTOPO2 = 1. GPU D6 PASS only on the Tenerife 0227 d01 (9 km) YSU domain (worst-hour T2 0.016 K) plus the pristine YSU oracle columns. E41 on that program: STACK 88 B, conservative kInput flags. Same PBL restriction as `topo_wind=1`. | as above |
| `aer_opt=1` (Tegen climatology in RRTMG SW) | Experimental; **GPU 24 h D6 PASS** (Tenerife 3 domains); E41 scan INCOMPLETE | Pristine chain `aerosol_in` → `aer_time_int` → `aer_p_int` → `RRTMG_SWRAD` (iaer=6): unit chain bitwise (2,703 points, 520 julians, 37 columns); SW release path inside the frozen tier-1 RRTMG bounds on 186 real Tenerife columns × 5 dates and a ×10 stress arm. Climatology shipped as a SHA-pinned 6.0 MB asset (no wrfinput field needed). Nests receive the parent's `aerod` by SINT, as WRF Registry `df=(p2c)`. GPU vs original CPU-WRF `aer_opt=1`, Tenerife 0227 3-nest, 24 h: D6 PASS d01–d03, complete field coverage (worst-hour T2 0.04/0.03/0.08 K; clear-sky surface SW RMSE 0.15/0.16/0.35 W/m² against a 20–23 W/m² aerosol effect). E41 measured on a 3 h re-compile of the same tree (the 24 h binary was lost): STACK 32 B, no flags, scanner verdict INCOMPLETE (slim AOT metadata). Needs `ra_sw_physics=4` on every domain; LW carries no aerosol. Nested: 6·nz·ny·nx REAL carry per domain. | [AEROSOL.md](../AEROSOL.md), [AE01](../../proofs/w41_aerosol/ae01_summary.json) |
| `aer_opt=3` (Thompson mp28 aerosol optics) | Experimental; **GPU 24 h D6 PASS** (Swiss); E41-disclosed | Pristine `gt_aod` + `calc_aerosol_rrtmg_sw` + `RRTMG_SWRAD` (iaer=10): `gt_aod` bitwise, band optics ≤ 3e-6 relative; 5 of ~157,000 cells in one ill-conditioned column judged against WRF's own one-ulp ensemble (AE09). GPU vs original CPU-WRF `aer_opt=3`, Swiss 42×42 mp28, 24 h: D6 PASS, all 369 history fields in all 25 frames (worst-hour T2 0.23 K, U10/V10 0.51/0.56 m/s, RAINNC 0.81 mm vs 0.80 mm for the aerosol-off control); `TAOD5502D` within 1e-4. Needs `mp_physics=28` (single domain, `wif_input_opt=1`) and `ra_sw_physics=4`. **E41 FAIL** on the conservative kInput rule: three fusions (658/879/1,370 HLO instructions) from pre-existing Thompson mp28 source families (the v0.4.0 mp28 aerosol-off run had two, 741/1,439; not byte-identical kernels); STACK 16 B, non-leaf 0; the aerosol code adds no flagged fusion. | [AEROSOL.md](../AEROSOL.md), [AE07](../../proofs/w41_aerosol/ae07_summary.json), [AE09](../../proofs/w41_aerosol/ae09_summary.json) |
| `aer_opt=2` | Refused | `aer_type`/AOD550 inputs are not ported. | [Catalog](../../src/gpuwrf/io/scheme_catalog.py) |
| `icloud=3` (`cal_cldfra3`) | Experimental; **GPU 24 h D6 PASS** (Tenerife 3 domains); E41 PASS | Literal REAL4 port byte-identical to pristine WRF 4.7.1 on 640 real columns (0 of 84,480 values differ); compiled port inside WRF's FMA/one-ulp envelope; XLA `log10` ≤ 15 ulp (≤ 1.2e-7) in the small-condensate branch, disclosed; 9/9 mutants. Replaces WRF's whole `icloud=1` block including the `icloud_bl` merge, in WRF order; history `CLDFRA` = radiation input. GPU (RTX 5090) vs original CPU-WRF `icloud=3`, Tenerife 0227 3-nest, 24 h, release defaults: D6 PASS 3/3 domains, 75 frames, 0 failures (d03 worst-hour T2 0.088 K, PSFC 2.5 Pa, U10/V10 0.25/0.33 m/s, RAINNC 0.05 mm); cloudy share at +24 h CPU/GPU 4.04/4.05 %, 3.58/3.60 %, 8.51/8.52 %, pattern correlation 0.97–0.99 (`CLDFRA` has no D6 bound). E41 PASS (max STACK 72 B). Cost 0.8–1.1 ms per radiation call and domain, ≤ 114 MB scratch; no new carry. RRTMG LW/SW 4/4 only; refuses `mp_physics=0/1/3`. | [radiation-cloud-fraction.md](../radiation-cloud-fraction.md), [oracle tools](../../proofs/w41_icloud/) |
| `icloud=2` (`cal_cldfra2`) | Experimental; **GPU 24 h D6 PASS** (Tenerife 3 domains); E41 PASS | Byte-identical to pristine `cal_cldfra2`. GPU vs original CPU-WRF `icloud=2` (gen2 dmpar WRF 4.7.1, 4 ranks), Tenerife 0227 3-nest, 24 h, release defaults on merged main `93346f7979` (incl. the `SWDDIR` and boundary-record changes): D6 PASS d01–d03, 25/25 frames, 375/375 fields, 0 non-finite (worst-hour d03 T2 0.084 K, PSFC 3.69 Pa, U10/V10 0.24/0.29 m/s, RAINNC 0.029 mm; d01/d02 T2 0.010/0.015 K). `CLDFRA` binary in both arms; +24 h cloudy share CPU/GPU 0.68/0.67, 1.11/1.11, 4.10/4.05 %, pattern correlation 0.93–0.95. E41 PASS 4/4 programs, max STACK 32 B, no flags; `cal_cldfra2` 0.03 ms per call and domain. Same restrictions as `icloud=3` except the `mp_physics=0/1/3` refusal. Both options also refuse `mp_physics=5/11/50–55/85/95` and `cu_physics=7/10`; `cu_physics=11` keeps `icloud=1`; `icloud=0` was not bound in v0.4.1 (bound in v0.4.2, above). | [radiation-cloud-fraction.md](../radiation-cloud-fraction.md) |

### Microphysics change

| Option | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `mp_physics=14` WDM5, `26` WDM7: CCN number (`Nn`) small-grid compiler barrier | CPU component BYTE; GPU 200-step smoke finite; E41 still FAIL | Only the `Nn` argument of the WDM5/WDM7 adapters passes through the existing small-grid barrier (grids ≤ 4096 columns; larger grids trace no extra operation), as the v0.4.0 WDM6 fix. CPU: 68 component checks byte-identical, 336/336 shared-consumer tests, default identities exact. CPU byte equality is a component regression, not a GPU-byte or WRF-fidelity proof: the barrier can repartition GPU floating-point operations. The 24 h D6 rows below are carried-forward v0.4.0 evidence; no changed-artifact forecast was compared. Compiler (CPU, real composite programs): WDM5 875-instruction kInput transpose and WDM7 1,211-instruction transpose removed; temps identical; ptxas 13.2→13.3 s (WDM5) and 22.1→24.2 s (WDM7). GPU: 200 steps finite on both (observed pre-mask NF0). Residual E41 FAIL: WDM5 STACK 536/400 B, WDM7 625-instruction transpose / 368 B. F2 open (WDM7 hail `qh` is not in the pre-mask census). No speed, compile-time or VRAM claim; the v0.4.0 24 h D6 rows below are unchanged evidence. | [README](../../proofs/w41_wdm57/README.md) |

### Tools and workflows

| Command / option | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| `gpuwrf da check`, `da analyse` | Experimental, opt-in, CPU analysis, d01 only | Offline LETKF (U, V, T, QVAPOR; MU fixed). Analytic Kalman limits, Lorenz-96 twins, moisture budget and rebalance oracles, 41/41 mutants; real-file end to end on a guarded development case (12 members, 108 of 392 sonde values assimilated, T O−B 1.90 K → O−A 1.24 K, no negative or added water); the written `wrfinput_d01` passes the case preflight. Development cases (descriptive, not confirmatory): pooled 12 h station MSE T2 −15 %, Q2 −29 %, U10 −27 %, V10 +6 %. Forecast skill not established. Not supported: nests, MU analysis, cloud/radiance observations, `supersaturation` other than `keep`. ALISIOS data only through the governance plug-in. | [docs/da.md](../da.md), [qualification](../../proofs/w41_da_core/QUALIFICATION.md) |
| `gpuwrf da` ensemble options (`inflation`, perturbed boundaries, variable localisation) | Experimental; calibration FAIL disclosed | Lorenz-96: RMSE 0.608 vs 0.751, spread/RMSE 1.03. Swiss CPU-WRF twin (20 members, analyses 03/06/09 UTC): analysis error 39 % below the old design; **overdispersive**: prior spread 2.59/2.32/2.53 × actual error at 06/09/12 UTC against the registered band 0.5–2.0 (FAIL); not retuned. Not run on the GPU member engine or with real observations. | [docs/da_ensemble.md](../da_ensemble.md), [design](../../proofs/a1_da/ens/ENS_DESIGN.md), [Swiss twin](../../proofs/a1_da/ens/results/SWISS_OSSE.json) |
| `gpuwrf da cycle`, `da members` | Experimental, opt-in, d01 | All members of a window in one process on one shared step program; zero-update control arm and LETKF arms; file seam or restart seam. CPU (Swiss, 2 members × 2 cycles): zero-update restart arm 7/7 equal to the continuous run, LETKF file seam 4/4 equal to separate runs, repeatability 8/8; 35/35 mutants. GPU (Canary outer domain 120×70, RTX 5090, 10 members × 2 windows × 6 h, zero-update arms): 28 member runs completed; the compared histories byte-identical to separate fresh processes (member m01 window 0 7/7, window-1 file seam 7/7, restart seam 6/6; other members not byte-compared); first member 415 s (cold), later members 11.0–14.1 s vs 38 s for a separate warm process; host RSS flat 5.67 GiB. No LETKF analysis inside the GPU loop yet. Refuses any input under protected (ALISIOS) roots. Disk ≈ 12–18 GB for 10 members × 2 Canary windows. | [docs/da-cycle.md](../da-cycle.md) |
| `gpuwrf run --surface-controls CONTROLS.npz`, `gpuwrf surface-params basis/expand` | Experimental, opt-in, Noah-MP only | 11 Noah-MP parameters as per-cell runtime inputs (`z0mvt`, `eg`, `vcmx25`, `mp`, `albsat`, `albdry`, `rhol`, `rhos`, `laim`, `saim`, `shdmax`; branch requirements: `laim`/`saim` need `dveg` 1/3/4, `shdmax` `dveg=4`, `vcmx25`/`mp` `opt_crs=1`). Table-edit oracle byte-exact for every control (released code on an edited `MPTABLE.TBL`); 49/49 tests in both carry modes; 12/12 mutants. GPU (RTX 5090): an edited Swiss 24 h run loaded the first run's executable (warm 42.8 s vs cold 456.1 s, cache unchanged); Canary 2-domain 3 h cold 686 s, warm 52.6/51.3 s. Unedited (prior) controls are bitwise equal to the release program (357 variables × 25 frames) and pass D6 vs CPU-WRF (T2 RMSE 0.085 K). Edited-table response vs original CPU-WRF (1,764 land cells, hourly): median pattern correlation LAI 1.00, EMISS 0.99995, ALBEDO 0.996, U10/V10 0.987/0.989, UST 0.974, LH 0.896, HFX 0.850, T2 0.587, TSK 0.485; median response amplitude ratio 0.986–1.009 (the residual is consistent with run-to-run noise; inference, not proof). E41 PASS (identical native code; STACK ≤ 16 B Swiss, ≤ 32 B Canary). Cost when on: +1 kernel and 37 f32 planes per domain. Restart refuses changed, added or missing controls (6/6, CPU). Refuses `--decompose`. No Tenerife 3-domain GPU run; batched members API only (a batched lane is not bitwise equal to its solo run, also without controls). | [surface_params.md](../surface_params.md), [receipts](../../proofs/w41_trackp/receipts/) |
| `gpuwrf surface-params eki init/update/collect`, `osse` | Experimental scaffold | Ensemble Kalman inversion tools; checked on a synthetic twin (OSSE) only: truth ×(1.6, 0.7) recovered as (1.45, 0.63). No parameter set estimated from real observations; the real-station pilot is planned, not run. ALISIOS station inputs only through the guard. | [surface_params.md](../surface_params.md#ensemble-kalman-inversion-scaffold-synthetic-tests-only-in-v041) |
| `gpuwrf optimize released`, `plan`, `run`, `score` | Experimental workflow (orchestration only; no model code changed) | `released` lists the 47 seven-scheme configurations of the v0.4.0 checks (44 one-domain, 2 two-domain, 1 three-domain; single-domain E41 17 PASS, 26 FAIL, 1 UNKNOWN); other combinations are refused. `run` adds admission-sized parallel members (`--pool-gib`, `--host-gb-per-member`), rungs, pause/resume (`--chunk-minutes`, exit 3 PAUSED; `--resume`) and sealed bundles (`--bundle`, `--bundle-consumer`). CPU dry run (Swiss, 2 configurations × 2 rungs + held-out, synthetic dates): 8 members over 7 resumes. GPU (RTX 5090): Swiss 3 h members paused after rung 1 and resumed; warm members 38–49 s, cold 466/491 s, 2,214–2,216 MiB per process, bundle unchanged; Tenerife d01 `cu_physics` 0 and 1 in parallel at a 1.0 GiB pool: 1,712/1,744 MiB, 6.38/6.41 GB host. E41 of the smoke programs INCOMPLETE (slim AOT metadata), STACK ≤ 152 B. 19/19 mutants. The GPU runs used the orchestration before the final review fixes (those are covered by tests). Output is a held-out diagnostic, not a skill result; no real station tournament run. Host RAM (6–6.5 GB per cold member) binds first. | [regional_optimization.md](../regional_optimization.md), [receipts](../../proofs/w41_optimize/) |
| `skebs=1`, `sppt=1`, `spp=1` (+ `spp_conv/pbl/lsm`) | Experimental; **GPU 24 h D6 PASS**, single domain | Swiss 24 h on RTX 5090 vs the original serial CPU-WRF 4.7.1 with the same namelist and seeds (15 km length scales): D6 PASS on 4 arms (365/363/374/379 compared fields × 25 frames), stochastic pattern fields within 5.2e-7–1.9e-6, seeds equal in 25/25 frames, NF0, E41 PASS (STACK ≤ 16 B). The serial WRF random-number sequence is reproduced exactly by a parallel generator (4.85 → 0.36 ms per step; with all schemes 1.99 → 1.12 s per forecast hour). Run overhead +7 to +23 % and +74 MiB VRAM against an off run from another session (indicative). SPP fields `FIELD_CONV/PBL/SF` are written to history. CPU: 4/4 RNG mutants. Not checked: nested domains, GPU restart (CPU restart exact), ensemble spread/skill against stations. F2 repairs comparable to the stochastic-off run (172,848–176,737 vs 174,717). WRF's default length scales give NaN patterns on small domains (see upstream WRF issues). | [GPU qualification](../../proofs/w41_stoch/GPU_QUALIFICATION.md), [stochastic.md](../stochastic.md), [upstream issues](../UPSTREAM_WRF_BUGS.md) |
| `gpuwrf ens prepare --mode stochastic`, `--mode multiphysics` | Experimental preset tool (CPU) | Writes member namelists and a `member_manifest.json` with seed words (`--members`, `--schemes` default `skebs,sppt`, `--lengthscale-km`, `--nens-start`; `--physics-rows` for 12 multiphysics rows, 7 of them E41 PASS). Refuses SPPT outside `bl_pbl_physics` 0/5 or `cu_physics` 0/1, SPP without MYNN, seed collisions and length scales the grid cannot resolve. 27 tests, 7/7 mutants. Members inherit the status of their options. | [ensemble.md](../ensemble.md) |
| `gpuwrf aot-key --input-dir A [--namelist N] [--compare-input-dir B] [--compare-namelist M] [--json F]` | New, CPU only, nested cases | Loads a nested case like `run` (CPU process) and prints the AOT cheap key of every step program (root advance, per-domain advances, fused cascades) through the product key path; never steps (with nested AOT off or a scan loop it reports UNAVAILABLE without running a step). Compare mode: SHARED / NOT SHARED per program with the differing key component; a missing key is UNAVAILABLE, never SHARED (exit 3). Real WN3 0227 vs 0707 on the fixed tree: d01, d02, d03 and fused d02 SHARED, keys equal to an independent per-component recompute, ~6 min CPU. Keys carry the CPU target: compare cases with each other, not with GPU cache file names. Mutants M1–M4 killed. | [tests](../../tests/test_aot_key_cli.py) |

### Packaging and inputs

| Item | Status | Evidence and limits | Receipts |
| --- | --- | --- | --- |
| Boundary/lower-boundary record count not a compile dimension | Merged (default program) | `wrfbdy_d01`/`wrflowinp_d0N` record counts leave the compiled program (traced 4-record windows), so 48 h and 72 h inputs reuse the 24 h program; real.exe-only keys (`num_metgrid_*`) leave the memory-plan digest. 48 h vs 72 h inputs: same cheap key and lowered program on Swiss, Canary and Tenerife (CPU); CPU steps byte-identical to v0.4.0 incl. a 6 h bracket change and an 18 h window refresh; GPU frames 0–3 byte-identical to the sealed v0.4.0 bundle (Swiss 4/4, Tenerife 12/12); 24 h D6 vs original CPU-WRF PASS 6/6 domains; E41 hard STACK PASS 3/3 (max 32 B). Identities change by design (Swiss `e1102c20`, Canary `9b676bec` before the `SWDDIR` fix). RW2/RW4: the v0.4.1 bundle loads its 24 h-compiled programs for 48 h inputs. | [GPU requalification tools](../../proofs/w41_bdyrec/gpu_requal.sh), [composed requalification](../../proofs/w41_bdyrec/composed/session.sh) |
| Repository files > 10 MiB as release attachments | Merged | Three inherited files (29.1, 27.5 and 11.6 MB) are SHA-pinned attachments of the GitHub release; `tools/release_attachments.py stage` fetches them (cache → git history → release URL, every source SHA-checked); `verify.py` gate `oversized_tree`. | [tool](../../tools/release_attachments.py) |
| Sealed compile bundle consumers | Merged (release tooling) | Comparison runs that share one sealed compiled bundle consume it through a copy-on-write overlay and report any new program; memory plans travel with the bundle but are not compared. | — |
| Nested fused program: one program and key per grid | Merged | v0.4.0 `fused_jit` closed over the parent/child `OperationalNamelist`, so the Noah-MP `TMN` leaf of each fused domain (one 2-D leaf per domain; other case leaves dead) was baked as an HLO constant; the AOTKEY-v5 cheap key hashed it, so every new date missed the cache (~14 min cold fused compile per date reported by the ALISIOS operational user; a sealed bundle served only its priming date). Fix: namelists are `fused_jit` arguments, as for the root advance; bound-table discovery kept in the fused key. CPU on real WN3 0227 vs 0707: fused key equal and lowered HLO byte-identical; 3-date regression test fails on the unfixed builder and on a re-closed-child mutant; release identities Swiss `b977712fe325` / Canary `321aa651067f` exact. Cost [I]: ~60 fewer dense constants (~5 MB executable), `TMN` ops no longer constant-folded. Nested numerics are not claimed bitwise vs v0.4.0; on the final program (B1, pin `bcd156d19a`): Tenerife 0227/0502/0614 3-nest 24 h in parallel, D6 vs original CPU-WRF PASS 9/9 domains, one fused cheap key `d17836664ef3…` over the three dates (`ONE_KEY`, loaded from the AOT blob in all three), E41 hard STACK PASS. | [regression test](../../tests/test_fused_cascade_case_blind.py), [B1 result](../../proofs/w41_bdyrec/receipts/b1_RESULT.json), [fused keys](../../proofs/w41_bdyrec/receipts/b1_FUSED_KEYS.json) |
| v0.3.0–v0.3.4 shared caches (disclosure) | Measured on v0.3.3 | The v0.3.x cheap key (AOTKEY v4) ignored closures, so a cache shared across dates could load another date's fused program silently (`GPUWRF_AOT_VERIFY` off by default). Of 108 closed-over leaves only `TMN` differs between dates and is consumed; it differs only on water cells (WN3 d02 29,666/31,239, d03 8,289/10,323; 0 land cells), where `real.exe` sets `TMN = SST(t0)` and every atmosphere-facing output is `where(is_land, …)`. Land `TMN` date-invariant: 90/90 WN3 pairs (10 issues) and 48/48 nine-domain Canary pairs (4 issues). GPU A/B (RTX 5090, v0.3.3, issue 0707 24 h with the 0227 fused program vs fresh compile vs `GPUWRF_AOT_VERIFY=1` recompile): 72/72 wrfout byte-identical (376 variables per frame), frozen D6 0 failures, noise floor 0. Nine-domain grid: inferred from inputs + code path, not run end to end. Restart checkpoints would carry the foreign water-column soil temperatures (never written to wrfout). Mitigation for other grids: `GPUWRF_AOT_VERIFY=1` (mismatch → quarantine + recompile, ~6 min on v0.3.3). | [impact record](../../proofs/w41_v033impact/IMPACT.json), [land `TMN`](../../proofs/w41_v033impact/TMN_LAND_INVARIANCE.json), [GPU A/B](../../proofs/w41_v033impact/gpu147/) |
| v0.4.1 compiled release bundle (date-blind) | Sealed (`bundle_sha256` `24a7b24f2373…`, source commit `104c97de98`, version 0.4.1) | Required: warm replay from the sealed bundle on ≥ 3 issue dates of one WN3 grid with 0 new compiles (fused programs included), wrfout byte-identical to the priming run for the priming date, a 48 h replay on the same programs, `gpuwrf aot-key` SHARED for the date pairs, E41 on every bundled program. Results: six consumer replays (Tenerife 0227 72 h, 0502 3 h, 0614 48 h, 0227 with 48 h inputs 3 h; Swiss 3 h; Canary 3 h) `compiled_new=0`, `aliases_new=0`, `BUNDLE_OK` before and after every arm; d01 `5a03c654c3ff…` and fused d02 `438af7f6605c…` one key per program for all Tenerife arms, all sealed blobs; RW1 replays byte-identical to the compiling runs (219/219, 4/4, 6/6 files); CPU `gpuwrf aot-key` SHARED for d01/d02/d03/fused d02 on all three pairs (CPU keys carry the CPU target, so they are compared pairwise, not with GPU file names); E41 hard STACK and coverage PASS (tool INCOMPLETE: compact AOT kInput UNKNOWN). Not published; delivered to the operational user. | [RW verdict](../../proofs/w41_bundle041/receipts/RW_VERDICT.json), [bundle seal](../../proofs/w41_bundle041/receipts/BUNDLE.json) |
| CPU test suite | Merged | 41 failing/erroring nodes of the v0.4.0-era full CPU suite (pre-existing): 40 repaired (test-only, plus the NTU `mp_physics=56` and CAM-MG `mp_physics=11` modules importing the portability facade, CUDA-lowered jaxpr/StableHLO byte-identical before and after); the three PRODUCT-RISK rows resolved from the original WRF source in the product's favour. The remaining node is the README approval digest, re-sealed on the final README text. Strict xfails keep their cited reasons (FLG: limits below). Release identities exact before and after. | [GPU code identity](../../proofs/w41_redfix/FACADE_IMPORT_CUDA_LOWERING.json) |

### Nested boundary-layer qualification

**Superseded in v0.4.2** (see the nested boundary-layer table above). v0.4.1 text: v0.4.0 qualified non-default boundary-layer schemes on single domains only,
and that rule stands for v0.4.1: YSU (`bl_pbl_physics=1`), Shin-Hong (`11`), ACM2 (`7`) and EEPS
(`16`) are not validated inside nested domains (the `topo_wind=1` comparison above ran YSU on
Tenerife one-way nests). Their comparison inside a three-domain Tenerife set-up with original
CPU-WRF twins of the same configuration was still running at the v0.4.1 source freeze
(`104c97de98`); the catalog changes and D6/E41 results follow after the release.

## GPU requirements (unchanged from v0.4.0)

The release Pallas/Triton fast path requires **NVIDIA Ampere or newer,
compute capability ≥8.0**. **Pascal and Turing are unsupported.** This is a
backend prerequisite, not a guarantee of VRAM capacity, per-card validation
or a measured speedup. [Requirement and measured witness](../../proofs/r_docs040/GPU_REQUIREMENTS.json).

On a GTX 1080 Ti (Pascal, capability 6.1), Triton refused the release defaults.
A box-only legacy interpreted path passed the frozen bounded D6 comparison
over three Swiss forecast hours; warm whole-run GPU time was 33.16 s/forecast-hour
versus 9.61 for original CPU-WRF on the same six-core Ryzen 5 5500, **3.45× slower [M]**. These are
debug-box clocks including initialization/output, not release S1, a 24h gate
or evidence of a supported fallback. No Turing run is claimed.
[Source, clocks and scope](../../proofs/r_docs040/GPU_REQUIREMENTS.json).

## Carried forward from v0.4.0: physics selectors

These rows are the v0.4.0 record, unchanged for v0.4.1 except where a v0.4.1 note is added.
Each row cites one exact v0.4.0 configuration (`cNN` = row of the
[v0.4.0 66-configuration ledger](FINAL_COMPOSITES_v0.4.0.md), followed by the measured source
commit), not arbitrary combinations. The v0.4.1 default program is unchanged
(see "Default program" above); opt-in rows were not re-run on the v0.4.1 tree. Fields without a
frozen bound are reported, not judged (REPORT_ONLY).


### mp_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no microphysics | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Kessler warm-rain | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Lin et al. (Purdue-Lin) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | WSM3 simple-ice | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | WSM5 | Composite D6 PASS24h (E41-disclosed) | c05/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Ferrier (new Eta), HRW | Composite D6 PASS24h | c62/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 6 | WSM6 graupel | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Goddard 4-ice | Composite D6 PASS24h | c64/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 8 | Thompson | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 9 | Milbrandt-Yau 2-moment | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 10 | Morrison 2-moment | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | CAM 5.1 microphysics | Composite D6 PASS24h (E41 UNKNOWN) | c89/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41UNKNOWN. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 13 | SBU-YLin | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 14 | WDM5 | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). **v0.4.1:** `Nn` small-grid compiler barrier, CPU component outputs byte-identical (not re-run as a forecast); see [Microphysics change](#microphysics-change). |
| 15 | Ferrier-Aligo advected rime | Experimental column-only; excluded | Original CPU-WRF crashes at first output under both initialization conventions. [Exclusion](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 16 | WDM6 | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 17 | NSSL (legacy) | Composite D6 PASS24h (E41-disclosed) | c12/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 18 | NSSL 2-moment 4-ice w/ predicted CCN | Composite D6 PASS24h (E41-disclosed) | c13/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 19 | NSSL one-moment with hail | Composite D6 PASS24h (E41-disclosed) | c14/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 21 | NSSL one-moment without hail | Composite D6 PASS24h (E41-disclosed) | c15/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 22 | NSSL two-moment without hail | Composite D6 PASS24h (E41-disclosed) | c16/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 24 | WSM7 (separate hail/graupel) | Composite D6 PASS24h | c17/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 26 | WDM7 (separate hail/graupel) | Composite D6 PASS24h (E41-disclosed) | c18/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). **v0.4.1:** `Nn` small-grid compiler barrier, CPU component outputs byte-identical (not re-run as a forecast); see [Microphysics change](#microphysics-change). |
| 27 | UDM 7-class | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 28 | aerosol-aware Thompson (water/ice-friendly) | Composite D6 PASS24h (E41-disclosed) | c19/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 29 | RCON (Thompson aerosol-aware, liquid-phase mods) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 30 | HUJI spectral-bin (fast) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 32 | HUJI spectral-bin (full) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 38 | Thompson w/ 2-moment graupel/hail | Composite D6 PASS24h (E41-disclosed) | c65/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 40 | Morrison 2-moment w/ CESM-NCSU RCP4.5 aerosol | Composite D6 PASS24h (E41-disclosed) | c20/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 50 | P3 1-ice, 1-moment cloud water | Composite D6 PASS24h | c21/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 51 | P3 1-ice + double-moment cloud water | Composite D6 PASS24h | c22/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 52 | P3 2-ice + double-moment cloud water | Composite D6 PASS24h | c23/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 53 | P3 1-ice 3-moment + double-moment cloud water | Composite D6 PASS24h | c24/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 55 | Jensen-ISHMAEL | Composite D6 PASS24h (E41-disclosed) | c87/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 56 | NTU multi-moment | Composite D6 PASS24h (E41-disclosed) | c88/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 95 | Ferrier (old Eta) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 96 | Madwrf | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 97 | Goddard GCE | Composite D6 PASS24h (E41-disclosed) | c61/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### cu_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no cumulus | Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Kain-Fritsch (new Eta) | Default suite; Composite D6 PASS24h | c36/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Betts-Miller-Janjic (BMJ) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | Grell-Freitas ensemble | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | Scale-aware GFS SAS | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 5 | Grell 3D ensemble | Composite D6 PASS24h (E41-disclosed) | c13/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 6 | Modified Tiedtke | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Zhang-McFarlane (CAM5) | Experimental:E41; validated GPU24h D6 | c29/7468605609: rawD6 PASS,frames25,E41FAIL. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 10 | Modified Kain-Fritsch (PDF trigger) | Composite D6 PASS24h | c30/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | Multi-scale Kain-Fritsch (MSKF) | Composite D6 PASS24h | c31/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 14 | KIM Simplified Arakawa-Schubert (KSAS) | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 16 | New Tiedtke | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 93 | Grell-Devenyi ensemble | Composite D6 PASS24h | c17/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 94 | 2015 GFS SAS (HWRF) | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 95 | previous GFS SAS (HWRF) | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 96 | previous new GFS SAS (YSU) | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | previous Kain-Fritsch | Composite D6 PASS24h (E41-disclosed) | c14/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### bl_pbl_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no boundary-layer | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | YSU | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Mellor-Yamada-Janjic (MYJ) TKE | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | Hybrid EDMF GFS (ACM-GFS) | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | QNSE-EDMF (Quasi-Normal Scale Elimination) | Composite D6 PASS24h (E41-disclosed) | c26/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | MYNN TKE | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | ACM2 (Pleim) | Composite D6 PASS24h (E41-disclosed) | c05/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 8 | Bougeault-Lacarrere (BouLac) | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 9 | UW (CAM5) | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 10 | TEMF (Total Energy Mass Flux) | Composite D6 PASS24h (E41-disclosed) | c27/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | Shin-Hong scale-aware | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 12 | Grenier-Bretherton-McCaa | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 16 | TKE + TKE-dissipation (epsilon) | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 17 | TKE + TKE-dissipation + TPE | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | MRF | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### sf_sfclay_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no surface-layer | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | revised MM5 Monin-Obukhov | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Monin-Obukhov (Janjic Eta) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | NCEP GFS surface layer | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | QNSE surface layer | Composite D6 PASS24h (E41-disclosed) | c26/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | MYNN surface layer | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Pleim-Xiu surface layer | Composite D6 PASS24h | c43/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 10 | TEMF surface layer | Composite D6 PASS24h (E41-disclosed) | c27/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 91 | old MM5 surface layer | Composite D6 PASS24h (E41-disclosed) | c61/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### sf_surface_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no surface temp prediction | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | thermal diffusion (5-layer slab) | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 2 | Unified Noah LSM | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 3 | RUC LSM | Composite D6 PASS24h | c33/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | Noah-MP LSM | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | CLM4 (Community Land Model v4) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 6 | CTSM (Community Terrestrial Systems Model) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 7 | Pleim-Xiu LSM | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 8 | SSiB (Simplified Simple Biosphere) | Composite D6 PASS24h (E41-disclosed) | c40/36fcfaa64d: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### ra_lw_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no longwave radiation | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | RRTM | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | CAM | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 4 | RRTMG | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Goddard longwave | Composite D6 PASS: accepted6h | c45/5c15881ede: D6PASS,6h/7frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | FLG (UCLA) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |
| 14 | RRTMG-K (KIAPS) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 24 | fast RRTMG (GPU/MIC) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 31 | Earth Held-Suarez forcing | Composite D6 PASS24h | c39/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | GFDL (Eta) longwave (semi-supported) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |

### ra_sw_physics

| Code | WRF scheme | Status (v0.4.0 evidence) | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no shortwave radiation | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Dudhia | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Goddard shortwave | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | CAM | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 4 | RRTMG | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Goddard shortwave (new) | Composite D6 PASS: accepted6h | c45/5c15881ede: D6PASS,6h/7frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | FLG (UCLA) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |
| 14 | RRTMG-K (KIAPS) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 24 | fast RRTMG (GPU/MIC) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 99 | GFDL (Eta) (semi-supported) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |


## Carried forward from v0.4.0: other options and workflows

These feature rows describe admitted implementation scope and the specific evidence available. A physics-row PASS does not automatically qualify these additions. Selection/inputs, active-regime coverage and feature-specific comparisons remain required.

| Option | Status / scope | Receipt |
| --- | --- | --- |
| `moist_adv_opt=0/1/2`, `scalar_adv_opt=0/1/2`; `diff_opt`, `km_opt=0/1/4` within supported pairs | WRF transport/default binding repaired; exact combination qualification TBD-VALIDATION. Constant-K paths without native CLI support refused. | [Catalog](../../src/gpuwrf/io/scheme_catalog.py), [binding](../namelist-compatibility.md), [source/scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `diff_opt=2/km_opt=2` LES TKE; `km_opt=5` SMS-3DTKE | Experimental: E41; c38/c37 full24h bounded D6/coverage PASS at dd619e480; extra-scalar/nested/PBL restrictions apply. | [Ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [SMS qualification](../../proofs/s1_km5/QUALIFICATION.md) |
| `km_opt=3` | GPU-refused by the integrated CLI validator before JAX/device work. Historical Swiss GPU nonfinite at1h; Swiss/Madrid original CPU references crashed. CPU/reference scope remains experimental. | [Exclusion](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `dfi_opt=1/2/3` | c49/50/51 source5c15881ede DFI1/2/3 full24h D6/coverage+E41PASS. DFL has its natural24frames, not padded25. Single-domain/forcing restrictions remain; see [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [DFI oracle/evidence](../../proofs/s1_dfi12/), [scope and source](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `grid_fdda=1/2`, `obs_nudge_opt=1` | `grid_fdda=1`: c77/c78; `grid_fdda=2`: c79; all source5c15881ede full24h scientific PASS/E41PASS. `obs_nudge_opt=1`: c55/5c15881ede full24h scientific PASS/E41FAIL disclosed. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Spectral](../../proofs/s1_specnudge/), [observation](../../proofs/s1_obsnudge/), [ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `grid_sfdda=1/2` | c77/5c15881ede scientific PASS, E41PASS; c78/5c15881ede scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Surface qualification](../../proofs/s2_sfdda/), [actual admission ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `sf_urban_physics=1` | c36/04cb09f816 scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Source/input receipt](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `sf_urban_physics=2/3`, `use_wudapt_lcz`, anthropogenic heat/morphology | c56/04cb09f816 scientific PASS, E41FAIL; c57/2ed8c04c26 scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Class-D initialized/held producers are merged. **v0.4.1, Experimental:** BEP+BEM rooftop PV (`PV_FRAC_ROOF>0` → `EP_PV_URB3D`) and green roofs (`GR_FLAG=1`, `GR_TYPE` 1/2 → `QGR_URB4D`, `TGR_URB4D`, `DRAINGR_URB3D`): pristine-WRF column oracle, REAL8 0 violations on 8 PV/green-roof arms (incl. `GR_TYPE=1`, dry soil below the wilting point, and a constructed green-soil/roof contrast for `wall_gr`), REAL4 inside WRF's own REAL4-vs-REAL8 envelope (green-roof fluxes: frozen caps); CPU coupled smoke finite; GPU 9 h smoke (Madrid, PV 0.3 + green roof 1.0) rc 0, 0 nonfinite, all four fields written every frame — proves finite execution and written fields, not fidelity; E41 FAIL of the c57 class (Noah-MP water/BEM/sfclay `broadcast_subtract_transpose` kInput fusion, STACK 976 B vs c57 968 B, which the green-roof code joins; no PV/GR-off control compiled). No CPU-WRF forecast comparison; `GR_FRAC_ROOF>0` with `GR_FLAG=0` refused. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Urban/LCZ evidence](../../proofs/v040/urban_lcz/), [PV/green-roof oracle](../../proofs/w41_writergaps/README.md), [history fields](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer) |
| `sf_lake_physics=1` | c83/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Genuine NCAR lake cells/default-depth mode; nested not validated. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Lake evidence](../../proofs/s1_lake/), [configuration](../lake-model.md) |
| `sf_ocean_physics=1`, `isftcflx=1/2` | c84/5c15881ede scientific PASS, E41PASS; c85/04cb09f816 scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. PWP refused; nested not validated. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Qualification](../../proofs/s2_oml/QUALIFICATION.json) |
| Noah-MP `dveg=1–9`, `opt_run=1–8`, `opt_soil=1–3`, `opt_irr=0–3`, other admitted `opt_*`; `sst_skin` | Experimental branch-specific qualification: c68/c69/c70 source5c15881ede select dveg7/8/9, opt_run8 and opt_infdv1/2/3 respectively, full24h scientific PASS/E41PASS. No blanket options pass. Crop/tile/map-selected irrigation needs valid geography and lacks3D validation; sf_surface_mosaic=1 is refused. | [Land qualification](../../proofs/u2_land/QUALIFICATION.json), [exact selector coverage/exclusions](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [MMF inputs](../../proofs/s2_mmf/INPUT_REQUIREMENTS.md) |
| `sst_update=1`, `slope_rad=1`, `topo_shading=1` | c60/5c15881ede scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Selector inventory and WN3 source](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `topo_wind=1/2` | v0.4.1 option; see [Namelist options](#namelist-options). | [GPU arms](../../proofs/w41_topowind/GPU_ARMS.json) |
| `nwp_diagnostics=1`, `prec_acc_dt`, `do_radar_ref=1`, WRF-Solar diagnostics | Experimental online products; frozen field bounds apply only where present, radar can be REPORT_ONLY. Feature-specific final gate TBD-VALIDATION. | [Online diagnostics](../../proofs/u2_diag/QUALIFICATION.json) |
| tslist, selected station profiles | Implemented CPU-oracle sampling/export; final composite/regime gate TBD-VALIDATION. | [Original tslist oracle](../../proofs/u2_tslist/oracle/manifest.json) |
| Regional streams / window statistics | Experimental; instant/mean/min/max definitions and restart restrictions explicit. | [Qualification](../../proofs/u2_streams/QUALIFICATION.json) |
| `skebs`, `sppt`, `spp`, `spp_conv/pbl/lsm`, `rand_perturb`, `stoch_force_opt` | v0.4.1: `skebs`/`sppt`/`spp` GPU 24 h D6 PASS single domain with the same seeds as CPU-WRF, see [Tools and workflows](#tools-and-workflows); `rand_perturb`, `stoch_force_opt` remain experimental statistical checks. | [v0.4.1 qualification](../../proofs/w41_stoch/GPU_QUALIFICATION.md), [v0.4.0 checks](../../proofs/u2_stoch/QUALIFICATION.md) |
| `windfarm_opt=1` Fitch | c46/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Manifest](../../proofs/u2_windfarm/manifest.json), [scope](../wind-farms.md) |
| `tracer_opt=2`, `gpuwrf_tracer_file` | c90/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Tracer values remain REPORT_ONLY; WRF-Chem and two-way tracer feedback refused. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Original active reference/ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [component qualification](../../proofs/u2_tracers/QUALIFICATION.json) |
| `feedback=1`, `smooth_option=2` | c42/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Raw historical CLAT failure retained; arbitrary nested combinations remain unqualified. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Two-way CPU evidence](../../proofs/u2_twoway/QUALIFICATION.json), [actual GPU ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| Multi-GPU independent ensembles / domain decomposition | Launcher and CPU virtual-rank evidence; real GPU scaling/efficiency TBD-VALIDATION. | [v0.4 scope/source inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [multi-GPU runbook](../../scripts/runpod/RUNBOOK.md) |
| Moving/vortex-following nests, adaptive time step, unsupported WRF-Chem/Fire/Hydro/wave paths, WRFDA (3D/4D-Var, radiances; the v0.4.1 offline LETKF `gpuwrf da` is separate, see above), HUJI30/32, CLM4/CTSM, RRTMG-K14/fast24 | Refused/out of this native CLI release scope; recognized reference/build paths do not establish operational support. | [Catalog](../../src/gpuwrf/io/scheme_catalog.py), [scope inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |


## Carried forward from v0.4.0: Noah-MP selector admission

This enumerates every selector in the backend-dark `SUPPORTED` table, including
selectors whose only admitted value is their default. Values are admission
ranges, not every-combination scientific approval. **Default-qualified** means
the frozen default tuple; **experimental CPU/oracle/branch-specific composite
scope** applies to nondefault values, with the exact selected GPU cases in the
[FINAL ledger](FINAL_COMPOSITES_v0.4.0.md) and the original
[land qualification](../../proofs/u2_land/QUALIFICATION.json).

| Selector | Default | Admitted scalar values | Validation class / limits |
| --- | --- | --- | --- |
| `dveg` | 4 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_crs` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_btr` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_run` | 3 | 1, 2, 3, 4, 5, 6, 7, 8 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_sfc` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_frz` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_inf` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_rad` | 3 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_alb` | 2 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_snf` | 1 | 1, 2, 3, 4, 5 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_tbot` | 2 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_stc` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_gla` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_rsf` | 1 | 1, 2, 3, 4 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_soil` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_pedo` | 1 | 1 | Default tuple only; other values refused; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_crop` | 0 | 0, 1 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_irr` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_irrm` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_infdv` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_tdrn` | 0 | 0, 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |

Prepared geography and actual MPTABLE parameters remain required for the
selected branches. BATS albedo needs its seven table coefficients; dynamic
VIC, tile drainage, crop/irrigation and MMF retain their explicit pairing/input
rules. Surface mosaic and active unsupported urban/aerosol branches stay
refused. No absent geography or output field becomes an initialized zero pass.


## Operator commands and workflow classes

These are the v0.4.2 command names (unchanged from v0.4.1). The v0.4.0 rows were
[checked against the CLI](../../proofs/r_docs040/CLI_GUIDE_AUDIT.json); the v0.4.1 additions were checked against `gpuwrf <command> --help` of the merged tree. Help
verification establishes the interface, not model fidelity or a new forecast
skill result. Forecast-launching workflows inherit the scheme/composite
classes above. [Usage and input schemas](USER_GUIDE_v0.4.2.md).

| Command | Class and delivered scope |
| --- | --- |
| `run` | GPU forecast; default-qualified or per-composite/E41-disclosed/INCOMPLETE/refused as selected. |
| `namelist-support` | CPU/backend-dark admission catalog; scientific classes remain in this matrix. |
| `doctor`, `manifest`, `reproduce` | CPU offline case/provenance tools; [source/fixture scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). No forecast launched by reproduce. |
| `post` | CPU offline forecast/application products; [definitions and qualification](../post.md). |
| `verify` | CPU offline station verification with explicit optional observation fetching; [scope](../station_verification.md). |
| `ens products/verify/prepare/launch` | CPU products/statistics/planning; launch schedules admitted forecasts. [Workflow evidence](../ensemble.md); real multi-GPU scaling unqualified. |
| `optimize identity/plan/score/run`, `calibrate fit/apply` | CPU/source-qualified regional planning/scoring/calibration; run schedules forecasts. [Held-out study rules](../regional_optimization.md); tools do not establish new skill. |
| `perturb` | CPU prepared-case perturbations; [member preparation scope](../ensemble.md). |
| `export` | CPU offline NetCDF/Zarr/GRIB2 output with optional dependencies; [source/fixture evidence](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| `tslist-export`, `stream-check` | CPU offline station/stream interfaces; [source inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| `cycle` | CPU same-grid land/snow cycling (not atmospheric DA; for that see `da`); [qualification](../cycling.md). Distinct from atmospheric DA and crash restart. |
| `da check/analyse/cycle/members` | v0.4.1, experimental, d01 only: CPU LETKF analysis; `cycle`/`members` launch admitted forecasts in one process. See [Tools and workflows](#tools-and-workflows). |
| `surface-params basis/expand/eki/osse`, `run --surface-controls` | v0.4.1, experimental: Noah-MP parameter controls without recompiling; EKI scaffold on synthetic tests only. |
| `optimize released` | v0.4.1: CPU listing of the 47 released configurations (`--max-dom`, `--e41`). |
| `ens prepare --mode stochastic/multiphysics` | v0.4.1: CPU member-namelist presets; members inherit their options' status. |
| `quickstart` | CPU onboarding plan/WPS-real preparation; optional GPU run inherits forecast gates. [Inputs/source rules](../onboarding.md). No separate onboard/cycling executable. |


## Qualification limits that apply across the matrix

- F2: zero raw output NaN/Inf does not justify negative pre-mask events or repairs. Final hail receipts do not count qh in their census. [Quartet](../../proofs/v040/gpuqual_qhmass/QUALIFICATION.json).
- Original MYNN rmol cold-start undefined read: use the admitted compiler-ZERO original reference where designated, disclose raw-original spread, retain frozen bounds. [Source/reference evidence](../../proofs/t_p3fix/UPSTREAM_WRF_BUGS.md).
- P3 native strict/FMA residuals, Jensen diagnostic residuals, WDM5 number-kind spread and NTU source-diffusion undershoots stay visible. [Known issues](../../KNOWN_ISSUES.md#v040-draft-qualification-limits).
- The nine formerly omitted class-D fields now have merged WRF-defined producers or held constants; unsupported active branches are refused. Missing fields would still make coverage INCOMPLETE. [Producer evidence](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer).
- The 66 registered v0.4.0 scientific scopes are complete. Station skill, portability/real multi-GPU performance and unrun optional regimes remain unqualified. A warm whole-run benchmark exists for v0.4.1: [release notes, Speed](../../release_notes/RELEASE_NOTES_v0.4.1.md#speed) (B1: Tenerife 3 cases parallel 6.131 s per case-hour; Canary 24 h 4.36 s/fc-h from launch, harness without the C-auto pool; matched A/B vs v0.4.0: stepping ±1.7 %, whole run +2.5/+3.3 % launch basis, one v0.4.0 arm excluded for host disk load; [B1](../../proofs/w41_bdyrec/receipts/b1_RESULT.json), [A/B](../../proofs/w41_bdyrec/receipts/s1ab_AB.json), [follow-up](../../proofs/w41_bdyrec/receipts/s1ab_FOLLOWUP.json)). No every-combination or later-source PASS is inferred from these rows. [Snapshot scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- **v0.4.1 finding, default path, fixed in v0.4.1:** v0.4.0 and every earlier release (bug since the SW port; v0.3.3 producer == v0.4.0 to 1.5e-5 W/m²) wrote `SWDDIR` (and the derived `SWDDIF`) from the delta-scaled cloud optical depth; WRF writes the unscaled `ztdbt_nodel` product (`module_ra_rrtmg_sw.F` 8577–8600). Pristine RRTMG_SWRAD oracle on 186 real Tenerife 0227 columns: the v0.4.0 `SWDDIR` differs by 31–129 W/m² on cloudy day columns (clear columns 0.27 W/m²); GSW/SWDOWN and heating rates were unaffected, but with `slope_rad=1`/`topo_shading=1` the split sets the slope-adjusted surface shortwave ([I] slope sweep on 81 cloudy day columns: worst-orientation SWNORM error max 31/58/83 W/m² at 10/20/30° with February sun, 41/84/102 W/m² with July sun). v0.4.1 (merged `28046dabbe`): SWDDIR ≤ 0.32 W/m² vs the pristine oracle on 186 Tenerife + 107 Swiss columns, every other SW output ≤ 0.42 W/m², the v0.4.0 product as mutant fails 17–149×; kernel registers unchanged, STACK 0; composed 24 h D6 6/6 PASS and E41 STACK PASS (see Default program). The `aer_opt=1/3` path already emitted WRF's form. Clear-sky `SWDDIRC`/`SWDDNIC` are not written by the port. [Known issues](../../KNOWN_ISSUES.md#v041-direct-beam-surface-shortwave-swddir), [oracle tests](../../tests/v025/w41_swddir/test_swddir_oracle.py).
- v0.4.1 options carry their own code-quality (E41) notes in the rows above; none is part of the default program.
- FLG (`ra_lw/ra_sw_physics=7`, experimental CPU-only): the 17 strict original-WRF band-bound tests are xfailed (strict=True), pre-existing since v0.4.0; each cites the per-band disposition in [FG93](../../proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json) (max band error 0.0014 W/m²; 10 bands outside the native-compiler envelope, not independently source-bounded). Known limitation, not a pass.
