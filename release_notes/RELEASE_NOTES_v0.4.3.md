# v0.4.3 — the v0.4.2 model, qualified for three-nest research runs

**Release date: 21 October 2026.** Every number in these notes is measured unless it is
marked as an estimate.

v0.4.3 answers a list of questions from an operational user before a long series of
three-nest, 72-hour research forecasts (9, 3 and 1 km over Tenerife). It does not change the
forecast model. It adds a qualification table for that setup against the original CPU-WRF,
a status for every reported item, and a small robustness addition to the run receipts.

**The forecast model is the one of v0.4.2, and we checked the compiled programs, not only the
source code.** The model source of v0.4.3 differs from v0.4.2 only in the version string. The
run command also records in its receipt whether the computer was suspended during the run
(see below); that code runs outside the compiled model. The compiled programs of the v0.4.3
compile bundle were compared with those of the v0.4.2 bundle, program by program, and every
forecast of the release checks was compared byte by byte with the same forecast of v0.4.2
(all 21 compiled programs and all 716 output files identical). Results of v0.4.2 therefore hold for v0.4.3 unchanged.

[Per-option support matrix](../docs/release/SUPPORT_MATRIX_v0.4.3.md) ·
[Command guide](../docs/release/USER_GUIDE_v0.4.3.md) ·
[Known issues](../KNOWN_ISSUES.md) ·
[v0.4.2 notes](RELEASE_NOTES_v0.4.2.md)

## At a glance

- **Same forecast model as v0.4.2.** All 21 compiled programs and all 716 release-check output files
  are identical to v0.4.2.
- **Three-nest qualification.** Eight 72-hour Tenerife forecasts against the original CPU-WRF. On the
  62 fields with frozen limits (D6), 23 of the 24 forecast domains pass and one (27 February, 1 km) is
  limited by the CPU-WRF noise floor. Four forecasts are qualified; four fail the file-integrity check
  (trace amounts that are exactly zero on the GPU). 10 m wind after hour 40 stays held.
- **Reported items.** A status for each item an operational user reported, below.
- **Host suspend.** The run's summary file records whether the computer slept during the run.
- **Speed.** Unchanged from v0.4.2 (release manager ruling: no regression; details under Speed).

## Same forecast model as v0.4.2: how we checked

A new version number alone makes every compiled program new: the compile cache is keyed by
the version, and the compiler's kernel choice can differ between two compiles of the same
source, which changes results at round-off level (v0.4.2 against v0.4.1 showed this). For
v0.4.3 we removed that freedom and checked the result at four levels.

1. **Source.** The traced model code (519 source files that can enter a compiled program)
   is identical to v0.4.2 except the version line. The new receipt code is not among them.
   The model fingerprints used since v0.4.0 are unchanged (Swiss Alps `b977712fe325`, Canary
   Islands outer domain `321aa651067f`).
2. **Compile keys.** The compile keys of every program (program, settings, grid, input shapes, tables and environment) equal those of v0.4.2 except the parts that contain the version number. With the version line set back to 0.4.2, the source tree including the new receipt code gives exactly the keys of v0.4.2 for all 16 key records checked. So the receipt code has no effect on the compiled model.
3. **Compiled programs.** The v0.4.3 bundle was compiled with the kernel choices recorded by
   the v0.4.2 compile (the compiler's tuning results were copied in before compiling; they
   only select between equally correct kernels). For every program — the outer, middle and
   inner domain step programs and the radiation start-up programs — we compared the
   optimised program and the machine code (GPU binary) with v0.4.2: all 21 programs are the same: same optimised program, same GPU binary, byte for byte. The compile added no tuning results of its own (1,227 before and after).
4. **Forecasts.** Ten release-check forecasts wrote 497 output files: Swiss Alps 3 h, Canary Islands two domains 3 h, Tenerife three nests 72 h and 24 h with `icloud = 0`, and six replays of other dates and lengths (up to 48 h). Every file is byte-identical to the same file of v0.4.2. A second 72-hour run from the sealed bundle compiled nothing and wrote the same 219 files again.

## Run receipts record host suspend

On 20 October an operational host hibernated twice while a batch of forecasts was running; the
user's batch script stopped the running case when a GPU status query timed out, and seven queued
cases failed the same way. A run that survives a suspend keeps going, but its wall
times are then not valid. `gpuwrf run` now
compares two system clocks (one counts suspended time, one does not) at the start and end of a
run and writes `host_suspend` into `nested_pipeline_run.json`: `suspended_s` (seconds the host
was suspended during the run) and `timing_valid` (false when that exceeds one second). It
prints a warning when the host slept, and a failed run's error message says that the host was
suspended and how to resume from a checkpoint. It never blocks or delays a suspend and never
stops a run. This code runs before and after the forecast, outside the compiled model; the
compile keys do not depend on it (see above).

## Three-nest qualification table

We compared eight 72-hour forecasts on the 9/3/1 km Tenerife set-up with the original CPU-WRF 4.7.1
run from the same inputs, with a rule fixed before any result was looked at. Six forecasts use the
operational user's own inputs (WN3), two use ERA5 inputs (a cold low and a strong trade-wind day). All
are issued at 18 UTC; the scored hourly history starts at 00 UTC the next day. They were run with the v0.4.2 bundle, whose compiled programs are identical to those of
v0.4.3 (see above), so the results hold for v0.4.3.

A forecast is qualified when, on every domain, each of the 62 fields with a frozen limit (D6) either
passes or is limited by the CPU-WRF noise floor in all 73 hourly frames, and the file-integrity check
passes on every frame. With X the error of the GPU forecast against CPU-WRF, L the frozen limit and C the
same error between two CPU-WRF runs of the same case: in hours 0–24 a field must meet X ≤ L; from hour 25
on it may instead meet X ≤ max(L, 4C/3), but only where the case has its own second CPU-WRF run (2 of the
8 cases: 27 February and 2 March); in the other six any X > L is a failure. The D6 limits cover these 62 fields only; the other output variables have no limit and are
covered by the file-integrity check.

| Forecast | Inputs | D6 limits (62 fields), 3 domains × 73 hours | File integrity | Result |
| --- | --- | --- | --- | --- |
| 27 February 2026 | WN3 | 9 and 3 km pass; 1 km limited by the noise floor of its own CPU-WRF twin (rain at hour 72: 1.14 times, vertical wind at hour 69: 1.20 times the twin difference, allowed up to 4/3) | Pass | **Qualified** |
| 14 June 2026 | WN3 | Pass | Pass | **Qualified** |
| 8 June 2026 | WN3 | Pass | Pass | **Qualified** |
| 20 January 2026 | WN3 | Pass | Pass | **Qualified** |
| 2 May 2026 | WN3 | Pass | Fail: trace ice, graupel and rain zero on the GPU | Not qualified |
| 20 February 2026 | WN3 | Pass | Fail: 1 km cloud water zero at one hour | Not qualified |
| 2 March 2024 | ERA5, cold low | Pass | Fail: canopy water after a light-rain difference | Not qualified |
| 18 February 2024 | ERA5, strong trade wind | Pass | Fail: first light rain and 1 km cloud water zero on the GPU | Not qualified |

The four failures share one pattern: in a few frames the GPU field is exactly zero where CPU-WRF holds a
trace amount above the frozen allowance. The rule keeps them as failures; the cause is open
([known issues](../KNOWN_ISSUES.md#v043-zero-output-where-cpu-wrf-has-trace-amounts)).

**10 m wind after hour 40: held.** In all eight forecasts and all domains, the bounded 10 m and 3-D wind
fields pass their D6 limits from hour 41 to 72, except one value limited by the CPU-WRF noise floor
(27 February, 1 km vertical wind at hour 69). The row stays held: its
release is the operational user's decision, four forecasts fail the integrity check, only two have a
second CPU-WRF run, and on the cold-low case the 1 km 10 m wind differs from CPU-WRF 2.5 times as much as
two CPU-WRF runs differ from each other (still within the limit).

The full table, per domain and field class, with every identity and input checksum:
[`proofs/v043/w43-teacher/TEACHER_QUAL.md`](../proofs/v043/w43-teacher/TEACHER_QUAL.md).

## Status of the reported items

| Reported item | Status | Short answer |
| --- | --- | --- |
| Qualification of 9/3/1 km three-nest forecasts over 72 hours | See the table above | Case by case against the original CPU-WRF |
| Nine-domain forecast fails the limits on 1 km domains | **Open / inconclusive**, no cause localised | No defect in the GPU model was found; in a GPU check of equal-size neighbouring nests that share one compiled program, over one ordinary and one radiation parent step, a disturbance did not leak between them. Nine-domain runs are not qualified ([known issues](../KNOWN_ISSUES.md#v043-nine-domain-tenerife-forecast-differs-from-cpu-wrf-on-the-1-km-leaves)) |
| Fewer heavy-rain cells on the 1 km domain | **Open** | Microphysics, boundary layer, surface layer, advection, diffusion and the acoustic step each agree with WRF 4.7.1 for one call on real inputs, on CPU and GPU; the coupling over full time steps is not yet checked |
| Weak coastal wind near El Médano | No evidence of a systematic GPU-vs-CPU bias | Mean GPU minus CPU-WRF difference at most 0.007 m/s over three 24-hour cases; the formal 72-hour CPU-WRF noise floor and a matched v0.4.2 72-hour check are pending. The deficit against the station comes from the choice of days and model cell (not a model bug) |
| Three or nine domains change the shared domains | Not a model bug on the GPU | The three shared domains are byte-identical in one GPU check: the W9N case of 16 March 2026, hours 0–3 (later hours and other dates are not checked); original CPU-WRF changes them at round-off level. Compare runs with the same number of domains |
| 31 May 2025 case stopped at hour 58 | Not a model bug | Batch time limit during heavy disk load; on a quiet disk the case ran 72 hours in 564 s. This rerun is a diagnosis, not a qualification of that case |
| Host suspend stopped a batch | Fixed in v0.4.3 | The model was not at fault; the suspend is now recorded in the run's summary file (see above), with no effect on results |
| 10 m wind after hour 40 | Held | Bounded wind fields pass D6 (one value noise-floor limited) in all eight forecasts; released only when the user qualifies it (see the qualification section) |
| Urban canopy (SLUCM), `km_opt` 2 and 5 in nests, 333 m nests | Not in v0.4.3 | Deferred |

## Speed

v0.4.3 runs the same compiled programs as v0.4.2, so its speed is that of
[v0.4.2](RELEASE_NOTES_v0.4.2.md#speed). We still measured it on the Tenerife three-nest forecast of
27 February 2026, alternating v0.4.3 and v0.4.2 runs of 24 hours in one session on the same computer:

| 24-hour run, three nests | v0.4.3 (runs 1 / 2 / 3) | v0.4.2 (runs 1 / 2 / 3) |
| --- | --- | --- |
| Step time per forecast hour, 1 km domain (median) | 6.25 / 6.04 / 5.96 s | 6.75 / 6.03 / 6.27 s |
| Whole run per forecast hour | 7.73 / 7.69 / 7.66 s | 8.30 / 7.74 / 7.75 s |

- **Pre-registered verdict: INCONCLUSIVE.** The fixed rule counts only runs during which the CPU
  monitor recorded every second without gaps; it counted three of the six runs (v0.4.3 runs 2 and 3,
  v0.4.2 run 3), fewer than the two per version it needs. The other runs missed that coverage check
  by a few samples; their monitors showed no competing load. The verdict stays as it is.
- **Release manager ruling: NO REGRESSION.** All six runs agree within 10 %: v0.4.3 / v0.4.2 = 0.958
  for the 1 km step time, and 0.958 plus the largest spread between runs of one version (12 %) is
  1.078, below the 1.10 limit of the rule. The compiled programs are identical.
- Earlier the same day, a 72-hour v0.4.3 run took 8.67 s per forecast hour on the 1 km domain,
  against 6.15 s for v0.4.2 on its release day. In the later session v0.4.2 itself took 6.03 s, so
  that difference came from load on the computer, not from v0.4.3.
- Memory: GPU 5,268 MiB per forecast (v0.4.2: 5,236 MiB), host 6.20 GiB (v0.4.2 release day: 5.92 GiB).

## Appendix: real-world release checks

| Check | What it tests | Result |
| --- | --- | --- |
| Warm start | A second run on the compiled bundle compiles nothing and writes identical files | Pass: nothing compiled, 219 of 219 files identical |
| Other dates | Other issue dates on the same grid, and a 48-hour input, use the same programs | Pass: 2 May and 14 June (48 h) and a 48-hour input of 27 February ran on the bundle programs without compiling |
| 72-hour Tenerife forecast | Three nests, 72 h, against the original CPU-WRF; speed and memory against v0.4.2 | Forecast: byte-identical to v0.4.2, so the [v0.4.2 result](RELEASE_NOTES_v0.4.2.md#appendix-real-world-release-checks) holds unchanged (pass; 1 km rain at hours 69–72 and vertical wind at hour 69 limited by the CPU-WRF noise floor; trace graupel at hour 72 a registered exception). Speed: pre-registered verdict INCONCLUSIVE; release manager ruling NO REGRESSION (all six same-session runs within 10 %, ratio 0.958); see [Speed](#speed) |
| Bundle replay | The sealed bundle serves three dates without compiling | Pass: six replays (Swiss Alps, Canary Islands, 2 May with and without `icloud = 0`, 14 June 48 h, 27 February 48-hour input), nothing compiled, bundle unchanged after each |

Compile bundle: `d2f4f197b8affdecc0c297faae3593eb9d4df658cc3a58fc24f6523518aeee70`. The bundle was compiled with the kernel tuning results of the v0.4.2 bundle (1,227 entries; the compile added none). Its GPU-binary check gives the same result as the v0.4.2 bundle: largest stack 32 bytes, and the kernel-size part cannot be evaluated on the compact compiled programs, as before.
