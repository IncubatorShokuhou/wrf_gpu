# v0.4.2 — fixes and checks after v0.4.1

**Release date: 19 October 2026.** Every number in these notes is measured unless it is
marked as an estimate.

v0.4.2 follows up on what operational users and our own release checks found in v0.4.1.
It adds no new physics menu. It answers a question about the v0.4.1 shortwave fix, makes
WDM6 microphysics about three times faster, validates four more boundary layers inside nested
domains, adds a cloud-free radiation switch (`icloud=0`) for cloud–radiation studies, and
hardens the tools around the model.

The default model program is the one of v0.4.1: the compiled step program of the default
physics is unchanged (Swiss Alps `b977712fe325`, Canary Islands outer domain `321aa651067f`,
as in v0.4.1). History `COSZEN` now holds the value of the last radiation call, as in WRF
(see Fixes). Because the output writer changed, compiled programs are keyed anew and the first
v0.4.2 run of each grid compiles once. The compiled step program is unchanged and `COSZEN` changes by design. Because v0.4.2 compiles its programs anew, other fields differ from v0.4.1 at round-off level and the differences grow with lead time, as between two separate compiles of one version; fidelity against CPU-WRF is re-checked (D6, see the
appendix).

[Per-option support matrix](../docs/release/SUPPORT_MATRIX_v0.4.2.md) ·
[Command guide](../docs/release/USER_GUIDE_v0.4.2.md) ·
[Known issues](../KNOWN_ISSUES.md) ·
[v0.4.1 notes](RELEASE_NOTES_v0.4.1.md)

## At a glance

| Change | Who is affected | Checked against the original CPU-WRF |
| --- | --- | --- |
| Direct-beam shortwave (`SWDDIR`) fix of v0.4.1 confirmed; opt-out switch added | Forecasts with slope radiation or terrain shading | Code path traced against WRF 4.7.1; repeat forecast of a 72-hour Tenerife case |
| History `COSZEN` = WRF's held radiation value | Everyone reading `COSZEN` (default output) | Within 2.4e-7 of CPU-WRF on three domains, 73 hourly frames |
| WDM6 microphysics about 3× faster, same output | `mp_physics=16` (WDM5/WDM7 share the change) | Output byte-identical to v0.4.1 |
| YSU, Shin-Hong, ACM2 and EEPS validated inside nests | `bl_pbl_physics=1/11/7/16` in nested runs | 24 h, three Tenerife domains, each scheme against its own CPU-WRF run |
| Cloud-free radiation, `icloud=0` (opt-in) | Cloud–radiation sensitivity studies | 24 h, three Tenerife domains, against CPU-WRF with `icloud=0` |
| `gpuwrf aot-key` never writes a compile cache | Users of sealed compile bundles | Keys equal to v0.4.1 on the Tenerife and Canary cases |
| Comparison scorer with bounded memory; benchmark harness uses the production memory pool | Validation and benchmark users | Scores byte-identical; benchmark outputs byte-identical |

## The shortwave fix of v0.4.1, checked again

v0.4.1 changed the direct-beam surface shortwave (`SWDDIR`) to follow WRF. On one 72-hour
Tenerife forecast (issued 20 April 2026), an operational user saw the v0.4.1 run further from
CPU-WRF on cloudy south-facing slopes of the 1 km domain than the unfixed v0.4.0 run, in the
afternoon of 22 April.
We checked this in three ways.

- **Code path.** Every step between RRTMG shortwave and the slope-adjusted surface flux was
  traced in WRF 4.7.1 and in `wrf_gpu`: the direct beam from the unscaled optical depth, the
  diffuse fraction held between radiation calls, the slope and shading correction applied
  every step, and the swap of flat and slope fluxes after the land model. The two match.
  On clear cells the slope factor of `wrf_gpu` equals CPU-WRF's (mean difference 7e-6 over
  15,611 cell-hours), so the sun and terrain geometry agree.
- **Where the difference came from.** Splitting the slope-surface shortwave error into a part
  from the flat-ground flux and a part from the slope correction: in the affected hours the
  flat-ground part had a root-mean-square of 169–257 W/m² and the slope part 4–15 W/m².
  The v0.4.1 run had more cloud over those slopes in those hours. At matched cloudiness the
  v0.4.1 diffuse fraction is closer to CPU-WRF than the v0.4.0 one.
- **A second v0.4.1 run.** A repeat of the same forecast with v0.4.1 (same code, new
  compile, so tiny rounding differences) did not show the deficit: on the same slopes and hours
  its root-mean-square error was 140–173 W/m², against 170–269 W/m² in the first run. A test
  rule written before the repeat was inconclusive: the repeat still had slightly colder land
  afternoons than the v0.4.0 runs (land 2 m temperature 0.01–0.03 K lower on average), which
  we keep tracking.

Result: the fix stays the default; the operational user closed the question on 19 October 2026.

For users who need continuity with v0.4.0 output, **`GPUWRF_SWDDIR_V040_SPLIT=1`** restores
the v0.4.0 direct/diffuse split. Set it before the process starts. It compiles its own
program (new cache keys) and cannot be combined with `aer_opt=1` or `3` (the run stops with an
error). The compiled shortwave
kernel is byte-identical in machine code to the v0.4.0 kernel (checked by compiling both on
the CPU for the RTX 5090 target); a GPU forecast with the switch was not run for this release.
The switch reproduces v0.4.0, not WRF; use it only for comparisons. See
[environment settings](../docs/release/USER_GUIDE_v0.4.2.md#restore-the-v040-shortwave-split-gpuwrf_swddir_v040_split).

## New and changed options

### Cloud-free radiation, `icloud=0` (opt-in)

`icloud=0` runs RRTMG radiation as if there were no clouds, while microphysics, boundary-layer
clouds, rain and dynamics run unchanged. A pair of runs with `icloud=1` and `icloud=0`
separates the radiative effect of clouds from the rest of the forecast. This follows WRF 4.7.1:
the radiation driver sets `CLDFRA` to 0 and RRTMG reads no cloud water, ice or droplet sizes.

- History `CLDFRA` is 0 in every forecast frame; all-sky and clear-sky fluxes are equal.
- Needs RRTMG longwave and shortwave; the cumulus and microphysics restrictions of `icloud=2/3`
  apply.
- Against the unchanged WRF radiation driver on 16 real Tenerife columns (thin and thick
  cloud, ice, mixed phase, clear sky): within 0.12 % of the frozen RRTMG limits.
- 24-hour forecast, Tenerife 9/3/1 km (issued 27 February 2026), against an original CPU-WRF
  run with `icloud=0`: no limit exceeded on any of the three domains in any of the 25 hourly
  frames. Largest hourly root-mean-square 2 m temperature difference 0.07 K (1 km domain).
- Use both runs of a pair from the same compile cache or bundle: two separate cold compiles
  can differ by rounding.

Guide: [radiation cloud fraction](../docs/radiation-cloud-fraction.md#what-changes-with-icloud--0-cloudradiation-decoupling-switch).

### Boundary layers inside nested domains

Until v0.4.1, YSU, Shin-Hong, ACM2 and EEPS were validated on single domains only. In v0.4.2
each passed the 24-hour comparison with an original CPU-WRF 4.7.1 run of the same namelist
inside three nested domains (Tenerife 9/3/1 km, issued 7 July 2026, revised MM5 surface
layer). On every domain: no limit exceeded, no missing or non-finite field. The largest
error relative to its limit was 0.11 (EEPS, 10 m wind on the 1 km domain).

| `bl_pbl_physics` | Scheme | Worst hourly root-mean-square 2 m temperature difference (9 / 3 / 1 km) |
| --- | --- | --- |
| 1 | YSU | 0.025 / 0.015 / 0.036 K |
| 11 | Shin-Hong | 0.021 / 0.013 / 0.043 K |
| 7 | ACM2 | 0.019 / 0.022 / 0.062 K |
| 16 | EEPS | 0.014 / 0.033 / 0.061 K |

Limits: one forecast date and one domain set-up, with `sf_sfclay_physics=1` only; the
compiled-code quality scan of these nested programs was not run. The GPU forecasts were made
with the released v0.4.1; v0.4.2 keeps their compiled step programs and changes the `COSZEN`
output and opt-in code that these runs do not use (a new compile differs at round-off level).
[Details](../docs/release/NESTED_PBL_QUALIFICATION_v0.4.2.md).

## Speed

- **WDM6 microphysics** (`mp_physics=16`): the rain fall-out loop now stops after WRF's own
  number of sub-steps for each column instead of always running the 64-step maximum, and the
  remapping step no longer waits on the host. On the Swiss Alps 3 km WDM6 configuration of the
  v0.4.0 comparisons, forecast stepping went from 18.3 to 6.0 s per forecast hour on one RTX
  5090 (3.05×, alternating runs in one session), and a 3-hour warm run from 71 to 34 s. In those
  runs the history files were byte-identical to v0.4.1, so the change itself does not alter
  results. WDM5 and WDM7 (`mp_physics=14/26`) use the same code;
  their speed was not measured.
- **Benchmark harness.** Our benchmark harness touched the GPU before the model set up its
  memory pool, so the measured runs fell back to on-demand allocation. `gpuwrf run` was never
  affected. With the corrected harness, forecast stepping on the Canary Islands 24-hour case
  is 6 % faster than the old harness reported (3.16–3.19 against 3.38–3.39 s per forecast hour,
  alternating runs; outputs byte-identical). The whole-run figure of v0.4.1 measured this way
  is 4.6 s per forecast hour, from the one run whose disk was not slowed by other jobs; we chose
  that run after the session, and over all runs the corrected harness was not faster for the
  whole run. Three Tenerife 24-hour forecasts in parallel took 6.3 s per case-hour (19.7× the
  CPU-WRF throughput on 3 × 4 cores).
- **v0.4.2 benchmark** (release bundle, warm, corrected harness, one RTX 5090): Canary Islands
  9/3 km 24 h in 115 s from process start to the last history file, 4.81 s per forecast hour
  (mean of two runs, 4.85 and 4.77; 19.2× the 92.3 s of 12-core CPU-WRF); three Tenerife 24 h
  forecasts in parallel 6.23 s per case-hour (19.8× the 123.4 s of CPU-WRF on 3 × 4 cores).¹
  72-hour Tenerife release check: in alternating runs of the v0.4.2 and v0.4.1 bundles in one session, v0.4.2 took 0.94 times as long for the 1 km forecast stepping and 0.96 times for the whole run, within the 8 % spread between runs of one version, so no speed change is claimed (see the appendix).
- **Earlier figures:** the v0.4.1 numbers in the README history (4.4 s per forecast hour, 21×;
  6.1 s per case-hour, 20×) and in the [v0.4.1 notes](RELEASE_NOTES_v0.4.1.md#speed) were measured
  with the pre-v0.4.2 benchmark harness (on-demand allocation, see above). Numbers from different
  sessions on this shared host are not a version comparison; the matched comparison of v0.4.2
  with v0.4.1 is the RW3 check in the appendix.

¹ The three-forecast run counts as disturbed under the rule fixed before the session: the data
disk was slow during it (mean wait 53.7 ms, above 50 ms for 68 s). By that rule the number is
still reported. The two Canary Islands runs were undisturbed and wrote byte-identical files.

The README charts are still the v0.3.2 measurements.

## Fixes and robustness

- **History `COSZEN` follows WRF.** WRF writes the cosine of the solar zenith angle held from
  the last radiation call (evaluated half a radiation interval after the call). `wrf_gpu`
  wrote it for the output time, up to 0.057 off. v0.4.2 writes the held value: against CPU-WRF
  on the 20 April Tenerife forecast it agrees within 2.4e-7 on all three domains over 73 hourly
  frames. Only the output changes; the forecast itself is unchanged.
- **`gpuwrf aot-key` writes no compile cache.** In v0.4.1 the tool could add files to the
  compile cache it was pointed at, including a sealed release bundle. It now computes the keys
  in a private temporary cache that is deleted on exit, also on Ctrl-C or `SIGTERM`, and warns
  when the environment points into a sealed bundle. The keys are unchanged: on the Tenerife
  forecasts of 27 February, 2 May and 14 June and the Canary Islands case they equal the
  v0.4.1 keys.
- **Comparison scorer.** `scripts/wn3_score.py` now reads one pair of history files at a time.
  A 72-hour three-domain score needs about 2.4 GB of memory instead of 15–17 GB, with
  byte-identical results. A non-finite value in the CPU-WRF reference now fails the
  comparison; before, it was skipped silently. None of the references we re-scored contains one.
- **WDM6 speed-up** (see Speed): output byte-identical, so the v0.4.0 comparisons of WDM5,
  WDM6 and WDM7 with CPU-WRF still apply.

## Known issues

- **Fewer heavy-rain cells over the 1 km Tenerife domain** on day 2 and 3 of one 72-hour
  forecast (20 April 2026): cells with at least 10 mm accumulated rain were about 12–15 % fewer
  than in two CPU-WRF runs (comparing means) at 48 and 72 hours, beyond the spread of four GPU and two
  CPU runs. The domain-total rain is not systematically lower. v0.4.0 and v0.4.1 behave the
  same. One day only; the cause (Thompson microphysics and sedimentation under heavy rain) will
  be investigated with a second rain case. [Known issues](../KNOWN_ISSUES.md#v042-heavy-rain-cells-on-the-1-km-tenerife-domain).
- **The 1 mm accumulated-rain limit can be exceeded by CPU-WRF itself.** On the same forecast,
  two CPU-WRF runs that differ only in their MPI decomposition differed by 1.20 mm
  (root-mean-square) on the 1 km domain after 17 hours. On such days the rain limit measures
  the chaotic growth of small differences rather than how faithful the port is. The limits stay
  frozen.
- **WDM5/WDM6/WDM7 sub-step cap** (since the port, unchanged): rain fall-out uses at most 64
  sub-steps per microphysics call; WRF has no cap. It matters only for long microphysics time
  steps over very thin lowest layers in heavy rain (for a 120 s step, layers under about 22 m);
  on the tested Swiss case WRF needs at most 6. Removing the cap is planned for a later release.
- **Two CPU tests fail, as on v0.4.1:** a catalogue-text check of the BEP/BEM urban notes
  (`tests/test_v040_urban_slucm_wiring.py`) and a test of the data-assimilation member
  driver that inspects an internal function (`tests/a1_da/test_member_driver.py`). Both fail
  identically on the v0.4.1 release; the model code they cover is unchanged.
- **End of the 72-hour release check:** on the 1 km Tenerife domain v0.4.2 has no trace graupel at hour 72, where CPU-WRF keeps 3 cells with at most 6.6e-12 kg/kg; a second CPU-WRF run trips the same file check, so it is recorded as an exception (appendix).
- **Compiled bundles:** settings that are not in a compiled bundle (for example `GPUWRF_SWDDIR_V040_SPLIT=1`, other physics or other domains) compile once on first use. The code-quality scan of the bundle programs is incomplete, as in v0.4.1: the compact bundle programs carry no fusion sizes; the per-thread stack and coverage checks pass.
- All v0.4.1 limits remain: see [KNOWN_ISSUES](../KNOWN_ISSUES.md).

## Upgrading from v0.4.1

- Namelists run unchanged. `icloud=0` is new and opt-in; `icloud=1` stays the default.
- The compiled step program is unchanged and `COSZEN` changes by design. Because v0.4.2 compiles its programs anew, other fields differ from v0.4.1 at round-off level and the differences grow with lead time, as between two separate compiles of one version; fidelity against CPU-WRF is re-checked (D6).
- The first run of each grid compiles again (new cache keys). A v0.4.1 compile cache or
  bundle is not reused; rebuild compiled bundles with v0.4.2.
- Checkpoints record the exact model source; resuming a v0.4.1 checkpoint with v0.4.2 is
  refused. Finish running forecasts with v0.4.1.

## Evidence

| Item | Evidence |
| --- | --- |
| `SWDDIR` audit, repeat run, test rule | [regime-gate spec and repeat-run verdict](../proofs/w42_swdfid/), [opt-out tests](../tests/v025/w42_swdfid/test_swddir_v040_optout.py) |
| `COSZEN` | [tests](../tests/v025/w42_swdfid/test_history_coszen_held.py), [CPU-WRF fixture receipt](../proofs/w42_swdfid/coszen_fixture_receipt.json) |
| `icloud=0` | [oracle and forecast](../proofs/w42_icloud0/), [tests](../tests/v041/test_icloud0_oracle.py) |
| Boundary layers in nests | [summary](../proofs/w41_nestqual/NESTQUAL_SUMMARY.json), [page](../docs/release/NESTED_PBL_QUALIFICATION_v0.4.2.md) |
| WDM6 speed-up | [byte-identity and loop tests](../tests/test_w42_wdm6_fallout_loops.py) |
| `gpuwrf aot-key` | [receipts](../proofs/w42_aotkey/), [tests](../tests/test_aot_key_cli.py) |
| Scorer | [summary](../proofs/w42_scorer/SUMMARY.md), [finite-value tests](../tests/test_wn3_score_finite_gate.py) |
| Benchmark harness | [session table](../proofs/w42_harness/receipts/S1H_TABLE.md), [verdict](../proofs/w42_harness/receipts/S1H_VERDICT.json) |

## Appendix: real-world release checks

As for v0.4.1, the released program is checked the way operational users run it (RW1 warm
cache, RW2 new date from a warm cache, RW3 72-hour three-domain Tenerife forecast against the
original CPU-WRF, RW4 sealed bundle on several dates).

| Check | Result |
| --- | --- |
| RW1 warm cache | **Pass.** A second run of Tenerife 27 February (72 h) from the sealed bundle wrote all 219 history files byte-identical to the first run; the Swiss Alps and Canary Islands replays matched their first runs byte for byte (4 and 6 files). Nothing compiled |
| RW2 new date from a warm cache | **Pass.** Tenerife forecasts for 2 May (3 h) and 14 June (48 h), and 27 February with 48-hour input files (3 h), loaded the programs compiled for 27 February: one cache key per program for all dates, nothing compiled. `gpuwrf aot-key` on the CPU reports every program shared for all three pairs of cases |
| RW3 fidelity (Tenerife 27 February, 72 h, three domains) | **Pass.** On the outer and middle domains no limit is exceeded in any of the 73 hourly frames. On the 1 km domain the accumulated rain exceeds its 1 mm limit at hours 69–72 (root-mean-square difference 1.26–1.98 mm) and the vertical wind at hour 69 (0.333 against 0.30 m/s); at each of these the two CPU-WRF runs started from slightly different conditions also differ by more than the limit (rain 1.51–2.11 mm, vertical wind 0.330 m/s), so the rule frozen in v0.3.3 counts them as limited by that noise floor, not as failures (v0.4.1: the same four rain hours). Every CPU-WRF variable and global attribute is present in every frame. The file check flags trace graupel at hour 72 on the 1 km domain: v0.4.2 has none, CPU-WRF 3 cells with at most 6.6e-12 kg/kg. A second CPU-WRF run trips the same check at the same hour, and v0.4.2 carries comparable graupel up to hour 70 (154 against 217 cells), so this is recorded as an exception, not a failure |
| RW3 resources and speed against v0.4.1 | **No regression.** Alternating 24-hour runs of the v0.4.2 and v0.4.1 bundles in one session, with the rule fixed before the session (three v0.4.2 and two v0.4.1 runs counted): forecast stepping on the 1 km domain took 0.936 times as long with v0.4.2, the whole run 0.959; the largest spread between runs of one version was 8.2 %, so ratio plus spread is 1.018, below the 1.10 limit. The 72-hour pair (reported only: the load watcher did not cover the whole runs): whole run 6.76 against 7.15 s per forecast hour; peak GPU memory 5,236 against 5,268 MiB per process; host memory 5.92 against 5.91 GiB; compiling 612 against 628 s |
| RW4 compiled bundle | **Pass.** The sealed v0.4.2 bundle replayed Tenerife 27 February (72 h), 2 May, 14 June and 27 February with 48-hour inputs, the Swiss Alps and Canary Islands cases, and `icloud=0` on 2 May. Nothing compiled in any run, and the bundle was unchanged before and after each run |
| `icloud=0` from the bundle (Tenerife 27 February, 24 h) | **Pass.** Compiled into the bundle as its own programs and replayed on 2 May without compiling. Against the original CPU-WRF `icloud=0` run no limit is exceeded on any domain in the 25 hourly frames (largest hourly root-mean-square 2 m temperature difference 0.07 K, 1 km domain); `CLDFRA` is 0 in both models; the per-thread stack and coverage parts of the code-quality scan pass |
