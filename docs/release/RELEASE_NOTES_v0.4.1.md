<!-- Copy of release_notes/RELEASE_NOTES_v0.4.1.md with links re-based for the docs site. Edit the original, then regenerate this copy. -->
# v0.4.1 — data assimilation, terrain winds, aerosols and cloud fraction

**Release date: 18 October 2026.** Every number in these notes is measured unless it is
marked as an estimate.

v0.4.1 adds features requested by operational users of v0.4.0: an
experimental ensemble data-assimilation cycle, WRF's terrain correction of surface
winds, aerosol in RRTMG shortwave radiation, two more WRF cloud-fraction options for
radiation, Noah-MP surface parameters that can be changed without recompiling, and
regional physics selection that runs on the GPU. Stochastic physics (SKEBS, SPPT, SPP)
now has a 24-hour comparison with the original CPU-WRF using the same random seeds.

Every new option is opt-in. With none of them selected, the default physics is that of
v0.4.0 apart from the fixes listed below. Three changes touch the default program: the
boundary-record change (same outputs, new compiled-program identity), the fix of the
direct-beam surface shortwave (`SWDDIR`, see Fixes), which changes results with slope
radiation, and the nested-program change, which passes inner-domain inputs to the program
when it runs instead of compiling them in (see Fixes). All three are in v0.4.1 and were
checked together against the original CPU-WRF: 24-hour comparisons passed on the Swiss Alps,
Canary Islands and Tenerife cases, and on three Tenerife forecast dates with the final
program. Default program in v0.4.1: Swiss Alps `b977712fe325` and Canary Islands outer
domain `321aa651067f` (v0.4.0: `3356e24924a9` and `196aed9f9973`).

[Per-option support matrix](SUPPORT_MATRIX_v0.4.1.md) ·
[Command guide](USER_GUIDE_v0.4.1.md) ·
[Known issues](../../KNOWN_ISSUES.md) ·
[v0.4.0 notes](RELEASE_NOTES_v0.4.0.md)

## At a glance

| Feature | How to use it | Status | Checked against the original CPU-WRF |
| --- | --- | --- | --- |
| Ensemble data assimilation (LETKF) | `gpuwrf da check / analyse / cycle / members` | Experimental, outer domain only | Algebra, file and cycle checks; no forecast-skill claim |
| Terrain correction of surface winds | `topo_wind = 1` or `2` (YSU or Shin-Hong boundary layer) | Experimental; 24 h comparison passed | Swiss Alps 3 km (two boundary layers) and Tenerife 9/3/1 km |
| Aerosol in shortwave radiation | `aer_opt = 1` (climatology) or `3` (Thompson aerosol-aware) | Experimental; 24 h comparison passed | Tenerife 9/3/1 km (`aer_opt=1`), Swiss Alps 3 km (`aer_opt=3`) |
| Radiation cloud fraction | `icloud = 3` or `2` | Experimental; 24 h comparison passed | Tenerife 9/3/1 km (each option) |
| Noah-MP surface parameters as runtime inputs | `gpuwrf surface-params`, `gpuwrf run --surface-controls` | Experimental | Swiss Alps 24 h, response to edited parameters |
| Regional physics selection on the GPU | `gpuwrf optimize released / plan / run / score` | Experimental workflow | Workflow runs on CPU and GPU; no skill claim |
| Stochastic physics | `skebs`, `sppt`, `spp`; `gpuwrf ens prepare --mode stochastic` | Experimental; 24 h comparison passed | Swiss Alps 3 km single domain, same random seeds as CPU-WRF |
| Check which compiled programs two cases share | `gpuwrf aot-key --input-dir … --compare-input-dir …` | New tool, CPU only | Not applicable: no forecast is run |

"24 h comparison passed" means: a 24-hour GPU forecast and an original CPU-WRF 4.7.1
forecast from the same inputs, compared hour by hour on every output field against the
limits fixed before the v0.4 comparisons (for example 1.5 K for 2 m temperature), with no
failure. It covers the tested configuration only, which is why these options stay
experimental: other combinations and other regions have not been compared.

## New features

### Ensemble data assimilation (LETKF), experimental

`gpuwrf da` combines an ensemble of short forecasts with observations and writes an
analysis `wrfinput_d01` from which the next forecast starts. The analysis is a Local
Ensemble Transform Kalman Filter (LETKF) for wind, temperature and water vapour on the
outer domain; the dry column mass is held fixed. It runs on the CPU between forecasts.

- `gpuwrf da check --config FILE` validates a configuration; `gpuwrf da analyse`
  writes the analysis; `gpuwrf da cycle` runs forecast/analysis windows; `gpuwrf da
  members` runs ensemble members.
- `da cycle` and `da members` run all members of a window in one process on one compiled
  step program. On the GPU (outer domain of the Canary Islands case, 10 members, two
  6-hour windows) the first member to run took 415 s including compilation and each further
  member 11.0–14.1 s, against 38 s for a separate warm process. All 28 member runs
  completed. The history files compared with separate fresh runs were byte-identical: the
  window-0 forecast of member m01 (7 files), a window-1 file seam (7 files) and a restart
  seam (6 files); the other members were not byte-compared. These GPU windows used the
  zero-update control arm; the analysis step itself was exercised on the CPU.
- Observations: radiosondes (IGRA2), surface stations (ISD-Lite) and CSV/JSON tables.
  Cloud observations can only be withheld for verification.
- Checks: the analysis reproduces the exact Kalman update in analytic tests, behaves as
  expected in Lorenz-96 twin experiments, keeps total water consistent, and writes a
  file that passes the model's own case checks. On one development case the temperature
  misfit to the assimilated soundings fell from 1.90 K before to 1.24 K after the analysis.
- **Not established: better forecasts.** On three development cases the 12-hour station
  errors fell for 2 m temperature, 2 m humidity and 10 m zonal wind and rose for 10 m
  meridional wind, and the model lost an analysed trade-wind inversion within 2–3 hours.
- **Known limitation, too much spread:** the optional ensemble settings (perturbed
  boundaries, adaptive and additive inflation) produced a prior spread 2.3–2.6 times the
  actual error in their registered twin experiment, outside the required range 0.5–2.0.
  With default settings the opposite happens: the spread decays by 28–76 % per 6 hours.
- Not supported: nested domains, analysis of the dry column mass, cloud or radiance
  assimilation.

Guides: [data assimilation](../da.md), [ensemble options](../da_ensemble.md),
[cycling in one process](../da-cycle.md).

### Terrain correction of surface winds (`topo_wind`)

WRF's `topo_wind` adds surface drag over rough sub-grid terrain and, with option 1,
corrects 10 m winds in valleys. As in WRF 4.7.1 it acts only with the YSU
(`bl_pbl_physics=1`) and Shin-Hong (`11`) boundary layers; with MYNN or another
boundary layer the option changes nothing and a warning says so.

- `topo_wind=1` (Jiménez and Dudhia 2012) needs the sub-grid terrain variance `VAR_SSO`
  in `wrfinput`. WPS writes it only when the `varsso` geography data are installed; a
  file without it is refused, as in WRF.
- `topo_wind=2` (Mass and Ovens) uses the geogrid terrain variance `VAR`.
- The correction fields equal those of the original WRF start-up code bit for bit on the
  Swiss and Tenerife grids.
- 24-hour comparisons with CPU-WRF passed for `topo_wind=1` on the Swiss Alps 3 km case
  with YSU (worst hourly root-mean-square 2 m temperature difference 0.48 K) and with
  Shin-Hong (0.26 K), and on the Tenerife 3 and 1 km nests with YSU; `topo_wind=2`
  passed on the Tenerife 9 km outer domain. The GPU reproduces WRF's effect: over Swiss
  land the mean 10 m wind drops by 0.28 m/s on the GPU and by 0.32 m/s in CPU-WRF.
- Code-quality note: the compiled surface-layer kernel with `topo_wind=1` and YSU uses
  400–496 bytes of per-thread stack, above our 256-byte limit. Results are correct; the
  speed cost has not been measured.

Guide: [namelist notes](../namelist-compatibility.md).

### Aerosol in shortwave radiation (`aer_opt`)

- `aer_opt=1` uses WRF's Tegen monthly six-type aerosol climatology in RRTMG shortwave.
  The climatology ships with `wrf_gpu`, so no extra input is needed; nested domains
  receive the aerosol from their parent as in WRF.
- `aer_opt=3` uses the Thompson aerosol-aware scheme's water- and ice-friendly aerosol
  (`mp_physics=28`, single domain) for the shortwave optics and writes the column
  aerosol optical depth `TAOD5502D`.
- Both need `ra_sw_physics=4` (RRTMG) on every domain; `aer_opt=2` is not supported.
  Longwave radiation carries no aerosol, as in WRF without WRF-Chem.
- 24-hour comparisons with CPU-WRF passed for `aer_opt=1` on the three-domain Tenerife
  case (clear-sky surface shortwave within 0.15–0.35 W/m², where the aerosol itself
  removes 20–23 W/m²) and for `aer_opt=3` on the Swiss Alps case (`TAOD5502D` within
  0.0001 of CPU-WRF).
- Code-quality note: with `aer_opt=3`, three compiled fusions of the existing Thompson
  aerosol-aware microphysics exceed our size limit; the aerosol code adds none. The scan
  of the `aer_opt=1` run is incomplete (it was repeated on a re-compile).

Guide: [radiation aerosol](../AEROSOL.md).

### Radiation cloud fraction (`icloud`)

- `icloud=3` is WRF's scale-aware cloud fraction (`cal_cldfra3`): a relative-humidity
  cloud fraction with radiation-only cloud water and ice. `icloud=2` is WRF's simple
  on/off cloud fraction (`cal_cldfra2`). Both replace the default `icloud=1` treatment,
  including the merge of MYNN sub-grid clouds, exactly in WRF's order. History `CLDFRA`
  is the cloud fraction that radiation saw.
- Both need RRTMG longwave and shortwave; `icloud=3` refuses `mp_physics=0`, `1` and `3`.
- `icloud=3` passed a 24-hour comparison with CPU-WRF on the three-domain Tenerife case
  (worst hourly root-mean-square 2 m temperature difference on the 1 km domain 0.09 K;
  cloudy share at +24 h within 0.02 percentage points of CPU-WRF on each domain). The
  extra cost is about 1 ms per radiation call and domain.
- `icloud=2` passed a 24-hour comparison with CPU-WRF on the same Tenerife case (worst
  hourly root-mean-square 2 m temperature difference on the 1 km domain 0.08 K). `CLDFRA`
  is on/off in both models, and the cloudy share at +24 h is within 0.05 percentage points
  of CPU-WRF on each domain. The cloud fraction itself costs 0.03 ms per radiation call and
  domain.

Guide: [radiation cloud fraction](../radiation-cloud-fraction.md).

### Noah-MP surface parameters as runtime inputs, experimental

`gpuwrf run --surface-controls CONTROLS.npz` replaces selected Noah-MP surface
parameters (for example leaf area, albedo, emissivity, roughness) with per-cell fields
without recompiling: every parameter set on one grid reuses the same compiled program.
`gpuwrf surface-params` builds the parameter basis of a case and expands parameter sets.

- Every control gives the same result as running the released model with the
  parameter table (`MPTABLE.TBL`) edited the same way.
- A Swiss Alps 24-hour GPU run with unchanged parameters is bit-identical to the
  released program and passes the comparison with CPU-WRF.
- With eight edited parameters, the GPU forecast moves like CPU-WRF with the same table
  edit: median hourly pattern correlations over land are 0.97–1.00 for emissivity,
  albedo, 10 m wind and friction velocity, 0.85–0.90 for heat fluxes and 0.49–0.59 for
  skin and 2 m temperature; median response amplitudes are within 1.5 % of CPU-WRF.
- Not done yet: calibration against real stations. The ensemble Kalman inversion tools
  (`gpuwrf surface-params eki`, `osse`) were checked on a synthetic twin experiment only.

Guide: [surface parameters](../surface_params.md).

### Regional physics selection on the GPU

`gpuwrf optimize` compares physics configurations for your region on past dates, with
held-out test dates. In v0.4.1 the study's forecasts run on the GPU end to end, several
at a time, with pause and resume.

- `gpuwrf optimize released` lists the 47 configurations that passed the v0.4.0
  comparisons with CPU-WRF (44 single-domain, 2 two-domain, 1 three-domain);
  `--e41 PASS` keeps only those without a code-quality note.
- `optimize run` runs forecasts in rungs, several at once as GPU and host memory allow,
  and can pause (`--chunk-minutes`) and continue later (`--resume`). It can use a sealed
  compiled bundle (`--bundle`, `--bundle-consumer`) so that members load programs
  instead of compiling them.
- Checked end to end on the CPU and on an RTX 5090: a Swiss Alps study with 3-hour
  members paused after the first rung and resumed; warm members took 38–49 s and cold
  ones 466–491 s, at about 2.2 GiB of GPU memory each. Two Tenerife 9 km members ran
  side by side at about 1.7 GiB each. Host memory, about 6–6.5 GB per member that compiles, is the
  first limit.
- What it does not show: which physics is best for your region. The result is a
  held-out diagnostic for your dates, not a validated choice of scheme, and no real
  station tournament has been run yet.

Guide: [regional optimisation](../regional_optimization.md).

### Stochastic physics and ensemble presets

- SKEBS, SPPT and SPP (`skebs`, `sppt`, `spp`, and `spp_conv/pbl/lsm`) now have a
  24-hour comparison with the original CPU-WRF on the Swiss Alps case, using the same
  namelist and random seeds: all four tested set-ups pass the field limits on every
  hourly frame, the random patterns agree to within 2e-6, and the seeds match in every
  frame.
- The random-pattern generator was rewritten to run in parallel on the GPU while giving
  WRF's serial sequence exactly; it takes 0.36 ms per step instead of 4.85 ms.
- `gpuwrf ens prepare --mode stochastic` writes one namelist per member with distinct
  seeds (`--members`, `--schemes`, `--lengthscale-km`); `--mode multiphysics` writes
  members from a list of 12 physics rows. The tool refuses settings it cannot support,
  for example SPPT with a boundary layer other than none or MYNN or a cumulus scheme
  other than none or Kain–Fritsch, SPP without MYNN, colliding seeds, or a length scale
  the grid cannot resolve.
- Not yet checked: nested domains, restart on the GPU, and the spread and skill of a
  stochastic ensemble against stations.

Guides: [stochastic physics](../stochastic.md), [ensembles](../ensemble.md).

### Check compiled-program reuse before a run (`gpuwrf aot-key`)

`gpuwrf aot-key --input-dir CASE` loads a nested case on the CPU and prints the cache key
of every compiled step program the run would look up. It needs no GPU, compiles nothing
and runs no forecast step. With `--compare-input-dir OTHER_CASE` it says for each program
whether the two cases share it (`SHARED`) or not (`NOT SHARED`, with the part of the key
that differs). A compile cache or bundle prepared on one case serves another case exactly
for the shared programs.

- `--namelist` and `--compare-namelist` use another `namelist.input`; `--json FILE` writes
  the full report.
- Nested cases only (`max_dom` of 2 or more). The keys are computed for the CPU, so compare
  cases with each other, not with the file names in a GPU cache.
- On the Tenerife three-domain grid, forecasts issued on 27 February and 7 July 2026
  share all four programs with v0.4.1; the comparison took about 6 minutes on the CPU.
  The command exits with code 3 if a key cannot be computed, and never reports such a
  program as shared.

Guide: [command guide](USER_GUIDE_v0.4.1.md#check-compiled-program-reuse-gpuwrf-aot-key).

## Fixes and robustness

- **Direct and diffuse surface shortwave (default physics).** Found during this release:
  v0.4.0 and every earlier release wrote the direct-beam surface flux (`SWDDIR`, and
  `SWDDIF` derived from it) from the delta-scaled cloud optical depth, whereas WRF uses the
  unscaled one. On cloudy daytime columns of the Tenerife case
  `SWDDIR` differed from WRF by 31–129 W/m². The horizontal total surface shortwave and
  the heating rates were not affected. With slope radiation or terrain shading
  (`slope_rad=1`, `topo_shading=1`), however, the direct/diffuse split sets the shortwave
  reaching sloping ground, so on cloudy slopes the surface energy budget and the forecast
  could differ from WRF; an estimate on real cloudy Tenerife columns puts the error of the
  slope shortwave at up to 31–102 W/m² on 10–30° slopes, depending on season and slope
  direction. v0.4.1 follows WRF: against the original WRF routine `SWDDIR` now agrees
  within 0.32 W/m² on 293 real columns from the Tenerife and Swiss cases. The fix changes
  the default program; it passed the 24-hour comparisons with CPU-WRF listed at the top.
- **Nested runs reuse their compiled program on every forecast date.** When an inner
  domain has nests of its own (for example the Tenerife 9/3/1 km and the nine-domain
  Canary set-ups), `wrf_gpu` compiles that domain and its nests as one program. In v0.4.0
  this program carried one piece of case data as a fixed constant: the Noah-MP deep-soil
  temperature (`TMN`) of the inner domains. Its cache key followed that value correctly,
  so every new forecast date compiled the nested program from scratch (about 14 minutes
  per date in the Tenerife operational set-up), and a sealed compiled bundle served only
  the date it was built from. In v0.4.1 the program receives these inputs when it runs:
  one program per grid and one cache key for all dates. A regression test checks that
  three forecast dates on one grid give one key and one compiled program; it fails without
  the fix. Nested forecasts with the new program: three Tenerife forecasts (issued 27
  February, 2 May and 14 June 2026; 24 hours, run side by side on one GPU) all loaded the
  same compiled nested program and passed the comparison with original CPU-WRF on all three
  domains: all 25 hourly frames complete, no limit exceeded. Users of v0.3.0 to v0.3.4 with
  a shared compile cache: see the note below.
- **One compiled bundle for all dates.** The v0.4.1 compiled release bundle is built from
  these date-independent programs. It ran Tenerife forecasts for 27 February (72 hours),
  2 May and 14 June, a 27 February case with 48-hour input files, and the Swiss Alps and
  Canary Islands cases without compiling anything (see the appendix).
- **No known failing CPU tests.** Of the 41 CPU tests that already failed on v0.4.0, 40
  were repaired. Most were stale tests or test fixtures; in the three cases where a test
  and the model disagreed, the original WRF source decided, each time in the model's
  favour. The NTU (`mp_physics=56`) and CAM-MG (`mp_physics=11`) modules now load the
  shared GPU-portability layer; their compiled GPU code is byte-identical. The 41st test
  is the README approval check, which is re-sealed on the final README text. Expected
  failures stay marked with their reasons (see FLG under Known issues).
- **WDM5 and WDM7 compiler safeguard** (`mp_physics=14` and `26`). On grids of up to 4096
  columns, the cloud-condensation-nuclei number now passes through the same small-grid
  compiler barrier that fixed a WDM6 compile stall in v0.4.0. On the CPU the schemes'
  outputs are bit-identical to v0.4.0 (component checks); on the GPU the barrier can
  regroup floating-point operations, so GPU results are not claimed identical. Both
  schemes ran 200 GPU steps with finite output; their 24-hour comparisons with CPU-WRF are
  the v0.4.0 ones and were not re-run. Two oversized compiled fusions are gone; the
  remaining per-thread stack (368–536 bytes) is still above our limit, and no speed change
  is claimed.
- **Input length no longer forces a recompile.** The number of records in `wrfbdy_d01` and
  `wrflowinp_d0N` leaves the compiled program, so a case prepared for 48 or 72 hours reuses
  the program compiled for a 24-hour case. Outputs stay byte-identical: on the GPU the
  initial and first three hourly history files of the Swiss Alps and Tenerife cases equal
  those of v0.4.0 byte for byte, and 24-hour comparisons with CPU-WRF passed on all six
  domains of the Swiss Alps, Canary Islands and Tenerife cases. The compiled-program
  identity changes because the program now takes the records as an input.
- **Large repository files moved to release attachments.** Three historical test and
  evidence files over 10 MiB are now SHA-pinned attachments of the GitHub release,
  fetched with `tools/release_attachments.py`.
- **Sealed compile bundles stay sealed.** Compiled bundles shared by several comparison
  runs are now consumed through a copy-on-write overlay, so a run can no longer change
  the bundle it uses.

## Speed

Measured with the release program on one RTX 5090 with an AMD Ryzen 9 9950X host. As in
the README, GPU times are the whole process (start-up, loading the compiled programs, the
forecast and writing every output file) and the CPU-WRF reference counts only WRF's main
loop.

| Workload | CPU-WRF | v0.4.1 on RTX 5090 | Ratio |
|---|---:|---:|---:|
| Tenerife 9/3/1 km, three 24 h forecasts in parallel | 123.4 s per case-hour (3 × 4 cores) | **6.1 s** per case-hour | **20×** |
| Canary Islands 9/3 km, 24 h, one forecast | 92.3 s per forecast hour (12 cores) | **4.4 s** per forecast hour | **21×** |

- Tenerife: the forecasts issued on 27 February, 2 May and 14 June 2026 ran side by side
  from a warm compile cache and finished 72 case-hours in 441 s. These are the forecasts
  of the CPU-WRF comparison above.
- Canary Islands: 105 s from process launch to the last of 48 history files. This run used
  the benchmark harness, which in this release starts without the per-case GPU memory
  pool that `gpuwrf run` sets up; `gpuwrf run` may therefore be slightly faster (estimate).
- **v0.4.1 against v0.4.0** on the Canary Islands case, alternating runs of both versions
  in one session (two measured runs each): forecast stepping is the same within 1.7 %; the whole run
  takes 2.5–3.3 % longer from process launch (5.19 and 5.23 s against 5.06 s per forecast
  hour). The other v0.4.0 run (over 8 s per forecast hour) was slowed by foreign disk load
  on the host and is left out. The same v0.4.1 program took 4.4 s in the benchmark session
  above and 5.2 s here: on this shared host the spread between sessions is larger than the
  difference between the versions.
- **Against the README figures:** the README charts (3.3 s, and 5.5 s for four forecasts at
  once) were measured with v0.3.2, before the WRF-fidelity work of v0.3.3 to v0.4.0. The
  current model is slower than v0.3.2 on these workloads; we have not measured how much of
  that each release added. v0.4.0 made no speed claim of its own; the matched comparison
  above is the version-to-version check. Energy was not measured again.

## Boundary layers inside nested domains

Not part of v0.4.1. v0.4.0 validated the non-default boundary layers on single domains
only, and that rule stands: YSU, Shin-Hong, ACM2 and EEPS are not validated inside nested
domains. Their comparison inside a three-domain Tenerife forecast with original CPU-WRF runs
of the same configuration was still running when v0.4.1 was frozen; the results will follow
in a later release ([support matrix](SUPPORT_MATRIX_v0.4.1.md#nested-boundary-layer-qualification)).

## Known issues

- Data assimilation: see the limits above (outer domain only, no skill claim, spread
  calibration not met).
- Code-quality notes of v0.4.1 options (per-thread stack or fusion size above our
  limits; results correct, speed not measured): `topo_wind=1` with YSU, `aer_opt=3`,
  and the remaining WDM5/WDM7 stack. The full list is in the
  [support matrix](SUPPORT_MATRIX_v0.4.1.md).
- **FLG radiation tests at strict WRF bounds are expected failures** (pre-existing since
  v0.4.0). FLG (`ra_lw_physics=7`, `ra_sw_physics=7`) remains experimental and CPU-only. Its 17
  band tests against the original WRF at strict bounds are marked as expected failures, each
  citing the recorded per-band disposition (largest band error 0.0014 W/m²; ten bands lie
  outside the native-compiler envelope and are not independently bounded).
  [Per-band dispositions](../../proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json).
- All v0.4.0 limits remain: see [KNOWN_ISSUES](../../KNOWN_ISSUES.md).

## Note for users of v0.3.0 to v0.3.4: nested runs with a shared compile cache

In v0.3.0 to v0.3.4 the nested program described under Fixes also contained the deep-soil
temperature `TMN` as a constant, but its cache key did not include it. A compile cache
shared between forecast dates could therefore load, without notice, a nested program
compiled for another date. v0.4.0 already gave each date its own key (and recompiled);
v0.4.1 removes the constant.

What we measured with v0.3.3:

- `TMN` is the only date-dependent value inside that program that the model reads. It
  differs between dates only over water, where `real.exe` sets it to the day's sea-surface
  temperature. Over land it was identical in every pair of dates we checked: 90 pairs from
  ten Tenerife three-domain forecasts and 48 pairs from four nine-domain Canary forecasts.
  Over water the value reaches only a placeholder soil column that is masked out of the
  surface fluxes, so it never reaches the atmosphere or the output.
- End to end on the Tenerife three-domain grid: a 24-hour forecast issued on 7 July 2026
  that loaded the program compiled for the 27 February forecast wrote history files
  byte-identical to a fresh compile, all 72 of them (3 domains × 24 hours). For the
  nine-domain Canary grid the same result follows from the identical land values and the
  same code path; that grid was not run end to end.
- Restart checkpoints written in such a run carry the other date's values in the
  placeholder water soil columns. They never appear in the output.

If the land `TMN` of your grid changes from date to date (for example with a different
preprocessing chain), these results do not carry over. With v0.3.x, set
`GPUWRF_AOT_VERIFY=1` when one compile cache serves several dates: the run then rebuilds
the nested program from its own inputs, compares it with the loaded one and recompiles on
a mismatch (about 6 minutes on v0.3.3). Upgrading to v0.4.1 removes the need for it.
[Impact record](../../proofs/w41_v033impact/IMPACT.json),
[land `TMN` check](../../proofs/w41_v033impact/TMN_LAND_INVARIANCE.json).

## Upgrading from v0.4.0

- Existing namelists run unchanged; the new keys default to the v0.4.0 behaviour
  (`topo_wind=0`, `aer_opt=0`, `icloud=1`). The `SWDDIR` fix changes `SWDDIR`/`SWDDIF` on
  cloudy columns, and forecasts with slope radiation or terrain shading can change against
  v0.4.0.
- `topo_wind=1` needs `VAR_SSO` in `wrfinput`: rerun WPS geogrid with the `varsso` data
  if your files lack it.
- Compiled programs from v0.4.0 are not reused: the compile cache is keyed by version, so
  the first v0.4.1 run of each grid compiles again.
- Nested grids now compile once per grid instead of once per forecast date, and one
  compiled bundle serves every date of its grids. `gpuwrf aot-key --compare-input-dir`
  shows which programs two cases share before you run them.
- Checkpoints: `--surface-controls` runs refuse to resume with different, added or
  missing controls. `aer_opt=3` runs and nested `aer_opt=1` runs carry extra restart
  fields (`TAOD5502D`, the nest aerosol). Resuming a v0.4.0 checkpoint with v0.4.1 is not
  supported: a `gpuwrf run` checkpoint records the exact model source it was written with,
  and a resume with any other source is refused with a source-identity error. Finish a
  running forecast with v0.4.0, or start it again with v0.4.1.

## Evidence

Technical evidence for each item, kept out of the summary above:

| Item | Evidence |
| --- | --- |
| Data assimilation | [core qualification](../../proofs/w41_da_core/QUALIFICATION.md), [ensemble design](../../proofs/a1_da/ens/ENS_DESIGN.md), [cycle tools](../../proofs/w41_da_cycle/) |
| `topo_wind` | [GPU comparisons](../../proofs/w41_topowind/GPU_ARMS.json), [WRF start-up oracle](../../proofs/w41_topowind/CTOPO_ORACLE.json) |
| `aer_opt` | [oracle and forecast receipts](../../proofs/w41_aerosol/) |
| `icloud` | [oracle](../../proofs/w41_icloud/), [`icloud=2` GPU comparison](../../proofs/w41_icloud2/README.md) |
| Surface parameters | [receipts](../../proofs/w41_trackp/) |
| Regional optimisation | [receipts](../../proofs/w41_optimize/) |
| Stochastic physics | [receipts](../../proofs/w41_stoch/) |
| WDM5/WDM7 safeguard | [compiler and GPU evidence](../../proofs/w41_wdm57/README.md) |
| Release attachments | [staging tool](../../tools/release_attachments.py) |
| Nested program, one key per grid | [three-date regression test](../../tests/test_fused_cascade_case_blind.py) |
| v0.3.x shared-cache impact | [impact record](../../proofs/w41_v033impact/IMPACT.json), [land `TMN` check](../../proofs/w41_v033impact/TMN_LAND_INVARIANCE.json), [GPU comparison](../../proofs/w41_v033impact/gpu147/) |
| `gpuwrf aot-key` | [tests](../../tests/test_aot_key_cli.py) |
| CPU test repairs | [NTU and CAM-MG GPU code identity](../../proofs/w41_redfix/FACADE_IMPORT_CUDA_LOWERING.json) |
| `SWDDIR` fix | [WRF oracle tests](../../tests/v025/w41_swddir/test_swddir_oracle.py), [Swiss oracle columns](../../proofs/w41_swddir/s01_summary.json), [slope estimate](../../proofs/w41_swddir/slope_estimate.py) |
| Input length out of the compiled program | [GPU comparison tools](../../proofs/w41_bdyrec/gpu_requal.sh) |
| Speed, nested forecasts on three dates | [benchmark run](../../proofs/w41_bdyrec/receipts/b1_RESULT.json), [one nested key for three dates](../../proofs/w41_bdyrec/receipts/b1_FUSED_KEYS.json), [v0.4.1 against v0.4.0](../../proofs/w41_bdyrec/receipts/s1ab_AB.json), [follow-up](../../proofs/w41_bdyrec/receipts/s1ab_FOLLOWUP.json) |

## Appendix: real-world release checks

Besides the comparisons above, v0.4.1 is checked the way operational users run it. Each
check runs the released program on the GPU:

- **RW1 warm cache:** a second run of a case loads every compiled program from the cache.
- **RW2 new date from a warm cache:** a forecast for another date on the same grid, also
  with 48-hour input files, loads the programs compiled for the first date.
- **RW3 72-hour three-domain forecast:** Tenerife, issued 27 February 2026, compared with
  the original CPU-WRF on every output field (field limits and file completeness), with
  speed (seconds per forecast hour), peak GPU memory, host memory and compile time
  against v0.4.0.
- **RW4 compiled bundle:** the sealed release bundle replayed on at least three forecast
  dates without compiling.

| Check | What ran | Result |
| --- | --- | --- |
| RW1 warm cache | A second run of Tenerife 27 February (72 h), Swiss Alps (3 h) and Canary Islands (3 h), each after a first run that compiled | **Pass.** Every compiled program loaded; all history files byte-identical to the first run (219, 4 and 6 files) |
| RW2 new date from a warm cache | Tenerife forecasts issued 2 May (3 h) and 14 June (48 h), and 27 February with 48-hour input files (3 h), on the programs compiled for 27 February | **Pass.** Nothing compiled; one cache key per program for all dates; `gpuwrf aot-key` on the CPU reports every program shared for the three pairs of cases |
| RW3 fidelity | Tenerife 27 February, 72 h, three domains, against the original CPU-WRF | **Pass.** On the outer and middle domains no limit is exceeded in any of the 73 hourly frames. On the 1 km domain the accumulated rain exceeds its 1 mm limit at hours 69–72 (root-mean-square difference 1.27–1.73 mm), but two CPU-WRF runs started from slightly different conditions differ by more (1.51–2.11 mm), so the rule frozen in v0.3.3 counts these hours as limited by that noise floor, not as failures. All output files are complete. The same forecast with v0.4.0 exceeds the rain limit from hour 67 (1.09–1.90 mm), by more than v0.4.1 at each of these hours, and fails the file check on one trace-graupel frame |
| RW3 resources | The same 72 h forecast, v0.4.1 against v0.4.0, both from their compiled bundles | Peak GPU memory 5,268 against 5,370 MiB per process (−1.9 %); host memory 5.97 against 5.74 GiB (+4.1 %); start-up to the first forecast hour 42.8 against 40.8 s (+5 %); compiling took 628 s (v0.4.1, this 72 h forecast) against 775 s (v0.4.0, its 3 h bundle build), so the bases differ |
| RW3 speed | The same 72 h forecast | Not conclusively measured on this host: other workloads ran on the GPU host cores during the comparison runs. A single 72 h pair without that control took 7.6 % longer with v0.4.1 for the whole run. The matched Canary Islands comparison shows unchanged stepping (±1.7 %, see Speed), and in the least-disturbed Tenerife pair the forecast part took 0.7 % longer with v0.4.1. A clean re-measurement follows as an addendum to these notes. |
| RW4 compiled bundle | The sealed v0.4.1 bundle on Tenerife 27 February (72 h), 2 May, 14 June and 27 February with 48-hour inputs, plus the Swiss Alps and Canary Islands cases | **Pass.** No program compiled in any of the six runs; the bundle was unchanged before and after each run |

Receipts: [real-world checks](../../proofs/w41_bundle041/receipts/RW_VERDICT.json),
[benchmark and nested-program runs](../../proofs/w41_bdyrec/receipts/b1_RESULT.json),
[version comparison](../../proofs/w41_bdyrec/receipts/s1ab_AB.json).

## Addendum (19 October 2026): RW3 speed, measured again

The RW3 speed cell above says the comparison could not be measured cleanly on release day. We repeated it
overnight on the same Tenerife forecast (issued 27 February 2026, three domains, 24 hours), running the two
sealed compiled bundles, v0.4.1 and v0.4.0, alternately: v0.4.1, v0.4.0, v0.4.1, and so on. Nothing compiled in
any run, both bundles were unchanged before and after every run, and every v0.4.1 run wrote files byte-identical
to every other v0.4.1 run and to the release-day v0.4.1 run (the same holds for v0.4.0).

The host is shared, so a watcher checked the four CPU cores that feed the GPU once per second. A run was stopped
and repeated (at most three tries per slot) when another program used more than one full core, or all other
programs together more than two cores, for more than 10 seconds. In the second session a run also counted only if
the model's own main thread waited for a CPU less than 1.5 % of its run time, the run's CPU pressure stayed below
2.5 seconds, and the data disk was not stalled. The rules and the pass rule were written down, with checksums,
before the first run of each session: pass if the v0.4.1/v0.4.0 ratio plus the largest spread between runs of the
same version stays at or below 1.10.

| Session | Counted 24 h runs | Seconds per forecast hour, whole run | v0.4.1 / v0.4.0 | Largest same-version spread | Result |
| --- | --- | --- | --- | --- | --- |
| 1 (21:52–23:15 UTC) | 3 × v0.4.1, 4 × v0.4.0 | v0.4.1 7.47–7.96, v0.4.0 7.43–7.97 | 1.019 | 7.3 % | **Pass** (1.091 ≤ 1.10) |
| 2 (23:19–01:57 UTC, stricter rules, heavier host load) | 4 × v0.4.1, 3 × v0.4.0 | v0.4.1 7.41–8.06, v0.4.0 7.20–7.95 | 1.011 | 10.4 % | **Not decided:** the host load changed during the session, which widened the spread between runs of the same version |

In the five pairs where both back-to-back runs counted, v0.4.1 took 0.971 to 1.030 times as long as v0.4.0 for
the whole run (mean 1.004). For the forecast part alone the ratio was 0.979 to 1.037 (mean 1.016). A 72-hour
v0.4.1 run took 6.65 seconds per forecast hour, as on release day (6.63). Its v0.4.0 partner does not count: the
load watcher fell behind and did not cover the whole run, which the pre-registered acceptance rule rejects. During
that run a system file-indexing job also stalled the data disk (recorded as a disturbance, not an acceptance
criterion in session 1), so there is no counted 72-hour pair.

**Conclusion:** in these 24-hour runs v0.4.1 was not slower than v0.4.0 by anything near the 10 % limit, and
they did not reproduce the 7.6 % of the single release-day 72-hour pair. Session 1 passes the pre-registered rule
narrowly (1.091 against 1.10) and session 2 does not decide it (1.114); in the five counted back-to-back pairs the
whole runs were within about 3 % of each other, and the forecast part may be 1–2 % slower with v0.4.1, within the
run-to-run noise of this shared host. These are measurements of the counted 24-hour runs, not a general bound for
the two versions, and there is no counted 72-hour comparison. In the same 24-hour runs the peak GPU memory per
process was 5,268 against 5,370 MiB and host memory 6.0 against 5.8 GiB.

Receipts: [session 1 result](../../proofs/w41_rw3add/receipts/S1_VERDICT.json),
[session 2 result](../../proofs/w41_rw3add/receipts/S2_VERDICT.json),
[both sessions and pairs](../../proofs/w41_rw3add/receipts/SUMMARY_BOTH.json),
pre-registered rules [session 1](../../proofs/w41_rw3add/receipts/PREREG.json) and
[session 2](../../proofs/w41_rw3add/receipts/PREREG2.json),
[release-day attempts](../../proofs/w41_bundle041/receipts/ATTRIB2b_ACCEPT.json).
