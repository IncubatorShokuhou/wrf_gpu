# Noah-MP runtime parameters (W2): qualification in progress

Critic phase-A FIX repairs are ready for the delta check (section at the end). GPU gates remain open;
no GPU or forecast-performance claim.

W2 extends the existing explicit `--surface-controls` mechanism. The default
control set stays unchanged. Every selected field rides the existing traced
clock operand; the namelist retains only its names/shapes/dtypes signature.
`apply_surface_controls` replaces local parameter bundles inside the step trace,
after reading the clock operands. It never changes the namelist's static holder.
There are no new prognostic leaves, model equations, or kernel source changes.
Omit the W2 names to disable this family.

| Control | Pristine WRF meaning and consumers | Admission range |
|---|---|---|
| `smcmax` | SOILPARM MAXSMC, saturated volumetric moisture/porosity; ENERGY, PHASECHANGE, WATER | (0,1], >= SMCREF and existing SMOIS; urban WRF value .45 preserved |
| `dksat` | SOILPARM SATDK, saturated conductivity; WATER and derived KDT | 1e-12..1 m/s |
| `bexp` | SOILPARM BB, Clapp–Hornberger exponent; ENERGY, PHASECHANGE, WATER | 1e-6..30 |
| `psisat` | SOILPARM SATPSI, positive magnitude of saturated matric potential; ENERGY, PHASECHANGE | 1e-8..100 m |
| `rsmin` | MPTABLE RS, minimum Jarvis stomatal resistance; requires opt_crs=2 | 1e-6..1e5 s/m |
| `hvt` | MPTABLE canopy top; PHENOLOGY snow burial, ENERGY wind/displacement, radiation geometry | 0..100 m; HVB <= HVT |
| `hvb` | MPTABLE canopy bottom; PHENOLOGY burial and opt_rad=1 geometry | 0..HVT |
| `z0sno` | MPTABLE snow-covered ground momentum roughness; ENERGY | 1e-8..1 m |
| `snow_emis` | MPTABLE snow-covered ground emissivity; ENERGY | .5..1 |

Ranges are physical admission bounds, not a calibration recommendation. Soil
controls use `(4,ny,nx)` fields, matching WRF's four soil layers. Vegetation and
snow controls use `(ny,nx)` fields. Explicit arrays allow spatial changes; soil
class factors refer to ISLTYP, vegetation class factors to IVGTYP. DWSAT is a
separate SOILPARM column and is not silently recomputed from DKSAT/BEXP/PSISAT.
Non-Schaake and non-table soil combinations remain refused until qualified.

WRF authority: `module_sf_noahmpdrv.F` TRANSFER_MP_PARAMETERS (HVT/HVB:1477–1478,
RSMIN:1525, soil:1634–1640, KDT:1691, urban override:1694–1700, FRZX:1705).
The shared input validator checks existing SMOIS and SH2O against controlled
porosity on the host. It is imported byte-for-byte from main `f79da49f62`
(original helper `d49b259636`, blob `6daa5a9a`). Glacier admission uses the
initialized ISICE & XICE<=0 partition, and urban scaling retains WRF's .45 cap.
Urban includes both the table's ISURBAN and LCZ51..61, matching original
URBAN_FLAG. Three additional host controls check table/LCZ51/LCZ61 known caps,
explicit wrong-cap refusal and CLI/API parity with the urban correction on
(LS45:3PASS); no timestep or physics edit is involved.

| Gate | Current evidence |
|---|---|
| OFF Swiss canonical native program | [M CPU trace] exact `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a`; LS39 rebased6040d4b79b on mainc4a829ece4 |
| OFF pre-topo PROD canonical native program | [M CPU trace] exact `321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b`; post-topo composed-main identity remains a separate open gate |
| OFF raw CUDA HLO | [M CPU native lower] Swiss BYTE22,468,110B SHA3641e8be/1080native calls, including approved loader hook; final schema-safe pre-topo PROD BYTE21,905,496B SHA269260d3/1088calls (LS26) |
| ON default compiled bytes / changed values same cheap key | [M CPU] HVT, DKSAT and Jarvis RSMIN48/48BYTE. BEXP/PSISAT measured association class below. Each joint8-control cohort uses one key/two traces. Actual Swiss native-at-trace all8: different value digests, SAME production cheap key e3f2f7d6 and canonical trace f2871d7b (LS30). Actual GPU AOT still open |
| Pristine changed-parameter oracle, snow/frozen coverage | [M original WRF + compiled CPU] ALL9 controls x6columns:486 frozen field groups PASS (LS19–21); independent HVT snow-burial consumer matches original FVEG0→.2233572751, omission mutant killed (LS22–23). The six-column set has no frozen SOIL (critic F5); frozen-soil cohort added below: 4 soil arms x3 columns, 108 field checks + 12 water + 12 energy PASS (LS50–52) |
| Registered water / energy closure | [M]54/54 water and54/54energy PASS under the manager-approved, pristine-measured absolute-floor amendment below |
| Shared porosity / production loading | [M CPU]legacy+new consumer suite67PASS/5protected-case skips; rebased context delta23PASS. Actual control-file loader binds from initialized carry while production namelist.land stays None. Legacy bind/expand/validate ignores poison land_state (LS29–30) |
| Effective-cap / checkpoint seams | [M CPU]188 checker tests PASS; 83 checkpoint cases PASS. Default None:24 paired checker outcomes/reports BYTE; no SMCMAX:6 paired checkpoint outcomes/reports BYTE. Ignore-cap mutant killed2 genuine assertions; skip-retained mutant killed1, zero setup errors (LS31–37) |
| Native E41 / warm-root GPU A/B / memory | OPEN; manager night window required |
| Owner ACK / independent critic | Controls owner ACK22:16:13Z; CLI basis/class-axis GRANT22:28:29Z. Critic pending full gate |

The compiled CPU ON-default round-off class is measured on four real day/night
forest/grass columns. BEXP changes LH in one cell by1ULP (2.384e-7 W/m2,
relative1.148e-7), QFX in one cell by2ULP (1.137e-13 kg/m2/s, relative1.375e-7),
and Q2 in two cells by1ULP (9.313e-10 kg/kg, relative7.524e-8). PSISAT changes
one QSFC cell by1ULP (1.863e-9 kg/kg, relative1.189e-7), one LH cell by1ULP
(9.537e-7 W/m2, relative7.386e-8), and one QFX cell by1ULP (4.547e-13,
relative8.842e-8). Capturing only those ON operands as constants, while retaining
the same apply path and all model equations, restores48/48 OFF output bytes
versus43/48 with runtime operands. Optimized HLO retains the static BEXP-derived
literals and PSISAT×GRAV product, while the traced case computes from operands.
These are regression/transport measurements; changed values use original WRF
as the physical oracle.

The retained frozen/snow ON-default joint8 arrays differ in5/48 leaves: QSFC
(land and returned flux) one cell each by2ULP, max abs4.657e-10/rel1.993e-7;
LH two cells by at most4ULP,5.960e-8 W/m2/3.448e-7; QFX two cells by at most
6ULP,2.842e-14 kg/m2/s/3.928e-7; Q2 two cells by1ULP,2.328e-10/7.077e-8.
These diagnostics come directly from retained LS16 raw arrays (LS42), without
rerunning a numerical reference. The cold intervention (LS43) freezes only
the ON operands, retaining the same application and physics, and restores
48/48 OFF bytes versus43/48 dynamic. This independently localizes the cold
class to operand constantness too. The6ULP QFX maximum is disclosed alongside
the warm1–2ULP class; native GPU magnitudes remain unmeasured.

The six-column workload contains four retained real day/night columns, a real
WN3 frozen state with explicitly prescribed missing caller metadata, and that
state with a declared snowpack stress input. Pristine WRF is rerun on exactly
the supplied inputs and edited SOILPARM/MPTABLE files. All physics objects stay
unchanged. The wrapper sets original caller soil/FICEOLD flags, resets irrigation
rates, supports the Jarvis caller choice, and writes lossless REAL decimals.
Nighttime ALBEDO uses the original driver's held-value rule (SALB<=-999); the
raw internal sentinel is retained in the original failed scoring report.

Manager authority `2026-10-23T22:42:56Z` amended the near-zero C27 ratio gate
before re-evaluation. Its binding rule is now
`native <= max(1.5*pristine_unperturbed, measured_floor)` AND `native < .1 mm`.
The floor is the maximum original WRF REAL residual under32 fixed, distinct,
one-ULP soil-water input members on the **same parameter arm and column**.
Sixteen sign vertices plus16 fixed ternary patterns use seed20261023; SMOIS and
SH2O each move at most one REAL ULP. Pairs within one ULP move coherently to
preserve liquid<=total. No samples are rejected or chosen after measuring a
residual. The288 WRF runs (1728 column-members) give floors7.65e-6..1.98e-5mm.
All per-column values are in `LS18_floor/floor.json`; successor reports
`LS19_warm_floor.json` and `LS19_cold_floor.json` preserve prior ratio failures.
The same small cold residual exists with W2 OFF; its actual arrays are retained.
Energy keeps the frozen .05W/m2 bound. No field/energy tolerances changed.

Only host validation uses the initialized land context. The sole approved
loader edit passes `carry.noahmp_land` inside the existing controls branch;
legacy controls ignore it. Parameter-only checkpoint branches check retained
SMOIS/SH2O before the old early return, without imposing atmospheric analysis
checks. Joint soil edits pass the controlled cap to the same shared checker.
Owner restart helpers remain source-byte-identical to569c8a2226.

[I source] the eight common float32 fields carry80 bytes per horizontal cell
(four4-layer soils and four2-D fields); Jarvis RSMIN adds4 bytes. Across the
PROD domains this is3,171,120B common plus158,556B Jarvis. This is an operand
size, not a peak-memory measurement. Glacier column gathers, executable
temporaries, launches, readiness, actual VRAM/RSS and warm-root cost remain
unmeasured until the native night qualification. No hot-loop host transfer or
new state leaf is introduced.

Night qualification is one40-minute manager step for BOTH families. The frozen
replacement is `DAY_NIGHT_v6/proofs/w44_lsmparams/night40.sh`, SHA256
`d08e5979b1bd005af93ce7be15c2d6d760683e3831cbc260cbb8a36e89627bf5`.
It requires explicit GA_START_UTC/GA_END_UTC and2390s remaining, bounds its
whole payload to2370s, acquires official locks with timeout1 and keeps nested
GNU timers foreground. The top controller stays outside its one payload
group, forwards TERM and reaps descendants. Final LS47 CPU shims prove TOP-group
TERM during GPU (2.368s) and actual E41 RAM scope (2.538s), plus whole-timeout
(3.189s):0 survivors and dummy flock reacquired in every case. A healthy
two-family shim takes six private lock calls, returns0 and deletes both
consumed caches. E41 and cleanup have explicit same-line QUIET exits; failed
unfinished scans retain only the required cache for CPU continuation.
Invoke directly from night_seq.sh;
an extra caller RAM scope masks the caller's cancellation group (LS40 failure,
retained). No GPU was started. Native E41, warm A/B, final post-topo identity,
release RW gates and one independent critic remain open.

Manager's October24 day-slot launch-smoke prerequisite is met by LS49. The
same shell and six child commands run through private device boundaries on a
one-column real daytime Noah-MP workload, with all product fast flags retained.
Each child records `JAX_PLATFORMS=cuda` before changing only that device selector
to CPU, calls actual `noah_mp_step` through the production control/clock path,
and produces48 finite leaves plus changed-value output without a third JIT.
No native prime/measure/E41 PASS can be written by the private smoke path.
The product environment retains the installed JAX's preferred
`XLA_CLIENT_MEM_FRACTION=.25`, clears its deprecated alias to avoid conflicts,
clears stale JAX_PLATFORM_NAME and sets CUDA_VISIBLE_DEVICES=0. The installed
GPU-option parser independently returns cuda/cuda_async/.25/preallocate=true
without initializing a GPU. The earlier claim that the original allocator
name was misspelled was incorrect; LS48's supported deprecated-alias smoke is
retained as superseded evidence, with LS49 checking the final environment.

[M CPU mini] common8 prime/warm/profile29.226/9.227/8.414s;
Jarvis28.561/7.542/7.480s; entire exact-launch smoke96.146s. Both caches are
deleted and the private flock is reacquired. These fit the735/115/115s child
timeouts in the smoke; actual full-PROD GPU and E41 costs remain unmeasured.
`receipts/LAUNCH_SMOKE_V6.json` and its six child receipts preserve the product
environment and device substitution. Admission is GA-window-only for day or
night; no worker GPU job is started before the manager's sequence.

Temporary measurements, logs and pins:
`<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams/`. MAIN EXPERIMENTS contains every
LS01–49 measurement, including failures. Program-identity comparisons are
regression checks; the changed-table original WRF runs provide fidelity evidence.
No GPU-speed or release-readiness claim is made.


## Critic phase-A FIX repairs (2026-10-24, Opus takeover of the lane)

Verdict `cr-w44-lsmparams` FIX on `6d1b2e58a7` (VERDICT.md SHA256 `90d111e8…`). All repairs are CPU.

**F1 catalog revert.** Rebased onto the params owner registry `6528908b32`; the catalog hunk adds only the nine
names and the qualifier, keeping `rad_diag`, `icloud_bl {0,1}` and the MODIS note (critic source pre-check 15:29Z).
Catalog/namelist consumers: 192 PASS, 1 FAIL = the main reader-registry test fixed later on main `bd43e17c95`
(F1_catalog_readers.log). The lane remains stacked on unmerged params; after params merges it is rebased onto
main, dropping the params copies (w44-release 15:26Z order).

**F2 irrigation, F3 tile drainage: refused.** `IrrigationStatic.parameters` keeps its own initial
SMCMAX/BEXP/DKSAT/DWSAT and `SelectedHydrologyParameters` keeps the initial KDT/FRZX, so soil controls with
`opt_irr` 1–3 or `opt_tdrn` 1–2 would be silent partial edits. `inactive_reason` now refuses them loudly;
deletion-sensitive tests cover every soil control and option value (test_admission, 4 controls x 2 options).

**F4 cold start: refused when it matters.** WRF NOAHMP_INIT (`module_sf_noahmpdrv.F:2084–2097`) splits the
initial frozen soil water with the edited BEXP/SMCMAX/PSISAT; the port's `_noahmp_init_sh2o` uses the original
table. `bind_domain_controls(..., cold_start=True)` now refuses CHANGED `smcmax`/`bexp`/`psisat` on any
active-land layer with initial TSLB < 273.149 K. Unchanged values, warm initial soil and checkpoint
resume/branch (SH2O restored, `cold_start=False` from the nested loader) stay admitted. The loader change is one
keyword on the already-granted `_load_domains` line. The batched path (B>1) has no land state on its namelist and
therefore refuses changed values of these three fields (fail-closed). Tests: refusal for each of the three,
identity and restart admissions (test_admission), checkpoint tests bind with `cold_start=False` like the loader.

**F5/F5' frozen soil in the pristine oracle.** `columns.frozen_soil_columns()` adds the retained real WN3
`frozen_barren16` (top soil 265.53 K) and `frozen_grass10` (270.35 K), a declared warm-rain copy of barren16
(276.15 K, 10 mm/h), and — after the critic's delta verdict (F5') — a declared WET-frozen copy of barren16
(top-layer SH2O = SMC = .12) with and without the warm rain. `oracle.py --rows frozen` runs original WRF with
unchanged tables (base) and SMCMAX/DKSAT/BEXP/PSISAT x1.1.
- Correction (critic CR09, confirmed by the stricter census on LS51): in the retained frozen layers the liquid
  water is already BELOW the opt_frz=1 supercooled limit (barren16 L1 .0674 < .0971), so PHASECHANGE never
  froze there. The small SMC changes reported for LS50/LS51 were frost-deposition differences, not phase change;
  the earlier sentence "soil ice moves with BEXP/PSISAT/SMCMAX" was a misattribution.
- [M] LS65 census (`frozen_census.py`: freezing = L1 SH2O_OUT < SH2O_IN and ice gain > 10x the deposition
  bound): both wet-frozen columns freeze in every arm (base L1 .1200 -> .0971 = the supercooled limit, ice
  gain 2.29e-2 m3/m3, deposition 0). The limit binds: L1 SH2O_OUT vs base BEXP +.01427, SMCMAX +.00969,
  PSISAT +.00178 m3/m3 (2,900x / 1,900x / 360x the 5e-6 field tolerance). DKSAT moves SH2O by 3.6e-5 on the
  wet-rain column. Census PASS.
- The C27 residual subtracts precipitation `(PRCPCONV+PRCPNONC)*dt`; the first wet floor showed exactly
  P*dt = 0.25 mm. Dry receipts are numerically unchanged. 32-member pristine floors (LS65_floor):
  5.5e-6..1.6e-5 mm.
- [M] Port vs original WRF on the 5 frozen columns (LS66, `score.py --cohort frozen`, release defaults, CPU):
  all four arms PASS the frozen field bounds (45 checks each, SH2O/SMOIS max error 2.2e-8 against the 5e-6
  bound, also where the supercooled limit binds), water within max(1.5 x pristine, floor) and < .1 mm, energy
  residual 0.000 W/m2; one cheap key, two traces; ON-default 46/48 leaves byte-equal (class below).

**F6 deletion-sensitive wiring test.** `tests/w44_lsmparams/test_wiring.py` runs the actual `noah_mp_step`
through `traced_controls` + `apply_surface_controls` on the warm cohort and a cold cohort (frozen surface,
snowpack, two frozen-soil columns, rain stress), one compiled program per route:
- every control changed alone changes outputs;
- (F4') the `cold_start=` expression of the single bind in `_load_domains` is evaluated from the source AST:
  fresh True, resume False, branch False (test_admission). [M] mutants of the loader source (force False,
  drop the keyword, drop the branch term) each fail with a genuine AssertionError, healthy PASS;
- every documented consumer bundle is observed (full apply differs from the variant that keeps only the other
  bundle's new value). Routes under the admitted options: SMCMAX/BEXP/HVT both, DKSAT/HVB/RSMIN
  static.parameters, PSISAT/Z0SNO/SNOW_EMIS EnergyParams (PSISAT's static copy is read only by refused
  groundwater options). HVT's phenology route needs active snow burial, so its arms use HVB x.01 in both
  reference and changed;
- soil ice is active on the frozen columns; values never retrace (4 programs: OFF, full, two route variants).
- [M] Critic mutants (LS52, plugin modules unchanged): no-op 18 FAIL, static-only 11 FAIL, energy-only 11 FAIL,
  all genuine AssertionErrors, 0 setup errors; healthy: all wiring tests PASS.
- The CPU suite runs the legacy f64 Noah path (conftest); there OFF reads f64 table gathers while controls are
  WRF REAL, so ON-prior differs at parameter-rounding level (max rel ≈2e-9) and the class check is skipped. The
  module re-runs itself under the release defaults in a clean child (junit: only that self-test may skip).
- [M] Release defaults (Noah REAL path), all nine controls at prior values incl. Jarvis RSMIN (LS53/LS54; LS67 re-run
  with the 7-column cold cohort incl. the wet-frozen columns: 22 PASS, same class, constants restore all OFF bytes):
  warm LH/Q2/QSFC 1 ULP, QFX 2 ULP and land EAH 1 ULP in one cell (6.1e-5 Pa of 1826 Pa); cold LH 4, QFX 3,
  QSFC 2, Q2 1 ULP and the energy-closure residual ERRENG in one cell by 1.53e-5 W/m2 (a value near zero; 2 ULP
  of its ~120 W/m2 flux terms, far below the .05 W/m2 gate). EAH and ERRENG are new members of the disclosed
  class (the joint-8 non-Jarvis sets showed only QSFC/LH/QFX/Q2). Causal control in the same test: the same
  values captured as compile-time constants, same apply path, restore every OFF byte on both cohorts — the
  differences are operand-constness association, not a physics change. 22 PASS + the self-test skip.

## v7 GPU launcher for the 25.10 window (replaces DAY_NIGHT_v6)

Manager decision 23:59Z/00:35Z: pin the stacked head (src tree `ea2b3cd5` = `28bc238ecb`); if params merges
first, rebase and rebuild; otherwise prove schema-2 OFF identity of the post-rebase tree afterwards.
- `gpu_v7.sh` (production: PROD 2 domains, families lsm + jarvis, 2370 s) and `cuda_mini_v7.sh` (Swiss F20
  43x43, max_dom 1, family lsm, 1200 s) differ ONLY in the grid/length block. Both derive from `night40.sh`:
  E41 now runs in its own `systemd-run --user --scope MemoryMax=12G MemorySwapMax=0` (no shared RAM ticket),
  the first failing family stops the step, CPU phases use cores 24,25,28, and `root_gpu.py` asserts E225
  alignment from the case namelist before any model work (PROD 54/18 s, Swiss 18 s, hours 3).
- `launch_smoke_v7.py` runs the exact entry with the REAL `root_gpu.py` model path on CPU (only the device
  selector, the lock, nsys and timer values are substituted; E41 runs the real tool on the CPU cache and
  records its expected 'no native CUDA ELF' refusal). The v6 smoke used a one-column stand-in model.
- [M] What the real-path smokes found (LS56–LS62): an import-origin assert comparing a resolved module path
  with the unresolved `src` symlink; `differences()` rebinding its reference dict on the first non-BYTE
  leaf (IndexError — certain on GPU as well, because ON-default has the disclosed rounding class). Both fixed.
  On CPU each Swiss program needs 6–20 min to compile and ~4 GB host RAM; a 12G scope is OOM-killed at the
  third executable (journal, LS60/LS62). Manager 01:27Z: one CPU smoke in a 24G own scope (LS63, queued).
- [M] LS64 (pre-merge pin) and LS70 (re-pinned on the merged head GPU_v7m @c4a570d950 = origin/main c2025fd7ad +
  lsmparams) pass ALL phases of `cuda_mini_v7.sh` on the real model path on CPU in a 24G own scope (1559 s / 1541 s):
  one AOT key across values, changed values load with 0 new native blobs, E41 reaches the cache, measure and
  profile rc 0, caches deleted. Manifest `receipts/GPU_V7_MANIFEST.json`; smoke receipt `receipts/LAUNCH_SMOKE_V7_LS70.json`.
- [M] FINDING (root level, CPU): on the full nested root (Swiss, one root step) ON at the prior values differs from
  OFF by more than the column-level class: history LH in 244 of 1,849 cells by up to 533 ULP (1.34e-5 W/m2),
  land/accumulator leaves up to 786 ULP with relative <= 6.17e-5 in up to 1,040 cells, State leaves only on
  near-zero values (<= 4.5e-12). Identical before and after the merge. [I] likely iteration/branch decisions of
  the Noah-MP energy-balance solver flipped by 1-ULP operand changes; not yet causally proven at root level.
  Reported to manager and critic for a decision; the GPU magnitude is measured in phase B.
- [M] LS71 (manager decision (b), CPU, same case/pin, own 24G scope; tools `root_cause.py`,
  `root_cause_compare.py`, `root_cause_analysis.py`): the hypothesis above is NOT confirmed.
  (A) The constants arm (the same prior values as compile-time constants through the same apply path) removes ALL
  19 ON≠OFF leaves, but it is not byte-identical to OFF: SMOIS 3 cells at 1 ULP and SH2O 10 cells at ≤2 ULP, all
  in frozen layer-1 columns (TSLB 267.9–273.13 K, ISLTYP 16 ×9, 7 ×1). Traced ON equals OFF bitwise in SMOIS/SH2O,
  so the same values reproduce OFF there: this is the constant route's own folding, not a value mismatch.
  (B) No exit-iteration flips: TV and TAH are bitwise identical ON vs OFF in every column (TG 1 cell 1 ULP); the
  OFF census arm was lost to a launcher bug (systemd-run expanded `${arm}`) and is not needed for that conclusion.
  The ON≠OFF cells: 325 land columns over IVGTYP 1/5/7/10/12 and ISLTYP 3/6/7/12/16, no urban, ice or water
  columns. Median 1–2 ULP per leaf; the large ULP counts sit on near-zero evaporation (QFX max 786 ULP at
  4.5e-8 kg/m2/s against a field maximum of 1.0e-5, max abs 4.55e-12; LH max abs 1.34e-5 W/m2; QSFC 2 cells 23 ULP;
  EAH ≤2 ULP; energy accumulators max abs 2.4e-4). [I] few-ULP operand-constness differences amplified by
  cancellation in near-zero moisture fluxes. Stopped and reported; GPU phase B waits for the manager/critic ruling.

## ON-default round-off class at root level (manager ruling 2026-10-25 05:12Z / 05:16Z)

Ruling (1): ACCEPTED as an operand-constness round-off class. The causal control is LS71: the same default values
captured as compile-time constants remove all 19 differing carry leaves. Exit-iteration flips are excluded
(canopy TV/TAH are bitwise identical).

Measured table [M CPU, LS71]: Swiss d01, one nested root step, release defaults, pin c4a570d950 (src b60781a9),
traced controls at their default values vs the default program. Receipt `receipts/LS71_ONDEFAULT_TABLE.json`
(sha256 6e697f45; first table b0409fbf has the same numbers), tool `root_cause_table.py` + `field_table.py`.
Scale: field max = max |field| over the 1,742 Noah-MP land columns without land ice (State.xland = 1 and
IVGTYP ≠ ISICE; fills ≥ 1e30 excluded). Packed history families are split into named slices. ULP counts alone
mislead on near-zero values: QFX reaches 786 ULP at 4.5e-8 kg m-2 s-1, against a field max of 1.0e-5.

| Field | Units | Cells | ULP median / max | Max abs | Field max | Max abs / field max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `land_history.Q2B` | kg kg-1 | 84 | 1 / 23 | 1.07e-08 | 0.0056 | 1.9e-06 |
| `State.qsfc` | kg kg-1 | 2 | 22.5 / 23 | 1.07e-08 | 0.00603 | 1.8e-06 |
| `noahmp_land.qsfc` | kg kg-1 | 2 | 22.5 / 23 | 1.07e-08 | 0.00603 | 1.8e-06 |
| `land_history.EVG` | W m-2 | 238 | 2 / 648 | 1.91e-05 | 29.4 | 6.5e-07 |
| `energy_accumulators.ACLHF` | J m-2 | 241 | 2 / 600 | 0.000244 | 462 | 5.3e-07 |
| `land_history.EVB` | W m-2 | 211 | 1 / 29 | 5.01e-06 | 9.51 | 5.3e-07 |
| `history_diagnostics.lh` | W m-2 | 244 | 2 / 533 | 1.34e-05 | 25.7 | 5.2e-07 |
| `State.qv_flux` | kg kg-1 m s-1 | 225 | 2 / 720 | 4.09e-12 | 8.8e-06 | 4.7e-07 |
| `land_history.EDIR` | kg m-2 s-1 | 227 | 2 / 786 | 4.55e-12 | 1e-05 | 4.5e-07 |
| `State.qfx` | kg m-2 s-1 | 227 | 2 / 786 | 4.55e-12 | 1e-05 | 4.4e-07 |
| `land_history.TR` | W m-2 | 15 | 7 / 62 | 1.79e-07 | 0.486 | 3.7e-07 |
| `land_history.ETRAN` | kg m-2 s-1 | 15 | 10 / 46 | 7.11e-14 | 1.9e-07 | 3.7e-07 |
| `land_history.Q2V` | kg kg-1 | 43 | 1 / 4 | 1.4e-09 | 0.00549 | 2.5e-07 |
| `history_diagnostics.q2` | kg kg-1 | 67 | 1 / 4 | 1.4e-09 | 0.00551 | 2.5e-07 |
| `land_history.CHB2` | m s-1 | 1 | 6 / 6 | 2.79e-09 | 0.012 | 2.3e-07 |
| `land_history.SHG` | W m-2 | 40 | 1 / 4 | 1.53e-05 | 70.5 | 2.2e-07 |
| `energy_accumulators.ACGRDFLX` | J m-2 | 39 | 1 / 13 | 0.000244 | 1.47e+03 | 1.7e-07 |
| `State.fltv` | K m s-1 | 23 | 1 / 3 | 1.12e-08 | 0.0833 | 1.3e-07 |
| `State.theta_flux` | K m s-1 | 18 | 1 / 3 | 1.12e-08 | 0.084 | 1.3e-07 |
| `State.hfx` | W m-2 | 18 | 1 / 3 | 1.14e-05 | 95 | 1.2e-07 |
| `land_history.TGB` | K | 2 | 1 / 1 | 3.05e-05 | 279 | 1.1e-07 |
| `noahmp_land.tg` | K | 1 | 1 / 1 | 3.05e-05 | 280 | 1.1e-07 |
| `State.t_skin` | K | 1 | 1 / 1 | 3.05e-05 | 280 | 1.1e-07 |
| `noahmp_land.t_skin` | K | 1 | 1 / 1 | 3.05e-05 | 280 | 1.1e-07 |
| `land_history.TRAD` | K | 1 | 1 / 1 | 3.05e-05 | 280 | 1.1e-07 |
| `land_history.TGV` | K | 1 | 1 / 1 | 3.05e-05 | 281 | 1.1e-07 |
| `land_history.T2B` | K | 2 | 1 / 1 | 3.05e-05 | 281 | 1.1e-07 |
| `history_diagnostics.t2` | K | 1 | 1 / 1 | 3.05e-05 | 282 | 1.1e-07 |
| `land_history.T2V` | K | 1 | 1 / 1 | 3.05e-05 | 283 | 1.1e-07 |
| `land_history.FIRA` | W m-2 | 6 | 1 / 2 | 7.63e-06 | 73.7 | 1e-07 |
| `land_history.GRDFLX` | W m-2 | 41 | 1 / 11 | 7.63e-06 | 81.5 | 9.4e-08 |
| `land_history.GHV` | W m-2 | 54 | 2 / 73 | 7.63e-06 | 87.8 | 8.7e-08 |
| `energy_accumulators.ACHFX` | J m-2 | 16 | 1 / 2 | 0.000122 | 1.71e+03 | 7.1e-08 |
| `land_history.IRG` | W m-2 | 29 | 2 / 35 | 3.81e-06 | 56.3 | 6.8e-08 |
| `noahmp_land.eah` | Pa | 45 | 1 / 2 | 0.000122 | 2e+03 | 6.1e-08 |
| `land_history.GHB` | W m-2 | 17 | 1 / 1 | 3.81e-06 | 64.5 | 5.9e-08 |
| `land_history.IRB` | W m-2 | 3 | 1 / 1 | 3.81e-06 | 66.1 | 5.8e-08 |
| `land_history.SHB` | W m-2 | 9 | 1 / 1 | 1.91e-06 | 44.1 | 4.3e-08 |
| `State.el_pbl` | m | 2 | 2 / 2 | 9.54e-07 | 120 | 7.9e-09 |
| `State.qke` | m2 s-2 | 9 | 1 / 5 | 2.98e-08 | 25.1 | 1.2e-09 |
| `State.qsq` | (kg kg-1)2 | 2 | 2 / 3 | 8.13e-20 | 9.3e-08 | 8.8e-13 |

Constant route (b): the constants arm itself is not byte-identical to the default program. It leaves a separate
1–2 ULP residual in frozen layer-1 soil water only (TSLB 267.9–273.13 K; ISLTYP 16 ×9, 7 ×1). Traced ON equals
the default program bitwise in these two fields, so this residual is constant folding in the frozen-soil path,
not a value mismatch:

| Field | Units | Cells | ULP median / max | Max abs | Field max | Max abs / field max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `noahmp_land.smois` | m3 m-3 | 3 | 1 / 1 | 2.98e-08 | 0.439 | 6.8e-08 |
| `noahmp_land.sh2o` | m3 m-3 | 10 | 1 / 2 | 2.98e-08 | 0.439 | 6.8e-08 |

GPU gate (c), pre-registered 05:16Z before any GPU data (`receipts/GPU_ONDEFAULT_GATE.json`). The GPU prime
records the same table per domain, with the same code (`root_gpu.py` → `field_table.py`, record only). Every field
of every domain must satisfy max abs / field max ≤ 4e-6, which is 2× the CPU envelope of 1.9e-6. Cell counts are
disclosed as on CPU; a non-float difference fails. Evaluator `ondefault_gate.py`. If the bound is exceeded: stop
and report, no re-tuning. Launcher pin GPU_v7t @86c86274d1 (entries unchanged: mini 88a032e4, production 188cc488).
- [M] LS72 exact launch smoke of the v7 mini on the new pin GPU_v7t @86c86274d1 (own 24G scope, 1277 s): LAUNCH_SMOKE_PASS on the
  real model path on CPU, one AOT key, changed values load 0 new native blobs. The recorded `default_field_table`
  in prime, measure and profile is identical to the LS71 table (41 fields, max 1.91e-6). `ondefault_gate.py` →
  IN_CLASS: this is a CPU smoke result, not the GPU verdict. Receipts `receipts/LAUNCH_SMOKE_V7_LS72.json`,
  `receipts/LS72_ONDEFAULT_GATE_CPU_SMOKE.json`, manifest `receipts/GPU_V7T_MANIFEST.json`.
