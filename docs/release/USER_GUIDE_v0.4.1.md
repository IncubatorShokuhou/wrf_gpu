# v0.4.1 quickstart and command guide

Use the guide matching your installed release. This is the v0.4.1 guide (released
18 October 2026). The default physics and its validation are those of v0.4.0, apart from the
default-path fixes listed in the release notes; v0.4.1 adds opt-in options and tools,
marked **new in v0.4.1** below. See the
[support matrix](SUPPORT_MATRIX_v0.4.1.md), the
[release notes](../../release_notes/RELEASE_NOTES_v0.4.1.md) and the v0.4.0
[upgrade note](UPGRADE_v0.4.0.md) before changing an existing operational setup.

## Install and check the case

The GPU fast path requires NVIDIA Ampere or newer, compute capability >=8.0.
Pascal and Turing are unsupported. Capacity depends on grid, selected physics
and cold compilation; it is separate from architecture admission.

For the tested CUDA 13/JAX 0.10 setup, use Python 3.11+ and a compatible NVIDIA
driver. Install into an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax[cuda13]==0.10.*"
python -m pip install -e .
export GPUWRF_WRF_ROOT=/path/to/WRF

gpuwrf --help
gpuwrf namelist-support --json
```

Run these commands from the cloned repository. `GPUWRF_WRF_ROOT` supplies the
original WRF runtime/source tables needed by selected options. Large packaged
physics tables use pinned, content-verified assets fetched on first selected
use from the GitHub release of your version. Their cache root is `GPUWRF_TABLES_DIR` or the default user cache. For an
offline installation, prefill the matching pinned assets and set
`GPUWRF_TABLES_OFFLINE=1`; a missing or mismatched asset stops the run. CAM-MG
`mp_physics=11` additionally requires `triton==3.6.*`. Optional product tools
have their own extras, documented in their linked guides.

`namelist-support` reports backend admission, not coupled D6 qualification.
The versioned support matrix attaches scientific and resource evidence to
exact combinations. Do not infer all-combination support from a catalog entry.

## Run the bundled forecast

The small Swiss example includes a prepared `namelist.input`, `wrfinput_d01`
and `wrfbdy_d01`. Inspect it and write output to a new directory:

```bash
gpuwrf doctor examples/switzerland_d01
gpuwrf run --input-dir examples/switzerland_d01 \
  --output-dir runs/switzerland --domain d01 --hours 24 \
  --scratch-dir runs/switzerland_scratch
```

The first invocation compiles before advancing. Later runs reuse compatible
executables, but a new grid, physics selection or version can require a new
compile. Cold compiler host RAM and warm forecast memory are different
budgets. The [c11 resource record](../../KNOWN_ISSUES.md#c11-resolved-compiler-stall-and-residual-resources)
retains the old compile cap and the successful changed-source qualification.

Read the [example notes](../../examples/switzerland_d01/README.md) for dates,
selected schemes and expected output. History uses familiar WRF variable
names and dimensions. Headers alone do not establish physical agreement.

## Run your own prepared case

Prepare a real WRF case with WPS/real first, or use the `quickstart` onboarding
workflow below. Supply `namelist.input`, `wrfinput_d0*`, `wrfbdy_d01` and any
selected forcing/static bundles. For fixed nests:

```bash
gpuwrf doctor /path/to/case
gpuwrf manifest /path/to/case --output runs/case_manifest.json
gpuwrf reproduce runs/case_manifest.json --case-dir /path/to/case
gpuwrf run --input-dir /path/to/case --output-dir runs/my_case \
  --domains-from-namelist --hours 24
```

`doctor` reports input, support, CFL and memory checks. `manifest` creates
input/source/environment provenance; `reproduce` verifies it without starting
a forecast. These commands do not qualify a new configuration scientifically.
Use fresh output paths and keep the case namelist with the run receipt.

A case with at least two CPU `wrfout` histories can select compatibility replay.
Keep reference outputs outside the standalone input directory and check the
reported mode. For a single domain use `--domain`; for the full namelist nest
set use `--domains-from-namelist`.

## Restart and extend a forecast

Native checkpoints retain selected active state and held diagnostics. Schema,
source/configuration binding, domain shape and data checks are enforced.
Do not edit a checkpoint to bypass a refusal.

```bash
gpuwrf run --input-dir /path/to/case --output-dir runs/my_case \
  --domains-from-namelist --hours 24 \
  --checkpoint-dir runs/checkpoints --checkpoint-interval-steps 120
```

Resume with the verified generation path written by that run:

```bash
gpuwrf run --input-dir /path/to/case --output-dir runs/my_case \
  --domains-from-namelist --hours 24 \
  --resume-checkpoint runs/checkpoints/VERIFIED_GENERATION
```

`VERIFIED_GENERATION` denotes the actual authenticated generation directory,
not a literal directory created by this guide. `--extend-run` permits a later
forecast end with a matching resume; it does not extend available boundary or
forcing data. Old active checkpoints may be incompatible after the new carry
fields described in the [upgrade guide](UPGRADE_v0.4.0.md). Corrected historic
precipitation totals require a fresh forecast.

## New namelist options in v0.4.1

Three opt-in `&physics` settings are new. Leaving them out keeps the v0.4.0
behaviour.

```text
&physics
 topo_wind = 1, 1, 1,   ! per domain; only with bl_pbl_physics = 1 (YSU) or 11 (Shin-Hong)
 aer_opt   = 1,         ! one value; needs ra_sw_physics = 4 on every domain
 icloud    = 3,         ! one value; needs ra_lw_physics = 4 and ra_sw_physics = 4
/
```

- `topo_wind=1` needs `VAR_SSO` in `wrfinput_d0N` (WPS geogrid with the `varsso`
  data); without it the run is refused, as in WRF. `topo_wind=2` uses the
  geogrid `VAR`. With a boundary layer other than YSU or Shin-Hong the option
  has no effect and a warning is printed.
- `aer_opt=1` uses the shipped Tegen climatology; `aer_opt=3` needs
  `mp_physics=28` on a single domain and writes `TAOD5502D`; `aer_opt=2` is
  refused. See the [aerosol guide](../AEROSOL.md).
- `icloud=3` refuses `mp_physics=0/1/3`. Both `icloud=2` and `icloud=3` passed one 24-hour
  comparison with CPU-WRF (Tenerife, three domains).
  See [radiation cloud fraction](../radiation-cloud-fraction.md).

`gpuwrf doctor` and `gpuwrf run --dry-run` check these pairings before any GPU
work. Status per option: [support matrix](SUPPORT_MATRIX_v0.4.1.md#namelist-options).

## Ensemble data assimilation (new in v0.4.1, experimental)

`gpuwrf da` writes a LETKF analysis `wrfinput_d01` from an ensemble of
forecasts valid at the analysis time and your observations (radiosondes,
surface stations, CSV/JSON tables). Outer domain only; the analysis runs on
the CPU.

```bash
gpuwrf da check   --config da.json            # validates and lists every input
gpuwrf da analyse --config da.json --out analysis_00z
gpuwrf da cycle   --config cycle.json --out runs/da_cycle --dry-run
```

`da cycle` runs forecast/analysis windows with all members of a window in one
process on one compiled program; `da members` runs a member manifest. The
configuration schema, observation formats and the moisture rules are in the
[data-assimilation guide](../da.md); the optional ensemble settings and their
known overdispersion are in [ensemble options](../da_ensemble.md); the cycle
layout is in [cycling in one process](../da-cycle.md). A better forecast is not
established; verify against independent observations first.

## Noah-MP surface parameters without recompiling (new in v0.4.1, experimental)

```bash
JAX_PLATFORMS=cpu gpuwrf surface-params basis --input-dir CASE --output basis.npz
gpuwrf surface-params expand --basis basis.npz --output rough.npz \
    --class z0mvt:2=1.25 --set albdry=0.95      # entries without a prefix are d01
gpuwrf run --input-dir CASE --output-dir OUT --surface-controls rough.npz
```

`gpuwrf run` without domain options runs d01 only, so this control file holds d01
entries only. For a nested case, prefix entries with their domain (`--set
d02:albdry=0.95`) and run all domains with `--domains-from-namelist`; a control
for a domain outside the run is refused. Every parameter set on one grid reuses
the same compiled program. Eleven
Noah-MP parameters can be controlled (`sf_surface_physics=4` only). A resume
with different, added or missing controls is refused. The ensemble Kalman
inversion tools (`surface-params eki`, `osse`) are checked on synthetic twin
experiments only. See [surface parameters](../surface_params.md).

## Check compiled-program reuse (`gpuwrf aot-key`)

```bash
gpuwrf aot-key --input-dir /path/to/case_a \
  --compare-input-dir /path/to/case_b --json keys.json
```

`aot-key` prints the cache key of every compiled step program of a nested case
(the outer domain, each nest, and inner domains compiled together with their
nests). With a second case it prints `SHARED` or `NOT SHARED` per program and
names the part of the key that differs. Use it before a run to see whether a
compile cache or bundle prepared on one case serves another: shared programs
are loaded, the others are compiled. It runs on the CPU, compiles nothing and
runs no forecast step; comparing two Tenerife three-domain cases takes about 6
minutes.

`--namelist` and `--compare-namelist` select another `namelist.input`;
`--hours` sets the forecast length the keys are computed for (default 3). The
exit code is 0 when every key was computed and 3 when a program is
`UNAVAILABLE`, which is never counted as shared. Nested cases only. The keys
carry the CPU target, so compare cases with each other, not with the file names
in a GPU cache. In v0.4.1 all forecast dates of one grid share their programs;
with v0.3.x, run with `GPUWRF_AOT_VERIFY=1` when one cache serves several dates
(see [known issues](../../KNOWN_ISSUES.md#v030v034-nested-programs-from-a-compile-cache-shared-across-dates)).

## Prepare ensembles and independent cases

For already prepared cases, create a reproducible member plan and launch it:

```bash
gpuwrf ens prepare --mode batch --cases CASE_A CASE_B \
  --output-dir runs/member_plan --hours 24
gpuwrf ens launch --plan runs/member_plan/plan.json
```

`--mode physics` uses a common case and selected namelists; `--mode lagged`
uses the declared case times. New in v0.4.1, two presets write the member
namelists from one case:

```bash
gpuwrf ens prepare --mode stochastic --case CASE --members 10 \
  --schemes skebs,sppt --lengthscale-km 15 --hours 24 --output-dir runs/stoch
gpuwrf ens prepare --mode multiphysics --case CASE --members 7 \
  --hours 24 --output-dir runs/physics
```

`--mode stochastic` gives every member distinct WRF stochastic-physics seeds;
SKEBS, SPPT and SPP passed a single-domain 24-hour comparison with CPU-WRF
using the same seeds ([stochastic guide](../stochastic.md)). `--mode
multiphysics` uses rows of a table of physics combinations from the v0.4.0
checks. Inspect `ens prepare --help` and the
[ensemble guide](../ensemble.md) for inventory formats and member semantics.
The scheduler checks GPU/host budgets and queues work. It supports independent
cases; real multi-GPU scaling remains separately unqualified.

After forecasting, compute products from an explicit member inventory:

```bash
gpuwrf ens products --members members.json --variable T2 \
  --output runs/ensemble_T2.nc
```

The member inventory defines IDs and output paths. Exact alignment is the
default; intersection alignment deliberately changes the compared support.
`ens verify` reports probabilities, scores and reliability against declared
observations. Statistical tools are not a claim of improved forecast skill.

## Weather products, verification and export

List supported derived products, then select the products appropriate to your
fields and assumptions:

```bash
gpuwrf post --list-products
```

[Forecast products](../post.md) and [application products](../post_apps.md)
explain inputs, units, optional dependencies and interpretation. `verify`
compares forecast data with canonical station observations:

```bash
gpuwrf verify --stations stations.csv --obs observations.csv \
  --model runs/my_case/wrfout_d01* --output-dir runs/verification
```

See [station verification](../station_verification.md) for wind frame,
precipitation interval, station-height and land-aware sampling conventions.
A download is requested only with `--fetch-observations`; plain `--obs` is
an offline observation file.

For a surface subset in CF NetCDF:

```bash
gpuwrf export --input runs/my_case/wrfout_d01* \
  --output runs/surface.nc --format netcdf --variables T2 PSFC U10 V10
```

`export --help` covers pressure levels, regions, Zarr and explicit GRIB grids.
`--raw` preserves native WRF dimensions/attributes and makes no CF claim.
`tslist-export` converts existing station text output; `stream-check` validates
regional stream definitions without starting a forecast.

## Regional optimisation and land/snow cycling

The executable commands are **`optimize`** and **`cycle`**. Onboarding is
**`quickstart`**, not a separate `onboard` command.

List the physics configurations a study may use, then plan an admitted,
date-held-out regional physics study from its JSON definition:

```bash
gpuwrf optimize released --max-dom 1 --e41 PASS
gpuwrf optimize plan --study study.json --output runs/study_plan.json
gpuwrf optimize run --study study.json --output runs/regional-study --chunk-minutes 60
gpuwrf optimize run --study study.json --output runs/regional-study --resume
```

New in v0.4.1, `optimize run` runs the forecasts itself, as many at once as
GPU and host memory allow, and can pause after a time budget (exit code 3)
and resume. `optimize score` consumes forecasts; `optimize score` consumes
forecasts rather than manufacturing them. `calibrate fit/apply` provides
chronological station bias/MOS workflows. Read the
[regional optimisation guide](../regional_optimization.md) for study schemas,
training/validation separation and explicit admission rules. A prepared study
or successful tool invocation is not a new held-out skill result.

Cycle compatible land/snow state into a **new** input file:

```bash
gpuwrf cycle --source previous_cycle/wrfout_at_new_init \
  --wrfinput new_case/wrfinput_d01 --output new_case/wrfinput_cycled_d01
```

The source must match the new initialization time and grid. The output does
not overwrite the template or automatically replace the next case input.
Stage it as the new case's selected `wrfinput_d01` only after checking the
[cycling guide](../cycling.md). Cycling is distinct from crash restart and
from atmospheric data assimilation.

## Onboard a region with WPS/real

Inspect memory profiles and the preparation interface first:

```bash
gpuwrf quickstart --profiles
gpuwrf quickstart --help
```

`quickstart` plans by default. `--prepare` explicitly requests download and
WPS/real execution; `--run` additionally requests the GPU forecast. Provide
the requested region, UTC cycle, WPS/WRF/geography roots and source. Check
source availability and download limits before execution. The
[onboarding guide](../onboarding.md) explains GFS/IFS/AIFS fields, external
preprocessors and prepared-case outputs.

## Interpret the qualification record

The [v0.4.1 support matrix](SUPPORT_MATRIX_v0.4.1.md) is the authoritative feature
list; its v0.4.0 rows are carried forward unchanged. Default qualification, per-composite D6, E41-disclosed resources,
CPU-only/API-only and refused options are distinct. The
[validation ledger](FINAL_COMPOSITES_v0.4.0.md) retains actual clocks and
sources; the [fix list](FIXES_v0.4.0.md) links original-WRF evidence and failures.

In the final registered suite,65 configurations pass full 24-hour scientific
scope and c45 passes its accepted six-hour scope.
Bounded D6 does not cover every inventoried field: unbounded fields remain
REPORT_ONLY, missing fields remain incomplete, and output finiteness does not
justify intermediate negative-event repairs. E41 and F2 disclosures remain in
[known issues](../../KNOWN_ISSUES.md).
