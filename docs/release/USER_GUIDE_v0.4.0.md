# v0.4.0 quickstart and command guide

Use the guide matching your installed release. This is the v0.4.0 released
guide: frozen defaults are qualified, and all 66 registered configurations now pass their accepted scientific
scope (65 full 24-hour + c45 accepted six-hour). See the [support matrix](SUPPORT_MATRIX_v0.4.0.md),
[66-case ledger](FINAL_COMPOSITES_v0.4.0.md) and
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
use. GitHub release assets currently await owner authentication; remote
runtime-table fetches fail until upload. Their cache root is `GPUWRF_TABLES_DIR` or the default user cache. For an
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

## Prepare ensembles and independent cases

For already prepared cases, create a reproducible member plan and launch it:

```bash
gpuwrf ens prepare --mode batch --cases CASE_A CASE_B \
  --output-dir runs/member_plan --hours 24
gpuwrf ens launch --plan runs/member_plan/plan.json
```

`--mode physics` uses a common case and selected namelists; `--mode lagged`
uses the declared case times. Inspect `ens prepare --help` and the
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

Plan an admitted, date-held-out regional physics study from its JSON definition:

```bash
gpuwrf optimize plan --study study.json --output runs/study_plan.json
```

`optimize run` schedules the prepared forecasts; `optimize score` consumes
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

The [support matrix](SUPPORT_MATRIX_v0.4.0.md) is the authoritative feature
list. Default qualification, per-composite D6, E41-disclosed resources,
CPU-only/API-only and refused options are distinct. The
[validation ledger](FINAL_COMPOSITES_v0.4.0.md) retains actual clocks and
sources; the [fix list](FIXES_v0.4.0.md) links original-WRF evidence and failures.

In the final registered suite,65 configurations pass full 24-hour scientific
scope and c45 passes its accepted six-hour scope.
Bounded D6 does not cover every inventoried field: unbounded fields remain
REPORT_ONLY, missing fields remain incomplete, and output finiteness does not
justify intermediate negative-event repairs. E41 and F2 disclosures remain in
[known issues](../../KNOWN_ISSUES.md).
