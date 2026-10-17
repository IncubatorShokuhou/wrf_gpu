# Regional physics selection and station calibration

`gpuwrf optimize` selects among explicitly GPU-admitted physics configurations.
`gpuwrf calibrate` fits station bias correction or simple linear model output
statistics (MOS). Offline planning, scoring and calibration use NumPy/pandas and
the merged station verifier. They add no model kernels, transfers, temporary
device arrays or compilation. Model execution delegates to the existing
ensemble/batch scheduler. This implementation is tested on the CPU; the full
workflow has not yet been demonstrated end to end on a GPU. It does not certify regional forecast
improvement or promote a scheme to release status.

## Study and admission

Prepared WRF cases must already exist. The tool does not download observations,
run WPS, change the original namelists or edit model code. Supply a canonical
station metadata CSV and local observation CSV as defined in
[station verification](station_verification.md). Choose one initialization per
UTC date, at least two earlier training dates and two later held-out dates.
The declared date must match the namelist and wrfinput initialization. Requested
hours must contain integral root time steps and have complete boundary coverage
(dt=54 seconds requires multiples of three hours).

Example study (paths resolve relative to its JSON):

```json
{
  "schema": "gpuwrf.optimize.study.v1",
  "region": "my-region",
  "objective": "T2",
  "hours": 24,
  "budget_case_hours": 144,
  "test_start": "2026-02-01",
  "stations": "stations.csv",
  "observations": "observations.csv",
  "admissions": ["qualified-control.json", "admitted-alternative.json"],
  "cases": [
    {"id": "jan10", "date": "2026-01-10", "input_dir": "cases/jan10", "domain": "d01"},
    {"id": "jan20", "date": "2026-01-20", "input_dir": "cases/jan20", "domain": "d01"},
    {"id": "feb10", "date": "2026-02-10", "input_dir": "cases/feb10", "domain": "d01"},
    {"id": "feb20", "date": "2026-02-20", "input_dir": "cases/feb20", "domain": "d01"}
  ],
  "sampling": {"method": "bilinear", "land_aware": true, "wind_frame": "grid"}
}
```

An admission is an **operator-supplied qualification decision**, not an
optimizer-generated scientific verdict. Obtain the source binding with
`JAX_PLATFORMS=cpu gpuwrf optimize identity`. Each admission file supplies the exact
seven physics selections (scalars broadcast to all domains, or explicit lists),
domain count, version, source SHA256 and existing GPU evidence identities:

```json
{
  "schema": "gpuwrf.optimize.admission.v1",
  "status": "GPU_QUALIFIED",
  "gpuwrf_version": "0.4.0",
  "model_sha256": "<source identity from the measured model>",
  "max_dom": 1,
  "physics": {
    "mp_physics": 8, "bl_pbl_physics": 5, "sf_sfclay_physics": 5,
    "sf_surface_physics": 4, "ra_lw_physics": 4, "ra_sw_physics": 4,
    "cu_physics": 0
  },
  "evidence": [{"path": "gpu-validation.json", "sha256": "<actual evidence SHA256>"}]
}
```

`GPU_SMOKED_ADMITTED` is also accepted for an explicitly admitted experimental
configuration and remains labelled that way in the recommendation. It does not
mean a coupled forecast passed. CPU-oracle-only and generic catalog
`IMPLEMENTED` status cannot grant admission. Source/version/domain drift or
changed evidence refuses the study. The tool checks identities and the current
operational catalog, pairing, binding and nested restrictions; it cannot
authenticate the scientific content or authority of an arbitrary local admission
file.
The operator must bind evidence for the actual configuration, input regimes and
other unchanged case settings, including dynamics, cadence and land options.

By default the admitted configurations form the search. An optional
`search_space` maps **all seven** physics keys to lists of candidate options;
their Cartesian product is generated and intersected with admissions before
WRF validation. A list-valued per-domain option must be nested inside the
options list, e.g. `"cu_physics": [[1, 0, 0], [5, 0, 0]]`. No option without an
admission is run automatically. At most 4096 combinations are generated.

## Planning, existing forecasts and execution

```bash
JAX_PLATFORMS=cpu gpuwrf optimize plan --study study.json --output plan.json
JAX_PLATFORMS=cpu gpuwrf optimize score --study study.json \
  --forecasts forecasts.json --output selection.json
```

The plan reports stable `p-...` configuration IDs, rounds, rejected pairings,
admission provenance and the required budget. Existing forecast inventory:

```json
{"schema": "gpuwrf.optimize.forecasts.v1", "forecasts": {
  "p-<ID from plan>": {
    "jan10": ["control/jan10/wrfout_d01_2026-01-10_00:00:00", "control/jan10/wrfout_d01_2026-01-10_01:00:00"],
    "jan20": ["control/jan20/stations.csv"],
    "feb10": ["control/feb10/stations.csv"],
    "feb20": ["control/feb20/stations.csv"]
  }
}}
```

List complete selected-domain history files or one canonical station forecast
CSV per case. Populate every required training arm and the eventual winner's
held-out arms. Missing arms fail rather than shortening a round. Offline labels
associate supplied files with configurations; the tool records file hashes but
does not attest that externally supplied forecasts ran the labelled physics.
The `run` workflow prepares and records its actual private namelists instead.

```bash
gpuwrf optimize run --study study.json --output regional-study --max-parallel 1
```

No GPU work is required to plan or score. For execution use your normal GPU
environment, not `JAX_PLATFORMS=cpu`. The existing launcher controls GPU and RAM
admission. A round submits a candidate's new
dates as a batch; independent dates can run in parallel, candidates are processed
sequentially. Outputs, source manifests, selected namelists, budget charges and
launcher records remain under the new output directory. Failed batches stop
without publishing a recommendation. Existing destinations are refused.

The compute budget is integer **case-hours**, not a guaranteed wall time or GPU
energy cap. Compile, initialization, IO and scheduler waits are additional wall
cost. The plan reserves all held-out dates for one winner. Successive halving
starts with two training dates (three when necessary to avoid a singleton batch),
roughly doubles dates per survivor and retains the best half, rounded up.
Already evaluated dates are reused. Training ends when all training dates are
used; the best remaining configuration is then evaluated once on all test dates.
Ties break by stable configuration ID. Later dates never determine which
configuration wins. Small training sets can eliminate configurations that would
improve on other weather regimes; choose representative dates in advance.
This allocation follows the
[successive-halving resource principle](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html).

## Objectives, support and uncertainty

Objectives: `T2` (degC, temperature errors also K), `WS10` (scalar speed m/s),
`precip` (total precipitation rate mm/h) and `solar` (instantaneous downward
shortwave `SWDOWN`, W/m²). `all` combines all four and requires explicit positive
`scales`, e.g. `{"t2":1,"wind":1,"precip_rate":1,"solar":100}` in those units.
A single objective defaults to scale 1 in its physical units. The loss is
the mean of `station-balanced RMSE / scale` across selected variables. Each
station has equal weight within a variable; its available errors receive equal
weight. All candidates in a round use the exact same station/time/variable/rain
interval intersection, including case ID. Counts and per-variable RMSEs are
reported. At least two stations and support on every requested case are required.

WRF history extraction reuses `verify`: bilinear/nearest land-aware sampling,
optional explicit lapse-rate and wind-height correction, scalar speed sampled
before vector norm, and complete total rainfall differencing. Missing intervals
are unavailable, accumulation resets/buckets are refused. Observed rain rates
integrate onto the forecast's complete intervals. Solar is an isolated extension
using the same sampler; canonical CSV variable `solar` means instantaneous W/m²,
not hourly mean irradiance or PV output. UTC/offset timestamps, unique keys,
finite numbers and registered stations are mandatory. No NaN can disappear in
an aggregation. Solar-only history should include a regular surface diagnostic
such as T2, as required by the shared verifier.

Initialization rows are read for rainfall differencing but excluded from skill.
Both issue dates and actual scored intervals respect `test_start`: training
valid times precede it, held-out interval starts are at/after it. The held-out
result includes a reproducible 400-replicate UTC date-block bootstrap 95%
interval (seed 0). All stations of a date are resampled together. This interval
describes the selected configuration on supplied test dates; correlated days,
few dates, station selection and repeated studies can understate uncertainty.
It is not proof of improvement against CPU-WRF or a release check.

## Station calibration

```bash
JAX_PLATFORMS=cpu gpuwrf calibrate fit --stations stations.csv --obs past-obs.csv \
  --forecasts past-forecasts.csv --method bias --output station-model.json
JAX_PLATFORMS=cpu gpuwrf calibrate apply --model station-model.json \
  --forecasts future-forecasts.csv --output corrected-stations.csv
```

Historical CSVs contain one selected forecast per station/time/variable/interval,
across dates. The exporter must choose initialization and lead consistently;
this first version does not fit separate lead/season/regime models. Temperature,
scalar speed, rain rate and solar are supported; vector wind/direction is not.

Per station/variable, `bias` predicts `x + mean(y-x)`.
`mos` fits `a+b*x`, where `b=sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))²)` and
`a=mean(y)-b*mean(x)`. These are
[ordinary least-squares coefficients](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm).
MOS refuses constant predictors. At least three earlier training dates per
scored group are needed. Expanding chronological cross-validation fits only
dates before each next validation date, excludes intervals crossing its start,
and records every fold's coefficients, support, raw/corrected RMSE and bias.
It never scores a fitted row as a held-out row. Final deployment coefficients
refit all historical pairs after the validation scores are computed.
Every exported station/variable must have at least one chronological held-out
pair. Fitting refuses the complete model if a group lacks that evidence, even
when other stations have validation scores. Application also checks per-group
held-out support, refusing groups without that support in previously saved models.

Speed, rainfall rate and solar predictions apply the documented statistical
constraint `max(0,a+b*x)`; affected counts are reported. This is a station product
constraint, never a model-state repair. Temperature is unconstrained. Application
requires every valid time and interval start to follow the last fitted valid
time. Unknown station/variable groups fail rather than borrowing coefficients.
The output is canonical scalar CSV plus a provenance sidecar; raw vector wind
components are omitted because they would disagree with corrected scalar speed.
All training/model/forecast files carry SHA256 identities. Past correction skill
does not guarantee future improvement; examine the held-out scores.

Memory scales with extracted station rows and the candidates surviving a round,
not retained full forecast grids. Grid readers stream frames; input hashing and
NetCDF reads have CPU/disk cost. Bootstrap computations operate on station rows.
Actual execution memory/compile cost is that of the admitted physics and existing
batch launcher; this feature makes no new GPU speed or memory claim.
