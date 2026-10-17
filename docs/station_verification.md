# Station verification

`gpuwrf verify` extracts surface predictions and compares **both forecast sources
against station observations on identical support**. CPU-WRF and the driving
model are comparison forecasts. This offline command does not run a forecast or
change its compiled program. NumPy, pandas and netCDF4 perform the work; no JAX
arrays, GPU allocation or model compilation is needed by this package.

```bash
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 gpuwrf verify \
  --stations stations.csv --obs observations.csv \
  --model gpu/wrfout_d03_* --compare cpu/wrfout_d03_* \
  --output-dir verification
```

Select **one domain and one issue** per source. Multiple frames per file are
supported. Duplicate times, changing grids and mixed domains/issues are refused.
You may replace either wrfout list with one CSV of station predictions; this
supports driving models without guessing their grid, units or wind convention.
Use the same declared corrections for both sources. A station CSV is already
sampled/corrected by its producer; extraction flags never re-correct its values.

The command writes `verification.json`, `verification.md` and a standalone
`verification.html`. JSON records input SHA-256 hashes, source support counts,
station metadata, interpolation nodes/weights and corrections. A source with no
common pairs for a variable is **unavailable**, never a zero error. A completely
empty comparison exits 2. Invalid data exits 2. Descriptive results have status
`DIAGNOSTIC_ONLY_NO_PROMOTION`; there is no automatic release/skill verdict.

## Station metadata

```csv
station_uid,lat,lon,elevation_m,landmask,wind_height_m,roughness_m
example:station,28.4,-16.3,120,1,10,0.03
```

`station_uid` is a stable string, like ALISIOS's `source:native_id`. Latitude and
longitude are geographic degrees; elevation is metres above mean sea level.
Land mask is 1 for land, 0 for water. Wind height and roughness are metres above
the local surface. Optional defaults (recorded in the report) are land=1,
wind height=10 m and roughness=0.03 m. Supply observed site metadata when known.
ALISIOS station exports may use `station_id`, `altitude_m` and
`altitude_verified_m`; verified altitude takes precedence, as in its skill engine.

## Canonical observation/forecast CSV

```csv
station_uid,valid_time,variable,value,interval_start
example:station,2026-01-01T01:00:00Z,t2,18.2,2026-01-01T01:00:00Z
example:station,2026-01-01T01:00:00Z,wind,5.1,2026-01-01T01:00:00Z
example:station,2026-01-01T01:00:00Z,wind_direction,270,2026-01-01T01:00:00Z
example:station,2026-01-01T01:00:00Z,gust,8.2,2026-01-01T00:00:00Z
example:station,2026-01-01T01:00:00Z,precip_rate,0.4,2026-01-01T00:00:00Z
```

| Variable | Units | Definition |
|---|---|---|
| `t2` | °C | Air temperature at 2 m; WRF `T2[K] - 273.15` |
| `wind` | m/s | Scalar wind speed, WRF 10 m by default |
| `wind_direction` | degrees | Meteorological **from** direction clockwise from north |
| `gust` | m/s | Explicitly supplied gust statistic over the declared window |
| `precip_rate` | mm/h | Mean rate over `(interval_start, valid_time]` |

Timestamps must have UTC or an explicit offset; native ObsDB `ts_utc` is the
documented exception. All timestamps normalize to UTC. Instantaneous variables
use `interval_start=valid_time` (the column may be omitted if there is no rain).
For gust, `interval_start=valid_time` explicitly means an instantaneous diagnostic;
a window maximum must declare its start. Point gusts and interval maxima never
match. Rain requires an explicit positive interval. Overlapping rain intervals,
duplicate keys, negative speeds/rates, and NaN/Inf are rejected. Optional
`qc_flag` excludes nonzero flags before observations are ingested; exclusions are
counted. Missing variables are absent rows, not NaN placeholders.

Forecast CSVs also accept ALISIOS's `prediction_value`, `prediction_u_ms`,
`prediction_v_ms` and `prediction_dir_deg` (expanded into direction rows).
If components are supplied, both must be finite on every wind
row. A CSV must select one source and one issue. Scalar speed and mean-vector
components remain separate quantities. Without components, vector scores derive
u/v from the supplied scalar speed and direction; this assumption belongs to the
station CSV producer.

If the optional `issue_time` column is present, every row must declare the same
non-missing, nonempty issue. Omitting that column is supported for canonical CSVs.

## ALISIOS observations (read-only exports)

```bash
gpuwrf verify --stations alisios_stations.csv --obs obsdb_export.csv \
  --obs-format alisios --obs-precip-hours 1 \
  --model gpu/wrfout_d03_* --compare cpu/wrfout_d03_* --output-dir verification
```

Export `station_id` (or `station_uid`), `ts_utc`, `variable`, `value`, `qc_flag`
from the approved ObsDB authority; this tool never connects to or modifies the
live ALISIOS database. Native variables supported are `t2_c`, `wind_speed_ms`,
`wind_dir_deg`, `wind_gust_ms`, `precip_rate_mmh` and `precip_mm`.
Other variables and nonzero QC flags are excluded and counted. Duplicate native
station/timestamp/variable measurements are ambiguous and rejected.
Every admitted native speed, gust and precipitation measurement must be
nonnegative, and every native direction must be in [0,360], **before** hourly
aggregation. A valid aggregate cannot hide an invalid native measurement.

Within the UTC hour beginning at `h`, temperature/speed use arithmetic means,
direction uses the unweighted circular mean, and gust uses the maximum, following
ALISIOS `skill/engine.py::_observations`. Undefined opposing direction means are
reported as unavailable. Temperature/speed/direction keep the ALISIOS hour-start
label `h`. **Gust is converted to an explicit `[h,h+1h]` window ending at `h+1h`**;
this avoids matching an hourly maximum with a point history sample at `h`.
Precipitation requires `--obs-precip-hours`: rates/accumulations are declared to
represent an interval ending at the original `ts_utc`, not an assumed daily
counter. `precip_mm` is divided by that duration. Export only the correct native
precipitation channel; overlapping rate and amount rows are rejected.

## Optional METAR/SYNOP observation source

[Meteostat 2](https://dev.meteostat.net/python/api/meteostat.hourly) provides
explicit [METAR and NOAA ISD-Lite observation providers](https://dev.meteostat.net/providers).
NOAA ISD combines hourly/synoptic observations from multiple reporting networks
([NOAA source description](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)).
Install the optional extra, add a string `meteostat_id` column to stations, and
request a provider explicitly:

```bash
pip install 'gpuwrf[verify]'
gpuwrf verify --stations stations.csv --fetch-observations isd-lite \
  --start 2026-01-01T00:00:00Z --end 2026-01-02T00:00:00Z \
  --model gpu/wrfout_d03_* --output-dir verification
```

Use `--fetch-observations metar` for METAR. Station identifiers are preserved
including leading zeros. The string station ID lets Meteostat resolve native
provider identifiers/metadata; no interpolated Point is requested. The request uses UTC, only the selected
observation provider, `fill=False`, `clean=True` and metric units. It never requests
forecast providers or Meteostat's blended hourly dump (which can contain model
substitutes). Provider missing measurements are excluded/countable at ingestion;
Inf is refused. Metric speed/gust are converted from km/h to m/s. Provider rain
and gust windows are hourly and end at the record time. Downloaded canonical
`observations.csv` and provider metadata are retained for reproducibility.
Provider availability and measurement conventions remain the operator's concern;
no live-download qualification is claimed by the offline test suite.

## Sampling and corrections

Default `--method bilinear` inverts the four-node curvilinear lat/lon cell map
and applies its weights, with longitudes unwrapped around each station. This is
lat/lon quadrilateral interpolation; it does not claim to be WRF's projected
`ll_to_xy` interpolation. It preserves analytic bilinear cell fields. `nearest`
chooses the closest eligible mass node in the containing cell. Both refuse
out-of-domain stations; no distant extrapolation occurs. Land-aware sampling
requires WRF `LANDMASK`, drops nodes of the other surface class and renormalizes
remaining positive weights; it refuses cells with no compatible support.
`--no-land-aware` deliberately permits coastal mixing and is recorded.

Wind defaults to grid-relative `U10/V10`; it is rotated with WRF
`COSALPHA/SINALPHA` or the [NCAR wrf-python rotation equations](https://github.com/NCAR/wrf-python/blob/develop/fortran/wrf_user.f90)
and projection metadata (Lambert, polar stereographic, Mercator, unrotated lat/lon).
Unsupported projection/rotation metadata is refused. Use `--wind-frame earth`
only for inputs already rotated to east/north. Scalar speed is computed before
rotation (stored fp32 rotation fields need not have exactly unit norm) and interpolated as
`interp(hypot(u,v))`; components are interpolated independently. This matches
ALISIOS `_sample_wind_rows`, and retains sub-grid variance that would be lost in
`hypot(interp(u),interp(v))`. Direction is `atan2(-u_earth,-v_earth)` modulo 360.
An exactly cancelled/negligible mean vector has undefined direction and is
excluded from direction rows with a reported count, while its valid speed and
vector components remain available.

Corrections are explicit and disabled by default:

- `--lapse-rate 0.0065`: `Tstation = Tgrid - Γ (zstation - interp(HGT))`,
  Γ in K/m. This fixed environmental lapse assumption does not represent
  inversions or surface-layer stability; inspect corrected and raw skill.
- `--wind-height neutral-log`: multiply speed/components/gust by
  `ln(zstation/z0)/ln(10/z0)`. This assumes neutral flow, zero displacement and a
  declared station roughness. Default `as-is` compares raw 10 m wind; the report
  counts stations whose measured wind height differs from 10 m.
- `--gust-variable NAME --gust-window-hours H`: use an actual gust field, with
  its declared maximum window ending at output time (`H=0` for instantaneous).
  WSPD10MAX is a maximum of sampled speed, not automatically a gust. No gust is
  synthesized. Truth and forecast windows must match exactly.

Rain is `(RAINC + RAINNC [+ RAINSH])` differenced between adjacent output frames,
then divided by elapsed hours. Differences are checked before interpolation so
a local reset cannot be hidden by positive changes in adjacent cells. The first frame establishes a baseline, not a
rain amount. Negative differences/resets and nonzero bucket counters are refused;
reconstruct cumulative totals beforehand. There is no clipping or zero fill.
All rain sources and truth are integrated as piecewise-constant rates over exact,
fully covered windows. When comparing, the baseline's windows govern; otherwise
the candidate's windows govern. Gaps are unavailable. Instantaneous fields have
no time interpolation. WRF reads are streamed one frame at a time; only surface
arrays and station rows are retained, with no full 3D history loaded.

## Score definitions and evidence limits

Every scalar score uses `error = forecast - observation`, with bias = mean(error),
MAE = mean(abs(error)), RMSE = sqrt(mean(error²)). Vector wind RMSE is
`sqrt(mean((uf-uo)² + (vf-vo)²))`, without division by two. Direction errors wrap
to `[-180,180)` and report bias/MAE/RMSE plus **median absolute circular error**.
Direction requires observed and **every forecast** speed strictly above 4 m/s,
as in ALISIOS `_direction_stats`; the same eligible rows are used for both arms.
Vector wind has its own reported N when direction truth is unavailable.

Rain events use inclusive `rate >= threshold` in mm/h (defaults 0.1, 1, 5).
Given hits H, misses M, false alarms F and correct negatives C,
`Hrandom=(H+M)(H+F)/N`, `ETS=(H-Hrandom)/(H+M+F-Hrandom)`, and
`FBI=(H+F)/(H+M)`. Undefined denominators are JSON null, not NaN or a perfect
score. These are the standard Gilbert equitable threat score and frequency bias
definitions from the [CAWCR verification guide](https://www.cawcr.gov.au/projects/verification/).
Scores use rate thresholds, not mixed 1 h/3 h accumulation thresholds.

Comparison reports give candidate-minus-baseline RMSE and its ratio on exact
common support; a zero baseline RMSE has no defined ratio. Aggregate scores are
pooled station/time pairs, not normalized multi-variable skill-ladder scores.
This tool does not reproduce ALISIOS publication eligibility, HQ evidence floors,
date-block confidence intervals or issue-date weighting. Correlated station-hours
are not independent cases; small descriptive improvements do not prove superiority.
Formal adoption still uses the project's real-station skill process.

Tests use committed hand-computed score values, analytic interpolation fields,
the NCAR rotation formula, unit/window/QC failure cases and an actual subprocess
CLI. GPU forecasts are outside this K0 lane's qualification.
