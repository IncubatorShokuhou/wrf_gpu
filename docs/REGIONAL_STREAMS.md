# Regional output streams (development v0.4)

Put `gpuwrf.streams.json` beside `namelist.input` for a native `gpuwrf run`.
The feature is opt-in. Hourly full-domain history keeps its normal cadence.
Check a file before a forecast with `gpuwrf stream-check --input gpuwrf.streams.json --dt 18`.

```json
{
  "version": 1,
  "streams": [{
    "name": "airport",
    "domain": "d01",
    "interval_steps": 1,
    "buffer_frames": 32,
    "region": {"stations": [{"id": "runway", "i": 20, "j": 20}]},
    "variables": {
      "T2": ["instant", "min", "max", "mean"],
      "RAINNC": ["instant", "accumulation"]
    }
  }]
}
```

Alternatively use `interval_seconds` (for example 300), which must be an exact
multiple of that domain's timestep. `interval_steps: 1` samples every own step.
Each stream has a unique name and exactly one domain. Multiple streams may have
independent regions, variables and intervals. `variables: ["T2", "U10"]` selects
instantaneous values only. Unknown settings and unavailable fields are refused.

Regions select mass-grid centers; they do not average spatially:

| Region | Definition |
|---|---|
| `{"bbox": [-17, 27, -16, 29]}` | Inclusive west, south, east, north in degrees |
| `{"polygon": [[-17,27],[-16,27],[-16,29]]}` | Longitude/latitude vertices; inclusive boundary |
| `{"stations": [{"id":"site", "lat":28.3, "lon":-16.6}]}` | Nearest mass center by locally scaled lon/lat distance; outside coordinate envelope refused |
| `{"stations": [{"id":"site", "i":20, "j":20}]}` | Exact WRF 1-based mass-grid indices; useful with tslist locations |

Bboxes/polygons do not wrap the dateline. An empty region is refused. Geographic
station lookup is a nearest-grid convenience, with its selected i/j and grid
coordinates recorded in the output; it does not reproduce WRF tslist projection
rounding. Use exact i/j when matching an existing tslist station.

Available fields are T2 (K), Q2 (kg/kg), U10/V10 (m/s), PSFC (Pa), PBLH (m),
HFX/LH/SWDOWN/GLW (W/m²), TSK (K), and RAINC/RAINNC (mm). Surface and radiation
diagnostics are the held in-step fields, without running physics again. Their
cadence therefore follows the forecast physics; requesting a faster stream does
not increase radiation calls. TSK is State's skin temperature. Visibility and
other derived diagnostics are currently refused.

Statistics use uniform post-solve own-step samples in `(window_start, window_end]`,
before nest feedback, with windows anchored to forecast start. `instant` is the
last sample, `min`/`max` are extrema, and `mean` is the arithmetic sample mean.
`accumulation` is only allowed for RAINC/RAINNC and means **end minus start** of
the cumulative field, preserving negative differences. It is not a sum of
cumulative rain values. Any sampled NaN/Inf fails the stream, including interior
samples in an instantaneous-only window. There is no lead-zero frame. A final
incomplete interval is retained in a checkpoint; it is not published as a full
interval. End-time extension continues that interval.

Output is compact NetCDF under `OUTPUT/streams/wrfout_stream_NAME_DOMAIN_FIRST_LAST.nc`.
Each file holds a bounded chunk of complete intervals with `Time` and `point`
dimensions, Times, CF time bounds, XLAT/XLONG, point IDs and 1-based i/j. Instant
fields keep their WRF names; statistics add `_min`, `_max`, `_mean`, or
`_accumulation`. These are regional products, not full-grid WRF restart/history
files. WRF auxhist variable selection, independent alarms and frame packing map
to this subset; polygon selection and window statistics are extensions.

Regions are selected once on the host. Samples, reductions, rainfall baselines
and the bounded frame buffer stay in the device carry. Transfer occurs only at
host segment boundaries, never inside the timestep executable; a bounded async
writer serializes NetCDF access using the process-wide lock. `buffer_frames`
sets the maximum retained frames per stream, increased only if necessary to fit
one root-step's child frames. Memory is proportional to selected points,
channels and buffered frames. Very frequent large regions increase traffic and
output size; small regions are the intended use.

Stream names and coordinate checksums stay outside static timestep metadata.
Moving a region or renaming a stream with the same point count, variables and
interval can reuse the compiled program; restart still binds the exact stream
file and checks selected coordinates. Changed shapes or variable/statistic
selections require a new executable. Buffer flushes preserve committed placement.

Native B=1 single-domain/nested runs and operational checkpoint/resume are
supported. Checkpoints preserve partial windows and authenticate already
published chunks. Resume verifies committed files, removes only authenticated
later chunks, and refuses foreign/corrupt collisions or configuration changes.
Both internal pickle and NetCDF carry schemas preserve the device window state;
importing a generic carry does not recover its output journal. Homogeneous
batching and CPU-history replay are refused. WRF tslist can coexist on fresh
runs; tslist's own restart restriction still applies.

Qualification receipts are in `proofs/u2_streams/`; see the lane handoff for
measured scope. A component comparison with original CPU-WRF auxhist fields
qualifies sampling/window arithmetic, not a long coupled forecast or station RMSE.
