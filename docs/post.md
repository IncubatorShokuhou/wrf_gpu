# Offline forecast products

`gpuwrf post` reads existing CPU-WRF or gpuwrf history and writes one
CF-annotated NetCDF file. It never advances a forecast. NumPy/xarray handle the
data; thermodynamic products additionally require the optional MetPy extra:

```sh
pip install 'gpuwrf[post]'
gpuwrf post --list-products
gpuwrf post --input wrfout_d01_* --output products.nc \
  --product wind10,wind-height,cloud,precip,snow \
  --heights 100 200 --rolling-hours 1 3 6
gpuwrf post --input wrfout_d01_* --output convection.nc \
  --product instability,srh,freezing,thermal --storm-u 8 --storm-v 3
```

Use files from **one stationary domain and one accumulation origin** per call.
Input files are sorted by their internal UTC `Times`, including multiple records
per file. Duplicate times, changing grids, inconsistent fields and mismatched
bucket/origin metadata are rejected. Quoting a glob prevents shell expansion;
pass expanded paths. The minimal product `wind-speed10` only needs U10/V10.
`--product` can be repeated. Application modules may add names to the same registry.

## Input and output conventions

Inputs use standard WRF names, dimensions and units: pressure Pa, temperatures K,
geopotential m²/s², winds m/s, dry-air mixing ratios kg/kg, water-equivalent
accumulations mm, snow depth m and snow water mass kg/m². Unlabelled legacy WRF
variables are interpreted in these units. Unit conversion of differently encoded
files is the caller's responsibility. Potential temperature is `T+300`, and
absolute temperature is `(T+300) * ((P+PB)/100000) ** (287/1004.5)`, using ARW
Rd/Cp. Mass height is the adjacent-interface mean of `(PH+PHB)/9.81`, less HGT.
PSFC/T2/Q2 provide the surface parcel; Q2 and QVAPOR are **mixing ratios**, converted
to vapour pressure before dewpoint. No RH/moisture floor or saturation clipping
is applied. Sounding diagnostics require strictly decreasing pressure, increasing
height and positive mixing ratio (zero mixing ratio has undefined dewpoint).

All consumed numerical inputs and computed diagnostics must be finite. Input
NaN/Inf/fill values fail the command. Physically undefined levels, the first
precipitation interval and unavailable rolling windows are written as CF missing
values, with explicit `*_valid`/`*_status` flags. The in-memory representation uses
finite `-9999` fill (phase uses `-1`); xarray decodes these to NaN when reading the
output. These documented missing values are distinct from invalid calculations.
Flags have units `1`. Output Time is UTC, latitude/longitude are auxiliary CF
coordinates where available, and every diagnostic supplies units, a long name
and sampling metadata. Source file paths and resolved options are recorded.
Output is atomically replaced only after a successful complete write; an input
file cannot be the output.

**Sampling limit:** a history sample is instantaneous unless explicitly labelled
as a window. Hourly files cannot provide sub-hourly maxima, gusts or precipitation
timing. CAPE/SRH and cloud values are samples, not maxima over the previous hour.
`time_bounds` describes each previous history interval (the first is invalid).
Rolling variables give their trailing duration separately in `window_seconds`
and their validity flags; the previous-interval bounds do not describe a rolling
window. No interpolation in time is used to manufacture a missing window edge.

## Definitions

| Product name | Output and units | Definition and assumptions |
|---|---|---|
| `wind-speed10` | `wind_speed_10m`, m/s | `sqrt(U10²+V10²)`; rotation invariant. |
| `wind10` | speed m/s; `wind_from_direction_10m`, degrees | Earth-relative meteorological direction **from**, clockwise from north. Calm speed exactly zero gives direction 0 by convention. |
| `wind-height` | east/north components and speed m/s, direction degrees; `bulk_wind_shear` m/s; `wind_veer` degrees | Destagger U/V by adjacent arithmetic means, rotate to earth, linearly interpolate **components** in AGL height. U10/V10 anchor at 10 m; lowest model mass height must exceed 10 m. One or more increasing requested heights; no extrapolation in any column. With two or more heights, shear is the vector difference magnitude between the lowest/highest requested heights; veer is signed clockwise upper-minus-lower direction in [-180,180), missing if either endpoint is calm. A single height emits winds alone. Height is a CF coordinate in m AGL. No logarithmic/power-law wind profile is assumed. |
| `cloud` | `cloud_cover_low/mid/high/total`, fraction 0–1 | Uses model **CLDFRA**, not an RH proxy. Low: p≥680 hPa; mid: 440≤p<680 hPa; high: p<440 hPa. Within each layer and for the full column, contiguous levels with CLDFRA>0 overlap maximally (block maximum); blocks separated by a clear level overlap randomly: `1-product(1-block_max)`. Empty pressure layers give 0 and a `cloud_layer_*_sampled=0` flag. No extrapolation below terrain or above model top. Total is computed from the full profile, not by randomly combining the three layer totals. This declared maximum-random convention follows the RRTMG overlap concept; it does not claim to reproduce a particular radiation call or UPP/wrf-python cloud diagnostic. |
| `precip` | `precipitation_accumulated`, mm; `precipitation_interval`, mm; `precipitation_rate`, mm/h; `precipitation_rolling_*h`, mm | Total liquid equivalent = RAINNC+RAINC+RAINSH if RAINSH exists. Positive BUCKET_MM or `--bucket-mm` reconstructs RAINNC/RAINC using `residual + bucket_mm*I_RAIN*`. Required counters must exist and be nonnegative integers; nonzero counters with unknown/disabled bucket size are refused. RAINSH has no WRF rain bucket. Each interval is the cumulative difference over its actual timestamps; rate = amount / elapsed hours. Rolling totals subtract the exact endpoint accumulations, including irregularly spaced intermediate files; a missing exact start timestamp yields missing + valid=0. Negative accumulations or decreases are refused; the tool cannot infer a restart reset. Accumulated origin is inherited from the model, not the first supplied file. |
| `precip-type` | `grid_scale_precipitation_phase`, category | **Limited composition diagnostic**, qualified for Thompson MP_PHYSICS=8: 0 none, 1 liquid, 2 solid, 3 mixed over the previous history interval. Solid water equivalent = ΔSNOWNC+ΔGRAUPELNC+ΔHAILNC (hail may be absent for mp=8), total = ΔRAINNC. Uses separate per-field reset checks. Solid fraction ≥1−1e−5 counts as solid; zero counts liquid; others mixed. Rejects solid amount above total by >1e−5 mm. Excludes deep/shallow convective precipitation. It cannot distinguish snow from graupel as a dominant weather type, or rain from freezing rain. Other schemes and thermodynamic precipitation type are deferred. |
| `snow` | `snow_depth`, m; `snow_water_equivalent`, kg/m²; `snowfall_interval_water_equivalent`, mm | SNOWH and SNOW are instantaneous surface storage. Snowfall is ΔSNOWNC over the preceding history interval, water equivalent, **not snow depth** and not storage change. SNOWNC represents grid-scale snow/ice; the Thompson reference excludes graupel/hail. Other schemes inherit their own SNOWNC definition. No fixed snow-to-liquid ratio is used. |
| `thermal` | wet bulb, heat index and apparent temperature, K | Wet bulb: MetPy `wet_bulb_temperature` (Normand). Heat index: NWS/WPC Steadman fallback, Rothfusz regression and humidity corrections; shade/light-wind assumptions, not a solar heat-load model. Apparent temperature: MetPy `apparent_temperature`, wind-chill/heat-index piecewise, with T2/Q2 and 10 m speed, `face_level_winds=False`, `mask_undefined=False`. Surface RH must be in [0,1]; supersaturation is refused rather than clipped. These are instantaneous outdoor diagnostics. |
| `instability` | `surface_based_cape/cin`, J/kg; `lcl_pressure/lfc_pressure`, Pa; `lcl/lfc_height_agl`, m; `lifted_index`, K | MetPy surface-based parcel at PSFC/T2/Q2 plus every model mass level. CAPE/CIN use MetPy's default bottom-LFC/top-EL integration; CIN is signed (≤0). This differs from wrf-python's most-unstable 500 m mixed parcel `cape_2d`. LCL via MetPy `lcl`; LFC via `lfc(which='bottom')`; missing if no LFC exists. Heights interpolate z in log pressure and are missing outside model vertical coverage. LI is environment-minus-parcel temperature at 500 hPa; missing when 500 hPa is outside the sounding. A 2 m surface sample is treated as z=0/PSFC for the parcel. No subgrid lifting or entrainment is modelled. |
| `srh` | `srh_0_1km`, `srh_0_3km`, m²/s² | Signed **total** SRH via MetPy's hodograph line integral in earth coordinates. Requires explicit `--storm-u/--storm-v` (m/s); 0,0 deliberately selects a stationary storm. No implicit Bunkers storm motion. U10/V10 are held constant from 10 m to ground; mass-level winds are destaggered and rotated. Layer tops are interpolated by MetPy; profiles must cover 3 km AGL. |
| `freezing` | lowest freezing level and wet-bulb-zero heights, m AGL, with status flags | Add the surface thermodynamic sample at z=0 to the model profile; find the lowest downward 273.15 K crossing and linearly interpolate temperature in height. If the surface is at/below freezing, height=0/status=2; a resolved crossing has status=1; no sampled crossing has missing/status=0. Multiple elevated inversions are not collapsed into a melting-energy precipitation-type model. Wet bulb at every level uses MetPy Normand. |

Wind direction, shear/veer and SRH require COSALPHA/SINALPHA. Rotation is
`ue=u*cosalpha-v*sinalpha`, `vn=v*cosalpha+u*sinalpha`; the pair must form a unit
rotation. When inputs are **known** to already be earth-relative and omit both
fields, explicitly pass `--option 'wind_frame="earth"'`. The program never
guesses from the projection or from the file's origin. Historical surface-only
GPU files can therefore safely yield `wind-speed10` while directional/profile
products are refused.

UTCI is **deferred**: mean radiant temperature, a declared radiation geometry
and independent implementation qualification are not available here. Solar
geometry/clearness and energy/agricultural products belong to application
extensions and are not implied by the core names above.

## Python API and application registration

```python
from gpuwrf.post import read_wrfout, run_products, write_products
ds = read_wrfout(["wrfout_d01_2023-01-15_06:00:00", "wrfout_d01_2023-01-15_07:00:00"])
products = run_products(ds, ["wind10", "precip"], {"rolling_hours": [1]})
write_products(products, "products.nc")
```

`read_wrfout(paths, variables=None)` eagerly loads requested variables, closes
each file, and supplies datetime64 Time. `run_products(ds, names, options=None)`
also accepts an already prepared xarray dataset with the same dimensions and
Time coordinate. `gpuwrf.post.registry.register(name, runner, required=(),
optional=())` adds a product; the runner receives `(ds, options)` and returns an
xarray Dataset. Runners must provide finite output and explicit sampling/units.
`post.apps.register_products()` is loaded lazily when that optional package is
installed and must be idempotent. Application options pass through repeatable
`--option key=JSON` values. Only requested fields and optional fields declared
by their registry entries are read by the CLI.

## Evidence and cost

Tracked fixtures in `tests/fixtures/post/` are **unmodified field crops** from
the real Swiss RD11 CPU-WRF 42×42×44 case (three irregularly spaced times) and
historical GPU surface history (two times). `manifest.json` records original
and extracted-file SHA256 values and exact crops. The extractor in
`proofs/u1_post/generate_fixtures.py` reads references without modifying them.
Tests use independent equations, published worked cases and upstream MetPy,
including active clouds/precipitation, rain buckets, rotation/destaggering,
irregular windows, missing levels, reset refusal and NaN/±Inf in every consumed
field. The GPU fixture qualifies surface wind speed and missing-input refusals;
**no full-profile GPU history qualification or GPU forecast is claimed**.
The CPU crop was selected from a whole-grid census for nonzero precipitation
and nonsupersaturated surface conditions. Separate unmodified supersaturated
surface crops exercise the documented thermal-index refusal; those cells were
not clipped or silently excluded from a claimed whole-domain product.

Post-processing executes on the CPU and introduces no model-step launches,
allocations, host/device transfers or model compilation. The normal forecast
parser adds only the post subcommand's argument registration and defers the post
package import until this command is selected; its small parser startup overhead
has not been benchmarked separately.
The inherited top-level gpuwrf package configures JAX on import; post code does
not import JAX or initialize its backend. This wave's commands/tests pin
`JAX_PLATFORMS=cpu`. CLI memory scales with the selected input fields across all
supplied times plus product arrays; split long histories into overlapping
windows to bound memory. NumPy products vectorize over columns; MetPy
parcel/wet-bulb/SRH calls run on CPU and can be expensive on large grids. No GPU
performance claim follows from these offline checks.

Measured CPU cost on the tracked three-time 2×2×44 crop (MetPy 1.7.1,
core 29, nice 19; includes selected-field reads and NetCDF writing): wind/cloud/
precipitation/snow products took 1.589 s wall / 1.480 s CPU, and thermal/parcel/
SRH/freezing products took 0.872 s wall / 0.663 s CPU. Peak process RSS was
481,924 KiB (~471 MiB), dominated by interpreter/library overhead at this tiny
size; this is not a full-domain memory estimate. JAX's backend remained
uninitialized. The reproducible driver is `proofs/u1_post/benchmark.py`;
the lane's `bench.json` records source identity, array bytes, options and costs.

## References

- [WRF output definitions and precipitation buckets](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/output.html).
- Pristine WRF `phys/module_mp_thompson.F` lines 1298–1309: RAINNC total,
  SNOWNC snow+ice, GRAUPELNC separate (qualification restricted to mp=8).
- [RRTMG shortwave source and cloud overlap conventions](https://github.com/AER-RC/RRTMG_SW), maximum-random overlap.
- [wrf-python diagnostic definitions](https://wrf-python.readthedocs.io/en/latest/diagnostics.html);
  the RH-derived `cloudfrac` and `cape_2d` parcel definitions differ from ours.
- [MetPy calculations](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.html),
  [LFC selection](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lfc.html),
  [SRH](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.storm_relative_helicity.html).
- [NWS/WPC heat index equation](https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml).
