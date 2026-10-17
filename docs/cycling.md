# Soil and snow cycling

`gpuwrf cycle` carries a previous forecast's land state into a new initialization.
It can reduce the repeated soil moisture, soil temperature and snow spin-up caused
by replacing a regional model's evolved surface with each forcing analysis.
It does not assimilate observations or guarantee better forecast skill; compare
cycled and uncycled forecasts against independent observations for your region.

```bash
gpuwrf cycle \
  --source previous/wrfout_d01_2023-01-15_03:00:00 \
  --wrfinput new-real/wrfinput_d01 \
  --output next-run/wrfinput_d01
```

Prepare `next-run` with the new run's namelist and boundary files before running
the forecast. Repeat the command separately for each domain. The output must be
a new path; neither input is modified. The output retains the new atmosphere,
SST, time, forcing and static fields. An embedded `GPUWRF_LAND_CYCLE` JSON
attribute records both input paths and SHA256 hashes, selected time index, UTC
valid time, per-field transferred-value hashes, masks and cell counts. The CLI
prints that record plus the output hash. Output publication is atomic.

The source is one CPU-WRF/gpuwrf NetCDF `wrfout` or WRF NetCDF restart file. Its
`Times` must contain exactly one record matching the new wrfinput's single
initialization time. The command does no temporal interpolation, extrapolation,
grid remapping or clipping. It refuses different horizontal coordinates,
projection, terrain, grid ID, land-use tables/categories, soil categories, soil
layer centers/thicknesses, masks, field shapes, dtypes or units. Missing metadata
also causes refusal. Noah (`sf_surface_physics=2`) and Noah-MP (`4`) are supported.
Other surface schemes need their own complete cycling contract.
This offline file support does not extend the model's admitted land schemes:
the native CLI currently admits Noah-MP, while Noah classic remains an API path.

The basic transferred variables are `SMOIS`, `SH2O` (m³/m³), `TSLB`, `TSK` (K),
`SNOW` (kg/m², numerically mm water), `SNOWH` (m), and `SNOWC` (fraction).
`CANWAT` (kg/m²) is included when present. Only cells with `LANDMASK=1`,
`XLAND=1`, and `LAKEMASK=0` are changed. Water/lake cells retain the new input.

Noah-MP requires the full available snow-column and canopy set: `ISNOW`, `TSNO`,
`SNICE`, `SNLIQ`, `ZSNSO`, `SNEQVO`, `ALBOLD`, `TV`, `TG`, `TAH`, `EAH`, `CANLIQ`,
`CANICE`, `FWET`, `LAI`, `XSAI`. Missing required source fields cause refusal;
use a fuller history stream or restart. If present on either side, optional
`TAUSS`, `SMCWTD`, `CM`, `CH`, `ALBEDO`, `EMISS`, `ZNT`, `QSFC`, `WA`, `WT`, `ZWT`,
`LFMASS`, `RTMASS`, `STMASS`, `WOOD`, `STBLCP`, `FASTCP` also require a source and
are copied. Source fields absent from the input are created with matching WRF
metadata and dimensions. Accumulated runoff/fluxes are not cycled.

Canopy temperature/humidity/wetted fraction, phenology and carbon fields are
copied only where source `FVEG>0` and the category is not `ISICE`; `FVEG` is
retained as the explicit activity mask. Aquifer `WA`/`WT` exclude glaciers.
Pristine WRF writes missing-value sentinels at these inactive cells; the command
preserves the new input there. Zero is accepted as valid active state. All
numeric fields reject NaN/Inf; missing active values and impossible bounds cause
refusal. Conservative sanity bounds include moisture/fractions in [0,1], liquid
water ≤ total moisture, temperature 150–350 K (inactive snow layers may be zero),
nonnegative water storage and snow depth, and `ISNOW` in {-3,-2,-1,0}.
Snow/soil column consistency is checked both before publication and before
restoration: inactive snow slots must be zero, active snow/soil thicknesses
positive, soil thicknesses consistent with DZS, and layered mass/depth consistent
with SNOW/SNOWH. Thin bulk-only snow with ISNOW=0 is valid. Rounding allowances
come from [pristine WRF arithmetic](../proofs/u2_cycling/COLUMN_BOUNDS.md);
inconsistent columns are refused without clipping or reconstruction.

The gpuwrf Noah-MP input loader recognizes the provenance marker and restores
transferred fields after its cold-input defaults. This is necessary because
ordinary real.exe inputs have uninitialized snow/canopy fields and zero SH2O.
Unmarked inputs use the existing cold initialization. The marker and transferred
value hashes are checked at load time; editing cycled state requires a new cycle.
The marker is a gpuwrf extension: CPU-WRF's ordinary cold-start initializer does
not honor it. Use an actual WRF restart workflow when continuing CPU-WRF.

Evidence includes literal independent copy tests and original Swiss CPU-WRF
hourly state at hours 1–3, frozen field values and the production Noah-MP input
loader. New-init test templates use the Swiss grid with explicitly retimed
initialization metadata; they do not claim independent real.exe analyses at
those times. Cycling adds offline file copying and one input-only restore pass;
it introduces no timestep kernels, synchronization or persistent device state.
