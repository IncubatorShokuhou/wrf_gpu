# Export and custom geographical inputs

`gpuwrf export` reads existing CPU-WRF or gpuwrf history offline. It uses NumPy/xarray;
no forecast or GPU is started. Install optional projected-grid/GRIB support with
`pip install 'gpuwrf[export]'`. NetCDF uses the existing netCDF4 dependency, and
Zarr uses the existing Zarr dependency. GRIB2 additionally uses ecCodes and SciPy.

```bash
gpuwrf export --input wrfout_d01_* --output surface.nc \
  --variables T2 PSFC Q2 U10 V10 HGT
gpuwrf export --input wrfout_d01_* --output pressure.zarr --format zarr \
  --variables T U V QVAPOR PH --pressure-levels 850 700 500 --chunk 128
gpuwrf export --input wrfout_d01_* --output tenerife.nc \
  --variables T2 U10 V10 --region -17 -16 28 29
gpuwrf export --input wrfout_d01_* --output selected.nc --raw \
  --variables RAINNC RAINC QCLOUD U --region -17 -16 28 29
```

Pass files from one stationary domain. Actual WRF `Times` determine UTC sampling;
filenames do not determine time. Duplicate times, moving grids, inconsistent
field inventories and NaN/Inf in selected inputs are refused. A region is the
smallest rectangular **index envelope** containing grid centres inside the bbox;
on a curved grid, that rectangle can include centres outside the bbox. Dateline
crossing boxes are unsupported. Staggered raw crops retain the adjacent faces.

CF conversion writes `Conventions=CF-1.10`. The supported variables are:

| WRF selection | Output | Definition/units |
|---|---|---|
| T2 | temperature_2m | Instantaneous 2 m air temperature, K |
| PSFC | surface_pressure | Instantaneous surface pressure, Pa |
| Q2 | mixing_ratio_2m | 2 m water-vapour/dry-air mass ratio, kg/kg (CF units `1`) |
| U10, V10 | eastward_wind_10m, northward_wind_10m | Instantaneous earth-relative 10 m wind, m/s |
| HGT | terrain_height | Surface altitude above model sea level, m |
| T | temperature | `(T+300)*(P_total/100000)^(287/1004.5)`, K; moist-theta inputs explicitly marked USE_THETA_M are refused |
| U, V | eastward_wind, northward_wind | Adjacent C-grid face average, rotated to earth-relative wind, m/s |
| QVAPOR | mixing_ratio | Water-vapour/dry-air mass ratio, kg/kg |
| PH | geopotential_height | Adjacent full-level mean of `(PH+PHB)/9.81`, m |

`T`, `U`, `V`, `QVAPOR`, and `PH` require explicit pressure levels in hPa.
Target pressure levels must be strictly increasing or strictly decreasing; unordered
selections are refused, and valid selections retain their requested slab order.
Total pressure is `P+PB`; interpolation is linear in log pressure in each column.
Nonmonotonic pressure, levels below ground or above the sampled profile are refused;
there is no extrapolation, nearest-level substitution or hidden terrain mask.
Wind rotation requires valid COSALPHA/SINALPHA, including for U10/V10.

CF files contain UTC time, 2-D latitude/longitude, explicit pressure in Pa (positive
downward), and 2 m/10 m height metadata. Projected grids include metre x/y coordinates
and a grid mapping derived from WRF's MAP_PROJ/TRUELAT1/TRUELAT2/STAND_LON/MOAD_CEN_LAT.
Lambert conformal, polar stereographic, Mercator, and unrotated geographical grids
are supported, using WRF's spherical 6,370,000 m Earth. Rotated geographical grids
are refused. x/y consistency is checked to 10 m to accommodate WRF float32 lat/lon.

`--raw` keeps arbitrary selected WRF variables, their units, staggering and attributes
in NetCDF or Zarr. It does not claim CF conversion or reinterpret accumulated rain.
XLAT/XLONG/HGT are retained for location. For precipitation rates/totals, including
WRF buckets and counter resets, use the defined products in `gpuwrf post`.

Zarr stores use format 2 for client compatibility, consolidated metadata, time/pressure
chunks of 1 and horizontal chunks bounded by `--chunk`. Use a new output directory.
Files are written beside their destination and published only after successful writes.
Memory scales with selected source variables across the requested frames, plus derived
arrays; cropping reduces derived-array size, and Zarr chunking controls storage rather
than source loading. Split large exports into time batches to limit host memory.

GRIB2 is an explicit subset of the table above, using WMO parameter/fixed-surface
codes and IEEE float64 packing. Pressure levels must resolve to whole Pa. Whole-minute UTC timestamps are verifying times,
with zero lead and generating-process type “post-processed forecast”; original forecast
reference/lead lineage is not preserved. The originating centre is unspecified (65535),
not inherited from the ecCodes example. No accumulated precipitation is inferred.
Native regular lat/lon grids write directly. For projected grids specify an explicit
regular lat/lon target, inside the source convex hull:

```bash
gpuwrf export --input wrfout_d01_* --output surface.grib2 --format grib2 \
  --variables T2 PSFC U10 V10 --grib-grid -16.7 -16.4 28.2 28.5 31 31
```

Regridding is linear interpolation in geographical degrees (not conservative area
remapping); extrapolation is refused. This subset does not encode native projected
GRIB grids, rotated grids, precipitation intervals or arbitrary WRF species.

## Verification rules

For the complete qualification selection install `pip install 'gpuwrf[export,post,dev]'`;
optional-library cases explicitly skip when their dependency is absent. Qualification
receipts require those cases to run with no skips. Tests reopen NetCDF/Zarr with xarray, check all exported data arrays against independent
face-average/rotation formulas and MetPy log-pressure interpolation, and compare projection
coordinates to independently specified PROJ definitions. The documented CF checks cover
Conventions, coordinate units/standard names, time decoding, pressure axis/positive,
scalar height references, absence of `_FillValue`/`missing_value` on NetCDF dimension
coordinates, strict pressure-coordinate ordering, grid_mapping links and Appendix F mapping attributes. These
are targeted rules, not a claim that an external cfchecker was run. ecCodes round trips
check message count, numeric values, locations, UTC timestamps, level types and parameters.
A real original CPU-WRF Swiss history check preserves T2/PSFC/HGT values exactly in both
NetCDF and Zarr. No forecast-physics validation is implied by format conversion.

References: [CF grid mappings](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.6/build/cf-conventions.html),
[WMO/ecCodes fixed surfaces](https://codes.ecmwf.int/grib/format/grib2/ctables/4/5/),
[reference-time meaning](https://codes.ecmwf.int/grib/format/grib2/ctables/1/2/), and
[generating process](https://codes.ecmwf.int/grib/format/grib2/ctables/4/3/).

## Data compatibility: terrain and land-use edits

Custom geography must pass through WPS/geogrid/metgrid and original WRF `real.exe`
to produce consistent wrfinput/wrfbdy. Editing geo_em alone does not update an existing
wrfinput. Terrain affects atmospheric vertical coordinates and boundary consistency;
editing HGT directly is a **dry-run interface test**, not a scientifically valid forecast
initialization recipe. Regenerate real.exe inputs after changing geography.

```bash
gpuwrf doctor path/to/custom-real-case
gpuwrf run --input-dir path/to/custom-real-case --output-dir runs/custom --dry-run
```

NUM_LAND_CAT counts 21/24/40/61 are accepted by the category interface when the namelist
and input metadata agree, with finite integer LU_INDEX/IVGTYP inside the declared range.
Supply matching LANDUSE.TBL/MPTABLE.TBL and soil tables, consistent land/water masks,
vegetation categories and land fractions. A count is not qualification of arbitrary table
contents or an urban physics scheme; doctor checks table/category conventions separately.

Doctor is the shared category authority. Confirmed LCZ categories 31–41 without urban
physics fail loudly before a run, including `--dry-run`; confirmation uses LCZ in MMINLU
or use_wudapt_lcz. Ambiguous numbering in a custom non-LCZ table remains a warning, since
indices alone do not identify a classification. Unsupported urban/LCZ namelist switches
still fail the existing namelist gate. A structural dry-run can use placeholder/missing
inputs; its `land_category_check.unchecked` lists files outside the NetCDF category check.
Use doctor for full input validation. The compatibility gate exercises edited original
Swiss real.exe inputs with each category count and a confirmed LCZ refusal; it does not
run WPS, real.exe, or a modified-terrain forecast.
