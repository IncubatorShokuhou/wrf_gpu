# Application products from WRF history

These opt-in `gpuwrf post` products read either CPU-WRF or gpuwrf history.
They run offline with NumPy/xarray; they do not change a forecast or compile
model kernels. Solar geometry alone needs the optional dependency:

```sh
pip install 'gpuwrf[apps]'
gpuwrf post --input wrfout_d01_* --output wind.nc --product app_wind \
  --option hub_height=100 --option rotor_diameter=100 \
  --option 'power_curve="turbine.csv"'
gpuwrf post --input wrfout_d01_* --output solar.nc --product app_solar \
  --option tilt=30 --option azimuth=180 --option dc_rating_w=5000
```

See `gpuwrf post --help` for the core history reader and output options.
`--option key=JSON` is repeatable: numbers and booleans are unquoted JSON,
strings need JSON quotes. Each command processes one stationary domain.
Inputs use conventional WRF units: K, Pa, m, m/s, kg/kg **dry-air** mixing
ratios, mm precipitation and W/m² radiation. Times are decoded UTC history
timestamps. Missing required fields, NaN/Inf, negative hydrometeor/radiation
inputs (except visibility's literal UPP concentration rule), non-increasing times and extrapolation outside resolved wind profiles
are refused. Standard WRF scalar winds are destaggered before interpolation.

| Product | Output | Sampling and scope |
|---|---|---|
| `app_wind` | `hub_wind_speed`, `rotor_equivalent_wind_speed` (m/s); optional `wind_power` (W), `wind_energy` (Wh), `hub_air_density` (kg/m³) | Instantaneous speed/power, cumulative trapezoidal sampled energy |
| `app_solar` | `ghi`, `dni`, `dhi`, `poa_irradiance` (W/m²), `pv_cell_temperature` (°C), `pv_dc_power` (W) | Instantaneous irradiance and PV module DC power |
| `app_icing` | `supercooled_liquid_water` (g/m³), `icing_potential` (0/1), on model mass levels | Instantaneous thermodynamic screen |
| `app_fire` | `hdw_instantaneous` (hPa m/s) | Instantaneous sampled lower-layer wind/VPD product |
| `app_fwi` | `fwi_ffmc`, `fwi_dmc`, `fwi_dc`, `fwi_isi`, `fwi_bui`, `fwi_fwi`, `fwi_dsr` (unitless); `fwi_rain_24h` (mm) | Daily local-standard noon recurrence; preceding noon-to-noon rain |
| `app_frost` | `air_frost`, `surface_frost` (0/1) | Instantaneous temperature thresholds |
| `app_visibility` | `visibility` (m), `lowest_level_rh` (%), `fog_potential` (0/1) | Lowest resolved mass level |

Hourly history cannot recover ten-minute wind distributions, peak gusts,
within-hour frost minima, cloud droplet spectra or actual power production.
All hazard products are diagnostic screens, without calibrated probabilities
or operational aviation/fire warning certification.

## Wind energy

`hub_height` defaults to 100 m AGL, `rotor_diameter` to 100 m, and `samples`
to 9 (integer ≥3). The entire rotor must be above ground and within the wind
profile. U/V are destaggered; a 10 m U10/V10 anchor extends the profile down to
10 m. Each component is linearly interpolated in geometric height AGL.
No log-law, stability or below-profile extrapolation is substituted.

For sample heights spanning the rotor, segment boundaries lie halfway between
adjacent samples. Exact circular segment areas `Ai` sum to `A=πR²`:

```
REWS = (sum((Ai/A) * vi³))^(1/3)
```

This is the [DTU/IEC rotor-equivalent wind definition](https://orbit.dtu.dk/files/93592493/Rotor_equivalent_wind_speed.pdf).
`veer=true` (default) projects each sample wind onto the hub wind direction
and retains signed cubic flux contributions; `veer=false` uses speed only.
In a strongly reversing profile the signed equivalent speed can be negative;
power-curve estimation then refuses that profile rather than clipping it.
At a calm hub the projection direction is undefined, so speed-only is used.
A common horizontal map rotation cancels from scalar speed and this relative
projection: no earth-relative direction is claimed. Vertical velocity, rotor
tilt, turbulence, wakes and control dynamics are omitted. Snapshot REWS is
**IEC-style**, not a claim of IEC-certified ten-minute power measurement.

Power requires a user CSV with exactly these columns (W, not kW):

```csv
speed_ms,power_w
0,0
3,0
10,2000000
24,2000000
25,0
```

At least two finite rows, strictly increasing nonnegative speed and
nonnegative power are required. Linear interpolation is used within the curve;
power is zero outside its endpoints. Include the actual cut-in/cutout points.
For a pitch-regulated approximation, density changes the speed used to look up
power: `v_ref = REWS * (rho/rho_ref)^(1/3)`, with `reference_density=1.225` kg/m³.
Moist-air density is computed from P+PB, WRF dry potential temperature and
QVAPOR and interpolated at the hub; thermodynamic extrapolation is refused.
Density correction is not a universal turbine-control or stall model.

`wind_energy` starts at zero at the first supplied timestamp, then integrates
sampled power by the trapezoidal rule over actual (possibly irregular)
intervals. This is a sampling approximation, not annual energy yield or a
sub-hourly time average. Curtailment, turbine availability and electrical
losses are not included.

## Solar and PV

`SWDOWN` supplies GHI. If **both** `SWDDNI` and `SWDDIF` exist, they supply
DNI and DHI directly. Supplying only one is refused. Otherwise, the documented
[Erbs decomposition](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.erbs.html)
estimates them from GHI, true solar zenith and day of year. It uses Spencer
extraterrestrial irradiance and pvlib's default minimum cosine 0.065 and
87-degree DNI cutoff. At greater zenith DNI=0 and DHI=GHI. WRF radiation fields
are snapshots, while Erbs is empirical: decomposed DNI/DHI can have substantial
errors, especially broken cloud, snow, complex horizons and short samples.
The output records whether native components or decomposition were used.

Latitude/longitude, terrain elevation and UTC times feed
[NREL SPA via pvlib](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html),
using `nrel_numpy`. Geometry uses pvlib's standard-atmosphere pressure estimate
from elevation and default refraction conditions. True zenith is used for Erbs;
apparent zenith is used for panel incidence. Grid movement is refused.

Fixed-plane `tilt` defaults to 30° from horizontal, `azimuth` to 180° clockwise
from north (90° east), `albedo` to 0.2. Isotropic transposition is:

```
POA = DNI * max(0, cos(incidence))
    + DHI * (1 + cos(tilt))/2 + GHI * albedo * (1 - cos(tilt))/2
```

Beam power is zero at supplied zenith ≥90° (SPA apparent zenith in
`app_solar`). No tracking,
terrain/horizon shading, anisotropic sky, spectral or incidence-angle modifier
is modelled. These assumptions match pvlib's isotropic transposition.

Cell temperature defaults to the simple NOCT approximation
`Tcell = T2_C + (noct_c-20)*POA/800`, with `noct_c=45` °C. Supply
`cell_temperature_c` to bypass it. The approximation has no wind-cooling model.
[PVWatts DC](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html)
is `Pdc = dc_rating_w*(POA/1000)*(1+gamma_pdc*(Tcell-25))`, bounded below at zero.
Defaults are `dc_rating_w=1000` W, `gamma_pdc=-0.004` /°C. POA is treated as
effective irradiance: no IAM/spectral/soiling loss is deducted. Output is **DC**;
no inverter, grid AC, clipping, storage or net-metering model is implied.

## Icing, frost and visibility

Icing potential uses WRF QCLOUD+QRAIN, dry-air density and temperature.
`supercooled_liquid_water = 1000*rho_d*(QCLOUD+QRAIN)` g/m³ when
`-40 < T_C < 0`, zero otherwise; `icing_potential=1` wherever this is positive.
The [NWS icing conditions](https://www.weather.gov/source/zhu/ZHU_Training_Page/icing_stuff/icing/icing.htm)
support screening for subfreezing liquid, including rare liquid near -40°C.
The screen cannot infer droplet size, collection efficiency, aircraft speed,
ice shape, accretion rate or severity; it is not UPP IFI. Warm-surface transient
icing, wet snow and ice-crystal engine icing are outside this definition.
Negative liquid mixing ratios are refused, including tiny negative traces
present in some CPU-WRF history. No silent tolerance or clipping is applied
to make this screen accept an entire such grid; valid columns remain usable.

`app_frost` flags `T2_C <= air_threshold_c` and `TSK_C <= surface_threshold_c`,
both default 0°C. The thresholds and equality convention are recorded. TSK is
model skin temperature, not a crop-leaf thermometer; no duration, crop-specific
injury or probability is inferred.

Visibility follows [NOAA UPP CALVIS method R](https://noaa-emc.github.io/UPP/CALVIS_8f_source.html).
For lowest-level mixing ratios qv/qc/qr/qi/qs, compute
`rho=p/[287.04*T*(1+0.608*qv)]` and mixture volume per dry-air mass
`V=(1+qv)/rho+(qc+qr)/1000+(qi+qs)/917`. Mass concentrations are `C=1000*q/V`
in g/m³. The extinction coefficients (km⁻¹) are summed:

```
beta = 144.7*Ccloud^.88 + 2.24*Crain^.75
     + 327.8*Cice + 10.36*Csnow^.7776 + 1e-10
visibility_m = 1000 * min(24.135, -ln(.02)/beta)
```

As in original CALVIS, raw hydrometeors enter `V`, then each concentration
is replaced by `max(0,C)` for extinction. This is the reference algorithm,
including its treatment of finite negative condensate traces. Output metadata
reports `negative_condensate_input_count`; invalid mixture volumes are refused.
Dry clear air reaches the 24135 m ceiling. Graupel, haze/aerosols and
humidity-only extinction are omitted; there is no claim of full RUC-GSD
visibility parity. `fog_potential` flags visibility<1000 m **and** QCLOUD>0
**and** RH≥`fog_rh_percent` (default95%). RH is over liquid from Bolton saturation
vapor pressure, bounded at saturation for diagnostics. A lowest model level
above a shallow fog bank can miss it; rain/snow visibility alone is not fog.

## Fire weather

`app_fire` implements an instantaneous sampled
[Hot-Dry-Windy screen](https://hdwindex.fs2c.usda.gov/abouthdw.html): maximum
wind speed times maximum surface-adjusted vapor-pressure deficit in the lowest
`hdw_top_m` (default500 m). Profile coverage through the chosen top is required;
only actual levels at or below the top participate. Temperature is dry
adiabatically adjusted to the **lowest model-level pressure** with mixing
ratio retained. VPD uses Bolton saturation over liquid, in hPa. The two maxima
need not be collocated. This follows the structure of
[Srock et al. (2018)](https://doi.org/10.3390/atmos9070279), with lowest model
level replacing the surface and no added 2 m point. It is not the operational
daily afternoon maximum, climatological percentile or a fire-spread forecast.

`app_fwi` is the **classic daily Canadian FWI** from Van Wagner & Pickett
(1985), with CFFDRS latitude/month day-length corrections. It uses CFFDRS's
continuous FFMC moisture coefficient `250*59.5/101` consistently in FFMC and
ISI, rather than historical rounded147.2. It is not the distinct FWI2025,
hourly FFMC, overwintering/startup calculation, fuel classification or fire
spread model. The [CFFDRS reference implementation](https://github.com/cffdrs/cffdrs_py)
provides independent full-precision tests, and its published1985 table supplies
an additional sequential check.

```sh
gpuwrf post --input wrfout_d01_* --output fwi.nc --product app_fwi \
  --option utc_offset_hours=0 --option ffmc_previous=85 \
  --option dmc_previous=6 --option dc_previous=15
```

The fixed `utc_offset_hours` is **local standard time**, without daylight
saving. Exact noon timestamps and an exact preceding-noon accumulation sample
are required. Early noons lacking 24h history are skipped. Subsequent noons
must be consecutive daily noons; gaps and a missing first eligible noon are
refused. The supplied previous codes apply at the noon **before the first
output noon** (the UTC date is recorded); there are no silent initial defaults.
85/6/15 above are illustrative startup values, not valid codes for every site.

T2, RH derived from T2/PSFC/Q2, and U10/V10 (converted to km/h) approximate the
station noon observations that FWI expects. Preceding24h rain uses
RAINNC+RAINC(+RAINSH), including I_RAINNC/I_RAINC counters if `BUCKET_MM` or explicit
`bucket_mm` is positive. Counter availability/integer values and continuous,
nondecreasing rain are checked; accumulation resets are refused. Rain time
bounds are written explicitly. Previous codes must reflect the location and
season; a forecast with arbitrary startup codes is not a fire-danger analysis.
RAINSH remains unbucketed, as in pristine WRF `module_diag_misc.F`; WRF has
no I_RAINSH field.

Daily FWI has a subset of history times. Write it separately from instantaneous
products with different times; never fill the intervening times with zeros or
carry forward a daily code under an instantaneous label.

## Validation and cost

`tests/test_post_apps_numeric.py` checks disk areas by independent quadrature,
worked density/power/energy/threshold cases, pvlib Erbs/isotropic/PVWatts, NREL
SPA's published example, NOAA UPP extinction equations and 108 full-precision
CFFDRS cases across wet/dry regimes, all months and latitude bands. The tracked
fixtures record reference commits and source SHA256. Dataset/CLI tests cover
WRF units and staggering, noon/rain windows, metadata, optional fields, missing
inputs and NaN/Inf refusals. Reference packages are test extras, not imported by
`gpuwrf run`. Deletion-sensitive mutants verify that formula gates detect
real scientific changes.

Products allocate NumPy arrays only when selected. Wind interpolation loops
over requested heights with vectorized columns; rotor sampling uses small
bounded sets. SPA evaluates one timestamp at a time. Daily FWI loops over
noons with vectorized grid cells. Loading history and writing outputs cost CPU
RAM/I/O; there are no model/GPU kernel launches or forecast compile changes.
Processing one domain and bounded time spans keeps memory under user control.
