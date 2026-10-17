# LCZ, urban morphology and anthropogenic heat

**Status in v0.4.0:** The single-layer urban model (`sf_urban_physics=1`) is **validated 24 h**: it passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. LCZ land use (`use_wudapt_lcz=1`), NUDAPT morphology and anthropogenic heat are **experimental**: checked against WRF's urban code, not yet in a full 24-hour forecast comparison. LCZ with BEP/BEM (`sf_urban_physics=2/3`) is not supported.

These options extend the opt-in single-layer canopy model with Noah-MP:

```fortran
&physics
 sf_surface_physics = 4,
 sf_urban_physics = 1,
 use_wudapt_lcz = 1,
 distributed_ahe_opt = 2,
/
```

LCZ with BEP/BEM (`sf_urban_physics=2/3`) is not supported: the multi-layer
model currently has only one urban type (type 2). Mixed LCZ types would need
per-type geometry, prognostic state and PBL sources; that is separate work.

WRF 4.7.1 uses **LCZ land categories 51–61**, mapped to urban types 1–11
and `URBPARM_LCZ.TBL`. An ordinary ISURBAN cell becomes type 5 with LCZ
enabled. Older 31–41 LCZ data must be regenerated with the matching WPS/real.exe;
custom and NLCD category numbers are not automatically converted. Provide a
consistent `NUM_LAND_CAT` and the WRF tables. The daily CPU-history replay
pipeline refuses these options; use the native single-domain/nested loader.
WRF also refuses LCZ tables when the domain's maximum urban type is at most 3
(module_physics_init.F:3353). The loader preserves that upstream consistency
check; supply a properly prepared LCZ case rather than relabelling old classes.

`FRC_URB2D` is the urban area fraction (0–1). WRF keeps a supplied value in
(0,1], otherwise uses the selected category's table fraction. It blends rural
Noah-MP and SLUCM fluxes with that fraction. Morphology can be supplied as
prepared wrfinput `LP_URB2D`, `MH_URB2D`, `STDH_URB2D`, `HGT_URB2D`,
`LB_URB2D`, `LF_URB2D`, or as geogrid-style `URB_PARAM`: one-based slots
91,92,93,94,95 and 96–99 respectively. Heights are metres, plan and frontal
area indices are dimensionless. WRF enables morphology only where area-weighted
height HGT and mean height MH are positive; otherwise category geometry applies.
NUDAPT LF(1) is an experimental deterministic convention that matches WRF
**only for a single zero-initialized cell**, as described in
[upstream behavior](UPSTREAM_WRF_BUGS.md). WRF reuses undefined U10/V10 locals
across tile cells; compiler initialization does not reset them per cell. In a
two-cell diagnostic run of the original WRF code, identical cells differ by
HFX 6.273636 W/m2, U10 0.992088 m/s and TSK 0.301880 K. First-cell results are
bit-identical to the one-cell reference; an isotropic LF control has zero
differences. A full forecast comparison with the original CPU WRF has not been
done for this option. LCZ without morphology is unaffected. Invalid geometry or a first model
level too close to the canopy fails before stepping. Distributed drag, green
roofs, irrigation and CH_SCHEME=1 remain refused.
WRF's actual NetCDF names for LP, HGT and LB are `BUILD_AREA_FRACTION`,
`BUILD_HEIGHT` and `BUILD_SURF_RATIO`; the loader accepts these Registry names
and their Fortran grid-member aliases. STDH is passed verbatim: WRF labels
the field `m2` although its urban formulas use it as a height standard deviation.

Table `AHOPTION=1` applies the table's 24-bin local-solar diurnal profile to
AH (W m-2). `ALHOPTION=1` applies its 48-bin profile and seasonal multiplier
to ALH. Both follow the radiation driver's held solar hour angle, including
its half-radiation-interval midpoint. WRF holds the simulation-start calendar
month for the seasonal multiplier (the caller passes grid%julday).

WRF's separate namelist control is `distributed_ahe_opt`, not `ah_option`.
The `AHE` input is a finite REAL field with 288 month/hour bins (January hours
0–23, February 24–47, etc.) and units W m-2. WRF selects the simulation-start
month even across a month boundary; hours cycle in UTC. Option 2 adds the selected value
to HFX after the surface flux blends, without urban-area scaling. Option 1
adds AHE/1200/DZ8W to the bottom PBL potential-temperature tendency; this port
requires its WRF source-tendency MYNN path. Table AH and distributed AHE are
additive, as in WRF; configure them deliberately to avoid double-counting a
physical emissions inventory.

Beyond-WRF extension: `urban_ahe_file='/absolute/path/hourly.nc'` selects a
gridded hourly AHE stream with WRF `Times(Time,DateStrLen)` and
`AHE(Time,south_north,west_east)` in W m-2. Times must start exactly at run
initialization, advance hourly, and cover the entire forecast. Values are
held for each UTC forecast hour. The whole stream is validated and loaded
once; steps perform only device array selection. Restart carries include
these exact inputs, and missing feature inputs are refused.
The four REAL clock anchors ride the device carry too; forecast dates do not
enter the static urban bundle or force a new coupled-model compilation.

With these options off, the release program is unchanged. The evidence is a
one-cell column comparison with the original WRF code covering all eleven LCZ
types, active morphology, diurnal heat, and single/double precision tolerances.
These options have not been run in a GPU forecast comparison, and their GPU speed
and memory use have not been measured.


Measured CPU costs on a 42×42×44 Swiss city patch (12 urban cells, eight
logical CPUs) were 0.478–0.580 s per warm step; restart steps took
0.487–0.523 s. These CPU runs emulate the GPU kernels and do not estimate GPU
speed. Geometry's 99 sky-view iterations
run once on the host during loading. Nine REAL32 morphology grids, optional
longitude and four REAL32 clock anchors remain in the device carry. Measured
inputs occupy 2,102,704 bytes with monthly AHE, or 77,632 bytes with the two-record
hourly stream; longer hourly files scale linearly with record count. No per-step
file reads or host transfers were added. GPU launch count, peak GPU memory and compile time
for these options have not been measured. The 24-hour validation of the
table-only single-layer model does not cover these inputs.
