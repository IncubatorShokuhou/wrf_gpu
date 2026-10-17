# LCZ, urban morphology and anthropogenic heat

These options extend the opt-in single-layer canopy model with Noah-MP:

```fortran
&physics
 sf_surface_physics = 4,
 sf_urban_physics = 1,
 use_wudapt_lcz = 1,
 distributed_ahe_opt = 2,
/
```

LCZ with BEP/BEM (`sf_urban_physics=2/3`) remains refused: its new multilayer
bundle currently has one type-2 geometry/carry. Mixed LCZ requires per-type
grouped geometry, prognostic carries and PBL sources; that is a separate port.

WRF 4.7.1 uses **LCZ land categories 51–61**, mapped to urban types 1–11
and `URBPARM_LCZ.TBL`. An ordinary ISURBAN cell becomes type 5 with LCZ
enabled. Older 31–41 LCZ data must be regenerated with the matching WPS/real.exe;
custom and NLCD category numbers are not automatically converted. Provide a
consistent `NUM_LAND_CAT` and the WRF tables. The daily CPU-history replay
pipeline refuses these options; use the native single-domain/nested loader.
WRF also refuses LCZ tables when the domain's maximum urban type is at most3
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
NUDAPT LF(1) is an experimental deterministic convention with **one-cell
zero-init parity only**, as described in [upstream behavior](UPSTREAM_WRF_BUGS.md).
WRF reuses undefined U10/V10 locals across tile cells; compiler initialization
does not reset them per cell. In the sealed two-cell pristine diagnostic,
identical cells differ by HFX 6.273636 W/m2, U10 0.992088 m/s and TSK 0.301880 K.
First-cell results are BYTE with the one-cell fixture; an isotropic LF control
has zero differences. Coupled original CPU-WRF/D6 qualification remains
pending. LCZ without morphology is unaffected. Invalid geometry or a first model
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

The opt-out release program is unchanged. Qualification evidence uses a
pristine WRF one-cell column oracle with all eleven LCZ types, active morphology,
diurnal heat, and REAL4/REAL8 tolerances. GPU forecast qualification and
GPU timing/memory measurements remain pending.


Measured CPU interpretation costs on the 42×42×44 Swiss city patch (12 urban
cells, eight allowed logical CPUs at nice19) were0.478–0.580 s per warm control step on the final composed tree;
committed restart steps took0.487–0.523 s. These include CPU Pallas interpretation and
RN32 emulation and do not estimate GPU speed. Geometry's 99 sky-view iterations
run once on the host during loading. Nine REAL32 morphology grids, optional
longitude and four REAL32 clock anchors remain in the device carry. Measured
inputs occupy2,102,704 bytes with monthly AHE, or77,632 bytes with the two-record
hourly stream; longer hourly files scale linearly with record count. No per-step
file reads or host transfers were added. GPU launch count, peak VRAM, cold
compilation and cached executable readiness for these opt-ins remain unmeasured.
Existing table-only SLUCM GPU qualification does not cover the new inputs.
