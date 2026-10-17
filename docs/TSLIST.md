# WRF station time series

**Status in v0.4.0: Experimental.** It runs, but has not yet passed a 24-hour comparison with the original CPU WRF.

Place a WRF `tslist` in the input run directory passed to `gpuwrf run`. This
opts a fresh native single-case forecast into station output. Without that
file, the compiled forecast program has no station sampler or station buffer.

The first three lines are headers. Use WRF's fixed columns exactly:

```text
#-----------------------------------------------#
# 24 characters for name | pfx |  LAT  |   LON  |
#-----------------------------------------------#
Swiss centre              ctr    46.608    8.489
```

Description occupies columns 1–25, prefix 27–31, latitude 33–39, longitude
41–48. An `I`/`J` header in those same columns selects one-based `I7`/`I8`
indices. Prefixes are 1–5 letters, digits, `_` or `-`; duplicate or unsafe
prefixes are refused. LAT/LON supports the WRF spherical Lambert, polar
stereographic and Mercator projections (`MAP_PROJ=1/2/3`). Coordinates are
projected and rounded with WRF `NINT`; no land/water relocation is performed.
Each domain independently selects the stations it contains, retaining their
original tslist IDs. I/J addresses the same indices independently on each nest.
Outside-domain stations produce no files. Stations on WRF's extra high-edge
halo are refused because that halo is unavailable in the port's public State.

WRF controls in `&domains` apply globally:

| Key | Default | Meaning |
|---|---|---|
| `max_ts_locs` | 5 | Read at most this many stations; ignore later records with a warning; 0 disables output |
| `max_ts_level` | 15 | Global profile level count, first capped by root levels (WRF `e_vert(1)-1`), then available domain levels |
| `ts_buf_size` | 200 | Device sample capacity; a child uses at least enough rows for one complete root step |
| `tslist_unstagger_winds` | `.false.` | Optional WRF rotation-before-averaging for UU/VV; requires interior stations |

Every completed own step contributes one sample, starting at the first step's
valid time, before nest feedback. There is no artificial lead-zero sample.
Station-only gathers run inside the step program. Bounded buffers are copied
to the host between compiled forecast segments and flushed, including the
last partial segment. Short buffers can increase segment dispatches and host
synchronizations; a buffer covering a history interval avoids extra segment
boundaries. There is no per-step host transfer for the sampler.

Each station/domain writes `pfx.dNN.TS` and `.UU/.VV/.WW/.PH/.TH/.QV/.PR`.
The header, field widths and spacing match original CPU-WRF files, including
WRF's consecutive `F13.5` profile fields. Existing targets are refused.

| Stream | Values and units |
|---|---|
| TS identifiers | domain, forecast hours since simulation start, original station ID, one-based i/j |
| TS fields in order | T2 [K], Q2 [kg/kg], earth U10/V10 [m/s], PSFC [Pa], GLW/GSW/HFX/LH [W/m²], TSK/top TSLB [K], RAINC/RAINNC [mm], CLW [kg/m²] |
| UU/VV | Earth winds [m/s], WRF's staggered sampling by default |
| WW | Vertical velocity averaged to mass levels [m/s], for either wind option |
| PH | Geopotential height at mass levels MSL [m], using WRF g=9.81 |
| TH/QV/PR | Dry potential temperature [K], vapor mixing ratio [kg/kg], total pressure [Pa] |

GSW is net downward surface shortwave. CLW includes cloud water, rain, cloud
ice, snow and graupel over **all** mass levels, using WRF `calc_p8w` pressure
faces and its literal species/level REAL accumulation order. Profile height
selection does not truncate CLW. Surface fields use held in-step diagnostics;
sampling does not rerun surface physics or radiation. NaN/Inf batches fail
before file append.

The initial implementation supports Noah-MP with held in-step surface and
radiation diagnostics, fresh native single-case (B=1) runs, and static nests. Checkpoint/
resume, homogeneous batching and CPU-history replay with a tslist are refused.
Buffered tslist carries are also refused by restart serialization. WRF
restart-style file replacement/append and moving-station relocation are not
implemented. Other land models or unavailable halo sampling fail explicitly.
Solar diagnostics (`diags.solar_diagnostics != 0`) require an extended WRF
format and are refused. A list with no station inside any domain leaves the
forecast unsegmented and creates no station output.

Original CPU-WRF Swiss evidence is tracked under `proofs/u2_tslist/oracle`:
four completed steps, two stations, eight files each; LAT/LON and I/J with
unstaggered profiles. All five condensate species are additionally exercised
by a verbatim pristine `calc_p8w`/CLW source oracle on real Swiss pressure and
height columns. A short GPU run also checks buffer flushing, compiled-program
resources and the extra cost of sampling; the records are in
`proofs/u2_tslist/receipts/qualification.json`. Long-forecast fidelity has not
been checked.

The offline `gpuwrf tslist-export` tool runs on the CPU. It selects surface
variables and output intervals without altering the WRF raw stream:

```bash
gpuwrf tslist-export --input ctr.d01.TS --output ctr.csv \
  --variables T2,U10,V10,RAINNC --interval-seconds 60
gpuwrf tslist-export --input ctr.d01.TS --output ctr.nc --format netcdf
gpuwrf tslist-export --input ctr.d01.TS --output ctr-verify.csv \
  --format station-csv --station-uid swiss-centre --variables T2,U10,V10
```

Each export contains one station/domain. It preserves original forecast
hours and values; an interval keeps the first completed sample crossing each
positive interval alarm (dt=18, interval=60 keeps t=72,126,…). Filtering uses
the WRF file's rounded hours. No interpolation or missing sample is invented.
CSV column names identify variables; the table above defines their units.
NetCDF includes units and time origin. Export initializes no JAX backend and refuses an
existing destination. `station-csv` uses the station verification tool's
canonical CSV schema and validator, and can be passed directly to `gpuwrf verify --model`.
It requires the station UID from your station metadata table. It converts T2
to °C and earth U10/V10 to speed plus meteorological direction (and retains
components). Calm samples omit undefined direction. This schema supports
temperature and winds; plain CSV/NetCDF retains the other WRF fields without
inventing precipitation intervals from the first accumulated sample.
