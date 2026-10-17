# Case doctor and reproducibility

Inspect a native `real.exe` case before running a forecast:

```bash
gpuwrf doctor /path/to/case --namelist /path/to/namelist.input
gpuwrf doctor /path/to/case --budget wrfout_d01_2026-01-01_00:00:00 wrfout_d01_2026-01-01_01:00:00
gpuwrf manifest /path/to/case --output run-manifest.json
gpuwrf reproduce run-manifest.json
gpuwrf reproduce run-manifest.json --case-dir /relocated/case
```

These commands inspect files without launching a forecast, importing JAX,
initializing CUDA, or changing inputs. JSON goes to stdout. Doctor/reproduce
return 0 for PASS, 1 for failed checks, and 2 for invalid arguments/files. A
doctor PASS means its listed checks passed, with any warnings still applicable;
it is neither scientific validation nor a guarantee that a forecast will run.
Manifest creation records unsupported selections too and does not validate them.
`--output` creates a new file and refuses to overwrite an existing file.

Doctor checks each declared domain's mass/staggered dimensions against the
namelist (`e_we/e_sn/e_vert` include the staggered endpoint), variable dimension
names, initial time, land and soil categories, soil layers, LANDMASK (1 land,
0 water), XLAND (1 land, 2 water), and actual case-local LANDUSE.TBL section and
water labels when provided. Without a table it reports unknown compatibility.
ISLTYP uses NUM_SOIL_CAT/num_soil_cat, never NUM_LAND_CAT. A missing category
declaration uses WRF's 16 soil categories. Categories 31–41 without urban
physics are explicitly flagged: confirmed LCZ inputs are errors, and ambiguous
datasets (including NLCD) receive advice to identify their category scheme.
Numeric input fields are scanned record by record; missing values, NaN and Inf
are errors. Broad sanity ranges cover terrain, soil temperature/moisture,
water-point SST and vapor; unexpected units are warnings.

Root boundary times must start at the IC time, be strictly increasing and agree
with `interval_seconds`. Coverage includes the last tendency interval, bounded
by WRF's explicit next-boundary timestamps if present. Core boundary base and
tendency fields must be present with WRF's exact axis names and lengths:
PH (and optional W) uses bottom_top_stag, V uses south_north_stag on west/east
strips, and U uses west_east_stag on south/north strips. Boundary width must
match spec_bdy_width (WRF default 5) and fit the IC grid. Child boundaries come
from the parent.
With `sst_update=1`, every domain needs wrflowinp SST through the requested end
time; the doctor does not assume extrapolation past its final SST record.
Shared dimensions must match. IC/BC source attributes are compared as heuristics;
equal or absent attributes cannot prove matching meteorological sources.
Support classifications, operational refusals and pairing restrictions come
from the existing scheme catalog, operational validator and honour report.

Advice reports the typical WRF rule `dt_seconds ≈ 6 × min(dx,dy)_km`, adjusted
by the ratios along each actual parent chain (omitted parent_time_step_ratio
defaults to 1, as in WRF Registry.EM_COMMON:2323), the maximum adjacent terrain height difference
divided by grid spacing (dimensionless), and initial vertical CFL
`max(|W_lower|,|W_upper|) × dt / dz`. Height is `(PH+PHB)/9.80665` metres.
This is physical W, not WRF's contravariant terrain velocity or acoustic CFL;
initial conditions cannot predict later instability. Thresholds of terrain
slope 0.5 and physical vertical CFL 1 trigger advice, not admission limits.
These conventions follow the [WRF Users' Guide](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.2/users_guide_chap5.html)
and the [WRF output definitions](https://github.com/wrf-model/Users_Guide/blob/main/output_variables.rst).

VRAM is an **estimate for planning**, not a measured result for the inspected
case or a memory admission limit. It is calibrated on two measured cases. A
two-domain 9/3 km Canary Islands case has `(120×70+267×117)×44` cells, 4.00 GB
of live GPU memory and a 5.61 GB automatic memory-pool budget. A three-domain
9/3/1 km Tenerife case needs roughly 963 MiB outside the memory pool. Doctor
scales the two-domain case's live and planning bytes linearly with cells and
adds the three-domain allowance outside the pool. It reports GiB (2³⁰ bytes).
Different physics, grids, output and allocator reservations can change actual
use substantially: measured per-process peaks of 4.74 GiB (two-domain case) and
6,264 MiB (three-domain case) in other configurations show this dependence.

Offline budgets share conservation diagnostic units/constants and use pristine
WRF pressure-coordinate definitions for NumPy history-file reductions.
They use cell area `DX×DY/(MAPFAC_MX×MAPFAC_MY)` (isotropic MAPFAC_M fallback),
dry-column mass `(MU+MUB)×area/g`, and layer mass
`((C3F[k]-C3F[k+1])×(MU+MUB)+(C4F[k]-C4F[k+1]))×area/g`, with
`g=9.80665 m s⁻²`. Pristine `dyn_em/module_initialize_real.F:6742` defines face
dry pressure as `C3F×MU0+C4F+ptop`; ptop cancels between faces. C1F/C2F are
mass-coupling derivatives and must not be differenced as pressure coordinates.
Layers must have finite, positive pressure thickness; inverted/zero layers
are rejected. Differencing the 1-D coefficients avoids materializing full 3-D
interface pressures. Sigma ZNW fallback requires explicit HYBRID_OPT=0 and
absent C3F/C4F. Atmospheric water is the
sum of available QVAPOR/QCLOUD/QRAIN/QICE/QSNOW/QGRAUP/QHAIL mixing ratios
(kg/kg dry air) times layer dry mass (kg). Missing species are listed; they
are not silently claimed as measured zero. RAINC+RAINNC+available RAINSH in mm is kg/m² and
includes snow/graupel already, so separate snow accumulators are not added.
RAINSH is independent accumulated shallow-cumulus precipitation
(`phys/module_physics_addtendc.F:2291–2293`); its absence is reported. Nonzero
precipitation buckets require BUCKET_MM metadata. Negative state
values are retained. Records must have the same geometry, vertical coefficients
and field set. Outputs are instantaneous storage/accumulator totals (kg), plus
first-to-last changes at the supplied history times. With no lateral flux and
integrated evaporation observations, **closure is UNKNOWN**; these changes
are not conservation residuals or fidelity checks.

A manifest streams SHA256 over wrfinput, wrfbdy, wrflowinp, the selected
namelist and case-local parameter tables (including Noah-MP MPTABLE.TBL).
History files are excluded. It
records parsed and original namelist text, installed Python-source digest,
version, available Git commit, catalog support, Python/package versions,
declared GPUWRF flags and selected backend/allocator environment variables.
Unrelated environment variables and credential-named variables are excluded.
It does not contain model inputs; keep those separately. Hashing costs one
sequential read of every input, so manifest creation scales with case bytes.

Reproduce verifies the inventory, byte hashes, namelist, source bytes, support
and recorded numerical environment. Git commit and host system are disclosed;
identical source bytes across commits are accepted. It never executes a stored
command, restores environment variables, downloads inputs or starts a run.
`--case-dir` remaps only case-relative paths; external namelists keep their
recorded absolute path. Manifests are provenance records, not signed attestations
or GPU/driver determinism guarantees. Unknown schema versions are refused.

The tools add no timestep kernels, device launches, host/device transfers or
compile work. Numeric scans hold one variable record at a time; hashes use
1 MiB blocks. The forecast program is unchanged.
