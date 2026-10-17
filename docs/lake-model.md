# WRF lake model

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

`sf_lake_physics=1` opts into the WRF lake model with Noah-MP, Thompson (`mp_physics=8`),
MYNN (`bl_pbl_physics=5`) and MYNN surface layer (`sf_sfclay_physics=5`). The lake
code is also checked on the CPU against the original WRF routines. The tested
24-hour configuration used real lake cells with WRF's default lake depth. The default
`sf_lake_physics=0` keeps the released program unchanged.

The port implements `lakeini` and the full `Lake`/`LakeMain` timestep from
unmodified WRF `phys/module_sf_lake.F`: surface-flux/stability iterations,
coupled snow/lake/soil thermal diffusion, phase changes, convective mixing,
precipitation/evaporation and snow water/compaction/layer combine/divide.
Caller and saved payload use REAL32; WRF's explicit `r8` calculations stay DOUBLE,
including its rounded unsuffixed REAL constants.

Both CLI loaders bind `use_lakedepth`, `lakedepth_default` and `lake_min_elev`.
`LAKEFLAG=1` selects exactly `LAKEMASK=1`; otherwise WRF's water-category/elevation
selection applies. Lake points must be water (`XLAND=2`, as prepared by real.exe),
with soil category 1..19. Requested depth data requires `LAKE_DEPTH`/`lake_depth`
and valid `LAKE_DEPTH_FLAG`; use `use_lakedepth=0` for default depth. Missing
requested data fails before initialization, even on an all-land tile.

Lake columns are compacted once. Noah-MP runs on the complementary tile,
retaining two horizontal axes; lake columns never enter its kernel and their
unused Noah-MP payload stays unchanged. The Lake call follows the LSM slot and
overwrites HFX, LH, GRDFLX, TSK, QFX, T2, TH2, Q2 and ALBEDO. GRDFLX is
`eflx_gnet`, the flux at the surface interface; penetrating solar radiation is
handled inside the thermal solve. MYNN's bottom flux handles are reconstructed
from physical HFX/QFX and the independently built MYNN Exner input under its
existing flag. Surface-layer momentum, UST, roughness and QSFC remain
as WRF's caller leaves them. Radiation reads held lake TSK/ALBEDO.
Lake-free domains retain current Noah-MP GRDFLX/ALBEDO history without running
LakeMain or constructing lake atmosphere inputs.

Auxinput4 `sst_update`, DFI and other LSM/microphysics/PBL partners combined
with lake physics are refused. Each nested domain binds its own bundle; nested lake runs are not validated.
GPU memory use, launch counts and speed have not been measured.

The [fixtures](../proofs/s1_lake/fixtures/manifest.json) come from the full original
WRF `Lake` wrapper after `lakeini`, retaining REAL32 forcing rounding and the
surface-driver operand mapping. Error services and literal WRF `RCP` constants
are the only stubs. Source/caller/compiler/driver hashes are recorded; no shared
WRF object or instrumented full WRF executable is run.

All 55 initialization columns match every one of 22 fields at frozen `2e-7`
relative tolerance, zero absolute tolerance and exact metadata. All 33 active
timestep trajectories match 22 payload and nine surface outputs at frozen
`2e-6` relative tolerance and small absolute roundoff floors (`1e-10` payload,
`1e-6` surface), rejecting NaN/Inf. They cover open/ice water, shallow/deep lakes,
all five snow-layer counts and 120-step ice formation. Expected values are
pristine Fortran, never a second JAX implementation.

A real Swiss 42×42×44 CLI case with two explicitly synthetic lake mask/depth
and water-category tiles completed three full CPU RK/physics steps. Meteorology
and geometry were retained. Lake state stayed finite REAL32 and evolved; the
actual Noah-MP kernel received 1×1762 non-lake columns. Pickle and NetCDF retained
every lake leaf exactly. This short CPU test is separate from the 24-hour GPU
forecast comparison.
One further diagnostic step from that exact saved restart reported zero
pre-repair guard events and zero negative/nonfinite/above-cap moisture visits.

Three standalone DOUBLE helpers (Combo, MoninObukIni, Tridiagonal) match original
WRF at frozen `2e-14` relative tolerance. The separate DOUBLE QSat cold-limit
diagnostic has `1.09e-11` relative CPU-JIT FMA contraction error (eager meets
`2e-14`); barriers/fast-math switches did not remove it. The full timestep is
checked at WRF's saved REAL32 interface and unchanged trajectory bounds;
no DOUBLE-exact QSat claim is made.

`OperationalCarry.lake` stores 22 compact arrays, held surface outputs and the
pre-correction energy residual, plus two packed full-grid GRDFLX/ALBEDO planes.
All leaves are float32. Standard lake history/restart fields have 10 water/soil
property levels, 15 snow/soil levels and 16 interfaces; exact compact leaves are
also saved in the native extension. Active restarts reject missing/old schemas,
missing fields, wrong shapes/dtypes or nonfinite data. Inactive old files
accept the absent group. Output expands the carry without a lake re-solve.

Default configurations add no device work/storage. Lake payload costs 808 bytes
per column before diagnostics; two Swiss lake tiles use 15,808 bytes including
held fields/background. Noah-MP gather/scatter traffic is opt-in. GPU resource use
has not been measured; the CPU timings below are not GPU speed claims.
The two-column CPU Lake executable compiled in 111.02 s on one shared CPU core;
eight warm calls had a 0.722 ms median, 18,049 bytes of compiler-reported
temporary storage and 1,960 bytes of compiled output. These component values
exclude the Noah-MP partition and full forecast work.

Regenerate references with `proofs/s1_lake/generate.py` and `generate_helpers.py`,
using your own `--out` directory and a CPU-only environment. Private builds clean
up after use; small fixtures are tracked in the repository. `coupled_smoke.py` records the synthetic edits and original/
modified input checksums.
