# Noah-MP glacier timestep qualification

Both `opt_gla=1` and `2` now use the separate pristine-WRF glacier model at
`IVGTYP == ISICE`. They include glacier energy exchange, thermal diffusion,
ice phase change, snow-layer evolution and water balance. The operational
partition excludes those columns from natural-soil SFLX, composes with lake
partitioning and retains the previous-call `QSNOWXY` in all four restart forms.
Both options have `PhysicsStepSpec` entries and namelist bindings.

The scientific reference is the unmodified WRF 4.7.1
`phys/module_sf_noahmp_glacier.F`, SHA256
`a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece`.
The tracked NPZ assets, source/unit manifest and proof scripts are in this
directory. Full local artifacts are under
`<USER_HOME>/wrf_gpu2_lanes/s2-glacier/`; compact receipts are tracked in
[`receipts`](receipts).

## Scientific and integration gates

- **[M, original CPU-WRF]** 4,896 complete calls: 22 original Swiss columns
  plus 29 perturbed physical states, across 96 option tuples
  (`2 gla × 2 alb × 4 snf × 2 tbot × 3 stc`). Every defined output is literal
  binary32 byte exact, with zero nonfinite values and zero sensitivity of
  defined outputs to zero versus NaN compiler initialization.
- **[M, CPU compiled]** Both glacier variants pass the independent pristine
  input-ULP envelope on the 58-state compiled fixture. The envelope uses only
  original Fortran: individual input perturbations of ±1 ULP, all-low/high
  inputs and 128 seeded mixed directions. Its frozen gate permits one output
  ULP outside that envelope and rejects nonfinite values.
- **[M, real Swiss CPU coupled execution]** Each option completes three steps
  with zero nonfinite leaves and zero guard events. A resumed fourth step
  matches uninterrupted execution on all 183 carry leaves byte for byte.
  Actual driver output checks prove glacier routing, including the WRF
  vegetation sentinel and ice-soil moisture; these are not writer-only checks.
- **[M, tests]** G27: all 370 mandatory shared and affected consumers pass.
  After rebasing onto main `f202fa34955a`, G33: 302 affected tests pass,
  including `tests/contracts/`, the v0.18 fail-closed pins, all five option
  boundaries, namelist honour, catalog, NetCDF locking and restart consumers.
  The genuine deletion of the production dispatch branch is killed by one
  assertion with zero setup errors. A clean tracked archive passes all four
  selected source-binding, regime-census and compiled-envelope checks.
- **[M, numerical reuse]** Final candidate `721beb6ef7ab` retains identical
  full native-facade graphs, 183 initial leaves and 61 closed constants for
  both coupled-smoke configurations. Their graph hashes are `dd2e036f8b1d`
  and `adafaff6f507`. The completed CPU forecasts carry forward without reruns.

## Release-program impact

The 57-domain census finds **22 Swiss glacier points**, **zero in both PROD
domains**, and **zero in all 54 domains of the 18 WN3 cases**. Source and
pytree identity prove the ice-free path reduces exactly to main: 19 existing
arithmetic/driver modules remain byte identical, and the original 32/14 leaf
prefixes and `None` auxiliary data are preserved. This is a universal source
proof for WN3, not a claim of 54 measured executions.

The manager approved the Swiss change as a missing-physics fidelity fix. Its
new default native-facade jaxpr SHA256 is
`f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`,
replacing frozen `2bf12cb8`. PROD d01 remains exactly
`6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
These are CPU traces of the native facade, not GPU execution measurements.
The missing glacier timestep in shipped versions through v0.3.4 is recorded
in root `KNOWN_ISSUES.md`.

## Performance, memory and remaining owner validation

Ice-free domains add no timestep kernels or carry bytes. Glacier domains use
three Pallas column kernels plus natural/glacier gathers and scatters. Each
kernel keeps its local vertical recurrence inside the kernel. Swiss persistent
land state adds one float32 rate plane: **7,056 bytes**, increasing its land
carry from **373,968 to 381,024 bytes**. Intermediate stage buffers and tile
copies add traffic. The actual Pallas graph exposes two packed output buffers
per stage, totaling **8,096 / 8,272 / 8,976 bytes** for 22 columns, before
backend aliasing or dead-output elimination. These sizes exclude natural-tile
copies; actual GPU wall time and peak VRAM are still unknown.

CPU compilation of the six real sm120 kernel bodies totals **194.389 s** for
GLA1 and **170.151 s** for GLA2. All six cubins have STACK=0, LOCAL=0 and zero
PTX fp64 instructions; registers are 148/128/128 and 151/96/128. No non-leaf
callee exceeds 1,000 SASS instructions (maximum 54). These CPU compiler
artifacts establish bounded native compilation, not a measured GPU speedup
or a pass of the whole operational GPU executable's E41 gate.

GPU timing, peak VRAM, whole-executable E41 and Swiss 24-hour re-qualification
against CPU-WRF remain in the manager's final validation. No GPU was used in
this CPU-only wave. `UPSTREAM_WRF_BUGS.md` documents inactive output handling
and refusal of glacier `opt_snf=5`, whose precipitation branch is absent
upstream. Its caller-state note records WRF's defined per-point `QRAINXY`
hold: the port's zero matches the qualified cold-initialized path. Arbitrary
nonzero externally seeded `QRAINXY` is outside this qualification. The water
budget uses its correct local rain.
