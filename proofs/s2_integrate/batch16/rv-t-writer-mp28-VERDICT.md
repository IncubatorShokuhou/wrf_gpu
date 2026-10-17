# FIX — t-writer-mp28-emission

Reviewed candidate `cc5a953eacd10e50ccb1c639c0615c8232924916` (base `2af43322bec3e2d722884048a2ad7d92d483500f`).
The emission physics fix passes the registered CPU gate. One directly affected
shared oracle caller must be repaired before merge (E199).

**Required fix:** `tests/v025/b_thompson/oracle_reference/p0l3b7c_runner.py:221–223`
calls the changed startup helper on its constructed column and demands the old
mass-height emission. The real frozen production invocation now raises
`Refuse: port nwfa2d 62146.65625 does not round to the driver's 53290.4140625`
before scoring its first column. On exactly the same candidate/source, restoring
only the pre-fix helper in process passes all six original L3b columns, unchanged
WRF files/NPZs/manifests/tolerances. This is a new shared-consumer regression,
not a failure of the corrected startup field.

Repair the column caller to supply the original driver's frozen `nwfa2d/nifa2d`
as held State inputs and assert their actual adapter consumption/preservation.
Its column oracle must compare identical emission inputs to WRF, rather than
recompute startup emission using a different height context. Preserve every
frozen WRF output/input and numerical bound; retain the new real-file frame-zero
test as the independent startup check. Re-run the repaired L3b production/fp64
caller and a deletion-sensitive input-threading check. Do not revert the correct
interface geometry or widen a tolerance. The critic has made no product repairs.

Independent evidence on the immutable candidate:

- Pristine `start_em.F:873` computes interface `(PH+PHB)/g` and
  `module_physics_init.F:4530` passes it as HGT to `thompson_init`; original
  `module_mp_thompson.F:496–510` uses the two lowest interfaces, REAL4 constants,
  left-associated emission factors and no area/dt scaling. NIFA2D is untouched
  by that initializer. The source formula matches this chain; removing the old
  depth clamp also matches the original. State.ph is total geopotential; summing
  the REAL4 inputs in loader fp64 then rounding reproduces the REAL4 sum.
- Actual standalone CPU native initializer vs unchanged original c19/mp28
  file: QNWFA2D eager max **3 ULP**, max abs **0.015625**, **1678/1764** exact;
  compiled max **19 ULP**, relative **1.407930636e-6** below frozen
  **32eps32 = 3.814697266e-6**. QNIFA2D **1764/1764 BYTE**; changed-geometry
  held emission BYTE. Independent eager/compiled tests reject nonfinite output.
- **9 focused tests PASS**, **0 setup errors**. Own full-function pre-fix
  mutant produces **1 genuine AssertionError / 0 setup errors**.
- Two emission arrays with nonzero distinct water/dust values survive checkpoint
  and NetCDF restart plus subsequent coldstart/changed geometry: **8 BYTE checks**.
  The first proof harness had an incorrect hardcoded surface shape, correctly
  refused by E78; only that proof shape was repaired and the failed log retained.
- Official fresh full default jaxpr traces: Swiss **f672a573**, PROD d01
  **6fc715a2**, exact. Only the temporary-case location was redirected privately;
  zero forecast steps and no GPU execution.
- Only `_aerosol_surface_emission_columns` changes in one product file; every
  other top-level AST is identical to base, including the shared height helper
  and every other caller. Four original files match claimed SHA256 and every
  saved array byte. Ten packet hashes and fourteen clean archive files verify.
  Actual low/middle/high terrain branches cover **390/1296/78** cells.

Performance/memory: measured CPU native load+initialization+qualification
8.313s, RSS2051520KiB (~1.956GiB), no swap, 6G cap. This is diagnostic cost,
not a matched speed comparison. [I] Startup uses two float32 interface planes
instead of whole-column float64 mass-height intermediates. Existing initialized
mp28 hot-step work, carry/restart size and default graph are unchanged. Actual
GPU launch/compile/readiness/VRAM/performance effects are [U], with no new GPU claim.

Existing disclosed limits remain: separate 3D fallback mass geometry is unfixed;
affected full GPU forecast/E41/performance qualification is pending; QNBCA and
TAOD5502D remain uncomputed. Old defective GPU QNWFA2D is not relabelled PASS.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-writer-mp28`; source/provenance `AUDIT.json`, native `native/NATIVE_FRAME0.json`,
`healthy.xml`, `mutant.xml`, `RESTART_BYTES.json`, both identity JSONs,
`l3b.log`, `l3b_control/summary.json`. MAIN experiments RV01–RV10 logged.
CPU16–19 nice19, one heavy JAX process at a time, official RAM slots6G/noSwap,
QUIET guards, bounded timeouts, cacheNONE/GPUWRF_JAX_CACHE=0; no GPU.
