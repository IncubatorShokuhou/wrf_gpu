# CAM-MG allocation and producer selection

The public/daily forecast entry materialized CAM-MG histories for every MP
selection. Radiation then used their presence to override cloud fraction and
effective radii. Original WRF instead selects these overrides with
`is_CAMMGMP_used` (`module_physics_init.F:1362–1364`,
`module_radiation_driver.F:1433–1444`).

The fix excludes only the 20 CAM-MG fields from the historical concrete scan tail
when MP is not 11, and selects both radiation overrides with MP11. All other
legacy scan fields remain. The CAM science, native kernel, metadata, operational
entry files and strict restart validator are unchanged.

`QUALIFICATION.json` binds the receipts and their SHA256 values:

- All 12 C123 product failures pass; 338 targeted checks pass across the retained
  74 checks, one corrected new shape test, and 263 shared consumers.
- Active CAM11 carry/radiation outputs/constants/program remain byte identical.
  Eight non-CAM selections match the pre-CAM radiation program and compiled
  outputs exactly. These are regression proofs; the WRF selector source and
  retained original-WRF fixtures provide the scientific reference.
- Three independent allocation/radius/cloud mutants each cause one genuine
  assertion failure, with one healthy control and zero setup errors.
- Both full release identities remain exact: Swiss `f672a573`, PROD `6fc715a2`.

The native single/nested CLI uses the selected-only initial carry and bypasses
the faulty public-entry helper. Existing composite runs therefore were not
exposed to this allocation bug. This change preserves CAM11 and does not fix
the separate c89 precipitation failure.

There are no additional timestep operations or transfers [I: static selection].
The affected public entry saves 1,656 carry bytes on the measured 4×2×3 probe;
the same shape calculation saves 22,478,400/83,595,564 bytes on PROD d01/d02 [I].
Native CLI memory is unchanged. No GPU timing, peak VRAM or E41 claim is made.

The original combined shared suite hit its 6 GB cgroup cap before producing XML.
The affected operational routes and all shared consumer files subsequently pass
263 checks under the same cap, with JAX compilation caches cleared between tests
and process peak RSS 4,334,080 KiB. This is not a full option-matrix or full-suite
pass. The initial new test's `(6,4)` expectation was corrected to the API's
`(2,3,4)` shape; only that test was repeated. The original failing XML remains.

`gate.py SOURCE BASELINE_EXPORT PRE_CAM_COMMIT OUTPUT` executes the actual old
allocation/radiation functions in the same imported environment. The baseline
export needs `src/gpuwrf/contracts/state.py` and
`src/gpuwrf/coupling/physics_couplers.py` from `f7de5b32e`; the pre-CAM commit is
`0602830b4`. `mutant.py SOURCE BASELINE_EXPORT MODE OUTPUT` restores each faulty
behavior in memory (`allocation`, `radii`, `cloud`) without modifying source.
Use the official RAM wrapper, CPU backend/cache-off environment, assigned cores,
QUIET guard, timeout and lane pytest directory. All temporary snapshots and test
scratch from this lane have been removed.
