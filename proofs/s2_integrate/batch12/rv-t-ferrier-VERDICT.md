# MERGE — lane/t-ferrier

Reviewed candidate `6908324fd9f7dc3260537d6970772a1b0884dcc1` against its main base
`06f224fdc`. This approves the two-site Noah-MP lowering fix, with no requested fixes.

The defect is static REAL TIMEAN unary negation reaching installed Triton's
`lax.neg_p -> _minus` as `TypedNdArray`, rather than an MLIR value. The only
production change replaces `-timean` with typed REAL32 `-0 - timean` in
`groundwater_step` and `_equilibrium_soilwater`. This is exact negation for
finite REAL operands under round-to-nearest, including signed zero. Ferrier
equations, assets, scientific fixtures, precision declarations and admission
status are unchanged; mp5/15 remain admitted, mp85/95 remain refused.

Independent evidence on a detached candidate snapshot, CPU16–19/nice19,
MemoryMax6G/Swap0, persistent cache disabled, no GPU:

- RF01: 13 tests passed in 6.16 s, including real native CUDA-target lowering of
  both groundwater options and both admitted Ferrier entries, plus original-WRF
  groundwater/equilibrium/prognostic-water fixtures.
- RF03: all 1,872 finite literal outputs byte-equal to pristine-WRF fixtures;
  all 1,872 compiled candidate/control outputs byte-equal over five arms.
  Compiled old/new is regression evidence, separate from the pristine reference.
- RF04/RF05: actual c62/c63 first-step CUDA lowers pass with 149/150 Triton calls,
  42.179/37.807 s measured load+trace+lower, peak RSS 2,608,492/2,592,252 KiB.
  Both input SHA sets and complete StableHLO hashes exactly match worker FF10/FF12.
  These lower native Triton, never the Pallas interpreter at trace time.
- RF06/RF07: independently rerun official identity algorithm, adapting only its
  scratch path. Swiss FULL `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`
  and PROD d01 FULL `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` match.
- RF02 reproduces the two product lowering exceptions after deleting the fix.
  RF11 strengthens the check: the same assertion wrapper passes both positive
  cases, then yields exactly two genuine native-lowering AssertionErrors with
  `TypedNdArray`/`lax.neg_p` provenance and zero setup errors after deletion.
  RF10's first assertion wrapper used the wrong installed helper name; its
  instrumentation refusal is preserved, excluded from acceptance, and corrected
  only in RF11. No failed attempt is counted as a passing gate.
- RF08 verifies 29 compressed AND raw receipt hashes, all 13 declared source/
  asset hashes, three pristine reference source identities, and byte identity of
  both historically measured production blobs to the final candidate.

Pristine `module_sf_noahmplsm.F:314` declares TIMEAN REAL; :9595–9632 declares
groundwater fields/QDIS REAL and S_NODE alone REAL(KIND=8). The fix changes
neither these types nor the retained S_NODE casts. Ferrier DOUBLE saturation
code is byte-unchanged. The float64 warning persists on successful lowerings;
it does not explain this exception or constitute a new precision change.

Sibling audit: RF09 inventories all 77 named unary-negation sites in Noah-MP.
Both static TIMEAN sites are covered; none remains. Relevant other sites are
`groundwater_options.py:42,50,51` (field-derived fff/zwt/ka),
`groundwater.py:181,197,251,347,362,365,366,522` (water/dt/soil operands),
`water_hydro.py:289` (soil kdt), and `canopy_options.py:38` (czil gathered by
`energy.py:1058`). Inventory is not universal native qualification of every
possible scalar specialization; no speculative broad rewrite is warranted.

Resource assessment [source/I]: zero added State arrays, domain intermediates,
native calls, transfers or synchronizations; one existing scalar operation per
active water path changes and can fold within the existing kernel. Successful
CPU lowering RSS is measured above. GPU timing/VRAM are unknown; no speedup
claim is made. Unchanged default identities preserve their program structure.

Actual GPU executable/E41 and full CPU-WRF forecast D6 remain release-owner
gates. This verdict approves the localized lowering fix, not forecast fidelity
or release readiness. Snapshot and disposable review caches are removed after
sealing receipts; compact evidence remains in this directory and the lane root.
