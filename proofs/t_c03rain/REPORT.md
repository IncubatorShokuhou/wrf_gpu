# c03 Lin rain and MYJSFC history qualification

Status: **READY-FOR-CRITIC for the delta after `1612acc49`**.
No frozen tolerance changed.

The original c03 rain failure is a deficit, reaching 4.321870 mm RMSE at
24 h; 1.024437 mm is the first failure at 6 h. Graupel accounts for much
of the discrepancy (24 h GRAUPELNC RMSE 3.170302 mm; RAINC 0.078466 mm).
The reference is the retained original CPU-WRF c03 run, not an earlier GPU version.

The final physics candidate is immutable `1612acc49` at
`<USER_HOME>/wrf_gpu2_lanes/t-c03rain/CR24-tree`, source tree
`9bea5a29cb47e42873a4575d8dbefa0523e620ce`. The final run is
`<USER_HOME>/wrf_gpu2_lanes/t-c03rain/CR29_c03`.

## Changes and independent evidence

- Lin now receives terrain height for the bottom sedimentation Courant bound.
  Pristine public `lin_et_al`, six regimes at HT=0/1500 m: old 8/12 pass,
  corrected 12/12 pass under the existing Lin tolerances. Maximum corrected
  precipitation error is 8.6071e-8 mm. The 18,323-byte original fixture survives
  a clean Git archive with its recorded checksum.
- Lin now receives dry TH_PHY and recouples its result to prognostic theta_m
  through the existing WRF moist-physics prep/finish helpers. The old caller
  warmed actual c03 inputs by up to 2.71633 K, crossing freezing in 1,628 cells
  at 6 h. The independent literal REAL4 WRF thermodynamic fixture passes;
  removing either boundary conversion produces a genuine numerical assertion
  failure, with zero setup errors.
- MYJSFC's equivalent-roughness U10E/V10E are retained from the actual surface
  solve, persisted in optional REAL history leaves, restarted, and emitted by
  the shared single/nested output path. Pristine `module_physics_init.F:1326–1327`
  initializes them from U10/V10 before the first solve; the initial writer now
  follows that rule. The separate SF4 extension below applies the same WRF
  rule to QNSESFC. Other surface schemes have no added history leaves/fields.
- An output-only Eta barrier did not improve the actual executable: all five
  STACK flags, including the 2,456-byte maximum, were unchanged. It was discarded.
  Final E41 remains FAIL: the five old flags plus a Lin input fusion with
  STACK=392 B and kInput=917 HLO instructions; non-leaf flags are empty.
  Both actual native/HLO programs are pinned. The manager assigned further
  E41 work to t-e41opt after this result.

## Validation receipts

All receipt paths below are relative to `<USER_HOME>/wrf_gpu2_lanes/t-c03rain/`.

| Check | Evidence | Result |
|---|---|---|
| Pristine Lin terrain oracle | `CR02_oracle/comparison.json` | 12/12 corrected columns pass |
| MYJSFC oracle, adapters, 3 writer modes, 2 restart formats | `CR05.xml` | 19 pass, 1 pre-existing XPASS |
| Required shared consumers and contracts | `CR08_shared.xml` | 343 pass; 3 unavailable WN3 inventory skips |
| History consumers after reseed correction | `CR15_history_final.xml` | 16 pass; 1 unavailable authentic PROD-operand skip |
| Pre-existing stale history mocks | `CR16_baseline.xml` | 2 identical missing-`nwp_diagnostics` failures on unchanged main; assigned to t-redsuite |
| Final Lin caller and affected consumers | `CR24_caller.xml` | 56 pass; 2 unchanged checks reused |
| Actual bound operational Lin smoke | `CR27_lin_operational.xml` | 1 pass |
| Removed prep / removed finish mutants | `CR28_mutants.json` | 2 numerical assertions; 0 setup errors |
| Final frozen-tree default identities | `CR26_identity_seal.json` | Swiss f672a573 / PROD 6fc715a2 exact |
| Final GPU / full inventory / D6 | `CR29_c03/`, `FINAL_C03.json` | PASS: 25 frames, 341 numeric fields + Times; zero bounded failures, missing fields or nonfinite values |
| Final actual executable E41 | `CR29_c03/e41/` | FAIL disclosed; 2/2 programs scanned and pinned; further work assigned to t-e41opt |
| SF4 coupling, four restart formats, producer retention | `CR37_qnse.xml`, `CR38_retention.xml` | 107 composed passes; one corrected nonphysical test fixture |
| SF2/SF4 lead-zero, shared writer and history consumers | `CR41_consumers.xml` | 10 passes |
| SF4 assignment-deletion / ordinary-wind substitution mutants | `CR44_history_mutant.xml`, `CR45_producer_mutant.xml` | 3 genuine numerical assertion failures, 0 setup errors |
| Final SF4-tree identities and selected-program reuse | `CR40_final_identity/receipt.json` | Both defaults exact; complete c03 canonical program exact; real c26 full-step trace succeeds |

The full identity hashes are:

- Swiss: `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`
- PROD d01: `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`

The first diagnostic GPU arm (`CR09_c03`, source `501e61d52`) completed all
25 frames, rc=0, cold wall 867.4748 s, sampled process VRAM peak 5,862 MiB.
Terrain alone barely changed the rain failure; that arm is not final qualification.
Its two actual native/HLO PINs are preserved; disposable caches were deleted.
The historical lead-zero host serialization replay preserved every original
field/metadata byte and added two original-CPU-byte-equal fields. That replay
is explicitly excluded from final science: the final gate uses fresh raw GPU output.

Final RAINNC RMSE is 0.490215 mm at 24 h (old 4.321870 mm), with a maximum
over all leads of 0.490326 mm against the unchanged 1 mm limit. The final cold
qualification wall is 887.7593 s; sampled process VRAM remains 5,862 MiB.
U10E/V10E are present in every frame, with lead-zero bytes equal to CPU-WRF.
Their later numerical differences are REPORT_ONLY under the frozen manifest.
Disposable final caches and temporary files were deleted after pinning.

## Separate QNSESFC extension

Product commit `d41ee5873`, with test fixture correction `cfbb40c5c`, retains
QNSESFC's already-produced U10E/V10E in its surface diagnostics and writes them
into the optional REAL history leaves before the land overlay. Initial carry
construction and the actual Noah-MP loader reseed both allocate the leaves for
SF4, keeping the scan structure stable. Lead-zero output copies U10/V10 under
the pristine SF2-or-SF4 initialization predicate. The original QNSESFC equations,
column kernel, tables, original oracle and independent enclosure artifacts are
byte-identical to `1612acc49`; see `SF4_SOURCE_REUSE.json`.

The unchanged official `proofs/s1_port/release_identity.py` was run on the final
frozen SF4 tree. Swiss and PROD retain the full hashes above. The full c03
canonical selected program is identical before/after SF4:
`60c7d0db0bce8616f301dc2b4c8acd3128e262b0e7cfaff40852927319d001c6`
(39,389,076 characters). Its 24 h numerical evidence is reused; no duplicate
GPU forecast was run. Real c26 QNSE initialization and full-step tracing also
succeed; this is readiness evidence, not a new c26 D6 qualification.

The first SF4 adapter test accidentally used nonphysical restart byte tags as
atmospheric inputs: 106 tests passed and that probe failed. Only the fixture
changed; its corrected physical land/sea probe passes. CR37's `-k` expression
also deselected the entire Lin-named consumer file; CR41 explicitly executed
the ten relevant consumer nodes. Initial mutant launchers imported the CPU
guard too late and were administrative refusals, not mutation kills. Corrected
launchers produced the three numerical assertion failures reported above.

All own detached snapshots, disposable caches and identity scratch directories
were removed after their last consumer. Small raw outputs, receipts and native
PINs are retained. The Lin range through `1612acc49` is reviewed separately;
this packet requests review of the subsequent SF4, scorer and audit/proof delta.

## Performance and memory scope

The two new REAL surface history leaves cost 14,112 bytes on 42×42 [I].
Lin's additional preparation/finish is device arithmetic at its existing
microphysics boundary; an individually materialized legacy-fp64 profile is
620,928 bytes [I]. Fusion, extra launch count and actual incremental workspace
are not inferred from linear complexity. No full 3-D State barrier is retained.
The final GPU receipt supplies actual cold wall and sampled VRAM; these shared
qualification arms are not warm S1 benchmarks. Output copies occur at history
cadence; no host operation was added to the timestep arithmetic.

For SF4 the extra retained history is likewise 14,112 bytes on 42×42 [I]. The
fixed surface kernel already returns both winds; the extension adds no surface
solve or Pallas call, profile temporary, or in-step host transfer. Incremental
SF4 GPU timing/workspace remains unmeasured. Legacy SF4 restarts missing the now
required wind history fail the existing E78 validation instead of fabricating it.

## Main caller audit requested by the manager

See [MP_CALLER_AUDIT.md](MP_CALLER_AUDIT.md), its source-closure manifest, and
[the density clarification](MP_AUDIT_RHO_ADDENDUM.md).
All 29 active bindings plus passive 0 and eight requested unbound codes were
audited in about 15 minutes. Nine main callers lacked both conversions;
the eight non-Lin suspects were posted with file:line for manager assignment.
Only Lin was changed here. Additional density-convention and Goddard terrain
suspects remain source-audit observations, not new numerical verdicts.
