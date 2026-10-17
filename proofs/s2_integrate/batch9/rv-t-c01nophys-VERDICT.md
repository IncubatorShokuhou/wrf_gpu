# MERGE — lane/t-c01nophys

Reviewed candidate: `03d3f2375e8d2672baecaf9a4e92ceeff8c1c4fc`.
Measured product: `2e42e125417c3204fc1ec83f183b419b091e87a9`; the entire
`src/` diff between measured and reviewed commits is empty. Independent
critic used an immutable detached snapshot, CPU only, no product edits.

The correction matches the original WRF surface-driver lifetime. In pristine
`phys/module_surface_driver.F:1490`, `sf_sfclay_physics=0` returns before the
shared `PSFC=p8w` assignment at :1988 and the surface/land diagnostic writes.
`module_first_rk_step_part1.F:634,656` passes the actual grid history arrays.
The output Registry identifies these arrays directly, rather than defining a
post-step recomputation. Holding them reproduces the CPU reference quantity.

The PBL-on audit resolved an initially suspicious assignment correctly:
`module_pbl_driver.F:1033` writes **local** PSFC (:789–797), absent from its
dummy argument list and the caller. It does not write `grid%psfc`.
The admitted MYNN path does not receive T2/Q2/U10/V10/TH2 as output arguments;
microphysics receives none of the six held fields. Surface-layer-off,
PBL-off/on and microphysics-off/on PSFC precedence was checked independently.
TEMF/YSU combinations that can write near-surface diagnostics do not qualify
as admitted sfclay=0 counterexamples: pairing checks refuse those combinations.
This verdict concerns admitted configurations; it does not qualify unsupported
WRF pairings. Filtered DFI inputs are loaded after the prepass, and nested
feedback history overlays the held-map merge afterward.

Evidence independently checked:

- [M CPU] 35/35 focused writer, daily sync/async auxiliary-stream, original-WRF
  fixture, nonfinite/shape rejection and NetCDF-lock checks pass (15.60 s).
- [M CPU] Independent deletion mutant restores old last-map precedence:
  h1/h2/h24 fail with genuine PSFC AssertionError, h0 passes, zero setup errors
  (2.21 s). No exception-only failure is counted as a kill.
- [M CPU reading original CPU/GPU files] 150/150 whole surface field-frames
  byte-identical, six fields over all 25 frames. This is not GPU self-comparison.
- [M source/artifacts] 18/18 receipt hashes, five original fixture source-file
  hashes/crops, actual CPU/GPU input hashes, original CPU executable hash and
  two retained actual E41 executable PIN hashes verify. Actual run says GPU,
  rc0, matching original CPU receipt; two E41 entries/required d01 are covered,
  maximum STACK 8 B, no over-limit nonleaf/kInput flags.
- [M CPU trace] Both official release program identities reproduce exactly:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
  Host inspection additionally preserves later-map object identity for all
  eight active surface-layer codes. New loader/merge/TH2 predicates are inactive
  for sfclay>0; this is the source basis for unchanged physics-on history paths.

The frozen retained D6 receipt has zero bounded misses across 60 fields and
25 frames. Full D6 remains **INCOMPLETE**, with five soil-variable shapes
CPU=5/GPU=4, owned by t-writer. No F2 zero-event, release closure, or whole-run
speed claim is made. The worker's three shared-test failures are explicitly
documented with matched frozen-parent failures; the critic did not rerun the
entire shared suite because the focused and identity gates already passed.

Performance/memory: host-only one-time read of six arrays (42,336 bytes for this
case), dictionary-reference overlay without array copies. Existing measured
merge median is 489.840 ns disabled / 203.086 ns active; these are host
microbenchmarks, not forecast timings. Model sources and both release programs
are unchanged; no new timestep kernel, device transfer/allocation, baked table,
restart leaf or compile program. No further GPU validation was needed.

Artifacts: `RV01-tests.log`, `RV02b-scope.json`, `RV03-mutant.log`,
`RV04-swiss.json`, `RV05-prod.json`, `RV06-evidence.json`, `RV06-full-byte.json`
under `<USER_HOME>/wrf_gpu2_lanes/rv-t-c01nophys/`. RV02's initial constructed
PBL-PSFC counterexample is explicitly discarded as an ownership-reading error
in MAIN EXPERIMENTS; it is not a finding against the candidate.

Next: manager integration; carry the five-soil-shape limitation until t-writer
closes it. No requested fixes to this candidate.
