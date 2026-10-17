# MERGE — t-cfix-phyprep series 1

Head `cd518dc813bde7b323131802d3201fd875e65cb5`; base
`c0793d6948067c37797268d7ac063c74d48c1690`. Scoped CPU source/oracle/caller
acceptance for F4/F5 and the posted callers' F15/F16 changes. Composed GPU
qualification remains pending for c03/c04/c06/c07/c25.

Independent critic evidence [M CPU]:

- CP01b: 35 PASS, zero failures/errors, 136.70 s. All 13 producer oracle/caller
  gates, 17 changed YSU-fixture consumers, and five affected operational smokes.
- CP02/CP02b: independently rebuilt pristine phy_prep and
  conv_t_tendf_to_moist on the original CPU-WRF frame with the real constants.
  Frozen NPZ SHA `275e10030c8c46615213100ebcafb980ba22f67a70ab34cc70a32f68873f0ef1`
  reproduces exactly; all 62 arrays BYTE, zero nonfinite values, full provenance
  manifest exact. Oracle dry theta equals original CPU history T+300 BYTE on
  all 12×44 sampled levels (max absolute difference zero).
- CP03: five additional two-step actual `_physics_boundary_step` caller/carry
  cases PASS, 35.40 s. Real adapters, deterministic column-rate sentinels and
  inert RK isolate source/carry plumbing. Independent literal expectations
  check next-step dry T and retained theta_m/qv updates. No numerical dycore or
  forecast claim.
- CP04: native-facade CPU production traces EXACT:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
- CP05: independent in-memory MYJ dry-rate conversion deletion produces one
  genuine numerical AssertionError, zero setup errors/skips (6.30 s).
  Producer's 17 mutation definitions/receipts inspected.

Source matches pristine first_rk_step_part1 PBL/cumulus argument mapping:
dry TH/T, nonhydrostatic PI, hydrostatic P/P8W, surface face pressure, total RHO,
and g=9.81 geometry. MYJ/GFS/BMJ/GF/NTiedtke State-only paths use the old-state
conv_t_tendf_to_moist terms. Tiedtke's pressure/height change is explicitly in
scope. The shared selected-sf1 branch is AST-exact (CP00); all other admitted
shared PBL consumers require sf1. Direct adapter-only legacy calls can change.
No physics kernel, State schema, baked table or default MYNN path was changed.

The inherited total-density reconstruction uses the documented input-ULP bound
for rounded total geopotential and logarithm conditioning. Its disclosed
approximately 2.1e-5 relative residual is not fixed by this series. Numerical
tolerances were not altered. The corrected smoke fixture has consistent sigma
metrics and stores theta_m; it preserves the trigger/mutation assertions.

Efficiency [I/U]: affected callers add a hydrostatic-face scan and total-density
reconstruction plus column-shaped REAL intermediates. Actual GPU launch count,
temporary liveness/peak memory and compile/readiness delta are unmeasured.
Source complexity does not establish the producer's "negligible" cost claim.
Measure only the changed composed GPU programs; CP04 establishes unchanged
default programs. No GPU commands were run for this review.

Integration follow-up: F6 adds required RQC/RQI outputs to the MYJ/GFS adapter
ABI. Before composed consumer tests, update four phyprep test mock sites
(`_observe_myj`, `_install_mock` MYJ/GFS branches, and the GFS legacy test) with
zero condensate-rate outputs. This source-inferred integration detail was sent
to both producers through MAILBOX. The standalone posted series passes; F9 RK
conversion, surface handoff and other F4 callers remain separate series.

CP01's broad smoke arm was terminated incomplete after unrelated PBL compilation
reached approximately 12 GiB RSS. Its partial dots are not counted as passes;
the completed narrowed CP01b supplies the gate evidence. All attempts logged in
MAIN EXPERIMENTS.tsv. Private Fortran build tree removed after CP02; snapshot
removed after the final identity arm. Cache disabled throughout.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PP1/CP00*`–`CP05*`,
`test_full_carry.py`, `run_mutant.py`.
