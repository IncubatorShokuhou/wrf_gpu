# F2 — convective RAINCV of the stateless cumulus schemes reaches the land surface (cu 2/3/5/6/16/93)

Lane t-calleraudit · on top of F1 · product commit 3d37fde8d (scan_adapters.py, operational_mode.py) · CPU only.

## Defect (CALLER_AUDIT F2, owner-verified)
WRF passes every cumulus scheme's RAINCV to the land surface: module_surface_driver.F:1568
(RAINBL += RAINCV + RAINNCV) and :3283 (MP_RAINC = RAINCV) → module_sf_noahmpdrv.F:779
(PRCPCONV = MP_RAINC/DT). The port took PRCPCONV only from the held KF-layout rate tuple
(`carry.cumulus_tendencies[6]`, cu 1/4/14/94/95/96/99) or cuplus PRATEC (cu 7/10/11); BMJ (2), GF (3),
Grell-3D (5), Tiedtke (6), New-Tiedtke (16) and Grell-Devenyi (93) hold no such tuple, so
`operational_mode.py` fed PRCPCONV = 0: their convective rain never reached Noah-MP/RUC/SSiB.

## Fix
- The five adapters (tiedtke, ntiedtke, gf, grell_cumulus [g3/gd], bmj) return the RAINCV they
  accumulate (instantaneous per-call amount; G3/GD = PRATEC·DT like their rainc_acc) when asked
  (`return_raincv`; default call unchanged, OFF path creates no ops).
- The cumulus slot asks only when a land precipitation carry exists; the amount rides the in-step
  `_PhysicsStepForcing.convective_amount` (not a carry leaf → no restart schema change) to the post-RK
  land update: PRCPCONV = REAL(RAINCV)/REAL(DT) (noahmpdrv.F:779). KF/SAS/KF99/cuplus/default paths
  untouched.

## Evidence [M CPU]
- tests/v040/test_t_calleraudit_land_precip.py (+7, 44 total PASS): the ACTUAL
  `_physics_boundary_step` for each of cu 2/3/5/6/16/93 (cumulus producer + MP mocked, RK identity)
  ends with `noahmp_precipitation.prcpconv == f32(0.36)/f32(dt)` and prcpnonc 0, and the slot requested
  RAINCV exactly once; the real Tiedtke adapter (column mocked) returns the RAINCV it adds to rainc_acc.
- Mutants (F2_MUTANTS.json): 3/3 killed by genuine assertions, 0 setup errors (post-RK ignores the
  amount; slot never requests it; Tiedtke returns zero).
- Operational smoke tests/test_v013_operational_smoke.py 93 PASS (11.7 min, 16 GiB slot).
- Consumers 141 PASS (cumulus parity/oracle/wiring incl. Grell, GF, Tiedtke, NTiedtke, KF, t-rainnc
  callers, F2 missing-bundle, catalog, precipitation); shared 243 PASS.
- Default identities EXACT: Swiss f672a573, PROD d01 6fc715a2 (F2_SWISS/PROD_IDENTITY.json).

## Performance / memory
Non-default cumulus only: one (ny,nx) REAL division per step in the post-RK precipitation update;
no new launches of note, no carry growth.

## GPU reruns needed
Composites with Noah-MP and cu 2/3/5/6/16/93: c03 c04 c05 c06 c07 c13 c17 (all already in the F1 list).
