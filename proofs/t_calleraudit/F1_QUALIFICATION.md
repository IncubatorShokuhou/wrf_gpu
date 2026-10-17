# F1 — every admitted microphysics scheme feeds WRF's public precipitation slots to the land surface

Lane t-calleraudit · candidate on lane/t-calleraudit (product commit 00d0f31e9 + test/proof commits) · CPU only.

## Defect (CALLER_AUDIT F1, owner-verified both sides)
`_initial_carry_for_run` seeded the land precipitation carry (`noahmp_precipitation`) only for
mp 8/5/7/15/38 (operational_mode.py, old line 7793). For every other scheme Noah-MP (and RUC/SSiB)
saw no carry, so `noahmp_coupler.prc2d` returned ZERO rain, snow, graupel, hail and convective
precipitation: the land ran dry (no soil-moisture recharge, no snowpack from snowfall). Producers
already existed for mp 9/11/18–22/50–53/55/56 but were never consumed; the generic MP adapters
(1,2,3,4,6,10,13,14,16,24,26,40,97) and mp 28/17 returned no amounts.
WRF: module_surface_driver.F:1568 (RAINBL += RAINCV+RAINNCV) and :3283
(MP_RAINC=RAINCV, MP_RAINNC=RAINNCV, MP_SNOW=SNOWNCV, MP_GRAUP=GRAUPELNCV, MP_HAIL=HAILNCV) →
module_sf_noahmpdrv.F:779-783 (PRCP* = slot/DT).

## Fix (3 product files, mp8 default path untouched)
- Seed the carry for every `mp_physics != 0` (Noah-MP/RUC/SSiB).
- `scan_adapters.MP_PUBLIC_PRECIP_SLOTS`: the slots each WRF 4.7.1 `microphysics_driver.F` CASE
  actually passes; unpassed slots stay 0 as in WRF (e.g. SBU-YLin passes RAINNCV only, :2470-2500;
  WSM3/5/WDM5 no GRAUPELNCV). `_mp_public_precipitation` takes the per-call INSTANTANEOUS kernel amounts
  (`accumulator_increments`, never differences of rounded totals, B63); Kessler's public RAINNCV is
  WRF's last-sedimentation-sub-step value (`module_mp_kessler.F` DO WHILE nfall), kept by the kernel in
  `diagnostics['rainncv']` — upstream WRF behaviour reproduced, not repaired.
- 13 generic adapters gain `return_precipitation` (final return only wrapped; OFF path creates no ops);
  `_fp64_island` passes `(State, side)` tuples (narrow-back stays in the island frame, E134).
- mp28 returns its THOMPSONAERO amounts like mp8; mp17 alias returns NSSL amounts; NSSL and Milbrandt
  now carry HAILNCV and the all-phase RAINNCV total (previously hail was folded into rain, PRCPHAIL 0).
- Resume safety (E78): the nested/native restart identity hashes the whole model source
  (restart_store.nested_identity), so checkpoints written before this change are refused, never
  resumed with a missing carry.

## Evidence [M CPU]
- tests/v040/test_t_calleraudit_land_precip.py: 37 PASS (7.7 s): pristine Registry+driver slot oracle
  (independent of port code, E98); caller path through the real adapters + fp64 island (f64 and REAL
  carries) for all 13 generic codes; NSSL/Milbrandt hail+total; mp17/mp28 dispatcher; seed coverage;
  real WSM6 kernel on an active column (disjoint accumulators sum to the all-phase total, rtol 1e-5).
- Mutants (F1_MUTANTS.json): 5/5 killed by genuine assertions, 0 setup errors (old seed set, no slot
  filter, Kessler accumulator instead of public RAINNCV, NSSL / Milbrandt drop HAILNCV).
- Consumers: 526 PASS / 36 SKIP across 20 touched-seam files (precipitation, NTU, NSSL, SSiB, RUC, MP
  restart binding, MP savepoint/caller tests) + nssl_cumulus/test_caller 11 PASS after its mock was
  updated to the producer's actual contract (`rainncv` key, nssl2mom/driver.py:99);
  shared consumers 243 PASS; tests/test_v013_operational_smoke.py 93 PASS (13.9 min, 16 GiB slot).
- Default identities EXACT: Swiss f672a573, PROD d01 6fc715a2 (F1_SWISS/PROD_IDENTITY.json), re-run on the
  tree rebased onto main 8ce83ef62 (Shin-Hong merge); after the rebase my 37 + Shin-Hong/NSSL/precipitation/
  rainnc-caller tests 84 PASS. Shared/smoke/consumer counts above are from the pre-rebase tree (the rebase
  brought only the independent bl11 Shin-Hong change).

## Performance / memory
Non-default MPs only: one (ny,nx) amount dict per MP call + the existing elementwise
`precipitation_from_step` (5 REAL (ny,nx) carry leaves, 20 B/column) — negligible launches/bytes.
Default program unchanged.

## GPU reruns needed (program changes)
Every composite with Noah-MP/RUC/SSiB and mp ∉ {5,7,8,15,38}: c02 c03 c04 c05 c06 c07 c08 c09 c10 c11
c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c61 c87 c88 c89. Expected effect: land
surface wetter after rain/snow (LH/T2/snow cover), largest where precipitation occurred.

## Not changed / disclosed
- RUC still refuses mp≠8 (OM:7386 'not_wired'); its stale reason text is now outdated but RUC
  qualification with other MPs is out of this lane's scope.
- Convective rain for cu 2/3/5/6/16/93 still never reaches the land (CALLER_AUDIT F2) — next series.
- WRF 4.7.1 registers only mp=18 for NSSL; 17/19/21/22 are port aliases (MP owners).
