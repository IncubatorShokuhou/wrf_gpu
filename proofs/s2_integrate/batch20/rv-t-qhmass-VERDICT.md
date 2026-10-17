# MERGE — t-qhmass 9d1cad351

Reviewed exact head `9d1cad3515fa28b9ebae9bf7b16e2600dc20b3c8`, source tree
`b3e73a82625a54b5b4f60288d42237ac7137936d` (identical to `b9852764f`).
Independent critic: GPT-6.1 Sol, lane rv-t-qhmass, 2026-10-15.
No required patch changes. This verdict admits the CPU-qualified mass-loading
fix; it does not close GPU or full-forecast qualification.

## WRF source and species

- `module_em.F:171` supplies the full moist array to `calc_cq`;
  `module_big_step_utilities_em.F:856–870` sums every active mass species
  across adjacent levels. `pg_buoy_w:2474–2497` consumes that condensate loading.
- `module_first_rk_step_part1.F:208–224` supplies full moist, MUT, C1H/C2H and
  DNW to `phy_prep`; its unchanged downward pressure recurrence is at
  `module_big_step_utilities_em.F:4940–4973`.
- Surface/history PSFC uses this hydrostatic bottom interface: the surface
  caller binds `P8W=grid%p_hyd_w` at `module_first_rk_step_part1.F:633`, and
  `module_surface_driver.F:1988` takes `P8W(I,kts,J)`.
- `calc_p_rho` in `module_small_step_em.F:438–568` is a dry linearized
  relation. `calc_p_rho_phi`'s thermodynamic vapor correction uses P_QV
  (`module_big_step_utilities_em.F:1066–1072`). Neither needs an added hail
  gas term.
- The shared selector contains the frozen six core slots plus present
  `qh`, `qi2`, `qi3`, deduplicated from the central mass Registry. Inactive
  core slots are zero; inactive optional leaves are None. Scalar numbers,
  volumes, aerosol fields and rime partitions are excluded. All three changed
  callers use that selector; the native sweep consumes its optional tail once.
- Original Registry/check_a_mundo confirms separate hail in NSSL 17/18;
  17 resolves to 18, 22 disables hail, and Thompson 38 combines hail in qg.

## Independent measurements

- [M] 10 tests PASS in 8.68 s: eight focused tests plus two acoustic wiring
  tests. Original PSFC 95000 -> 95090 Pa at unchanged 1e-9 Pa absolute bound.
  Pressure/native explicit-RN results match extracted pristine REAL4 bodies;
  face-sum tolerance is the declared analytic roundoff bound.
- [M] Removing only qh produces three genuine numeric assertion failures,
  with zero setup errors: PSFC, pristine pressure and actual acoustic caller.
- [M] Native frontend 6/7/8/9 inputs PASS: one Pallas call, no runtime fp64
  arithmetic; 6/7/8-input kernel jaxprs match the parent exactly.
- [M] Fresh official release probes: Swiss `f672a573`, PROD d01 `6fc715a2`.
  Fresh full native-facade mp22 `68191143` and mp38 `70e2dbca` match the
  sealed pre-fix programs. These identities are regression evidence.
- [M] All 24 sealed receipt checksums and original WRF commit/file/body hashes
  verified. Raw XML contains 344 unique PASS = eight focused + 336 other
  shared, and one inherited BL11 assertion with zero setup errors. The manager
  identifies its separate test repair as `a367c1a57`.
- [M] Audited three complete two-step caller receipts: 45 pristine loading
  observations, zero non-finite values and zero guards. The five owned operator
  ASTs, shared helper and data match the tested successor at each measured
  source. NSSL 17/18 uses controlled original CPU-WRF h12 hail on a verified
  grid: 1815 domain cells stay active, and the interior field reaches the next
  RK byte unchanged. This establishes caller/carry behavior, not CPU-WRF
  trajectory parity. Repeating these unchanged numerical runs was unnecessary.

## Performance, memory and remaining qualification

[M] No-hail/default programs and native launch count are unchanged. The new
helper and central Registry reach the production AOT source fingerprint;
there are no new external baked assets. [I] Active hail adds a load/add at
each mass reduction; the native sweep reads its existing State buffer directly,
without a packed moisture temporary or new persistent carry. [U] Composed GPU
time, peak memory, device operations and E41 remain unmeasured by this lane.

The current matrix independently confirms the full changed-source GPU D6/E41
rerun set: c07(9), c12(17), c13(18), c14(19), c17(24), c18(26), c64(7),
c88(56). The manager also requires v-run mp18/mp18_cu5/mp24/mp26. Coordinate
reuse of equivalent runs; existing affected PASS remains provisional. Unchanged
22/38 paths may reuse this fix's program identity evidence.

The direct writer's no-diagnostics PSFC fallback still omits optional masses.
This disclosed source-only follow-up is outside the three owned sites; the
operational pipeline supplies the corrected diagnostic. Assign the writer
follow-up separately. Global F2/station/24–72 h and release gates remain open.

Evidence: `REVIEW.json`, `receipts/`, `audit.py`, `identity.py`; MAIN experiment
rows RQ00–RQ10 include the two corrected critic metadata-reader setup failures.
Raw mutant XML/log are gzip-compressed without content changes; REVIEW.json
records both compressed and original checksums.
