# MERGE — F2 correction + F24

Reviewed `412e0eeff..74995985f3254183b81d4815e5a1ea22941eeeee`, product `76860e2d1`.
The superseded `c3379876f` has exactly the same product source; 749 adds the completed smoke packet.
The previous F2 FIX is closed. The new Noah-classic/PX precipitation wiring meets the registered CPU/source gates. No product edits or GPU work were performed by the critic.

## Independent results [M CPU/source]

- **78 PASS**, 0 failures/errors, 24.90 s: the F24 and F1/F2/F2-fix tests plus precipitation/restart consumers (`RC01-tests.xml`).
- **12 PASS**, 0 failures/errors, 14.05 s: the independent real-CU-adapter two-step matrix that previously produced six MP0 failures. Every CU2/3/5/6/16/93 × MP0/MP6 case asserts operational suite admission, consumes zero on its first surface call, then consumes the previous call's RAINCV/DT through the actual State-to-column view and Noah forcing assembler (`RC02-caller.xml`). Leaf CU solvers, dynamics and land solve are sentinels; this verifies caller wiring, not forecast fidelity.
- **2 PASS**, 4.30 s: actual Noah-classic/PX surface hooks using existing WRF-derived smoke bundles reach their real solver arguments with active amounts: Noah PRCP=.005 mm/s, FFROZP=.4375; PX PRECIP=5e-6 m/s (`RC04-surface.xml`). The solvers are intercepted after the real forcing assembly, before evolution.
- **2 PASS**, .99 s: the exact changed hydration precipitation block preserves the five amount/DT slots, combines RAINSHV in the existing convective projection, and handles absent/present optional SR (`RC08-hydration-prefix.xml`). This is a projection-block test, not a complete original-WRF restart import.
- **7 genuine assertion kills**, 0 errors: restoring the MP0 seed exclusion kills all six corrected two-step cases (10.79 s); replacing Noah-classic FFROZP with zero kills the actual solver-input check (2.19 s). Mutations are confined to process memory; production source bytes remain clean (`RC05-seed-mutant.json`, `RC06-sr-mutant.json`).
- Fresh gfortran REAL4/no-FMA evaluation of the untouched WRF statements reproduces **288 rows / 864 outputs plus sources, inputs and program hash byte-for-byte**. Fresh and tracked receipt SHA256: `686f815f1f0e9ad1edade800302736939314ba8822db5ea8c2abb26e06530336` (`RC07-oracle-comparison.json`). The independent RC01 oracle comparison meets the frozen sampled gates: PRCP worst 1 ULP, PX PRECIP worst 2 ULP, FFROZP exact. These are observed limits on the registered matrix, not a universal conditioning claim.
- Both independently re-run release program identities are EXACT: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` (`RC03-swiss.json`, `RC03-prod.json`).

Total independent healthy checks: **94 PASS** (78+12+2+2). Oracle rebuild and identities are additional evidence. The producer's 93/93 per-test smoke receipt is present in 749; its broader 2,313 PASS / 15 SKIP / 3 classified pre-existing failures is reused, not relabeled as an all-green suite.

## Source and restart assessment

The seed selector now follows the actual LSM consumer independently of MP selection. MP0 returns zero microphysics amounts, so the newly seeded convection-only path both collects RAINCV and avoids the former bare-State/tuple incompatibility. The six original missing paths are closed without loosening admission or tolerances.

Pristine surface_driver.F:1568–1570 and :4539 define stepwise RAINBL and its reset for bldt=0. Noahdrv.F:835 and :859–866 use PRCP=RAINBL/DT and FFROZP=SR; surface_driver.F:2708 enables FRPCPN when SR is present. PXLSM.F:482 converts the same amount to m/s. The slight sampled quotient/sum rounding differences are disclosed and satisfy the registered gates; no physical clipping is added beyond the original MAX.

SR comes from the producer's existing WRF SR, rather than a generic frozen-channel reconstruction. Generic Lin/WSM/WDM/Goddard diagnostics already expose SR; Morrison/Morrison-aero now forward their existing output. Ferrier/Goddard4ice/NSSL/Milbrandt/CAM/NTU/Thompson-hail forward their own SR. P3 requests its existing SR output only for Noah-classic. The explicit Thompson8/28 fallback matches the original `mp_gt_driver` numerator including ice and its left-associated total. Kessler/SBU/Jensen do not receive an SR argument in the original driver and preserve zero. Noah-MP filters the extra optional field before its column view.

Pickle format v6 rejects pre-v6 active Noah-classic checkpoints; new pickle and NetCDF SR round-trips passed. Native checkpoint source identity continues to refuse older model-source snapshots. A five-field restored tuple defaults SR to None, which Noah-classic refuses before the land solve. Existing SSiB validation accepts the inactive None channel without dropping its original five required precipitation leaves.

**Qualification limit:** full original-WRF restart hydration remains fixture-dependent and Noah-MP-specific (`hydrate_wrfrst` requires a Noah-MP carry). Its reader currently omits SR from `wanted`; a future SR-enabled original-WRF import must add that inventory field and qualify the complete path. The projection-block check above does not establish Noah-classic original-WRF restart interchange. This does not affect the qualified native/pickle/NetCDF carry round-trips or the new API LSM forcing.

## Performance and remaining gates

Noah-MP release program identity is unchanged, with SR=None and the same five device precipitation leaves. Noah-classic adds one fp32 surface SR leaf: **4 bytes/column [I static payload]**; newly participating LSM/MP0 paths retain the existing five rate leaves (20 bytes/column [I]). P3/Noah-classic adds a requested output. The changes add surface arithmetic and forwarding, with no new vertical solve, host callback, timestep transfer or explicit synchronization. Actual non-default GPU launches, peak allocation, compile/readiness and timing are **[U]**; unused SR dictionary entries are not by themselves proof of zero native-output cost.

No additional composite is introduced by F24 because the current matrix uses neither Noah-classic/PX nor MP0+LSM+CU. Existing F1/F2 affected composite GPU/fidelity gates remain binding; this verdict does not declare those forecasts qualified. Noah-classic/PX remain API-only under existing admission. No new independent product fix is required in this reviewed range.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/F24/`; durable copy under `proofs/rv_t_calleraudit/F24/`. All heavy probes used the official 6G/no-swap RAM wrapper, JAX CPU, lane affinity/nice19, explicit QUIET guard and timeouts. No persistent cache was used. Snapshot, pytest and temporary oracle build directory are removed after the last dependent arm. Tracked log/XML copies trim trailing whitespace; original raw receipts remain unchanged externally.

Count clarification: the final source collects 71 owned checks (50 land-precipitation + 21 F24), rather than the packet's 72; adding the seven existing precipitation tests gives RC01's measured 78. This is a packet counting discrepancy, with no changed outcomes or tolerances.
