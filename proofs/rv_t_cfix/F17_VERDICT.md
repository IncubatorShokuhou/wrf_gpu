# F17 — FIX: legacy radiation cadence reaches GBM one interval late

Reviewed `2daaa8e6d3`, implementation `b222e98d21` + acceleration `9fc68f2413`, base `50517329d8`.
Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PM3/`; decisive `PM08-refresh.json`, `PM08.xml`, `test_gbm_refresh.py`.

**Blocking caller error:** the admitted bl12/sf1/LW1/SW0 configuration refreshes legacy radiation after GBM. In the independent actual `_physics_step_forcing` check, entry RTHRATEN is −3e−5 K/s and the radiation sentinel returns −7e−5. GBM receives −3e−5 first, then the carry is refreshed to −7e−5: order `PBL → radiation`, with a 4e−5 K/s discrepancy (57.14% of this constructed current rate). The actual suite resolver accepts the configuration. This is a caller/cadence witness with isolated kernels, not a measured forecast error.

Pristine `module_first_rk_step_part1.F:264` calls radiation before PBL `:1113`; `module_pbl_driver.F:2099` passes current grid%rthraten to GBMPBL. Candidate OM `:9658` restricts early legacy refresh to bl1, `:9822` passes the still-held rate to bl12, and `:10153` performs the late refresh. Existing F17 tests exercise only run_radiation=False and therefore miss this error.

Required repair: refresh legacy radiation on the step-entry state before GBM on refresh steps, once; preserve held steps, RK coupling and the default identities. Add a refresh/hold/two-step caller check. No global radiation rewrite is necessary for this narrowly scoped repair. If choosing broader changes, coordinate with the radiation owner.

Healthy evidence retained:

- [M CPU] Six producer numerical/caller gates plus one independent two-step real-adapter/carry check pass. The latter verifies held RTHRATEN and next-call prognostic TKE; solver/RK numerics are isolated.
- [M original WRF] Fresh unmodified public GBMPBL builds reproduce REAL4 with/without radiation and the DOUBLE transcription, all 13 fields per arm, raw output hashes and build manifest. The 16 real c08 columns use explicitly **constructed** RTHRATEN; this is not a history-field claim. The established compiler-zero-init policy handles the upstream KPBL uninitialized read. DOUBLE transcription gate <1e−9 normalized; REAL gate 4× pristine REAL/DOUBLE spread +1e−6, PBLH/KPBL gates unchanged.
- Clean-build tooling defect disclosed: builder invokes make_gbm_fixture before writing build_zero.json. The reviewer assembled the manifest from the already completed pristine binaries/dumps, then verified all numerical values/digests; no recompilation or tolerance change was needed.
- Independent RTHRATEN deletion is one genuine assertion failure, zero setup errors/skips. Both full mp8 default hashes independently EXACT (Swiss f672a573 / PROD 6fc715a2).

Efficiency: producer reports 11.635→17.385 ms per 1760-column component call versus the old simplified search (+49.4%; correct work differs), and 19.318→17.385 ms versus the literal implementation. The timed ACC module is BYTE-identical to final source. However, the raw timing script names a live checkout and its receipt lacks immutable tree/input hashes; these are producer component observations, not a qualified S1/whole-step speed result. State-machine acceleration preserves WRF sequential sums. Actual GPU peak/temp/launch cost and E41 remain open.

Do not merge F17 as gate-green until the refresh-step error is repaired. Recheck only the changed caller/cadence artifact and necessary default identity binding; retain the exact-source original-column evidence. No producer edits/GPU run by this critic. Snapshot/builds/caches removed.
