# v0.4.3 CPU test hygiene — w43-reds

Candidate: `5724bd928a`, based on frozen main `888804310b` (v0.4.3).
Original before/after evidence: `9d0f8150ab`, based on main `218e27ea28` (v0.4.2).
Status: **READY-FOR-CRITIC**. Fix commits: `f79180f9f5` (memo introspection), `5724bd928a` (native history fixture).

| Test / cohort | Classification | Action | Evidence |
|---|---|---|---|
| `tests/a1_da/test_member_driver.py::test_process_aot_reuse_shares_only_blob_loaded_executables` | Pre-existing stale test introspection | Follow the record-window wrapper to the inner AOT closure; preserve the memo checks | Baseline [M CPU]: 1 failure, `KeyError: '_aot_calls'`, 4.52 s. Repaired node passes within the full module. Frozen-candidate negative control: publishing `Compiled` calls causes one genuine `AssertionError`, zero setup errors (3.94 s). |
| All consumers of `_memo_of` | Same test file only | Run the whole member-driver module | [M CPU] **55 passed, 2 skipped**, 18.07 s, no failures or setup errors. |
| `tests/test_full_history_default.py::test_default_full_and_explicit_reduced_retain_identical_common_fields` | Pre-existing stale callback fixture under release defaults | Pass the synthetic initial-base carry required by native RK output; retain every field-equality check | Baseline [M CPU, FAST1]: 1 failed, 17 passed. Final **18 passed in each of FAST0 and FAST1**. Deleting default FULL history causes one genuine inventory assertion failure, zero setup errors. Shared fixture unchanged. |
| Release CPU cohort | Prior final.sh dependency list + changed/importing consumers since v0.4.2 + AOT/version consumers + fixed history module | 65 files; sequential 6G/12G tiers, final changed file in 2G scope | [M CPU] **1,990 passed, 15 skipped**, zero failures/errors/missing files. |
| Named release-default CPU subset | Explicit `GPUWRF_FAST_DEFAULTS=1` | Seven files, each in a fresh pytest process | [M CPU] **113 passed, 2 skipped**, zero failures/errors. Live worker environment recorded. |
| Full test collection | CPU, ordinary suite defaults | Collect all `tests` | [M CPU] **19,660 tests**, zero collection errors or module skips. |

All 84 original `assert` statements and both assertion-call ASTs across the two edited files are unchanged, including the history test's exact array-equality call. The candidate `src` tree is byte-identical to the v0.4.3 freeze (`6f98d8d8558a3a32b5145a6410a62e294932a1d0`); no product code changes. Both fixes change only test setup, adding no model launches, transfers, allocations, or compile work. C-auto fingerprints only package sources, excluding these test edits. No GPU timing or scientific-fidelity claim is made. RW1–RW4 are not triggered by these test-only changes.

The 15 RC skips are explicit: 11 require optional `miepython`, two require an admitted ALISIOS guarded session, and two require the absent external WN3 case. The defaults subset has the same two guarded-session skips. No new skip or xfail was added.

Validation reuse: the final fixture change affects only `test_full_history_default.py`. Every other `src/` and `tests/` Git blob matches the completed 64-file run. Those results, the other six defaults files, and collection carry forward; only the changed module is rechecked in both modes. Its test names and parametrization remain unchanged.

Raw receipts: `<USER_HOME>/wrf_gpu2_lanes/w43-reds/`. Persistent JAX caching was disabled; all snapshots, pytest scratch and temporary caches have been removed after their final consumers. The pre-freeze broad arm was cancelled in the queue before running any tests; replacement measurements used immutable v0.4.3 snapshots. The final RAM scope END was reported to the manager at 07:54Z; the manager owns restoring the temporary slot count.

`FINAL_RECEIPT.json` binds the final counts and provenance. `VALIDATION_REUSE.json` proves unchanged source/test blobs; `COLLECTION_REUSE_PROOF.json` pins test names/signatures/parametrization. `RC_REPORT.json`, both final history XML files, and both mutant XML files preserve raw results. `FINAL_RC_FILES.txt` is the 65-file list shared with w43-release; its later curated-clone `final.sh` gate remains with that lane.
