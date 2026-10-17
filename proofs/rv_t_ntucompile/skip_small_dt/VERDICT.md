# MERGE — NTU perf #3: skip unused small_dt calls

Target **346cbeb5b65c6d58e36b7586207af02baf1a6ded**, base `04cb09f816`.
Route to **s2-integrate before the FINAL c88 rerun**. No required fixes.

- **224 independent CPU checks PASS**, zero failures/errors/skips. Replacing the helper with an always-call implementation causes **one genuine assertion failure, zero setup errors**.
- All **80/80 retained G08 GPU output arrays are BYTE and finite**; all **912/912 NP14 frozen CPU output arrays are BYTE and finite**, with identical NPZ SHA `0f70d7e4...`. GPU comparisons are regression evidence; CPU byte equality supports reuse of the existing oracle qualification. Neither is a new coupled-forecast fidelity claim.
- All **43 NTU runner source files** match between the GPU-measured snapshot and the submitted commit, and both aerosol NPZ banks match. GPU runner asserts GPU/official lock, session logs record rc0, and the two original-WRF input-frame hashes match the CPU manifest.
- Exact AST comparison after undoing only the wrapper/import proves the generated `advance_small` arithmetic, arguments, all 29 merge assignments and later consumers are unchanged. Every result is merged with the same `m9 & alive` mask, and no result is read later. `small_dt` copies its input dictionary and has no external write effect; the zero-result branch is never selected into state when no level is active. NumPy/literal reference execution always calls the original routine.
- The offline emitter independently reproduces **all four generated files BYTE**, including unchanged scalar routines. The candidate's own clean immutable native lower has raw TTIR SHA **3071fcff...**, identical to the measured implementation; optimized TTIR differs only in the exact source filename. The new condition is **one resident scf.if inside the column program**, not a host timestep decision.
- Independently rerun FULL defaults: Swiss **7379c7d7**, PROD **6487f5b0**, both exact.

Same-session, same-work G08 medians (seconds/call, four calls with the first excluded):

| Real original-WRF state | Base | Candidate | Reduction |
|---|---:|---:|---:|
| c88 +1h, 1764×44 | 0.388235 | 0.222948 | **42.574%** |
| c88 +12h, 1764×44 | 0.280647 | 0.223763 | **20.269%** |

These are measured kernel-only gains, not whole-forecast speedups. G08 kernel compile was 46.71→49.80 s; that observation is not a full R3 result. There are no additional column launches, public buffers or host transfers; the reduction/branch and filler values remain local to the existing Pallas program. Actual peak VRAM and GPU E41 on the final composed candidate remain open. The producer's CPU NP13 resource prescreen used dirty working source (retained TTIR matches this commit); its 54 s timing is not reused as a rigorous immutable timing claim.

Keep the disclosed experimental status and existing scientific deviations. Actual two-warp E41, F2, full24h D6/inventory and whole-forecast performance are for the planned FINAL c88 composite. Later G09/other optimizations are outside this verdict.

Evidence: [REVIEW.json](REVIEW.json); raw reviewer receipts `<USER_HOME>/wrf_gpu2_lanes/rv-t-ntucompile/skip/`.
