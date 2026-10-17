# MERGE-WITH-FIXES — NTU byte-preserving performance change

Target `4df5022bc7f556d1252ac88e245f5680e88b2ea9`, base `297f4344b9`.
Numerical/performance review gates PASS. One catalog wording correction is required at integration; no model fix or repeat numerical run is required for that correction.

- Independent CPU suite: **221 PASS, 0 failures/errors/skips**. Restoring the exact base `find_rc0` gives **one genuine assertion failure, zero setup errors**, because Newton again runs 50 trips. The test's old-reference AST exactly matches base after counter/function renaming.
- Independently compared retained outputs: **80/80 GPU arrays BYTE and finite** on real original-WRF c88 +1h/+12h columns, and **912/912 frozen CPU arrays BYTE and finite** (NPZ SHA `0f70d7e4...`). GPU scripts assert the GPU and official lock; lock/session logs show completed arms. Input frame hashes match the original CPU output manifest. The CPU arrays are validation-reuse evidence, not a new WRF fidelity result.
- Submitted target's own CPU CUDA-target lower has raw TTIR SHA **41ef3401...**, identical to the GPU-measured snapshot. Optimized TTIR matches after replacing only the exact source filename. Kernel closure files and both aerosol NPZ banks are BYTE across the measured snapshots and target. Compiler parameters remain two warps; no GPU rerun was needed for the rebase.
- Independently rerun FULL defaults: Swiss **7379c7d7**, PROD **6487f5b0**, exact against base's approved batch53 IDs.
- **Same-session G06 measured:** 1.68753 → 0.38899 s/call (**4.338x**) at +1h; 1.14183 → 0.28603 (**3.992x**) at +12h. Four calls/frame, reported median excludes the first, every call blocks for completion. The claimed 4.8x/6.2x ratios additionally use the earlier G05 four-warp measurements; they are cross-session ratios, not a matched end-to-end forecast speedup.

Required metadata fix: `src/gpuwrf/io/scheme_catalog.py`, mp56 note, still says current measured runtime is 1.15–1.70 s/call. Label it explicitly as the **pre-shortcut two-warp baseline**, or replace it with **0.286–0.389 s/call on the two G06 original-WRF states, kernel-only**. Preserve the disclosed historical four-warp STACK3496 and the fact that actual two-warp E41 remains pending. This is a string-only correction; source/program identity can carry these results forward.

The change retains WRF DOUBLE arithmetic. Finite fractions keep the same Newton updates; undefined fractions retain NaN and stop extending the column-wide loop. Public measured outputs remain finite and byte-identical. No additional launches, global buffers or host transfers arise from the change; predicates/selects and changed warp allocation affect local compiler resources. Actual two-warp STACK/VRAM are **not** proved by the old four-warp scan (CPU prescreen STACK4672 is not the actual gate). Full24h c88 final-composite D6/E41, peak memory, whole-forecast speed and F2 remain pending as directed by the manager at 02:44Z. Existing experimental qualifications/deviations are unchanged.

The Jensen evidence-pin failure is outside this diff and already has a separate r-pkg040 repair verdict; it is not attributed to this change. The 24-hour saving estimate remains inferred.

Route to **s2-integrate**: integrate the exact reviewed NTU implementation with the one catalog wording fix. Later G07/order/other performance changes are outside this verdict.

Evidence: [REVIEW.json](REVIEW.json); raw reviewer receipts `<USER_HOME>/wrf_gpu2_lanes/rv-t-ntucompile/perf/`.
