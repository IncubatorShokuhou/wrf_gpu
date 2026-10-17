# MERGE — same-critic P3 radar FIX closure

Target: `35e66b76b6fa2ee05ff4f7399ebffc49d12941eb`, `lane/t-writer-p3-radar`.
Review scope: only the two requested fixes since reviewed `6e911aa42`. This decision supersedes the two-item FIX verdict for that earlier target; the earlier signed evidence is preserved unchanged.

Both items close:

1. The common `held_diagnostics` helper forwards the selected Operational or RUC carry's existing radar object on every applicable return. The actual daily loop routes Operational, RUC3, SSiB8, lake, TEMF and mixed cases into this helper. Independent rerun: **25 passed, zero errors**, including 13 route combinations for P3 and NSSL, 4.852 s. Exact object/bytes survive through publication. Restoring the actual prior helper independently gives **10 genuine assertion failures, 3 working controls, zero errors**, 3.066 s. FDDA contributes only `HFX_FDDA`; moving the radar overlay after it introduces no field conflict.
2. Literal/FMA NPZs and the P348 report are tracked and byte-identical to the originals already verified in the first review. Their manifest seals the same strict 0.0003 dBZ band and registered WRF contraction maximum 4.766279220581055 dBZ. The new test checks every reference hash, ties the report to the existing accepted qualification, and asserts the fixed bound. Independently executed the changed loader/assertions from an eight-file clean git archive: **PASS**. Appending bytes to the archived FMA file yields **one genuine hash assertion** before comparison. The producer's sealed actual three-step clean-archive test also passed; its eight raw receipt hashes and twelve packet hashes verify.

Source audit confirms the sole product change is `daily_pipeline.py::_surface_diagnostics_for_output`. All other product files are byte-identical; the rest of that module and the numerical test's step/comparison tail are AST-identical. Prior numerical, default-program identity, restart, native CPU and retained CPU-frame evidence therefore carries forward. No model/default trace or forecast was repeated in this delta review.

Performance/memory: the fix forwards the same radar object via a host dictionary update; it adds no array copy, scattering calculation, device operation or synchronization to the timestep. Previously missing routes now incur the required radar output transfer/IO. No new speed claim. GPU/E41/full c23 D6/VRAM/runtime qualification of the 27-output ABI remains pending; the prior inherited count failures and unrelated broad-smoke memory refusals remain disclosed.

All work used CPU12,13,28,29, nice19, JAX CPU/CUDA hidden, official 6G/no-swap slots, QUIET guards and timeouts. Snapshot, archive and pytest scratch removed; persistent cache disabled; no GPU commands or producer edits.

Verdict: **MERGE** for this reviewed candidate and the two requested fixes.
