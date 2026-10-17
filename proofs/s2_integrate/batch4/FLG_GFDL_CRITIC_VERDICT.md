# MERGE — lane/s2-flggfdl cffc0c1b4

Final reviewed commit: `cffc0c1b43e8964104a644586401c4f0a1f453a7`. This supersedes the conditional admission verdict for `dc5139f7e`; `ed5cd08f3` is superseded by the final seal. Scope remains **EXPERIMENTAL CPU-ONLY paired FLG7/7 and GFDL99/99**. No outstanding fixes remain within that scope.

Both metadata conditions are closed. All four catalog entries now contain the correct separate FLG/GFDL numerical disclosures and link the tracked per-band disposition. The independent bound receipt, original critic verdict/evidence, mutant harness and clean test XML match the critic's retained originals byte-for-byte. All three fixture hashes and every per-band count/error/disposition match the independent assessment: seven source-bounded strict deviations, ten residual bands with 285 values outside the source envelope, experimental/D6-pending only.

Independent metadata verification against `dc5139f7e` confirms:

- The only changed product file is scheme_catalog.py; its AST is identical after restoring precisely four note strings. Every other product blob, including the corrected caller, solvers and runtime assets, is unchanged.
- The only changed test source has added comments; its AST is identical. Assertions, numerical bounds, XFAIL selections and mandatory corruption checks are unchanged. Test fixtures are unchanged.
- The final seal alters only documentation/manifests and trailing ASCII space/tab in two copied mutant trace files. Both original and tracked SHA256 values are verified; numerical traceback content and test outcomes are unchanged.
- `git diff --check dc5139f7e..cffc0c1b4` passes.

**Compiled-zero disposition is resolved:** its hard failure is the intentional corruption mutant, not an unmodified-candidate failure. Retained independent results are clean **89 PASS / 17 XFAIL / 0 FAIL / 0 ERROR**, versus mutant **1 hard numerical assertion failure / 1 XFAIL / 0 setup errors**. The hard regression must continue to reject zero outputs; it has not been marked XFAIL. The small residuals remain explicitly disclosed, not relabelled strict or native-envelope passes.

Verification artifact: `final_metadata_gate_cffc0c1b4.json`; reproducible host-only checker: `final_metadata_gate.py cffc0c1b43e8964104a644586401c4f0a1f453a7`. An initial whitespace comparator incorrectly added a final newline to XML; that comparator failure was discarded and corrected without changing producer evidence or numerical results.

Earlier independent caller/seam/mutant/default-identity results and unchanged producer native-oracle/restart/coupled/artifact evidence carry forward. No numerical, model, compile, GPU or identity rerun was performed or needed for this metadata delta. Arithmetic, launches, temporaries and transfers are unchanged; prior resource measurements remain the evidence, with no new cache-readiness or performance claim.

Full forecast D6, GPU/native Pallas and release qualification remain pending under the established scope. Reviewed lane untouched; no snapshots or caches created. Manager owns integration.
