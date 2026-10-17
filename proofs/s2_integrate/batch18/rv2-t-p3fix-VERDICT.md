# MERGE — MP52 compiler fix

Reviewed `lane/t-p3fix` at `0bb2ad921b8f4eb4a6d7a49ce90571d9d7f3d852` against critic base `ee7232f50`. Acceptance covers the MP52 compile fix only. No required fixes.

The production diff contains exactly two files: `_p3_memory.py` and `phys_p3.py`. New primitives and four-warps selection are used only for `mp == 52`; other P3 codes retain their stock load/store operations and two-warps selection. No shared compiler pass, vendor installation, arithmetic implementation, table, physical bound, or coupled call boundary changes.

The memory semantics are sound for this call site. All vector references and results are float32; the existing Pallas address computation produces byte addresses aligned to four bytes. The column grid owns each output column. Level indices remain unclamped; the `level < nz` predicate suppresses all padded global accesses. Lookup corners stay within the unchanged coordinate bounds; gamma and DERF indices remain bounded. Load fallback values are explicitly converted to REAL4 before assembly. Scalar operations retain stock lowering and its single-writer election. Vector output exchanges preserve all 32 bits and serialize identical redundant-lane writes from sliced layouts; output references are fresh, written only at the end and never read by the kernel. Inline assembly is marked impure, and final atom instructions survive actual native compilation.

Independent CPU checks on a detached snapshot of the reviewed commit:

- [M] Native SM120 compile: lower 2.742 s, compile 40.063 s, REG255, STACK112 B, shared1024 B; no non-leaf >1000 SASS and no unknown callgraph. Producer's 63.990 s is reproducible as a completed compilation with the same resource result; timing differences under shared CPUs are not a speed claim.
- [M] 34 focused memory/interface/fail-closed controls pass in 1.06 s. Deleting the actual native fill cast triggers the required typed-IR `AssertionError`: one assertion, zero setup errors, 3.161 s.
- [M] Final-source CPU interpretation of 3 steps, 8 columns and 44 levels gives all 84 field-frames / 26,472 values byte-identical to the retained pre-fix baseline, with no nonfinite values. Regression evidence only.
- [M] Full default program identities rerun and exact: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
- [M] 19 sealed receipt hashes, 18 raw artifact hashes and all 25 GPU whole-file hashes verified independently. Reviewed source/data/packaging trees equal measured GPU commit `b297eef1f` byte for byte. Retained actual GPU receipt asserts `gpu`, rc0 and this measured commit.
- [M] Retained frozen D6 statistics independently checked: all 3,100 scalar comparisons for 62 bounded fields pass; all 372 compared fields have complete finite pairs over 25 frames. Overall inventory stays **INCOMPLETE: REFL_10CM**, not full D6 PASS.
- [M] Both retained paid JIT/AOT E41 entries cover d01 and pass; actual ELF SHA and both PIN hashes verified. Actual P3 REG255/STACK168/shared1024/non-leaf0, with no oversized kInput transpose. Static transfer check on both changed files passes.

Performance and memory: one existing column kernel and unchanged ABI/global buffer count. No new launches, intermediate global arrays, or timestep host transfers are introduced by the diff. Four-warps geometry, atomic exchanges, redundant-lane writes, REG255 and actual STACK168/STL42 have real costs. The previously uncompilable baseline has no valid matched execution timing; no warm speedup, cached readiness, roofline or peak-memory delta is claimed. The retained cold 480.167 s forecast and sampled 5374 MiB VRAM are qualification evidence under shared resources, not S1/R1 closure.

Release limitations remain binding: active `REFL_10CM` producer/coverage gap; 17,896,142 negative F2 phase visits and extra-P3-species coverage remain unattributed; compiler-ZERO original-WRF/MYNN uninitialized-RMOL sensitivity and prior MP50 eight FMA residuals retain their experimental disclosures. These are not newly closed by a memory/compiler fix.

No GPU used by this critic. Original references, producer outputs, PINs and ELF untouched. One failed private identity staging setup was logged as SETUP_FAIL; only the identity probes were retried, successfully. Scientific runs were not repeated. Review artifacts are in this directory; final cleanup is recorded separately.
