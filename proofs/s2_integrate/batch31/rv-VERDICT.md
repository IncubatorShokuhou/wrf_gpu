# MERGE — lane/t-p3mp53 f552e5fb2

Independent CPU-only review of the MP53 compiler fix. All pre-registered lane gates pass. No production files were changed by this critic and no GPU job was launched.

- [M] Native SM120 compile reproduced in 9.004 s (2.398 s lowering), REG255, STACK48 B, shared1024 B, STL11; callgraph PASS. This is an isolated CPU compiler prescreen, not forecast timing.
- [M] 108 targeted tests PASS / 1 unrelated archive test deselected. Deleting the actual predicate reshape branch produces exactly one AssertionError and zero setup errors.
- [M] Final-source Pallas interpretation matches all 66 retained baseline field-frames / 20,136 REAL4 values byte-for-byte across three steps, eight columns and 44 levels. This is regression evidence.
- [M] Independent full program hashes match Swiss f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d and PROD d01 6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b.
- [M] Canonical Pallas program hashes for MP50/51/52 match exact pre-change source 11eee66ed. The baseline modules were read directly from Git before import; no production checkout was modified.
- [M] Independently rescanned both paid executable envelopes: rich JIT and rich AOT E41 PASS, full required d01 coverage, identical native ELF SHA6c73c119815e7f79. Actual MP53 REG241 / STACK0 / shared1024 / STL0, no oversized nonleaf callee or kInput transpose fusion.
- [M] Verified 22 sealed receipt hashes, 17 raw artifact hashes, all 25 GPU output frame hashes, measured/final src+data+packaging Git identities, actual GPU/CPU-ZERO input equality, frozen scorer and tolerance hashes. Retained GPU receipt asserts cuda:0 and rc0 on source38d001552: 24 h / 4800 steps / 25 frames, 431.920 s cold, sampled process VRAM5376 MiB.
- [M] Retained original-WRF ZERO c24 D6 covers 367 numeric fields plus Times, every frame, with no missing fields/schema failures/nonfinites. Independently checked 3,100 frozen scalar limits across 62 bounded fields and 25 frames: PASS. Unbounded fields remain REPORT_ONLY. No forecast or numerical scoring repeat was needed.

Memory/source audit: existing typed MP52 primitives are unchanged except documentation, now selected for MP53. REAL4 vector loads use correctly typed fallback values and predicated accesses. Level indices remain unclamped and masked at nz=44/padded width64 (E106); table coordinates retain established in-range bounds. Scalar precipitation stores retain the stock single-writer mask. Atomic exchanges make replicated vector output writes race-free; each column owns disjoint output addresses. Bool singleton reshape is bit-preserving, enabled only in the MP53 kernel, and reset in finally. No arithmetic, thresholds, tables, outputs, extra launches, global buffers or host transfers were added.

Performance costs remain explicit: four-warps geometry, atomic exchanges and register pressure; isolated STACK48/STL11 differs from the actual paid STACK0/STL0 kernel. The observed cold qualification wall and sampled VRAM support this lane only; no warm S1 speedup, roofline, R1/R2 or global release claim is made.

Open risks retained: 17,951,617 negative pre-mask phase visits (15,623,747 interior), no original-WRF amplitude attribution, and QZI/QIR/BIR pre-mask coverage unestablished. Global F2 remains OPEN. The ZERO-init original-WRF RMOL sensitivity and stack-specific private PTX/Pallas API limitations remain disclosed.

Production/test diff --check passes. Two trailing-whitespace lines occur only in sealed producer raw assertion dumps; preserving their hashes is preferable to editing historical evidence. This does not change the scientific or compiler verdict.

Evidence: AUDIT.json, D6_LIMITS.json, native/receipt.json, scope_{baseline,candidate}.json, swiss.json, prod.json, tests.xml, mutant.xml, and e41/wave_gate.json in this lane's artifact directory. Reproduction commands and bounded run outcomes are recorded in R01..R11 command receipts and MAIN EXPERIMENTS.tsv. Review caches, IR, pytest directories and snapshot are removed after their final reader; producer originals remain untouched.
