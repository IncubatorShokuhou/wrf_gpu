# MERGE — mp55/mp56 native compile fix

Reviewed `c2e549f66cc29870b3c2cf3ef22e2d18576360ae` (production source `f8894da622`) against `8d31e07f83`.
Scope: compiler readiness and the memory-lowering correction. Full forecast and release gates remain separate.

- Independent CPU checks: **154 passed, 0 failed/errors/skips**. Frozen pristine-WRF REAL4/REAL8 columns, typed memory, native evidence, wiring and catalog consumers passed. Producer's 427 shared/targeted checks were audited from XML and reused.
- Exact original-source versus final interpretation: NTU 44/12 levels and Jensen 44/40 levels have identical StableHLO bytes (`dc56b5e0`, `0bece37a`, `485507c6`, `ceb6fbff`). Memory substitution reverses exactly to the original kernel files. Stock discharge preserves output bytes; no physics arithmetic or tolerances changed.
- A genuine in-process deletion of the GPU bounds rule triggers **2 assertion failures, 0 setup errors**. The earlier CPU-guard import-order refusal was excluded.
- Independently rerun default program IDs: Swiss **f672a573** and PROD **6fc715a2**, exact.
- `_in_bounds` matches stock vector-gather wrap-once then clamp semantics. In-range indices are unchanged. Column stores use valid level indices; masked padding lanes retain a false predicate, so the address correction cannot create padding writes (E106). Scalar DOUBLE loads retain the stock route.
- GPU receipts bind exact final memory-helper/kernel/launcher blobs to paid `3c7389e5a`, asserted GPU, official lock, original-case input hashes and rc0. Decompressed JIT/AOT payload hashes reproduce the E41 receipts. Whole-step compile: Jensen **366 s**, NTU **311 s**; sampled VRAM **6550/6016 MiB**. Actual c88 receipt wall is **924.212 s**, not the summary's 919.1 s.
- One-hour original-CPU bounded comparisons: **0 failures, 0 non-finite values**, frozen manifest `9df4fab8`. Jensen has 385 common/384 numeric fields; NTU 381/380. Strict inventory is **INCOMPLETE**: Jensen lacks `REFL_10CM`, NTU lacks `QHAIL`. These are not full D6 or 24-hour PASS claims; the manager was notified.
- E41 **FAIL remains disclosed**: Jensen STACK **5472 B**, NTU **3496 B**, above 256 B. Rich JIT has no nonleaf/kInput flags. AOT HLO-size metadata remains UNKNOWN; its cubin matches rich JIT. This follows opt-in policy and grants no default-gate waiver.

No new launches, global buffers or host transfers arise from this lowering. Atomic output exchanges, bounds instructions and local spills remain GPU costs. NTU's measured cloudy-column runtime **1.35–1.85 s/call** is a major open performance limitation; the manager's separate performance work remains necessary. The new helper is in the AOT source-key closure, so source changes invalidate that key; no new cached-readiness or default-speed measurement is claimed. Other MPs' production files are unchanged.

Verdict: **MERGE this compile fix**. Preserve experimental/disclosed status and pending full24h D6, missing-field, F2 and NTU performance work. No producer source fix is required for this scoped verdict.

Evidence: [REVIEW.json](REVIEW.json). Raw reviewer receipts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-ntucompile/`.
