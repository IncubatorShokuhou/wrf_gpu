# Writer producer fields — CPU qualification

Scope: manager decisions 2026-10-14 19:05Z and 21:59Z. This is a writer,
producer-retention and restart gate, not full forecast/GPU qualification.

| Group | Implementation |
| --- | --- |
| Computed outputs | SBU `ri3d` → `RIMI`; GBM `kethl` → `EXCH_TKE`; held Thompson surface emissions; the actual six Goddard radiation-input radii. |
| Inactive constants | Guarded Morrison aerosol fields, Jensen classifications, legacy stochastic work fields, NSSL NWP-off `REFD_MAX`, Noah-MP/FASDAS `HFX_BOTH/QFX_BOTH`. Original CPU proofs cover 24 field/configuration pairs and 225 complete files, every cell/all 25 frames. |
| Deferred fields | Nine class D fields disclosed in `KNOWN_ISSUES.md`; class C ice diagnostics and separately unassigned P3/Jensen radar remain outside this candidate. |

`QUALIFICATION.json` binds 504 distinct passing CPU checks and all receipts.
Three accepted real source deletions each produce one genuine assertion and
zero setup errors. The earlier copied-globals RIMI mutant is explicitly
discarded; only `TWF10-delete-rimi` qualifies. Clean-archive asset checks pass.

Both official full default program hashes match: Swiss `f672a573…`, PROD
`6fc715a2…`. `DEFAULT_IDENTITY_REUSE.json` proves the final FASDAS addition
changes only two host IO files; nine model/initialization subtrees are exact.
No GPU work was performed. CPU default-trace peak RSS was 2.90/3.70 GiB.
`PERFORMANCE_SCOPE.json` separates source-derived storage from unmeasured
opt-in GPU launch, compilation, E41 and whole-forecast costs.

`FINAL_SCHEMA.tsv` is the review table; empty issue cells are rendered as `-`.
The hashed full report is retained under `<USER_HOME>/wrf_gpu2_lanes/t-writer/`:
57 schema coverage passes, nine remaining producer-gap failures, three input
provenance refusals and 14 incomplete rows; 1,677 original frames compared.
The current c00/c21/c22 namelists differ from their sealed original CPU inputs,
so their 75 frames were refused. No case or original reference was modified.
c53 remains incomplete with one frame; c78's complete constant proof does not
qualify c53's missing 24-hour reference. Older initialized schemas retain their
actual source/hash and explicit reuse annotations.

Changed active producer outputs still need full GPU comparisons against original
CPU-WRF, E41 and measured resource/performance qualification after ALISIOS DONE.
