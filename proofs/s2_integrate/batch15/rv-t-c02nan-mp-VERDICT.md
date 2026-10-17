# MERGE — t-c02nan numeric repairs

Reviewed packet `065e4539689bf9ed2c6df0ba267a6d09fcb5787f`; scope only numeric
`1b077d489`, `37984ffb7`, `8236fdf52` and their required tests/fixtures/proofs.
The already reviewed SF1/UST changes are excluded. No producer files changed,
no new GPU run, no numerical tolerances changed. This accepts the repairs;
C08 full release qualification remains incomplete.

**Independent verification [M CPU].** On clean immutable `5e8c3001d` with
`src/gpuwrf` tree `8b7681b3f92db9d48f7e956c30a69f00f04693c3` (BYTE identical
to measured GPU `8236fdf52`): 40 focused tests PASS, 40.42 s. They cover the new
real failure fixtures, Morrison warm/cold/dry regimes and caller wiring, and
Goddard pristine column/diagnostic fixtures. Independent deletion of only
`ni, ns, nr, ng = ni_c, ns_c, nr_c, ng_c` produces genuine AssertionErrors in
both reference-kind tests, with exactly qs/Ns NaNs at levels 17–19 and zero
setup errors. Rebuilt pristine Morrison REAL4 and REAL8 from unchanged WRF
module bodies using the original wrapper and module initialization: 15/15
input echoes BYTE, all reference columns/scalars BYTE, finite and repeatable.
Compiler-defined zero initialization also reproduces the saved references.

**Source and caller correctness.** Goddard WRF :4000 rounds the saturation
quotient before testing SSI. An approximate positive SSI with zero DM enters
the :4063–65 deposition normalization with a zero denominator. The new
`div.rn.f32` preserves this WRF arithmetic; no physical branch, clamp or NaN
mask was added. The CPU literal remains unchanged after helper inlining.
The paid native original/candidate comparison is genuine: original nine-field
NaN signature, candidate 38/38 fields finite, pristine TH BYTE. Internal native
SSI itself was not tapped; CPU-derived counterexample inputs are labelled.
Morrison WRF :2116–20 and slope setup :2232/39 write corrected particle numbers
before NSAGG :2434. The four aliases forward values already computed by the
port, preventing a negative raw Ns fractional power. GBMPBLINIT :1243–52
initializes cold-start TKE to REAL .001: independent actual loader check gives
all mass levels `0.0010000000474974513`, BYTE vs the pristine CPU frame. This
is an initialization repair, not claimed to be the Morrison NaN cause.
Only these four model files differ from base main `f4fd9de77`; other MP scheme
implementations and the Morrison-aero copy are unchanged.

**Default identity [M CPU].** Fresh FAST1 native-facade probes reproduce Swiss
`f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`
and PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
An initial FAST0 identity attempt was stopped before any hash was accepted;
the invalid attempt is logged. No successful numerical gate was repeated.

**Paid GPU evidence independently audited, not rerun.** Both receipts attest
`8236fdf52`, GPU execution, 24 h / 4800 steps / 25 hourly frames, rc0. All input
digests and ten retained executable/metadata PIN hashes verified; actual JIT
and AOT have the same ELF per case. Every D6 frame's coverage/schema/nonfinite
record checked, plus finite sample counts for every numeric field.

| Case | Original CPU-WRF D6 | Raw nonfinite | Actual JIT+AOT E41 |
|---|---|---|---|
| C64 | PASS; 25 frames, 367 numeric fields, 62 bounded PASS | 0 | PASS; max STACK16 B |
| C08 | INCOMPLETE only EXCH_TKE; 25 frames, 359 common numeric fields, 62 bounded PASS, 0 failures | 0 | FAIL; STACK408 B and 5 large transposes |

**Performance/memory.** The number aliases add no arithmetic, persistent
arrays, launches or transfers. The GBM initialization replaces the wrong
MYNN seed at initialization only. Goddard retains one native program and adds
one precise division, with measured component registers 166→163, stack/local
0 B, shared1024 B, compile26.62→26.52 s and ~94.28→94.21 us two-column
host+device median. Those timings are component observations, not a speedup
or whole-domain/S1 claim. C08 inherited worst STACK408 B remains; its Morrison
f64 fallback is unchanged and separate. Whole forecasts were cold and parallel,
so their walls are not performance benchmarks.

**Open release work.** EXCH_TKE output coverage, C08 E41 and the independent
F2 negative-moisture audit remain open. Preserved phase observations include
C64 12,798,532 and C08 18,578,752 negative visits; these are repeated field/phase
observations, not unique cells or an every-event justification. This verdict
does not waive those gates. No required change to the three scoped repairs.

Evidence: `tests.xml`, `mutant.json`, `oracle.json`, `oracle-provenance.json`,
`swiss.json`, `prod.json`, `initial.json`, `audit.json` and reproducible helpers
in this directory and tracked under `proofs/rv_t_c02nan_mp/`.
