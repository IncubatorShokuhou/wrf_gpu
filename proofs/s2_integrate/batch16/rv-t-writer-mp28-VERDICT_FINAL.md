# MERGE — E199 caller repair accepted

Same critic delta review of `02c88f5383691fc300cc768d5c6f81db90ef1205`, superseding only the E199 condition
in the preserved original `VERDICT.md` (SHA f1d76367687b11b7d872e7ee4f8a866c74bb32b89c534ccafb852d8b9a02bcaf).

The L3b caller supplies the hash-bound original driver's NWFA2D/NIFA2D as held
State inputs, checks their exact values at the real adapter emission call,
requires exactly one call, and checks their preservation on return. These
checks also execute on every registered original mutant arm. Startup emission
remains covered by the unchanged original-CPU frame-zero test.

Independent immutable-final-source results: **12/12 L3b PASS** (6 production,
6 fp64), all original discriminators and unchanged numerical bounds; **1 focused
fp64 test PASS**; own deletion of both actual held-input seeds gives **1 genuine
consumption AssertionError / 0 setup errors**. All twelve result arrays,
decisions, registered dtype vectors and kernel-direct scores match the producer's
numeric packet exactly. No numerical oracle inputs or thresholds were changed.

Full product `src` tree is byte-identical to reviewed cc5a953ea; parent frame-zero
test/fixture/qualification are BYTE. Twenty frozen WRF/NPZ/compare/verify/manifest/
tolerance files, 8 delta packet hashes, 28 producer raw-receipt hashes and 26
archive files independently verify. Every runner definition outside `run_port`
is AST-identical, including frozen input checks and numerical scoring/verdicts.
The fp64 checker now recognizes only the three existing WRF REAL radius leaves
as REAL32 and positively checks that dtype, matching unchanged operational
precision; the registered seven-field fp64 vector is retained exactly.

Prior independently accepted native eager3ULP/compiled19ULP, QNIFA2D/held BYTE,
9 focused checks, genuine geometry mutant, 8 restart byte checks and full
Swissf672a573/PROD6fc715a2 default identities carry forward by exact product/
fixture identity. No repeat of those completed gates and no GPU execution.

Measured changed-caller CPU costs (diagnostics, not forecast speed):
L3b wall 3:20.41, peak RSS 3745856KiB;
focused healthy wall 0:08.88; independent mutant wall
0:06.17; swaps0 in every arm, official6G/noSwap RAM slot.
This test-only repair adds no product launches, allocations, traffic, compile/
readiness cost, carry or restart storage. Existing GPU/E41/performance qualification,
separate 3D fallback geometry and QNBCA/TAOD5502D disclosures remain open; old
defective GPU QNWFA2D is not relabelled PASS.

Artifacts: `E199_CLOSURE.json`, `E199_AUDIT.json`, `e199_l3b/`,
`e199_healthy.xml`, `e199_mutant.xml`, own `e199_mutant.py/source.json`.
MAIN RV11–RV14 logged; no critic product edits. Next: manager merge exact candidate.
