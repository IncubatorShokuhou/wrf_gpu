# MERGE — t-cfix-phyprep series 2

Reviewed source at `cfe70e98e6`, series-2 implementation `7739b86af0`, base
`1b1e31e575`. CPU scope: F15 hydrostatic pressure staging for G3/GD, QNSE,
TEMF, EEPS and KEPS. GPU coupled qualification remains pending.

[M CPU/source] PQ01: 27 tests PASS (all 22 phyprep gates and five changed Grell
consumers). PQ02: the independent original-WRF reference is reused with exact
fixture/builder/shared-helper identity and fresh pristine source/constants SHA
checks. It is the previously rebuilt REAL4/8 oracle, not old port output.

PQ03/PQ03b: seven actual two-step operational caller cases PASS for this series
(G3/GD, QNSE MF0/MF1, TEMF, EEPS/KEPS). Real argument assemblers and dispatch
execute; sentinel column boundaries and an isolated RK mass/moisture update
require fresh next-step hydrostatic pressure. No numerical dycore/forecast claim.
PQ06: all four unmocked QNSE/TEMF/EEPS/KEPS operational smokes PASS (54.54 s).
PQ05: independent QNSE PMID reversion to nonhydrostatic pressure causes one
genuine numerical assertion, zero errors/skips (3.08 s).

PQ04 both default native-facade CPU traces EXACT on the final composed candidate:
Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.

Source matches pristine first_rk_step_part1/pbl_driver mappings: G3/GD P/P8W
and PSUR, shared QNSE surface/MF/PBL PMID/PINT, TEMF P3D/P3DI, EEPS P3D and
KEPS P8W. Exner retains nonhydrostatic pressure. Physics kernels, carries,
tables and unrelated admitted callers are unchanged by this series. The Grell
smoke fixture now uses consistent sigma mass/pressure/heights and retains its
assertions. Reference tolerances are unchanged (64-ULP hydrostatic chain bound).

Efficiency [I/U]: new hydrostatic-profile work replaces the extrapolated fields.
EEPS currently calls the surface-pressure wrapper and profile helper on identical
operands; the wrapper delegates to that helper. Reuse/CSE on the actual GPU is
unmeasured; one profile call can provide PSFC too. No source-only speed, peak
memory or readiness claim is accepted. Measure the changed composed GPU programs.

Review-harness initialization exceptions from the first PQ03 attempt are logged
separately and excluded from numerical failures. Only its four valid cases were
reused; the other six were rerun with valid carries. EPS's two-step CPU harness
uses the qualified forced-wide mode preserving REAL QKE. No production guard was
removed. The producer's known unrelated external-reader-key failure was not
rerun or waived by this scoped review.

Next: merge the scoped series, compose, then GPU D6/timing c05/c13/c17/c26/c27/
c10/c11. Original source-bound scientific gates remain binding. No GPU or
producer edits; all experiments MAIN-logged, cache disabled, snapshot cleaned.
Artifacts `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PP23/PQ01*`–`PQ06*` and helpers.
