# MP-density review: MERGE ab5545b28

Candidate `ab5545b28fc0209229f3aca9d3230f972408da14`, base `e50076096`, model tree `a84a7c0d91e8de972b6423e04760351c5f4638d8`. This is the separately requested NSSL 17/18/19/21/22, Ferrier 5/15 and Goddard 7 caller fix. Verdict **MERGE for the reviewed CPU/source gates**; affected composed GPU/24-hour D6 qualification remains required.

| Gate | Independent result [M CPU unless stated] |
|---|---|
| Affected caller + NSSL column/radar regressions | 38 PASS, 0 failures/errors, 3 intended Ferrier/Goddard REAL-only skips |
| Compiled production adapter prefixes, observed at native-kernel call | 13 PASS: NSSL five IDs in actual fp32/fp64; Ferrier 5/15 and Goddard 7 in actual fp32 |
| Common geometry helper | Both actual carry dtypes PASS; maximum rho relative error 1.16655e-7, maximum DZ absolute error 3.81470e-5 m, within unchanged per-level input-ULP bounds |
| Fresh pristine REAL4 statement oracle | Seven untouched statements; source/constants/program/inputs and all six output bit patterns exactly match the frozen fixture |
| Moist-factor restoration | 16 genuine numeric assertions: NSSL 11, Ferrier 3, Goddard 2; 0 setup errors |
| NSSL one-metre clamp restoration | 1 genuine numeric assertion, actual 1 m versus expected 0.5 m; 0 setup errors |
| Swiss default trace identity | `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` EXACT |
| PROD default trace identity | `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` EXACT |

Pristine wiring was inspected directly: `solve_em.F:3750` calls `moist_physics_prep_em` with al/alb and PH/PHB, then `:3809` passes RHO=grid%rho and DZ8W. `module_big_step_utilities_em.F:5480/5488/5542` forms REAL z=(ph+phb)/9.81, raw face differences and dry rho=1/(al+alb). The earlier `(1+qv)/ALT` at :4856 is overwritten before MP. Native calls receive these arrays: NSSL DN/DZ at `module_microphysics_driver.F:2263/2265`, Ferrier 5/15 at :2956/:2991, Goddard at :2370/:2371. The new compiled probe catches the real adapter call during tracing and compiles the operand expressions; it substitutes no physical kernel output and uses no host callback.

Only three adapter rho/DZ operand paths change. The NSSL driver diff is docstring-only by AST; the shared helper blob and all other model paths are exact to base. Producer a86a model tree is exact to this final head. Existing native-kernel oracle results are reused for unchanged code; the NSSL fixture tests deliberately inject the frozen fixture DN at the seam and are kernel/packing regressions, not evidence of production density. The separate production caller tests and compiled prefixes establish the new density boundary. No redundant 1,355-test or forecast rerun was performed.

The independent oracle is a compiled pristine-statement boundary fixture, not a full WRF forecast: real CPU c05 PH/PHB/MUT/hybrid operands for three lowest levels, with a disclosed constructed finite ALB operand. It covers hypsometric option 2, required by public CLI binding. Low-level option 1 is not qualified here. No-grid tests preserve the legacy dry-EOS convention and are regression evidence. The manifest records units, precision, input-rounding tolerances and source hashes. The two default traces execute zero forecast steps, preserve the pre-F20 registered identities, and make no new release/fidelity/speed claim.

Efficiency [I source]: no extra MP evaluation, adapter host callback, persistent State leaf or precipitation/radar computation is introduced. NSSL replaces EOS work with three pressure profiles and a LOG; Ferrier/Goddard already used the same geometric density path and now omit the moisture factor. Goddard also retains its height expression. [M CPU/compiler, isolated 3x2x3 helper] fp32/fp64 argument bytes 120/240, result bytes 160/304 including tuple overhead, temporary bytes 0; cost model 276/366 FLOPs and 18 transcendentals, optimized geometry HLO has zero observed fp64 arithmetic operations. These tiny CPU results do not establish GPU fusion, traffic, peak VRAM, launches or full-step readiness; those costs remain [U] pending changed-artifact GPU measurement. Final affected c12–c16/c62–c64 and v-run mp18/mp18_cu5 D6/coupled gates remain with the validation owners.

Process: RV25 was refused by the inner QUIET guard after RAM admission (zero tests). RV25b ran only after fresh manager RESUME. RV27 was discarded because the reviewer initially treated NSSL's trailing vertical axis as a leading axis; RV27b fixes the observer only and passes. The xunit2 classifier was corrected to inspect assertion messages/tracebacks rather than a nonexistent `type` attribute; no tests were repeated for that parser correction. Every heavy arm used the official 6 GB/no-swap RAM slot, CPU 24,25/nice19/JAX CPU, timeouts, disabled caches and bytecode. No GPU, producer edit or foreign reference edit. Three earlier sealed verdicts remain verified and unchanged.

Evidence is bound by `FINAL_SEAL.json`; `QUALIFICATION.json` checks every count, numeric mutant and full default digest. Reviewed snapshot and pytest scratch are removed after their last dependent arm. Completed 2026-10-15T21:51:49Z.
