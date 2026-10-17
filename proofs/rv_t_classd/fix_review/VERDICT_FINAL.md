# rv-t-classd final verdict — MERGE

Same-critic fix-delta review of `065409847c` versus the previously reviewed `165de18b3`.
Source tree `src=58ef3e58c9539049d7689ef381c71e75c6717734`; `src/gpuwrf=10917379e6aba91aeefb6acac80c8cd543469d09`.
The original FIX report and its SHA-sealed evidence remain unchanged. This verdict supersedes its four required corrections only.

| Finding | Final disposition | Independent evidence |
|---|---|---|
| F1 native QGR restart | CLOSED | Healthy full carry round-trips byte-for-byte. Missing QGR, `(1,1)` shape, float64 storage, deleted whole urban group, absent configuration and wrong bound domain are refused at public save and hash-authenticated read. Production `nested_identity` emits the expected BEP/BEM option/domain binding. `RV14_native.json`; additional affected tests in RV13. |
| F2 Morrison baked constants | CLOSED | Changing the actual snow coefficient changes actual reflectivity/HLO and now changes non-null AOT keys for mp10 and mp40, including a child namelist captured by a fused-call closure. The tests also confirm stale-blob lookup misses and radar-off keys remain unchanged. `RV15_keys.json`; RV13 key regressions. NSSL REAL-log split constants are keyed too. |
| F3 REAL logarithm | CLOSED at the assigned CPU/source gate | Both one-moment variants retain frozen pristine S6 REAL4 BYTE parity on 680 levels each; REAL8 parity remains within its original bound. Full radar lowering retains legitimate DOUBLE islands and has no f64 log. Isolated new REAL logarithm lowering contains no f64 at all. `RV13_tests.xml`, `RV16_real_log.json`. |
| F4 alias/provenance incident | CLOSED | Independently ran the new staging helper: three inputs are private regular copies, all 83 table links resolve inside pristine WRF, an attempted symlinked write is refused, and every foreign input stays at its sealed hash. Retained c08r2 inputs match the measured/original hashes separately; all 25 output hashes equal the prior run; the private binary has only the two declared pathname patches. `RV17_staging.json`. Original incident evidence is retained; history-residual attribution is now explicitly a hypothesis. |

## Validation and reuse

[M CPU] **194 affected tests pass, zero failures/errors/skips**, in 95.64 s. Selection: NSSL one-moment radar, green-roof history/native checkpoints, Morrison radar and AOT keys, shared online native restart and SSiB binding. Frozen assertions/fixtures were not loosened or edited.

RV18 independently removes the new checkpoint validation and substitutes the inaccurate plain REAL logarithm: **one checkpoint assertion failure and two numerical-parity assertion failures, zero setup errors**. A hostile alias write is also refused before changing its protected target.

RV19 proves that only three model files changed: NSSL diagnostics, checkpoint admission and AOT-key accounting. Every pre-existing NSSL diagnostic function except the one-moment radar's final log expression is AST-identical, and that radar's complete pre-log calculation is AST-identical. All other model files and radar oracle fixtures are byte-identical. The original independent default hashes and pristine Morrison rebuild/CPU metadata/reference evidence therefore carry forward. The unchanged probe's successor receipts on `05c1b1775` (identical source tree to `065409847`) confirm full Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` and PROD `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.

## Performance, memory and limits

F1 and F2 add host-side checkpoint checks and metadata-key hashing; they add no timestep arithmetic or device arrays. The BEM carry and history cadence are unchanged from the first review. The new logarithm is vector arithmetic with host-generated split constants; its isolated CPU compiled artifact has arguments/output/temporary buffers of 40,808 bytes each for 10,202 values, with no f64 operations. Its CPU compiler cost model counts 190 floating operations per value. These are CPU/compiler diagnostics, not GPU launch, runtime or VRAM measurements. GPU fusion/contraction behavior, E41, startup/readiness and changed-artifact runtime still require the existing opt-in qualification before release.

An additional independent glibc `log10f` sensitivity sample gives 10,201/10,202 exact results and maximum one REAL ULP error; this is not a universal correct-rounding claim or a replacement for the unchanged frozen S6 BYTE gate. No gate was widened. The existing c08 reconstructed-history residual up to 15.45 dBZ remains an attribution hypothesis, not a coupled-trajectory pass. The disclosed mp40 MSKF defined-zero treatment of upstream out-of-bounds cumulus arrays remains outside an active original-CPU composite qualification. The separate NSSL/Ferrier/Goddard density finding is explicitly outside this verdict.

The earlier broad operational-smoke run interrupted by signal 143 remains incomplete and is not counted as passing. Earlier producer-wide receipts remain producer evidence. The critic ran no forecast or GPU jobs and grants no roofline or full-release clearance.

**MERGE `065409847c` for the reviewed Class-D scope. No remaining required fix from F1–F4.** The manager owns integration and the affected GPU/coupled validation; the separate density branch is not included.
