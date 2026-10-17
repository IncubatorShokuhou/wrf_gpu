# rv-t-classd composition verdict — MERGE

Reviewed final composed successor `54cee8bd8`, source tree `src=e62b4e3bde56b64d10ad05db3f3a08492c28f169`, `src/gpuwrf=f65c2869f35c0b07097854a114c83bd337e9de1b`.
Manager scope: **only** the composition delta and both-active Morrison caller test. The earlier Class-D and calleraudit F1 scientific reviews are reused; this does not reopen or expand them.

The source merge `4c910db58560a6e37d1e663ab554de7b13163268` joins reviewed Class-D `065409847c92f382dd76e53aed22774f165e76b8` with F1 integration `947ea8d747496ccd5f593245c894aa462b68f9fb`. The immutable measured snapshot was `601842b9b9c0d8fca81157cefa9c854f3d01386a`; final `54cee8bd8` has byte-identical model source and both scoped caller test files, so no execution is repeated. The later edits consist of receipts/notes and two NSSL caller mocks; those unrelated mock edits are outside this narrow verdict and are droppable per the author.

**Composition source passes.** Both `morrison_adapter` and `morrison_aero_adapter` accept independent precipitation/radar flags and produce `State[, precipitation][, radar context]`. The operational both-active branch requests both and receives a triple; the former `precip=None` is removed. Each adapter contains exactly one underlying Morrison tendency call. The numerical pre-return calculation and radar-context builder are AST-identical to Class-D; `_mp_public_precipitation` and the tuple-preserving `_fp64_island` are AST-identical to accepted F1. Both channels come from that same tendency object. Evidence: `PRECHECK.json`, `RV23_source.json`.

**RV21: 10/10 scoped CPU checks PASS**, zero failures/errors/skips:

- mp10 and mp40, REAL carry, two compiled operational dispatcher steps; one adapter invocation per step with both flags, including mp10's fp64 compatibility wrapper.
- Positive total precipitation and cold-column snow; the next land call receives the previous step's held precipitation. Radar stays held before the alarm and updates from the same MP-exit context at the alarm.
- Radar on/off yields byte-identical prognostic State and land precipitation.
- Legacy and REAL adapter callers preserve State and individual side outputs for plain, precipitation-only, radar-only and both-active selections.
- The two existing operational caller cases now compare both-active precipitation with the radar-off precipitation, replacing their old `precip=None` expectation.

The new test is a compiled **caller seam**, with RK and the Noah-MP numerical solver replaced; the solver stub records the actual precipitation argument. It does not claim forecast fidelity. Reusing the already qualified conversion/radar functions for expected caller outputs is wiring regression evidence; the physical operator evidence remains in the two prior reviews.

**RV22: one genuine assertion failure, zero setup errors.** Erasing only the both-active precipitation values, while retaining valid State, radar and precipitation schema, fails the positive-total-precipitation assertion at line 131 of the new caller test. The original test passes in RV21. The queue crossed host hibernation; process completion and the XML assertion signature were checked after resume. This is not a performance measurement.

**Efficiency and remaining gates.** The composition does not add a second Morrison evaluation, change its numerical body, or add a prognostic State field. It returns the required existing per-call precipitation amounts alongside the existing MP-exit radar context. The compiled seam verifies one adapter invocation per step; this is not a GPU kernel-launch measurement. Actual GPU fusion/temporary buffers, E41 and coupled-forecast qualification remain for the final composed class-A candidate, once, under the manager's rerun policy. No GPU or deviceless GPU compile was run. CPU durations are test diagnostics, not performance evidence, including during the hibernation interval.

**MERGE the reviewed composition in `54cee8bd8`.** Prior verdicts/seals remain immutable. No full numerical revalidation, default-program claim, gate/tolerance change or separate density-branch inclusion is made by this review. Approval covers the composed model and both-active Morrison caller tests; the unrelated NSSL mock edits remain explicitly outside scope.
