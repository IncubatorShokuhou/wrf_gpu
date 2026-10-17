# Frozen-column emission consumer repair

The same critic's FIX verdict identified a caller mismatch: L3b recomputed
startup emissions on a constructed column, then refused because the corrected
startup helper produced 62146.65625 instead of the frozen driver's
53290.4140625. This column solve must use the original driver's held inputs.

The runner now seeds NWFA2D/NIFA2D from the hash-bound frozen inputs, verifies
their exact values at the actual adapter emission call, and verifies their
preservation in the returned State. Its original numerical verdict, bounds,
manifests, WRF files and NPZs remain unchanged. The independent real-file
frame-zero startup test is unchanged.

The fp64 setup check also follows the existing operational WRF REAL radius
contract: the three radius diagnostics must be float32, while the registered
seven-field fp64 input vector remains exact. Those diagnostics do not feed the
column solve. The initial setup refusal and missing pytest-parent error are
retained as failed harness receipts.

Six production and six fp64 L3b arms pass, including every original registered
discriminating mutant, at unchanged bounds. A focused fp64 caller check uses
the stronger exact-value observer. Deleting the real held-input assignment
must fail at the adapter consumption assertion with zero setup errors.

`E199_QUALIFICATION.json` binds final results and raw receipts;
`L3B_FROZEN_INPUTS.json` seals twenty original files. `E199_SOURCE_REUSE.json`
proves eleven selected product functions, the aerosol kernel, frame-zero test,
original fixture and parent qualification are unchanged from the reviewed
repair. The final observer change removes only two downcasts: all column
computations, inputs and verdicts are AST-identical to the twelve measured
arms, so their numerical evidence is reused.

These changes affect the test caller only. Product stepping, memory, restart
schema and default program are unchanged. CPU oracle costs are diagnostic
costs, not forecast performance. GPU qualification, the separate 3-D aerosol
initialization issue and existing uncomputed-field disclosures remain open.
