# Pristine WRF 4.7.1 undefined local predicate

`phys/module_bl_mfshconvpbl.F:731` declares `GTEST_LOCAL_LCL2` without
initializing it. Line 935 assigns true only on a second-half LCL crossing;
line 997 reads it together with the first-half predicate. The corresponding
port predicate starts false, following the standing wave initialization policy.
Pristine source bodies are never edited.

The private oracle now explicitly uses `-finit-real=zero -finit-integer=0
-finit-logical=false`, with production-style floating-point traps disabled.
Q20 compared all fields from 15 regimes, REAL4/REAL8, and four heap-fill values:
120 executions are byte-identical to the original Q07 oracle. Q07 separately
compared `-finit-logical=false` and `true`; every reported output was identical.
The inactive path has dry fraction one, so its cloudy contribution has zero
weight. These observations establish initialization insensitivity for the
censused columns, not for every possible atmospheric state.

Receipts: lane artifacts `Q20/zero_identity.json`, `Q20/zero.json`, and
`Q07/logical_initialization_analysis.json`; source and compiler hashes are in
the packets. Full coupled forecast qualification remains pending.
