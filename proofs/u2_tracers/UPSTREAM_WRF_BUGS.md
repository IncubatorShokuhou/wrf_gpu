# Tracer PBL source and compiler audit

This record concerns only pristine `phys/module_pbl_driver.F` routines `diff4d`,
`diff`, and `invert`, extracted without changing their bodies. It establishes
column mixing evidence, not coupled forecast fidelity.

The literal default reference uses REAL32, `-O0 -ffp-contract=off`, bounds checks,
`-finit-real=zero -finit-integer=0 -finit-logical=false`, and no FP traps.
Its six fixtures are byte-identical to the original snan-initialized fixture:
SHA256 `6501902034c3ae51e90c408f6946a7cac09288bb47006ba9f93d18e8cfb561ab`.
A fresh snan sensitivity arm also reproduces all 3,360 output values byte for
byte, with all values finite and caller inputs byte-identical.

No uninitialized read was found in this extracted path at the tested bounds
(`kts=1`, `kte=1,2,8`). `diff4d` initializes the complete heat/momentum exchange,
density, face density, surface/volume, and implicit/explicit source work arrays
before filling active levels. It fills scalar and thickness levels before the
solve. `diff` assigns every used `cddz` face and every active matrix/RHS row;
`invert` reads only those assigned rows. Its diagnostic flux output is assigned
over all active levels. This audit does not cover other callers or nonunit lower
vertical bounds: `diff` literally sets `iz1=1`, irrespective of `kts`.

The original source forces bottom heat exchange to zero and pins the top mass
level by setting its matrix row to identity. These are its actual boundary
conditions and must be preserved. They are not a symmetric no-flux boundary.

The independently compiled original FMA reference uses
`-O2 -ffp-contract=fast -march=native -mfma` with the same bounds checks, zero
initialization, exact inputs, and unchanged routine bodies. Its SHA256 is
`7478897fea775bc677860d64239f17dbaf41922f8252d45a8549c1cb5faba864`.
Compared with the literal original, its native compiler spread is:

| Case | Different values / total | Maximum absolute tendency spread (s^-1) | Values exceeding fixed literal bound |
| --- | ---: | ---: | ---: |
| heterogeneous | 152 / 768 | 1.7695128917694092e-8 | 0 |
| strong_exchange | 296 / 768 | 1.5925616025924683e-7 | 99 |
| zero_exchange | 0 / 768 | 0 | 0 |
| constant_profile | 107 / 768 | 1.7660635620586618e-8 | 0 |
| one_level | 0 / 96 | 0 | 0 |
| two_levels | 2 / 192 | 3.3760443329811096e-9 | 0 |

The fixed literal bound remains
`16*eps32*(abs(q)/DTBL + abs(literal tendency))`. The largest original-WRF
FMA/literal spread is 5.684880813 times that bound, in strong exchange.
This is original-WRF compiler sensitivity, not evidence of an upstream physics
bug. The prior native JAX diagnostic has identical case difference counts and
maximum errors; that observation alone does not establish JAX/FMA byte identity.
The FMA fixture supports an independent compiled-path comparison without changing
the literal gate or setting global product compiler flags. Coupled D6 and the
existing coupled `1e-4` gate remain unchanged and decide forecast acceptance.

Source SHA256:
`c1c083365a4e780381fce09b97231cd322913590f089ac91aa9bfc1f6be54424`.
Extracted routine SHA256:
`02e78b5181047611bcc87eae949a9c8fa2ce1359bca5e6fa03667d5181c47fe5`.
Common caller-input SHA256 (REAL32 operands, field identities, and shapes):
`b7ffb47cc817c11267d6b1eaf796208bbc91b2247c8cdcb9e5230e4fa37d0183`.
Both manifests record compiler identity, flags, exact source/input provenance,
and regeneration commands. Lane receipts are `TR-PBL05_literal_zero.json`,
`TR-PBL06_sensitivity.json`, and `TR-PBL07_fma_spread.json` under
`<USER_HOME>/wrf_gpu2_lanes/u2-tracers/`.
