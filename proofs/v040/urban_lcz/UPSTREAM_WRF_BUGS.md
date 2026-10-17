# NUDAPT uninitialized direction inputs

See [the source audit and measured sensitivity](../../../docs/UPSTREAM_WRF_BUGS.md).
Pristine bodies are unmodified. Compiler zero/false initialization at caller
entry gives LF(1) for the **one-cell** reference; it does not reset reused
U10/V10 between tile cells. The port's fixed LF(1) is a deterministic
experimental convention, not whole-tile compiler-zero parity.

The [sealed two-cell diagnostic](evidence/multicell/README.md) measures
HFX 6.273636 W/m2, U10 0.992088 m/s and TSK 0.301880 K differences with identical
array inputs/state. First-cell frozen fields are BYTE; isotropic LF control
has zero differences. Original four column results/tolerances are unchanged.
Coupled original CPU-WRF/D6 qualification is pending; LCZ without morphology
is unaffected. Zero-vs-infinity initialization sensitivity also remains material.
