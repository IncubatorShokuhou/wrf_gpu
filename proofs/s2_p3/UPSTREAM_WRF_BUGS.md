P3 pristine compiler-initialization audit

P348 builds the byte-identical original `module_mp_p3.F` with production FP
traps off and compiler-defined zero REAL/integer/logical initialization. The
external driver reloads the exact original frame0 operands before each arm.
All 140 saved fields across four variants and three calls are BYTE to the
original P305 reference. A second build uses REAL SNAN, integer42 and logical
true initialization; all 140 fields remain BYTE with no nonfinite output.

No material uninitialized-output dependency was observed in this corpus. This
does not claim every possible P3 regime is free of upstream uninitialized
scratch reads. No pristine computational statement was changed.

Compiler contraction has a material effect even on identical operands; its
independent measured spread and the source-derived radius discontinuity bound
are documented separately in `COMPILED_BOUND.md`. These are numerical
conditioning effects, not initialization defects.
