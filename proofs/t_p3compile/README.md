MP52 compiler qualification
==========================

MP52 is P3 with two ice categories. MP53 is the three-moment variant.

The captured MP52 kernel stalls in Triton 3.6's TritonGPUCoalesce pass.
The last entered memory operation is the first 64-element column load.
A bounded C++ stack sample reaches `buildCoalescedEncoding` through
`mlir::getSlice` and repeated `getForwardSliceImpl`; the expanded level-local
process graph is already sufficient to reproduce the stall without a
sedimentation loop. Moving the category loop to runtime still times out.

The fix is scoped to MP52's Pallas memory operations. Typed global load
instructions avoid that pass's whole-graph tensor-pointer analysis. Final
vector outputs use atomic exchange because sliced layouts can replicate
lanes; the exchanged old bits are unused. Uniform scalar loads and scalar
precipitation stores retain stock Triton lowering. The load's masked fill
is explicitly converted to the reference's REAL4 dtype before assembly,
including under an x64 trace. No shared compiler pass or vendor installation
is changed. P3 arithmetic, lookup formulas, buffers and call boundaries stay
unchanged. Four warps reduce the measured native stack pressure.

`interpret.py` compares all outputs over three steps against the unchanged
CPU Pallas implementation. This is byte regression evidence, not WRF
fidelity. `cubin_probe.py` lowers the actual operational 44-level input and
output ABI, compiles it on CPU with the captured SM120 target, checks typed
memory operands, and scans native resources. This prescreen does not replace
the scan of the executable compiled and used by the actual GPU forecast.

`gpu_case.py` runs a private c23 input copy under the official GPU lock.
`score_zero.py` scores those outputs against the already qualified 25-frame
ZERO-init original-WRF reference using the unchanged frozen D6 scorer and
manifest. Missing variables and nonfinite values remain explicit failures
of coverage or fidelity; no bounds are changed.

Performance costs to disclose are the output exchanges, four-warps launch
geometry, register use, stack spills, shared scratch and cold readiness.
The change adds no kernel launches, global intermediate arrays or in-step
host transfers. Forecast wall time is cold qualification under shared CPU
and IO, not a warm S1 benchmark. The original non-repeatable WRF reference
and its uncertainty disclosure remain in `proofs/t_p3fix/QUALIFICATION.json`.

Final measured qualification (`QUALIFICATION.json`): CPU interpret has 84
byte-identical field-frames / 26,472 values. The single-column CPU native
prescreen compiles in 63.99 s (112-byte stack). The actual c23 GPU run writes
all 25 frames in 480.17 s cold, with 5,374 MiB sampled process VRAM; its real
MP52 kernel uses REG255, STACK168 and 1,024 bytes shared scratch. Both paid
JIT and AOT executable entries pass the complete E41 gate. All 62 frozen
bounded fields pass D6 over 24 h with zero misses and nonfinite values.
Full inventory coverage remains incomplete only for missing `REFL_10CM`.
The real native fill-cast deletion is rejected (one assertion, zero setup
errors); 34 focused controls pass. F2 negative phase visits remain open and
are preserved verbatim in the sealed receipt. Both full default program
IDs remain exact. This closes the compile-fix gate, not global release gates.
