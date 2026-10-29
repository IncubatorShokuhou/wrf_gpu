# D1 root cause and scope

The original GPU receipt reports 101 unequal ON-default versus OFF leaves
(26 parent, 75 child), including U/V/W, theta, qv and pressure. It is historical
regression evidence, not a CPU-WRF fidelity reference. The original receipt is
`<DATA_ROOT>/wrf_gpu2/scratch/w44-params/D1_EVAL/GATE.json`.

Two defect mechanisms were measured on the required RC base:

- **Scalar folding:** before the field solve, `_dampmag(18, .2)` is
  `0x40666666` with the released Python scalar fold and `0x40666667` with a traced
  REAL coefficient. The same one-ULP seam occurs at dts 9 and 4.5. Host-binding
  the two RK stage strengths preserves the selected fold without timestep host
  work. A deleted-precomputed-operand mutant fails the numeric assertion;
  pristine WRF `advance_w` bounds and changed-value response pass.
- **Fusion topology:** the vertical-CFL three-control subset alone changes XLA's
  fusion choices and reproduces STACK 288 B. c_s alone and the diffusion6 plus
  Rayleigh subset each pass at 32 B. The failing native
  `input_transpose_fusion_99` maps to V advection plus pressure gradient,
  Coriolis and curvature: 412 HLO instructions, 49 parameters, 18 transposes.
  Its three scalar parameters are pressure-gradient coefficients, not the
  runtime controls. Original-protobuf caller provenance is retained in the
  private DL01 attribution receipt; printed remapped StackFrames were not used.

The output-only V pressure-gradient barrier removes 107 instructions and 15
parameters from that fusion; STACK falls to 32 B. The source-only counterfactual
without the seven-line guard/comment/barrier addition remains a 288 B failure.
Final parent and child native scans pass all preregistered stack/callee/HLO
limits. This is a CPU CompileOnly prescreen, not actual GPU qualification.

Final source `7c83a67469` passes the complete CPU first nested root: all 355
carried leaves are byte-identical OFF versus ON-default; all three arms are
finite. Doubling c_s changes 101 leaves, keeps both domain keys, and leaves the
two-entry concrete JIT cache unchanged. Parent, force-down and child OFF
full-constant HLO plus located/unlocated native kernel IR are exactly identical
to the RC baseline (22 comparator checks; canonical jaxpr was not rerun).

These results establish the CPU lane gate and identify demonstrated scalar and
compiler mechanisms. Quantifying how each mechanism contributes to the original
101 GPU leaf differences, and proving default byte preservation for the new
GPU-compiled program, remain for the preregistered actual GPU A/B. The product
family remains refused; no default, fidelity tolerance or release gate changed.
