# Ferrier c62 CUDA lowering fix

The real c62 failure is in Noah-MP `groundwater_step`, at `exp(-timean)`.
TIMEAN is a static REAL table value. Installed Triton `neg_p` passes its
`TypedNdArray` literal straight to `_minus`, which expects an IR value. Ferrier
itself lowers successfully. The preceding float64 truncation warning persists
after the fix and is not the cause of this exception.

The two production edits replace `-timean` by REAL32 `-0 - timean` in the
groundwater and equilibrium-water paths. Under round-to-nearest this is exact
negation, including both signs of zero. No field, table, parameter, tolerance,
physics equation, caller, carry, restart schema or Ferrier DOUBLE island changes.

CPU-only evidence (never GPU execution or forecast qualification):

- Original real c62: lowering fails in `neg_p`; 34.887 s, 2,489,896 KiB RSS.
- One-operation diagnostic source clone: real c62 lowers, 149 Triton calls.
- Fixed real c62: PASS, 149 calls, 49.511 s, 2,595,784 KiB RSS.
- Fixed real c63 (mp15 / opt_run2): PASS, 150 calls, 49.899 s, 2,585,828 KiB RSS.
- Admitted Ferrier mp5/15: native CUDA lowers, REAL32 outputs and TTIR DOUBLE
  saturation islands checked. mp85/95 remain unadmitted.
- Original-WRF literal byte gate: 1,872/1,872 finite values, five groundwater /
  equilibrium / prognostic-water arms. Compiled old/new outputs also byte equal
  for all 1,872 values. Existing documented compiled deviations remain unchanged.
- Positive component tests: 13 passed across FF06/FF07, excluding the two
  superseded FF06 interpreter-only instrumentation failures.
- Deleting both fixes: two genuine CUDA lowering failures, zero setup errors.
  FF09's earlier CPU-guard setup refusal is preserved and is not a mutant kill.
- Required contract / legacy-MP refusal / NetCDF-lock consumers: 37 passed.
  Total positive tests: 50. Swiss `f672a573` and PROD `6fc715a2` identities exact.

`QUALIFICATION.json` seals the receipts, final production source and the two
official default-program identities. Experiments FF01–FF16 are in MAIN's ledger.
Logs/XML use lossless gzip with compressed and original byte hashes recorded.
The pre-rebase measurement history is retained at `lane/t-ferrier-pre-rebase`.

Resource assessment: one existing scalar operation is replaced per water path;
there are no new State arrays, intermediates at domain size, Pallas calls, or
host/device transfers. The static scalar is folded inside the existing column
kernel [I]. Successful real CUDA-target lowering remains below 2.6 GiB RSS [M].
GPU timing, peak VRAM, executable E41 and full CPU-WRF forecast D6 remain for
release validation; no runtime speedup or numerical forecast claim is made.

Reproduce with `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, the lane core/memory/QUIET
rules, and a detached snapshot:

```text
python proofs/t_ferrier/lower_case.py SOURCE_ROOT OUTPUT_JSON [--input CASE/input]
python proofs/t_ferrier/check_precision.py SOURCE_ROOT OUTPUT_JSON
python proofs/t_ferrier/mutant.py SOURCE_ROOT OUTPUT_XML
python proofs/t_ferrier/release_identity.py SOURCE_ROOT OUTPUT_JSON [CASE]
```

The optional `lower_case.py --groundwater-control` is the historical diagnostic
for an unpatched source tree (FF05), not the fixed-tree measurement path.
