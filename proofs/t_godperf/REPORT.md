# Goddard radiation tile diagnosis

The cheap fix changes both Goddard column-tile limits from 128 to 2048.
On actual Swiss c45 initialized columns (1764 columns, 44 levels), the
2048 specialization produced all 16 outputs byte-identically to the old
128 specialization, for both night and noon solar geometry. The same initialized
thermodynamics were used at noon; this is not an h12 forecast state.

## Component measurements

Actual RTX 5090, official GPU lock, immutable source `187f7b1f7`, all four
variants compiled before execution. Times below are first execution after
explicit compilation, outside the Nsight capture window. They include graph
initialization and are **not** a warm forecast/S1 measurement.

| Tile cap | Night (s) [M] | Noon (s) [M] | Executable temps (MB) [M] | Compile (s) [M] | Exact outputs |
|---|---:|---:|---:|---:|---|
| 128 | 53.525 | 53.389 | 13.969 | 5.685 | baseline |
| 1024 | 7.743 | 7.694 | 60.359 | 6.071 | 16/16, both regimes |
| 2048 | 3.664 | 3.587 | 83.523 | 5.610 | 16/16, both regimes |
| 4096 | 3.648 | 3.614 | 83.523 | 1.364 | 16/16, both regimes |

The component first-execution reduction is 14.6–14.9×. Both 2048 and 4096
cover the 1764-column input without tiling: this experiment does not establish
4096-column performance. Choose 2048 to bound the working set on larger grids.
Additional measured temporary storage is 69.554 MB; argument/output storage
remains 6.569/0.720 MB. These are compiled component allocations, not process
peak VRAM or a whole-model memory gate. The 4096 compilation reused the same
untiled shape and is not an independent cold-compile improvement.

Every actual component executable passed E41: 195/199/179/179 static native
kernel entries, maximum STACK 0/0/8/8 B, no non-leaf or kInput failure.
Static kernel entries are not dynamic launches per radiation call.

## Cause and remaining work

1. **Serial tiny tiles multiply launch work.** The old cap repeats the full
   radiation recurrence fourteen times on the Swiss domain; the fix runs it
   once. The timings track that factor closely [M/I].
2. **LW nested level-pair and weighted loops remain expensive.** The actual
   paid c45 rich optimized HLO (`57b5…` JIT, `93f525…` program) contains 44
   LW-attributed while sites and 311 fusion sites across its mutually exclusive
   branches; SW has 7/60. In one LW branch the innermost level-pair body calls
   ten weighted-sum loops. Their known trip counts total 52 iterations, each
   with two fusions and one copy. With 10 bands, 990 level pairs and 14 tiles,
   that projects 21.6 million weighted-loop fusion/copy executions alone [I,
   static projection, not measured CUDA node count]. Several weighted results
   are calculated in every band before a selection discards them. A native
   column/band kernel, or carefully qualified specialization of these loops,
   is the next substantial optimization; it is outside this two-constant fix.
3. **fp64 is required here.** Pristine `module_ra_goddard.F:139` selects
   `SELECTED_REAL_KIND(15,307)` for the SW/LW solvers. The port's explicit DOUBLE
   islands match that choice. LW has 2401 f64-result compute instructions out
   of 4063 source-attributed compute instructions; SW has 640/1252 [M static
   instruction counts, not FLOP/time shares]. Converting these solvers to fp32
   would change the reference arithmetic.
4. **No E38 mega-fusion evidence.** The largest Goddard-attributed optimized
   fusion in the paid program has 307 HLO instructions, kind kLoop. Original
   whole-program E41 passed. This diagnosis does not establish a roofline bound.

## Profiler limitation

GP04 attempted node-level Nsight capture of full-domain calls after the
completed byte checks. The official 6 GB/no-swap scope was OOM-killed after
five profiled calls; no report was published. The kernel journal confirms
`run-p1690349-i1701823.scope` reached the cap. Preserve the completed benchmark
receipt and classify the trace as incomplete. Dynamic node counts, transfer
counts and device/host time attribution remain **unknown**. Manager disposition
on 2026-10-15: this is an optional diagnostic, not a lane gate; hand off now and
take no additional GPU slot. A future optional capture should cover one real
128-column tile and one untiled full-domain call. Do not repeat the completed
four-size byte checks.

## Fidelity and integration scope

The final Goddard module's AST equals the measured specialization after exactly
two integer-literal substitutions. LW, SW, helper and table bytes match both
the measured source and paid c45 source `90af92cfc`. No equations, precision,
cadence, history fields, initialization policy or oracle limits changed.

The existing pristine-reference/caller/history/restart/NetCDF checks passed
114/114 on final candidate `8741b8440`. The full Swiss default-program hash
remains `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD remains `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
The final actual c45 200-step/cadence-33 program lowers for CUDA on CPU in
44.81 s, with 155 native Pallas sites and zero interpreted sites. This is a
frontend admission, not a whole-program CUDA compilation or GPU execution.

Actual whole-model GPU E41, forecast performance and c45 D6 on the changed
program belong to v-prep's final composed run; the old paid program is not
transferred as a new-program qualification. The upstream compiler-zero
reference policy and experimental scheme disclosures remain in force.

Artifacts live under `<USER_HOME>/wrf_gpu2_lanes/t-godperf/`; committed receipts
contain sanitized metrics and hashes. Raw Nsight data is private and disposable.
