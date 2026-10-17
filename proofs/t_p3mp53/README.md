# MP53 compiler qualification

P3 mp=53 (three-moment ice) timed out during the retained c24 GPU compile;
only frame zero existed and no numerical verdict was available.

The stock 44-level native kernel reproduces the TritonGPUCoalesce stall on
CPU with the captured SM120 target. Extending MP52's typed global loads and
atomic output exchanges clears Coalesce but exposes an invalid SSA error in
the first TritonGPURemoveLayoutConversions pass. A reused level predicate's
later category expand is moved below earlier uses. Two/four warps, a local
category-select rewrite, and replacing floating selects did not fix this.
Replacing only the 18 boolean singleton expansions with reshape did.

The retained source candidate applies typed memory and four warps only to
MP52/MP53, and enables boolean singleton reshape only inside the MP53 Pallas
kernel. All other contexts retain the previous JAX operations. No scientific
arithmetic, threshold, field, buffer or call boundary changes. No vendor or
shared compiler pass changes. The discarded category-select rewrite is absent.
The predicate ContextVar is reset even when tracing raises.

`cubin_probe.py` captures the actual current 44-level adapter inputs and all
21 outputs including held radar, runs the ordinary TTIR prepasses, and
compiles with the captured SM120 target on CPU. This isolated prescreen is
not a scan of an executable used by a forecast. The source candidate compiles
in 9.494 s after 2.692 s lowering; native STACK48, REG255, shared1024 bytes
(the Triton metadata reports shared256), STL11, no oversized nonleaf callees.

`interpret.py` compares the actual Pallas functions over three steps on the
frozen MP53 original-WRF operands, extended to 44 levels solely for shape
regression. All 66 field-frames / 20,136 values are byte-identical. This is
regression evidence; independent fidelity uses the ZERO-init original-WRF
24-hour c24 reference, retained unchanged, and the frozen D6 scorer/manifest.
The raw non-repeatable reference and its sensitivity disclosure are preserved
in the existing P3 qualification. No bounds or reference values are changed.

108 focused tests pass, including P3 fixtures, lookups, radar, transport,
restart and contracts. Deleting the actual predicate reshape branch causes
one assertion failure with zero setup errors. The first mutant invocation's
late cpu_guard import was a setup refusal, not a kill; the fixed invocation
imports cpu_guard before JAX. Both full release IDs are exact: Swiss f672a573
and PROD d01 6fc715a2. Later proof-only commits have identical src/data/package
trees to the validated candidate; numerical checks are reused with that proof.

The output exchanges, register pressure and spills remain performance costs.
This change adds no launches, global intermediate arrays, or host transfers
inside the timestep. The single manager-approved 24-hour c24 GPU arm completed all 25 frames in
431.920 s cold, with 5,376 MiB sampled process peak VRAM. Both actual JIT/AOT
executables passed complete E41 coverage; the actual MP53 kernel uses REG241,
STACK0 and STL0 (distinct from the isolated prescreen). Frozen D6 against the
sealed ZERO-init original-WRF reference passes all 25 frames with zero bounded
misses, nonfinite values or missing fields. The scorer and 9df4 tolerance
manifest are unchanged. All scientific outputs and exact paid executable
encodings are retained; caches, raw diagnostic IR and snapshots are removed.

The recorded core-species pre-mask census has zero nonfinite phase visits,
but 17,951,617 negative visits (15,623,747 interior). Their amplitude is not
attributed to original WRF, and QZI/QIR/BIR pre-mask coverage is not established.
Global F2 remains OPEN. This closes the MP53 compile/c24 qualification lane;
it does not claim warm S1 speedup, R1/R2 or global release-gate closure.
