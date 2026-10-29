# Diagnostic qualification plan (CPU-first; no GPU self-launch)

Current candidate: out-of-loop read-only history consumer, manager ACK
2026-10-27T23:11:41Z. No forecast/runtime/compile/IO/nesting source changes.
Model-program source identity carries existing CPU/GPU programs by construction;
there is no changed GPU compiled artifact to time, scan or requalify. Do not
spend a cold compile to validate a Python JSON consumer. Existing opt-in
producer limits remain binding: radiation history-only scope, MYNN E242 FAIL.

Current CPU arms: frozen archive; known-answer/QC/source-byte checks; transport
all 21 existing pristine-WRF MYNN terms and local residuals through the real
NetCDF consumer; source-tree identity; deletion mutants; full root collection.
Caps 180 s each, 8 GiB/no swap, 12/28, nice19, JAX_PLATFORMS=cpu, QUIET guards,
GPUWRF_JAX_CACHE=0. Private archive/pytest scratch under lane root, remove after
the last arm; raw projected NetCDF/JSON deleted by the reference arm immediately.

Future separately gated producer variant (estimates in docs/optimizer_profiles.md):

1. Freeze one composed candidate and a same-source OFF control. Use fresh
   CPU force-down operands for WN3 0227 d01 and fused d02/d03. An E156-faithful
   deviceless replica must use native Triton lowering, the captured target and
   actual flags/autotune conditions; compare physical fusions/FMA status. CPU
   byte equality alone cannot pass. Coordinate RAM >8 GiB through the manager.
2. Only after manager scheduling/lease, provide an exact night_seq_v10 launcher
   and sequencer CPU smoke. GPU command must be
   `scripts/with_gpu_lock.sh --label v45-diag -- timeout ...`; CUDA assertion
   and platform receipt required. No GPU child of a parent CUDA-pool owner.
   Secret-looking env is stripped by the sequencer; no nsys environment output.
3. Separate cold-setup arm ≤3600 s, private OFF/ON JAX/Triton/CUDA caches and
   explicit resolved namelist/cadences. Scan the actual cached executables:
   STACK≤256 B, nonleaf≤1000 SASS, kInput≤500 optimized HLO. Any violation FAIL.
   Do not use retained research executable caches as a new exact-source proof.
4. Warm identity arm ≤1800 s, abort on first mismatch: all carried prognostic,
   physics/scratch/land-history leaves, including hidden leaves, by dtype,
   shape and bytes at steps 1/2, then WN3 three-nest 3 h histories. Compare ON
   with OFF as regression identity, not WRF fidelity. If mismatch, reject the
   observer; do not interpret its new diagnostic rates as production rates.
5. Original CPU-WRF term checks on matched exact operands/call times, including
   active day/night plumes, followed by paired resource timing/profile arms
   ≤1800 s. Report nsys kernels/copies/round trips, bench s/fc-h, live/reserved
   VRAM and host RSS with A/A noise; no performance estimate from arithmetic
   counts. Broader RW gates belong to the manager's final composition.

Pre-register a final disk estimate before booking: one source archive plus
private caches and bounded output, respecting the **current** ALISIOS floor.
No large captures or reference deletion are authorized by this plan. Retain
small SHA-bound receipts, remove only exact owned disposable paths after all
dependent arms and approval-bound reference checks.
