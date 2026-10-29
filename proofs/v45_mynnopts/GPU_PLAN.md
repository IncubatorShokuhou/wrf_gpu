# MYNN options: preregistered GPU work (not yet booked)

CPU-only development. Manager admission, valid v45-mynnopts lease, official GPU lock,
ALISIOS window and final immutable source pin are prerequisites. No native claim from
the CPU oracle. Do not run a forecast on the live worker checkout.

1. Per option (icloud_bl=0, bl_mynn_edmf=0, cloudpdf=0, cloudpdf=1), compare final
   default against selected program, plus a repeated default A/A. Use separate cache
   directories per arm (E237), release flags and 44-level real PROD day/night active
   columns. Assert backend=='gpu' in every measured process. Oracle reference is
   pristine CPU-WRF, unchanged existing tolerances. Record finite values, activity,
   compile/readiness, compiled memory and matched warm component latency.
2. Full PROD nested root: matched OFF/ON, finite carry/source/history; default
   constant-aware native-first identity E222/E224. Explicit NVTX3 labels for nsys;
   count launches, allocations and transfers, retain hashes before deleting raw traces.
   CPU/JAX comparisons are regression evidence only. No new loop host transfer allowed.
3. Run E41 on each actual compiled executable after cold setup; parse detailed verdict,
   never treat INCOMPLETE/rc2 as PASS. Any STACK>256B, nonleaf>1000 SASS, kInput>500
   optimized HLO instructions fails. No gate changes.
4. Manager's final non-identical composition runs affected RW1–RW4, including original
   CPU-WRF WN3 0227 72h/D6 and prior-release timing/memory; >10% regression STOP.

Each PDF child <=9min (compile, oracle, timing and E41); one two-child entry <=20min.
Cold setup <=60min and native/profile <=30min,
always GNU timeout. Only manager-scheduled long arms may run full cases. Proposed
scratch total <=3GiB on ROOT, no DATA writes except source. Full-case cache/output
needs a separately approved disk plan with measured size and current shared floor.
Build one exact-call night_seq_v9/v10 launcher per admitted plan, smoke it THROUGH
that sequencer (same argv/caps/env, CPU/private flock only) before manager booking.
The full-root launchers remain pending. The component CPU SEQ_SMOKE receipt
passes; it does not establish GPU readiness or native qualification.

The bounded first-entry operator launcher is `night_operators.sh operators`, with
MO_SOURCE_ROOT pointing at a source-pinned archive and MO_OUTPUT_ROOT at the owned
lane directory. Its body and arguments are identical in CPU smoke and GPU mode,
apart from backend/private-flock substitution and native-only export/E41. It compares
both PDFs on all384 frozen WN3 night/Swiss day columns in separate child processes,
then records20 warm calls and compiled memory. It is component-only; all coupled
and full-model native-cost/release gates remain pending. It is not a full-root launcher.
Its environment is an explicit safe allowlist, and it uses the canonical official
lock wrapper in the main checkout (an extracted source archive has no Git metadata
for the wrapper's self-verification). The final plan records launcher SHA/source pin.

F2 cache/export contract: each invocation creates a unique
`MO_OUTPUT_ROOT/cache_runs/<backend>_<UTC>_<pid>/pdf0` and `pdf1`. The launcher
derives and explicitly allows all six private controls in each child:
`JAX_COMPILATION_CACHE_DIR` and `GPUWRF_JAX_CACHE_DIR` → `jax/`,
`TRITON_CACHE_DIR` → `triton/`, `CUDA_CACHE_PATH` → `cuda/`,
`XDG_CACHE_HOME` → `xdg/`, `TMPDIR` → `tmp/`. Incoming unrelated cache paths
are replaced by these owned per-PDF paths. A Python contract checks all six
before initializing the backend and sets JAX's actual cache configuration.
CPU smoke disables persistent caching; native mode enables it. JAX persistence
and CUDA cache limits are each64MiB per PDF. The aggregate checked hard cap is
384MiB per PDF / 768MiB per run, including native/export/temp evidence. These
are planned native caps; no native allocation measurement is claimed. Together
with the <=1.2GiB slim source and <=0.2GiB test scratch this remains below3GiB.

Immediately after the actual native `.lower(...).compile()`, the measured
`Compiled` object is serialized with the project AOT serializer. Rich trace
metadata/full optimized HLO are required (`GPUWRF_AOT_KEEP_TRACE_METADATA=1`).
Each arm retains `native/operator.xlaexec`, its metadata and SHA. The project
`wave_cubin_gate.scan_executable` scans that exact blob as a component, without
mislabeling it as a domain step. The same STACK256/nonleaf1000/kInput500 rules
apply; missing native coverage or UNKNOWN/INCOMPLETE never passes. The component
report records the actual E41 result. CPU smoke reports native export/E41 N/A.
After the arm, only the five exact cache/temp directories are removed; native
executable and E41 receipts remain for the last review reader. Clean those
bounded retained artifacts only after that reader, preserving small SHA receipts.
No full-model qualification is inferred from a component E41 pass.
