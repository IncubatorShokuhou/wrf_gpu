# W4/W5 precompile qualification

Status: **READY-FOR-CRITIC — W5 CPU subgate only**. W4 is **NOT QUALIFIED**;
actual GPU admission, byte comparisons and release gates remain open. No GPU
job or GPU waiter has been started by this lane.

W4 uses the E156 CPU-init/native-Triton recipe on immutable source snapshots.
Disk kernel caching and split-module shortcuts are excluded following PS06's
1,453,580 non-finite outputs despite matching SASS. Autotune level 0 is explicit
for the initial probe; its future cold GPU comparison must use the same options.

| Probe | Measured CPU result | Limit |
|---|---|---|
| W4P01 | Direct compiled-thunk export, 8,419 B, 0.457 s process | API feasibility only |
| W4P02 PROD d01 | Compile 152.103 s, process 370.895 s, peak RSS 6,415,300 KiB; executable 208,759,676 B | Artifact produced; later concrete-call pickle failed |
| W4P02 recovery | E41 PASS on retained rich AOT, one d01 program, no new model compile | CPU prescreen; actual GPU artifact gate still required |

W4P02 source `0fdf00f662`, input manifest
`<USER_HOME>/wrf_gpu2_lanes/w44-perf/prod_case/inputs.json`, sm120 target SHA
`b753abdddbace1e913de3767a1afbdfeba2b272e77f2be93658c766968e4960f`.
Work: PROD two nests, dt 54/18, d01 one root step, radiation cadence 33;
FAST1/CENSUS1/SLIM0/AOT_SLIM0/RRTMG_MP_RE1. No GPU execution. Private persistent
JAX cache disabled; temporary Triton/CUDA/init directories removed by the helper.
Artifact and full receipts:
`<USER_HOME>/wrf_gpu2_lanes/w44-precompile/prod_d01/`.

The helper failed after native export while pickling concrete arguments with a
JAX Device. Product AotMeta serialization succeeded. The next helper records
leaf hashes/placement instead; the existing native is reused for its scan.
The full compile timer is 152.103 s; XLA's 138.244 s diagnostic is one internal
phase and must not be substituted for the full compile wall.

CompileOnly's fingerprint uses `PJRT C API\nNVIDIA GeForce RTX 5090` for
`cuda_driver`; the released real GPU uses `PJRT C API\ncuda 13000`. An unchanged
CPU metadata record therefore cannot pass the product target comparison.
Any target attestation/re-export must be explicit and separately measured;
no fabricated matching fingerprint or relaxed loader is included here.

Still required: PROD d02 plus admitted WN3 d01/fused d02+d03 artifacts, actual
GPU target/load admission, one ordinary and one radiation root plus a real 3 h
segment BYTE versus a same-options cold GPU build, loaded-artifact E41,
readiness and runtime/memory costs. Manager scheduling/admission is pending.

W5 adds `gpuwrf precompile --namelist-variants FILE`: a backend-dark queue and
ordinary GPU priming fallback, with per-variant receipts and conservative cache
reuse. CPU artifacts are never installed into the shared runtime cache by W5.
User documentation: `docs/precompile.md`. W5 numerical work is the existing
`gpuwrf run` entry point; the only existing product file edited is its additive
CLI registration. Default model source/arithmetic are unchanged.

W5 focused tests plus existing CLI/prefetch consumers: **41 PASS, 13.40 s**
(CPU only). This includes process-group timeout/descendant cleanup and exact
lock-descriptor inheritance, cache corruption/missing metadata, changed inputs,
external runtime tables, and binding the selected namelist in the actual input
view. GPU execution remains unmeasured. Further source/queue/mutant receipts
will be appended after running the immutable composed candidate.

Completed CPU evidence:

- Default traced-source closure: **520 files BYTE-identical** to the pre-topo
  main baseline, digest `ce9a39f2b8c0f66e0b78a324c584e243ae4782709950113cfbb701c2514e201e`.
  Swiss `b977712f` / PROD `321aa651` default identities are carried forward on
  this exact-source proof; no duplicate numerical validation was performed.
- Actual PROD and admitted WN3 0227 queue preparation: **1.178 / 0.925 s**,
  peak process RSS **359,372 KiB**, **zero backend initialization attempts**,
  no executable cache created. WN3's eight input hashes match the sealed v0.4.3
  manifest. The queue report now includes a concrete `execute_argv` array.
- Four mutants killed through genuine assertions: cache integrity, selected
  namelist binding, official FD inheritance, external table identity. Initial
  plugin setup refusals were counted as zero kills; importing `cpu_guard`
  before JAX fixed the harness. Final report and environment-digest changes
  each passed their affected focused test. Raw environment values are not
  persisted in cache receipts.
- Strict fingerprint comparison against the sealed GPU target fails even
  after equalizing XLA_FLAGS in the diagnostic control. No target metadata or
  loader checks were changed to turn this into a pass.

Evidence: `receipts/W5_CPU_PACKET.json`, `W5_SOURCE_QUEUES.json`,
`W5_MUTANTS.json`, `ADMISSION_CPU.json`. The final product code is
`cdbe5098049074750821c3f5cbdef5fa7721994c`; the final delta affects only the new
tool's environment digest, outside the unchanged model source closure.

The default timestep has no added device operations or arrays by exact model
source identity. Added CLI registration/import cost was not isolated. Planning
wall/RSS above are measured CPU costs; GPU priming wall, VRAM, readiness and
RW1–RW4 are unmeasured and are required before runtime/release qualification.

A sealed **two-minute admission-only night launcher** is ready:
`<USER_HOME>/wrf_gpu2_lanes/w44-precompile/admission_tools/proofs/w44_precompile/night_admission.sh`,
SHA256 `56f74db15a10c3d7a7f7281da2676d3ebb245f3b9c8fb54069d178df50be691b`.
It binds exact raw native/meta bytes and source `0fdf00f662`, takes a 6G RAM
slot before the official GPU lock, and calls the ordinary strict loader.
It never compiles a fallback, relabels metadata or advances a model step.
Five CPU dry checks passed (valid window, before window, insufficient time,
missing window, unknown argument). This precheck cannot establish scientific
qualification even if a load succeeds. The manager must schedule it with a
real UTC window; the system clock offset is still material.

Remaining CPU domain artifacts are being prepared independently of that
admission check. Completing them does not waive the target or GPU byte gates.
