# v05-hcb — CPU diagnosis and compile-once runtime partition design

**The original GPU native-difference cause remains OPEN. No product source or defaults changed.**

| Contract gate | Result |
|---|---|
| G1: original C2b first differing function and responsible pass | **OPEN.** Original binaries were deleted before this lane. CPU min5/min1 difference is refuted, but that replay omits a GPU-only pass. |
| G2: native-identical one-WHILE-graph design and preregistration | CPU design/proof ready; actual GPU load, capture, byte identity and speed unmeasured. |
| G3: no product default change | PASS; only lane proof tools, receipts and notes. |

Two independent fresh CPU builds used the exact failed source `700a382f8b`, real PROD force-down operands, 151 native Pallas calls, captured sm120 target, normal autotune level 4, require-complete, and a private copy of the 1,227 retained tuning files. Both lowered HLO keys are the original C2b d02 key `f6bf7a5e6fbd97f9ee702f5cbb631e7934c039c6bf7463f98f6640de79ccb8b6`. CPU compile parallelism was 1, rather than the GPU arm's 8. This is a controlled CPU refutation, not full GPU pipeline equivalence.

| Native record | CPU min5 = min1 | Original GPU min5 | Original GPU min1 |
|---|---|---|---|
| 0 | `c383bcdb…` | `c383bcdb…` | `c383bcdb…` |
| 1 (PTX) | `a7131336…` | `8ab1f0a4…` | `c5b95e82…` |
| 2 (ELF) | `9b0fa76e…` | `c78660b1…` | `07a3ecf9…` |

Full hashes and exact comparisons are in [H03_CPU_PAIR.json](H03_CPU_PAIR.json). The first differing CPU PTX function is **none**, because the entire PTX record is equal. The first differing original GPU function is **unknown**; aggregate opcode counts cannot recover it.

[Pinned XLA `GpuCompiler::AddFusionAutotuningPass`](https://raw.githubusercontent.com/openxla/xla/b6f37ab/xla/service/gpu/gpu_compiler.cc) returns immediately for a null stream executor. Consequently, setting normal autotuning and require-complete on CompileOnly does not exercise the GPU NativeEmitter-versus-BlockLevelEmitter fusion-autotuner. The CPU native records even equal the earlier autotune-0 proof. All 1,227 observed tuning-file opens are accounted for by the harness's explicit precompile hashing loop; there were no additional tuning-file opens. [H01_NORMAL_CPU.json](H01_NORMAL_CPU.json) records this audit. This identifies a missing stage in the old proof, **not the specific cause of C2b's GPU difference**. Cache misses, GPU tuning choices and cold nondeterminism remain unclassified for that deleted artifact.

There is a second independent loader issue. [GpuAotCompilationResult::LoadExecutable](https://raw.githubusercontent.com/openxla/xla/b6f37ab/xla/service/gpu/gpu_aot_compilation_result.cc) supplies process debug flags to [GpuExecutable::FromProto / Create](https://raw.githubusercontent.com/openxla/xla/b6f37ab/xla/service/gpu/gpu_executable.cc), which runs the thunk pass pipeline with those flags. The retained module's options are a separate input. Thus editing stored DebugOptions208 alone does not select the reload partition. [H02_PROOF.json](H02_PROOF.json) proves preservation of 101,178,413 native-chunk bytes and every other rest byte, but its stored-option simulation is **not runtime attestation**. The earlier CB06/cbmini construction needs this qualification too.

The proposed construction loads the **same complete donor blob** twice, in fresh processes whose flags differ only in runtime graph-min-size 5 versus 1. [runtime_meta.py](runtime_meta.py) derives only the runtime fingerprint, preserves the original compile fingerprint in a receipt, and leaves the standard fingerprint, blob, target and ABI guards enabled. It refuses any additional flag delta, damaged blob, duplicate flags or changed stored min. Derived metadata are for explicit private providers only; automatic cheap-key/HLO cache aliasing is forbidden. This is proof code awaiting review, not a product cache-policy exception.

Runtime min1 is process-wide: auxiliary graph partitions and d01 can change too. The preregistered [GPU_PLAN.json](GPU_PLAN.json) therefore requires identical native coverage for all numeric auxiliary work, one frozen real initial carry pack, zero scored compiles, observed graph IDs/WHILE capture, terminal and history bytes, host time per d02 call, kernels/copies/nodes, wall per root, readiness, own-PID VRAM/RSS and auxiliary recording/update costs. If full explicit executable coverage is unavailable, preparation stops before timing. Warm ABBAAB and profiling use separate bounded sessions; no launcher or window is armed.

Measured CPU costs: min5 compile **165.890 s**, total **731.334 s**, peak RSS **6.349 GiB**; min1 compile **143.038 s**, total **412.391 s**, peak RSS **6.252 GiB**. These are setup costs on shared CPUs, not speed claims. CPU E41 prescreen passed and carries to min1 by exact native identity; actual GPU E41 remains separate. The final proof suite passed **18/18**. Deleting either derivation produced its intended assertion failure with zero setup errors. Source URLs, hashes and the loader call chain are in [H04_RUNTIME_DESIGN.json](H04_RUNTIME_DESIGN.json).

Native arithmetic, serialized allocation/alias data and kernel work are identical in the proposed pair. Runtime graph creation can still change memory, synchronization, conditional-node overhead, updates and readiness; these costs are unmeasured. This design does not reduce the underlying S3 kernel-node count and makes no WRF-fidelity claim.

The manager decision is whether to schedule the separate frozen GPU `N5_A / N5_B / N1_C` diagnosis in the plan, or retain G1 as deferred. Identical-option cold controls distinguish nondeterminism from a graph-size-dependent difference before pass bisection. Preserve native bytes, first-function excerpts and pass dumps until attribution finishes. A CPU-only replay cannot recover the deleted C2b PTX or exercise the omitted pass.
