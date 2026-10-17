# Backend portability

The model retains its JAX interfaces and selects execution by backend capability.
CUDA uses the existing Pallas/Triton kernels. On other JAX backends, a local
Pallas facade lowers the same kernel bodies to ordinary JAX operations with
Pallas's interpreter. Tridiagonal solves use a JAX Thomas recurrence outside
CUDA. These paths retain the release's precision and WRF fidelity switches.

The interpreter is an execution implementation, **not an independent scientific
reference**. The portability gate exercises the same native kernels through the
facade and direct Pallas interpretation, within documented REAL tolerances, plus
finite release-default Swiss CPU steps. CUDA callable/source and release jaxpr
identity preserve the NVIDIA program. This lane does not execute new CUDA arms.
Original-WRF component fixtures supply separate scientific checks where available.

Legacy reference paths are not WRF-faithful; non-CUDA runs use the native kernels
through the facade. In particular, the old full acoustic composition has B13 W/PH
ownership differences, stale `mu_work`/total reconstruction, and downstream pressure
differences; the old PD reference also differs. These are recorded diagnostics,
not portability acceptance references. No legacy equations are fixed in this lane.

| Backend | Dispatch | Qualification |
|---|---|---|
| NVIDIA CUDA | Original Pallas callable, Triton kernels and PTX rounding | CUDA program identity checked on CPU; prior GPU receipts apply to unchanged work |
| CPU | Pallas bodies lowered to JAX; JAX Thomas solves | Two release-default Swiss steps finite; targeted component checks; full forecast qualification pending |
| AMD ROCm | Portable JAX route; no NVIDIA setup | Hardware untested; capability dispatch tested with simulated devices |
| TPU | Portable JAX route; no NVIDIA setup | Hardware untested; DOUBLE islands and integer-64 support require qualification |
| Apple Metal | Portable JAX route when a compatible JAX plugin is installed | Hardware untested; required precision support unqualified |
| Intel / other PJRT GPU | Portable JAX route when a compatible plugin is installed | Hardware untested; required precision support unqualified |

Use an explicit JAX platform pin for reproducible backend selection, for example
`JAX_PLATFORMS=cpu gpuwrf run ...` or `JAX_PLATFORMS=rocm gpuwrf run ...` with an
installed matching JAX backend. This change does not install accelerator plugins
or emulate unavailable float64/integer64 support. A rejected plugin or unsupported
primitive remains an error; the model does not silently downcast DOUBLE islands.

## NVIDIA assumptions and their boundaries

| Assumption | Location | Portable behavior |
|---|---|---|
| GPU-required allocation | `contracts/state.py` | State constructors use the selected JAX device; empty/unavailable backends still fail in JAX |
| `cuda_async`, pool sizing, NVIDIA VRAM queries | `cli.py`, `runtime/gpu_allocator.py`, `integration/nested_pipeline.py` | No allocator re-exec, CUDA defaults, or C-auto NVIDIA query on a non-CUDA target |
| NVIDIA VRAM/lock preflight | `runtime/gpu_preflight.py` | Explicit skipped/backend receipt on non-CUDA; CUDA locking and admission unchanged |
| `nvidia-smi` compute capability in cache tag | `runtime/compile_cache.py` | CPU/ROCm/TPU/Metal/Intel tags do not invoke NVIDIA tools |
| GPU command buffers, autotune, parallel compilation XLA flags | `runtime/xla_autotune.py` | No CUDA flag injection for non-CUDA targets; operator-supplied flags remain operator policy |
| Pallas/Triton and PTX inline assembly | `kernels/*.py`, `nesting/sint_kernel.py`, `coupling/boundary_apply.py` | Local facade selects JAX interpretation; rounding helpers use portable arithmetic |
| NVIDIA compiled thunks / cubin metadata | `runtime/aot_executable.py` | NVIDIA-only compaction/export/load transforms require a CUDA device |
| `CUDA_ROOT`, `ptxas`, `cuobjdump`, Nsight | Development profiling/build scripts, `backends/cuda_tile/`, `backends/jax/bench.py` | Outside the portable forecast path; no portable E41, roofline or Nsight claim |

Bootstrap detection does not initialize JAX: explicit platform pins win, AMD
visibility/device nodes distinguish ROCm from the generic `gpu` alias, and NVIDIA
device nodes/visibility establish the default CUDA target. Kernel dispatch inspects
the actual PJRT client's platform version and device kind. An unknown accelerator
uses the portable route.

## Parity and limits

`tests/test_portability.py` checks vendor selection, original CUDA callable identity,
CUDA-free lowering, state selection, and the hydrostatic recurrence. The companion
`tests/test_portability_kernels.py` compares direct stencils and real-operand dycore
operators with separate equations. Existing component gates cover radiation, PBL,
microphysics, land, cumulus and nesting.

New REAL comparisons reject NaN/Inf in both arms and bound maximum absolute error
by a documented number of float32 epsilons times the reference field scale (normally
64, coupled and boundary operators 8, hydrostatic 4). EOS perturbation pressure uses total pressure
as the cancellation scale. Random streams use exact equality. Existing component
gates retain their original tolerances; none are widened by backend dispatch.
CPU contraction, reduction order and subnormal handling differ from CUDA PTX RN32,
so interpreter success does not imply GPU byte equality.

`proofs/s1_port/FAMILY_INVENTORY.json` maps all 31 Pallas modules to component
checks and source-identity receipts. `proofs/s1_port/release_identity.py` adapts
the established release probe for CPU-only native-program tracing: initialization
uses the portable route, then the local facade selects the original callable for
the trace. Merely mocking `jax.default_backend()` no longer establishes CUDA
capability. This proof does not compile or execute a CUDA executable.

[M, CPU/source] The family checks cover 120 unique passing tests, with an observed
native factory in each of the 31 modules. The static audit covers 48 factory sites
and retains CUDA-equivalent source bytes and line counts. The CUDA-callable mutant
produces one genuine assertion failure and zero setup errors. These are portability
and source-identity results, not a new GPU or whole-forecast scientific validation.

Supplemental EOS and UV comparisons disable the CPU algebra-simplification pass locally to
avoid the known E130 division reassociation; production flags are unchanged.
BouLac uses the release's WRF REAL constants in both arms. The acoustic reference
helper predates specified/nested ownership (B13), so the retained diagnostic wrapper supplies
WRF's `advance_uv` update bounds from `module_small_step_em.F:735–751` before
downstream operators. Its wider coefficient/work outputs are compared at native
REAL storage precision, with finiteness checked before conversion. Those old-path
diagnostics still fail and are kept outside the portability acceptance suite.

CUDA returns the original `pallas_call` object, adds no wrapper to native kernel
tracing, and retains kernel source lines. [I, from program/source identity] The
facade introduces no additional native timestep launches, temporary arrays or
host/device copies. The rebased upstream initialization ownership guard is retained.
Capability checks occur at trace time;
source fingerprints change, so existing executable caches may need warming again.
Portable interpreter compile cost, memory use and speed require measurement for
each workload and device. No ROCm/TPU/Apple/Intel forecast validation is claimed.

[M, CPU] The smoke uses the real Swiss RD11 case (42×42 columns, 44 levels,
18 s timestep), release defaults and the normal operational advance wrapper,
without changing model dispatch. Two steps leave all 167 floating-point carry
leaves finite and invoke no NVIDIA tools. On shared core 29, load took 309.882 s,
the first step including cold trace/compile took 1376.216 s, and the second step
took 1.890 s. Process peak RSS including compilation was 6,359,104 KiB. These
are workload-specific CPU costs, not GPU timings or a CPU-WRF speed comparison.
Times use the monotonic clock; the host suspended during this arm, extending its
calendar duration. The receipt retains the existing REAL-context dtype warning.
The default event census was disabled (`null`), so this smoke establishes final
carry finiteness rather than a pre-mask event or trajectory-fidelity qualification.
The receipt is `proofs/s1_port/receipts/PORT32_cpu_smoke.json`.

The final gate seal is `proofs/s1_port/receipts/QUALIFICATION.json`. Required shared
consumers cover 252 cases: 251 executed passes and one previously qualified CAM
pass reused after exact program and all 80 constants matched. A clean tracked
archive passed 19 portability checks and 21 CLI checks; all 526 archived source
files matched their committed bytes. The model tree qualified here is
`8063c7f205647bf0caa67a6d121db9fbdd2ba00e`, composed on main `489a91483`.
Later integration changes require the manager's normal composition review.
