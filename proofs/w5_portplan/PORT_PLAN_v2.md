# PORT_PLAN v2: fast GPU kernels beyond NVIDIA for wrf_gpu v0.5.0 (plan only)

Lane w5-portplan · Opus 5.5 · 2026-10-17. Base `lane/w5-portplan` 5797475922 = main v0.4.0 tree.
- v2 incorporates the Astra critique (`<USER_HOME>/wrf_gpu2_lanes/w5-portplan-astra/PORT_PLAN_CRITIQUE.md`, 11:52Z). The disposition of each revision is in §5.
- v1 stays frozen at `PORT_PLAN.md` (sha256 f7582fed…).
- No GPU was used, and nothing was implemented, rented or bought.
- Evidence comes from CPU source reading, two CPU probes (W5P01 offline compile census, W5P02 rounding codegen probe) and existing receipts.
- Labels: [M] measured or read from source/receipts, [I] inferred, [U] unknown, [W] primary web source checked 2026-10-17.
- Machine-readable companion: `PORT_INVENTORY.json`. Per-module census: `PORT_CENSUS.md`.
- Captured TTIR: `W5P01/ttir_archive.tar.zst`; the inventory's TTIR paths are archive members. Tools: `tools/`, also committed to `lane/w5-portplan:proofs/w5_portplan/`.

## 0. Answer

**Recommendation (conditional on the M1 go/no-go, §4.2).**
- Keep one kernel source. The 50 Pallas modules are `jnp` code traced into Triton.
- On Triton-capable GPUs, run those same kernels natively instead of through the interpreter.
- Select every implementation through a **qualified capability tuple**: device arch + PJRT/JAX/plugin version + launch route + arithmetic policy + cache ABI. Never select on the string `gpu`.
- Isolate the few vendor-specific constructs in one `vendor_ops` module: WRF-literal RN ops, math externs, typed memory, gather, launch geometry and control-flow policy.
- XLA glue stays XLA. The interpreter remains the CPU/TPU/unknown route.
- No second implementation and no new DSL. A hand kernel or a per-family XLA variant is allowed only for a measured blocker (§2 (c), (e)).

**What the evidence supports, and what it does not.**
- [M source] JAX 0.10 contains a ROCm branch in its Pallas-Triton lowering (`pallas_call_registration.py:86,103`). It also has OCML math tables and a hip `gtsv` lowering.
- [W] AMD ships `jax-rocm7-plugin 0.10.0+rocm7.14` (our JAX version) for gfx90a/942/950/1100/1101/1200/1201 ([AMD install matrix](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)).
- [M CPU, W5P01] The *portable* (non-CUDA-branch) bodies of the release kernels lower to ROCm Triton IR and compile with **stock** Triton 3.6. Result: gfx942 58/58 PASS, gfx1100 57/58 PASS [b_core_noah_canopy_flux TIMEOUT > 900 s wall at nice 19], sm_75 57/58 PASS [b_core_noah_canopy_flux TIMEOUT > 900 s wall at nice 19], sm_61 55/55 PASS (3 large kernels not compiled: drivers stopped at FINAL) of 58 sites. VGPR spills/scratch: gfx942 none; gfx1100 `dyn_acoustic_fp32` b_core_coef_fp32 112 VGPR spills / 452 B scratch; `fused_vertical_implicit` <lambda> 126 VGPR spills / 508 B scratch.
  - This is discovery evidence for route A2.
  - It is only a proxy for route A1, because XLA's bundled Triton is another compiler (the CAM-MG precedent).
  - It is not hardware evidence and not an E41 result.
- [M CPU] Two release modules do not lower for ROCm: `boundary_apply` (scalar couple) and `dyn_ring_relax_fp32`. Their non-CUDA RN branch is CPU-prescreen code (f64 + `reduce_precision`, which has no Triton rule).
- [M source] The non-CUDA facade today runs Pallas `interpret=True`, a sequential `while_loop` over the whole grid (`hlo_interpreter.py:413–472`). On PROD d02 that means 31,239 sequential Thompson iterations per call.
- [M wz1080] On a GTX 1080 Ti, interpreted release defaults did not finish forecast hour 1 of the 43×43 Swiss case in 6 min, i.e. > 360 s/fc-h, against CPU-WRF 9.6 s/fc-h on the same box.
- This route is short-smoke and component qualified on its documented scope (S1-port). It is **not** a speed route.
- [M] Pre-Ampere NVIDIA has no native route today:
  - XLA's in-graph Triton refuses cc < 8.0 (wz1080 receipt).
  - The product preflight refuses such GPUs with exit 69 (`gpu_preflight.py:518–530`). `GPUWRF_ALLOW_UNSUPPORTED_GPU` changes admission only.
  - A native pre-Ampere route therefore needs the stock-Triton route A2. In our code A2 is CUDA-only (row 15) and sits behind that cc ≥ 8.0 preflight.
- [W] **Pallas-Triton deprecation.** The JAX 0.11.0 changelog says: "The Triton backend is deprecated and will be removed in a future version of JAX" ([Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)). Our whole hot path depends on it. The response is to pin the exact validated tuple now, with a bounded compatibility spike (M6) before any JAX upgrade.
- [W] **ROCm control flow.** Upstream XLA `rocm_command_buffer.cc` (commit 65ad34a5303, 2026-10-15) still returns `UnimplementedError` for conditional and while nodes (found by Astra).
  - Control flow that CUDA keeps inside command buffers becomes host-driven on ROCm.
  - That makes device control flow and readiness the **first** acceptance gate (M1). It is no longer a late risk.

**Speed.** Current AMD end-to-end speed is [U].
- §3 gives per-family *planning scenarios* against named assumptions. They are not acceptance thresholds and are never multiplied into CPU-speedup claims.
- Acceptance numbers come from M1/M4 measurements against an equally declared CUDA run: B=1 wall, and the maximum-resident-case throughput at the real memory limit.

**Effort [I].**
- Kernel-reuse base: 20.5 lane-days (LD). That is M0–M5 + M2b + M7 in §4.2.
- Plus risk reserves: 3–8 LD for control-flow/readiness integration and 1–4 LD for the pre-Ampere launch/cache/capture route beyond M2b.
- **Conditional envelope: 24.5–32.5 LD**, excluding reviews, hardware waiting and optional-scheme compiler repairs.
- M6 (6–10 LD, conditional) is separate.
- Re-estimate after M1. A NO-GO there changes the architecture, not just the schedule.

**Owner decisions (§4.4):**
- D1 target tiers;
- D2 gate policy;
- D3 AMD rounding;
- D4 hardware access (RDNA) and budget;
- D5 the exact JAX tuple pin;
- D6 the rule for hand kernels.

## 1. Inventory today

### 1.1 Kernels, routes and historical time ranking

[M source] Kernel modules:
- 50 files call `pallas_call` outside the facade. Astra's independent AST audit agrees: 72 factory sites in 50 files.
- 31 modules (12.1k lines, 48 factory sites) are on the release path (`proofs/s1_port/FAMILY_INVENTORY.json`).
- The other 19 (5.9k lines) are options and diagnostics: CAM-MG/ZM, P3, NTU, Jensen, QNSE, SMS ×2, MSKF ×2, KF-CuP ×2, CuPlus, Ferrier, Goddard, Milbrandt, NWP/radar diagnostics, stochastic RNG.
- Several kernels also trace retained physics code; Thompson-full, for example, reuses `physics/thompson_column.py`.

Routes and pins:
- All kernels use the Triton lowering; `JAX_PALLAS_USE_MOSAIC_GPU=false` is pinned.
- The default route is A1, the in-graph `__gpu$xla.gpu.triton` custom call with XLA-bundled Triton.
- CAM-MG uses A2, stock Triton 3.6 + jaxlib `triton_kernel_call` (`kernels/_triton_kernel_call.py`). XLA-bundled Triton miscompiled it on sm_120.

Time ranking [M, historical]: BP83 RC3 (`925b22dda2:tests/v025/b_phys/BP83/ROOFLINE_RC3.md`).
- What it is: BP82 RC2 measured roots mapped onto RC3 HLO, 3 roots per class, ranking only. Its modelled floor is not a DRAM roofline certificate.
- Ordinary root (1 d01 + 3 d02 steps): 33.1 ms kernel+copy, 3,662 launches.
- Radiation root: 81.7 ms, 11,136 launches; RRTMG accounts for 45.1 ms.
- Ordinary families in ms: acoustic 9.84 · MYNN 4.03 · advection 3.30 · carry glue 2.50 · PD 2.08 · lateral boundary 1.71 · DtoD 1.64 · other 1.64 · RK 1.27 · Thompson 1.26 · layout 1.24 · diffusion 1.12 · Noah-MP 0.56 · sfclay 0.38 · force-down 0.28.

Families mix Pallas kernels and XLA glue, so the Pallas/XLA split is [U].
- No WN3 family table exists.
- The only v0.4 numbers are per-root totals. PROD hour-2 is 55.05 ms/root. WN3 0227 hour-2 is 0.1153 s/root = 7.686 s/fc-h, which is not whole-run S1 (`<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/C5_353a85f8a1_20261016/PERFORMANCE.json`).
- The v0.4 release notes make no new S1/roofline claim.
- **M0 therefore takes one fresh, cadence-weighted v0.4 CUDA family profile** (PROD + WN3) before any tuning target is set.

### 1.2 CUDA-only assumptions (release path, plus option rows where marked)

| # | Assumption [M source] | Where | Off-CUDA today | v0.5.0 action |
|---|---|---|---|---|
| 1 | `cuda_native()` means both "NVIDIA" and "native Triton". Its 24 sites mix three meanings: (i) native code shape (facade; `rad_mcica` lane clamp + KISS-kernel enable; Milbrandt gather/roll); (ii) vendor asm; (iii) CUDA libs/AOT. Native dispatch also lives in the NTU/Jensen/P3 adapters, and CAM-MG sets `interpret=not cuda_native()` | `backends/pallas.py`, `rad_mcica.py:119,179`, `physics/{ntu,jensen_ishmael,p3,cammg}/adapter.py`, 20 more | interpreter (sequential grid loop) | split into `triton_native(tuple)` and `vendor()`; audit the resolved call path per scheme, not only imports |
| 2 | PTX inline asm for WRF-literal RN ops (`{op}.rn.f32`) | defined in `dyn_rk_fp32._rn`, `dyn_ring_relax_fp32._rn`, `boundary_apply.py:1579`, `sint_kernel.py:112`; **consumed** by MYNN tridiag2, `dyn_boundary`, `phys_hydrostatic_real`, NWP diagnostics, SMS (options), plus Ferrier/Goddard/Milbrandt/P3 locally | either plain jnp, which AMD contracts to FMA (W5P02), or f64 + `reduce_precision`, which does not lower (W5P01) | `vendor_ops.rn`: PTX on CUDA, AMDGCN `v_{op}_f32` on hip; every consuming family re-runs its exact-REAL contract |
| 3 | PTX typed global memory (`ld.global`, `atom.global.exch`) | `_column_memory/_p3_memory/_qnse_memory` (NTU, Jensen, P3, QNSE options) | [M] AMD compile CRASH (PTX register constraint) | It exists for two reasons: to avoid an NVIDIA Coalesce stall (E157) and to keep replicated lanes from racing. On hip, use AMDGCN `global_load`/`global_atomic_swap`, or prove plain `pl.store` safe with a lane-replication test |
| 4 | Math externs registered for CUDA only (`__nv_*`) | `_ntu_math.py` erf, `milbrandt2mom/rounding.py` | [M] NTU `KeyError: 'rocm'` | add OCML entries |
| 5 | `tt.gather` custom lowering | Thompson sed/full, KF, CAM-MG, Milbrandt | compiles on AMD (census) | keep |
| 6 | cuSPARSE `gtsv` (legacy paths) | `physics/tridiagonal_solver.py`, `dynamics/vertical_implicit_solver.py` | JAX Thomas `lax.scan`, one kernel per level | release path uses the Pallas tridiag2; legacy: hip gtsv |
| 7 | Command buffers `+CONDITIONAL,+WHILE` | `runtime/xla_autotune.py:396–460` | flags skipped. [W] ROCm conditional/while nodes are Unimplemented upstream; there is also a W7900 command-buffer segfault workaround | **M1 gate**: count DtoH/syncs/launches/allocations per step; budget static scheduling if needed |
| 8 | `cuda_async`, capped prealloc, C-auto via `nvidia-smi` | `cli.py`, `gpu_allocator.py`, `nested_pipeline.py` | JAX defaults (BFC, 75 % prealloc) | memory fraction from `get_compiled_memory_stats()` (E34) + PJRT `memory_stats()`; report true peak |
| 9 | `nvidia-smi` for the cache tag, preflight and VRAM | `compile_cache.py`, `gpu_preflight.py`, `gpu_allocator.py`, `ens/cli.py` | skipped | PJRT `device_kind` / `memory_stats` |
| 10 | Compiled-thunk AOT, slim module, cubin metadata | `aot_executable.py`, `aot_slim_module.py`, `jax_cache_thunks.py` | NVIDIA only | **M1 gate**: fresh-process cached readiness on ROCm (R3 ≤ 60 s); B12n shows that Triton re-emission at load cost 40 s on CUDA |
| 11 | `ptxas`/`CUDA_ROOT` (B48), E41 cubin gate | tooling | n/a | E41 stays; an AMD counterpart must be explicitly approved (§3.1) |
| 12 | Global x64; DOUBLE islands; int64 KISS | `_fast_defaults.py`, `rad_mcica`, `stochastic_rng`, Thompson | run on AMD/Intel Max/old NVIDIA; Arc A lacks fp64 hardware; Apple Metal documents no float64 | tier policy (D1) |
| 13 | Launch geometry tuned for 32-lane warps | all kernels | wave64 on CDNA doubles the threads per program | per-arch geometry table (M3) |
| 14 | XLA in-graph Triton requires cc ≥ 8.0 | XLA handler; preflight refuses (exit 69) | no native pre-Ampere route | A2 below sm_80 (M2b + M5 feasibility) |
| 15 | A2 is CUDA-only | `_triton_kernel_call.py:97–170`: `CUDABackend`, PTX, `platform != 'cuda'` raises | unusable on ROCm and pre-Ampere | generalize to `HIPBackend`/hsaco and any CUDA cc (M2b); JAX's own non-default branch is the template |
| 16 | Pinned Pallas-Triton lowering; JAX ≥ 0.11 deprecates it | `_fast_defaults.py`, every kernel | — | pin the exact tuple; M6 spike |

### 1.3 What the S1-port facade already gives

- backend-dark bootstrap;
- PJRT vendor detection;
- the original CUDA callable kept unchanged;
- interpreter parity for 31 modules (120 tests);
- a 2-step CPU smoke.

This is the right skeleton. v0.5.0 changes what the facade returns for a qualified Triton tuple and moves the vendor bodies into `vendor_ops`.

### 1.4 W5P01 census (discovery, not acceptance)

Scope [M]:
- The capture plugin (`tools/w5capture.py`) re-traces every `pallas_call` of the 44 S1-port family nodes in a CPU process. It therefore records the **portable bodies**, with `cuda_native()` False (E115/E117 trap).
- It lowers each one with `lower_jaxpr_to_triton_module` for `rocm` and for `cuda`/120, and compiles with **stock Triton 3.6**: hsaco, or PTX + bundled `ptxas` 12.8.
- Shapes are the tests' shapes. Most kernels use the real 44 levels (acoustic, Thompson, MYNN tridiag2); RRTMG uses 16 layers.
- Kernels that unroll `nz` (MYNN tridiag2, Thompson, KF) can change their code at other shapes. M0 recaptures at real WN3/PROD shapes and flags.
- Of 79 captured release invocations, 46 portable bodies give byte-identical ROCm/CUDA TTIR. The other 33 differ only in math extern symbols (`__ocml_*` vs `__nv_*`). This says nothing about the production CUDA bodies, which carry PTX asm in 5+ modules (§1.2 row 2).

Denominators over 58 release kernel sites (per platform, plus 2 lowering failures):

| Family | Kernel sites | gfx942 PASS · max VGPR · min occ | gfx1100 PASS · max VGPR · min occ | gfx90a / gfx950 / gfx1201 PASS | sm_75 PASS · max regs | sm_61 PASS |
|---|---:|---|---|---|---|---|
| acoustic | 7 | 7/7 · ≤366 · 1 | 7/7 · ≤256 · 5 | 5/7 +2 / 5/7 +2 / 5/7 +2 | 7/7 · ≤255 | 6/7 +1 |
| acoustic (legacy advance_w) | 2 | 2/2 · ≤384 · 1 | 2/2 · ≤256 · 5 | 1/2 +1 / 1/2 +1 / 1/2 +1 | 2/2 · ≤255 | 1/2 +1 |
| MYNN | 6 | 6/6 · ≤159 · 3 | 6/6 · ≤118 · 12 | 6/6 / 6/6 / 6/6 | 6/6 · ≤116 | 6/6 |
| advection | 6 | 6/6 · ≤62 · 8 | 6/6 · ≤117 · 12 | 6/6 / 6/6 / 6/6 | 6/6 · ≤203 | 6/6 |
| PD transport | 4 | 4/4 · ≤90 · 5 | 4/4 · ≤168 · 9 | 4/4 / 4/4 / 4/4 | 4/4 · ≤245 | 4/4 |
| lateral boundary | 4 | 4/4 · ≤36 · 8 | 4/4 · ≤49 · 16 | 4/4 / 4/4 / 4/4 | 4/4 · ≤49 | 4/4 |
| RK/EOS | 9 | 9/9 · ≤96 · 5 | 9/9 · ≤191 · 8 | 9/9 / 9/9 / 9/9 | 9/9 · ≤113 | 9/9 |
| Thompson | 4 | 4/4 · ≤223 · 2 | 4/4 · ≤189 · 8 | 3/4 +1 / 3/4 +1 / 3/4 +1 | 4/4 · ≤242 | 4/4 |
| diffusion | 2 | 2/2 · ≤36 · 8 | 2/2 · ≤75 · 16 | 2/2 / 2/2 / 2/2 | 2/2 · ≤64 | 2/2 |
| Noah-MP | 4 | 4/4 · ≤185 · 2 | 3/4 · ≤111 · 12 (TIMEOUT) | 2/4 +2 / 2/4 +2 / 2/4 +2 | 3/4 · ≤206 (TIMEOUT) | 3/4 +1 |
| GWDO | 1 | 1/1 · ≤77 · 6 | 1/1 · ≤73 · 16 | 1/1 / 1/1 / 1/1 | 1/1 · ≤64 | 1/1 |
| physics coupling | 1 | 1/1 · ≤17 · 8 | 1/1 · ≤14 · 16 | 1/1 / 1/1 / 1/1 | 1/1 · ≤32 | 1/1 |
| nest force-down | 1 | 1/1 · ≤90 · 5 | 1/1 · ≤176 · 8 | 1/1 / 1/1 / 1/1 | 1/1 · ≤205 | 1/1 |
| RRTMG | 6 | 6/6 · ≤135 · 3 | 6/6 · ≤121 · 10 | 6/6 / 6/6 / 6/6 | 6/6 · ≤154 | 6/6 |
| KF | 1 | 1/1 · ≤243 · 2 | 1/1 · ≤177 · 8 | pending / pending / pending | 1/1 · ≤238 | 1/1 |

`+k` = k sites without a result yet: either not compiled because the drivers were stopped at FINAL, or (gfx90a/gfx950/gfx1201) TTIR larger than 400k characters, which was not attempted. Lowering failures (no TTIR, so not counted above): `boundary_apply` scalar couple and `dyn_ring_relax` (`reduce_precision`). Per-module detail with VGPR/SGPR spills and scratch: `PORT_CENSUS.md`.

Reading it:
- gfx942: no completed isolated release compile has a VGPR spill or scratch.
- gfx1100 (RDNA3, 256-VGPR limit): the two largest acoustic-family kernels **spill**. `b_core_coef_fp32` has 112 VGPR spills / 452 B scratch (347 s compile); legacy `fused_vertical_implicit` has 126 / 508 B. These are the first RDNA targets of M3: block the unrolled level loop, or use smaller blocks.
- SGPR spills occur (KF 91, Thompson-full 76, SW band sums 32, Noah 20, acoustic 14 on gfx942). They are not proven free.
- KF and Thompson-full use 243 and 223 VGPR on gfx942 (2 waves/SIMD).
- `b_core_coef_fp32`, a 3.5 MB unrolled TTIR, and legacy `fused_vertical_implicit` reach 366 and 384 VGPR at 1 wave/SIMD.
- Thompson-full carries 1,730 f64 instructions out of 22,772; these are the DOUBLE islands.
- AMD compile time is a real cold-start item:
  - Thompson-full takes 109–130 s.
  - `b_core_coef_fp32` takes 228 s.
  - Noah canopy (1.17 MB TTIR, Python-unrolled Newton iterations) on gfx942: PASS (1353.16 s wall, 185 VGPR, occ 2), CPU 1142.33 s user, measured in the 1 h re-run after a 900 s timeout. It still exceeds 900 s on gfx1100 and on sm_75 (stock Triton + ptxas), so it is a pipeline cost, not an AMD one (R10).
- All of this is per isolated kernel. Whole-program fusion and layout cliffs (A19/A40/A47) are not covered.
- Options (9 of 19 modules captured): gfx942 3/16 PASS, 1 FAIL, 2 TIMEOUT, 7 CRASH, 3 not attempted/pending; PTX-asm kernels (Ferrier, QNSE, typed memory) CRASH on the PTX constraint; NTU erf LOWER_FAIL (cuda-only table); `nwp_helicity_column` AMD PassManager FAIL.

## 2. Options with evidence

**(a) Native Pallas-Triton on ROCm: CHOSEN, with two launch routes.**
- **A1, in-graph:** the XLA-ROCm bundled Triton. It needs no code beyond the capability tuple, and AMD's CI tests this route (upstream: gfx942/950; RDNA/MI200 downstream [W]). The default.
- **A2, stock Triton + `triton_kernel_call`:** the per-kernel escape when A1 fails a gate, and the only native pre-Ampere route.
  - Our wrapper is CUDA-only today. jaxlib binds the ROCm plugin's `_triton` (`jaxlib/gpu_triton.py:20,36`), and JAX's non-default branch shows the hsaco variant.
  - Generalize it in M2b. Do not switch all kernels to A2 for uniformity: its capture, compile, readiness and memory behaviour need their own qualification.
- [W] Precedent: marin runs one Pallas-Triton flash-attention source on MI300X, MI350X, H100 and GB200 (PR #10014, 2026-10). No large stencil/physics code with published cross-vendor roofline fractions was found.

**(b) Pallas Mosaic GPU: REJECTED for portability.**
- [W] Its API names NVIDIA Ampere/Hopper/Blackwell only ([Mosaic API](https://docs.jax.dev/en/latest/jax.experimental.pallas.mosaic_gpu.html)).
- It has no AMD path (ROCm issue jax#34669 closed as not continued). `pallas_call()` support was removed in 0.11.2.
- Our kernels are not matmul-shaped.

**(c) JAX/XLA implementations: no whole-model fallback, but selectively allowed.**
- The legacy column paths lose a lot. Measured on CUDA (fused kernel vs legacy XLA path):

  | Family | Gap |
  |---|---|
  | RRTMG LW | 9.1× (C20) |
  | KF | 273× (C21) |
  | MYNN plume | 50× (C18) |
  | Thompson | ≈ 10× (C24) |

  Several legacy paths are also not WRF-faithful (B13).
- A tuned fused-XLA variant of a **pointwise/stencil** family remains a legitimate per-family choice on a given tuple. Conditions: it preserves the equations and passes the same measured fidelity and speed gates. It needs no new science.

**(d) Thin DSL / code generator: only the minimal form.**
- The de-facto DSL already exists: `jnp` in Pallas. Owning its jaxpr→TTIR step is M6, a **compatibility spike**, not assumed future-proofing.
- The installed `lowering.py`/`primitives.py` (3,627 lines) import private `state`, `indexing`, `GridMapping`, tracing and MLIR APIs. Vendoring them does not freeze those contracts.
- A new HIP/SYCL DSL (GT4Py/DaCe, Kokkos) or a rewrite in the Triton Python DSL behind `jax_triton` would cost ≈ 18k kernel lines plus a full re-qualification. Rejected.
  - GT4Py was already excluded in M2.
  - Kokkos had the highest agent-iteration friction (ADR-001).

**(e) Per-vendor hand kernels (HIP C++/FFI): bounded escape, evidence-gated.** v1's 2-family cap and 50 % rule are replaced by a measured criterion. A hand kernel is allowed when all three hold:
- the family is on the measured critical path of a qualified tuple;
- after M3 tuning it exceeds its scenario band;
- rocprof/ISA evidence names a compiler limitation.

Each hand kernel needs Owner approval (D6), because it doubles maintenance and per-vendor oracle runs (≈ 3–5 LD each).

**(f) Older NVIDIA (sm_61/70/75): conditional feasibility via A2.**
- [M] A1 refuses below cc 8.0. The legacy XLA path ran at 24.8 s/fc-h steady on the 1080 Ti Swiss case, 2.8× slower than 6-core CPU-WRF (wz1080).
- [M CPU] Stock Triton PTX assembles with `ptxas` for sm_75 and sm_61 (sm_75 57/58, sm_61 55/55 of 58 sites compiled so far, ≤ 255 registers on sm_75). That is promising, but it bypasses the XLA compiler that refuses the product program.
- M5 is therefore a feasibility budget. It covers target selection, buffer aliases, stream/capture behaviour, cache reload and WRF qualification on the owned wz1080 box first.
- Pre-Ampere support for the v0.4.0 `GPUWRF_FAST_DEFAULTS=0` legacy path is not a goal.

### 2.1 Precision and the gate policy

- **Scope of W5P02 [M CPU].** It compiled `a*b+c` and `a/b+c` for gfx942/gfx1100 and counted instructions; it ran no numerics.
  - Plain code contracts to `v_fma_f32`.
  - `enable_fp_fusion=False` or AMDGCN inline asm `v_mul_f32`/`v_add_f32` keeps separate RN ops.
  - f32 `/` compiles to the `v_div_scale/fmas/fixup` (IEEE) sequence, while CUDA emits `div.full.f32` (E84).
  - This establishes codegen structure only. Whether AMD's local division semantics move a *coupled* forecast closer to or further from CPU-WRF is [U]; D6 and the oracles judge that.
- **Required semantics on the final primitive/compiler path:**
  - frozen WRF contracts, including cancellation, ties, signed zero, normals/subnormals (hsaco reports `denorm_mode_32 = 3`, preserve), and active wet/cold cases (E88/E102/E154);
  - math externs (OCML vs libdevice) [U] in ULP terms, covered by the same contracts;
  - fp64 DOUBLE-island cost on RDNA (advertised 1:32) [U] until measured.
- **Gate policy (D2).** Fidelity is per qualified tuple, against original CPU-WRF (D6 manifest on the manager-pinned WN3 release set) and pristine-WRF component oracles at frozen tolerances, plus F2. Bitwise equality is required only in three places:
  1. within one tuple, for determinism and AOT identity (two warm runs from one cache, E99);
  2. for exact-integer streams (KISS McICA, stochastic RNG);
  3. for existing exact-REAL component contracts against the pristine literal, such as MYNN tridiag2 vs literal WRF REAL (BP56) and the SINT/EOS literal gates. These are re-run per vendor against the *pristine reference*, never against CUDA.
- End-to-end CUDA↔ROCm byte equality is never required. GPU-vs-GPU comparisons are labelled regression only.

## 3. Families, support matrix, validation, hardware

### 3.1 Per family

- **t ratio** is a planning scenario: t(AMD)/t(RTX 5090) device time. It assumes bandwidth scaling × an AMD efficiency of 0.6–0.9, and that the B=1 host gap is not included. It is [I] and not an acceptance threshold.
- **LD** is that family's share of the M2–M4 effort, not additional work.

| Family (historical share) | Modules | Vendor-specific today | Verdict | LD | t ratio scenario MI300X / RX 7900 XTX [I] | Device validation (frozen tolerances) |
|---|---|---|---|---:|---|---|
| Acoustic (29.7 %) | `dyn_acoustic_fp32` (7) | none in body; geometry | reuse; geometry sweep (`coef`: gfx942 366 VGPR, occ 1; gfx1100 VGPR spills + 452 B scratch) | 0.5 | 0.6–1.0 / 1.8–2.6 (latency-bound w/mass loops, A54) | D7 pristine acoustic oracle; mass guards 0 |
| MYNN (12.2 %) | `phys_mynn_plume/tridiag/cloudmix/boulac`, `fused_edmf_plume` | tridiag2 uses `dyn_rk._rn` (RN ops) | small change (inherits `vendor_ops.rn`) | 0.5 | 0.5–0.9 / 1.6–2.4 | 48 WRF mean checks; tridiag2 exact vs literal WRF REAL (rule 3) |
| Advection (10 %) | `dyn_flux_fp32`, `dyn_momflux_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | momentum 12/12; pristine REAL4 operators |
| XLA glue/layout/DtoD (≈ 18 %) | XLA | XLA backend | XLA-ROCm; executing-binary scan (R5) | 0.5 | [U] | D6 + integrity |
| PD (6.3 %) | `dyn_pd_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | PD operator gates |
| Lateral boundary (5.2 %) | `dyn_ring_relax_fp32`, `dyn_boundary_fp32`, `boundary_apply` | RN ops; 2 kernels do not lower off CUDA | small change (M2a) | 1.0 | 0.4–0.7 / 1.9–2.6 | B36 ring tests; NumPy scalar-mass gate; D6 rings |
| RK/EOS (3.8 %) | `dyn_rk_fp32`, `dyn_uvtend/rhsph/omega_fp32` | `dyn_rk._rn` (PTX) | small change (M2a) | 0.5 | 0.4–0.7 / 1.9–2.6 | RK/EOS 222 checks; rhs_ph E133 gate |
| Thompson (3.8 %) | `phys_thompson_full/prep/sedimentation` | `tt.gather`; DOUBLE islands | reuse (`tt.gather` compiles) | 0.5 | 0.7–1.3 / 1.2–2.5 (register-bound; f64 cost [U]) | gate-2/L3a/L3b; mixed-phase pristine; adversarial ice/graupel (E154) |
| Diffusion (3.4 %) | `dyn_diff_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | 246 REAL field checks |
| Noah-MP / GWDO / coupling (3.0 %) | `phys_noahmp_columns/radiation`, `phys_gwdo_column`, `phys_hydrostatic_real` | hydrostatic uses `dyn_rk._rn` | reuse; small change (hydrostatic); Noah canopy compile time | 0.5 | 0.7–1.3 / 1.5–2.5 | Noah energy/phenology + REAL water rule; GWDO pristine |
| Nest force-down (0.9 %) | `nesting/sint_kernel` | PTX rn/div mode | small change (M2c) | 0.5 | 0.5–1.0 / 1.5–2.5 | SINT exact host-transcription gate |
| RRTMG (radiation root) | `rad_lw_transfer`, `rad_sw_band_sums/quadrature`, `rad_mcica` | `cuda_native()` code-shape branch (KISS kernel, lane clamp) | small change (`triton_native`) | 0.5 | 0.5–1.0 / 1.5–2.5 | LW/SW 18 + clear-sky 4 WRF oracles; KISS bitwise (rule 2) |
| KF (d01) | `phys_kf_column` | `tt.gather` | reuse; geometry (243 VGPR) | 0.5 | 0.8–1.5 / 1.5–3.0 | 5 frozen + 16,800 real + 8,400 deep-active columns |
| Options (19 modules) | see §1.1 | PTX typed memory/asm, cuda-only externs, adapters | per-scheme state (§3.2) | 2.0 | n/a | own component gates; `experimental` until qualified |


### 3.2 Support matrix and states

States per (backend × arch × scheme):

| State | Meaning |
|---|---|
| `interp` | functional only |
| `codegen` | CPU census green |
| `native-smoke` | archetypes on hardware |
| `component` | device oracles green |
| `qualified` | final composed WN3 D6 + F2 + readiness + memory on that exact tuple |

Rules:
- MI300X does not qualify RDNA, gfx1100 does not qualify gfx1101/1200/1201, and A1 receipts do not qualify A2.
- CUDA scheme receipts do not transfer. Each of the 19 option schemes carries its own state; schemes without one fail clearly or run as `experimental` with the execution path disclosed.
- The v0.4 support matrix (`docs/release/SUPPORT_MATRIX_v0.4.0.md`) is the template.

### 3.3 Validation design (CPU-WRF and pristine WRF are truth)

1. **CPU discovery (no hardware).**
   - The census runs path-filtered, only when kernel or `vendor_ops` files change. It does not run on every commit.
   - It keeps separate compiled / failed / timed-out / not-attempted counts and SGPR/VGPR-spill, scratch and occupancy fields.
   - Interpreter parity runs where the touched family has it.
2. **Archetype smoke on hardware (M1).**
   - Archetypes: stencil + RN32 (`dyn_rk` EOS, boundary couple); recurrence (acoustic w, MYNN tridiag2); DOUBLE + integer (Thompson-full, McICA KISS).
   - Each runs against its original-WRF fixture, including active wet/cold, inactive and padded lanes, odd grids, halos, cancellation, subnormals and integer overflow.
   - Assert in the measured process: native dispatch, actual device, native hsaco/cubin present, and 0 interpreter loops in the lowered program.
3. **Composed control flow and readiness (M1, gate).**
   - One ordinary + radiation + KF cadence sequence on the exact tuple.
   - Counts per step: transfers, syncs, launches and allocations, with fixed-trip acoustic loops and branch predicates.
   - Fresh-process cached readiness and true peak memory.
   - A finite forecast alone cannot pass this gate.
4. **Component fidelity on the device.** All frozen pristine-WRF oracles of the advertised families run at frozen tolerances (§2.1 rule 3).
   - Use the existing dedicated device-runner pattern with an in-process device assertion; the `tests/v025` CPU guard and GPU lock/lease rules stay.
5. **Final composed qualification (once, on the final candidate).**
   - D6 on every case of the manager-pinned WN3 release set, with manifest-bound field denominators.
   - Output integrity, F2 census and strict arm, and restart/AOT reload.
   - Station/long-run scope as required by the release.
   - PROD is iteration/debug only.
   - Unchanged native artifacts reuse their receipts (validation-reuse rule).
6. **Performance and resources.**
   - S1 whole-run, from process start to the last file, with B=1 and the maximum-resident-case throughput against a declared CUDA run.
   - R1 peak memory; R2 scaling α ≤ 1.10 on 3 sizes; R3 cold and cached readiness ≤ 60 s.
   - S3 transfers/ops; S4 roofline residual from rocprof counters.
7. **Static resource gate.** E41 stays for NVIDIA binaries.
   - An AMD counterpart (e.g. scratch 0 and VGPR spill 0 on the *executing* binaries, bound to the gate result) is a proposal that needs manager approval.
   - An offline stock artifact is never called an E41 result.

### 3.4 Hardware access and budget (nothing booked)

| Target | Source [W] | Plan | Cost |
|---|---|---|---|
| MI300X gfx942 | [DigitalOcean $2.59/h](https://docs.digitalocean.com/products/droplets/details/pricing/) (billed while stopped), [Runpod from $2.39/h](https://www.runpod.io/gpu-models/mi300x) | M1 15 h + M3/M4 35 h | ≈ $130–220 + storage |
| RDNA3/4 gfx1100/1201 | **no verified rental** (v1's DataCrunch quote is not on the current [Verda list](https://verda.com/pricing)) | borrow, quote, or buy one RX 9070 XT / 7900 XTX (≈ €650–900) — **D4** | [U] |
| MI250X gfx90a | Cirrascale $1.28–1.60/h (secondary source) | 10 h, best effort | ≈ $16 [U] |
| sm_61 | owned wz1080 box | M5 | 0 |
| sm_70/75 | vast.ai/Runpod V100/T4 $0.14–0.25/h (secondary) | 10 h each | ≈ $5 [U] |

Rental rules:
- pin the image by digest together with the wheel and compiler hashes;
- secure profiler/counter access;
- budget cache storage and teardown;
- every receipt records the platform, `device_kind`, ROCm/driver version and tuple hash;
- never share ROCm and CUDA caches;
- bring outputs home and score them on our CPU against CPU-WRF.

The $250–450 v1 figure stays only as a conditional MI300X compute allowance.

## 4. Architecture, milestones, risks, decisions

### 4.1 Architecture

```
kernel bodies (jnp in Pallas, unchanged science) ──► vendor_ops: rn · extern · gather · typed_mem · geometry · ctrl-flow policy
capability table: qualified tuple → {A1 | A2 | interp}; unqualified → interp + loud warning
jaxpr ──► TTIR (JAX lowering, pinned tuple; M6 spike for later JAX)
   ├─ A1 in-graph __gpu$xla.gpu.triton (CUDA ≥ sm_80 default; ROCm default)
   └─ A2 stock Triton (nvidia any cc | amd) + triton_kernel_call (per-kernel escape; pre-Ampere)
XLA glue stays XLA; selective fused-XLA family variants allowed under §2 (c)
```

`vendor_ops` interface:

```python
triton_native(tuple) -> bool       # native code shape (today's CUDA shape) for qualified tuples
vendor() -> "cuda" | "rocm" | "interp"
rn(op, a, b)                       # cuda PTX "{op}.rn.f32" | rocm AMDGCN "v_{op}_f32" (div: IEEE "/") | interp f64+reduce_precision
extern(name, x)                    # __nv_* | __ocml_*
gather(x, idx); typed_load/typed_store   # tt.gather; PTX typed mem | AMDGCN global_load/global_atomic_swap | plain
geometry(kernel, tuple) -> (block, num_warps, num_stages)   # default = today's CUDA values
```

The CUDA program stays byte-identical by default: the raw HLO/jaxpr identity check, E151.

### 4.2 Milestones and dependencies

| M | Content | LD | Depends on | Gate |
|---|---|---:|---|---|
| M0 | Discovery: census productized (`scripts/portability/compile_census.py`, path-filtered); recapture at real WN3/PROD shapes and flags; one fresh v0.4 CUDA family profile (1 GPU slot); pin the exact tuple (JAX/jaxlib/plugin/Triton/ROCm hashes) | 1.5 | — | report with full denominators (no pass bar) |
| M2a | Minimal `vendor_ops` (rn/extern/gather/typed-mem) + `triton_native` split; the boundary lowering fix | 2 | M0 | census lowers 31/31; CUDA raw HLO identical |
| M1 | **Feasibility go/no-go on MI300X** (time box 3 LD + 15 GPU-h): capability tuple, memory policy, archetypes (§3.3.2), composed control flow + readiness (§3.3.3) | 3 | M2a | §3.3.2–3 green. NO-GO → re-plan |
| M2b | A2 generalized (`HIPBackend`/hsaco, any CUDA cc); per-kernel route table | 2 | M2a | A2 compiles for gfx942/gfx1100/sm_70/75; CUDA unchanged by default |
| M2c | Remaining `vendor_ops` consumers (MYNN tridiag2, SINT, `rad_mcica`, option externs/typed memory) | 1 | M1 | per-consumer exact-REAL contract green |
| M3 | Per-arch geometry from measured bottlenecks on MI300X **and one RDNA target** (D4) | 3 | M1, M2c | family device times reported vs scenarios (no pass bar) |
| M4 | Qualification per advertised tuple (§3.3.4–7) | 5 | M3 | `qualified` states |
| M5 | Pre-Ampere A2 feasibility on wz1080 (sm_61), then T4/V100 | 2 | M2b | `native-smoke` → `qualified` or documented NO-GO |
| M7 | Docs, support matrix, `gpuwrf doctor` vendor checks | 1 | M4 | review |
| M6 | Compatibility spike on a named next JAX: one column, one stencil, one custom primitive, with aliasing, cache reload and device execution | 6–10 | M4 | conditional; must precede any JAX upgrade |

Effort and timing:
- The base is 20.5 LD. With the control-flow/readiness and pre-Ampere reserves (§0), the envelope is **24.5–32.5 LD**.
- Elapsed time follows the dependency chain, not total LD divided by agent count. The critical path is M0 → M2a → M1 → M2c → M3 → M4 → M7 ≈ 16.5 LD + reserves + hardware access.
- M2b → M5 runs in parallel with M1–M4. M6 follows M4.

### 4.3 Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Pallas-Triton deprecation [W] | exact tuple pin; M6 |
| R2 | XLA-ROCm bundled Triton miscompiles or hangs (cf. CAM-MG, E157) | A2 per kernel; archetype + component oracles |
| R3 | **Certain:** ROCm conditional/while command buffers are unimplemented upstream, so control flow is host-driven and the host gap grows. B=1 gap on CUDA is already ≈ 13–22 ms of a 46–55 ms root. Concurrency hides it only if the host CPU, streams and memory allow | M1 gate + 3–8 LD reserve: static scheduling, graph segments or a bounded runtime change, with owner and budget |
| R4 | Readiness without compiled thunks (R3 ≤ 60 s) | M1 fresh-process measurement; reserve |
| R5 | Whole-program fusion/layout differs on ROCm (A19/A40/A47) | executing-binary scan + rocprof family table |
| R6 | RDNA less tested upstream; W7900 command-buffer issue | an RDNA target in M3/M4 (D4) |
| R7 | Register pressure: wave64 occupancy 1–2 (KF, Thompson, coef on gfx942) and [M] VGPR spills on RDNA's 256-VGPR limit (coef, legacy advance_w on gfx1100) | geometry table; census SGPR/VGPR fields |
| R8 | Cloud capacity, profiler privileges | 2 providers; short resumable arms |
| R9 | Option schemes regress silently | per-scheme state; fail-closed or `experimental` |
| R10 | Cold-compile time of large unrolled kernels with stock Triton [M CPU]. Noah canopy took 1,142 CPU-s on gfx942 and exceeded 900 s on gfx1100 **and on sm_75**, so the cost is not AMD-specific; acoustic `coef` 228 s; Thompson-full 109–130 s. How A1 (XLA-bundled Triton) fares is [U]. That threatens R3 cold ≤ 600 s on every new tuple and on A2 | persistent/AOT cache per tuple; compile-time column in the census; if needed, replace Python-unrolled iterations by in-kernel loops (E38) under the exact-REAL contracts |

### 4.4 Owner decisions

- **D1 Target tiers.** Proposed targets:
  - Supported: NVIDIA sm_80+, AMD gfx942/gfx950, and one RDNA family.
  - Best effort: gfx90a and sm_70/75.
  - Assessment only:
    - Pascal (A2 on wz1080);
    - Intel (`jax[oneapi]` 0.11.2-alpha excludes Pallas and has executable-serialization failures [W]; Arc A lacks fp64 hardware);
    - Apple (the JAX-Metal plugin documents no float64, which blocks the DOUBLE islands [W]);
    - TPU (interpreter route).
  - Rule: a "supported" tier is published only in the `qualified` state, on its exact tuple.
- **D2 Gate policy** as in §2.1.
- **D3 AMD rounding:** AMDGCN asm RN ops, the exact analogue of our PTX. Accept IEEE division on AMD; its forecast effect is judged by D6/oracles.
- **D4 Hardware/budget:**
  - the MI300X allowance;
  - how to get RDNA access (borrow, quote, or buy ≈ €650–900, which needs a slot and a PSU);
  - which RDNA generation is the supported one.
- **D5 Pin** the exact validated JAX 0.10.0 + jax-rocm7 0.10.0 + Triton 3.6 + ROCm 7.14 tuple. M6 must pass before any upgrade.
- **D6 Hand kernels / XLA variants:** approve per family under the §2 (c)/(e) rule.

## 5. Critique disposition (Astra, `PORT_PLAN_CRITIQUE.md`)

| # | Revision | Decision | What changed / why |
|---|---|---|---|
| 1 | Control flow + readiness as an early gate | **Accepted** | M1 is now a go/no-go with archetypes + a composed cadence sequence: transfers, syncs, launches, allocations, cached readiness, memory. A finite run cannot pass. R3 is marked certain with a 3–8 LD reserve. The upstream `rocm_command_buffer.cc` evidence is cited |
| 2 | Bound census claims | **Accepted** | Census relabelled as discovery for portable bodies + stock Triton (A2 input, A1 proxy). Full denominators, separate SGPR/VGPR/scratch fields; the "no A19 cliff" and "shape-independent" wording removed. MYNN/Thompson/KF unroll `nz`, RRTMG used 16 layers, so M0 recaptures at real shapes. TTIR neutrality restated as portable bodies only. No E41 claim |
| 3 | Pre-Ampere route + dependencies | **Accepted** | M2b generalizes A2; M5 is a conditional feasibility on the owned wz1080 box; reserve 1–4 LD beyond M2b. The DAG now orders M0 (discovery) → M2a → M1 → … The v1 "hard failure" wording is corrected: preflight already refuses cc < 8.0 with exit 69 |
| 4 | Precision overclaims | **Accepted** | Removed the "1–2 ULP D6-irrelevant", "WRF-closer forecast" and "no new fp64 cost" claims. W5P02 is scoped to codegen. Added the edge-case contract list. Imported `_rn` consumers (MYNN tridiag2, `dyn_boundary`, hydrostatic, SMS, NWP) mapped in §1.2. The D2 vs tridiag2 inconsistency is resolved by rule 3 (exact-REAL contracts against the pristine reference) |
| 5 | Historical vs projected numbers | **Accepted** | BP83 is labelled historical ranking. WN3 is the 7.686 s/fc-h hour-2 receipt (v1's 7.4 had no receipt). The "75 % Pallas" claim is withdrawn. Ratios are planning scenarios; AMD speed [U]; the CPU-speedup multiplication is removed. A fresh v0.4 profile is added in M0. B=1 and max-resident throughput are measured against a declared CUDA run |
| 6 | Support/rental explicit | **Accepted** | 5-state support matrix per backend × arch × scheme. RDNA rental marked unverified, so access becomes Owner decision D4. Image digest pin, profiler access, storage and teardown added. Options carry per-scheme states, and CUDA receipts do not transfer |
| 7 | M6 as a compatibility spike | **Accepted** | M6 is a bounded spike on a named next JAX (column, stencil, custom primitive, aliasing, cache reload, device execution), 6–10 LD conditional. Exact tuple pin now (D5). No uniform A2 switch. Intel needs native launch + readiness, so it stays assessment-only |
| — | Effort envelope 24.5–32.5 LD | **Accepted** | §0/§4.2; re-estimated after M1 |
| — | Missing alternatives: selective fused XLA; drop the 2-family cap and 50 % rule | **Accepted, with one retention** | §2 (c)/(e) use the measured critical-path criterion. *Retained:* Owner approval per hand kernel (D6), because a second source per vendor is a lasting maintenance and oracle cost that the measured criterion alone does not price |
| — | Checklist items | **Accepted** | Interpreter wording; WN3 release set; manifest-bound denominators; R1/R2/R3/S3/S4/restart/station; device-runner pattern instead of weakening the `tests/v025` guard; one final validation with receipt reuse; Mosaic and Apple wording corrected with primary links |

Nothing in the critique was rejected outright. The one partial retention is D6 (above).

**Post-v1 census corrections.** These came from compiles that were still running at v1.
- gfx1100 VGPR spills in two acoustic-family kernels. v1 said "0 VGPR spills on every completed compile", which held only for the kernels completed at the time.
- Noah canopy compiles on gfx942 after 1,142 CPU-s, and exceeds 900 s on gfx1100 and sm_75 (R10).
- gfx90a, gfx950 and gfx1201 compile 51/51 of the ≤ 400k-character kernels.
