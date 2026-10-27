# PORT_PLAN v1: fast GPU kernels beyond NVIDIA for wrf_gpu v0.5.0 (plan only)

Lane w5-portplan · Opus 5.5 · 2026-10-17 · base `lane/w5-portplan` 5797475922 (main v0.4.0 tree). No GPU was used.
The work was CPU source reading plus two CPU probes (W5P01 offline AMD/NVIDIA compile census, W5P02 rounding probe).
Labels: [M] measured or read from source/receipts, [I] inferred, [U] unknown, [W] external web source checked on 2026-10-17.
Machine-readable companion: `PORT_INVENTORY.json`, in the same directory.

## 0. Answer in one page

**Recommendation.** Do not write a second implementation. The kernel bodies in the 50 Pallas modules are plain `jnp` code traced into
Triton, and that code is already vendor-neutral. The NVIDIA-only parts are thin. On every GPU that Triton supports, run the *same* Pallas kernels
natively instead of through the interpreter, and isolate the few vendor-specific constructs in one small `vendor_ops` table
(see §1.2). The table covers WRF-literal rounding ops, math externs, typed memory primitives, register gather and launch geometry.
The XLA glue stays XLA. The interpreter route stays only for CPU, TPU and unknown devices.

**Evidence that this is cheap.**
- [M] JAX 0.10 already contains a ROCm branch in its Pallas-Triton lowering (`pallas_call_registration.py:86,103`: `num_stages=1`, hsaco).
- [M] JAX 0.10 also has hipSPARSE `gtsv` and OCML math tables.
- [W] AMD ships `jax-rocm7-plugin 0.10.0+rocm7.14`, our exact JAX version. It covers MI210/MI250/MI300/MI325/MI355 and RX 7900/9000.
- [M] W5P01 captured the 58 release-path kernel sites of 29 of the 31 release modules, lowered every one to ROCm Triton IR on CPU and compiled each offline with stock Triton 3.6 for gfx942 (MI300) and gfx1100 (RX 7900). Results: gfx942 57/58 PASS [b_core_noah_canopy_flux TIMEOUT (900 s wall at nice 19; 1 h re-run W5P01/run_giant.sh)], gfx1100 55/55 PASS (3 large kernels still compiling), sm_75 51/51 PASS (7 large kernels still compiling), sm_61 51/51 PASS (7 large kernels still compiling) of 58 sites; AMD VGPR spills 0 and scratch 0 B on every completed compile.
- [M] The other 2 modules fail to lower for ROCm: `coupling/boundary_apply.py` (scalar couple) and `kernels/dyn_ring_relax_fp32.py`. Both are lateral-boundary kernels whose non-CUDA RN branch is CPU-prescreen code (f64 + `reduce_precision`, which has no Triton rule).
- [M] W5P01: the kernel IR is vendor-neutral. Of 79 captured release kernel invocations, 46 lower to byte-identical ROCm and CUDA TTIR. The other 33 differ *only* in math-library symbol names (`__ocml_*` vs `__nv_*`), after normalising debug locations.
- [M] W5P02: the AMD equivalent of our PTX `mul/add.rn.f32` rounding ops works. An AMDGCN inline asm `v_mul_f32`/`v_add_f32` compiles and is never contracted into an FMA.

**What the current "portable" route really is.** [M source] The non-CUDA facade runs Pallas in interpret mode, which is a sequential `while_loop`
over the whole kernel grid (`jax/_src/pallas/hlo_interpreter.py:413–472`). For example, Thompson runs `grid=(ncol,)`, which on PROD d02
is 31,239 sequential iterations per call. That route is correct but unusable for speed on any GPU. v0.5.0 replaces it on GPUs.

**Strategic risk found while planning.** [W] JAX 0.11.0 (2026-07-16) deprecated the Pallas-Triton backend: "will be removed in a future
version of JAX … switch to the official Triton bindings and jax_triton". Our whole hot path is Pallas-Triton.
- v0.5.0 stays on JAX 0.10.x, which has a ROCm plugin, so nothing breaks now.
- Before any JAX upgrade, the plan owns the jaxpr→TTIR step (milestone M6).
- That milestone reuses the stock-Triton + `triton_kernel_call` route that already exists, GPU-qualified for CAM-MG in `_triton_kernel_call.py`.
- This makes the port robust against both the deprecation and future vendors.

**Expected speed [I, wide error bars]** relative to the RTX 5090. Today's baselines are PROD 3.6 s/fc-h on v0.3.x/RC (BP83/CP147) and WN3 7.4 s/sim-h on v0.4.0.

| GPU | Single case | Multi-case throughput |
|---|---|---|
| MI300X | 0.7–1.2× | 1.5–2.5× |
| RX 7900 XTX | 0.45–0.65× | |
| RX 9070 XT | 0.35–0.5× | |
| MI250X (1 GCD) | 0.6–0.9× | |
| V100 | 0.4–0.6× | |
| T4 | 0.15–0.25× | |

How the ranges were built [I]: the device part scales with memory bandwidth × a Triton-AMD efficiency of 0.6–0.9. The host/launch gap does not scale; it is ≈ 13–22 ms of a ≈ 46–55 ms ordinary root on the 5090. So the MI300X roughly halves device time on a single case, but the gap caps the gain. Multi-case runs hide the gap.

Against CPU-WRF 12-rank (92.3 s/fc-h, PROD), this means ≈ 18–31× on MI300X, ≈ 11–17× on RX 7900 XTX and ≈ 4–6× on T4 (from the 5090's 25.6×).

**Effort.** v0.5.0 core (M0–M5, M7) ≈ 19 lane-days plus reviews; deprecation insurance (M6) ≈ 6–10 lane-days.
Cloud qualification ≈ US$ 250–450, with no hardware purchase needed (§3.2).

**Owner decisions needed (§4.4):**
- support tiers;
- the cross-vendor gate policy (no bitwise across vendors);
- the WRF-literal rounding policy on AMD;
- the budget and the optional purchase of one consumer AMD card for CI;
- the JAX version pin;
- whether per-vendor hand kernels are allowed at all.

## 1. Inventory today

### 1.1 Kernels and their device-time share

[M source, W5P01 static scan]
- 50 source files call `pallas_call` (the facade excluded). 31 modules (12.1k lines) are on the release-default path (`proofs/s1_port/FAMILY_INVENTORY.json`, 48 factory sites).
- The other 19 (5.9k lines) are option schemes and diagnostics: CAM-MG/ZM, P3, NTU, Jensen, QNSE, SMS(×2), MSKF(×2), KF-CuP(×2), CuPlus, Ferrier, Goddard, Milbrandt, NWP/radar diagnostics and stochastic RNG.
- Many kernels also trace retained physics functions in place (e.g. Thompson full reuses `physics/thompson_column.py` code objects), so the real kernel source is larger than these files.
- All release kernels use the Triton lowering. `JAX_PALLAS_USE_MOSAIC_GPU=false` is pinned in `_fast_defaults.py`.
- All release kernels go through the default in-graph custom call `__gpu$xla.gpu.triton`, which XLA compiles with its bundled Triton.
- The exception is CAM-MG, which is opted into stock Triton 3.6 plus jaxlib `triton_kernel_call` (`kernels/_triton_kernel_call.py`). XLA's bundled Triton miscompiled its control flow on sm_120.

Device time comes from the newest family table that exists, BP83 RC3 (PROD, RTX 5090, release defaults; `925b22dda2:tests/v025/b_phys/BP83/ROOFLINE_RC3.md`).
- Ordinary root (1 d01 + 3 d02 steps): 33.1 ms kernel+copy, 55.8 ms profiled wall, 3,662 launches.
- Radiation root: 81.7 ms, 144.9 ms wall, 11,136 launches; RRTMG accounts for 45.1 ms of it.
- No WN3 family table exists. WN3 shares are [I] equal to the PROD shares.
- PROD and WN3 run the same release schemes; only KF is limited to d01.

| Family (ordinary root) | ms/root [M] | Share | Native modules | Vendor-specific constructs in the kernel |
|---|---:|---:|---|---|
| Acoustic | 9.835 | 29.7 % | `dyn_acoustic_fp32` (7 kernels) | none |
| MYNN | 4.028 | 12.2 % | `phys_mynn_plume/tridiag/cloudmix/boulac`, `fused_edmf_plume` | none |
| Advection | 3.304 | 10.0 % | `dyn_flux_fp32`, `dyn_momflux_fp32` + XLA | none |
| Carry glue, layout, DtoD, other | ≈ 6.0 | ≈ 18 % | XLA-generated fusions and copies | XLA backend (portable) |
| PD transport | 2.076 | 6.3 % | `dyn_pd_fp32` | none |
| Lateral boundary | 1.709 | 5.2 % | `dyn_ring_relax_fp32`, `dyn_boundary_fp32`, `boundary_apply` | PTX `mul/add.rn.f32` (scalar couple) |
| RK / EOS | 1.274 | 3.8 % | `dyn_rk_fp32`, `dyn_uvtend_fp32`, `dyn_rhsph_fp32`, `dyn_omega_fp32` | PTX `.rn.f32` in EOS/stage masses |
| Thompson | 1.264 | 3.8 % | `phys_thompson_full/prep/sedimentation` | `tt.gather`, f64 DOUBLE islands |
| Diffusion | 1.118 | 3.4 % | `dyn_diff_fp32` | none |
| Noah-MP, sfclay, GWDO | 0.98 | 3.0 % | `phys_noahmp_columns/radiation`, `phys_gwdo_column` | none |
| Nest force-down | 0.282 | 0.9 % | `nesting/sint_kernel` | PTX `.rn` / `div` mode |
| RRTMG (radiation root, 2/h) | 45.1 | — | `rad_lw_transfer`, `rad_sw_band_sums/quadrature`, `rad_mcica` | int64 KISS; a `cuda_native()` clamp branch |
| KF (d01 only) | small | — | `phys_kf_column` | `tt.gather` |

[I] About 75 % of ordinary device time runs inside our Pallas kernels and about 25 % in XLA-generated fusions and copies.
The XLA part runs on ROCm without code changes, but its fusion and layout decisions may differ from CUDA (A19/A40/A47 class), see R5.

### 1.2 CUDA-only assumptions (complete list for the release path)

| # | Assumption | Where [M] | Effect off-CUDA today | v0.5.0 action |
|---|---|---|---|---|
| 1 | `cuda_native()` gates the native Pallas callable **and** means 'NVIDIA'. The 24 call sites mix three meanings: (i) native-vs-interpreter code shape (facade, adapters, `rad_mcica` lane clamp and KISS kernel enable, Milbrandt gather vs roll); (ii) vendor asm (`boundary_apply`, `dyn_rk`, `dyn_ring_relax`, `sint`, Milbrandt rn/pow); (iii) CUDA libraries/AOT (`gtsv`, compiled thunks) | `backends/pallas.py`, `rad_mcica.py:119,179`, `dyn_ring_relax_fp32.py:20`, adapters of 7 option schemes | interpreter: sequential grid loop | split into `triton_native()` (cuda or rocm: choose the CUDA code shape) and `vendor()` (asm strings) |
| 2 | PTX inline asm for WRF-literal REAL rounding | `boundary_apply.py:1579`, `dyn_ring_relax_fp32.py:20`, `dyn_rk_fp32.py:148`, `sint_kernel.py:112`; options: ferrier, goddard, milbrandt, p3 | portable branch is either plain jnp, which AMD contracts to FMA (W5P02), or f64 + `reduce_precision` (`boundary_apply`, `dyn_ring_relax`), which does not lower (W5P01) | `vendor_ops.rn(op)`: PTX on CUDA, AMDGCN `v_*_f32` on hip; on AMD division is already IEEE |
| 3 | PTX typed global memory (`ld.global`/`st.global`/`atom.exch`) | `_column_memory/_p3_memory/_qnse_memory` (NTU, Jensen, P3, QNSE options) | not used on the release path | stock Pallas loads on hip (the workaround targets an NVIDIA Coalesce pass, E157); re-test |
| 4 | libdevice `__nv_*` externs registered with `cuda=` only | `_ntu_math.py` (erf), `milbrandt2mom/rounding.py` | lowering fails on rocm (options only) | add `rocm=` OCML entries (`__ocml_erf_f32`, …) |
| 5 | `tt.gather` custom lowering | Thompson sed/full, KF, CAM-MG, Milbrandt | TTIR op is vendor-neutral | [M] the `tt.gather` kernels (Thompson sedimentation, KF) compile for gfx942: 3/3 PASS, ≤ 243 VGPR, occ ≥ 2, VGPR spills 0, SGPR spills 91 |
| 6 | cuSPARSE `gtsv` via `lax.linalg.tridiagonal_solve` | `physics/tridiagonal_solver.py`, `dynamics/vertical_implicit_solver.py` (legacy paths) | non-CUDA falls back to a JAX Thomas `lax.scan`, one kernel per level (B3) | release path uses the Pallas tridiag; legacy path: JAX already has a `hip` gtsv lowering |
| 7 | XLA command buffers `+CONDITIONAL,+WHILE` | `runtime/xla_autotune.py:396–460` | flags skipped off CUDA | probe on ROCm (hipGraph); [W] the W7900 known issue needs `--xla_gpu_enable_command_buffer=` |
| 8 | `cuda_async` allocator, capped prealloc, C-auto pool from `nvidia-smi` | `cli.py`, `runtime/gpu_allocator.py`, `nested_pipeline.py` | defaults (BFC, 75 % prealloc) | `XLA_PYTHON_CLIENT_MEM_FRACTION` from `get_compiled_memory_stats()` (E34) + PJRT `memory_stats()` |
| 9 | `nvidia-smi` for the cache tag, preflight and VRAM | `compile_cache.py`, `gpu_preflight.py`, `gpu_allocator.py`, `ens/cli.py` | skipped (S1-port) | tag from PJRT `device_kind`; VRAM from `memory_stats()` |
| 10 | Compiled-thunk AOT export, slim module, cubin metadata | `aot_executable.py`, `aot_slim_module.py`, `jax_cache_thunks.py` | NVIDIA only | legacy JAX cache on ROCm; [U] readiness (B12n re-emits Triton at load: 40 s on CUDA before the fix) |
| 11 | `ptxas`/`CUDA_ROOT` choice (B48), E41 cubin gate (`cuobjdump`) | tooling | n/a | AMD analogue: hsaco metadata gate (VGPR spill, scratch, instruction count) = W5P01 tool |
| 12 | Global x64; DOUBLE islands (Thompson 611 f64 ops, CLWRF REAL(8), bdy rdt; BD60); int64 KISS | `_fast_defaults.py` `JAX_ENABLE_X64`, `rad_mcica`, `stochastic_rng` | fine on AMD/Intel Max/older NVIDIA; Arc A has no fp64; Apple and TPU have no f64 | tier policy (§4.4 D1) |
| 13 | Launch geometry tuned for 32-lane warps (`num_warps`, block 128/256, levels padded to 64) | all kernels | wave64 on CDNA: same `num_warps` means 2× the threads per program | per-arch geometry table (M3) |

### 1.3 What the CPU-portable facade (S1-port, `docs/portability.md`) already gives us

The facade already provides the following [M]:
- backend-dark bootstrap: no NVIDIA tools, allocator or XLA flags off CUDA;
- vendor detection from PJRT `platform_version`/`device_kind`;
- the original CUDA callable object kept unchanged;
- interpreter parity for all 31 modules (120 tests, 64-eps REAL bounds);
- a CPU forecast smoke (Swiss, 2 steps).

This is the right skeleton. v0.5.0 changes only what the facade returns on a Triton-capable GPU, plus the vendor-specific bodies.
The parity tests become the CPU pre-screen of every AMD kernel.

### 1.4 W5P01 offline compile census (CPU only, new)

Method:
- A pytest plugin (`tools/w5capture.py`) re-traces every `pallas_call` that the 44 S1-port family nodes execute. The interpreted execution itself is unchanged; the shapes-only mode skips it.
- The plugin lowers each kernel jaxpr with JAX's own `lower_jaxpr_to_triton_module(…, 'rocm')` and `(…, 'cuda', 120)`.
- Each TTIR is then compiled by stock Triton 3.6 offline: hsaco for gfx942/gfx1100, and PTX → `ptxas -v` (bundled 12.8) for sm_75/sm_61.
- The run used lane cores and a 1 GB cgroup, with no GPU.
- Raw results: `W5P01/compile_rep.jsonl`; captured IR: `W5P01/all/` (186 unique TTIR); representatives: `W5P01/rep/` (58 per platform).

| Family | Module | Kernels | gfx942 PASS · max VGPR · VGPR/SGPR spills (scratch) · min occ | gfx1100 same | sm_75 PASS · max regs · spill B | sm_61 PASS |
|---|---|---:|---|---|---|---|
| acoustic | `dyn_acoustic_fp32.py` | 7 | 7/7 · 366 · 0/14 · 1 | 6/6 · 156 · 0/0 · 9 | 5/5 · 163 · 0 | 5/5 |
| lateral boundary | `dyn_boundary_fp32.py` | 4 | 4/4 · 36 · 0/0 · 8 | 4/4 · 49 · 0/0 · 16 | 4/4 · 49 · 0 | 4/4 |
| diffusion | `dyn_diff_fp32.py` | 2 | 2/2 · 36 · 0/0 · 8 | 2/2 · 75 · 0/0 · 16 | 2/2 · 64 · 0 | 2/2 |
| advection | `dyn_flux_fp32.py` | 3 | 3/3 · 56 · 0/0 · 8 | 3/3 · 114 · 0/0 · 12 | 3/3 · 203 · 0 | 3/3 |
| advection | `dyn_momflux_fp32.py` | 3 | 3/3 · 62 · 0/0 · 8 | 3/3 · 117 · 0/0 · 12 | 3/3 · 111 · 0 | 3/3 |
| RK/EOS | `dyn_omega_fp32.py` | 1 | 1/1 · 39 · 0/0 · 8 | 1/1 · 43 · 0/0 · 16 | 1/1 · 46 · 0 | 1/1 |
| PD transport | `dyn_pd_fp32.py` | 4 | 4/4 · 90 · 0/2 · 5 | 4/4 · 168 · 0/0 · 9 | 4/4 · 245 · 0 | 4/4 |
| RK/EOS | `dyn_rhsph_fp32.py` | 1 | 1/1 · 96 · 0/0 · 5 | 1/1 · 191 · 0/0 · 8 | 1/1 · 113 · 0 | 1/1 |
| RK/EOS | `dyn_rk_fp32.py` | 5 | 5/5 · 32 · 0/0 · 8 | 5/5 · 46 · 0/0 · 16 | 5/5 · 44 · 0 | 5/5 |
| RK/EOS | `dyn_uvtend_fp32.py` | 2 | 2/2 · 92 · 0/0 · 5 | 2/2 · 189 · 0/0 · 8 | 2/2 · 96 · 0 | 2/2 |
| MYNN | `fused_edmf_plume.py` | 1 | 1/1 · 159 · 0/2 · 3 | 1/1 · 96 · 0/56 · 16 | 1/1 · 116 · 0 | 1/1 |
| acoustic (legacy advance_w) | `fused_vertical_implicit.py` | 2 | 2/2 · 384 · 0/0 · 1 | 1/1 · 140 · 0/0 · 10 | 1/1 · 128 · 32 | 1/1 |
| GWDO | `phys_gwdo_column.py` | 1 | 1/1 · 77 · 0/0 · 6 | 1/1 · 73 · 0/0 · 16 | 1/1 · 64 · 0 | 1/1 |
| physics coupling | `phys_hydrostatic_real.py` | 1 | 1/1 · 17 · 0/0 · 8 | 1/1 · 14 · 0/0 · 16 | 1/1 · 32 · 0 | 1/1 |
| KF | `phys_kf_column.py` | 1 | 1/1 · 243 · 0/91 · 2 | 1/1 · 177 · 0/15 · 8 | pending | pending |
| MYNN | `phys_mynn_boulac.py` | 1 | 1/1 · 39 · 0/0 · 8 | 1/1 · 36 · 0/0 · 16 | 1/1 · 32 · 0 | 1/1 |
| MYNN | `phys_mynn_cloudmix.py` | 1 | 1/1 · 27 · 0/0 · 8 | 1/1 · 25 · 0/0 · 16 | 1/1 · 63 · 0 | 1/1 |
| MYNN | `phys_mynn_plume.py` | 2 | 2/2 · 64 · 0/0 · 8 | 2/2 · 63 · 0/0 · 16 | 2/2 · 69 · 0 | 2/2 |
| MYNN | `phys_mynn_tridiag.py` | 1 | 1/1 · 109 · 0/0 · 4 | 1/1 · 118 · 0/0 · 12 | 1/1 · 96 · 0 | 1/1 |
| Noah-MP | `phys_noahmp_columns.py` | 3 | 2/3 · 130 · 0/20 · 3 | 2/2 · 111 · 0/0 · 12 | 1/1 · 206 · 0 | 1/1 |
| Noah-MP | `phys_noahmp_radiation.py` | 1 | 1/1 · 80 · 0/0 · 6 | 1/1 · 96 · 0/0 · 16 | 1/1 · 64 · 0 | 1/1 |
| Thompson | `phys_thompson_full.py` | 1 | 1/1 · 223 · 0/76 · 2 | 1/1 · 189 · 0/2 · 8 | pending | pending |
| Thompson | `phys_thompson_prep.py` | 1 | 1/1 · 120 · 0/0 · 4 | 1/1 · 104 · 0/0 · 12 | 1/1 · 72 · 0 | 1/1 |
| Thompson | `phys_thompson_sedimentation.py` | 2 | 2/2 · 27 · 0/0 · 8 | 2/2 · 29 · 0/0 · 16 | 2/2 · 36 · 0 | 2/2 |
| RRTMG | `rad_lw_transfer.py` | 2 | 2/2 · 59 · 0/0 · 8 | 2/2 · 70 · 0/0 · 16 | 2/2 · 64 · 0 | 2/2 |
| RRTMG | `rad_mcica.py` | 1 | 1/1 · 58 · 0/0 · 8 | 1/1 · 82 · 0/0 · 16 | 1/1 · 56 · 0 | 1/1 |
| RRTMG | `rad_sw_band_sums.py` | 2 | 2/2 · 135 · 0/32 · 3 | 2/2 · 121 · 0/0 · 10 | 2/2 · 154 · 0 | 2/2 |
| RRTMG | `rad_sw_quadrature.py` | 1 | 1/1 · 25 · 0/0 · 8 | 1/1 · 20 · 0/0 · 16 | 1/1 · 38 · 0 | 1/1 |
| nest force-down | `nesting/sint_kernel.py` | 1 | 1/1 · 90 · 0/0 · 5 | 1/1 · 176 · 0/0 · 8 | 1/1 · 205 · 0 | 1/1 |

Totals (kernel sites PASS/compiled): gfx942: 57/58, gfx1100: 55/55, 75: 51/51, 61: 51/51

Non-PASS: `diagnostics/nwp.py` nwp_helicity_column gfx942 FAIL: RuntimeError: PassManager::run failed; `kernels/phys_noahmp_columns.py` b_core_noah_canopy_flux gfx942 TIMEOUT: ; `kernels/phys_qnse_columns.py` qnse_sfc_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `kernels/phys_qnse_columns.py` qnse_pbl_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `physics/ferrier/native.py` ferrier_mp5_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `physics/ferrier/native.py` ferrier_mp15_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'

Reading the census [M]:
- No completed release compile has a VGPR spill or scratch on either AMD target, so there is no A19-class register/stack cliff. Only SGPR spills occur, and those spill into VGPR lanes cheaply.
- Register pressure sits where it sits on CUDA. KF and Thompson-full use 243/223 VGPR on gfx942, i.e. 2 waves/SIMD; on NVIDIA KF uses 242 registers (E18).
- Two acoustic-family kernels reach 1 wave/SIMD on gfx942: the 3.5 MB unrolled `b_core_coef_fp32` (366 VGPR, 228 s compile) and the legacy `fused_vertical_implicit` advance_w (384 VGPR). Both are first targets for the M3 geometry sweep. Halving `num_warps` on wave64, or blocking the unrolled level loop, are the obvious knobs. Their gfx1100 compiles are still running at v1 time.
- RDNA (gfx1100, wave32) needs more VGPRs per lane but keeps 8–16 waves.
- sm_75 peaks at 245 registers without spills.
- Thompson-full on gfx942 carries 1,730 f64 instructions out of 22,772: these are the DOUBLE islands, as expected.
- AMD compile time is a cold-start (R3) item: Thompson-full takes 109 s on contended CPU, and the large Noah/acoustic kernels take longer (see the pending list).

Caveats:
- Compile success is not run success. Lowered shapes are test shapes; block sizes, and hence register use, are shape-independent for these kernels.
- Occupancy is the compiler's static estimate in waves per SIMD (maximum 8 on gfx942, 16 on gfx1100).
- The kernel bodies are exactly the ones a ROCm process traces today: `cuda_native()` is False there too.

## 2. Options with evidence

**(a) Pallas-Triton native on ROCm: CHOSEN as the v0.5.0 route.**
- Evidence: [M] the JAX 0.10 ROCm lowering branch exists, the W5P01 census results, and the hip `gtsv` and OCML tables.
- [W] The jax-rocm7-plugin 0.10.0 is the same JAX version as ours. RDNA3/4 and CDNA2/3/4 are official targets.
- [W] Upstream JAX CI covers only gfx942/gfx950; RDNA and MI200 are tested in AMD's downstream CI. The marin project runs one Pallas-Triton source on MI300X, MI350X, H100 and GB200 (2026-10).
- Two launch variants exist:
  - A1: in-graph `__gpu$xla.gpu.triton`, with XLA-ROCm's bundled Triton. It is zero code and what AMD tests.
  - A2: stock Triton + `triton_kernel_call`. It exists for CUDA, and jaxlib binds the ROCm plugin's `_triton` module for it (`jaxlib/gpu_triton.py:20,36`). The W5P01 census compiles exactly this route's input (TTIR → stock Triton 3.6). For A1 it is a strong proxy only, because XLA bundles another Triton commit.
- Start with A1. Keep A2 as a per-kernel fallback, the CAM-MG precedent: XLA-bundled Triton miscompiled a kernel there that stock Triton compiled correctly.

**(b) Pallas Mosaic GPU: REJECTED.**
- [W] It is NVIDIA-only (Hopper `wgmma`, Blackwell `tcgen05`) and its API breaks each release; `pallas_call()` support was removed in 0.11.2.
- It has no AMD backend, and sm_120 support is [U]. Our kernels are not matmul-shaped, so its tensor-core strength does not apply.

**(c) Pure JAX/XLA per family: REJECTED as a performance route; it stays as the CPU/TPU route.**

The measured cost of losing the fused kernels on CUDA:

| Family | Fused kernel vs legacy XLA/lax path |
|---|---|
| RRTMG LW | 9.1× (C20) |
| KF | 273× (C21) |
| MYNN plume | 50× (C18) |
| Thompson | ≈ 10× (C24) |
| `advance_w` | 2× (C1) |
| Whole ordinary root | flags OFF 0.786 s vs ON 0.483 s on the LW6-era tree (A24) |

- The legacy XLA paths are also not WRF-faithful for acoustic and PD (B13; `docs/portability.md`). Option (c) would therefore mean maintaining a second, worse implementation.

**(d) Thin kernel DSL / code generator: ADOPTED in its minimal form.**
- Our de-facto DSL already exists: `jnp` inside Pallas, lowered jaxpr→TTIR.
- The minimal adoption is to own that lowering in M6, and to add `vendor_ops`.
  - Vendor the pinned `jax/_src/pallas/triton/lowering.py` + `primitives.py` (3,627 lines, Apache-2) together with our ~10 custom lowering rules.
  - It builds through `jaxlib.triton.dialect`, the generated MLIR bindings (3.9k lines). If jaxlib drops them with the backend, the vendored emitter switches to generic-form MLIR ops, which stock Triton parses.
- JAX's own suggestion, rewriting every kernel in the Triton Python DSL behind `jax_triton`, would mean about 18k kernel lines plus the reused physics code (≈ 40–60 LD) and a full re-qualification. Rejected.
- A new DSL that emits HIP/SYCL (GT4Py/DaCe, Kokkos) would mean rewriting about 15k kernel lines and re-qualifying every WRF oracle. GT4Py was already excluded in M2 (ADR-001: Python 3.13 break).
- Kokkos had the "highest agent iteration friction" in the M2 bakeoff (`.agent/decisions/ADR-001-backend-selection.md:65`).
- For Intel GPUs, the only GPU target that stock Triton lacks, add the Intel XPU Triton backend later instead of a second DSL.

**(e) Per-vendor hand kernels (HIP C++) for the top families: DEFERRED, evidence-gated.**
- Allowed only if, on hardware, a family reaches less than 50 % of the roofline fraction it reaches on CUDA after the M3 tuning.
- Allowed for at most 2 families (candidates: acoustic w/mass, RRTMG band sums).
- Cost: a second source per family plus a per-vendor oracle re-run, about 3–5 lane-days each.

**(f) Older NVIDIA (sm_61/70/75).**
- [W] JAX CUDA-12 wheels support sm_52+; CUDA-13 wheels support sm_75+.
- [W] Triton officially supports cc 8.0+; PyTorch guards at ≥ 7.0.
- [M] W5P01 results: PTX from the same TTIR assembles with `ptxas` for sm_75 51/51 and sm_61 51/51 (max 245 registers on sm_75, spills only in the legacy `fused_vertical_implicit` kernel).
- Route: the same kernels with the CUDA-12 JAX wheel. There is no "XLA-only" fast route, see (c).
- Pascal sm_61 stays best effort: Triton does not support it, and the census shows that ptxas accepts all 51 completed kernels for sm_61; the Triton runtime launch on Pascal remains [U] (one vast.ai 1080 Ti smoke in M5 decides).

### 2.1 Precision traps and the gate policy

- **FMA contraction (E84/E111/E123).** [M W5P02] Triton contracts `a*b+c` to `v_fma_f32` on gfx942/gfx1100, just as on CUDA. Every place where WRF-literal REAL rounding matters already uses explicit RN ops on CUDA. Those sites need `vendor_ops.rn`, and AMDGCN `v_mul/v_add/v_sub_f32` inline asm compiles and stays uncontracted. Per-kernel `enable_fp_fusion=False` is an A2-only alternative.
- **Division.** [M W5P01/W5P02] On AMD, f32 `/` compiles to the IEEE-correct `v_div_scale/fmas/fixup` sequence. On CUDA, XLA and Pallas emit `div.full.f32` (≤ 2 ulp, E84), while WRF Fortran is IEEE. AMD is therefore *closer* to WRF here and ULP-different from CUDA. Accept this and record it.
- **Subnormals.** [M] The hsaco kernels keep f32 denormals (`.amdhsa_float_denorm_mode_32 3`), as CUDA does with no ftz. CPU flushes them (E117). This is fine because CPU-WRF reference data contains subnormal hydrometeors.
- **Math library.** OCML and libdevice differ by about 1–2 ulp in `exp/log/pow`. This is D6-irrelevant but makes cross-vendor bitwise comparison impossible.
- **fp64 DOUBLE islands.** These include Thompson, CLWRF REAL(8) and the boundary `rdt`.
  - RDNA3/4 have fp64 at about 1:32 (RX 7900 XTX ≈ 1.5–1.9 TF), comparable to the RTX 5090 at 1:64 with 1.6 TF. There is no new cost there.
  - CDNA is 1:1 or 1:2.
  - Arc A-series (no fp64 hardware), Apple Metal and TPU (no f64) cannot run the islands natively. Emulation (double-float) would be a fidelity project of its own, hence the Assessment tier.
- **Integer-64.** KISS McICA and the stochastic RNG are exact-integer streams and must be **bitwise** on every vendor. That is an exact gate: [M] the census reports the KISS lane kernel compiles on gfx942 (1/1 PASS, ≤ 58 VGPR, occ ≥ 8, VGPR spills 0) and gfx1100 (1/1 PASS, ≤ 82 VGPR, occ ≥ 16, VGPR spills 0); the stochastic RNG (option) compiles on gfx942; bitwise stream equality is a hardware gate in M4.
- **Determinism.** Our release kernels use no atomics (static scan: atomic count 0 outside the NTU/Jensen options). XLA-ROCm reductions may use atomics, so run-to-run byte identity must be re-proven per vendor: two warm runs from one compiled cache (E99). `--xla_gpu_deterministic_ops` is the fallback.
- **Gate policy (proposal, Owner decision D2).**
  - Fidelity is judged per vendor against original CPU-WRF and pristine-WRF oracles at the *frozen* tolerances: D6 manifest, frozen component oracles, F2 census.
  - Bitwise is required only (i) within one (vendor, arch, toolchain) tuple for determinism and AOT identity, and (ii) for exact-integer kernels (KISS).
  - Never bitwise CUDA ↔ ROCm. Any GPU-vs-GPU comparison is labelled regression.

## 3. Per kernel family: verdict, effort, expected speed, validation

Effort is in lane-days (LD) beyond the shared infrastructure of §4.2. Speed is the [I] device-time ratio vs CUDA on the same class of GPU: memory-bound kernels × the bandwidth ratio × a Triton-AMD efficiency of 0.6–0.9, from §1.4 occupancy and the literature [W].

Column legend:
- **t ratio** = t(AMD)/t(RTX 5090) for the family's device time [I]. A value below 1 is faster.
- **Census** = the gfx942 W5P01 result. Occupancy is in waves/SIMD; spills are SGPR spills, VGPR spills were 0 everywhere.
- **LD** is the family's share of the M2–M4 milestone effort, not extra work on top of it.

| Family (share) | Modules | Census (gfx942) | Verdict | LD | t ratio MI300X / RX 7900 XTX | Device validation (frozen tolerances) |
|---|---|---|---|---:|---|---|
| Acoustic (29.7 %) | `dyn_acoustic_fp32` (7 kernels) | 7/7 PASS, ≤ 366 VGPR, occ ≥ 1, VGPR spills 0, SGPR spills 14 | reuse as-is; geometry retune; `coef` TTIR is 3.5 MB (unrolled), so watch cold-compile time | 0.5 | 0.6–1.0 / 1.8–2.6 (latency-bound w/mass loops, A54) | D7 pristine acoustic oracle; BC27 PROD shapes; mass guards 0 |
| MYNN (12.2 %) | `phys_mynn_plume/tridiag/cloudmix/boulac`, `fused_edmf_plume` | 6/6 PASS, ≤ 159 VGPR, occ ≥ 3, VGPR spills 0, SGPR spills 2 | reuse as-is | 0.5 | 0.5–0.9 / 1.6–2.4 | 48 WRF mean checks (4 P0 sets); BP56 tridiag2 bitwise vs literal WRF REAL *per vendor* |
| Advection (10 %) | `dyn_flux_fp32`, `dyn_momflux_fp32` | 6/6 PASS, ≤ 62 VGPR, occ ≥ 8, VGPR spills 0 | reuse as-is | 0.25 | 0.4–0.6 / 1.9–2.4 (bandwidth-bound) | momentum 12/12; pristine REAL4 operators |
| XLA glue/layout/DtoD (≈ 18 %) | XLA-generated | n/a | XLA-ROCm as is; whole-program hsaco gate (R5) | 0.5 | 0.5–1.2 / 1.8–3.0 | D6 + output integrity |
| PD (6.3 %) | `dyn_pd_fp32` | 4/4 PASS, ≤ 90 VGPR, occ ≥ 5, VGPR spills 0, SGPR spills 2 | reuse as-is | 0.25 | 0.4–0.6 / 1.9–2.4 | PD operator gates |
| Lateral boundary (5.2 %) | `dyn_ring_relax_fp32`, `dyn_boundary_fp32`, `boundary_apply` | 4/4 PASS, ≤ 36 VGPR, occ ≥ 8, VGPR spills 0; `boundary_apply` scalar couple + `dyn_ring_relax`: LOWER_FAIL (reduce_precision) | **small change**: 2 kernels need `vendor_ops.rn`, because their non-CUDA RN branch does not lower | 1.0 | 0.4–0.7 / 1.9–2.6 | B36 ring tests; NumPy scalar-mass gate; root-boundary D6 rings |
| RK/EOS (3.8 %) | `dyn_rk_fp32`, `dyn_uvtend_fp32`, `dyn_rhsph_fp32`, `dyn_omega_fp32` | 9/9 PASS, ≤ 96 VGPR, occ ≥ 5, VGPR spills 0 | **small change** (`dyn_rk` PTX rn → `vendor_ops`) | 0.5 | 0.4–0.7 / 1.9–2.6 | RK/EOS 222 checks; rhs_ph E133 association gate |
| Thompson (3.8 %) | `phys_thompson_full/prep/sedimentation` | 4/4 PASS, ≤ 223 VGPR, occ ≥ 2, VGPR spills 0, SGPR spills 76 | reuse as-is; `tt.gather` compiles; DOUBLE islands stay f64 | 0.5 | 0.7–1.3 / 1.2–2.5 (register-bound column; f64 islands at 1:32 on RDNA) | gate-2/L3a/L3b 30/30; mixed-phase pristine; adversarial ice/graupel columns (E154) |
| Diffusion (3.4 %) | `dyn_diff_fp32` | 2/2 PASS, ≤ 36 VGPR, occ ≥ 8, VGPR spills 0 | reuse as-is | 0.25 | 0.4–0.6 / 1.9–2.4 | 246 REAL field checks (BD09–13) |
| Noah-MP / GWDO / coupling (3.0 %) | `phys_noahmp_columns/radiation`, `phys_gwdo_column`, `phys_hydrostatic_real` | 5/6 PASS, ≤ 130 VGPR, occ ≥ 3, VGPR spills 0, SGPR spills 20 | reuse as-is | 0.5 | 0.7–1.3 / 1.5–2.5 | Noah energy/phenology 11/11 + REAL water rule; GWDO pristine |
| Nest force-down (0.9 %) | `nesting/sint_kernel` | 1/1 PASS, ≤ 90 VGPR, occ ≥ 5, VGPR spills 0 | **small change** (PTX rn/div mode → `vendor_ops`) | 0.5 | 0.5–1.0 / 1.5–2.5 | SINT exact host-transcription gate (E121) |
| RRTMG (radiation root, 45 of 82 ms) | `rad_lw_transfer`, `rad_sw_band_sums`, `rad_sw_quadrature`, `rad_mcica` | 6/6 PASS, ≤ 135 VGPR, occ ≥ 3, VGPR spills 0, SGPR spills 32 | reuse; `rad_mcica` takes the CUDA (Triton-native) code shape on AMD too (KISS kernel + lane clamp, §1.2 row 1) | 0.5 | 0.5–1.0 / 1.5–2.5 | LW/SW 18 + clear-sky 4 WRF oracles; KISS stream **bitwise** |
| KF (d01 only) | `phys_kf_column` | 1/1 PASS, ≤ 243 VGPR, occ ≥ 2, VGPR spills 0, SGPR spills 91 | reuse; highest register pressure, so the geometry sweep matters | 0.5 | 0.8–1.5 / 1.5–3.0 | 5 frozen + 16,800 real + 8,400 deep-active WRF columns |
| Option schemes (19 modules) | CAM-MG/ZM, P3, NTU, Jensen, QNSE, SMS, MSKF, KF-CuP, CuPlus, Ferrier, Goddard, Milbrandt, diagnostics, RNG | captured 9 of 19 modules; gfx942 3/8 PASS; PTX-asm kernels (Ferrier, QNSE, typed memory) CRASH 'constraint f'; NTU erf extern LOWER_FAIL (cuda-only table); `nwp_helicity_column` AMD PassManager FAIL | small change for the PTX typed-memory/extern users; reuse for the rest | 2.0 | n/a | existing component gates on device; opt-in experimental tier (as on CUDA) |


### 3.1 Validation design (no GPU-vs-GPU truth)

1. **CPU pre-screen, every commit, no hardware.**
   - Interpreter parity (the existing 120 tests).
   - The W5P01 compile census for gfx942/gfx1100/gfx1201 + sm_75/sm_80/sm_120 as a CI gate: lowering, compile, spills = 0 on release kernels, scratch, VGPR occupancy floor. This is the AMD E41 analogue.
   - Runtime: 1–2 h on lane cores today; [I] ≈ 10 min with a TTIR cache keyed by sha.
2. **Component fidelity on the device.** The frozen pristine-WRF oracles (acoustic D7, Thompson gate-2/L3/mixed-phase, MYNN 48 mean checks, KF 5 + 16,800 columns, RRTMG LW/SW + clear-sky, Noah energy/water, GWDO, SINT) run with `JAX_PLATFORMS=rocm` at the frozen tolerances. KISS streams must be bitwise.
   - This needs one device-test switch, because `tests/v025/conftest.py` forces CPU: add a `GPUWRF_DEVICE_TESTS=<platform>` opt-in that asserts the platform/`device_kind` in-process (E13).
3. **Forecast fidelity vs CPU-WRF.**
   - PROD 24 h (D5 reference) and the three WN3 gate cases 0227/0502/0614, 24 h (D15 references; the inputs must be copied to the cloud host).
   - Swiss 24 h (D30, the shipped example).
   - Each is scored with the frozen D6 manifest (all 375 fields compared, 62 bounded), output integrity, the F2 census (0 nonfinite, strict-guard arm) and the station scorer for 0227.
4. **Determinism and performance.**
   - Determinism: two warm runs per vendor byte-identical.
   - Speed: the S1 proxy (3 h, segments 2–3) and the 24 h whole run, with `rocprofv3` kernel traces mapped to families (the BP83 rules).
   - Memory and throughput: R1 VRAM vs CPU RSS, and the multi-case throughput (D28 mode, N cases per GPU).

### 3.2 Cloud qualification plan and cost (no owned AMD hardware)

| Hardware | Provider / price [W] | Hours | Cost |
|---|---|---:|---:|
| MI300X ×1 (gfx942) | DigitalOcean/AMD Dev Cloud $2.59/h, RunPod $2.39 (often out of stock), median $4.42 | 50 | $130–220 |
| RX 7900 XTX (gfx1100) | DataCrunch ≈ $0.39/h | 50 | ≈ $20 |
| MI250X (gfx90a) | Cirrascale $1.28–1.60/h | 10 | ≈ $16 |
| T4 (sm_75), V100 (sm_70), 1080 Ti (sm_61) | vast.ai/RunPod $0.14–0.25/h | 10 each | ≈ $8 |
| Intel Max 1550 | Intel Tiber cloud preview (free, reservation) | assessment | 0 |

- **Total ≈ $250–450.**
- RDNA4 (gfx1201) has no cloud offer [W]. Buying one RX 9070 XT (≈ €650) or RX 7900 XTX (≈ €900) for the local CI box is optional (D4).
- Rules for cloud arms:
  - pinned container `rocm/jax:rocm7.14.1-jax0.10.0`;
  - receipts must record `jax.devices()[0].platform`, `device_kind` and the ROCm version;
  - never mix ROCm and CUDA caches;
  - inputs are uploaded as sha-checked bundles (`release_bundles` tooling), and the outputs that come back are scored on our CPU.

## 4. Recommended architecture, milestones, risks, decisions

### 4.1 Architecture: "one kernel source, native on every Triton GPU"

```
kernel bodies (jnp in Pallas, unchanged) ──► vendor_ops (rn, extern, gather, typed mem, geometry)
        │
        ▼  capability table: device → {cuda, rocm} native | {cpu, tpu, unknown} interpret
jaxpr ──► TTIR  (JAX lowering today; vendored pinned copy in M6)
        │
        ├─ A1 in-graph __gpu$xla.gpu.triton (XLA-bundled Triton; CUDA default, ROCm default)
        └─ A2 stock Triton (nvidia | amd | later intel-xpu) + triton_kernel_call / jax.ffi (per-kernel opt-in; all kernels after M6)
XLA glue/fusions stay XLA (portable); per-arch geometry table; hsaco/cubin static gate in CI
```

Principles:
- One source.
- Vendor differences live in one module; no more `if cuda_native()` scattered across 20 sites.
- The CUDA program stays byte-identical until the Owner decides otherwise: the M6 switch of CUDA to A2 is a separate, D6-requalified change.

### 4.2 Milestones (v0.5.0 central feature)

| M | Content | LD | Gate |
|---|---|---:|---|
| M0 | Productize W5P01 as `scripts/portability/compile_census.py` + CI test (TTIR cache, hsaco/cubin gate); fold W5P02 | 1.5 | census green for all release kernels on gfx942/gfx1100/sm_75/sm_120 |
| M1 | Capability table (native on rocm); runtime: memory fraction from compiled stats, PJRT `memory_stats`, cache tag from `device_kind`, command-buffer probe; rocm extras in installer/doctor | 3 | MI300X: PROD 3 h runs, finite, platform asserted |
| M2 | `vendor_ops` (rn/extern/gather/typed-mem) + replace the release-path `cuda_native()` bodies (boundary_apply, dyn_rk, sint, rad_mcica) + option-scheme externs | 3 | CPU census + interpreter parity; CUDA raw HLO / jaxpr identical (E151) |
| M3 | Per-arch geometry table (block, num_warps, num_stages) from census occupancy + one on-device sweep (MI300X, RX 7900 XTX) | 3 | family device time vs the §3 expectation |
| M4 | ROCm qualification: device oracles, PROD + 3 WN3 + Swiss 24 h D6, F2, determinism, R1, perf receipts | 5 | §3.1 levels 2–4 green |
| M5 | Older NVIDIA: CUDA-12 wheel env, sm_70/75 smoke + one 24 h D6 (T4 or V100); sm_61 probe | 2 | Best-effort tier documented |
| M6 | Deprecation insurance: vendored jaxpr→TTIR lowering, A2 for all kernels on ROCm; CUDA A2 only behind a flag | 6–10 | ROCm D6 re-run; CUDA unchanged by default |
| M7 | Docs, support matrix, `gpuwrf doctor` vendor checks | 1 | doc review |

- v0.5.0 core (M0–M5, M7) is 18.5 LD; it can run as 2–3 parallel lanes in ≈ 7–9 working days.
- M6 can ship as v0.5.1 if needed, but it must land before any JAX ≥ 0.11 upgrade.

### 4.3 Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | [W] Pallas-Triton deprecation (JAX 0.11) | pin 0.10.x; M6 |
| R2 | XLA-ROCm bundled Triton miscompiles or hangs (cf. CAM-MG sm_120, E157 Coalesce hang) | A2 per kernel; device oracles |
| R3 | No CONDITIONAL/WHILE command buffers on ROCm: host-driven loops (A37: 13 DtoH/root) and a bigger host gap | measure in M1. On CUDA the unprofiled ordinary root is ≈ 46–55 ms (BP72/v0.4.0 RC) against ≈ 33 ms device busy, so the host/launch gap is already ≈ 13–22 ms (nsys inflates it, E185). On ROCm graphs matter even more |
| R4 | AOT readiness (R3 gate) without compiled thunks | [U] measure the cached load on ROCm; the persistent JAX cache fallback works |
| R5 | XLA fusion/layout decisions differ on ROCm (A19 recompute cliff, A40/A47 layout tax) | hsaco static gate on the whole-program cache (E41 analogue) + rocprof family table |
| R6 | RDNA is less tested upstream; W7900 command-buffer segfault (known AMD issue) [W] | RX 7900 XTX arm in M4; command-buffer probe |
| R7 | Wave64 register pressure in column kernels (Thompson/KF/Noah) | census VGPR/spill → geometry table |
| R8 | Cloud availability (MI300X often sold out) | book 2 providers; the arms are short and resumable |
| R9 | Maintenance: every kernel change must stay portable | census in CI: a CPU minute per kernel, no hardware |

### 4.4 Owner decisions

- **D1 Tiers.**
  - Supported: NVIDIA sm_80+ and AMD gfx942/gfx950 + gfx1100/1101/1200/1201.
  - Best effort: gfx90a, sm_70/sm_75.
  - Assessment only: Pascal, Intel (`jax[oneapi]` has no Pallas; Arc A has no fp64), Apple (no maintained JAX GPU, no f64), TPU (interpreter route).
- **D2 Gate policy.** Per-vendor fidelity vs CPU-WRF at frozen tolerances; bitwise only within a vendor tuple and for integer kernels; never cross-vendor (§2.1).
- **D3 Rounding.** On AMD, implement WRF-literal RN ops with AMDGCN asm (exact analogue of our PTX), and accept AMD's IEEE division as WRF-closer than CUDA.
- **D4 Budget.** Cloud ≈ $250–450, plus optionally one consumer AMD card for CI (≈ €650–900; needs a slot and a PSU in a host).
- **D5 JAX pin.** Stay on 0.10.x through v0.5.0; M6 is a precondition for any upgrade.
- **D6 Hand kernels.** Allowed only under the (e) rule: after M3, < 50 % of the CUDA roofline fraction; max 2 families.
