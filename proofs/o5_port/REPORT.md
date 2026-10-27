# o5-port — v0.5.0 portability groundwork (M0 + M2a minimal), decision-independent

Lane o5-port · Opus 5.5 (claude_2) · 2026-10-24 · branch `lane/o5-port`, base main `3fb0622d7b` (pre-topo).
No GPU was used. Every result below is CPU code generation or source/binary inspection; nothing ran on an
NVIDIA or AMD device in this lane. Labels: [M] measured/read, [I] inferred, [U] unknown.
Valid for every plausible answer to Owner decisions D1–D6 (nothing here selects a tier, a budget or a hand kernel).

## 0. Results in one screen

1. **`vendor_ops` facade (M2a minimal)** — AMDGCN literal-RN ops, admitted-ROCm native dispatch
   (`GPUWRF_ROCM_TRITON_NATIVE=1`), per-kernel interpreter fallback (`GPUWRF_ROCM_INTERPRET_KERNELS`).
   **CUDA program byte-identical** [M]: Swiss + PROD d01 canonical and raw HLO, production convention and the
   nested d01→d02 path (force-down + child step), every Pallas kernel incl. source locations. Ready for a critic.
2. **Compile matrix, production bodies, real shapes** [M CPU]: product route A1 (XLA deviceless) compiles the
   whole PROD d01 step for sm_80/86/89/90/120 and for **gfx942 (MI300X)**, gfx90a, gfx1100 (§3.3); XLA refuses
   sm_75 (cc < 8.0) and CUDA-13 ptxas refuses sm_61/70. Stock Triton (A2) compiles all 61 bodies for sm_70…120,
   but is not a proxy for A1 (coef kernel 255 regs + spill vs 40–48 regs).
3. **ROCm control flow quantified before any rental** [M strings, I static]: the pinned ROCm binaries cannot put
   conditionals/whiles into HIP graphs; one PROD d01 step contains 93 conditional executions and 435 while
   iterations → ≈ 93 host round trips per d01 step on ROCm (§3.3, §4 G1).
4. **Deviceless XLA-ROCm on our CPU host** [M] with the exact AMD tuple (§2) — M1 Phase 0 runs here, for free.
5. **Side result:** the existing OFF-identity probes were blind to 4 MYNN/Noah Pallas kernels; corrected probe +
   test delivered (§6).


## 1. `vendor_ops` facade (M2a minimal) behind an exact OFF-identity gate

Commit `448d09028f` (`src/gpuwrf/backends/vendor_ops.py`, facade `backends/pallas.py`, 5 call sites).

| piece | CUDA (default) | admitted ROCm (`GPUWRF_ROCM_TRITON_NATIVE=1`) | everything else (CPU, unadmitted ROCm, unknown) |
|---|---|---|---|
| `vendor()` / `triton_native()` | `cuda` / True | `rocm` / True | `interp` / False (+ one-time warning on ROCm) |
| `pallas_call` | original JAX callable | original JAX callable | interpreter (unchanged S1-port route) |
| literal REAL `+ - *` (`dyn_rk._rn` and its importers MYNN tridiag2, `dyn_boundary`, hydrostatic, NWP; ring relax; boundary scalar couple; SINT) | PTX `{op}.rn.f32` (untouched lines) | AMDGCN `v_{add,sub,mul}_f32` / `v_add_f64 …,-$2`, `v_mul_f64` inline asm | plain ops (interpreter) / f64+`reduce_precision` prescreen where it was before |
| literal `/` | PTX `div.rn` (SINT `div.full` option) | IEEE `/` (AMDGCN `v_div_scale/fmas/fixup`; Owner D3) | plain |
| `rad_mcica` KISS lane kernel + lane clamp | on | on | off (bitwise jump-ahead stream) |

Design rule (E109/E134): every CUDA emitter keeps its Python frame and line (edits are in-place,
line-count preserving; `git diff -U0` hunks are `-n,+n`), so the CUDA program is byte-identical by construction.
Option schemes with PTX typed memory / `__nv_*` externs (NTU, Jensen, P3, QNSE, Ferrier, Goddard, Milbrandt,
CAM-MG) gate on `cuda_native()` and stay interpreted on ROCm (per-scheme state `interp`).

**OFF identity [M, CPU, real cases, `tools/capture_real.py` + `tools/compare_identity.py`]:** base snapshot vs
candidate snapshot behind ONE import alias (E58/E94):

| case | canonical jaxpr (release identity) | raw CUDA-target HLO | Pallas kernels: TTIR / TTIR+locations / files | verdict |
|---|---|---|---|---|
| Swiss d01 42×42×44 | b977712f = b977712f | 3641e8be = 3641e8be (22,468,110 B, file bytes equal) | 56/56 equal (launch params via raw HLO) | **IDENTICAL** |
| PROD d01 120×70×44 | 321aa651 = 321aa651 | 269260d3 = 269260d3 (21,905,496 B, file bytes equal) | 55/55 equal incl. num_warps/num_stages | **IDENTICAL** |
| PROD d01, production convention | (not traced) | c7183eff = c7183eff (20,824,167 B) | 61/61 equal | **IDENTICAL** |
| Swiss d01, production convention | (not traced) | 09ca2808 = 09ca2808 (21,386,725 B) | 62/62 equal | **IDENTICAL** |
| PROD nested d01→d02, production convention: force-down (SINT + scalar couple) / d02 child step on the force-down output | (not traced) | e37821d9 = e37821d9 / ab837481 = ab837481 (+ d01) | 123/123 equal (all three programs) | **IDENTICAL** |

**Schema-2 re-proof (critic F1: constant values) [M CPU]:** `scripts/off_identity_probe.py` native-first, base tree vs
final candidate — PROD nested (root d01 + production force-down + d02 child step; 393/48/375 previously elided
constants now hashed) and Swiss root: full-constant HLO, legacy HLO, per-name Pallas counts, every kernel with and
without source lines, canonical jaxpr and its constant values — **IDENTICAL** (`proofs/o5_port/identity/IDv2_*.json`).

**Against current main** (lane HEAD `741288143f` = main `b14a9801cb` + facade, merged; probe run on the wz1080 CPU per
the 21:34Z RAM rule): PROD nested 22/22 and Swiss 10/10 checks **IDENTICAL** (`identity/vs_main_b14a9801cb/`).
Pre-merge re-run (manager 01:12Z; main moved to `f86d44d3c5`: H-CB, registry, o5-coast motion of `boundary_apply.py`):
lane `da16d568b6` = facade + merged `f86d44d3c5` vs `f86d44d3c5`, same probe and box, both trees verified file-by-file
against `git ls-tree` — PROD nested 22/22 and Swiss 10/10 **IDENTICAL**, incl. `kernels_ttir_sha16` and
`kernels_ttir_loc_sha16`; all probes COMPLETE, 161/159 Pallas calls native, 0 interpreted. CPU tests 65 passed, 2 heavy
probe tests skipped (covered by this run) (`identity/vs_main_f86d44d3c5/`).

The table holds for the first candidate `448d09028f` and again for the final candidate `d13aee96b5` (adds the
per-kernel ROCm interpreter fallback). *Production convention* = additionally `platform.runtime_backend()` →
`cuda`, i.e. the program the GPU really runs: the established probe convention misses 6 kernel bodies on PROD d01
(McICA KISS ×2, MYNN tridiag2/plume/cloudmix, Noah snowwater; 141 vs 157 lowerings), two of which this lane touches.
The base tree for the production rows is a hard-linked copy of the candidate snapshot with the 7 changed files
restored from `3fb0622d7b`, verified file-by-file against `git ls-tree` (1667 files, 0 mismatches).

The base raw-HLO SHAs equal the w44-params W304 receipts (independent tool, other alias path), so the tool
reproduces the established identity exactly. Receipts: `<USER_HOME>/wrf_gpu2_lanes/o5-port/ID_{swiss,prod}{,_final}.json`, `ID_{prod,swiss}_production_final.json`, `base_hl_provenance.json`.

Tests [M CPU]: `tests/test_vendor_ops.py` 11 PASS (selection on CPU/CUDA/unadmitted/admitted ROCm; ROCm TTIR of
the real `_rn` chain = AMDGCN asm with no `arith.mulf/addf` (deletion-sensitive); CUDA TTIR still PTX RN; f64
spellings; ring relax + scalar couple now lower for ROCm without f64 (W5P01 failures); stock-Triton gfx942
compile: literal chain 0 FMA, plain control ≥ 1 FMA). 12 seam test files on the candidate (sint 52, portability
kernels 49, mcica 14+32+3, mynn tridiag REAL 5, ring/b36, EOS, record windows, init cache): 0 failures.

## 2. Exact toolchain tuples (pin candidates for D5) [M]

| tuple | component | version | fingerprint (sha256 prefix) |
|---|---|---|---|
| CUDA (today's product) | jax / jaxlib | 0.10.0 / 0.10.0 | RECORD bb072cc2 / dd5315ab |
| | jax-cuda13-plugin / -pjrt | 0.10.0 / 0.10.0 | RECORD 3560bbc6 / e6550a12; `xla_cuda_plugin.so` a98368eb (388,136,440 B) |
| | XLA's assembler (pip nvidia/cu13 `ptxas`) | 13.2.78 | a0212ca1 — **rejects `sm_61` and `sm_70`** ("not defined for option gpu-name") [M] |
| | stock Triton (A2 route, CAM-MG) | 3.6.0+git9844da95 | RECORD f6bd2aee; bundled `ptxas` 12.8.93 c960a4f2 (accepts sm_61/70/75/…/120) |
| ROCm (candidate, AMD index `repo.amd.com/rocm/whl-multi-arch`) | jax-rocm7-plugin / -pjrt | 0.10.0+rocm7.14.0 | RECORD 0de00079 / f0e9912f; `xla_rocm_plugin.so` eff57c95 (511,532,480 B) |
| | TheRock SDK wheels | rocm 7.14.0, rocm-sdk-core 7.14.0, rocm-sdk-libraries 7.14.0 | RECORD 55fff7ae / 91360a35 / b6ba5633 |
| ROCm (PyPI build, NOT the AMD-index one) | jax-rocm7-plugin / -pjrt 0.10.0 | ROCm/xla `a4814ea3`, rocm-jax `29ee4dbc` | wheels 3793b64d / fe0a4055; needs the `[with-rocm]` amd_* wheels that are on no public index we found |

Consequences [M→I]: (a) pre-Turing NVIDIA (Volta sm_70, Pascal sm_61) cannot use today's CUDA-13 tuple at all
(XLA's ptxas refuses the arch before the cc ≥ 8.0 Triton refusal even matters) → those tiers need a separate
CUDA-12 tuple (`jax-cuda12-*`) **and** the A2 route; (b) the ROCm tuple must be the AMD-index build
(`+rocm7.14.0` local version) because its SDK wheels are installable without `/opt/rocm`; (c) [M] JAX 0.11.2 and
0.12.0 (PyPI) still ship `jax/_src/pallas/triton/*` (6 files) — deprecated but present, so the pin buys time,
it does not remove the M6 risk.

## 4. Exact ROCm gaps (status after this lane)

| # | gap | evidence | status / owner of the fix |
|---|---|---|---|
| G1 | Device control flow: ROCm command buffers cannot hold CONDITIONAL/WHILE nodes | [M] both 0.10.0 ROCm plugin binaries (PyPI ROCm/xla a4814ea3, AMD-index +rocm7.14.0) contain "Graph conditionals are not yet supported on HIP graphs." and "Conditionals are not supported on ROCM."; product injects `+CONDITIONAL,+WHILE` only for CUDA (`xla_autotune._gpu_is_target`) | open — M1 gate; static per-step count of host round trips from the deviceless ROCm executable in §3.3 |
| G2 | Pallas dispatch on ROCm was interpreter-only | `backends/pallas.py` | **closed (opt-in)**: `GPUWRF_ROCM_TRITON_NATIVE=1` returns the native callable; default stays interpreted + warning until a tuple is qualified |
| G3 | WRF-literal RN ops were PTX-only; ring relax + boundary scalar couple did not lower off CUDA (f64 `reduce_precision`) | W5P01; `tests/test_vendor_ops.py` | **closed in code generation**: AMDGCN `v_*_f32/f64` asm; both kernels lower for ROCm with 0 f64 tokens [M CPU]. Numerical contract on AMD hardware = M1 archetype oracles |
| G4 | Division semantics: CUDA `div.rn` (and SINT/XLA `div.full`) vs AMD IEEE sequence | W5P02, vendor_ops | D3 decision; component oracles on hardware |
| G5 | Option schemes: PTX typed memory (NTU/Jensen/P3/QNSE), `__nv_*` externs (NTU erf, Milbrandt), Ferrier/Goddard asm | grep of `elementwise_inline_asm`/`__nv_` sites | stay **interpreted** on ROCm (gated by `cuda_native()`); per-scheme state `interp` until ported (M2c) |
| G6 | Legacy cuSPARSE gtsv paths | `physics/tridiagonal_solver.py`, `dynamics/vertical_implicit_solver.py` | ROCm takes the pure-JAX Thomas scan (release path uses Pallas tridiag2) |
| G7 | Allocator/VRAM policy (`cuda_async`, C-auto via `nvidia-smi`) | `gpu_allocator.py` | ROCm gets JAX BFC defaults (75 % prealloc) → R1 needs a ROCm memory fraction from `get_compiled_memory_stats()` (E34) |
| G8 | AOT compiled thunks / slim module / E41 cubin gate are CUDA-only | `aot_executable.py:387` returns None off CUDA | ROCm readiness = JAX persistent cache only → M1 must measure fresh-process readiness (R3 ≤ 60 s); AMD static gate proposal: `tools/amdgpu_elfs.py` (VGPR/SGPR spills, scratch from the executing code objects; needs manager approval) |
| G9 | `nvidia-smi`-based cache tag / preflight | `compile_cache.py`, `gpu_preflight.py` | preflight is skipped on ROCm (tested); tag falls back to backend name — fine for one ROCm tuple per cache dir |
| G10 | `JAX_PLATFORMS=gpu` expands to [cuda, rocm] and a missing ROCm becomes fatal (w44-warmstart INBOX 02:12Z) | INBOX | ROCm launchers must pin `JAX_PLATFORMS=rocm` |
| G11 | Pre-Ampere NVIDIA: XLA Triton refuses cc < 8.0; CUDA-13 ptxas refuses sm_61/sm_70 | wz1080 receipt; [M] ptxas 13.2 | separate CUDA-12 tuple + A2 route (M2b/M5) |
| G12 | Pallas-Triton deprecation (JAX 0.11) | [M] still shipped in 0.11.2 and 0.12.0 | pin 0.10.0 tuples (§2); M6 before any upgrade |
| G13 | Release-identity probe ≠ production program | [M] P04 vs P05: the established probe convention (local facade native, backend-dark guards on CPU) omits 6 production kernel bodies (McICA KISS ×2, MYNN tridiag2/plume/cloudmix, Noah snowwater: 141 vs 157 Pallas lowerings on PROD d01) | this lane proves OFF identity in BOTH conventions (§1); recommend the manager add the production convention to future OFF gates |

## 5. Costed M1 plan — MI300X go/no-go (once AMD credits arrive)

What moved OFF the rental by this lane [M]: the exact ROCm tuple installs without `/opt/rocm` (§2), the whole
production program can be compiled for gfx942 on our CPU with XLA-ROCm's own Triton (§3.3), the literal-RN
spelling exists and lowers (§1), and the static host-round-trip count per step is computable (§3.3). The rental
therefore starts with a pre-built, hash-pinned recipe and spends GPU hours only on execution evidence.

**Phase 0 — here, no rental (≈0.5 LD, CPU; PROD d01 part DONE tonight, §3.3):** re-run `tools/run_a1_rocm.sh` on the
final candidate for PROD nested + WN3-0227 (3 domains) shapes (the probe's `--nested` mode gives the child programs);
freeze the gfx942 VGPR/scratch table and the control-flow census (PROD d01: 93 conditionals + 435 while iterations per
step — the number Phase 3 must reproduce with rocprofv3 and budget); prepare the
device-runner scripts (asserted `jax.devices()[0].platform=='gpu'` + `device_kind` contains MI300, native
dispatch asserted by an EM_AMDGPU code object in the executable, `GPUWRF_ROCM_TRITON_NATIVE=1`,
`JAX_PLATFORMS=rocm` (G10)); package PROD + Swiss + WN3-0227 inputs (<2 GB) and the CPU-WRF reference stays home.

**Phase 1 — provision + smoke (GPU-h 1.5):** pinned image digest, venv from the §2 tuple (+ `rocm-sdk-device-gfx942`
7.14.0 for library kernels), `rocm-smi`/`rocprofv3` available (R8); `tests/test_portability.py`,
`tests/test_vendor_ops.py` on device; one 2-step Swiss d01 interpreted (S1-port route) as a functional baseline.

**Phase 2 — archetype oracles native (GPU-h 4):** frozen pristine-WRF oracles, unchanged tolerances, device runner:
`dyn_rk` EOS + stage masses (RN32 literal), boundary scalar couple + ring relax (B36 ring tests), MYNN tridiag2
**bitwise vs literal WRF REAL** (rule 3), acoustic w (D7), Thompson-full gate-2/L3a/L3b + adversarial ice/graupel
(E154), McICA KISS **bitwise** (integer stream), Noah-MP energy + REAL water rule, KF frozen + deep-active columns.
Every oracle fails closed on a non-native dispatch.

**Phase 3 — composed control flow + readiness + memory (GPU-h 5):** PROD d01 and Swiss 1 h native; rocprofv3
kernel + HIP-API trace of 1 ordinary + 1 radiation + 1 KF root: launches, DtoH/HtoD, `hipStreamSynchronize`,
allocations per step vs the static census (§3.3); fresh-process cached readiness (JAX persistent cache; no AOT
thunks on ROCm, G8); peak memory (`memory_stats`) with a fixed MEM_FRACTION.

**Phase 4 — first forecast evidence (GPU-h 4.5):** WN3-0227 3-nest 3 h native, census on, wrfout home → D6 vs
CPU-WRF at 3 h on our CPU (labelled preliminary, not qualification); B=1 s/fc-h and 3-case parallel throughput
against the same-day declared RTX 5090 run.

**Total ≈ 15 GPU-h + 3 LD**, MI300X at the plan's [W 2026-10-17] $2.39–2.59/h ≈ **$36–39 compute**, plus ≈ 3 h
setup/idle/transfer overhead and storage ⇒ **budget $60–80**; stop-loss at 20 GPU-h.

**GO** (all): Phase-2 oracles green at frozen tolerances (bitwise where rule 2/3); composed PROD/Swiss/WN3 runs
native, 0 nonfinite, F2 census 0; host round trips per step measured and within a costed plan (static
scheduling / graph segments / unrolling, 3–8 LD reserve, G1); cached readiness ≤ 60 s or a measured, owned fix.
**Escape inside the window:** a kernel that miscompiles or hangs on XLA-ROCm's Triton is routed to the interpreter
(`GPUWRF_ROCM_INTERPRET_KERNELS`) or, after M2b, to the A2 stock-Triton route that already runs the whole release
program on Pascal (§7) — so one bad kernel does not end the window.
**NO-GO → re-plan** (any): an archetype oracle fails with no source-level fix inside the window; XLA-ROCm Triton
miscompiles/hangs a release kernel and A2 also fails (R2); host-driven control flow makes the B=1 root > 3× the
RTX 5090 root with no structural remedy in sight; memory does not fit one WN3 case.
Speed numbers are reported, never extrapolated into CPU-speedup claims (critique #5).

## 3. Capability-tuple compile matrix (production bodies, real shapes, release flags)

**Inputs [M].** `tools/capture_real.py` loads the real case on the CPU through `nested_pipeline._load_domains`
(as the release probe), then traces one production step of PROD d01 (120×70×44, release defaults incl. KF) in
the *production convention*: local facade AND `platform.runtime_backend` set to the vendor, `jax.default_backend()`
'gpu' (E156), LAPACK/cuSPARSE `tridiagonal_solve` restored after the CPU load. Every Pallas kernel is captured
where JAX lowers it (`lower_jaxpr_to_triton_module`), with its real grid and `num_warps/num_stages`.
CUDA: 61 unique bodies / 157 lowerings / 153 Triton custom calls. ROCm (vendor_ops, admitted): the same 61
bodies (1:1 by source+grid); AMDGCN asm replaces PTX RN (2062 → 1878 asm ops: divisions become IEEE `/`),
f64 tokens identical (3721; Thompson DOUBLE islands + weak literals, E193).

Two routes, kept apart (critique #2):
- **A1 (product route):** XLA's own pipeline on a CompileOnly client — `topologies.get_topology_desc(platform=…,
  target_config=…)`, the whole production step lowered against the virtual device and `.compile()`d
  (XLA-bundled Triton + ptxas 13.2 for CUDA; XLA-ROCm bundled Triton + LLVM AMDGPU + lld for ROCm). Target
  configs: captured RTX 5090 (control) and spec-sheet fabrications [I] for A100 sm_80, RTX 3090 sm_86,
  RTX 4090 sm_89, H100 sm_90, T4 sm_75, MI300X gfx942, MI250X gfx90a, RX 7900 XTX gfx1100
  (`proofs/o5_port/targets/`). CUDA flags = product command-buffer policy; autotune level 0 (no device).
  Native resources from the serialized executable: cubins via `scripts/e41/cubin_scan.py`, AMDGPU code objects
  via `tools/amdgpu_elfs.py` (AMDHSA notes).
- **A2 (escape / pre-Ampere route):** stock Triton 3.6 on each captured TTIR (`tools/compile_matrix.py`),
  ptxas -v with Triton's bundled 12.8 (+ the cu13 13.2 ptxas as second opinion for sm ≥ 75); hsaco for AMD.

No device executed anything: these are compile-feasibility and static-resource results, not E41 results of an
executing binary, and not speed.

### 3.1 A2 — stock Triton 3.6, NVIDIA, 61 production kernel bodies of PROD d01 [M CPU]

| arch | compiled | FAIL / TIMEOUT | kernels with spill or stack | max regs | slowest compile |
|---|---:|---:|---:|---:|---:|
| sm_70 (V100) | 61/61 | 0 / 0 | 3 | 255 | 226 s |
| sm_75 (T4) | 61/61 | 0 / 0 | 2 | 255 | 230 s |
| sm_80 (A100) | 61/61 | 0 / 0 | 3 | 255 | 245 s |
| sm_86 (RTX 30) | 61/61 | 0 / 0 | 2 | 255 | 209 s |
| sm_89 (RTX 40) | 61/61 | 0 / 0 | 2 | 255 | 285 s |
| sm_90 (H100) | 61/61 | 0 / 0 | 2 | 255 | 184 s |
| sm_120 (RTX 50, control) | 61/61 | 0 / 0 | 2 | 255 | 230 s |
| sm_61 (GTX 10, Pascal) | 61/61 | 0 / 0 | 2 (**coef 496 B stack / 492 B spill**, w 32 B) | 255 | 271 s |

Only three kernels ever spill or use a stack frame (ptxas 12.8; the cu13 13.2 ptxas agrees for sm ≥ 75 except where noted):
- **`dyn_acoustic_fp32` `b_core_coef_fp32`** (3.5 MB unrolled TTIR) is at the 255-register cap everywhere and spills:
  stack **320 B (sm_70), 264–272 B (sm_75), 312 B (sm_80)** — above the E41 256 B line — vs 32 (sm_86/89),
  24 (sm_90), 112 B (sm_120). This is the first M3 geometry item for Volta/Turing/A100 (block the level loop or
  shrink the block), matching W5P01's gfx1100 VGPR spill for the same kernel. A `num_warps` sweep (2/4/8, CPU on the
  box, `wz1080/M3_coef_numwarps_sweep.jsonl`) does not fix it: 4 and 8 warps give identical spills on sm_61–80,
  2 warps spill 1.9–2.6 kB, gfx1100 keeps 112 VGPR spills, and 8 warps on gfx942 introduce 110 VGPR spills (256 cap)
  → the fix is structural (level blocking), not launch geometry.
- `rad_sw_band_sums` 16 B spill at sm_70/sm_80 only; `b_core_w_fp32` a 32 B local frame on every arch (no spill).
- Heavy but spill-free: Thompson-full 242–246 regs, KF column 240–246, PD scale-sl 242–244 (ptxas 13.2: 167–184).
Full per-kernel tables: `proofs/o5_port/A2N_prod.md`, `A2N61_prod.md`, `A2R_prod.md`; A1 tables `A1N_prod.md`,
`A1R_prod.md`; control-flow census `CF_sm80.json`, `CF_gfx942.json`. sm_61 needs ptxas ≤ 12.x (§2); on the real 1080 Ti
the spilling coef kernel costs only ≈ 0.17 ms per Swiss step (§7).

### 3.2 A1 — XLA deviceless compile of the whole production step, NVIDIA [M CPU; targets other than sm_120 fabricated [I]]

PROD d01 production step (final candidate), XLA's own pipeline (bundled Triton + ptxas 13.2), product command-buffer
flags, autotune level 0. Native resources from the serialized executable (`scripts/e41/cubin_scan.py`).

| target | result | compile | kernels | max regs (Pallas, 43 of 61 bodies matched by name) | max STACK (any kernel) | local mem | max kInput transpose instr | temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sm_80 A100 | PASS | 282 s | 1218 | 206 | 32 B | 0 | 421 | 881 MB |
| sm_86 RTX 3090 | PASS | 283 s | 1205 | 210 | 32 B | 0 | 351 | 903 MB |
| sm_89 RTX 4090 | PASS | 254 s | 1155 | 210 | 32 B | 0 | 407 | 500 MB |
| sm_90 H100 | PASS | 222 s | 1131 | 206 | 32 B | 0 | 368 | 549 MB |
| sm_120 RTX 5090 (captured target, control) | PASS | 282 s | 1103 | 200 | 32 B | 0 | 406 | 524 MB |
| sm_75 T4 | **refused**: `FAILED_PRECONDITION: Triton support is only enabled for Ampere GPUs (compute capability 8.0) and up, but got compute capability 7.5` | | | | | | | |

Reading it:
- On every Ampere+ target the product route stays inside the E41 static limits (STACK ≤ 256 B, kInput ≤ 500);
  this is a deviceless prescreen of the code XLA would load, not an executing-binary E41 result.
- **A1 ≠ A2 [M]:** the same 3.5 MB `b_core_coef_fp32` TTIR with the same launch parameters (4 warps, 3 stages) gets
  **40–48 registers and no stack from XLA's Triton** but **255 registers + 24–320 B spill from stock Triton 3.6**
  (ptxas 12.8 *and* 13.2). So stock-Triton results are not a proxy for the product route on this kernel — and
  pre-Ampere (sm_70/75), which can only use A2, inherits the spilling version (M5 risk, first M3 item).
- Temps differ by target (0.50–0.90 GB): XLA's fusion/tiling decisions depend on the device description, so
  memory (R1) must be re-measured per tuple.
- The sm_75 refusal reproduces the wz1080 hardware receipt without hardware; no CUDA-13 route exists below sm_80
  (§2: ptxas 13.2 also rejects sm_61/70).
Receipts: `<DATA_ROOT>/wrf_gpu2/scratch/o5-port/A1N_prod/` (summary.json, per-target kernel tables, optimized HLO),
`<USER_HOME>/wrf_gpu2_lanes/o5-port/A1N_prod.md`.

## 6. Side result: the OFF-identity gate itself was blind to 4 Pallas kernels (fixed tool)

[M] The established probes (`proofs/o1_nlbind/probe.py` → `proofs/s1_port/release_identity.py`, w44 `off_raw.py`)
trace the canonical jaxpr first on the CPU default backend; inner jits keep those interpreted traces (E161), so
the native CUDA lowering that follows omits MYNN cloudmix/plume/tridiag2 and Noah snowwater, and the McICA KISS
kernels are dropped because `gpuwrf.backends.platform` still reports the CPU. PROD d01: 138 Triton calls / 55 bodies
/ 141 lowerings instead of 153 / 61 / 157. Every OFF identity built that way says nothing about those kernels.

Fix (manager request 16:27Z): `scripts/off_identity_probe.py` (native production-convention pass first in a fresh
process, BLIND verdict if any `pallas_call` stays interpreted inside the native program, per-kernel TTIR census,
then `clear_caches` + canonical value), `scripts/off_identity_compare.py`, `tests/test_off_identity_probe.py`
(synthetic E161 reproduction: old order BLIND, native-first COMPLETE, `clear_caches` restores; heavy real-PROD
test: old order BLIND 138/55/141, new COMPLETE 153/61/157, canonical 321aa651 in both). All OFF results in §1 used
the equivalent of the new order (`--no-canon` production captures) in addition to the established convention.

### 3.4 A2 — stock Triton 3.6, AMD, the 61 ROCm production bodies (vendor_ops, admitted) [M CPU]

| arch | compiled | FAIL / TIMEOUT | VGPR-spill or scratch kernels | max VGPR | slowest compile |
|---|---:|---:|---|---:|---:|
| gfx942 (MI300X, wave64) | 61/61 | 0 / 0 | Thompson-full 2 VGPR spills (269 VGPR, occ 1) | 366 (coef, occ 1) | 181 s |
| gfx90a (MI250X, wave64) | 61/61 | 0 / 0 | Thompson-full 2 VGPR spills (270 VGPR) | 366 (coef) | 173 s |
| gfx1100 (RX 7900, wave32) | 61/61 | 0 / 0 | **coef 112 VGPR spills / 452 B scratch** (256-VGPR limit) | 256 | 163 s |

The previously failing ring relax and boundary scalar couple bodies now compile (AMDGCN RN asm).

### 3.5 Nested-path bodies (force-down + d02 child step of PROD), A2, all targets [M CPU, box]

The 62 kernel bodies that only appear on the nested path (`--nested` captures P11/P12: SINT force-down, nest scalar
couple, the 5 child-only d02 kernels `b_diff_advect_u/v_fp32`, `b_diff_large_step_u/v_fp32`, `boundary_spec_final_real`,
and d02-shape variants of the others) compile on **all 11 targets** (sm_61/70/75/80/86/89/90/120, gfx90a/942/1100):
62/62 each, 0 FAIL/TIMEOUT; the nest-only kernels 132/132 without spill; the only spills are the d02-shape coef kernel
(as in §3.1). Together with §3.1/§3.4 every one of the 123 Pallas bodies of the PROD nested production program compiles
for every target (`A2nest_prod.md`). The Noah canopy
kernel that timed out at test shapes in W5P01 compiles at the real PROD shape. Table: `A2R_prod.md`.

### 3.3 A1 on ROCm — XLA-ROCm deviceless compile of the whole production step [M CPU; fabricated AMD targets [I]]

Isolated venv with the §2 AMD tuple; same PROD d01 production step, vendor_ops admitted ROCm bodies (AMDGCN
literal RN), `lowering → XLA-ROCm (bundled Triton, LLVM AMDGPU, lld)`, autotune level 0. Native resources from
the AMDGPU code objects in the serialized executable (`tools/amdgpu_elfs.py`, AMDHSA notes).

| target | result | compile | kernels | max VGPR | VGPR spill | SGPR-spill kernels | scratch | temp |
|---|---|---:|---:|---:|---|---:|---|---:|
| gfx942 MI300X (wave64) | **PASS** | 227 s | 1096 | 278 (`b_core_coef_fp32`, 1 wave/SIMD) | none | 14 (incl. KF 84, Thompson-full 67, Noah snowwater 66, coef 20 SGPRs) | one XLA fusion, 68 B | 501 MB |
| gfx90a MI250X GCD (wave64) | **PASS** | 245 s | 1101 | 278 (coef) | none | 25 | 5 XLA loop fusions, 68–132 B | 625 MB |
| gfx1100 RX 7900 XTX (wave32) | **PASS** | 212 s | 1137 | 256 (RDNA limit) | **`b_core_coef_fp32` 26 VGPR spills** | 12 | coef 108 B | 523 MB |

- The product route works for AMD code generation end to end: all Pallas kernels (incl. ring relax, boundary
  couple, Noah canopy) and all XLA glue compile for the exact tuple. XLA-ROCm's Triton gives the coef kernel 278
  VGPRs (stock Triton 366) — again A1 ≠ A2.
- **Static control flow per PROD d01 step** (`tools/hlo_controlflow.py` on the optimized HLO; same structure on
  sm_80 and gfx942): 12 while + 5 conditional instructions → weighted by trip counts **20 while executions /
  435 iterations (1 unknown trip count) and 93 conditional executions**, ≈ 5,000–5,550 launch-like ops. On CUDA
  the product captures CONDITIONAL/WHILE into command buffers; on ROCm each conditional needs a device→host
  predicate read + sync and every loop iteration is host-driven ⇒ **≈ 93 host round trips per d01 step** [I], before
  the d02 steps. This is the number M1 Phase 3 must measure (rocprofv3) and budget (G1).
Receipts: `<DATA_ROOT>/wrf_gpu2/scratch/o5-port/A1R_prod_{gfx942,gfx90a,gfx1100}/`, `CF_{sm80,gfx942}.json`.

## 7. Pascal on real hardware (wz1080, GTX 1080 Ti sm_61, 11 GB; box-only experiment) [M GPU]

Owner made the box available 2026-10-24 18:17Z. Tree = lane/o5-port `446807db75` (src content hash 54cb5a0e, verified on
both hosts) for the shim (`a2_*`) runs; from 21:05Z `03fe7edaab` (= `446807db75` + the opt-in route: src diff only
`backends/vendor_ops.py` + `backends/pallas.py`; box `TREE_PROVENANCE`) for every `route_*`, profile and oracle receipt —
i.e. without the main merge `741288143f` (5 src files: boundary_apply, acoustic, dyn_acoustic_fp32, operational_mode,
runtime_file_readers.json; none in the oracle component paths). The box's existing venv read-only (JAX 0.10 + `jax-cuda12-plugin` 0.10.0 + stock Triton 3.6 + CUDA 12.9;
CUDA 13 dropped Pascal), box GPU lock. Receipts: `proofs/o5_port/wz1080/`.

1. **Kernel level** (`tools/pascal_smoke.py`): the product literal-RN kernel (PTX `.rn` mul/add/div) on the default
   in-graph route → `FAILED_PRECONDITION: Triton support is only enabled for Ampere GPUs … got 6.1` (as on paper);
   the same kernel through **A2** (stock Triton → PTX sm_61 → jaxlib `triton_kernel_call`) → **65,536/65,536 bitwise**
   vs a NumPy binary32 reference; a contractible control: 35,703/65,536 (FMA + division) — the RN spelling matters.
2. **Exact-REAL contract (plan rule 3) on Pascal** (`tools/pascal_oracles.py`, product flag): MYNN `tridiag2` vs the literal
   WRF REAL4 transcription (BP56 reference) **bitwise** on 4 batches (70×3, PROD d02 117×267, 8400, odd 131 columns × 44;
   1.75 M values) — the same contract as on the RTX 5090. **Frozen Thompson WRF oracles** (`tests/v025/b_thompson/run_oracles.py`,
   retained WRF inputs/outputs relocated read-only, `--require-gpu`): Gate-2, L3a and L3b **30/30 PASS** on the native
   Thompson-full kernel through the route (production REAL and fp64 states, built-in mutants caught; route install receipt
   in `wz1080/thompson_oracles/thompson_gate2_recheck.log`) — the RTX 5090 gate C24 reproduced on Pascal.
   **KF frozen WRF oracle** (`tests/v025/b_kf/kf_gpu_oracle.py`, KF column Pallas kernel through the route): **5/5 cases
   PASS** (all six tendencies, max rel ≤ 7.2e-5; RAINCV 8.6e-7 vs tolerance 2.8e-4) — `wz1080/kf_oracle.json`.
   **RRTMG SW vs pristine WRF V4.7.1 RRTMG_SWRAD** (`tests/v025/w41_swddir/swddir_oracle_runner.py`, release path: REAL entry +
   fused Pallas band kernel, `band_kernel_calls.one = 1`, E114): AE01 on the 186 real WN3 0227 columns, aer_opt=0 arms
   0/2/4/6/8 — worst frozen tier-1 ratio **≤ 0.42** (arm 8 heating; all other arms ≤ 0.024; SWDDIR max abs 0.32 W/m²);
   Swiss S01 own call ≤ 0.019 (SWDDIR 0.17 W/m²) → **PASS** (bound = 1) — `wz1080/sw_oracle/`.
   *Not usable as a Pascal oracle:* the M5 analytic Tier-1 RRTMG gate (`tests/v025/b_phys/rrtmg_precision_gate.py --mode
   oracle`, tree `dd51631c51` = main `b14a9801cb` + facade) **fails under release defaults on the box CPU (interpret) as well**
   (SW surface_down max abs 285.7 W/m², LW flux_down 48.0) and **passes with `GPUWRF_FAST_DEFAULTS=0`** (Tier-1 LW ≤ 1e-4,
   SW ≤ 0.09 W/m²). Bisected (21 single-process arms, `tools/rrtmg_tier1_{arm.py,bisect.sh}`): the only cause is
   `GPUWRF_RRTMG_MAXRAND=1` (WRF Registry cldovrlp=2, release default since v0.3.3): release − MAXRAND passes (CPU 0.173,
   1080 Ti 0.176 W/m²), legacy + MAXRAND fails (285.2). The analytic fixtures are cloudy in all 16 layers and encode
   random overlap → **test hygiene** (stale direct-run gate; pytest runs it with MAXRAND off), not a product or
   interpret-path defect; cldovrlp=2 fidelity is the pristine-WRF SW oracle above. Pascal GPU equals the CPU run to
   ≤ 3.5e-4 W/m² on every max error — `wz1080/rad_tier1/`.
3. **Facade on a real non-sm_120 NVIDIA GPU** (`tools/facade_gpu_smoke.py`): `vendor()=='cuda'`, original `pallas_call`
   object, `platform_triton_native()`; release kernels through the facade (stage masses ×7 outputs at the PROD d02 plane,
   scalar couple 44×117×267) **bitwise** equal to the CPU-interpreted RN32 reference.
4. **Whole release-default program, natively** — box-only shim `tools/shim_a2/sitecustomize.py` (`O5_A2_ROUTE=1`) lowers
   EVERY `pallas_call` through A2, so XLA never sees an in-graph Triton call; bodies, dispatch and the rest are the product's.
   Swiss d01 (43×43×44, dt 18 s), `gpuwrf.cli run --hours 3`, release defaults ON:

| arm (same box) | whole run | steady (h2–h3) | notes |
|---|---:|---:|---|
| **A2 native release defaults, warm (AOT hit)** | 46.6 s / 3 h (15.5 s/fc-h incl. init + Python start) | **2.44 s/fc-h** | forecast-only 14.6 s; RSS 2.7 GB |
| A2 native, cold | 391–412 s | — | ≈ 5.5 min compile (stock Triton for 61 kernels + XLA) |
| legacy XLA path + interpreted Pallas (wz1080 deliverable 1) | 33.16 s/fc-h | 24.84 | `GPUWRF_FAST_DEFAULTS=0` |
| release defaults, interpreted Pallas (deliverable 1) | > 360 s/fc-h | — | did not finish hour 1 in 6 min |
| CPU-WRF np6 (best CPU config on the box) | 9.61 s/fc-h | 8.94 | |

   - **D6 vs RD11 CPU-WRF, 3 h: PASS** (frozen manifest, 361 fields; worst lead T2 0.095 K, PSFC 0.34 Pa, U10/V10
     0.017/0.025 m/s, RAINNC 4.7e-5 mm, T 0.0047 K, U/V 0.0097/0.0072, W 0.0023 m/s, QVAPOR 8.4e-7 — about half the
     legacy path's errors).
   - **Deterministic:** all wrfout byte-identical across three runs, including two independent cold compiles.
   - AOT caveat: the cheap key includes every `GPUWRF_*` variable; the box runner exports `GPUWRF_LANE=<label>`, so a
     new label is an AOT miss (E32-class) — reuse the label for warm runs.
   - **Product flag (`GPUWRF_PRE_AMPERE_KERNEL_CALL=1`, `03fe7edaab`, no shim):** Swiss 3 h wrfout byte-identical to the shim
     runs. **Swiss 24 h:** steady median 2.5 s/fc-h (h2–h24), **whole run 99.5 s = 4.15 s/fc-h** from process start (forecast-only
     67.9 s, RSS 2.75 GB), 25/25 frames byte-identical warm vs cold; **24 h D6 vs RD11 CPU-WRF: PASS** (worst lead T2 0.11 K,
     PSFC 2.8 Pa, U10/V10 0.098/0.090 m/s, RAINNC 0.20 mm, T 0.045 K, U/V 0.12/0.082, W 0.023 m/s, QVAPOR 7.8e-6).
     Reference CPU-WRF Swiss 24 h: 9.85 s/fc-h (main host, D30) / 9.61 (box np6, 3 h) → the 1080 Ti is ≈ 2.3× faster whole-run.
   - **PROD (2 domains, `--max-dom 2`) 3 h:** warm steady 25.8 s/fc-h, process 162.8 s (init-heavy), RSS 5.0 GB, fits 11 GB,
     byte-identical cold/warm (CPU-WRF 12-rank 92.3 s/fc-h; RTX 5090 ≈ 3). PROD d01 alone: steady 2.95 s/fc-h. No PROD D6
     (the CPU-WRF reference is ALISIOS data).
   - **Where Pascal spends the time** (`tools/pascal_profile.py`, CUPTI via `jax.profiler`, Swiss, 20 single-step calls):
     device 11.25 ms/step vs 12.2 ms/step steady CLI wall → device-bound. XLA fusions (glue/layout) 5.96 ms (53 %), Pallas
     4.26 ms (acoustic 1.31, lateral boundary 0.52, Thompson-full 0.48, advection 0.40, diffusion 0.32, PD 0.31, RK/EOS 0.30,
     MYNN 0.29, Noah-MP 0.28, coupling 0.05), copies/memset 0.48, other 0.55. The DOUBLE islands do not dominate
     (Thompson-full 0.48 ms incl. its f64 at 1/32 rate); the XLA glue is the first Pascal lever. PROD d01: device 37.3 ms/step
     (≈ 85 % of the 44 ms steady CLI step), XLA fusions 20.2 ms (54 %), Pallas ≈ 14.5 ms (acoustic 3.7, KF 2.0, MYNN 1.5,
     advection 1.5, Thompson 1.4, PD 1.2, diffusion 1.0, RK/EOS 0.85, lateral boundary 0.84, Noah 0.29).
   **Consequence for D1:** Pascal need not be "assessment only": with a CUDA-12 tuple and the A2 route it runs the
   release program natively, deterministically and inside D6 on Swiss, 3.7× faster than 6-core CPU-WRF in steady
   state on this box. Productizing = M2b (per-tuple route table selecting A2 below cc 8.0) + M5 qualification.

## 8. Handoff (AGENTS §6)

- **Lane** o5-port · v0.5.0 portability groundwork (M0 + M2a minimal + M2b opt-in), decision-independent.
- **Merge candidate** lane/o5-port `03fe7edaab` (src) + receipts/docs up to HEAD; merge-tree onto current main CLEAN
  (main's 87 newer commits touch none of these files). Files: `src/gpuwrf/backends/{vendor_ops,pallas}.py`; one-to-four-line
  non-CUDA branches in `kernels/dyn_rk_fp32.py`, `kernels/dyn_ring_relax_fp32.py`, `coupling/boundary_apply.py`,
  `nesting/sint_kernel.py`, `kernels/rad_mcica.py`; `tests/test_vendor_ops.py`; `docs/portability.md`; separately the OFF probe
  (`scripts/off_identity_{probe,compare}.py`, `tests/test_off_identity_probe.py`, own critic cr-o5-port-probe).
- **Gate numbers [M]:** OFF identity vs base `3fb0622d7b` — schema 2 (full constants) IDENTICAL for PROD nested (root d01,
  force-down, d02 child) and Swiss; established convention canon b977712f / 321aa651 and raw HLO unchanged; production
  convention raw c7183eff / 09ca2808 unchanged. Tests: vendor_ops + portability 38 PASS; seam suite 12 files 0 failures;
  probe light 7 + heavy 9 PASS. GPU (1080 Ti): facade kernels bitwise vs CPU RN32; Pascal release program native, Swiss D6 PASS.
- **Cost:** CUDA ≥ 8.0 program byte-identical → 0 kernels/copies/VRAM/s/fc-h change. New paths only behind
  `GPUWRF_ROCM_TRITON_NATIVE`, `GPUWRF_ROCM_INTERPRET_KERNELS`, `GPUWRF_PRE_AMPERE_KERNEL_CALL` (trace-time GPUWRF_* → cheap key).
- **Open risks:** ROCm route never executed on AMD hardware (M1); ~93 host round trips per PROD d01 step on ROCm (G1);
  `b_core_coef_fp32` spills with stock Triton on sm_70/75/80 and RDNA (M3 geometry); pre-Ampere route qualified on one box only;
  the established release-identity probes are blind to 4 kernels and to constant values (use the new probe).
- **Next:** critic for the facade (and the probe delta); Owner D1 (Pascal/Volta/Turing tier via A2), D4 (MI300X credits) →
  M1 Phase 0 here (§5), then the rental.
