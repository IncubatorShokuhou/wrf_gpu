# t-cammgws — CAM-MG (mp11) native GPU column: stock-Triton route

**Defect.** On the RTX 5090 (sm_120) the native CAM-MG Pallas column returned RAINNCV/SNOWNCV = 0 (8945 frozen
violations; micro skipped every column, LTRUE=0). XLA's bundled Triton drops the activity loads — both the in-graph
`__gpu$xla.gpu.triton` route and the PJRT plugin compiler (`jax._src.lib.triton.compile`) — while stock Triton 3.6
compiles the identical TTIR correctly (root cause localized by t-rainnc RN56/RN57/RN59).

**Discarded structural workarounds (GPU, all FAIL 8945 / rain 0):** num_stages=1 (CG02), inline count-1 loops (CG04),
ptxas 13.1 and ptxas -O0 (CG04: not ptxas), forguard scf.for loops and distinct cond conjuncts (CG06/CG12: XLA
Triton verifier 'operand does not dominate'), loop-free activity (CG15), plugin-compiler kernel-call route (CG19).

**Product route.** `src/gpuwrf/kernels/_triton_kernel_call.py`: `KernelCallCompilerParams` (subclass with its own
registered GPU lowering rule; only `cammg_mp11_column` uses it) → unchanged TTIR → `stock_compile` (CUDABackend
stages ttir→ttgir→llir→ptx, make_ttir first) → jaxlib `triton_kernel_call`. Fails closed without a visible CUDA
device, without Triton, on Triton ≠ 3.6.x, or if the kernel needs global/profile scratch. AOT: stock compiler
identity is part of the mp11 baked-table key (`runtime/aot_baked_tables.py`). mp11 removed from the GPU-refused set
(`io/cammg_config.py`; km3 keeps the shared hooks).

**Evidence.** GPU CG25/CG26: c03 pristine column and the active 44-level c89 column equal the CPU literal on all
27 outputs (max_abs 0; RAINNCV 0.245054 / 3.310337). Frozen controls: 30/36 direct PASS; the 6 failing calls (cases
16–18) were a HARNESS artifact — those controls need non-default aerosol modes, the runner closed over defaults; CPU
CG30 reproduces the 6 GPU rows exactly with default modes and gives 0 outside with the case modes. **GPU CG28
(modes as executable args): 36/36 frozen controls PASS, 0 outside, fatal 0.** GPU CG33 active 44-level column:
RAINNCV/SNOWNCV 3.310337 = literal, worst nc3d 5.7e-6 / qc 2.7e-19 (FMA-contraction ULP; 0 with fusion off, CG26).
E41 PASS on every XLA executable (CG25/CG28/CG33) — these scans cover only the 8 zero-stack XLA glue fusions; the
CAM kernel is not inside the XLA executable. **Fresh-process AOT reload (CG35):** serialized executable 1b1d27e1…
(PTX 9e34e7f4…, embedded) deserialized in a new process with stock compilation forbidden: 27/27 outputs BYTE vs the
export process, fatal identical; first launch 26.1 s (in-process libnvptxcompiler assembly), warm 17.8 ms.
**Native artifact (CG35_native_artifact.json):** offline ptxas 13.2 of that exact PTX → cubin fc11e744…: 255 regs,
STACK 1240 B (E41 STACK flag, disclosed opt-in), 239,848 SASS, 4 leaf callees ≤466, no non-leaf >1k. All
consumers (nested, fused, batch) reach the kernel through the single call site physics/cammg/adapter.py →
kernels/phys_cammg.full_column (CG35_native_artifact.json call_sites).

**Critic FIX items (rv-t-cammgws cdd17405) closed:** (1) content-keyed compiler identity (all stock-Triton package
bytes + route/kernel sources; 1.04 s once per process) in the mp11 AOT key and the stock_compile memo; same-metadata
collision regression + mutants M5/M6; (2) CG35 reload receipt; (3) catalog CPU-ONLY prefix and plugin wording
removed; (4) E41 logs force-added, native artifact identity/scan sealed. CPU48 1296/1296 arrays BYTE; release identity Swiss f672a573 /
PROD 6fc715a2 unchanged; route/AOT-key/admission tests with 4 genuine mutants (CG23_M1–M4).

**Resources (disclosed, opt-in scheme).** Native kernel (ptxas 13.2 from the stock PTX, CPU): 255 regs, STACK 1240 B
(E41 STACK flag), spills 3824/5008 B, 239,848 SASS, 4 leaf callees ≤466 SASS. XLA executable E41 PASS. Stock compile
at lowering ≈62–78 s per shape; first launch per fresh process assembles the 5.2 MB PTX in-process (~26 s per
shape, not cached across processes). Warm 1-column call 17.9 ms (24 lev) / 33.8 ms (44 lev); temps 2060 B; one custom-call launch per call.
c89 (42×42×44, dt 18 s) estimate ≈ 0.2 s per MP call. Occupancy lever for v0.4.1: num_warps=1 (needs its own gate).

## DLF2 history field (mp11), 2026-10-16

c89 24 h GPU record: all bounded fields pass, NF0; only DLF2 missing vs CPU-WRF. WRF: `Registry/registry.cam:73`
`state real dlf2 ikj misc … rh "DLF2" "dq/dt due to export of cloud water into environment by shallow convection"
"kg/kg/s"`, allocated by cammgmpscheme (mp11) with `initial_data_value=0.`. Its **only writer** is
`module_shcu_camuwshcu_driver.F:400` (UW shallow cu, reached only via `CASE (CAMUWSHCUSCHEME)` in
`module_shallowcu_driver.F`); CAMMGMP reads it `intent(in)` (`module_mp_cammgmp_driver.F:184`; its :758/:766/:844
assign a local r8 copy). The port refuses `shcu_physics≠0`, so in every admitted mp11 run WRF writes the Registry +0.
Source audit (18 sites, file SHAs): `DLF2_SOURCE_AUDIT.json`. CPU proof: c89 all 25 frames, every cell +0 bits,
metadata `DLF2_CPU_ALL_FRAMES.json`. Product: `io/producer_history.py` (`CAMMG_ZERO_FIELDS`, emitted for mp11 when
`shcu_physics≠2`; a nonzero supplied value is refused) + `io/wrf_variable_metadata.py` (CPU-pinned attrs).
Tests `tests/v040/test_cammg_dlf2_history.py` (9: CPU proof, pristine single writer, writer bits+metadata in
subset/full streams, production nested writer in both stream modes, branch exclusion, contradiction refusal, shcu=2
refused before load); mutant DL-M1 (drop the branch) → 6 genuine failures. Gates `receipts/dlf2/`: 607 PASS, default
IDs Swiss f672a573 / PROD 6fc715a2 EXACT. Cost: host-side only — one shared read-only zero array per shape (c89
310 KB/frame), no traced-program or device change.
