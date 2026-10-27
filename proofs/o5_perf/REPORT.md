# o5-perf — runtime performance and memory analysis of the current tree (v0.5.0 roadmap steps 2–4, CPU part)

Status: **in progress** (2026-10-24, lane o5-perf, Opus 5.5). This is a CPU analysis of existing GPU artifacts plus a
default-off fix prototype on lane/o5-perf. No new GPU run yet. Labels: [M] measured, [I] inferred, [U] unknown.

## Summary

1. **PROD stepping carries ≈ 9 ms/root of host/dispatch idle. Since v0.3.3 a large part of it comes from an XLA
   graph-capture limitation (H-CB).** This is the joint reading with w44-perf (2026-10-24 21:3xZ):
   - Per-root stepping wall on v0.4.3 is ≈ 46.9 ms. It splits into ≈ 36 ms device on ordinary roots, ≈ 2 ms radiation/KF
     extra device, and ≈ 8.7 ms idle on the device timeline [M G43/N43/R43; mixed scopes, §3].
   - The v0.3.2 → v0.4.3 stepping regression is +3.4 ms/root (mean; the v0.3.2 A/A failed). It matches the growth of the
     device union, +3.43 ms/root (w44-perf `HCB_RECONCILE.json`).
   - The host residual (wall minus ordinary-root device union) is ≈ 11 ms/root in both versions. H-CB changed where it comes
     from, not measurably its size. It is a recovery opportunity, not the explanation of the regression.
   - *Correction of my 17:00Z reading:* "d02 call 30 → 37.7 ms" compared per-call host times that include blocking. In v0.4.x
     the d01 call returns right after an asynchronous graph launch, and the d02 call absorbs the wait for d01's device work.
     The per-root call sum (39.7–43.8 → 44.7–45.4 ms) is the right scope.
   - Since v0.3.3 (+WHILE capture) the PROD d02 step program runs completely outside CUDA graphs: 1,950 host-launched kernels
     plus 585 copy API calls per root, and a host-driven loop.
   - **Cause** [M source + CPU replay of the stored options, cross-checked on v0.3.2]: the pinned XLA's command-buffer pass
     never descends into a convertible WHILE, and it drops sequences shorter than `xla_gpu_graph_min_graph_size` = 5. d02's
     top level has 4 thunks.
   - **Fix:** `GPUWRF_STEP_CMDBUF=v1|v2` (per-jit options on the stepping programs). The deviceless gate is green: optimized
     HLO and native code are identical (§3).
   - Expected gain −0.25…−0.5 s/fc-h on PROD (≈ −5…−10 % S1) [I], ceiling ≈ 0.58 (the whole idle):
     - lower end: only the 3.5 ms/root inter-kernel gaps of the host-launched d02 kernels go away;
     - upper end: the 3.0 ms/root of host work between executables also overlaps with the device once the d02 call no longer
       blocks.
   - The GPU A/B (cbab) decides empirically; it is booked for the 25.10 window, and a 5-min CUDA mini is ready.
   - Only per-domain leaf child programs are affected. The WN3 release workload is already fully captured.
2. **Startup is 23 % of a 24 h PROD run** (27 s per process). 11.2 s of it is cache-hit compile-path work for ≈ 1,094 eager
   init programs (§4). Lever S-1 is worth −0.35…−0.5 s/fc-h on 24 h, ≈ −3 s/fc-h on 3 h runs, and is paid per case in WN3 par3.
3. **Device time per ordinary root is 34.4 ms, i.e. 79 % of the static unique-byte roofline (27.3 ms)** [M].
   - Native dycore kernels run at 83–96 % of peak; the column kernels (Thompson, MYNN, PD, boundary) at 15–37 %.
   - The XLA glue already runs at 91 % of HBM peak on its lower-bound bytes. It is bytes-bound, so further device gains need
     fewer bytes: native/fused child moisture-scalar stage (D-1, ≈ −0.13 s/fc-h), layout conversions (8.55 GB/root), and
     ultimately per-RK-stage fused kernels (≈ −30…−40 % device time, a restructuring) (§2).
4. **Release benchmark (WN3 par3, GPU-saturated):** device bytes and multi-process sharing matter, host gaps do not. CUDA MPS has
   never been tried and could gain 10–25 % [U] at zero numerical change (P3 launcher).
5. **Memory:**
   - R1 = 0.39× CPU-WRF RSS (PASS).
   - R2 PASS [M]: footprint α = 0.43 (leaf) and 0.61 (root); the carry is linear, temporaries are nearly flat.
   - The deviceless compile reproduces GPU memory statistics byte-exactly, so it is a valid capacity instrument.
   - Temporaries are dominated by RRTMG 4,096-column tile workspaces (sub-linear in columns, linear in levels).
   - The WN3 fused cascade does not donate its carry: ≈ 0.4 GB per case (§5).

## 0. What was analysed and why it applies to main

- Program identity. With every v0.4.4 option unset, main must keep the release identities (PROD 321aa651, Swiss b977712f;
  `_WAVE_V044_COMMON.md`). So the PROD stepping programs on main are the v0.4.3 bundle programs. The topo fix is not merged yet.
  The only exception would be a lane that broke OFF identity. [I from the wave rule; not re-proven here]
- Device evidence. The w44-perf node captures of v0.4.3 were reused. N43 = three ordinary roots 68–70. R43 = the
  radiation/KF window, roots 100–102. G43 = a graph-level capture of one whole segment, about 67 roots, without node tracing.
  The kernels are bound to the bundle's actual optimized HLO (`w44-perf/ACTUAL_HLO`, native records 0/1/2 verified) and to the
  runtime thunk → symbol table (`ORDINARY_THUNKS`).
- New tool: `roofline_join.py`. For every executed kernel it joins the node-trace duration to a static byte count from the same
  optimized HLO. The byte count is a streaming lower bound: each operand byte is read once and each output byte written once.
  It is slice-aware (dynamic-slice/gather reads, in-place dynamic-update-slice writes).
  Output: `<USER_HOME>/wrf_gpu2_lanes/o5-perf/roofline/ROOFLINE_V043.json`.
  Pallas byte counts are upper bounds, because the operands are whole state arrays (BP54 caveat). Achieved bandwidth is therefore
  only claimed for XLA fusions.
- Whole-run S1 evidence: HERO H1/H2 (v0.4.2, native programs identical to v0.4.3), audited in `w44_perf/receipts/HISTORICAL_RECEIPT_AUDIT.json`.

## 1. Where a 24 h PROD forecast spends its time (S1 ≈ 4.7 s/fc-h)

| Term | s/fc-h | Share | Evidence |
|---|---:|---:|---|
| Startup, process → first segment (27.2 / 26.0 s) | 1.11 | 23 % | [M] HERO |
| Ordinary-root device kernels: d02 28.6 ms + d01 5.4 ms + force-down 0.3 ms per root | 2.29 | 48 % | [M] N43 × 66.7 roots/h |
| Device copies inside roots (1.66 ms/root) | 0.11 | 2 % | [M] N43 memcpy |
| Radiation + KF extra device time (night window; daytime SW adds ≈ 0.05) | 0.13 | 3 % | [M] R43 |
| Host / dispatch idle on the device timeline (≈ 8.7 ms/root) | 0.58 | 12 % | [M] G43 gap union; split in §3 |
| Warm-up ramp k1–k4 above the steady period | ≈ 0.15 | 3 % | [M] HERO per-segment periods |
| Output callbacks, steady (0.11 s/h) and in k1/k2 | ≈ 0.10 | 2 % | [M] HERO output union |
| Finalization after the last segment | 0.04 | 1 % | [M] HERO |
| Sum versus measured 4.75 / 4.67 | ≈ 4.5 | | the rest is session spread (stepping 3.13 here vs 3.25 in HERO) |

The steady stepping is 3.23–3.25 s/fc-h in HERO and 3.10–3.13 in the w44 same-session proxy [M]. Device busy is ≈ 2.52 s/fc-h of
that [M/I]; the remaining ≈ 0.6–0.7 s/fc-h is device idle. **The stepping regression v0.3.2 → v0.4.3 (+0.23 s/fc-h) equals the
extra kernel time (+3.22 ms/root, +0.215 s/fc-h; N32 93.48 → N43 103.15 ms per three roots)** [M/I]. The w44-perf ordinary delta
attributes it to fidelity fixes, so it is not an avoidable regression.

## 2. Ordinary root on the device (N43, per root, 34.38 ms kernel time)

By kernel class [M]:

| Class | Launches/root | ms/root | Static GB/root | Achieved TB/s | Launches < 4 µs |
|---|---:|---:|---:|---:|---:|
| Pallas native kernels | 596 | 16.13 | (35.0, upper bound) | n/a | 68 |
| XLA transpose-kInput fusions | 576 | 6.82 | 8.55 | 1.25 | 296 |
| XLA kLoop fusions | 1,254 | 6.52 | 10.47 | 1.61 | 905 |
| XLA kInput (reductions etc.) | 544 | 3.90 | 9.24 | 2.37 (L2 reuse) | 381 |
| XLA Triton fusions (kCustom) | 148 | 0.43 | 0.54 | 1.26 | 111 |
| Unbound (force-down executables) | — | 0.58 | — | — | — |

- **The XLA glue is bandwidth-bound and already efficient per byte.** It moves 28.8 GB/root in 17.7 ms, i.e. 1.63 TB/s =
  91 % of the 1.79 TB/s HBM peak; L2 reuse lifts some classes above peak [M]. The device-side lever is therefore **fewer bytes**
  (fewer materialized intermediates and layout conversions), not faster kernels. This is the endpoint argument (a) of AGENTS §2
  for the XLA half of the device time.
- Layout conversions alone (transpose-kInput fusions) carry 8.55 GB/root in 6.82 ms (0.45 s/fc-h). This confirms A45/A47 at the
  byte level: the data really moves between column-major physics and level-major dycore.
- Tiny kernels: 1,761 launches/root are shorter than 4 µs, 3.05 ms/root in total (0.20 s/fc-h) [M].
- Byte budget versus the essential state traffic [I]. One ordinary PROD root reads and writes the d02 carry
  (347 MB of arguments) 3 times and the d01 carry once: ≈ 2.4 GB, ≈ 1.3 ms at peak HBM. The program moves ≥ 28.8 GB in XLA
  glue alone, ≈ 12× the carry. RK3 stages and acoustic substeps inherently re-read fields, so the floor is far above 1.3 ms,
  but there is room.

  Folding the bandwidth-bound glue (scalar stage, layout conversions, boundary faces, small kernels) into the native kernels,
  keeping only one read+write of each state field per RK stage, would plausibly bring the root from 34 ms towards ≈ 21–24 ms
  (−30…−40 % device time, ≈ −0.7…−0.9 s/fc-h) [I]. That is a restructuring of the step (per-stage fused kernels), not a
  per-kernel tuning task; it is the honest "(b) rewrite" item of AGENTS §2.

**Native (Pallas) kernels: unique-byte roofline** [M]. Method: `tools/ttir_args.py` decodes each kernel's embedded Triton IR and
traces every `tt.load`/`tt.store` pointer to its function argument. The floor counts the bytes of the arguments actually loaded
or stored. Joined with N43 time by `pallas_roofline.py` (receipt `PALLAS_ROOFLINE_N43.json`):

| Native family | ms/root | Unique-byte floor ms | % of HBM peak | Comment |
|---|---:|---:|---:|---|
| dycore/advection (acoustic, PD, flux, diffusion, RK, boundary) | 12.07 | 9.95 | 83 % | `b_core_uv` 96 %, `b_core_mass` 87 %, `b_core_w` 105 % → at the roofline |
| Thompson (`thompson_full_column`, 0.5 ms/call) | 1.64 | 0.55 | 33 % | compute/latency-bound (DOUBLE islands, per-column work) |
| MYNN (`mynn_tridiag2_real` 37 %, BouLac `_lambda_` 12 %) | 2.17 | 0.63 | ≈ 30 % | one thread per column, occupancy-bound (E150) |
| PD limiter (`b_core_pd_tend/scale_sl`) | 1.20 | 0.38 | 29–35 % | |
| `boundary_spec_final_real` / ring relax | 0.97 | 0.27 | 15–37 % | ring-only work on full-field arguments (floor overcounts) |
| **All native** | **16.13** | **11.23** | **70 %** | |

**Device-side verdict (S4, bandwidth view).**
- Floor: native 11.2 ms + XLA glue 28.8 GB at peak 16.1 ms = **27.3 ms/root**, against 34.4 ms measured, i.e. 79 % of the static
  byte roofline of the current kernel/fusion structure.
- Per-kernel tuning headroom is ≈ 7 ms/root ≈ 0.47 s/fc-h, concentrated in latency- or occupancy-bound column kernels:
  Thompson, MYNN tridiagonal/BouLac, PD limiter, boundary spec-final.
- Beyond that, only fewer bytes help (structure: layout conversions, glue fused into native kernels).

By source family [M] (static GB = slice-aware lower bound; Pallas families include upper-bound operands):

| Family (dominant source) | Launches/root | ms/root | GB/root | TB/s | < 4 µs |
|---|---:|---:|---:|---:|---:|
| dycore/advection | 1,169 | 19.34 | 41.6 | (2.15) | 412 |
| MYNN | 126 | 2.36 | 2.0 | 0.86 | 45 |
| Thompson | 5 | 1.65 | 1.0 | 0.63 | 1 |
| mixed MYNN+boundary+layout | 100 | 1.59 | 3.6 | 2.25 | 20 |
| boundary/nesting | 316 | 0.88 | 0.96 | 1.09 | 275 |
| mixed boundary+dycore | 222 | 0.86 | 3.1 | 3.64 | 192 |
| mixed MYNN+boundary | 145 | 0.80 | 2.7 | 3.42 | 58 |
| Noah-MP | 232 | 0.54 | 0.42 | 0.78 | 201 |
| surface layer | 127 | 0.40 | 0.08 | 0.25 | 103 |

Heaviest single kernels (d02 unless noted), ms/root [M]:

| Kernel | ms/root | Comment |
|---|---:|---|
| thompson_full_column (3 calls) | 1.50 | 0.5 ms/call, 0.57 TB/s on its operand bytes → latency/compute bound |
| loop_add_multiply_fusion.3 (3 calls) | 1.33 | kInput transpose emitter; 30 operands, 248 MB/call, 0.56 TB/s; moisture large step `_apply_moisture_large_step` + `_nested_scalar_stage_tendencies` |
| b_core_uv / b_core_mass (21 calls each) | 1.23 / 1.16 | acoustic |
| mynn_tridiag2_real (27 calls) | 1.13 | one thread per column, occupancy bound (E150) |
| loop_add_divide_multiply_fusion (3 calls) | 0.83 | kLoop, 0.68 TB/s; same moisture/scalar stage |
| b_core_w (+ w_p1/w_p2) | 0.76 (+0.95) | acoustic w |

The two moisture/scalar-stage fusions run at 0.56–0.68 TB/s instead of ≈ 1.5 TB/s. A column-layout operand (`f32[117,44,267]`)
forces the transpose emitter onto a 30-operand multi-output fusion. At 1.5 TB/s they would save ≈ 1.3 ms/root (0.085 s/fc-h) [I].

Ordinary-root kernel time by dominant source function (`source_rank.py`, receipt `SOURCE_RANK_N43.json`). Each kernel is
assigned whole to its most frequent source frame; this is a ranking, not a per-line cost [M]:

| Source function | ms/root | Cumulative | Launches/root | Note |
|---|---:|---:|---:|---|
| `kernels/dyn_acoustic_fp32.py:_call` | 5.23 | 15 % | 176 | native acoustic (uv/mass/w/p) |
| `runtime/operational_mode.py:_nested_scalar_stage_tendencies` | 2.18 | 22 % | 9 | **XLA glue**, child moisture/scalar RK stage |
| `kernels/phys_thompson_full.py:full_column` | 1.66 | 26 % | 7 | native |
| `kernels/phys_mynn_tridiag.py:solve_tridiagonal_real` | 1.29 | 30 % | 36 | native, occupancy bound (E150) |
| `runtime/operational_mode.py:_apply_moisture_large_step` | 1.29 | 34 % | 18 | **XLA glue**, moisture large step |
| `kernels/dyn_pd_fp32.py:advect_scalar_pd_fp32` | 1.28 | 38 % | 14 | native PD scalar advection |
| `kernels/dyn_uvtend_fp32.py:large_step_uv_fp32` | 0.79 | 40 % | 18 | native |
| `kernels/phys_mynn_boulac.py:boulac_length_native` | 0.79 | 42 % | 12 | native |
| `kernels/dyn_diff_fp32.py:horizontal_diffusion_fp32` | 0.65 | 44 % | 50 | native |
| `kernels/dyn_flux_fp32.py:advect_scalar_flux_fp32_stacked` | 0.59 | 46 % | 14 | native |
| unbound (force-down executables) | 0.58 | 48 % | 291 | |
| `runtime/operational_mode.py:_diffopt1_dry_forward_tendencies` | 0.56 | 49 % | 26 | XLA glue |
| `dynamics/flux_advection.py:specified_flux_faces` / `stage_omega_specified` | 0.53 / 0.51 | 52 % | 53 / 63 | XLA glue, boundary faces |
| `kernels/layout_pin.py:row_major` | 0.51 | 54 % | 84 | E41 root pins (compiler protection) |
| `physics/rrtmg_mp_re.py:calc_thompson_effective_radii` | 0.50 | 56 % | 4 | every step, like WRF; radiation consumer only |

Why these two fusions are slow [M HLO]:
- Their first operand (`get-tuple-element.13520`, `f32[117,44,267]`) is the layout-normalized form of a carry field held in
  column-major layout. It is produced by another transpose fusion (`input_add_multiply_select_transpose_fusion`, sources
  `_physics_sc_outside_spec`, MYNN cloudmix/`mix_ice_number`, `mynn_adapter_with_source_leaves`, `layout_pin.row_major`).
- So this is an instance of the A47 layout tax. A45/A47b found that carry layout pins alone move the transposes elsewhere
  without a net gain.

**Lever D-1: native or fused child moisture/scalar stage.** `_nested_scalar_stage_tendencies` plus
`_apply_moisture_large_step` spend 3.47 ms/root in 27 XLA launches, at 0.56–0.68 TB/s on their two largest fusions. A native
(Pallas) or layout-consistent fused version at ≈ 1.5 TB/s would save ≈ 2 ms/root ≈ **0.13 s/fc-h** [I]. Every new Pallas
consumer must match the operand layouts (E175/A62).

Radiation window (R43, `SOURCE_RANK_R43.json`) [M]:
- LW band flux sums are the largest single item, 7.53 ms per window-root average.
- The McICA max-random CDF (`kernels/rad_mcica.py:max_random_cdf` + `.step`) costs 4.05 ms/root average, i.e. ≈ 12 ms per
  radiation pair, in ≈ 1,440 launches/root: it is launch-bound.
- `rrtmg_lw.py:_take_rows` gathers 2.78 ms/root average.

A single column-parallel CDF kernel would save ≈ 10 ms per radiation pair ≈ 0.02 s/fc-h (small: radiation runs 2×/h).
Native radiation kernels (`PALLAS_ROOFLINE_R43.json`) [M]: `rrtmg_lw_band_sums` runs at 32 % of peak (74 MB unique per launch, 58.7
launches per window root). SW floors are overcounted at night (little work, full arguments).

## 3. Host / dispatch gap (≈ 8.7 ms/root ≈ 0.58 s/fc-h)

G43 (graph-level capture, one segment, ≈ 67 roots): device span 3,276 ms, busy union 2,696 ms → **580 ms idle** [M]:

| Idle component (per segment) | ms | ms/root | Mechanism |
|---|---:|---:|---|
| Inter-kernel gaps < 1 ms | ≈ 232 | 3.5 | 146,535 host-launched d02 kernels (≈ 1.6 µs each); see H-CB |
| Gaps > 1 ms with no CUDA API active (host Python/XLA work between executables) | 202 | 3.0 | per-execute host work (A57), dispatch |
| Gaps > 1 ms during launch / graph update | 80 | 1.2 | 83,922 `cuGraphExecKernelNodeSetParams` = 1,253 per root (d01 graph updated every call) |
| Output callbacks | 48 | 0.7 | |
| Transfers, allocation, other | 18 | 0.3 | |

Host CUDA API time per root, excluding synchronization waits [M]: `cuLaunchKernelEx` 4.2 ms, `cuMemcpyDtoDAsync` 1.7 ms,
graph node updates 0.9 ms, `cuGraphLaunch` 0.66 ms; plus 17 `cuStreamSynchronize` per root.

### H-CB finding: the PROD d02 stepping program runs entirely outside CUDA graphs

- **v0.4.3 [M N43/G43]:** the PROD d02 program has 1,950 kernels/root, and all of them are host-launched (`graph_id` none,
  `cuLaunchKernelEx`). Its 585 DtoD copies/root are separate API calls. The d01 program, by contrast, is one CUDA graph per root
  (1,335 kernels, WHILE/CONDITIONAL command buffers). The whole process builds only 3 graphs (`cuGraphCreate` 3,
  `cuGraphInstantiate` 3).
- **v0.3.2 [M N32]:** d02 had 1,671 graph kernels and 132 host-launched kernels per root (`+CONDITIONAL` only).
- Since v0.3.3 (6e38cbf173) the capture flag is `--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` [M source].
- Ruled out statically [M CPU]:
  - the stored DebugOptions of d01 and d02 are identical (field 258 = FUSION, CUBLAS, …, CONDITIONAL, WHILE);
  - the serialized thunk trees have the same kind set (kernel/copy/while/conditional, all copies DtoD);
  - both outer while loops lack a trip count;
  - no custom-call targets other than Triton.
  The capture decision is made at load/runtime (the serialized tree is pre-conversion).
- **Same-session host evidence (unprofiled; w44 Oct 23 S0 arms; `receipts/HOST_CALL_TIMES_OCT23.json`)** [M]. Median host
  time inside the per-domain `advance` call, segment 2 onwards:

  | Version | d01 call ms | d02 call ms | force ms | Root (seg 2–3) ms |
  |---|---:|---:|---:|---:|
  | v0.3.2 (A32 ×2) | 5.9–7.4 | **30.0–31.4** | 0.4–0.8 | 39.7–43.8 |
  | v0.4.0 (C40 ×2) | 2.7–2.8 | **37.6–37.8** | 0.5 | 44.8–45.4 |
  | v0.4.3 (B43 ×3) | 2.8–5.2 | **35.4–37.7** | 0.5–0.9 | 44.7–45.3 |

  - `host_s` includes blocking on in-call syncs. The root time equals the sum of the call times plus ≈ 4 ms of Python.
  - In v0.4.x the d01 call returns after one asynchronous graph launch (2.7–5.2 ms), so the d02 call also absorbs the wait for
    d01's device work. Its growth of +6…+7.7 ms is therefore mostly a **blocking shift**, not more host work.
  - The comparable scope is the per-root sum: +1…+5 ms (≈ +3.4 mean). That matches the device-union growth of +3.43 ms/root
    (w44-perf `HCB_RECONCILE.json`, node-traced ordinary roots; d02 kernel sum alone +1.9 ms, N32 26.7 → N43 28.6).
  - Correction: my 17:00Z reading ("d02 call +6…+7.7 ms with +1.9 ms device, so H-CB explains the regression") mixed these
    scopes. **Joint reading with w44-perf:**
    - the regression is not resolved on the host side;
    - wall minus ordinary-root device union is ≈ 11 ms/root in both versions (≈ 0.745 s/fc-h). Of that, ≈ 2 ms/root is
      radiation/KF extra device time, and ≈ 8.7 ms/root is idle on the device timeline (G43 below);
    - H-CB changed the source of the idle (v0.3.2: 20 + 21 command buffers around 10 + 11 host whiles; v0.4.x: d02 fully
      host-launched behind a host loop), not measurably its size.
  - The 24 h HERO run (v0.4.2 programs) shows the same pattern: d02 call 37.2 ms median, d01 5.7 ms, root 51.8 ms.
  - **Expected effect of capturing d02 [I]:**
    - lower end: the 3.5 ms/root inter-kernel gaps of the host-launched d02 kernels (G43) go away;
    - upper end: the 3.0 ms/root of host work between executables also overlaps with the device once the d02 call no longer
      blocks. A per-root host read of a device value would cap this [U].
    - Together that is −3.5…−7.5 ms/root ≈ **−0.25…−0.5 s/fc-h**. The ceiling is the whole idle, ≈ 0.58 s/fc-h.
- **ROOT CAUSE [M source + M thunk trees, CPU].** Pinned XLA b6f37ab, `xla/backends/gpu/runtime/command_buffer_conversion_pass.cc`
  (copy in `receipts/`), lines 466–600:
  - `Run()` appends a *convertible* WhileThunk to the pending sequence and `continue`s, so it never descends into the body.
  - `FlushCommandBuffer()` leaves a pending sequence with fewer than `max(1, xla_gpu_graph_min_graph_size)` thunks
    unconverted. The stored DebugOptions field 208 is 5.
  - The PROD d02 executable's top level is `[copy.1668, wrapped_convert, wrapped_convert.1, while.793]`, i.e. 4 thunks.
    Nothing is converted, and the whole program runs op by op with a host-driven while loop.
  - d01's top level has 12 thunks (8 root-pin transposes …) and becomes one WHILE graph.
  - With `+CONDITIONAL` only (v0.3.2), the while is not convertible, so the pass descends into the body.
  - Upstream XLA `main` fixed this: `CountCommandBufferSize` walks nested thunks.

  CPU replay of the pinned algorithm on the serialized trees (`cb_partition_sim.py`, receipts
  `CB_PARTITION_SIM_{PROD,WN3}.json`) reproduces the traces exactly:

  | Program (v0.4.3 bundle) | Top-level thunks | OFF (+WHILE, min 5) | V1 (+WHILE, min 1) | V2 (no WHILE, min 5) |
  |---|---:|---|---|---|
  | PROD d01 `13c9f753` | 12 | 1 CB, 1,651/1,651 | 1 CB, 1,651/1,651 | 22 CBs, 1,621 captured, 12 host whiles |
  | **PROD d02 `e3bbc3ec`** | **4** | **0 CB, 0/1,212 (all host)** | 1 CB, 1,212/1,212 | 22 CBs, 1,186 captured, 11 host whiles |
  | WN3 fused d02+d03 `14d79aee` | 814 | 1 CB, 2,605/2,605 | same | 41 CBs, 2,561 captured |
  | Swiss d01 `b986b63f` | 9 | 1 CB, 1,810/1,810 | same | 18 CBs |

  **Only the per-domain leaf child program (PROD d02) is affected among the sealed programs.** The WN3 release workload is
  already fully captured.
  - Cross-check on v0.3.2 [M CPU]: the replay reads the stored options (258/208) of each blob. The v0.3.2 executables
    (w44-perf `cache_v032`) store CONDITIONAL without WHILE; they give d01 20 CBs (1,252/1,280) and d02 21 CBs (1,034/1,065,
    11 host whiles). That is the mostly captured d02 of N32 (`CB_PARTITION_SIM_V032.json`). The same replay on the v0.4.3
    stored set (+WHILE) gives the d02 0/1,212 seen in N43.
  - Generalization [I]: only root programs carry the 8 root layout-pin transposes in their top-level sequence. Fused cascades
    (`domain_tree._fusable_parent`: all children leaves, no feedback, a root only in a 2-domain tree) have ≈ 800 top-level thunks.
  - So every per-domain NON-root step program is expected to hit the bug: PROD-type 2-domain runs, and every domain of a
    two-way (feedback) nested run, since feedback disables fusion. These configurations gain from the fix too.
  - V1 changes nothing for the other programs.
  - V2 de-captures d01 and the fused program into v0.3.2-style host-driven loops.
- **Deviceless gate GREEN** [M CPU; E156 recipe; v0.4.3 bundle source; captured sm_120 target; product command-buffer
  XLA_FLAGS; receipts `DEVICELESS_AB_PROD_d0{1,2}.json`]. For each domain, OFF / V1 / V2 were compiled from one lowered program:

  | Program | Optimized HLO | Native records 0/1/2 | Stored 208 / 258 |
  |---|---|---|---|
  | PROD d02 | 6e6bab37 ×3 identical | identical; record 0 c383bcdb equals the real GPU executable | 5 / +WHILE · 1 / +WHILE · 5 / −WHILE |
  | PROD d01 | d8d9bf04 ×3 identical | identical; record 0 965fa810 equals the real GPU executable | same pattern |

  A toy CompileOnly proof (`tools/jit_options_merge.py`) shows that jit-level `compiler_options` land in the serialized
  executable and keep the AOT path's option 435.
- Prototype on lane/o5-perf 32e037dc78:
  - `GPUWRF_STEP_CMDBUF=off|v1|v2`, default off.
  - `operational_mode.step_jit` re-wraps `_advance_chunk_fori`, `_advance_chunk_static_scan` and the fused-cascade jit only
    when the switch is on.
  - The switch is registered in `aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS`.
  - `tests/test_step_cmdbuf.py`: 4 PASS, deletion-sensitive both ways. Cheap-key env-registry tests: 4 PASS.
- **v0.4.4 integration path (after the cbab A/B picks V1 or V2):**
  1. Add `"GPUWRF_STEP_CMDBUF": "v1"` (or `"v2"`) to `src/gpuwrf/_fast_defaults.py:FAST_PATH_DEFAULTS`, the same mechanism
     as `GPUWRF_CARRY_DONATE`. `GPUWRF_STEP_CMDBUF=off` remains the opt-out. Update the release-default list tests (E183) and
     document the switch in docs/PERFORMANCE.md.
  2. The cheap key moves for all stepping programs (registered import-time constant), so the v0.4.4 bundle must be re-primed.
     That is needed anyway for the topo fix. The deviceless gate says the native code is unchanged.
  3. If cbab shows wrfout bytes identical to the sealed v0.4.3 receipts, the existing v0.4.3 numerical validation carries over
     under the validation-reuse rule (AGENTS §4). Only RW1 (warm, 0 compiles), the changed-artifact E41 scan and timing then
     need re-running for this change.
  4. Risk notes:
     - With V1, d02 becomes one WHILE graph per root. Its ≈ 2,000 kernel nodes are updated per call when buffer addresses
       change (d01 today: 1,253 `SetParams` per root, ≈ 0.9 ms host). This work is asynchronous before launch.
     - A42 measured conditional-node overhead on radiation band loops (+≈ 10 ms GPU per radiation root, LW7 tree). d01 already
       runs the same way in production, and cbab measures it for d02.
- Fix candidates for the deviceless + GPU gate (manager 16:36Z): per-jit `compiler_options` on the stepping programs only.
  - **V1:** `xla_gpu_graph_min_graph_size=1`.
  - **V2:** `xla_gpu_enable_command_buffer` without WHILE.
  - JAX merges jit-level options with the AOT path's `lowered.compile(export_compile_options())`
    (`pxla.py:1220`, `compiler_options_kvs = jit + call`).
  - The source change moves the cheap key (source fingerprint), so the programs get their own cache, and v0.4.4 needs a new bundle
    (needed anyway for the topo fix).
- **Next step:** a GPU VLOG probe (§6, P1).
- Estimated value [I]: see the upper bound above, **0.3–0.6 s/fc-h**. WN3 is not yet checked; its d02+d03 run as one fused program.

## 4. Radiation, KF, output, startup, compile

- Radiation (R43, night) [M], extra device time per call:
  - d01 +17.9 ms (LW 10.5, McICA sampling 3.7, SW 1.9);
  - d02 +42.9 ms (LW 27.1, sampling 9.6, SW 4.9).

  At 2 calls/h per domain this is ≈ 0.12 s/fc-h. KF adds ≈ +0.7 ms per call, ≈ 0.01 s/fc-h. McICA sampling is 22 % of the
  radiation time.
- Output and join:
  - PROD, history every hour [M HERO]: 0.11 s of callback union per steady hour, first-hour peaks 0.8–1.2 s; finalization
    after the last segment 0.9–1.2 s. Small.
  - WN3 3-nest, FULL 377-variable history [M, FINDINGS A68 SI50]: forecast frames 1.56 busy s/fc-h, t0 materialization 2.0 s,
    final join 0.88 s, 0.4 GB/case-h.
  - These are busy times that overlap the GPU (async writer thread). In GPU-saturated par3 they cost host CPU, not GPU time.
  - E196: bench "tail" includes the final proof/finite summary, not only the NetCDF join.
- **Startup: 29.8 s from launch per PROD process** [M HERO H1, v0.4.2 = v0.4.3 programs]. This is the CLI allocator re-exec
  2.6 s before t0, plus 27.2 s from t0 to the first segment. Within those 27.2 s:
  - JAX import 0.64 s; CLI/harness wrappers to 2.40 s.
  - **1,094 JIT compile requests, all persistent-cache hits.** They cost 11.2 s of compile-path time on the main thread:
    lowering 6.44 s, tracing 1.19 s, executable load 1.96 s, cache retrieval 1.59 s. JAX's `compile_time_saved` is 31.4 s.
  - They are overwhelmingly one-op eager programs. Examples: 238 `convert_element_type`, 224 `broadcast_in_dim`,
    182 `add`, 180 `multiply`, 132 `subtract`, 128 `true_divide`, 120 `dynamic_slice`. Larger ones are 22 `jit(wrapped)` at
    1.88 s and `_coupled_updates` at 0.69 s.
  - Each first use of a primitive at a new shape/dtype is traced, lowered and fetched from disk at ≈ 10 ms. The persistent-cache
    key needs the lowered HLO, so a cache hit never skips lowering.
  - Event bursts occur at t = 1–3, 6–12 and 17–26 s, with 4 s and 1 s gaps of no compile activity (input read, AOT load,
    initial radiation).

  w44 R43 (3 h proxy, 18.3 s pre-segment) confirms the same 1,094 lowerings (4.07 s lowering only). Six hydrostatic Pallas
  wrappers take 1.34 s of that, and the first SW init setup takes 2.98 s (the source-fingerprint walk; merged fix −2.69 s).
  **Lever S-1:** run domain init as a few AOT-keyed programs, or as host NumPy, instead of ≈ 1,000 eager dispatches.
  Expected [I]: −8…−10 s per process ≈ −0.35 s/fc-h on 24 h PROD (−7 % S1), ≈ −3 s/fc-h on a 3 h run, ≈ −0.2 s/case-h in WN3 par3.
- **WN3 par3 release benchmark** [M HERO, v0.4.2 programs, 3 × 24 h concurrent, 448.3 s = 6.23 s/case-h]:
  - The first file (t0 history at init) appears at +50…58 s after launch: concurrent init of 3 cases, 3 domains each.
  - Then 16.0–16.4 s per case per forecast hour with 3 cases concurrent, i.e. 5.4 s per case-hour aggregate.
  - Startup is 12 % of the batch wall (0.75 s/case-h). The steady part, 5.47 s/case-h, is GPU-saturated (D25), so the
    **device-time levers (bytes), not host gaps, set the benchmark**.
  - Per-case pool 4.17 GiB (C-auto), per-PID VRAM 5,236 MiB, host VmHWM 6.2–6.3 GB.

## 4b. S3 device ops per own step, compile and cached readiness

- S3, ordinary own step (N43 node trace) [M]:
  - d01: 1,335 kernels + 5 copies ≈ **1,340** ops;
  - d02: 652 kernels + 195 DtoD + 4 small copies ≈ **851** ops.

  Cadence-weighted hourly means [I, from the R43 radiation roots: d01 +4,153 kernels, d02 +7,924 kernels per radiation call,
  2 calls/h]: d01 ≈ 1,470, d02 ≈ 930. **S3 fails** both the committed ≤ 500 and the stretch ≤ 150. The device-time view in §2
  shows why counting launches alone is not the lever (E10). The countable part is the 1,761 launches/root below 4 µs (3.05 ms).
- Cold compile (w44 prime arms, PROD d01 + d02, GPU) [M]: v0.4.0 backend 254 s, process 335 s; v0.3.2 backend 312 s,
  process 390 s. R3 cached readiness (process → first segment, warm sealed bundle) 18–27 s ≤ 60 s: **PASS**.

## 5. Memory anatomy (first part)

Bundle C-auto memory plans [M, compiled memory stats of the actual GPU executables]:

| Case | Program | Arguments | Outputs | Aliased | Temporaries | Code | Live peak plan | Budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PROD | d01 | 127 MB | 115 MB | 115 MB | 525 MB | 129 MB | 1.98 GB | 3.23 GB |
| PROD | d02 | 347 MB | 304 MB | 304 MB | 643 MB | 134 MB | | |
| WN3 3-nest | d01 | 127 MB | 115 MB | 115 MB | 525 MB | 129 MB | 2.89–2.94 GB | 4.47–4.52 GB |
| WN3 3-nest | fused d02+d03 | 452 MB | 407 MB | **0** | 576 MB | 257 MB | | |

- **The WN3 fused cascade does not donate its carry [M source + plan].** `domain_tree.py:2348` is `@jax.jit def fused_jit(...)`
  without `donate_argnums`, even though `GPUWRF_CARRY_DONATE=1` is a release default. So 452 MB of inputs and 407 MB of outputs
  are live together. Donating would save ≈ 0.4 GB per WN3 case (≈ −14 % of the live peak) [I]. It may also save the entry copies
  of the while-loop carry (cf. A40/A65: −1.9 ms/root for PROD). Caller audit needed: async OutputSnapshot by reference, restart
  capture.
- HERO PROD per-PID VRAM peak 3,898 MiB, JAX peak 1.995 GB, host VmHWM 5.2 GB [M].
  R1 against the 12-rank CPU-WRF RSS of 9,903 MiB = **0.39×** (target ≤ 1.25×, PASS) [M/I].
- **The deviceless compile reproduces the actual GPU memory statistics exactly** [M]. memprobe of PROD d02 from the v0.4.3
  bundle source, captured sm_120 target, autotune 0: arguments 346,875,860 B, outputs 304,192,092, aliased 304,190,732,
  temporaries 642,901,760. These are byte-identical to the GPU C-auto plan. Deviceless compiles are therefore a valid R2 / VRAM
  instrument (E34 extended to CompileOnly).
- What fills the temporaries (XLA buffer assignment of PROD d02) [M]:
  - The single 643 MB preallocated-temp block is dominated by RRTMG column-tile workspaces at the 4,096-column tile:
    `f32[56,4096,140]` / `pred[4096,57,140]` (122.5 MB each, LW g-points), `f32[44,4096,112]` (SW), `f32[4096,45,14,12]`,
    `f32[4096,57,16]`.
  - Allocations with tile shapes total 843 MB (they are reused within the block), against 437 MB of 3-D field temporaries.
  - These workspaces are fixed per program once a domain has more than 4,096 columns, and they scale with levels. Temporaries
    are therefore sub-linear in columns but linear in `e_vert` (R2 below).
  - The block is allocated on every execution of the step program, although radiation runs only twice per hour.
- C-auto pool policy [M source `runtime/gpu_allocator.py`]: budget = (max(Σ program footprints, measured live peak) +
  speculative code allowance) × 1.20 + 0.5 GiB, plus ≥ 1 GiB outside the pool for context and modules.
  - PROD: live 1.98 → pool 3.23 GB → per-PID 3.9 GB.
  - WN3: live 2.94 → pool 4.47 GB → per-PID 5.2 GB.

  That is ≈ 45 % over the JAX live peak. With the desktop at ≈ 4 GB this allows 5 WN3 cases per 32 GB, but par3 throughput
  already saturates at 3 cases (D25). Capacity is therefore not today's benchmark limit: tighter safety factors only pay for
  larger domains (N333).
- R2 and capacity projection [I until the probes report]. Measured per-program coefficients (GPU plans = deviceless,
  44 levels):

  | Program | Columns | Arguments B/cell | Temporaries |
  |---|---:|---:|---:|
  | root d01 (KF) | 8,400 | 344 | 525 MB |
  | leaf child d02 | 31,239 | 252 | 643 MB |
  | fused d02+d03 | 41,562 | 247 | 576 MB |

  - Arguments (carry) are linear in cells, α ≈ 1.
  - Temporaries are dominated by the 4,096-column RRTMG tile workspaces: ≈ constant in columns above 4,096 and ∝ levels.
    That gives α ≈ 0.3 over 8k → 31k columns for the whole footprint.
  - **R2 [M, deviceless = GPU-exact (MEM01)]** (`receipts/R2_SCALING.json`):

    | Program type | Small | Large | α footprint | α carry | α temps |
    |---|---|---|---:|---:|---:|
    | leaf child | WN3 d03, 454k cells | PROD d02, 1.37M cells | **0.43** | 0.97 | 0.24 |
    | root (different physics) | Swiss, 77.6k cells (args 352 B/cell, temps 224 MB) | PROD d01, 370k cells | **0.61** | — | — |

    **R2 (α ≤ 1.10) PASS.** Carry is linear; temporaries are nearly flat once a domain exceeds the 4,096-column tile.
    (The WN3 fused deviceless probe failed on the w44 closure-inspection recipe, `KeyError('fused_jit')`; its actual GPU plan
    above is used instead.)
  - Projection recipe for a new geometry: live ≈ Σ_domains 250–345 B × nx·ny·nz + per-program temporaries (≈ 0.55–0.65 GB
    × nz/44) + boundary and output buffers (≈ 0.3–0.5 GB). Pool = (live + code) × 1.2 + 0.5 GiB; per-PID = pool + ≥ 1 GiB.
  - N333 4-nest (9/3/1/0.333 km) + e_vert 85 [I, geometry pending the GuardedRead admission posted 15:36Z]: levels ×1.93 doubles
    both carry and temporaries. With a d04 of ≈ 20–30k columns, the WN3 live peak of 2.9 GB becomes ≈ 6–7 GB, so per-PID
    ≈ 9–10 GB and ≈ 3 cases per 32 GB GPU. A deviceless compile of the real 4-nest inputs would replace this with exact
    bytes in ≈ 30 CPU-min.

## 6. GPU plan (immutable launchers in `<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch/`, copies in `proofs/o5_perf/launch/`)

- **P1b cbab (H-CB A/B), cap 55 min (expected ≈ 32), smoke PASS, posted.** Entry `o5_launch_cbab.sh cbab`, sha c45bb1c5.
  - OFF: the sealed v0.4.3 bundle, warm. Its warm-up arm W0 carries the conversion-pass VLOG, which confirms the cause at
    runtime and supersedes cbprobe for PROD.
  - V1/V2: the prototype snapshot `snap_cbfix` (sparse, 0.96 GB, commit 32e037dc78), with a private cache seeded with the
    bundle's 1,228 tunes.
  - Arms: W0, 2 cold primes, 9 interleaved measured arms, 3 graph-level nsys arms.
  - Gates: wrfout SHA of every arm == W0; d02 host call time and seg 2–3 ms/root per variant (A/A ≤ 5 %); nsys graph counts.
  - (P1 cbprobe, sha 10f3894a, stays valid if a WN3 VLOG run is wanted.)
- **P0 cbmini (CUDA mini, compile-free H-CB check), cap 5 min, CUDA MINI READY (sha 62cf329f).** One consumer view runs
  M_OFF (sealed, VLOG), then copies the patched PROD d01/d02 AOT entries into the view (only DebugOption 208 5 → 1; native
  records asserted identical; `CBMINI_PATCH_RECEIPT.json`), then M_V1P. Checks: wrfout SHA == sealed B6 receipt, d02 host
  ms/call, VLOG counts. Post-run: `cbmini_analyze.py` (which AOT source each domain loaded).
- **P2 bisect (v0.4.1 vs v0.4.3), cap 45 min, smoke PASS.** Pre-registered in `BISECT_PLAN.json`. Entry `o5_launch_bisect.sh bisect`.
  - Covers the split that the w44-perf repeat (v0.3.2/v0.4.0/v0.4.3, PROD, S0 harness) does not.
  - Uses the product CLI / product parallel runner of each sealed bundle: PROD 3 h solo × (1 warm-up + 3 measured) and
    WN3 par3 3 h × (1 + 2).
  - Interleaved; A/A ≤ 5 % per version and metric; wrfouts deleted per arm.
- **P3 mps (CUDA MPS for WN3 par3), cap 25 min, smoke PASS.** Entry `o5_launch_mps.sh mps`.
  - Same sealed v0.4.3 programs and product parallel runner. The only change: the 3 case processes connect to a private
    MPS daemon (spatial sharing instead of time-sliced CUDA contexts).
  - Arms N0 (warm-up), M1, N1, M2, N2; WN3 0227/0502/0614 for 6 h each; the wrfout SHAs of all arms must be identical.
  - Rationale [M/I]:
    - solo WN3 runs at 8–11 s per simulated hour (bundle priming receipts), par3 at 16 s per case-hour;
    - par3 aggregate is 5.4 s/case-h against an estimated device-sum of ≈ 4.2 s/case-h [I, PROD per-domain kernel time
      scaled to WN3 column-steps];
    - many WN3 kernels underfill the 170 SMs: one-thread-per-column Thompson/MYNN/BouLac, the 1 km d03 at 10k columns, and
      1,761 sub-4 µs launches per PROD root.
  - MPS was never tried (FINDINGS/EXPERIMENTS/INBOX). Potential 10–25 % of release-benchmark throughput [U] at zero numerical
    change.
- **Not planned without an Owner/manager decision:**
  - v0.3.3 (snapshot + cold prime ≈ 25 GPU-min; v0.3.4 is jaxpr-identical to it);
  - v0.4.0 WN3 par3 (its nested programs are not date-blind, so 0502/0614 would cold-compile);
  - a WN3 node-trace (family anatomy of the release workload; needs the S0 capture harness on a 3-nest case).

## 7. Ranked levers (expected s/fc-h on PROD 24 h S1; [I] unless noted)

| # | Lever | Expected gain | Status / cost / risk |
|---|---|---:|---|
| 1 | **H-CB: restore d02 command-buffer capture.** `GPUWRF_STEP_CMDBUF=v1` (min graph size 1) or `v2` (no WHILE), per-jit on the stepping programs only. Targets the ≈ 8.7 ms/root host/dispatch idle; d02 has been fully host-launched since v0.3.3. Recovery opportunity, not the cause of the v0.3.2→v0.4.x regression (joint reading with w44-perf) | **0.25–0.5** (ceiling ≈ 0.58 = whole idle) | Root cause [M]; deviceless gate GREEN (identical HLO + native code); prototype + CPU tests on lane/o5-perf; CUDA mini ready; GPU A/B cbab 25.10; Astra review. For v0.4.4: default flip after A/B; new bundle (cheap key moves) |
| 2 | **Startup S-1 exact part:** load the remaining large init programs through the existing `integration/init_kernels.py:InitKernelCache` (cheap-key AOT, as for the RRTMG init kernels). Candidates: 8 eager `hydrostatic_profiles` Pallas calls (1.82 s/process [M R43]), `build_child_boundary_package` 0.39, MYNN cold start 0.18, BouLac 0.12, `_to_columns` 0.08 s. Same executables, byte-exact | 2.6 s/process ≈ 0.11 (24 h), ≈ 0.9 (3 h) | small–medium |
| 3 | **Startup S-2:** avoid the 2.6 s C-auto CLI re-exec (`cli.py:_maybe_reexec_for_nested_allocator`, `os.execvpe`). JAX reads `XLA_PYTHON_CLIENT_*` (set by `gpu_allocator.configure_cli_pool`) when the backend initializes, and the product keeps imports backend-dark (E190/SI54). So when `jax._src.xla_bridge.backends_are_initialized()` is False, an in-process `os.environ.update(new_env)` gives the same pool; keep today's re-exec as the fallback. No numerics change | ≈ 0.11 (24 h), ≈ 0.9 (3 h) | small (receipts record `cli_reexec`) |
| 4 | **Startup S-1 rest:** ≈ 1,000 one-op eager init dispatches (3.9 s compile path + ≈ 1.5 s cache retrieval [M]). Jitting the top ≈ 20 eager callers (`physics_couplers` 375 lowerings / 1.39 s, Noah init, ozone, MYNN init …) is **not** byte-identical in general (E8), so it needs D6 | ≈ 0.15–0.2 (24 h), ≈ 1.5 (3 h) | medium, fidelity gate |
| 5 | D-1: native/fused child moisture-scalar stage (`_nested_scalar_stage_tendencies` + `_apply_moisture_large_step`, 3.47 ms/root in 27 XLA launches; transpose emitter at 0.56–0.68 TB/s) | ≈ 0.13 (PROD); ×2 child domains on WN3 | medium (layout match E175; PD fidelity gates) |
| 5b | Column-kernel occupancy: Thompson (33 % of peak), MYNN tridiagonal/BouLac (≈ 30 %, one thread per column, E150: batch independent solves / level-parallel), PD limiter (29–35 %), `boundary_spec_final` (15 %). Measured headroom vs the unique-byte floor ≈ 4.5 ms/root | ≈ 0.2–0.3 (PROD); more on WN3 (1 km d03) | medium–high (kernel work, WRF-oracle gates, four-eyes) |
| 6 | Between-executable host work (3 ms/root) + per-call d01 graph updates (1,253 node updates/root): stable buffer addresses / fewer executes. Partly absorbed by #1 once d02 is async | 0.1–0.2 | medium (A57/A61/A65 lineage) |
| 7 | Layout bytes: move physics↔dycore layout conversions into native kernels (8.55 GB/root of transposes) | 0.1–0.3 | high (kernel work, four-eyes) |
| 8 | Tiny-kernel glue (1,761 launches/root < 4 µs) | ≈ 0.1 | medium |
| 9 | WN3 fused-cascade donation | memory −0.4 GB/case; speed small | low; caller audit |
| 10 | Radiation McICA max-random CDF as one column kernel (≈ 12 ms and ≈ 2,900 launches per radiation pair) | ≈ 0.02 | low priority |
| 11 | Per-RK-stage fused kernels (restructuring; §2 byte budget) | −30…−40 % device time ≈ 0.7–0.9 | large (v0.5+ rewrite item) |
| B1 | **Release benchmark (WN3 par3): CUDA MPS for the parallel cases (P3)** | 10–25 % of s/case-h [U] (≈ 0.5–1.3 s/case-h) | GPU A/B only; numerics unchanged by construction (SHA check) |

Sum of the exact, small-to-medium items 1–3: ≈ 0.5–0.8 s/fc-h on 24 h PROD (S1 4.7 → ≈ 4.0) [I]. That is about the v0.3.2 level
(3.3) plus the startup that grew since then.
