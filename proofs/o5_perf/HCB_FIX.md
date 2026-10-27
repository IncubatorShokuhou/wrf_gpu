# H-CB: lost CUDA-graph capture of the PROD d02 step program — cause, fix prototype, gates

Lane o5-perf, 2026-10-24. Labels: [M] measured, [I] inferred. Code: lane/o5-perf 32e037dc78 (prototype); all tools and
receipts in `proofs/o5_perf/`.

## Symptom [M]

- **Node trace.** w44-perf N43 (v0.4.3 bundle, three ordinary PROD roots):
  - all 1,950 d02 kernels per root are host-launched (`graph_id` none, `cuLaunchKernelEx`);
  - 585 DtoD copies per root run as separate API calls;
  - d01 is one graph per root.
- **Graph-level trace.** G43 (one segment):
  - only 3 graphs exist in the whole process (`cuGraphCreate` 3);
  - 146,535 `cuLaunchKernelEx` calls and 17 `cuStreamSynchronize` per root.
- **v0.3.2 (N32).** d02 had 1,671 graph-captured + 132 host kernels per root.
- **Same session, unprofiled.** w44 Oct-23 S0 arms (`receipts/HOST_CALL_TIMES_OCT23.json`), median host time per call. The
  times include blocking on in-call syncs:

  | Version | d01 call | d02 call | Per-root sum |
  |---|---:|---:|---:|
  | v0.3.2 | 5.9–7.4 ms | 30.0–31.4 ms | 39.7–43.8 ms |
  | v0.4.0 / v0.4.3 | 2.7–5.2 ms | 35.4–37.8 ms | 44.7–45.4 ms |

  In v0.4.x the d01 call returns after an asynchronous graph launch, and the d02 call absorbs the wait for d01's device work.
  The per-call d02 growth is therefore mostly a blocking shift. The per-root sum is the comparable scope.
- **Joint reading with w44-perf** (`proofs/w44_perf/PERF.md` 5b083b4a81, `HCB_RECONCILE.json`) [I, mixed scopes]:
  - The mean wall delta v0.3.2 → v0.4.3 is +3.4 ms/root, the same as the device-union delta of +3.43 ms/root. The regression
    is not resolved on the host side.
  - Wall minus ordinary-root device union is ≈ 11 ms/root in both versions. Of that, ≈ 2 ms/root is radiation/KF extra device
    time, and ≈ 8.7 ms/root is device-timeline idle (G43 gap union).
  - H-CB changed **where** the idle comes from (v0.3.2: 20 + 21 command buffers around 10 + 11 host whiles; v0.4.x: d02 fully
    host-launched), not measurably its size. H-CB is a recovery opportunity, not the cause of the regression.
  - My 17:00Z wording "host-bound; H-CB explains the regression" is withdrawn.

## Cause [M source + CPU replay]

Pinned XLA b6f37ab, `xla/backends/gpu/runtime/command_buffer_conversion_pass.cc` (copy:
`receipts/xla_b6f37ab_command_buffer_conversion_pass.cc`, lines 466–600):

1. `Run()` appends a **convertible** `WhileThunk` to the pending sequence and `continue`s. It does not descend into the
   body; only a *non*-convertible While/Conditional is recursed into.
2. `FlushCommandBuffer()` leaves a pending sequence with fewer than `max(1, xla_gpu_graph_min_graph_size)` thunks
   unconverted. The stored DebugOptions field 208 is 5.
3. Since v0.3.3 (6e38cbf173), `--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` makes the outer step loop convertible.

   - PROD d02's top level is `[copy.1668, wrapped_convert, wrapped_convert.1, while.793]`, i.e. 4 thunks. It is flushed
     unconverted, so the **whole** program runs op by op, with the host-driven outer loop and its DtoH syncs.
   - d01's top level has 12 thunks (8 root layout-pin transposes, …) and becomes one WHILE command buffer.
4. Upstream XLA `main` fixed this (`CountCommandBufferSize` walks nested thunks).

**Replay.** `cb_partition_sim.py` re-implements the pinned `Run`/`Flush`/`IsConvertible` on the serialized compiled-thunk
trees. It censuses every entry (no unnamed or unknown kinds) and reproduces the traces exactly:

| Program | OFF | V1 (min 1) | V2 (no WHILE) |
|---|---|---|---|
| PROD d01 | 1 CB, 1,651/1,651 | 1 CB | 22 CBs, 12 host whiles |
| **PROD d02** | **0 CB, 0/1,212** | 1 CB, 1,212/1,212 | 22 CBs, 11 host whiles |
| WN3 fused d02+d03 | 1 CB, 2,605/2,605 | 1 CB | 41 CBs |

The replay now reads each executable's **stored** options (DebugOptions 258/208) instead of assuming them. It agrees with
the measured traces on two versions:

| Executable | Stored command-buffer set | d01 | d02 | Measured |
|---|---|---|---|---|
| v0.4.3 bundle | …, CONDITIONAL, **WHILE**; min 5 | 1 CB, 1,651/1,651 | **0 CB, 0/1,212** | N43: d01 one graph, d02 all host-launched |
| v0.3.2 (w44-perf `cache_v032`) | …, CONDITIONAL (no WHILE); min 5 | 20 CBs, 1,252/1,280, 10 host whiles | 21 CBs, 1,034/1,065, 11 host whiles | N32: d02 1,671 graph + 132 host kernels/root |

The counts are static thunk counts, not kernel launches, so they agree with N32 only qualitatively (`CB_PARTITION_SIM_V032.json`).

Scope [I]: per-domain non-root step programs. That covers PROD-type 2-domain runs and all domains of two-way (feedback) runs.
The WN3 release workload is unaffected. FINDINGS A42 ("WHILE gave no gain on LW7; the outer step while loops stayed host
WhileThunks") is the same mechanism.

## Fix prototype (default off)

`GPUWRF_STEP_CMDBUF=off|v1|v2`, resolved at import and registered in `aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS`.
`operational_mode.step_jit` adds the options only to the **top-level** dispatch objects of the stepping programs:
`_advance_chunk_fori` (with the carry donation when `GPUWRF_CARRY_DONATE=1`), `_advance_chunk_static_scan`, and the
fused-cascade `fused_jit`.

**Four-eyes fix F1** (w44-perfplan 21:31Z; lane/o5-perf 142dfebfe9, rebased on main f401f1af72):
- JAX 0.10 refuses `compiler_options` on a jit that is traced inside another jit (`pjit_staging_rule`).
- Several callers stage the step program into their own jit: fused cascades (the default for 3+ nests, i.e. WN3), vmapped
  lanes, and the parallel tile drivers. The first prototype would have crashed there.
- Those callers now get `_advance_chunk_fori_nested` / `_advance_chunk_static_scan_nested`. These are exactly the switch-off
  objects (same donation), so the staged programs are unchanged.
- `_advance_chunk` picks the nested object when any carry/start/clock leaf is a Tracer. The check runs only when the switch
  is on.
- `parallel/harness.py` and `parallel/tile_driver.py` call the nested object directly.

| Variant | Per-jit compiler options |
|---|---|
| v1 | `xla_gpu_graph_min_graph_size=1` |
| v2 | `xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,CONDITIONAL` (v0.3.2 set) |

With the switch unset, the jit objects are the same as on main (`*_nested` is the same object).

## Gates

| Gate | Status |
|---|---|
| Deviceless (E156; v0.4.3 source, captured sm_120, product cmdbuf flags): OFF/V1/V2 optimized HLO + native records 0/1/2 identical, only fields 208/258 differ | **GREEN** [M]: d02 HLO 6e6bab37 ×3, d01 d8d9bf04 ×3; record 0 equals the real GPU executable (`DEVICELESS_AB_PROD_d0{1,2}.json`) |
| jit-level options reach the executable and merge with the AOT path's call options | **GREEN** [M] toy CompileOnly proof (`tools/jit_options_merge.py`) |
| CPU tests (F2) | `tests/test_step_cmdbuf.py`: toy compiled options off/v1/v2/invalid + master switch; real objects under off / v1 / v1 + CARRY_DONATE (options on top-level objects only, nested objects option-free, donation equal); dispatch top-level / inside `jax.jit` / inside `jax.vmap` for fori and static_scan, plus the precondition that a nested option-jit raises, plus a dispatch mutant (always top-level) that is caught; fused cascade (small WN3-type nest) lowered under OFF and under v1 + CARRY_DONATE, then compiled on CPU: HLO equal and compiled fused options carry min graph size 1. Status: fast tests 10 PASS; fused + suites under v1 → see the READY line |
| CUDA mini (compile-free: sealed OFF vs patched V1p entries, field 208 only) | READY, cap 5 min, sha 62cf329f |
| GPU A/B cbab (F3, E176): every scored arm on ONE tree, `lane/o5-perf-cbab` a833b78686 = sealed v0.4.3 source 888804310b + the fix `src` diff (5 files), private cache seeded with the bundle's tunes. Arms: W0 sealed bundle (bytes control + VLOG); P0 SOFF cold, P1 V1 cold; SOFF_1 V1_1 V1_2 SOFF_2 SOFF_3 V1_3 warm, fresh processes; N_SOFF N_V1 graph-level nsys; V2 (P2, V2_1, V2_2, N_V2) only while the deadline allows. Gates: PRIMARY V1 bytes == SOFF; CONTROL W0 == SOFF; V2 == SOFF; seg2–3 ms/root per variant (A/A ≤ 5 %); idle per root from the nsys arms (G43 method, CPU afterwards) | cap 55 min (launcher c45bb1c5 unchanged; payload `cbab_session.sh` new sha in the READY line); 25.10 window |
| Release (v0.4.4): default flip = branch `lane/o5-perf-flip` commit 76e75bcd2c (one commit on 9cb3f842ad, cherry-pickable): `"GPUWRF_STEP_CMDBUF": "v1"` in `_fast_defaults.FAST_PATH_DEFAULTS` + the extras list of `tests/test_fast_defaults.py` (as `GPUWRF_CARRY_DONATE`) + `test_step_cmdbuf` default v1 / master opt-out off; CPU with the patch applied: `test_step_cmdbuf` + `test_fast_defaults` 11 PASS; `test_gpu_command_buffer_injection_equals_the_flag_file_xla_string` stays green (per-jit, XLA_FLAGS unchanged); bundle re-prime (cheap key moves); RW1–RW4; numerical validation reuse if bytes are identical | after cbab |

## Expected effect and risks

- Expected [I]: V1 makes d02 one asynchronous WHILE graph. The G43 idle components set the range:
  - lower end, −3.5 ms/root: only the inter-kernel gaps of the host-launched d02 kernels go away;
  - upper end, −7.5 ms/root: the 3.0 ms/root of host work between executables also overlaps with the device, because the
    d02 call no longer blocks. A per-root host read of a device value would cap this [U].

  That is ≈ −0.25…−0.5 s/fc-h on PROD (≈ −5…−10 % of S1); the ceiling is the whole idle, ≈ 8.7 ms/root ≈ 0.58 s/fc-h. w44's
  0.745 ceiling is the residual including ≈ 0.13 s/fc-h radiation/KF device time.
- V2 restores the v0.3.2 partition. Because the residual did not measurably change between versions, V2 is expected to be
  ≈ OFF within noise. It is measured as a control.
- cbab decides empirically: per arm, root wall, the nsys device union per root, and residual = wall − union.
- Risks:
  - Conditional-node overhead on d02 radiation roots (A42-type, +≈ 10 ms per radiation root on LW7; d01 already runs that way).
  - Per-call graph updates of ≈ 2,000 d02 nodes (asynchronous host work, ≈ 1.5 ms).
  - Every stepping cheap key changes, so a new bundle is needed (it is needed anyway for the topo fix).
- N1 (four-eyes): merging the default-OFF fix ALREADY moves every step/fused cheap key, because the new import-time
  constant `_STEP_CMDBUF` is registered in `aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS` (E80 rule) and so enters
  `module_const_env_hash`. The bundle re-prime is due at the merge, not only at the flip.
- N5: the staging check looks for Tracer leaves. A caller jit that closes over a CONCRETE carry would reach the
  option-carrying object inside a trace; JAX then raises loudly (ValueError), it never runs silently wrong.
  No such caller exists today; `jax.core.trace_state_clean()` is the fallback if one appears.
- N6: `test_fused_cascade_compiles_under_v1_with_unchanged_hlo` peaks at ≈ 6.9 GB RSS, so it carries the repo's
  `heavy` marker (run with `--run-heavy` in a ≥ 8–12G RAM slot).
