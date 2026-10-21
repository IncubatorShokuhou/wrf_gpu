# w43-robust — ALISIOS P2 robustness items (fix list 2026-10-20, item 5)

Lane w43-robust, Opus 5.5, 2026-10-20/21. Labels: [M] measured, [I] inferred, [U] unknown.
Data: ALISIOS admission A-V043-ROBUST-E2-20250531 (8c3b84f9; CALIBRATION_READ), guarded reads R04 (94 files, SHA == admission),
our own journal (`analysis/journal/suspend_20261020_extract.txt`), our own sar history (`analysis/sar/`, local time WEST = UTC+1).
No ALISIOS observation truth was read. GPU: one 72 h rerun of item 1 (2026-10-21, lease w43-robust); items 2–3 no GPU.

| # | item | status | one-line reason |
|---|---|---|---|
| 1 | 20250531_18z_ERA5_r1: > 19 min instead of ~9, d03 only to +58 h | NOT_A_BUG (engine) — ENVIRONMENT by the frozen rule: quiet-disk GPU rerun 2026-10-21 564 s, 73/73/73 frames, rc 0 [M]; slowdown = <DATA_ROOT> I/O contention, +58 h stop = harness SIGKILL at its wall cap | same bundle/CLI/inputs on a quiet disk (nvme3n1 mean util 20 %) ran in normal time to +72 h on all 3 domains; on 2026-10-20 wall time of all 29 E2 runs tracked <DATA_ROOT> disk saturation (Pearson 0.876) and iowait on the GPU arm's single host core (0.894); 0531 had the worst window (88 % util, 109 ms await, 70 % iowait) and was SIGKILLed at 1180 s while still writing |
| 2 | WT1 wind at τ > 40 h "partly HELD" | NOT_A_BUG (engine) — ALISIOS governance rule | HELD = S0 rule R17 (wind admitted only for τ ≤ 40 h); the 40 h limit came from one event (L6, W3_0115 d03 V10 h42–44, present against CPU IC replicate R only); S0-A5 (2026-10-11) already re-admits d02/d03 wind τ 41–72 as a separate opt-in stratum after 2 GPU/CPU pairs at 0.08–0.30 m/s |
| 3 | behaviour after host suspend/hibernation 2026-10-20 15:16–21:11Z | NOT_A_BUG (engine) — host event + harness reaction; guard FIXED (23c9ee33c3, critic accepted, merged to main 21f726b3b7) | the NVIDIA hibernate hook suspends the GPU while user processes keep running (~3 min); the E2 harness's `nvidia-smi` probe hung 10 s → it SIGKILLed the running case; the dying process blocked the kernel freeze → first hibernation failed, NVRM unmap asserts on teardown; 7 queued cases each failed their 10 s `nvidia-smi` pre-probe. Wake-up at 21:11Z was clean |

## 1. 20250531_18z_ERA5_r1 (E2 GPU v0.4.2, 3-nest 9/3/1 km, 72 h)

**What happened [M, admitted receipts]**
- Attempt a1 13:18:36–13:38:29Z, v0.4.2 sealed bundle via `bundle_consumer.sh`, both executables loaded from AOT
  (`real_load=true`, series `compiled_new = 0`) → no compile, no cache miss, no record-count recompile.
- GPU free at acquire (2.5 GB used of 32.6 GB), device memory steady 7.8 GB, host RSS steady 5.9 GB → no GPU sharing, no memory growth.
- Inputs cover +0..+72 h (their INPUT_MANIFEST QA: 12 boundary intervals, 13 forcing endpoints; wrfbdy/wrflowinp sizes normal vs 31 other cases).
- Killed by the E2 harness: `FAILED_ATTEMPT_1.error = "stage wall cap"`, cap 1147.9 s, observed 1180.1 s, rc −9;
  their own diagnosis: "slow with continued output progress; no demonstrated GPU hang".
  Start 2025-06-01 00Z → d01/d02 58 frames (+57 h), d03 59 frames (+58 h): d03 simply stopped where the SIGKILL hit.
- Attempt 2 at 15:11:57Z failed after 10 s in the hibernation cascade (item 3). The case was never re-run.

**Why it was slow [M sar + receipts, causality I]**
- Output growth (scratch bytes, harness samples): 0.66 GB per 27 s at the start vs 1.1 GB per 27 s in the control 20240103 (539 s run),
  then ~0.1–0.17 GB per 35 s after ~12 min → the run was ~0.6× from the start and ~0.1× late.
- The E2 GPU arm runs pinned to ONE host core (core 4, `with_gpu_lock ... pinned to GPU-arm cores 4`).
  sar for that core (local 14:20–14:40 = 13:20–13:40Z): %iowait 73 % / 76 % (vs ~10 % in normal runs).
- <DATA_ROOT> disk (nvme3n1, where E2 writes raw wrfout): %util 94.5 / 89.7 %, queue 110 / 93, await 67 / 177 ms in the same windows;
  16 % / 26 % in the 10-min windows before/after.
- All 29 E2 runs of the day (`analysis/sar_correlation.json`, `analysis/sar_correlate.py`): wall time vs mean core-4 iowait over the run
  window Pearson **0.894**, vs nvme3n1 %util **0.876**. The other two slow runs also sat in disk-busy windows
  (20260119_ERA5_r1 837 s: 62 % util / 103 ms await; 20240323_r1 772 s: 78 % / 39 ms); runs at ≤ 25 % util took 467–525 s.
- Why the engine feels the disk: one 3-nest 72 h case writes the FULL 375-variable history, ≈ 299 MB per output hour
  (d01 51 MB + d02 185 MB + d03 63 MB, uncompressed, fsync per file) ≈ 21.8 GB per case ≈ 44 MB/s at normal speed.
  The async writer has one thread and a bounded queue (`max_pending=2`, `io/async_wrfout.py:86`, `nested_pipeline.py:4051`), so a slow
  disk back-pressures stepping by design (bounded host RAM). %util measures device busy time, not throughput, so the run's own share of
  the busy time cannot be separated from foreign load with sar alone; the foreign writer is UNKNOWN.
- Who saturated <DATA_ROOT> at 13:20–13:40Z: [U] (sar has no per-process I/O history).

**Classification: ENVIRONMENT → NOT_A_BUG (engine) [M rerun].** Measured: the +58 h stop is the harness SIGKILL at its fixed wall cap
(≈ 2.2× median), with no compile, no input truncation and no GPU sharing; the same case on a quiet disk runs in normal time to +72 h
(rerun below), so neither the slowness nor the stop is case-intrinsic. Cause of the slowness: shared <DATA_ROOT> contention [M correlation +
quiet-disk control]; who saturated the disk on 2026-10-20 13:20–13:40Z stays [U]. The engine's sensitivity to a saturated disk is a property
of the bounded writer (documented, not changed).

**Recommendations to ALISIOS (their harness):** (a) wall cap by output progress (e.g. kill only if no new wrfout for N min) or ≥ 3× median;
(b) long cases with `--checkpoint-dir … --checkpoint-interval-steps …` so a killed case resumes with `--resume-checkpoint` instead of
restarting (P1 restart path, nested); (c) ≥ 2 host cores for the GPU arm (one core is shared by the dispatch thread and the writer thread);
(d) raw wrfout on a less contended filesystem, or avoid other heavy <DATA_ROOT> I/O during GPU series.

**Quiet-disk rerun RB03 [M] (2026-10-21 08:00:07–08:09:40Z real UTC, `analysis/rerun0531/`, score `result/RB03_score.json`).**
Same v0.4.2 sealed bundle via `bundle_consumer` (`compiled_new = 0`), same CLI (`--domains-from-namelist --emit-initial-history --hours 72`),
inputs SHA-identical to the admission (ADD1 + R04), GPU arm on ONE host core (taskset 3, like E2), raw wrfout on the same disk (nvme3n1, <DATA_ROOT>).
- rc 0, `PIPELINE_GREEN`, all domains finite; frames d01/d02/d03 = **73/73/73** (+0..+72 h), 21.8 GB written.
- Wall (CLOCK_MONOTONIC, launcher) **563.7 s**; receipt total 533.6 s, forecast-only 498.6 s. Normal E2 runs that day: 467–539 s.
- nvme3n1 during the run (113 × 5 s samples, includes our own writes): %util mean 20.4 / median 13.1 / p95 70.3 %, 12 % of samples > 50 %;
  write 64 MB/s mean, w_await 22.5 ms mean (2026-10-20 13:20–13:40Z: 90–95 % util, 67–177 ms await). No hibernation during the run.
- Frozen rule (pre-registered before the run): wall ≤ 600 s AND 73 frames/domain → **ENVIRONMENT**.

## 2. WT1 wind at τ > 40 h "partly HELD"

Sources (ALISIOS cross documents, read directly): `ATLAS/DATA_MAP_DETAIL/wrf.md` row WT1, `SURROGATE_S0_ERA_BINDING_V033_DRAFT_20261009.md`,
`SURROGATE_S0_ERA_BINDING_V033_AMENDMENT_A5_20261011.md`, `TO_CONSUMERS_S0_GENERATION_95c4d278_20261011.md`.
- HELD is an ALISIOS **admission status** of teacher rows, not a model-output state: S0 rule table R01–R16 admits wind (u10/v10/speed/vector)
  on d02/d03 only for τ ∈ [τ*, 40] h; R17 holds T2/wind outside that window (also R18 d01 not teacher scope, R19 moisture/cloud/rain held).
- The 40 h limit "stems only from L6: a single event (W3_0115, h42–44); the IC-P replicate passes" (A5). Our records: the 0115 V10 h42/h44
  rows were classified IC-R-specific, i.e. they appear against the CPU IC replicate R but not against replicate P (INBOX 2026-10-07
  mass-opus "Z0115 … class PROPERTY IC-R-specific"; A5 "das IC-P-Replikat besteht") → [I] realization sensitivity of that case;
  no GPU-vs-CPU engine defect was localized there (v0.3.3 release scoring, INBOX 2026-10-09 RD97).
- S0-A5 (2026-10-11) already re-admits d02/d03 wind τ 41–72 as `ADMITTED_DISCLOSED_LEAD` (separate EXT_41_72 file, opt-in), based on 2
  GPU/CPU pairs (0225, 0303) with d03 field RMSE GPU−CPU 0.12/0.30 m/s (τ 41–56) and 0.08/0.21 m/s (τ 57–72); re-HELD trigger > 1.5 m/s.
  "Partly HELD" therefore = default file (τ ≤ 40), τ > 72, τ < τ*, moisture R19 — all policy.
- Not a missing/constant field and not a time-alignment issue. The coastal WS10 +1.8 % (L8) is w43-coastwind's item.

## 3. Host suspend/hibernation 2026-10-20

**Timeline [M, our journal, UTC]** (`analysis/journal/suspend_20261020_extract.txt`)
- 15:07:33 E2 starts 20250426 (GPU arm on core 4).
- 15:09:25 logind "The system will hibernate now"; `nvidia-hibernate.service` runs `nvidia-sleep.sh hibernate`
  (`echo hibernate > /proc/driver/nvidia/suspend`, PreserveVideoMemoryAllocations=1 → VRAM saved to /var, 8.2 GB peak), done 15:09:29.
  User sessions are NOT frozen (`systemd-hibernate.service.d/nvidia-suspend-nofreeze.conf`: SYSTEMD_SLEEP_FREEZE_USER_SESSIONS=false),
  so processes keep running for ~3 min against a suspended GPU.
- 15:09:36 last 0426 gpu.log line; 15:10:10 E2 harness: `nvidia-smi --query-gpu=memory.used` "timed out after 10 seconds" → case killed (rc −9).
- 15:10:16–15:12:09 seven queued cases (0302, 0510, 0626, 0502, 0520, 0608, 0531-attempt-2) each fail after ~10 s: lock still held by the
  dying 0426 wrapper, then their own `nvidia-smi` pre-probe times out.
- 15:12:34–54 kernel freeze fails: 8 tasks refusing, one `python3` (PF_EXITING) in `uvm_release → uvm_va_space_mm_shutdown`
  (a CUDA process tearing down against the suspended GPU [I: the killed 0426 engine]); "Failed to put system to sleep … Device or resource busy".
- 15:12:57 39 × `NVRM: clientUnmapResourceRefMappings: Failed to auto-unmap … nvAssertFailedNoLog rs_client.c:1194` = teardown of that client.
  No Xid. 15:13:15 `nvidia-resume.service`.
- 15:16:41 second hibernation succeeds (no GPU process); wake-up logged "21:11:04" clean (no NVRM/Xid lines after resume).
- After the wake-up the system clock is ~1 h FAST (chrony 2026-10-20: "System clock wrong by −3599 s", not stepped, slewing; manager
  CLOCK SKEW 22:18Z) → the logged "21:11:04" wake is ≈ 20:11Z real, i.e. the host slept ≈ 4 h 55 min. The resume also logged
  "ACPI: PM: Hardware changed while hibernated, success doubtful!" and an NTFS mount is active on the host → [I] another OS ran during the
  hibernation and wrote local time (WEST = UTC+1) into the RTC. Host matter (owner: `chronyc makestep`; RTC policy), not engine.
  Consequence for every wall-clock deadline after a resume (launchers, `with_gpu_lock --timeout` E180, harness caps): use real UTC
  (manager `real_utc.sh`) or CLOCK_MONOTONIC; the receipt guard below uses CLOCK_BOOTTIME − CLOCK_MONOTONIC, which a wall-clock
  step or slew does not move.

**Our runtime across a suspend:** E60 (2026-10-02) recorded GPU arms freezing through a 2.5 h hibernation and continuing; byte identity
of a run that spans a suspend is [U] (never measured; timings across a suspend are invalid, E60). On this host the GPU is suspended while
the process still runs, so CUDA calls block (as `nvidia-smi` did) until resume [I]. Today's run was killed by the harness, so nothing
says our engine itself failed.

**Guard (implemented, commit 23c9ee33c3, `src/gpuwrf/runtime/host_suspend.py` + `cli.py`; host-only, no traced code):** the CLI measures suspended time as the growth of CLOCK_BOOTTIME − CLOCK_MONOTONIC over
the run and writes `host_suspend = {suspended_s, timing_valid}` into `nested_pipeline_run.json`, prints a warning, and names a detected
suspend in the failure message (with the resume hint when `--checkpoint-dir` was set). No refuse/abort: a run that survives is still valid;
a run that loses its CUDA context fails loudly anyway and resumes from its last verified checkpoint (P1 restart path). Never block
hibernation (owner rule).

**Recommendations to ALISIOS:** do not kill a case on an `nvidia-smi` timeout alone (it hangs while the GPU is suspended); use output
progress + `journalctl -k | grep -i hibernat`; back off the queue after a suspend instead of burning cases at 10 s each; checkpoint long cases.

Guard evidence [M, CPU]: `tests/test_host_suspend.py` 10/10 + `tests/test_cli_aot_prefetch.py` 5/5 PASS; four deletion mutants (receipt field
forced to "no suspend"; failure note removed; detector reduced to CLOCK_MONOTONIC alone; to CLOCK_BOOTTIME alone) each killed by a genuine AssertionError, 0 setup errors (`analysis/receipts/host_suspend_mutants.log`).
Neither `cli.py` nor `runtime/host_suspend.py` is in the AOT cheap-key trace import closure (519 files) → no AOT key change, default program
identities (Swiss b977712f / PROD 321aa651) untouched by construction (no traced module edited). New receipt key `host_suspend` is additive JSON.
