# W44 performance measurement plan (pre-GPU)

Gate: explain the utilization dips and each measured lost second since v0.3.2;
recover cheap avoidable costs while retaining WRF fidelity. This file registers
measurement, not a result. No runtime/model source is changed by these tools.

- Case: original PROD `s0_case_20260725`, namelist SHA-256 `5ef3a96be3324deb712b0e0114288244bcf5f53edb21461fee7374aa766cef4b`,
  two domains, 44 levels, dt 54/18 s, 3 h, same original input paths for every arm.
  The case's namelist starts at 00Z; retain that resolved work rather than silently
  correcting the historical directory's `18z` label. WN2 is a performance/debug
  case; these runs do not establish CPU-WRF fidelity.
- Minimal version split allowed by the contract: v0.3.2 tag `5f682e5ae9`,
  v0.4.0 sealed bundle `7bb5ae669b`, v0.4.3 sealed bundle pin `888804310b`.
  The latter is source-identical to the released tag. v0.3.3 is the next split
  only if v0.3.2→v0.4.0 is materially slower in the controlled comparison.
- A common frozen current S0 harness measures every source. The extra wrapper
  observes fused-cascade host calls, which ordinary S0's `advance` wrapper misses.
  Forwarding and three-root stop are checked without JAX. No tensor observation
  is added, and no JIT callable is replaced. Original GPU backend assertion stays.
- Main version comparison: `cuda_async`, explicit preallocation, fraction 0.25
  on all versions. CLI C-auto budgets differ with program memory; use a separate
  current-source C-auto bridge to quantify that policy. VRAM/RSS are disclosed.
- Separate cold-setup phases `prime32`, `prime40`, each timeout 1500 s. They
  populate private caches/layers. Existing sealed bundles are read-only through
  the approved consumer. CPU E41 scan follows each cold phase, before cleanup;
  it is not charged as GPU stepping. Never infer the actual GPU gate from the
  CPU AUTOTUNE0 screen. Keep each cache until its last dependent arm, then delete.
- One compare lock session (outer timeout 1800 s): warm-up per version, then
  `B43_1 A32_1 C40_1 B43_2 C40_2 A32_2 B43_3`. Keep every arm.
  Each version's A/A rule is `(max−min)/mean ≤ 5%`, separately for the official
  segment-2/3 proxy and for stepping. A failed rule yields INCONCLUSIVE for that
  wall metric, not a favorable subset. No post-hoc exclusion or retry.
- Report official S0 segment-2/3 periods (terminal period includes finalization),
  plus stepping/output callbacks, and finalization separately. Also disclose
  segment call periods excluding the final tail to diagnose E196 inflation.
  Process/harness launch→last original output mtime is recorded (E162). The
  3 h process rate includes startup; never call it the 24 h release S1.
- Dips: whole-process 1 s dmon per arm; warm segment 2 graph-level nsys on old
  and current, CUDA/NVTX/OSRT and Python sampling 1000 Hz. Report timeline gaps
  separately from unprofiled walls (E185). A second node-level capture covers
  exactly three complete roots per version solely for device family ranking.
  Rich optimized-HLO source frames, not kernel-name guesses, bind family work.
- Raw profiler artifacts stay in 0700 directories with 0600 files; query only
  necessary activity tables/columns, never TARGET_INFO/environment tables.
- No GPU command is queued until an explicit manager `GO GPU` window.
  RE-Toolkit lanes, ALISIOS and the nightly retain priority. No worker agents.

CPU preflight: six actual arm command environments passed a JAX-free probe;
direct/fused object forwarding and exact three-root capture passed. GPU command
execution/consumer behavior and warm-cache hits remain unmeasured until GO.

Supplement (separate immutable tool version; queued d0935fec is unchanged):
the first three roots of segment 2 (68–70) are ordinary. `radiation_session.sh`
captures current roots 100–102 instead, including d01's refresh at 100
(`cadence=33`) and d02's refresh at own-step 301 in root 101 (`cadence=100`).
It is requested separately, ~1 min / hard cap 180 s, for current-cycle family
weighting. Four JAX-free forwarding/stop checks cover early/late × direct/fused.

Night scheduling (manager 18:40:31Z): do not self-start or arm a waiter. The
manager's single sequencer calls immutable per-phase `night_launch_*.sh`
wrappers. They require GA_START_UTC/GA_END_UTC, reject a window shorter than the
phase's full cap, and take the official lock themselves with `--timeout 1`.
Actual locked time is bounded by the invocation deadline and rechecks the
window after lock acquisition. Receipts record refusal, admission, locked
start and completion; CPU dry runs never touch the official lock or GPU.
The original compare payload still requires the sequencer to post an explicit
`manager -> w44-perf: GO GPU ...` line before calling it.

The night R43 supplement additionally observes initialization callers, compiler
aval shape/dtype signatures, cache events, InitKernelCache hits and low-level AOT
loads (including background prefetch). No tensor values or device operations
are added by the observer. Its durations may overlap; its process wall is
diagnostic only. Unprofiled comparison arms remain the original frozen d0935fec
payload. A separate immutable tool version binds the supplemental observer.
