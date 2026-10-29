# D1 GPU qualification preregistration

Status: CPU preparation only. No GPU lease/window is requested or consumed by this
file. Product `diffusion_parameters` remains refused. Pin the final reviewed
candidate and identical input SHA manifest before scheduling.

Use PROD 20260725, d01 120x70x44 dt54 / d02 267x117x44 dt18,
Thompson/RRTMG/MYNN/Noah-MP/KF(d01), product output cadence 67/200,
fusion eligibility as resolved by `product_schedule.py`. Run the actual product
force-down before child lowering. These are iteration/component gates, not WN3
release, station, D6 or S1 evidence.

1. Four cold domain arms: OFF d01/d02, ON-at-defaults d01/d02; each <=16 min
   timeout. Use separate private OFF and ON cache roots (E237), identical explicit
   compiler options and matched source/input/environment manifests. Save actual
   loaded DebugOptions, key/HLO/blob SHA, compiled-memory statistics and real
   executable E41 results. Deviceless PASS is only a prescreen.
2. Fresh-process warm OFF/ON/CHANGED replay, <=20 min. Assert physical backend GPU.
   One complete nested first root per arm; compare every finite carry leaf with
   byte equality for OFF/ON defaults. CHANGED doubles c_s; independently exercise
   each remaining selected name on a real active consumer. Require changed
   response and SAME ON/CHANGED keys, loaded executable and zero new native blobs.
   Reuse CPU pristine-WRF changed-value oracle results only when source-exact.
3. Matched warm cost/profile, <=10 min, separate from initialization and key/host
   comparisons. Twelve alternating samples per arm plus an OFF/OFF noise control.
   Explicit SDK NVTX labels (E248), actual nsys kernel/copy counts/bytes per own
   step and cadence-weighted root, compiled buffers, total per-PID VRAM, host RSS,
   readiness/compile wall. No >10% warm regression; report smaller losses with
   attribution. Counts/traffic must explain added boundaries/allocations.

All GPU commands use timeout and `scripts/with_gpu_lock.sh --label v05-diffusion`;
a manager-approved live lease, sequencer exact-call CPU smoke and RAM/disk ACK
are prerequisites. No GPU-parent child process retains a CUDA pool. Strip secret
variables in child environment. Never print embedded nsys strings/environment;
extract exact known-label IDs and numeric fields only, raw reports 0600/private.

Gates: defaults BYTE, no nonfinite leaves, unchanged product OFF program, all
Pallas native census complete, STACK <=256 B, no non-leaf callee >1000 SASS,
no kInput transpose fusion >500 optimized HLO instructions, changed operands live
without recompilation. Any failure keeps the family refused and triggers causal
localization, not a tolerance change. Independent critic required before merge;
manager owns composed RW1-RW4/release qualification.

Disk budget: one immutable snapshot measured at 2.5 GiB [M], plus <=6 GiB
private caches/reports [I], gives an aggregate peak allocation <=8.5 GiB. Ask the
manager for the final location after RW3 ends. Before admission, measure free
space on that filesystem and resolve the current ALISIOS floor (365 GiB at this
revision). Admit only when free_GiB - total_peak_GiB >= floor_GiB: this plan
therefore requires >=373.5 GiB free while the floor is 365 GiB. Count every
concurrent staging copy or other planned allocation in total_peak_GiB; any
larger snapshot or revised bound requires an updated DISK-PLAN and the same
admission check before allocation. Post DISK-PLAN before allocation >2 GiB.
Remove exact disposable caches, raw reports and snapshots as soon as the last
dependent extraction/review completes; retain small SHA-bound receipts and
existing references.

Before sealing the later launchers, adapt the w44 research runner explicitly:
its supported binder currently refuses D1, so any bypass must be local to the
research fixture, with the shipped refusal unchanged. The bound family now also
contains an internal REAL[2] stage-strength operand. Record it as an array (the
old `float(np.asarray(value))` logging loop cannot handle it); accept only the
seven public names in user input. Bind after static dt/acoustic cadence is
final. Exact-call CPU smokes of the final launchers are required before the GPU
window. Existing w44 launcher readiness receipts do not transfer to this pin.
