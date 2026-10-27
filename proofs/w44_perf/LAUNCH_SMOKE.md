# Day/night launcher admission, 2026-10-24

Both v4 public launchers passed the manager's exact-child-environment CPU
model smoke before the 10:30Z deadline. They replace the old posted wrappers;
old wrappers remain unchanged. The unposted v3 draft was removed.

| Step | Immutable public entry | SHA-256 | Phase cap / reserved teardown |
|---|---|---|---|
| Matched repeat | `<USER_HOME>/wrf_gpu2_lanes/w44-perf/day_launch_repeat_v4.sh` | `ae09c4aee94bdf251615a6c52a8409ae7f00636c284563c2cb38d522e10f3181` | 600 / 60 s |
| Remaining host callers | `<USER_HOME>/wrf_gpu2_lanes/w44-perf/day_launch_hostprofile_v4.sh` | `7fea49889f71be96ffb50dd1e31f9cf7a47870c73ff0c4a1c859a66340bb626e` | 180 / 60 s |

The exact public entry runs with `--launch-smoke`. Each actual capture child
records `JAX_PLATFORMS=cuda` and the product allocator/cache environment before
the device substitution. JAX's device configuration becomes CPU; its environment
remains `cuda`. The real immutable-source Thompson full-column function runs
one original PROD rain column, 44 levels, dt 18 s. All 20 returned leaves are
finite for v0.3.2, v0.4.0 and v0.4.3. The real CLI pre-exec/argument path is
entered, and the unchanged production GPU guard truthfully returns 3 on CPU.
The smoke captures this expected refusal and never claims a GPU forecast pass.
No GPU lock, NVML sampler or CUDA trace is started in CPU mode.

The host step also invokes the actual Python observer around the model call:
5,536,246 calls were captured and the receipt parsed. Compile/model CPU times
were 2.018 / 3.080 / 3.069 s across the three source variants and 4.200 s with
the host observer. These tiny smoke times are not forecast speed measurements.
Both real bundle consumers returned zero, passed both sealed-bundle checks,
and reported zero new compiled/alias/runtime/other files. Actual model import
hashes match each selected immutable source. Twelve admission controls pass,
including exact/short teardown margins and normal execution before GA start.

Cap evidence comes from the completed Oct23 GPU session: the same ten warm
arms took **320.992 s total**, maximum **39.284 s** per arm. The repeat's
fixed interleave, three warmups and seven measured arms are unchanged; its
cap is reduced from 900 to 600 s. The original graph arm took **41.081 s**
and the mixed observer arm **40.975 s**. New GPU Python-profiler overhead is
unmeasured; its 180 s cap retains headroom. Each launcher requires its phase
budget plus 60 s before `GA_END_UTC`. There is no date/hour or >=19Z guard.

Cause/fix: no prior perf GPU-launch failure; all original night phases ended
zero. The old dry mode changed the child environment to CPU too early and did
not meet the new smoke rule. The new opt-in smoke substitutes only the device
layer and runs actual model code. The ordinary observer's function AST and
the S0 harness bytes are unchanged. No product model source changed.

[Repeat receipt](receipts/LAUNCH_SMOKE_REPEATCOMPARE_V4.json),
[host receipt](receipts/LAUNCH_SMOKE_HOSTPROFILE_V4.json),
[complete frozen manifest](receipts/DAY_LAUNCHERS_V4.json).

The required two `LAUNCH SMOKE PASS` lines were posted at 02:58:33Z. Execution
remains manager-only after a GO and the actual GA window. No GPU job is queued
by the worker. Disposable smoke COW layers were removed after the receipts;
paid GPU caches and old-source snapshots remain for the registered arms.
