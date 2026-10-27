# CUDA mini launch qualification (v7)

**GPU result (P74, 2026-10-25 00:02:54–00:03:56Z, manager smoke sequencer, cap 7 min, rc 0).**
The sealed mini `1933cf36` ran on the RTX 5090 (`platform: gpu`):
- GPU phase 47.3 s of its 300 s cap; whole arm 52 s.
- Component (real WN3 d03 day column, EDMF 1, two steps): 50/50 physical fields BYTE
  ON vs OFF, verdict `COMPONENTS_BYTE`. First compile was 10.7 s OFF and 21.8 s ON;
  warm medians 0.82 ms OFF and 1.31 ms ON.
- E41 on the two actual CUDA AOT programs: OFF 136 kernels and ON 188 kernels,
  max STACK 0 B, no conservative flags, coverage 2/2.
- The in-launcher oracle used the sealed old PBL bound, so it does not qualify.
  The pre-registered offline re-score (`twin_oracle.py` at `49a3636970`) gives
  `ORIGINAL_TERM_GATE_PASS` at both steps. Controls: 14 killed. G3 is vacuous
  because RQCBLTEN is zero in this column.
- Tightest margin: RQIBLTEN at 0.833 of its bound, a single 1-ulp(qi)/dt quantization step.
- No nsys report (no profiler range in the component-only mini), so cost accounting is UNKNOWN.

Receipt: [P74_CUDA_MINI_GPU.json](receipts/P74_CUDA_MINI_GPU.json). Production
scientific gates (coupled BYTE, four oracles, cost) remain pending.

**v8 re-seal (2026-10-25 01:10Z, tools `5c15780802`).** The manager asked for a re-seal once the
gate file changed; it changed with the critic-approved floor amendment `7a24833cd2`. These are
new launcher files; the v7 bytes are untouched (E163). Only `history_gate.py`,
`pristine_conditioning.py`, `twin_oracle.py` and the manifests differ from v7. The CPU-rehearsal
scope moves from 16 to 24 GiB. The GPU code and model pin (`snapshot_p68`) are unchanged.

| Launcher | Path under `<USER_HOME>/wrf_gpu2_lanes/w44-budget/launch_v8_5c15780802/` | SHA256 |
|---|---|---|
| Production (GPU phase 24 GiB) | `production.sh` | `8feb06738162693d07bebd4f2fa8f181b3f61565d3ec2dd85798b0fcbe106dc7` |
| CUDA mini (12 GiB) | `cuda_mini.sh` | `32a033cd7dc0328a37575878d6646a213e783300be85e67737fadccaaf6ccb6d` |
| CPU rehearsal only (never queue; 24 GiB) | `cpu_rehearsal_coupled.sh` | `adcbb4b83fe3d0e1f0e8736feaaee4c0c2359926c053893f5353655b097178a5` |

E225: both launchers pass the real dry-run (VALIDATED) and o5-coast `check_alignment`
(production 3 h, mini 0.03 h). P76 is the CPU all-phase rehearsal of the v8 mini; it PASSES:
COMPONENTS_BYTE, the sealed v3 oracle gives `ORIGINAL_TERM_GATE_PASS`, E41 passes, the scope is
12 GiB. The P74 GPU result carries over because the GPU code is the same and only post-phase
gate files differ. [Receipt](receipts/V8_SEAL.json). The v7 set below is withdrawn.

All v7 launchers share one tool set (tools commit `427ebb9924`, 11 SHA256-pinned
files) and the immutable model snapshot `snapshot_p68` at
`5438aca4b533ba8ed739ce94e90cf14cb98b86d4` (src tree `3c12f311`), the lane head
rebased onto main `aac9e78065`. GPU command builder, literal CUDA child
environment, official one-second GPU-lock acquisition, private process groups
and subreaper cleanup are unchanged from v6. Only manifests differ.

| Launcher | Path under `<USER_HOME>/wrf_gpu2_lanes/w44-budget/launch_v7_427ebb9924/` | SHA256 |
|---|---|---|
| Production (GPU phase 24 GiB, pending approval) | `production.sh` | `171c3c69a762086b9880485840af8f141f89efe93c5d43ac186dafbb1eda6905` |
| CUDA mini (12 GiB) | `cuda_mini.sh` | `1933cf364b5016d9a6bb235b55c40a46ae4e6468774eef23941d17696cb04b3c` |
| CPU rehearsal only (never queue; GPU cap 1 s; 16 GiB) | `cpu_rehearsal_coupled.sh` | `6847042bf833f503f3504288303d0ea3e791803154a13c69425fe896d09d8559` |

The earlier v7 pair `launch_v7_b4f2dcf149` (mini `275f8b1c`, production `d10a9c70`)
differs only by the fixed 12 GiB GPU-phase scope; it is superseded.

v5 (`87b987aa`) and v6 (`b9ccaae4` / `2d23b106`) are withdrawn: they pin the
pre-rebase model `42f60b7485`, and v5 failed RAM admission twice.

| Setting | Production | CUDA mini |
|---|---|---|
| Component cases (real WN3 d03 columns) | day/night × EDMF 1/0, 32 columns | day EDMF 1, 1 column |
| Coupled three-nest replays | prime + replay OFF, ON, EDMF-off, 3 h | none |
| Expected probe verdict | `FIRST_STEP_AND_REPLAY_BYTE` | `COMPONENTS_BYTE` |
| Original-term checkers after unlock | 4 | 1 |
| GPU phase hard cap | 3600 s | 300 s |
| CPU phase hard cap | 720 s | 360 s |
| Whole arm cap | 4500 s | 720 s |

Why no coupled replay in the mini: a cold compile of one new three-nest program
already exceeds 300 s (night v1: ON fused child compile did not finish inside
the remaining 306 s [M]). The mini therefore exercises the production
admission, lock, product CUDA environment, own memory scopes, component
builder with native AOT exports, actual E41 discovery and the pristine checker,
not the coupled CLI path. That path is rehearsed on CPU (`cpu_rehearsal_coupled`).

Expected mini GPU-lock time was estimated at about 3 minutes [I] (measured: 47 s): CPU-measured component
compile 53 + 99 s (P69), a GPU/CPU compile ratio of about 0.9 measured on the
WN3 d01 program (170 vs 189 s), plus roughly 30 s for startup and exports.

## CPU rehearsals (device substitution only)

The harness `mini_cpu_all_phases.py` runs the exact sealed shell. At the device
boundary it records the product child environment (`JAX_PLATFORMS=cuda`), then
runs on CPU with the P26–35 AVX setting; the official lock and CUDA profiler
are substituted. CPU model outputs feed the real pristine checker. After model
rc 0 the native-export boundary supplies a retained actual CUDA executable to
the unchanged E41 scanner: scanner plumbing only, never qualification of the
new executables. CPU compile uses a separate rehearsal allowance; sealed caps
are unchanged.

| Run | Launcher | Result |
|---|---|---|
| P65 | v6 mini (2 × 2 components + replays) | Components 4/4 BYTE, original terms 4/4 PASS; coupled prime killed by the kernel memcg OOM at the 12 GiB scope during the fused d02/d03 CPU compile (rc 143 at 1273 s, journal 16:06Z); E41 then scanned CPU executables. INCOMPLETE. |
| P68 | v7 mini, 2 cases | PASS: 2 × 50/50 BYTE, 2 original-term gates, E41 PASS; model phase 283 s CPU (compile 262 s) → too close to 300 s on GPU, re-sealed with one case. |
| P69 | v7 mini (current) | PASS: 50/50 BYTE, `ORIGINAL_TERM_GATE_PASS`, E41 PASS (coverage 2/2 programs, STACK 32 B), own scope 12 GiB / swap 0, model phase 168 s, post 40 s. |
| P73 | v7 mini `1933cf36` (byte-paired with production `171c3c69`) | PASS: 50/50 BYTE, `ORIGINAL_TERM_GATE_PASS`, E41 PASS 2/2, scope 12 GiB asserted from the manifest; model 201.6 s (compile 52 + 131 s), post 34 s. |
| P70 | CPU coupled crop rehearsal (12 GiB scope) | FAIL: d01 OFF compiled in 178 s, then memcg OOM at 12 GiB (anon-rss 12.54 GB) in the fused d02/d03 CPU compile. Needs a 16 GiB CPU scope (manager decision). |

## Launch configuration checks (E225)

`validate_launcher.py` runs the real `gpuwrf run --dry-run` with the exact argv
the probe builds and independently checks duration/dt/history/boundary
alignment; o5-coast `check_alignment.py` confirms. Production: 10800 s = 200 /
600 / 1800 steps, interval 21600 s / 54 s exact, 9 history files. Mini argv
(`--minutes 1.8` on the crop, only used by the rehearsal) aligns to 2 / 6 / 18
steps. The d01 history interval 3600 s is not a multiple of 54 s; WRF writes at
the first step past the alarm (3618 s), so this is informational.

## Resources

Memory caps come from the manifest (`gpu_memory_max`): mini 12 GiB (component
programs only, P69 fine), production 24 GiB, CPU coupled rehearsal 16 GiB. The
production value is pending manager approval: a cold WN3 three-nest GPU process
alone peaked at 13.6 GB host VmHWM (FINDINGS D26), and the CPU compile of the
fused child program exceeds 12 GiB (P65/P70). The GPU-side compile peak is unmeasured.

Each GPU and CPU-post child runs in its own systemd scope (`MemoryMax=12G`,
`MemorySwapMax=0`, `--expand-environment=no`); no shared RAM FIFO. The mini
writes its cache and outputs under `<USER_HOME>/wrf_gpu2_lanes/w44-budget/` on
ROOT (CPU rehearsal caches are deleted before any GPU run); its 23.6 MB inputs
on DATA are read-only. Every run refuses DATA < 365 GiB and ROOT < 20 GiB.
Production writes to DATA and needs a separate `DISK-PLAN` plus manager OK.
