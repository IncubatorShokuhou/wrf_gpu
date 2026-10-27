# w44-rc: v0.4.4 real-world qualification session

Candidate runtime: f157ee2a4dd7e6debbab7fe40867e9ef17045a7c, src aa961cb87176f4185aca4d61c99754cae74a2be6; src/tests EXACT approved7ce and the CPU-smoked source.
Composed IN items 1 topo, 5 QSQ initialization, 6 E84 solar/ozone RN, 7 params refusal, 8 version, 9 LSM/Jarvis refusal.
OUT constfold, root walk flip, H-CB. The manager pins the full deliverable commit after critic.
This plan succeeds w44-release/RW_SESSION_PLAN.md, retaining all numerical thresholds and reference rules.

## Status and source

[M CPU] Five canonical native-first comparisons and E41 CompileOnly prescreens complete. Actual night_seq_v9 five-entry CPU admission completed under the manager09:54 exception (CPU_ADMISSION_V4.json). Item9 is composed; final full suite is RUNNING once; launcher critic is reviewing0fbff22053. No final RC pin or GPU qualification yet.
No GPU requested or started. No numerical/speed/memory PASS inferred from composition.
All new tools/receipts/caches live under <DATA_ROOT>/wrf_gpu2/scratch/w44-rc. Source snapshot is detached.
Parent launcher tools are read-only; DERIVATION.json binds copied inputs. Original WRF references and v0.4.3 bundle stay read-only.

## Binding gates

RW1: every warm run rc0, zero new compiles, exact whole-file bytes vs its prime, all step programs loaded from AOT blobs,
bundle content check missing=changed=new=0 before/after each consumer. Record hourly outputs and original file mtimes.
RW2: WN3 0502 and 0227 48-hour-input warm replay, plus icloud=0 second-date replay. Same program shape uses one sealed native program.
RW4: three dates 0227/0502/0614 replay and CPU aot-key pairs shared. Real bundle consumer, fresh write layers.
RW3: WN3 0227, all three nests, full72 hours vs ORIGINAL CPU-WRF, D6 plus output_integrity_pass and frozen floor72 rule.
Speed: interleaved N/O24-hour arms, primary d03_step_median with frozen uncertainty rule. >10% regression is STOP.
VRAM per PID and host max RSS: >10% regression is STOP. N72/O72 disclosed separately.
E41: actual GPU executable scan before cache cleanup. STACK>256B or non-leaf callee>1000SASS is a hard failure. kInput>500optimizedHLO is diagnostic under manager ruling01:58:22Z; the raw scanner verdict remains unchanged.
Compact-AOT HLO is UNKNOWN, not zero. Complete program coverage, STACKPASS, no non-leaf>1000 and known native call graphs are required. Every kInput diagnostic/unknown is published separately.
Swiss: E84 changes its program; mandatory24-hour D6+integrity vs ORIGINAL CPU-WRF Swiss reference.
No older GPU release is a scientific reference. GPU-vs-GPU bytes establish replay/determinism only.

## Exact manager calls

Manager uses night_seq_v9.sh on a sealed JSON plan, no wrapper that substitutes a different command.
Manager ruling 2026-10-26T01:19:21Z replaces the <=5-minute mini with explicit cold setup.
Five exact entries: entry_prime_swiss.sh cap15, entry_prime_prod.sh cap30, entry_prime_wn3.sh cap55,
entry_rw_warm.sh cap45, entry_rw3.sh cap75. All caps are minutes, including bounded teardown/post.
Cold setup: Swiss baseline349s ->900s cap; PROD baseline626s ->1800s cap; WN3 default+icloud0 each530-943s
->3300s entry cap including600s post-E41 reserve. These are [I] generous setup margins on predecessor [M] cold costs; they also cover the exact-cap CPU first-root smokes (predecessor Swiss CPU1h cold950s vs warm373s, so a10min entry can expire in CPU compile). Caps are maximums, never timing claims.
The WN3 icloud0 prime/replay preserves the predecessor scope; no CPU-WRF icloud0 twin is claimed.
One fresh private RC bundle/cache under scratch/w44-rc/rc_bundle_<pin>. Each prime is sealed independently,
logs compile/setup cost, and writes its actual source/device/finite/producer receipt. Later stages refuse missing predecessors.
One full E41 scan follows the last prime, checking every producer inventory before any cleanup or sealing.
Then entry_rw_warm seals and runs RW1/RW2/RW4; the long entry runs RW3. Cold setup entries double as the GPU launch smoke
and fail fast, as explicitly authorized by the manager. No standalone CUDA-mini entries remain in the active plan.
Prime entries run Swiss3h/PROD3h/WN30227 3h/icloud0 3h. The warm entry seals the shared cache and replays all requested dates and Swiss24h.
The long entry runs WN30227 N72, N1/O1/N2/O2/N3/O3 at24h, O72 if resources allow. N72 is mandatory for fidelity.
The active plan is <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/NIGHT_PLAN_V4.json, with tools_v044_v4 seal e6d12afcc889b7a3db0930e6351650c8b728301524c48913ee8fbe45f6eaee10. The shared short-session run root is session_<pin>_<mode>_v4; all cold/warm producers remain there for exact byte/prefix comparisons. Each measured forecast asserts jax.devices()[0].platform=='gpu' inside the actual measured process after product C-auto setup.

CPU exact-call smoke: SEQ_SMOKE=1 night_seq_v9.sh PLAN 20261026_12z UNUSED. The entry observes WRF_GPU2_SEQ_SMOKE=1.
It uses CPU/private QUIET/private mailbox and the same command builder/phases/checkers. Manager2026-10-26T09:54:33Z explicitly approved Swiss actual first-root execution after full-duration validation; PROD/WN3 model phases are exact CLI --dry-run plus alignment, backed by all five completed full-shape native-first CompileOnly prescreens and host-only item9 identity. They are PLAN_ONLY / NOT_APPLICABLE, never model PASS. CPU profiling runs the exact PROD plan then real Swiss NVTX work; GPU timing/source attribution is NOT_APPLICABLE. Every other phase runs. It never acquires the real GPU lock. The actual full-size GPU cold primes remain mandatory fail-fast first steps. No further full-PROD CPU first-root retries or16G scopes are allowed.
A real retained v0.4.3 AOT cache is scanned to exercise the compact-AOT INCOMPLETE path. New-candidate native gates remain pending.
Launchers cannot be called READY until all five actual sequencer smokes complete with resources reaped and cr-w44-launchers accepts them.

## Resource admission (manager requests the window)

GPUDISK [I inherited measured baselines]: source snapshot2.5GiB; short incremental12GiB (start free>=377GiB), RW3 incremental31GiB
(start free>=396GiB), optional old72 only with the extra headroom. Absolute ALISIOS floor365GiB. Exact guard runs per arm.
CPU follow-up stage original72-hour reference~23GiB, through existing ALISIOS read admission only. No new ALISIOS input mutation.
GPU scope MemoryMax20G/noSwap (prior cold WN3 host peak13.0–13.2GB); CPU follow-up/smoke scope12G/noSwap; CPU8,9,12 NI19.
GPU scope uses a dedicated systemd scope, never the shared RAM-slot FIFO. Official with_gpu_lock.sh --label w44-rc only.
No whole-run arm exceeds its manager-scheduled long-window cap; all short research GPU arms remain<=30min.
E41 CPU scan runs after releasing the GPU lock, before cleanup, with600s reserved. Explicit cold setup and the warm/full session require manager scheduling; no GPU launch by this worker.

## Follow-up and cleanup

Reuse copied followup_short/followup_rw3 scoring recipes. Parse report JSON, never compare_wrfout_grid exit code alone (E247).
Retain N72 and the CPU reference until the72-hour scorer has completed; delete only own listed outputs after SHA receipts.
Retain compiled native caches until E41 and all dependent consumers post DONE. Never remove a reference or an ALISIOS case.
CPU gate reports are separate from GPU release results. final.sh and wheel smoke follow manager-approved RC and RW receipts.

Manager01:58:22 task: separate B6_prod_profile uses the same warm source/cache/args after the unprofiled B6. Explicit NVTX3 program/root ranges and numeric per-launch durations are required; artifacts0600/private and raw SHA retained until source/DAG binding is DONE. No profiled timing replaces unprofiled RW1/RW3 evidence. Report any non-executed diagnostic NOT_EXECUTED, not0time. See KNOWN_ISSUES_RC.md.
