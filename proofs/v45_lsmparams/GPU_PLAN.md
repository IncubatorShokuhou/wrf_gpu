# W2 GPU qualification preregistration

CPU preparation only. All nine `LSM_CONTROLS` remain refused in the product.
Pin the reviewed final source and input SHA manifest before any GPU window.
Old w44 GPU receipts failed the scientific and native gates; they do not qualify
this implementation. No gate or tolerance is changed here.

CPU prerequisite is now met on `c1782be2dc` (main `cb48138349` plus the kept
Noah changes): full nested root 355/355 default BYTE, CHANGED 130 leaves with
the same ON keys and concrete chunk cache, and parent E41 kInput407/STACK240.
This plan is NOT BOOKED and has not run on the GPU. Seal its final reviewed
source/input/launcher/sequencer SHA before requesting a window.

Work: real PROD 20260725, d01 120x70x44/dt54 and d02 267x117x44/dt18,
Thompson/RRTMG/MYNN/Noah-MP/KF(d01). Resolve actual product output/radiation
cadence using `proofs/w44_params/product_schedule.py`; preserve force-down and
own-step counts. Use two explicit contexts: released Noah options with the
eight applicable controls, and Jarvis `opt_crs=2` with all nine including RSMIN.
The latter is also the declared CPU full-root context. Changed-after-init arms
are retained-state parameter perturbations; they do not qualify changed frozen
soil cold starts, which remain refused by the existing admission policy.

Schedule the two contexts serially as separate qualification blocks. Each has
four cold caps of 16 min, one 20 min root replay and one 10 min cost/profile
arm: 94 min of model caps, or 106 min including the sequencer's 2 min per-step
allowance. A block may start only if the complete remaining cap fits before
the granted hard end. Never mix caches between contexts or silently skip an arm.

1. Cold setup: OFF/ON-default d01 and d02 separately, each <=16 minutes, with
   separate private OFF and ON caches. Use the same source/input/compiler/env
   manifest, captured actual DebugOptions and real GPU platform assertion.
   Record loaded key/HLO/blob SHA, readiness/compile wall, compiled memory and
   E41 from the actual executable. CPU CompileOnly is only a prescreen.
2. Fresh-process warm OFF/ON-default/CHANGED, <=20 minutes. One full nested
   first root per arm, every carry leaf finite. Record full byte comparison,
   every unequal field's max absolute error, field scale, median/max ULP and
   zero-reference changes. Default BYTE is required (manager 2026-10-28
   01:11Z, following the failed full-root CPU arm). Preserve the old 4e-6 table
   for diagnosis only; it is not an alternative pass condition. Never increase
   a limit after evaluation.
   CHANGED must respond, retain the actual ON domain keys/loaded executables,
   produce zero new blobs and keep all three private derived coefficients live.
   Independently exercise each name on its registered real active column;
   carry forward pristine-WRF evidence only when the affected source is exact.
   Report the resolved counter policy, pre-mask event counters (zero or WRF-
   justified) and actual-work counters for each arm. Distinguish documented
   diagnostic sentinel values from physical-state nonfinites. A disabled
   counter is unmeasured, never a zero event count. Observer transparency must
   be checked on the GPU-compiled program where claimed (E205).
3. Warm cost/profile: <=10 minutes, 12 alternating OFF/ON samples and OFF/OFF
   noise control, outside imports/init. Explicit SDK NVTX labels (E248), actual
   nsys kernel/copy counts and bytes per own step plus cadence-weighted root,
   per-PID VRAM, host RSS, compiled buffers and readiness. No >10% warm
   regression. Attribute smaller losses, any extra launches/allocations and
   coefficient host binding. No forecast/S1/station claim from a component run.

Native limits: STACK <=256 B, no non-leaf callee >1000 SASS instructions, no
kInput transpose fusion >500 **optimized HLO** instructions; native-first
complete Pallas census and full-constant HLO identity of the OFF program.
Any failure stops qualification and leaves the family refused. No retry with
looser limits. Manager owns composed RW1-RW4 and release qualification.

All GPU commands require manager-booked ALISIOS window, live lease and
`timeout ... scripts/with_gpu_lock.sh --label v45-lsmparams -- ...`.
The manager's gate uses CPU-LEASE-FIRST ordering (the existing
`scripts/gpu_gate.py` / manager `cpu_lease.sh` protocol): validate source, time,
disk, RAM and free GPU-lock status; acquire a unique CPU lease and CURRENT
CORE_TABLE row for the GPU arm's approved cores; allow at least 30 s for peers
to withdraw; require the 60 s quiet-core check (each core <=50% busy); recheck
time/resources; only then enter the sequencer and official GPU lock. Never
acquire/hold the GPU lock while waiting for a CPU lease. The expected lock-holder
cores are 3,19,13,29, subject to the manager's current allocation; never borrow
ALISIOS or 14,15,30,31. Release only this gate's CPU lease in EXIT/INT/TERM/finally
after GPU teardown, including abort/expiry. Worker CPU analysis stays 12,28
nice19 with JAX_PLATFORMS=cpu and the same-line QUIET guard.
GPU host scopes are capped at 16 GiB/Swap0 with manager RAM ACK; standing
>=12 GiB admission uses RAMRUN and MemAvailable>=40 GiB. The completed CPU
Root03 approval does not authorize any GPU arm.
No CUDA-pool-holding parent launches a GPU child. The final launcher must pass
the sequencer's exact-call CPU smoke with native gates labelled NOT-APPLICABLE.
Smoke receipts from old w44 launchers do not transfer. Launcher/sequencer smoke
remains a separate readiness item until its sealed final pin is exercised.
Strip secret-looking environment variables in children. Raw nsys files/exports
stay private 0600; extract only exact known NVTX IDs and numeric metrics.

E237 namespaces are private under
`<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams/gpu_qual/<reviewed-pin>/<context>/`.
OFF and ON each get distinct `jax`, `aot`, `triton`, `cuda` and `tmp` roots;
CHANGED loads ON's caches and must add zero entries. Export and record every
cache/tmp path in the measured child; use an allowlisted environment, never a
shared/default cache. Preserve one context through its last warm/profile reader,
then remove its exact cache roots before staging the next context.

The hard aggregate lane allocation is <=3 GiB, including retained CPU evidence.
Pre-admission bounds are [I]: immutable sparse source <=1.10 GiB, both OFF+ON
caches <=1.00 GiB combined, reports/compiler temporary files <=0.25 GiB,
retained small evidence <=0.50 GiB: total <=2.85 GiB plus 0.15 GiB headroom.
Measure actual sizes before booking and after every arm; stop before any bound
is exceeded. Extract E41/numeric summaries then delete consumed large binary
dumps; do not retain four domains' raw HLO/protos or two context cache sets.
Keep final reviewed source and necessary original case inputs read-only; no
DATA or /tmp payload writes are authorized. Read-only existing inputs are not
duplicated into staging. Delete this CPU snapshot/dumps after their critic's
last reader before GPU staging if needed to satisfy the retained-evidence cap.
Post DISK-PLAN before allocation and require filesystem free bytes minus all
concurrent reserved/new bytes to remain above the manager's live floor; the
snapshot/cache/reports/reserve are counted together. If measured cache/source
sizes do not fit 3 GiB, do not book or launch: revise the bounded staging plan.
Raw nsys/exports are 0600, bounded by the report cap and deleted after extraction.
