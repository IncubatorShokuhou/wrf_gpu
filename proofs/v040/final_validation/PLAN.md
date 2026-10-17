# v0.4.0 final validation preparation

Status at2026-10-15 05:01Z: **70/70 original references complete;32 requiredMPcodes covered. GPU cohorts active.**
The added c90 supplies the pristine-real initialized active tracer reference;
its full initial numerical carry/constants match paid c46, and that GPU trajectory
passes the new original reference's25-frame D6 gate without a repeat GPU run.
TR17 remains REPORT_ONLY (overallRMSE6.00146e-6, maximum absolute9.93975e-4).
Actual native-CLI honour checks now use the exact CLI parser and replay=False:
c77/c78 hit a surface-FDDA max_dom-list TypeError before stepping. Their originals
remain valid; both GPU cases wait for a host fix. CPU preparation and runtime
admission are separate gates; global ready_for_gpu is false while these2 remain.
Manager05:09Z permits c45 Goddard radiation5/5 to finish24h inside30min,
otherwise retaining the earlier **6 h accepted gate**. On627 the24h attempt
hit1770s after24frames(h0–23), with0bounded misses/rawNF0. The independent
first7-frame scoped D6 gate passed all346CPUfields at unchanged bounds;
actual E41 passed. See GOD5_6H_D6.json. Full24h remains INCOMPLETE; no second
Goddard24h GPU arm is scheduled. The earlier preparation details follow.
Manager04:23Z scope exception: c45 Goddard radiation5/5 had a **6 h gate only**,
at unchanged frozen D6 bounds. Separate hourly-checkpointed segments are each
capped at30min, with the full original25-frame CPU reference preserved and
hash-verified before selecting the first7 observations. The current24h partials
remain archived evidence. Qualification is pending until all7 frames pass.
Observed first-hour file interval590.978s includes readiness work and is not S1;
CPU-WRF full24h wall532.366s. Budget approximately60–90 GPUmin including restart
overhead; t-godperf owns performance attribution. No24h Goddard claim is made.
The manager23:34Z resume overrides the earlier nightly pause and04:30Z start.
Exact issue `20261014_12z` is `DONE`; the allocation endsOctober15 19:15Z.
WN3 full24h D6 passes all75 frames/three domains; actual E41 passes4 entries.
Historical12-file GPU byte regression failed separately; no F2 closure is inferred.
See [the completed release-case receipt](WN3_GPU14_D6.json) and
[tomorrow's ordered plan](TOMORROW_GPU_PLAN.md) and its machine-readable JSON.
The live cohort ledger carries subsequent results and preserves original evidence.
Every cohort, including standalone/nested, now has a30min total cap;
separate cold setup requires a new manager allocation. Heavy CPU jobs use
`scripts/with_ram_slot.sh --label v-prep`, with four global slots and CPU
singletons capped at12GiB/no-swap until19:15Z. Four-process GPU cohorts use one
aggregate40GiB/no-swap scope. At19:15Z the15GiB aggregate ceiling returns.
The historical execution details below do
not override these limits.
Owner: v-prep; final RC freeze belongs to the manager. Owner update2026-10-14:
v-prep runs GPU batches as soon as their matched CPU references are complete,
using the official lease/lock. The hard stop is **19:15Z**. This supersedes the
earlier CPU-only contract and 11:30–19:00 preparation window.
External working root: `<USER_HOME>/wrf_gpu2_lanes/v-prep`.
The sealed matrix, matched namelists, CPU receipts and output hashes are published
beside this file when their preparation gates close. Live receipts stay external.

## Coverage and the composite-count discrepancy

Main at the initial inventory (`3681f7b328e68502985e1bc6fa3d4ef0f3d0ad07`)
admits **26 mutually exclusive microphysics codes**. Therefore twelve to fifteen
single-domain programs cannot cover that set. Thompson-only cumulus7/10/11 and
Goddard5/5 radiation require additional MP8 programs; TEMF, QNSE and urban
pairing restrictions also constrain packing. The initial forty rows cover all
MP/CU/PBL/LW/SW codes; authentic SSiB, K1 and two-way rows extend the matrix.
Surface7 and active Lake1 have separate rows. This is an explicit superset of
the requested count, not an assertion that fifteen programs provide full coverage.
The Owner resolved scope in chat2026-10-14: one primary Swiss24h composite per
CLI-admitted MP (about33 after pending merges), pack other admitted options at
least once, plus nested and release-default runs. Required MP8 companions stay
explicit when actual pairing rules prevent packing into another MP code.
Reuse completed unchanged references.

Owner exception2026-10-14: exclude `km_opt=3` from the full-coverage GPU gate.
Status: **experimental, column-oracle-qualified; no coupled reference: original
CPU-WRF crashes (SIGSEGV ~00:04) on Swiss and Madrid with km_opt=3**.
The fresh unchanged-body compiler-zero Swiss attempt also fails after one frame.
Keep that evidence, schedule no GPU arm, and do not let this excluded option
block the remaining matrix. No further optional CPU variant is needed for this
preparation gate.

Owner12:59Z also excludes MP15 from coupled coverage: original WRF4.7.1 crashes
at first output with both canonical and compiler-zero profiles. It remains
experimental, column-oracle-qualified, with no coupled reference. See
`UPSTREAM_WRF_BUGS.md`; no output mask is accepted as a cure for the observed
prehistory pointer corruption. Preserve the failed reference and stop this
bounded investigation. Pack its independently admitted Noah options elsewhere.

The accepted Madrid soil2/3 and explicit irrigation1/2/3 targets are c71–c73.
Official NCAR archives total949368423B and the one original WPS/real case has
populated genuine soil maps. Crop/tile/irrigation-method maps do not cover that
geography. Those unavailable branches remain **experimental: needs
geographically valid NCAR maps, no 3-D validation**, column-oracle-qualified;
no synthetic map truth is substituted. Sprinkler/flood each exercise171 cells;
micro-irrigation has zero events in this particular weather.

`MATRIX.json` records the exact admission inventory, options and companion
settings of every program. `namelist_honour_report` is required before admission;
an actual original-WRF successful finite reference is a separate gate. No original
WRF failure is converted into a valid GPU validation target.

Owner-approved scope excludes API-only options. Catalog IMPLEMENTED land1/2/7
are **programmatic-API-only**: the native CLI
refuses them without explicit static/land bundles. They are excluded from this
CLI matrix with that reason, not described as unimplemented. CPU-only FLG7/GFDL99
radiation is excluded. Constant-K API-only paths are also excluded. MMF opt_run5
is CLI-admitted on its separately prepared authentic ice-free groundwater
geography; stock Swiss lacks active groundwater and cannot replace that case.

Late Ferrier5/15, Goddard7, Thompson-hail38, Jensen55, NTU56, CAM-MG11 and
Noah-MP option merges must trigger a new admitted-set inventory. Add each
new mutually exclusive MP program; pair compatible secondary options onto it.
Never change a running case's namelist or replace its reference. Exact code17
normalization to NSSL18 may support one byte-identity carry-forward certificate,
but remains a distinct requested namelist/normalization check.

## Original CPU-WRF references

Swiss inputs are the retained authentic 42x42x44 2023-01-15 case, dt18,
24h, hourly history and one frame per file. Madrid references use retained
original WPS/real outputs at 42x42x44, 9km, dt50, 2026-10-13 00Z; RUC has six
soil layers. SSiB's three-layer input is generated by **original real.exe** from
the retained Madrid `met_em` files, not by editing/interpolating a Noah wrfinput.
Its preprocessing commands, executable and all input hashes are retained.

`prepare.py` copies the original serial executable into each lane case and
changes exactly two equal-length instrumented diagnostic OPEN strings to private
paths. Reversing those changes must restore the canonical executable SHA
`52f914cb38e0899d8a1da54bd52f3e08d81ba0f6a7015d3159e48393f5226bd6`.
No pristine Fortran body or shared reference is edited. Original WRF Thompson
tables are reused read-only with hashes. These are shared nice19 serial reference
timings, never a four-rank CPU benchmark.

For schemes with documented material upstream uninitialized reads, a private
compiler-defined-zero build uses `-finit-real=zero -finit-integer=0
-finit-logical=false`, production FP traps off, and unchanged source-body hashes.
The global-init attempt exposed bare SAVE/automatic arrays in Morrison-aero;
that inactive module retains its exact original object/interface, documented in
the build receipt. MP40 itself uses the canonical original executable. Existing
source-wide initialization-sensitivity receipts from scheme lanes are carried
forward only when their pristine bodies and compiler profile match.

The nested row uses the full authentic WN3 120x70 +267x117 geometry, 44levels,
dt54/18, feedback1/smooth2, 24h from 2026-02-28 00Z. The serial executable
refuses nesting. The retained original dmpar executable is copied byte-exact
from the two-way lane (SHA `1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b`).
It runs as MPI1 on a granted core; rank-zero `rsl` logs are authoritative for
the WRF success marker. The serial refusal remains preserved separately.

`cpu_references.py` bounds each Swiss/Madrid reference at5400s and the nested
reference at21600s, pauses its own process groups during QUIET, and requires
success, all25 hourly observations/domain (DFL1: natural24), raw finite CPU fields and unchanged
actual input hashes. `CPU_OUTPUTS.json` contains SHA256, Times, full schema and
headers of every frame. Failed references retain their evidence and block their
GPU row. Private executable copies are removed after their last execution;
scientific inputs/outputs and reconstruction receipts are retained.

## K1 and active-regime checks

The K1 row combines tslist, NWP diagnostics,60min precipitation buckets,
hourly regional streams, WRF tracer_opt2, two declared Fitch turbines and
SKEBS/SPPT/SPP-PBL/RAND on the qualified MYNN5/5, Thompson8/KF1 path. Turbine
geometry is explicitly constructed forcing on unmodified real Swiss meteorology;
it is not an existing wind-farm observation or skill claim. All sidecars and
the original namelist are hash-bound to both processes.

Regional stream T2/RAINNC instant samples are checked against the matched
original hourly CPU fields at the declared grid point. No hourly reference is
misrepresented as an every-step min/mean oracle. Original tslist files, tracer
history, POWER, NWP and precipitation fields are retained for feature-specific
scoring; record active/nonzero counts before claiming a feature was exercised.
Require actual turbine power/drag, stochastic variance, nonzero tracers and
active lake cells. A flag on an inactive case is not qualification.

The nested two-way row excludes tracers because two-way tracer feedback is
explicitly refused. The stream/tslist restart restrictions are respected: test
crash/resume on a separately declared most-carry configuration that is actually
restart-admitted, then byte-compare its uninterrupted and resumed GPU outputs.
This byte test is regression/restart evidence, not CPU-WRF fidelity.

## GPU windows: compile once per composite, then24h

Run these recipes from a clean detached measured candidate with the pristine-data
symlink. Every receipt identifies its exact source; carry a result to the final
RC only with the required unchanged-program/input/work proof. v-prep is now
explicitly authorized to execute the batches during the manager's allocation.
Every window sources the driver bridge and uses the official v-prep lease/lock.
The parent shell holds no CUDA pool. Each child inherits the lock FD and asserts
`jax.devices()[0].platform == 'gpu'` before forecasting; it records actual CLI
inputs, source commit, environment, original-reference hash and launcher hash.

Example60min manager-approved daytime window:

```bash
bash proofs/v040/final_validation/gpu_window.sh '2026-10-14 12:30:00 UTC' c00 c26 c27 c45
```

`gpu_inside.sh` starts at most three ordinary Swiss processes concurrently,
each with a15%-VRAM capped pool. CAM and nested arms start alone with30%.
This leaves desktop/outside-pool compile headroom. This is an
inferred safe initial schedule, not measured VRAM/performance evidence. Each
case owns its JAX/autotune/Triton/CUDA caches. One cold forecast invocation
compiles all necessary ordinary/radiation/history programs and then advances
the full24h; no duplicate per-option forecast is scheduled. Nested runs are
submitted alone; require the complete original reference and a separately
approved up-to60min cold-setup window if its measured cost needs that.

Use30min windows for one ordinary Swiss group,60min for two groups after first
costs are known,90min for three to four groups. Start no late arm that would
cross the manager deadline or19:15Z. All boundaries are absolute UTC, and the
official lease remains authoritative. A failed case does not invalidate other
completed composites; preserve its cache until the actual executable audit.

Twenty completed predecessor GPU receipts measure5.09–15.19min cold composite
walls (median8.31min), recorded in `VP28-predecessor-GPU-costs.json` externally.
These are source-bound predecessor measurements, not final RC predictions.
Nine current arms completed by12:54Z; their three-process groups take roughly
17–22min and ordinary per-process sampled peaks are5.3–5.9GiB [M]. c00 true
release-default RE1 passes full-field D6/E41; opt-in rainfall/nonfinite/schema
failures stay open and have independent repair lanes. The present 67
active programs imply **400–600 allocated GPU minutes [I]** for all cold arms,
including standalone special/nested cases; remaining work at13:10Z is roughly
**320–500min [I]**, excluding other lanes' queue occupancy and any necessary
repair reruns. This supersedes the stale45-program300–500min estimate. Pending
MP55/56/11 add their actual admitted programs after merge. Source changes need
identity/reuse evidence before carrying any result to the final RC. These are
qualification walls under shared CPU/I/O load, never warm S1 claims.

The manager granted a second window on2026-10-15 (MAILBOX13:25:11Z). It starts
only after04:30Z and after the **new** ALISIOS nightly cycle reports
`verdict=DONE` in `<DATA_ROOT>/alisios/state/nightly_9nest/LAST_RUN.json`; record
its issue, updated_utc and file hash when starting. The observed20261013 cycle's
DONE is not tomorrow's permission. Use a renewed manager lease and the official
lock, keep host work on the assigned cores, and end every arm before19:15Z.
Carry today's exact successful programs; put remaining composites, newly merged
MP55/56/11 and affected final repair checks in tomorrow's queue. Do not repeat
completed numerical validation for a proven host-only writer change.

## E41 and frozen D6 consumers outside the GPU window

After each child exits, release the GPU lock and pipeline CPU consumers while
the next GPU group runs. Every cold cache is scanned with the restored full
`scripts/wave_cubin_gate.py`, including its actual AOT domain/fused-program
inventory from the product proof. Preserve rich HLO (`GPUWRF_AOT_SLIM_MODULE=0`)
and actual native PIN identity before cleanup. Release defaults retain the hard
E41 gate. Opt-in flags are disclosed under the manager's standing policy;
unknown or missing programs never become a PASS.

`score.py <immutable-source> <case-dir> --domains d01 [d02]` uses unchanged
frozen D6 SHA `9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc`.
Every CPU field is inventoried on every paired frame; dimensions/shapes must
match and no CPU field may be missing. Raw CPU and GPU arrays must be finite.
Every bounded field must pass on every observation; unbounded fields remain
REPORT_ONLY. Every frame's global metadata differences are recorded separately.
Validate sealed CPU output hashes before scoring. dt54 alarms can be at
01:00:18/02:00:36: require the actual sealed CPU Times, not fabricated exact-hour
labels. A complete decision requires all25 frames per domain and exact pair Times.
Owner12:59Z accepts DFL1's natural24 frames at00:09 through23:09; its GPU output
must have those exact24 Times. No padding, dropped CPU field or time-warp is
allowed. The original count-only refusal and all output bytes remain preserved.

No cache is deleted until E41's complete scan/native identity and the last
dependent audit have succeeded. Delete disposable cache/tmp/snapshot paths
exactly, under the same-line QUIET guard. Preserve failures, CPU/GPU wrfouts,
reference executable reconstruction facts and minimal scanned PINs. A scan
timeout leaves its cache intact and its verdict INCOMPLETE.

`postprocess.sh c00 c26 ...` runs the full E41 scanner against
`gpu_proof/nested_pipeline_run.json`, then the frozen D6/full-field reader and
`pin_cleanup.py`. It retains exact raw executable PINs for every scanned
program before removing only the case's disposable cache/tmp. c32 is retained
as the ordinary-compiler Goddard initialization control; c45 is its active
compiler-zero GPU validation target. c32 must not consume another GPU compile.

## Release-default carry-forward

On the final RC run the native-facade identity tool
`proofs/s1_port/release_identity.py` for Swiss and PROD d01. Swiss must match
approved glacier hash `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD d01 must match `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
Swiss's previous `2bf12cb8` is pre-glacier and cannot justify skipping its new
24h comparison against original CPU-WRF. Rowc00 now provides the actual new
matched CPU/GPU24h run with full-field D6 and E41 PASS on its recorded source.

PROD d01 identity alone does not prove WN3 children/fused programs identical.
For WN3 carry-forward compare each final compiled program/constant payload and
resolved namelist/inputs/cadence to the accepted v0.3.3/0.3.4 program. Include
all three domains and fused child executables. Only exact identity supports
carrying their existing24/72h D6/station gates. Run one final WN3 release-default
smoke and byte-compare with retained released output: regression evidence only.
Any changed WN3 program needs the affected original-CPU-WRF gates. The new
two-way row is not that one-way release-default identity certificate.

`WN3_PROGRAM_IDENTITY.json` records4/4 exact canonical native-facade programs
(d01,d02,d03,fused d02+d03), all closed constants and every input value/dtype/
shape on current model treecdac4e9 against retained independent RVT08 baseline
40799926d. The comparison uses the released census-OFF scope. An initial
census-ON trace added exactly13 DeviceCensus inputs/domain; that scope mismatch
is retained and superseded, with all other inputs/constants already identical.
This is a source-program certificate; GPU compiler realization and the retained
release's earlier scientific limitations remain separate. The actual3h/12-frame
byte smoke is queued with census OFF. Fidelity composites keep census ON and
report their actual counters.

Historical opt-in GPU PASS receipts can be reused only with this same exact
final-program identity and complete matched-field coverage. The old failed,
incomplete and unavailable-reference rows remain open until actually resolved.

Resume2026-10-14 16:26Z: fresh main e45892a4e (Lin/theta) snapshot; v-runMP28 then t-c02nan c02/c08 then v-prep. Interrupted c17/c21/c22/c23 are setup/incomplete and timing-invalid; bootstrap caches inventoried and cleaned after E41 refused absent stepping executables. All new bulk data/snapshots are in approved DATA scratch. Jensen55 c87 reference is25finite/complete; its hibernation-spanning timing is INVALID, with original receipt preserved. No arm starts after18:45 unless its cap ends by19:10; hard end19:15.
