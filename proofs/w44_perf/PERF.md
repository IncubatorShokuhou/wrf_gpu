# v0.3.2 → v0.4.x performance investigation

**PARTIAL — gate OPEN; three-version headline withheld by the A/A rule.**
The first controlled GPU session completed. The separate deterministic-SDS
cache-key fix is merged and warm WN3 replay passed; it does not explain the
startup lowerings. Device-side ordinary-root delta is source-attributed; the
host/graph side (lost d02 graph capture, H-CB) is reconciled below. Its
recovery is an [I] opportunity until o5-perf's GPU A/B.

## README copy — HOLD until a complete session passes

Provisional sentence (replace the v0.3.2 qualification only after the fresh
session passes): “On an RTX 5090, the same-session warm-cache PROD 3 h proxy
measured 3.10 and 3.13 s per forecast hour of stepping for v0.4.0 and v0.4.3;
the v0.3.2 comparison remains inconclusive because its repeats exceeded the
registered 5% noise limit.”

| Version | Stepping, seg 2–3 mean s/fc-h [M] | Including output/finalization s/fc-h [M] | Startup to first segment s, mean (range) [M] | A/A spread, stepping / period | Status |
|---|---:|---:|---:|---:|---|
| v0.3.2 | 2.904 | 3.008 | 14.044 (12.886–15.202) | 8.71% / 9.59% | FAIL; no relative speed claim |
| v0.4.0 | 3.102 | 3.221 | 17.688 (17.525–17.850) | 0.76% / 1.77% | PASS for stepping/period |
| v0.4.3 | 3.131 | 3.226 | 18.887 (17.197–22.092) | 1.13% / 1.38% | PASS for stepping/period; startup unstable |

All seven measured arms are retained, from one lock session on 2026-10-23,
with identical JAX 0.10.0, explicit cuda_async 0.25 preallocated pool, current
S0 harness, PROD inputs, and 200 root steps. Every arm completed six output
files and had zero JAX cache misses. v0.4.0→v0.4.3 stepping differs by 0.93%,
within the 5% acceptance band; the data support no resolved change at this
precision. Startup is descriptive here: its variability was not a registered
acceptance metric in the first session. These are 3 h iteration proxies,
not the release's 24 h whole-run S1 or WRF-fidelity evidence.
[Sanitized bench receipt](receipts/GPU_BENCH_OCT23.json).

The [fresh-session preregistration](REPEAT_PLAN.json) repeats all three versions
in one new manager-scheduled window, with independent per-version 5% A/A
checks for stepping, period, **and startup**. Oct23 values will not be pooled
into it or substituted for failed arms. The immutable night launcher is posted
to the manager; no worker-launched GPU job is queued.
The current [v4 admission packet](LAUNCH_SMOKE.md) includes exact-CUDA-env CPU
model smoke for both steps and measured caps. Both use only the actual GA
window, so they are eligible for the newly granted day slot or night slot.

## Warm startup caller attribution [M observed GPU, not an S1 timing]

The R43 observer measured 1,094 MLIR lowerings before the first segment,
totalling 4.067 s of lowering-event durations. Six hydrostatic Pallas wrapper
lowerings account for 1.335 s; child-boundary coupled updates account for
0.226 s; tiled MYNN cold-start accounts for 0.102 s. These are duration sums,
not a disjoint wall partition or proof that all duration is avoidable.

All four RRTMG init kernels hit their cache. The first SW init setup took
2.977 s, while its low-level blob lookup/load occupied only about 0.006 s;
Most cost precedes lookup. The observed
pre-segment wall was 18.274 s. The SDS defect is a latent determinism bug:
actual WN3 content walks visited 66,641 nodes and **zero SDS**, and the fixed
key path loaded the exact released native programs with zero new compiles.
No startup speed gain is attributed to that fix.

Caller profiling now identifies the first source-fingerprint AST walk as this
pre-lookup cost: 519 files, 3.55 million nodes, 3.078 s CPU. The approved
[content-addressed import cache](SOURCE_CACHE.md) measures 0.0592 s through
the actual configured cache path in fresh CPU processes, with zero AST parses
and unchanged fingerprints/AOT keys. This recovers about 2.69 s of CPU startup
work per short process [M CPU component]; an end-to-end GPU startup improvement
is not yet measured. Astra approved the source-cache subgate in review
`bb2bf67667`; product `2b8c56012f` is integrated in main, with the exact
reviewed source SHA verified at `d1db02bd62`.

The same CPU caller probe on the historical source trees finds 182 reachable
files / 0.979 million AST nodes in v0.3.2, versus 510 / 3.516 million in v0.4.0
and 519 / 3.553 million in v0.4.3. Unprofiled first calls take 0.730, 3.549 and
3.078 s respectively. These are one-process CPU diagnostics from separate
phases, not a matched GPU speed ratio; they identify source-scope growth as
additional startup work. The cache retains that conservative scope.
[Version caller profiles](receipts/FINGERPRINT_VERSION_PROFILES.json).

## Allocator bridge and sampled clocks [M, descriptive]

The separate same-session AUTO43 arm uses the released C-auto policy: a
3.248 GB budget (fraction 0.096468), versus the comparison's fixed 0.25 pool.
It completes with zero cache misses, 2.9855 s/fc-h stepping and 3.068 s/fc-h
including output/finalization. Per-PID reserved VRAM is 3,898 versus 8,826 MiB;
peak live JAX storage is exactly the same 1,995,383,656 B. This is one bridge
sample, so it does not establish a timing improvement or an allocator verdict.
It does show that the memory policy must be disclosed beside any timing.
[Bridge receipt](receipts/GPU_AUTO_BRIDGE.json).

The unprofiled warm arms provide only 3–4 one-second sensor samples after the
phase boundary margins. Sampled GPU clocks stay within 2,940–2,970 MHz and
memory clocks at 13,801 MHz. Old A/A arm means are 2,956.75 and 2,945.5 MHz;
these samples show no accompanying large clock reduction, but cannot exclude
subsecond clock changes or identify the cause of the failed wall A/A. Sampled
SM means are 83.75–86.5% old, 92.25–93.25% v0.4.0 and 86.5–90.5% v0.4.3.
Those sparse device-wide samples are not an SM-occupancy comparison.
[Sensor receipt](receipts/GPU_DMON.json).

## Current cycle [M node activity, with projection explicitly inferred]

The node captures contain exactly three ordinary roots (68–70) and a separate
three-root window (100–102) containing d01 radiation, d02 radiation and KF.
The corrected [ordinary-root census](ORDINARY_ROOT_DELTA.md) binds the actual
loaded AOT artifacts to the rich HLO donor, including interface, ELF and all
execution fields outside the retained HLO. Runtime-thunk instruction IDs,
launch dimensions, NVTX and graph-clone lineage provide the source context.
Generic shared symbols retain all possible call sites; unresolved prefixes
stay unresolved. This replaces the earlier family labels, which did not
account for deduplicated symbols and small eager executables. Mixed-source
fusion time stays whole; instruction counts never divide its time.

| Current ordinary device family | Kernel sum, three roots ms [M] | Ordinary rate projection s/fc-h [I] |
|---|---:|---:|
| Dycore and advection | 61.866 | 1.375 |
| MYNN | 14.680 | 0.326 |
| Mixed/shared fusions | 10.471 | 0.233 |
| Thompson | 5.594 | 0.124 |
| Boundary/nesting | 2.537 | 0.056 |
| Noah-MP | 1.716 | 0.038 |
| Surface layer | 1.202 | 0.027 |
| Layout-only | 0.714 | 0.016 |
| Other attributed work | 4.064 | 0.090 |
| Unbound or unattributed | 0.303 | 0.007 |
| **All kernels** | **103.146** | **2.292** |

The projection is the three-root sum × (3600/54)/3, not a cadence-weighted
hour or an end-to-end forecast measurement. The mixed radiation/KF window has
22,308 kernels and a 166.872 ms kernel sum. Its earlier complete family labels
are withdrawn: the stricter mapping leaves 8,188 activities / 35.661 ms outside
the proven module scope because their graph-creation events precede the late
capture. The window is not averaged as an ordinary root. Radt is 30 minutes
and cudt is 5 minutes; a complete hourly family split remains open.
The corrected ordinary source and per-domain counts are in
[the bound census](receipts/ORDINARY_ROOT_DELTA.json).

| Same node-capture scope | v0.3.2 [M] | v0.4.3 [M] |
|---|---:|---:|
| Kernels, three ordinary roots | 8,292 | 10,227 |
| Kernel sum ms | 93.476 | 103.146 |
| Device-operation active union ms | 97.073 | 107.354 |
| DtoD rows / decimal GB | 2,205 / 4.035 | 1,758 / 3.701 |
| DtoD duration sum ms | 4.034 | 4.972 |
| HtoD rows / bytes | 18 / 120 | 18 / 120 |
| DtoH rows / bytes | 39 / 39 | 30 / 30 |

These snapshots establish more kernels and fewer explicit copy bytes in the
newer ordinary capture. The d01 copy path also changes representation:
explicit DtoD activities plus graph copy-helper kernels go 429→423, while the
copy-helper kernel count goes 12→417. Large copied arrays have not disappeared
just because they no longer appear in the explicit memcpy byte counter.
All device activity rows (kernel+memcpy+memset) rise 10,554→12,033, **14.01%**;
kernel rows alone rise 23.34%. These are diagnostic, not a causal wall verdict:
the old version failed A/A, and node tracing changes dispatch cost. Small
transfers still matter as potential dependencies even when their byte count
is tiny. The graph-mode captures hide internal nodes/copies, so their copy
totals must not be compared as real traffic.

## Dips: measured overlap and remaining CPU attribution

The current graph capture spans 3,275.838 ms including the output boundary;
the kernel/copy/graph-envelope union is 2,696.341 ms. A graph envelope includes
internal gaps, so this is an upper bound on this process's device activity.
The three ordinary node roots span 142.576 ms with a 107.354 ms operation
union; the radiation/KF window spans 225.531 / 170.437 ms. These are activity
intervals, not SM occupancy or device-wide utilization.

Every observed idle gap above 1 ms is partitioned into disjoint time slices.
Only the overlap actually covered by an API/output interval receives that
label; a brief API overlap never labels the entire gap.

| Current graph capture, gaps >1 ms | Disjoint overlap ms [M] |
|---|---:|
| Outside CUDA API; CPU cause unresolved | 202.235 |
| Launch / graph update API | 80.143 |
| Output callback (clock alignment ±0.5 ms) | 48.844 |
| Transfer API | 9.847 |
| Other CUDA API | 3.805 |
| Allocation API | 1.652 |
| Sync API | 0.703 |

For the three ordinary node roots, uncovered time is 13.204 ms and launch/
update overlap is 11.211 ms; for the radiation/KF window these are 21.870 and
8.578 ms. Runtime `cuStreamSynchronize` wall totals 2.026 s in the graph
capture, but almost all overlaps active device work: calling all of it idle
would be wrong. CUDA control-flow graph coverage also differs: ordinary old
kernels have 7,767 graph nodes / 525 standalone launches, versus current
4,377 / 5,850. The 5,850 = 3 × 1,950 are exactly the d02 kernels; their
compiler cause is now resolved by o5-perf (H-CB, next section).

The Python sampling option produced no Python sampling tables in the SQLite
export. The uncovered time therefore remains a bounded CPU-attribution gap;
it is not assigned to Python, IO, compile, or a particular caller without
evidence. No warm JAX cache misses occurred. The compiler observer separately
locates init lowerings before stepping. Output is a visible dip contributor;
its unprofiled warm cost is about 0.095 s/fc-h. **The lane gate remains open
until the residual CPU gaps and final matched version result are resolved.**
[Every gap and API accounting](receipts/GPU_TRACE_SUMMARY.json),
[init caller receipt](receipts/GPU_INIT_CALLERS.json).

Existing same-thread NVTX scopes further locate 104.558 ms of the uncovered
202.235 ms inside XLA thunks (59.804 ms in the outer d01 while), and 97.677 ms
outside an enclosing thunk. OSRT waits cover only a few milliseconds. These
are execution scopes, not a claim that the thunk itself caused the stall.
[Compact scope receipt](receipts/CPU_GAP_SCOPES.json). A second immutable
night launcher profiles Python calls only during segment 2 to resolve the
remaining callers; it runs after the matched timing arm and its profiled wall
will never enter the README speed table.

`cudaProfilerStop` blocks for profiler cleanup (about 4–6 s). Its return
timestamp and profiled terminal S0 period are excluded from speed claims.
CUDA activity clocks are aligned to the profiler-start receipt event, with
±0.5 ms timestamp uncertainty. Actual cold v0.3.2 artifacts fail E41's large
kInput test; v0.4.0 native scans have stack 32 B and no measured violations,
but compact AOT HLO metadata remains UNKNOWN, so its aggregate verdict is
INCOMPLETE. Neither baseline is a proposed release candidate.
[Actual cold artifact scans](receipts/GPU_COLD_E41.json).

## Host/graph side: lost d02 graph capture (H-CB) [M source/CPU replay; effect I]

The family and cohort tables above count **device kernel time only**. The
host/graph side was found by o5-perf (lane/o5-perf, `proofs/o5_perf/HCB_FIX.md`,
receipts `CB_PARTITION_SIM_{V032,PROD,WN3}.json`, `HOST_CALL_TIMES_OCT23.json`):

- **Cause [M source + CPU replay of the stored options].** Since 6e38cbf173
  (v0.3.3) the step programs enable `+WHILE` command buffers. The pinned XLA
  conversion pass then appends the convertible outer step `while` to a pending
  sequence without descending into it. A top-level sequence shorter than
  `xla_gpu_graph_min_graph_size` (stored 5) stays unconverted. PROD d02's top
  level has 4 thunks → 0 command buffers, so all 1,950 d02 kernels/root are
  host-launched behind a host-driven loop. d01 has 12 top-level thunks → one
  WHILE graph. v0.3.2 stored no WHILE, so the pass recursed into the bodies:
  21 d02 / 20 d01 command buffers, with 11 / 10 host while loops.
- **Independent consistency with our captures [M].** The current node capture
  has 5,850 standalone launches = 3 roots × 1,950; v0.3.2 has 525 (175/root).
  The graph-level G43 capture has only 3 `cuGraphCreate` calls.
- **Host call times [M, o5-perf from our Oct-23 unprofiled arms].** Median
  host time per call, in ms (`host_s` includes blocking on in-call syncs, so it
  is not pure CPU work):

  | Call | v0.3.2 | v0.4.0 / v0.4.3 |
  |---|---:|---:|
  | d02 | 30.0–31.4 | 35.4–37.8 |
  | d01 | 5.9–7.4 | 2.7–5.2 |
  | Per-root sum | 39.7–43.8 | 44.7–45.4 |

  d01 now returns early (async graph); d02 absorbs the waiting.

**What this changes in the loss accounting [I, mixed scopes].** Per-root
stepping wall = seg 2–3 stepping × 54/3600. Subtract the node-capture device
union per ordinary root (97.073/3 = 32.36 ms old, 107.354/3 = 35.79 ms new):

| | v0.3.2 | v0.4.3 |
|---|---:|---:|
| Stepping wall per root (arm range) | 41.7–45.5 ms | 46.7–47.3 ms |
| Residual (non-device) | 9.3–13.1 ms | 10.9–11.5 ms |

[Arithmetic receipt](receipts/HCB_RECONCILE.json) (`reconcile_hcb.py`).
The residual shows no resolved growth inside the failed v0.3.2 A/A band. The
mean wall delta (+3.4 ms/root) is the same size as the device-union delta
(+3.43 ms/root).

This does not prove that kernel growth *causes* the wall growth. Stepping
includes radiation/KF roots, node tracing perturbs dispatch, and the v0.3.2
A/A failed. What the data do show:

- H-CB moved the host residual from 21 host while loops (v0.3.2) to one fully
  host-launched d02 program (v0.4.x).
- H-CB did not measurably enlarge that residual on PROD.
- H-CB is therefore a **recovery opportunity** for an ≈11 ms/root host residual
  that both versions carry. It is not an additional explained regression.

**Opportunity and gates.** Capturing d02 (`GPUWRF_STEP_CMDBUF=v1`,
`min_graph_size=1`, step jits only, default off) gives identical optimized HLO
and native records 0/1/2 in a deviceless compile of the PROD d01/d02 top-level
programs. Only DebugOptions 208/258 differ (o5-perf [M CPU]). This holds for
the symptom and those programs only: the v1 prototype as written must first be
fixed for nested callers. Under v1/v2 the fused WN3 cascade cannot be traced
(w44-perfplan H-CB verdict F1, PP15). Upper bound if the whole residual vanished:
11.2 ms × 66.7 roots/h ≈ 0.75 s/fc-h. o5-perf estimates −0.5…−0.6 s/fc-h. Both
are **[I], unproved envelopes**. Open risks:

- conditional-node overhead on radiation roots (A42: +≈10 ms per radiation root
  on LW7);
- per-call graph updates of ≈2,000 d02 nodes.

Evidence = o5-perf GPU A/B `cbab` (25.10 window): wrfout bytes identical,
interleaved warm roots, A/A ≤5%, nsys graph counts. Prerequisite: same-snapshot
OFF arms. Without them OFF (sealed v0.4.3 source) and V1 (prototype snapshot)
differ by 29 source files as well as by the option (E176). The WN3 release
workload (fused d02+d03, one command buffer) does not show the *symptom*; that
says nothing about the prototype, which must not break the fused cascade. This lane does not repeat
that A/B. A positive result would put PROD stepping *below* the v0.3.2 level;
it does not change any fidelity commit.

## Preserved timing evidence [M historical]

| Case / receipt | Whole reported s/fc-h | Pre-first-segment s | Steady period median s/fc-h | Per-PID peak MiB |
|---|---:|---:|---:|---:|
| v0.3.2 VAL32 selected | 3.305 | 15.008 | 2.625 | 8,794 |
| v0.4.2 HERO H1 | 4.847 | 27.216 | 3.453 | 3,898 |
| v0.4.2 HERO H2 | 4.770 | 25.969 | 3.399 | 3,898 |

These are different sessions and allocator policies. VAL32 measures its process
t0→receipt end. HERO measures harness launch→last original file (including a
CLI allocator re-exec); its measured-t0→receipt rates are 4.750/4.667. Steady
periods include output. New stepping-only medians are 3.229/3.247; the selected
old receipt omits callback arrays, so its matching stepping-only value is UNKNOWN.
The last segment's call period excludes finalization here; official S0 terminal
periods will be disclosed separately in the controlled comparison.

The historical 1.503 s/fc-h difference decomposes arithmetically as follows:

| Difference, mean HERO minus VAL32 | s/fc-h [M historical] |
|---|---:|
| Before the first segment | +0.483 |
| First-segment start through final-segment return | +0.898 |
| After final-segment return through receipt end | +0.024 |
| Remaining timing-basis difference | +0.099 |

This explains the accounting, not the necessity or code cause of any increase.
The exact three disjoint phase sums reproduce each receipt's `t_end`. Sources
and SHA-256 are in [the receipt audit](receipts/HISTORICAL_RECEIPT_AUDIT.json);
[the parser](receipt_audit.py) does not import JAX or inspect raw profiler data.

## Dips already visible in preserved dmon [M historical]

Whole-process sampled SM means are 53.7/56.2%, heavily diluted by startup.
Restricting to segment 2 onward, with a one-second margin at phase boundaries,
gives 79.0/81.3% over 77 samples each. Eight/three samples are below 50%, with a
minimum of 3%. Startup means are 6.3/6.0%. Thus the visible dips are not solely
startup. These are device-wide utilization samples including desktop work;
they are not a kernel busy-time union and cannot identify millisecond causes.
Host-local timestamps were aligned using Atlantic/Canary and checked against
each preserved launch timestamp (first sample within two seconds).

## Static screen [M CPU, not a GPU speed measurement]

The v0.3.2 tagged PROD d01 step compiled for captured SM120 with AUTOTUNE0:
147 native Pallas calls, 1,001 fusions, 399 transpose fusions, 786 native kernels;
maximum stack 32 B. Five kInput transpose fusions exceed 500 HLO instructions
(largest 1,006). The screen took 278.4 s, including 114.0 s backend compilation,
with 6.34 GB peak RSS. This is a reason to retain the later root layout fixes
while measuring their cost. The actual GPU artifact remains authoritative.

The same-target v0.4.3 screen completed in 328.4 s (126.3 s backend, 6.30 GB RSS).
It has 161 native Pallas calls, 1,375 fusions, 557 transpose fusions, 1,104 native
kernels, maximum stack 32 B, and zero kInput fusions above 500 instructions.
Both screens include all compiled branches; these are not cadence-weighted
launch counts. Source metadata from the same printed HLO assigns the largest
static fusion-count additions as follows:

| Dominant source file | v0.3.2 | v0.4.3 | Difference |
|---|---:|---:|---:|
| `coupling/boundary_apply.py` | 56 | 123 | +67 |
| `dynamics/flux_advection.py` | 55 | 108 | +53 |
| `kernels/layout_pin.py` | 0 | 44 | +44 |
| `physics/surface_layer.py` | 7 | 47 | +40 |
| `runtime/operational_mode.py` | 84 | 117 | +33 |
| `physics/noahmp/energy.py` | 4 | 33 | +29 |

Mixed-source fusions retain their instruction-source histogram in the artifact;
the dominant source is a static label, not a proportional time allocation.
Attribution is missing for 135 old and 35 new fusions, so individual source
count differences also reflect changed metadata coverage. Explicit static
`copy` instructions are 137/139; their byte traffic remains UNKNOWN.
The [summary](receipts/STATIC_SCREEN_SUMMARY.json) binds receipts and HLO hashes.
DEV-only temporary caches are removed after each screen; the immutable old
snapshot remains for its GPU arms. The child-domain screen (three own steps)
also completed:

| d02 static metric [M CPU AUTOTUNE0] | v0.3.2 | v0.4.3 |
|---|---:|---:|
| Native Pallas calls in trace | 139 | 151 |
| Fusion definitions | 781 | 928 |
| Transpose fusions | 87 | 129 |
| Native kernels | 641 | 753 |
| Maximum stack B | 32 | 32 |
| kInput fusions above 500 instructions | 1 (509) | 0 |
| Backend compile s | 95.4 | 107.4 |

Source-attributed fusion additions are led by `physics_couplers.py` (+48),
RRTMG LW (+24), SW (+16), constants (+13) and Noah-MP energy (+12).
They remain all-branch counts, not cadence-weighted device costs.
[Child summary](receipts/STATIC_SCREEN_D02_SUMMARY.json).

## Startup import hypothesis [M CPU diagnostic]

Fresh CPU processes imported JAX, CLI and pipeline in 1.434 s (v0.3.2) and
1.532 s (v0.4.3), with 187/243 gpuwrf modules, 20 import-time JAX compilations
each and 0.805/0.810 GB peak RSS. This does not reproduce the historical 11.6 s
startup increase as pure Python import growth. CUDA executable load, GPU init
and output preparation remain unmeasured. No lazy-import prototype is justified
by this screen. [Receipts](receipts/IMPORT_SUMMARY.json) preserve this scope.

## Verlust verstanden

**Ordinary-Root-Delta quellengebunden; kein schneller Prototyp unter den
bindenden Gates qualifiziert.** Die vorhandenen drei vollständigen PROD-Roots
68–70 zeigen 8.292→10.227 Kernel (+23,34%), aber 10.554→12.033 gesamte
Device-Aktivitäten (+14,01%) und +9,669456 ms summierte Kernelzeit.
Diese Trace-Kohorten sind kein +23%-Wallclock-Claim und kein vollständiger
Strahlungs-/Stundenvergleich. Der alte Wall-A/A-Test bleibt verletzt.

Die Commit-Zwecke in der folgenden Tabelle belegen bewusste WRF-Arbeit
beziehungsweise Compiler-Schutz. Ihre einzelnen Millisekunden sind ohne
kausale Ablation **nicht** isoliert. Insbesondere wird nicht behauptet,
dass jede zusätzliche Fusion wissenschaftlich unvermeidbar ist.

| Quellkohorte | alt | neu | Δ Kernel | Δ ms | Commit-/Zweckbeleg; Entscheidung |
|---|---:|---:|---:|---:|---|
| Dycore/Advektion | 3,132 | 3,681 | +549 | +4.683 | 150628ddf6, f3c5967654, 6e38cbf173 — Physik→RK, W-Dämpfung sowie Layout/Fission; Q1 ohne qualifizierten Prototyp geschlossen |
| Graph-Helfer außerhalb Modell-CUBIN | 18 | 516 | +498 | +0.824 | kein einzelner kausaler Commit — 405 davon Kopierpfad; keine zusätzliche Physik, nicht auf Anzahl optimieren |
| Noah-MP | 345 | 723 | +378 | +0.707 | 3814b53fc7, b4b36bcc66, 6e38cbf173 — QSFC/Q2 korrekt halten; Q2 No-Go ohne konkreten Export-Redundanzbeleg |
| Randbedingungen im Step | 678 | 894 | +216 | +0.975 | 0dcd90e126, 659482b8d0; Defaults 6e38cbf173 — WRF-RK-/spec_bdy_final-Takt behalten |
| Runtime-/Staging-Quellen | 288 | 414 | +126 | -0.502 | mehrere, nicht isoliert — kein eindeutig neues Physikverfahren |
| MYNN | 1,068 | 1,188 | +120 | +1.876 | a054d46773/ac278549e6, f771035c95 — qni-Mischung, WRF-Caller und Konstanten behalten |
| Oberflächenschicht | 264 | 351 | +87 | +0.300 | 05a7bac11f/ac278549e6, 36fe022a0b, f771035c95 — PSIQ, REAL-PSI und Callergrenzen behalten |
| Skalare HLO-Steuerung | 342 | 426 | +84 | +0.069 | kein einzelner kausaler Commit — Zähler/Steuerung, nicht neue Flächengleichungen |
| Gemischte/geteilte Fusionen | 1,179 | 1,245 | +66 | -0.567 | mehrere Quellen pro Fusion — Zeit bleibt ungeteilt; kein Name→Familie-Transfer |
| Force-down-Aufrufbereich | 297 | 357 | +60 | +0.057 | 6e38cbf173 — kompakte Streifen-Fences; nur 0.057 ms Mehrzeit, geringe ROI |
| Nur Layout-Quellen | 90 | 117 | +27 | +0.261 | 6e38cbf173 — Root-Pins sind Compiler-Schutz; kleines Gitter-Firewall hier inaktiv |
| Hydrostatische Vorbereitung | 0 | 12 | +12 | +0.368 | 6e38cbf173 — nativer REAL-Pfad ersetzt Vorbereitung |
| Quellframe nicht zuordenbar | 138 | 144 | +6 | -0.029 | offen — sichtbarer Rest, nicht einer Familie zugeschlagen |
| Physik-Vorbereitung/Kopplung | 42 | 39 | -3 | -0.081 | 150628ddf6, 6e38cbf173 — coupling/ ist nicht pauschal Randbedingung |
| Schwerewellenwiderstand | 54 | 48 | -6 | +0.017 | mehrere — kleine Fusions-/Zuordnungsverschiebung |
| Thompson | 63 | 51 | -12 | +1.235 | aaf2e85576, b7864e0553, 6e38cbf173 — weniger Familien-Kernel, mehr Zeit; Q3 ohne materiellen Gewinnbeleg geschlossen |
| Präfix/Epilog/kleine Executables nicht gebunden | 294 | 21 | -273 | -0.522 | offen — -273 keine isoliert belegte Rückholung |
| **Summe** | **8,292** | **10,227** | **+1,935** | **+9.669** | jede Aktivität genau einmal |

Fidelity wird nicht zurückgenommen: Physik-/PBL-/Strahlungstendenzen müssen
im richtigen RK-Verbrauch landen (150628ddf6); W-Dämpfung muss den WRF-Stufentakt
behalten (f3c5967654); Root-/Nest-Skalarränder brauchen die korrekten RK1-,
Flow-Dependency- und Finalaufrufe (0dcd90e126,659482b8d0). Die qni-, PSIQ-,
REAL-PSI-, Noah-QSFC/Q2- und Thompson-Mixed-/Driver-Korrekturen bleiben mit den
in der Tabelle genannten Commit-Belegen erhalten. 6e38cbf173 enthält zusätzlich
Compiler-Schutz und native Ersatzpfade; es ist kein einzelner neuer physikalischer
Prozess. Die alten exakt gebundenen HLOs haben acht/zwei oversized kInput-
Fusionen in d01/d02, aktuell null/null. Root-Pins und gezielte Schutzgrenzen
werden deshalb nicht pauschal entfernt. Die kleine-Gitter-Firewall ist auf
PROD inaktiv.

Ein erheblicher Teil des Zähleranstiegs ist Darstellung/Staging: +405
Kopierhelfer stehen expliziten d01-DtoD-Zeilen 417→6 gegenüber; die Gesamtzahl
der d01-Kopieraktivitäten inklusive Helfer ist 429→423. Speicherverkehr ist
dadurch nicht verschwunden. Native Pallas-Aufrufe steigen nur um 81; diese
Operationsklasse ist bereits in den Quellenkohorten enthalten und wird nicht
dazuaddiert.

Die begrenzten Audits sind abgeschlossen:

- **Q1:** 366 gebundene Aufrufstellen, 566 ausgegebene Werte, alle SSA-live,
  keine exakt doppelt ausgegebene Fusionsberechnung. Identische d02-Teilausdrücke
  existieren, werden aber nicht bereits als gemeinsame Ausgaben materialisiert;
  gemeinsames Bereitstellen würde neue Vollfelder bis rund 33 MB benötigen.
  Die konkreten bereits ausgegebenen d01-Werte gehören insgesamt zu einer
  überinklusiven aktuellen Kernelkohorte von nur 0,045536 ms/root.
  [Def-use-Belege](Q1_CROSS_FUSION_AUDIT.md).
- **Q3:** Zwei getrennte Integer-MAX-/Shared-/Barrier-Pfade des Warm-/Cold-
  Selektors sind im gebundenen vorhandenen PTX bestätigt. Die logische
  Zusammenlegung ist äquivalent, aber ≥0,5 ms/root Nettogewinn und die
  Ressourcen einer geänderten Variante sind nicht belegt. Statische BAR-Zahlen
  sind kein Sync-Zeitbeweis. Kein Prototyp/Lower. [Beleg](Q3_SELECTOR_AUDIT.md).
- **Q2:** Kein konkreter toter/doppelter EF-/ET-Export belegt; No-Go bleibt.
  Ein gespeicherter kurzer Noah-/Surface-Screen hat 375 Ausgaben, alle SSA-live,
  ohne exakt doppelt ausgegebene Berechnung. Keine Schutzbarriere verändert.

Die +7,100 ms nichtnative Dycore-Differenz, +1,007 ms Noah+Surface und
+2,020 ms native Thompson-Differenz pro drei Roots sind beobachtete
Kohorten-Mehrbudgets, **keine mathematischen Rückholgrenzen oder Q1/Q2/Q3-Gewinne**.
Es wurde kein Kandidat mit belegtem Nettopotenzial über der bindenden
0,5-ms/root-Schwelle und allen Memory-/Fidelity-Gates gefunden. Das ist das
gültige negative Ergebnis dieses kurzen Versuchs, kein Nachweis einer globalen
Roofline oder dass jede verbleibende Millisekunde unvermeidbar ist.

**Host-Seite (H-CB, Abschnitt oben):** Die Kohortentabelle zählt nur
Device-Kernelzeit. Der Nicht-Device-Rest pro Root liegt in beiden Versionen bei
etwa 11 ms (v0.3.2 9,3–13,1 im gescheiterten A/A-Band, v0.4.3 10,9–11,5) [I].
Der verlorene d02-Graph seit 6e38cbf173 verschiebt diesen Rest, vergrößert ihn
aber messbar nicht. Er ist eine Rückhol-Chance (o5-perf, GPU-A/B 25.10), keine
weitere erklärte Regression und keine Fidelity-Frage.

Der separat bereits integrierte Source-Fingerprint-Cache spart gemessen rund
2,69 s CPU-Arbeit pro frischem Prozess mit identischen Keys; er verändert keinen
Ordinary-Root-Kernel. Offene Arbeit bleibt ausdrücklich getrennt: der neue
gleiche-Session-A/A-Repeat für README/Startup, Hostprofil für verbleibende
Init-/Idle-Aufrufer und vollständige Strahlungs-/Stundenprovenienz. Nach dem
Netzteil-Ausfall 07:10–14:29Z sind die Dateien und beide Launcher geprüft.
Der Ersatzslot am 24.10 ist ausschließlich warmstart→raddiag→budget vorbehalten;
repeat_v4/hostprofile_v4 sind nach Managerplanung für den **25.10** vorgesehen.
Die unveränderten SHA256 und vollständigen Receipts stehen im
[Neustart-Integritätsbeleg](receipts/POST_POWER_LOSS_CHECK.json). Kein eigener
GPU-Start, keine bestehende numerische Validierung wiederholt.

## Recovery candidates

The frozen [plan](PLAN.md) defines the common pool/harness, interleaved version
order, 5% A/A acceptance rule, graph-level host timeline and separate three-root
node capture. The current bundle's numerical/native identity versus v0.4.2 is
already established by REL43-ID (21/21 programs); it is reused.

| Item to test | Reason / next evidence | Recoverable s/fc-h |
|---|---|---:|
| Source fingerprint AST scan | Integrated content-addressed cache 2b8c56012f; exact keys, 21 focused and 84 consumer passes | 2.69 s/process CPU measured; GPU S1 effect UNKNOWN |
| Remaining cached executable readiness and init lowering | 1,094 lowerings; six hydrostatic wrappers account for 1.335 s of duration sums | UNKNOWN |
| MYNN/Thompson and physics→RK changes since v0.3.2 | WRF fidelity commits source-attributed; bounded Q1/Q3 audits closed without qualifying prototype | UNKNOWN |
| Root layout pins and E41 barriers | E41 protection retained; copy representation resolved, no concrete dead/double EF/ET export | UNKNOWN |
| Output and host dispatch gaps | dmon confirms stepping dips; graph-level CUDA/OSRT/Python timeline needed | UNKNOWN |
| d02 graph capture (H-CB, o5-perf) | +WHILE since 6e38cbf173 leaves PROD d02 (4 top-level thunks < min 5) fully host-launched; deviceless HLO/native identical with `GPUWRF_STEP_CMDBUF=v1` for the PROD top-level programs only (symptom only; the v1 prototype must first be fixed for nested callers, fused cascade F1); GPU A/B cbab on 25.10 needs same-snapshot OFF arms first (E176) | ≤ 0.75 ceiling, −0.5…−0.6 estimate [I] |
| Graph parameter updates | D10/D14 already measured temp-buffer address churn despite carry donation; no supported persistent-temp flag in this XLA; avoid repeating the same hypothesis | UNKNOWN |
| Allocator policy | Historical pool peaks differ; common explicit pool plus current C-auto bridge | UNKNOWN |

The source-key candidate follows a measured caller attribution. No fidelity
change, old-release-as-truth comparison, or release gate waiver is proposed.

The localized child Thompson check is closed without a qualifying prototype: its three-root
native duration sum is 2.503 / 4.500 ms in the old/current node snapshots.
Actual cubins use 166 / 200 registers, no stack/local storage, 18,928 / 19,304
static SASS instructions and 171 / 341 static BAR instructions. Both launch
31,239 blocks of 64 threads. At the captured 65,536-register/SM limit, the
register-only resident-block upper bound changes from six to five [I]; this
is not a measured occupancy or causal latency proof. Static DOUBLE add/mul/FMA
counts are similar (1,100 / 1,091), so static instruction counts alone do not
explain runtime. Source changes include WRF driver's vapor copyback/negative
repair and mixed-phase single-pass semantics. Those fidelity rules must stay;
the existing selector has two separate MAX-/Shared-/Barrier paths, but no
measured material net benefit from combining them. Its nominal device
difference projects to 0.044 s/fc-h [I] for the whole native cohort, not a
recoverable selector budget. Q3 closes without using the one prototype/Lower arm.
[Exact native resource receipt](receipts/THOMPSON_RESOURCES.json).
