# V044 MYNN/PBL budget program

Status: diagnostic-only twin (manager design B), CPU component qualified for
EDMF-on and EDMF-off. Public bindings, separate carry, three restart APIs and
the history writer are implemented and CPU-checked. Coupled GPU identity,
GPU term checks, E41 and GPU cost remain pending. No feature is release-ready.

**Critic FIX (cr-w44-budget, 21:55Z) — 2026-10-24 23:35Z.** F1: the PBL rule was
pre-registered at `49a3636970` and the P75 re-score passes 168/168 with 64/64
control kills (see below). F3: as in WRF's `mynn_dmp_edmf` package,
`bl_mynn_edmf=0` domains omit MAXMF/MAXWIDTH/ZTOP_PLUME from history (host
writer only). F2: the schema-2 native-first full-constant OFF identity of main `aac9e78065`
vs `5438aca4b5` is IDENTICAL for Swiss and PROD nested (P72,
[receipt](receipts/P72_SCHEMA2_OFF_IDENTITY.json)).
The sealed v7 launchers are unchanged. Their in-launcher PBL verdict does not
qualify; the GPU arrays get the pre-registered offline re-score with
`twin_oracle.py` at `49a3636970`.

**Current state (2026-10-24 21:45Z, takeover).** Lane head rebased onto main
`aac9e78065`; the reader registry keeps only budget's own three sites. The
three composition-red seam tests (plus two hidden restart stubs with the same
arity bug) are fixed in tests only: 437 seam tests and `tests/v025/restart`
(250 PASS / 4 SKIP) pass. Canonical OFF identities on the rebased head:
Swiss `b977712f`, PROD `321aa651` EXACT ([receipts](receipts/CURRENT_GATE.json)).
GPU launchers v7 (model `snapshot_p68`) pass the E225 dry-run and alignment
checks; the component-only CUDA mini passes all phases on CPU (P69). See
[CUDA_MINI.md](CUDA_MINI.md). Coupled GPU BYTE, GPU term checks, E41 on the new
executables and GPU cost are still pending.

Truth is pristine WRF 4.7.1. `inventory.py` extracts the exact 21 requested names,
descriptions, units, dimensions and staggering from `Registry.EM_COMMON`, with
SHA256 pins for the Registry and the three original MYNN/PBL source files.

WRF calls the switch `tke_budget=1`; `bl_mynn_tkebudget` does not exist in this
Registry. EDMF history uses `bl_mynn_output=1`.

| Fields | Work needed |
| --- | --- |
| qSHEAR, qBUOY | Retain existing turbulence terms; apply original surface boundary and vertical destagger. qBUOY must include the original half plume TKE-production term. |
| qDISS, qWT | Already returned by the predictor and discarded; export them, preserving original unused-top convention. Default `bl_mynn_edmf_tke=0`. |
| Dtke | Cheap post-solve `0.5*(qke_new-qke_old)/dt`, with unused top zero. |
| edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_qc | Already computed by the one existing plume call. |
| edmf_ent | Requires exporting the transient per-plume entrainment ENT and its area-weighted mean; new native plume output. |
| sub_thl3D, sub_sqv3D, det_thl3D, det_sqv3D | Exact zeros in the original standard build: `env_subs=.false.` at module_bl_mynnedmf.F:343. |
| RTHBLTEN, RQVBLTEN, RQCBLTEN, RQIBLTEN, RUBLTEN, RVBLTEN | Raw PBL tendencies already exist at the adapter seam; preserve WRF post_run mixing-ratio conversion and mass-grid momentum convention. |

The TKE Registry metadata says `m2 s-2`, while the original update comment
describes rates in `m2 s-3`. Output metadata must stay as registered. The four
production/transport fields have Registry Z staggering, while the original
driver copies only kts:kte after destaggering; the writer needs an original-file
shape check. `Dtke` is a local MYNN rate, not total forecast TKE change including
dynamics. Compare `Dtke` with qSHEAR+qBUOY+qWT-qDISS using the same pristine budget
residual, including its density, surface and clipping conventions.

First gate: expose already-computed qWT through the compiled native MYNN program
on actual real-domain columns. Compare every pre-existing result by dtype, shape
and bytes, rejecting non-finite operands/results. No eager comparison qualifies.
If BYTE fails, stop and hand the measured difference to the manager before
adding terms, public State leaves, restart, writer or option bindings. Passing
this component screen is not a full coupled-program identity claim; that also
requires the WN3 0227 3-nest 3 h arm, one shared compile/autotune cache and E41.

| Required evidence | Status |
| --- | --- |
| Original term inventory | 21 exact fields; [P03 inventory](receipts/P03_inventory.json) |
| Compiled first prototype | Actual WN3 d01 GPU qWT prototype: **183/183 BYTE**. OFF 3 h complete; ON fused compile hit cap. [Night analysis](receipts/NIGHT1_ANALYSIS.json). Expanded direct returns rejected 18/25 CPU and reverted. Twin day/night component: 50/50 old MYNN outputs over two steps in each case; final corrected artifact also 50/50 in each. Coupled GPU twin proof pending. |
| Per-term pristine oracle, real day/night and active plumes | All 21 twin terms PASS on real WN3 d03 day/night column sets, both steps, against unmodified public WRF driver at the actual input states; selected-column CPU scope only. Original base [P14](receipts/P14_pristine_reference.json). |
| Pristine-relative budget closure | Twin residual vs original residual PASS both steps/day/night; no artificial zero closure. |
| Deletion-sensitive tests and non-finite controls | ENT initializer deletion: 1 genuine assertion kill, 0 setup errors, 25 failing active columns. Five amended-gate control methods reject 152 bad cases. Broader source-deletion coverage remains pending. |
| OFF identities Swiss b977712f / PROD 321aa651 | Exact on the composed public program using the official probe with temporary lane path relocated only: [Swiss](receipts/P46_OFF_SWISS.json), [PROD](receipts/P47_OFF_PROD.json) |
| Own cheap/AOT key, catalog refusal, restart schema, history writer | WRF switches bound per domain; distinct ON static key, optional 5/16-field carry, schema 1 in pickle/native/NetCDF restart, exact Registry writer. 149 structural checks PASS; final writer 34 PASS / 3 existing unavailable-reference skips vs main 33 / same 3. All 21 enabled fields read back. [Default writer](receipts/P45_DEFAULT_WRITER_GATE.json) matches main for 51 minimum and 375 full fields, every schema/metadata/data hash. |
| ON kernel/copy, VRAM and s/fc-h cost, E41 | GPU twin cost pending. Legacy paid-cache recovery: native max STACK 32 B/no measured violations, 8/8 coverage; overall INCOMPLETE because 3 slim AOT HLOs are unknown. |
| Item 2 bl_mynn_edmf=0 | Same original oracle and two-step component checks PASS day/night; all 10 EDMF histories exactly zero. Public binding and dependent selector refusals CPU-checked; GPU qualification pending. |

The historical minimal production-source prototype appended optional `State.mynn_qwt` and
column `qwt`, exports the predictor's existing result, and retains it at the
adapter/RK seam. OFF remains `None`; no public option, writer or checkpoint
support was enabled. It was retired for design B, including its State and
column slots. Production MYNN kernels/adapter and State/precision files now
match composed main `d1db02bd62` exactly. The old snapshot remains only for
the historical measured night receipt; it is not the current feature.

[P12](receipts/P12_cpu_lower.json) lowers actual WN3 0227 d01 OFF and ON on CPU
with the real namelist/clock, dt 54 s, cadence 33 and one traced step. This is a
type/structure check, not compiled GPU identity or WRF fidelity. The immutable
GPU plan at source commit `3fa48d159766101770a787fb9947c0622c7c4456` first checks
every old d01 carry leaf, then conditionally runs nested 3 h OFF/ON plus warm
ON/OFF in one official lock session, compares terminal and wrfout bytes, and
scans actual cache executables with E41 before cleanup. The manager started that
night run; the worker did not start GPU work. See the night analysis above.

Measured CPU component memory: output 6,548,168 to 6,858,640 B (+310,472 B),
temporary 18,957,560 to 18,535,736 B. CPU interleaved medians were 23.4844 and
23.3914 ms. These are component CPU observations; GPU kernels/copies, VRAM,
forecast cost and full-program identity remain unknown. No speed claim is made.

The reference-only P14 check uses the unmodified public WRF
`mynnedmf_driver` objects and the committed F20 caller/real-column fixture.
Exporting all requested terms with `tke_budget=1` preserved all 18 previous
public outputs byte-for-byte across six sets (216 OFF/ON field comparisons).
The active-plume column counts were day 9/22/25 and night 8/10/12 for PROD
d01/d02 and WN3 d03; nighttime inactivity must be measured per fixture.
The four sub/det diagnostics and unused top planes were exactly zero.
The pristine maximum absolute closure residual ranged from 0.0004031 to
0.013866 m2 s-3; this is reference evidence, not a candidate closure pass or
a new tolerance. The small reference arrays remain outside git under
`<USER_HOME>/wrf_gpu2_lanes/w44-budget/P14_pristine_reference`.

The manager's unattended entry point is now
`<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_launch.sh` (SHA256
`c42b1071f45cd28fe60b8252065ad4d0e06ed151f5d18e1a19c2f05c3cef28aa`).
It requires `GA_START_UTC`/`GA_END_UTC` and a full 1800 seconds remaining,
requests the official lock with `--timeout 1`, runs the unchanged immutable
prototype in a private process group, and tears down orphaned descendants on
timeout or cancellation. The 30-minute cap includes the GPU phase (up to 22
minutes), E41 (up to five minutes), and teardown margin. It reports both phase
return codes and preserves cold evidence for the manager's gate disposition.
[CPU controls](receipts/NIGHT_LAUNCH_CPU_CHECKS.json): 10 tests PASS, including a
TERM-ignoring `setsid` descendant retaining a dummy flock; the actual frozen
entry point refused an intentionally short window with rc 4 before any GPU
lock. The manager sequencer started this v1 launcher; its 22-minute GPU cap
expired during ON fused-child compilation. Its E41 subprocess initially missed
`PYTHONPATH`; CPU recovery resolved the import and measured the retained cache.

## Design B: production unchanged, history-only twin

The rejected return extension changed EL/QSQ/cloud state at the ulp level.
Commit `ce8efefd42` reverted it. The replacement leaves production kernel,
plume, adapter and return code unchanged. A private twin consumes the exact
state/surface/selected-parameter operands also supplied to production. Only
the original production result advances the state; twin values are history
only. An output-only barrier sufficed in the selected-column CPU program.
The private plume projects the current production wrappers/recurrences instead
of copying a stale coefficient stack. Surface shear uses the actual predictor
production to retain selected stability parameters. Tuned composition still
requires its own qualification.

The final CPU artifact checks every carried MYNN leaf at steps 1 and 2 on 32
real WN3 d03 columns per day/night set. Both cases pass 50/50 BYTE. This is not
a full-grid/coupled GPU claim. Night input includes nonzero resolved ice/snow.
Term references use the original public driver, including warm QSQ/cloud input
fields on step 2; this is same-input callee validation, not a CPU-WRF forecast.

The manager amended only the near-equilibrium Dtke comparison, before
re-evaluation: 25 original members (nominal plus 24 fixed one-ulp patterns),
on the actual input columns at each step. For column c,
`floor[c] = max_k(max_member Dtke[c,k] - min_member Dtke[c,k])`.
Every cell must satisfy `abs(candidate-ref) <= max(.05*abs(ref), floor[c])`.
Every column is reported; above-bound values fail. The other term bounds and
all carried-byte requirements stay unchanged.

The six raw PBL rates (critic cr-w44-budget F1) originally had an absolute
`1e-3` bound, which was vacuous: RQVBLTEN ×1.5, RQVBLTEN = 0 and an RQCBLTEN
sign flip all passed. The rule pre-registered at `49a3636970` uses the same
relative bound as the other terms, 5 % of the reference column maximum. It
pairs that bound with a per-column floor from the same 25-member ensemble,
`floor[c] = max_k(max_member R[c,k] - min_member R[c,k])`. RTHBLTEN needs this
floor in weak-tendency columns, where REAL total differencing leaves 1–2
ulp(θ)/dt. The P75 re-score of the four admitted runs is in
[receipts](receipts/P75_F1_RESCORE.json). Every original reference reproduces
the earlier receipts byte for byte, and all 168 term/step checks pass. At its
worst column RTHBLTEN reaches 0.667 of the bound and the other PBL rates
≤ 0.078. The three former survivors and five positive controls are each
killed on all 8 case-steps (64/64, none vacuous).

Amendment (cr-w44-budget, 00:37Z, committed `7a24833cd2` before any production
GPU array): RQVBLTEN, RQCBLTEN and RQIBLTEN also get a per-level quantization
floor, `4·spacing_f32(max(|q_in|, |q_in + r_ref·dt|))/dt`, which is zero below
2^-126. The GPU mini showed exactly one local ulp(qi)/dt at one level. The P77
re-score of the four CPU runs and the GPU mini still passes every term and
reproduces all references byte for byte. The floor binds at 32 RQV and 348 RQI
points. A 10-ulp injection is killed at all of them; the mini's RQI margin is
0.25 of the bound. [Receipt](receipts/P77_AMENDMENT_RESCORE.json). ENT kts is initialized to 0.001
in pristine WRF :5806; deleting that initializer fails in 25 active columns
(one genuine numeric assertion kill, zero setup errors).

[Final CPU receipts](receipts/FINAL_CPU_GATE.json) index all four component
and original-oracle runs. Each oracle receipt reports every column at both
steps, including the independently measured floor. The three previously
failing day columns at step 1 are:

| Column (zero based) | max absolute Dtke error | reference profile max | original floor | Amended gate |
| --- | ---: | ---: | ---: | --- |
| 28 | 3.87430e-7 | 7.47542e-6 | 3.47694e-6 | PASS |
| 29 | 4.56969e-7 | 3.52661e-6 | 3.03984e-6 | PASS |
| 30 | 3.77496e-7 | 3.42478e-6 | 2.17557e-6 | PASS |

These floors come only from the pristine ensemble, with no candidate-derived
margin. Across the four runs, all 168 term/step comparisons pass (P75, with
the corrected PBL rule). The maximum
absolute difference between candidate and original closure residuals is
1.31004e-7. The physical carry passes 200/200 field comparisons in total.

Measured cost below is for the CPU component running two MYNN calls on 32 by
44 columns. It is not a forecast or GPU cost estimate. Output bytes grow
237,968 to 474,848 in each case because both steps retain all history fields.

| Case | EDMF | OFF median ms | ON median ms | OFF temporary B | ON temporary B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Day | 1 | 3.61235 | 4.72664 | 1,011,640 | 1,668,808 |
| Night | 1 | 3.98067 | 5.30350 | 1,011,640 | 1,668,808 |
| Day | 0 | 2.12073 | 2.18172 | 819,960 | 727,416 |
| Night | 0 | 2.04855 | 2.18256 | 819,960 | 727,416 |

EDMF-on CPU compile plus first call was 17.90–19.31 seconds OFF and
39.46–40.31 seconds ON. GPU compile, memory, extra kernels and s/fc-h remain
unmeasured; this opt-in twin incurs another MYNN pass and history allocation.

## Composed program and next night control

Immutable v2: `<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_launch_v2.sh`, SHA256
`ef94812982d60f2dde5631143c768f220cd39dd15df8d789fa93f78328f93299`,
cap **60 minutes**. [CPU controls](receipts/NIGHT_V2_CPU_CHECKS.json).
It preserves the exact paid-cache paths, primes OFF and ON through actual
root cascades before replaying both 3 h runs, requests rich AOT metadata and
sets the pinned CPU `PYTHONPATH` for E41. It is still pinned to the original
qWT prototype `3fa48d1597`; it does **not** qualify the new design-B twin.
That v2 is superseded for the feature. Do not queue it as design-B qualification.

The current model pin is `42f60b74854a7f31ec8cff608ee82c38d057d41a`, at
`<USER_HOME>/wrf_gpu2_lanes/w44-budget/snapshot_p49`. It includes raddiag from
main, separate budget history and the legacy-EDMF conflict refusal. Public
caller regression passes 146/146 physical State/source leaves over two steps.
The composed OFF identities remain exact: [Swiss](receipts/P46_OFF_SWISS.json)
and [PROD](receipts/P47_OFF_PROD.json). Real WN3 d01 two-step CPU lowering
accepts all 21 resident history leaves and a distinct static key, with the twin
visible in debug IR. This is a type/dataflow check, not execution or GPU BYTE.

The immutable replacement is
`<USER_HOME>/wrf_gpu2_lanes/w44-budget/day_launch_v5.sh`, SHA256
`87b987aa871f538155f949ec7cfa5fff6b43c3b2547dcd918e8f8e40e8737e28`,
cap **75 minutes**: GPU setup/replays up to 60, CPU E41/oracles/accounting up
to 12, plus margin. It pins the product child to `JAX_PLATFORMS=cuda`, uses
one fixed cache for all programs, primes OFF/ON before replaying both 3 h
three-nest runs, and adds EDMF-off qualification. Standalone real day/night
two-step components run in the same GPU process. E41 also discovers their
actual native exports; its thresholds are unchanged. Pristine term checks
remeasure each 25-member floor at the actual GPU input states after unlock.

The exact sealed-shell CPU launch smoke passed: the child CUDA environment was
recorded before device-only CPU substitution, and the actual MYNN caller plus
twin executed on a frozen WN3 44-level column, yielding 21 finite fields in
16.627 seconds. Admission used the real clock helper with a simulated 11:30Z
day window, preserving the admission function. The scratch directory was checked
to contain the unique run name. No GPU/official lock was used. The outer launcher intentionally
returns incomplete for GPU/full-forecast qualification in that CPU control.
External top-group cancellation through the actual RAM scope also reaps the
worker and escaped TERM-ignoring sampler and releases the dummy flock.

Only the manager sequencer starts GPU work. GA timestamps govern admission and
the minimum remaining 75-minute cap; there is no calendar/date guard. This
entry works for the granted day slot and the night slot. V3 CUDA and V4 were
withdrawn: repeated smoke found that `systemd-run` expands `${RUN##*/}` as an
invalid environment variable, yielding an empty run suffix. V5 uses `basename`
and verifies the unique path. The new GPU cost remains
unmeasured; the cap uses measured predecessor phases plus conservative headroom
for rich AOT exports and the additional variant. The user-facing option text
is in [docs/mynn-budgets.md](../../docs/mynn-budgets.md); README wording belongs
to w44-docs and its feature-list proposal is in MAILBOX.
