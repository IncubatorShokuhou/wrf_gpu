# MYNN fresh-start QSQ and EL seed (w44-qsqseed)

Status: **READY-FOR-CRITIC (CPU scope, under the manager rulings below).** Source frozen at `dfce74ff96` on source base
`700a382f8b`; later branch commits contain tests/proofs only. No GPU arm in this
lane; the composed RC's RW gates remain the GPU qualification.

## Change and scope

Fresh, non-restart/non-cycling MYNN first calls now request the existing
`mynn_coldstart_qke_from_state(return_moments=True)` result and store QSQ and EL
alongside QKE. These are the level-2 equilibrium values from pristine WRF's
`mym_initialize`, not zero seeds or tuned clamps. The only model source delta is
the fresh block of `_mynn_state_with_first_call_qke` in
`src/gpuwrf/coupling/physics_couplers.py`. The initializer itself and cycling
block are source-exact against the frozen base; see `receipts/source_scope.json`.

Manager scope ruling: closure 2.6 carries QSQ; TSQ/COV are per-call diagnostics
and have no State leaves on this base. No State schema or optional dead path was
added. Restart and later-step lifecycle authority is retained.

## Original-WRF evidence [M, CPU]

The linked pristine REAL4 `mym_initialize` operator witness uses 32 real day/night
and Swiss ice/snow columns, 44 levels, dx 3000/9000, caller SQV and independent
Exner, separate work arrays, mixlength=1, scaleaware=1, spp=0. RMOL=0 is the
retained explicit qualification of WRF's undefined first-call local. Fixture
provenance and object/source hashes are under `tests/data/w44_qsqseed/`.

Existing frozen bounds: QKE median relative error <=1e-4; EL max error bound
.001+.002*abs(WRF); QSQ max error bound 2e-12+.002*abs(WRF).
The unchanged return-moments helper meets those bounds. The initial diagnostic
extension also measured TSQ/COV, but that extension was removed on the manager's
scope ruling; those fields are not runtime validation claims for this branch.

The full two-step runs use the real production loader and native kernels in
CPU Pallas interpret mode. Both wrfinput files are byte-exact to their original
CPU-WRF twins (`receipts/twin_inputs_match.sha256`, to be sealed). Comparisons
use the frozen interior ring >=5. These are CPU fidelity measurements, not GPU
performance or qualification claims.

| Case / field | Step 1 relative RMS | Step 2 relative RMS | Formal gate |
|---|---:|---:|---|
| Swiss QSQ | .0002465 | .0004383 | PASS |
| Swiss QC_BL | .0100278 | .0137064 | **MISSED .01 bound** |
| Swiss QI_BL | .0016824 | .0021831 | PASS |
| Swiss CLDFRA_BL | .0036069 | .0038907 | PASS |
| Swiss QKE control | .0005962 | .0010996 | PASS |
| PROD QSQ | .0009707 | .0011164 | PASS |
| PROD QC_BL / QI_BL / CLDFRA_BL | 0 | 0 | exact zeros in both, cloud check vacuous |
| PROD QKE control | .0001432 | .0002591 | PASS |

All recorded fields are finite. Swiss supplies active liquid/ice/cloud evidence;
PROD supplies nonzero variance and QKE evidence. The full run JSON retains RMS,
signed sums, max errors, and EL diagnostic differences.

## Manager causal ruling and independent residual

The original 1% QC bound is unchanged and formally missed on Swiss. The manager
approved acceptance of the seed fix through the existing causal intervention:
a port run with original CPU-WRF step-1 QSQ injected before step 2. The fixed port
must agree with that intervention within its own spread, per field. Our explicit
spread check is RMS(fixed-control)<=RMS(control-originalCPUWRF), normalized by the
same CPU RMS, for QSQ/QC_BL/QI_BL/CLDFRA_BL/QKE at step 2. Step 1 is before the
injection and is retained as deletion sensitivity evidence.

Swiss passes all five checks against the retained o5-n333 default injection
receipt. Fixed-control errors are .00027877/.00015533/.00008080/.00014644/.00012910;
control-CPU errors are .00030375/.01362858/.00217253/.00388960/.00110412 in the same
field order. PROD reproduces the same method with the exact frozen-base lifecycle and
original PROD restart QSQ. Four fields meet that strict spread check (the three
cloud fields are zero in both arms); QSQ formally **MISSES**: fixed-control
.00114749 versus control-CPU .00030932. Fixed-CPU QSQ is .00111642, inside the
frozen .002 init class.

Manager 09:44 ruling accepts PROD against that frozen init class, retaining the
strict spread miss. The old injection control has a different QKE seed: the
existing `return_moments=True` helper uses WRF init's resolved-only THLV, whereas
the old `False` path uses ordinary SGS-aware THLV that adds snow to ice. Pristine
driver :675-684 excludes snow before `mym_initialize`. On PROD, fixed QKE error
is .0001432/.0002591 versus the injection control's .0033531/.0033479; the control
is therefore not an equivalent QKE/variance reference on this case. This
snow-sensitive QKE correction is part of the identity change and is quantified
separately by the pristine snow witness and the real-grid init census.

FINDING [M]: Swiss step-2 QC_BL residual is .0137064 with the fix versus .0136286
with original-WRF QSQ injected; their difference is .0001553. [I] The remaining
SGS-PDF2 discrepancy likely reflects small mean-state/caller differences in
theta, vapor, hydrostatic pressure or independent Exner around saturation rather
than variance initialization. This is a separate finding in the main INBOX;
future original-operand transplants at condensation can localize it. No residual
fix or tolerance change is included here.

## Program and lifecycle scope [M]

Schema-2, native-first comparisons bypass **only** MYNN first-call initialization
in both trees. Everything else is exact, including full constant values,
canonical program/constants, every native Pallas body and its source locations:

- Swiss root: **IDENTICAL**, all 10 checks.
- PROD root, product parent-to-child force-down and forced child step:
  **IDENTICAL**, all 22 checks.
- All probes are COMPLETE, with zero interpreted Pallas calls in native CUDA
  lowering. These are deviceless CUDA-target program proofs, not GPU executions.

The real-grid init census changes only `qke`, `qsq`, `el_pbl`. All other leaves
retain their values, shape and dtype; State replacement passes them through
without arithmetic. PROD uses literal byte comparisons; Swiss uses exact numeric
comparison plus that unchanged passthrough. A later optional Swiss byte-check
strengthening did not run because the GPU-window flag prohibited new RAM tickets;
it is not a changed numerical model and no completed physics validation was
repeated. Static later-step/restart calls return the original State, compiled
later-step output is unchanged, and full-constant **cycling HLO is exact** on both
grids. No State schema, IO, restart format or literal leaf-count pin changed.

Tests: **124 PASS in FAST_DEFAULTS=0 and 124 PASS in FAST_DEFAULTS=1**. The REAL4
operator oracle uses the same fixed release/no-FMA qualification condition as the
existing F3a witness; lifecycle/forwarding/carry dtype tests exercise both parent
modes. The 19 initial tmp-path setup errors were retried after creating the lane
parent directory; the 105 successful tests were reused. The new oracle's initial
legacy-mode condition mismatch is preserved as a diagnostic, without changing
bounds or model source. See `receipts/tests_summary.json`.

## QKE identity change and snow mechanism [M]

The original-WRF snow witness has 12 active ice/snow columns out of 32. For those
columns, old helper QKE relative RMS error is **.241289**, fixed error
**4.15725e-7**, maximum fixed absolute error **1.57952e-6 m²/s²**. Setting snow to
zero only in the old helper makes its QKE **byte-identical** to the fixed helper
on all 32 columns. This isolates the resolved-only THLV difference and proves
that the newly selected helper follows pristine initialization, including snow.

Both *loaded cold benchmark states* have zero snow. Their direct QKE seed changes
are only rounding: Swiss max 4.76837e-7 (98 values, relative RMS 2.10e-9); PROD max
2.38419e-7 (988 values, relative RMS 8.46e-9). Therefore the measured full-step QKE
improvement cannot be attributed exclusively to snow on these cases. EL seeding
also changes the first call, and the release-default `GPUWRF_MYNN_TKEPROD_UP=1`
reads previous EL. The injection control does not hold those initialization
changes fixed. The snow mechanism is independently proven by the structured
oracle, and all of QKE/EL/QSQ are disclosed as part of this identity change.

## Cost and memory [M, CPU component only]

Matched old-lifecycle/fixed-lifecycle JIT measurements use real loaded State and
Grid, release defaults, seven warm samples, CPU cores 24/25/28, nice 19 and a
12 GiB no-swap scope. These are init-component measurements; no GPU s/fc-h or
whole-forecast speed claim follows from them.

| Grid | Warm init old → fixed | Temporary storage old → fixed | Compile old → fixed |
|---|---:|---:|---:|
| Swiss 44×42×42 | 26.966 → 24.262 ms | 9,656,744 → 10,284,776 B (+0.599 MiB) | 1.531 → 1.428 s |
| PROD 44×70×120 | 283.844 → 277.687 ms | 45,968,424 → 48,958,824 B (+2.852 MiB) | 2.257 → 1.235 s |

Argument and output byte sizes are unchanged in both arms. Peak CPU RSS for the
paired init probes is 2.45/3.57 GiB. Treat the small warm/compile timing differences
as measurements on a shared host, not a speedup claim. The later-step CPU wrapper
microcheck is 1.176 → 1.128 ms (Swiss), 4.683 → 5.874 ms (PROD); this includes
whole-State CPU return/copy/dispatch and is not a GPU overhead measurement. Actual
GPU kernel/traffic, VRAM, readiness and forecast timing remain binding RC RW gates.
The new work consists of retaining existing EL and computing/retaining QSQ in the
existing initialization pass, with no new State leaves or host data transfer.

## Review, artifacts and limits

- Model source: `dfce74ff96503476bc5a85d9e11778e9872e80bc`, src tree
  `8c899e7d6c18d338b141244ef142abce878e62da`; only one fresh-start block differs
  from frozen source base `700a382f8b`.
- `receipts/MANIFEST.json` seals the numerical, native-program, cost and input
  receipts. Original fixture source/object/wrapper hashes are in
  `tests/data/w44_qsqseed/provenance.json`; generate with `init_oracle.py`.
- Raw compact step arrays and all commands remain under
  `<USER_HOME>/wrf_gpu2_lanes/w44-qsqseed/`; shared original-WRF twins remain
  under `<DATA_ROOT>/wrf_gpu2/scratch/o5-n333/` and are preserved.
- All experiments, including failed harness setups and formal misses, are logged
  to the main EXPERIMENTS TSV; separate residual finding is in main INBOX.
- Own pytest temporaries and measured snapshot are cleaned after their last
  dependency. No GPU arm was requested or run. The manager composes this branch
  into the RC and owns GPU RW1–RW4 / full-duration release qualification.

The QC 1% formal miss and PROD strict own-spread miss are **not relabeled PASS**.
Acceptance is under the manager's explicit causal-floor and frozen-init-class
rulings. The residual is separate; this lane makes no further physics fix.
