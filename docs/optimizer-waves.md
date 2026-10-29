# Optimizer waves and external review

`gpuwrf optimize run --study study.json --output run` stops after every EKI
iteration or discrete selection rung. Exit status **3** means PAUSED. The
optimizer writes `run/waves/wave-NNNN.json`, `wave_state.json`, and
`progress.json`; it does not launch the next wave automatically. Chunk-budget
pauses within a wave remain restartable without a new approval.

Each wave report contains the evaluated ensemble, member scores, impact matrix,
admissibility, dispatched case-hours, member wall times and measured RSS/VRAM
where available. Missing reference/impact assessments explicitly say
`NOT_EVALUATED`; they never imply superiority to WN3 or the control. The ALISIOS
FORMAT_ONLY examples are parser fixtures, never scientific truth. Cloud and
reference scoring obey their own admitted-data contracts.

After reviewing the report, **alisios_mgr or a review model** writes
`run/waves/wave-NNNN.GO.json` with exactly these fields:

```json
{
  "schema": "gpuwrf.optimize.wave-go.v1",
  "decision": "GO",
  "run_id": "COPY wave_state.json run_id",
  "study_sha256": "COPY wave_state.json study_sha256",
  "wave": 0,
  "report_sha256": "COPY wave_state.json pending.report_sha256",
  "checkpoint_sha256": "COPY wave_state.json pending.checkpoint_sha256",
  "approved_by": "alisios_mgr"
}
```

The wave number must be the integer `pending.wave`. `approved_by` also accepts
`review:<name>` (letters, digits, dots, underscores and hyphens, maximum 64
characters). A GO is a review assertion, not a cryptographic signature: restrict
filesystem write permission to the operator/reviewer. The optimizer never writes
approvals. A symlink, missing/invalid/oversized file, different run/study/wave,
changed report/checkpoint, unknown field or unrecognized reviewer refuses
continuation before any model launch.

`decision` can instead be `STOP`, which records a durable stop; later resumes
launch nothing. `gpuwrf optimize best --output run` always retrieves the incumbent
or the explicit `NO_ADMISSIBLE_SOLUTION` status. Selection never uses scalar
loss alone. To attach the scoring authority's assessment, write
`waves/wave-NNNN.review.json` with `schema=gpuwrf.optimize.wave-assessment.v1`,
the exact `run_id`, integer `wave`, `report_sha256`, and `assessment` containing
the unmodified `gpuwrf.optimize.multiobjective.v1` result. Add
`assessment_sha256` to the GO/STOP token: it is SHA256 of that JSON serialized
with sorted keys, compact separators and no NaN (the same canonical digest as
the wave report). The review file must be regular, at most 8 MiB.

Only finite `scores[].loss` with `scores[].admissible=true` can update
`best.json`. HARM_VETO, missing truth and insufficient dates never win. The
authority's `objective_identity` freezes truth, WN3/control pairs and the scoring
denominator; changing it refuses comparison with an earlier incumbent. Review
sidecars preserve the impact matrix, tests and uncertainty without rewriting
the report that was approved. No authority assessment means no admitted winner.

Then run `gpuwrf optimize run --study study.json --output run --resume`.
Checkpoint/ledger writes fsync the file and parent directory. Consumption is
committed durably before continuing. An old GO cannot approve
a later wave. If interrupted during an approved wave, `--resume` continues that
same wave without consuming a second token. EKI checkpoints preserve float64
ensemble values, iteration/history, observation keys and the deterministic
per-iteration RNG seed schedule. Completed forecast receipts are reused by
content; failed attempts remain charged to the compute budget. The final EKI
iteration also pauses for review before prior/posterior held-out evaluation.

Continuous runs also bind the EKI implementation source and NumPy version.
An unrecorded or changed implementation refuses resume before any model work;
use the original implementation or a new output directory. In particular, the
reviewed ensemble-space solver has a different seeded Gaussian root from the
legacy dense solver, while replay within the same implementation stays exact.

`--wall-seconds` defaults to **1800 seconds per invocation**, including study
preparation, scoring and failed attempts. A host timer interrupts preparation
or scoring; the scheduler deadline stops new members and terminates active
process groups, with its bounded teardown grace additional. `--max-retries`
defaults to **1** additional attempt per candidate/date and is immutable across
resume. Every attempt remains charged to the case-hour budget. The ledger keeps
total wall time across invocations plus preparation, model/compile and
extraction/QC times; wave reports disclose residual budgets and failed batches.
The incumbent survives exhausted budgets and crashes. Concurrent invocations
of one output directory are refused before consuming approval or starting work.

`--profile-points points.json` requests offline profile extraction from held
histories after scoring, using the qualified profile reader. The file is a list
of `{id, region, domain, j, i}` points (`region`: `north_windward` or `sea`;
zero-based nonnegative grid indices). The observer is fixed across resume.
Per-member reports in `diagnostics/` are SHA-bound in the wave report; unavailable
nests or terms remain explicit. This option never enables model diagnostics or
adds model taps. With no points, profiles report `UNAVAILABLE`.

Every actual model-member attempt, including retries, registers through
`scripts/alisios_registry.py` before process creation and ends in scheduler
cleanup. Start failure prevents the model launch; terminal failure prevents a
success claim. The child receives its exact `ALISIOS_WRF_RUN_ID` and atlas path.
SIGKILL/power loss cannot run finally: reconcile the original invocation from
receipts using the registry's reviewed backfill, never invent a second start.
If model output is complete but the terminal registry append failed, the receipt
marks `registry_complete=false`: optimizer resume refuses to rerun or score it
until the original invocation and receipt are reconciled.

Use `--registry registry.json` to set `forcing`, `atlas_dir`, `smoke`, `by` and
`producer`. For example, a test uses
`{"forcing":"OTHER","atlas_dir":"/private/test-atlas","smoke":true}`.
Tests must never use production ATLAS files. With no policy the forcing is
explicitly `OTHER`; set the real forcing family for operational runs. Registry
policy is immutable across resume. Successful runs retain their histories for
scoring and register **DONE_HELD**, `raw_deleted=false`; crashes register FAILED,
interruptions ABORTED. DONE_CLEAN is valid only after receipted raw cleanup.
`--bundle` refuses older bundles without the per-member hook capability;
rebuild with v0.4.5 or newer to enable optimizer registry support.

Wave/registry bookkeeping is host-only and outside timesteps. It adds no model
kernel, device transfer, JAX backend or model subprocess. Costs are small JSON
checkpoints and two durable registry appends per member; numerical forecast
programs and their memory requirements are unchanged. GPU speed remains
unmeasured for this orchestration-only change.

Until an ensemble-space EKI build is present, productive full-native cloud pixel
vectors are refused before model work; the scorer must declare the admitted
`observation_operator=superobservations`. A second guard refuses more than 4096
observations before dense covariance allocation. The cloud scorer owns the
correlation-aware operator and its ALISIOS contract. This controller never
silently thins observations or changes the scoring denominator.
