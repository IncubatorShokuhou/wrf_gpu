# Cloud controls CPU gate

`VALIDATION.json` binds the measured source, gates, exact fixture supplements,
source reuse and retained raw XML/log SHA256s. All 65 lane tests and all 271
requested consumer nodes passed; the consumer count is deduplicated across
bounded runs after restoring omitted baseline fixtures. This is optimizer
plumbing and analytic EKI evidence, not WRF/GPU/regional-skill qualification.

The control/null branch holds every selected multiplier at 1 with the families
enabled. It does not establish ON/OFF physical identity. Existing binder/kernel
source is preserved; MYNN's existing experimental limits remain.

The semantic mutant plugins perform in-memory source interventions only. From
an immutable checkout, with a private output directory whose parent exists:

```bash
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=proofs/v45_cloudknobs:src CLOUD_MUTANT=M1 \
  python -m pytest 'tests/w45_cloudknobs/test_cloud_controls.py::test_unavailable_matrix[thompson_parameters.av_i]' \
  -p cloud_mutations --basetemp <USER_HOME>/wrf_gpu2_lanes/REVIEW/pytest -q --tb=short
```

Use the target recorded for each M1–M12 and M14 in `VALIDATION.json`. M13 uses
`-p cloud_mutations_e224` on the CPU key/full-constant test and deletes the
proof's full-constant rendering. A kill requires rc=1, an actual assertion
failure (`AssertionError`, `DID NOT RAISE`, or pytest's rewritten `E assert`),
and zero setup errors/skips. The retained M4 setup error is explicitly excluded.
Run CPU checks on the manager-assigned cores in an 8 GiB/no-swap scope and obey
the project's QUIET guard. No GPU, live ALISIOS roots or production registry is
needed by these tests. Native pixel memory policy is tested before case reads.

After rebasing on main `ac48a5285037`, one invocation passed 196 tests (65 lane,
40 continuous callers, 87 ensemble-space algebra, four CLI). The changed guard
mutants were rerun; M14 rejects interpreting memory capacity as correlation
admission. Native vectors remain refused without the ALISIOS contract, and
superobservations also require its admitted length. `VALIDATION_PRE_ENSEMBLE.json`
preserves the earlier gate; `ENSEMBLE_REBASE_REUSE.json` explains source reuse.

The wave/cloudscore integration rebase is sealed separately in
`rebase/RESULTS.json` and `rebase/RESOLUTION_HUNKS.json`. Base main
`25d959e6b7` contains separately reviewed cloudscore `51a472858e`;
`rebase/OWN_COMMITS.txt` lists only this lane's commits above that base.
The source pin is `4bdab52979`: 295 distinct touching tests passed across
bounded runs (68 owned), one unchanged optional archived-WRF smoke skipped,
and 21,769 tests collected with zero errors. The initial touching run's one
stale cloud count failure is retained and superseded by the final seven-pass
run. Two changed-behavior deletion checks produce three genuine assertions,
zero errors/skips. Scheduler/collector/checkpoint AST preservation and exact
source carry explain reuse; this does not add a WRF or GPU qualification claim.
