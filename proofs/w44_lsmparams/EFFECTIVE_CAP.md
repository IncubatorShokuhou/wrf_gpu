# Shared effective-SMCMAX admission

Manager authorization: `2026-10-24T00:17Z` current-thread approval of proposal
`b5c20c8981bf4111367eb314c5d36463d4e66525d4bede264a024b6735ee9672`.
Owner w44-override informed; the W2 critic reviews this delta with the family.

`validate_fields` and `validate_wrfinput` accept optional `effective_smcmax`.
Default `None` preserves the original table lookup, returned reports and
refusals. An explicit cap uses the same shared `validate_soil_porosity` helper
and runs even when the changed-field set has no soil field. This permits joint
state/parameter edits to use the actual runtime porosity. The original glacier
ISICE & XICE<=0 cap=1 and water exclusions remain in that unchanged helper.

Measured CPU evidence, immutable `3407c6a6b0` / `6040d4b79b`:

- `tests/w44_override` plus explicit-cap tests:188PASS,1.52s.
- Paired against main `c4a829ece4`:24 public-API return/refusal reports are
  byte-identical with the new argument omitted or explicitly `None`.
- Ignoring the cap is killed by2 genuine assertions,0setup errors: retained
  soil above a lowered cap and a valid increased-cap/soil-edit case.
- NaN/±Inf, invalid caps, glacier cap1, positive sea ice and water exclusions
  are covered by independent known answers.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams/LS31_cap_suite.*`,
`LS33_checker_identity/report.json`, `LS36_ignore_cap_mutant.*`.
These are host admission measurements, not forecast fidelity or GPU timing.
