# u2-optim CPU qualification

Source candidate: `329b4d525ae79c6ec06769ce1c57d936eb4ca83f`.
Measured detached candidate: `eeabf575d75bd1c1f7a1cce739719a2ad07c8bc1`.
The rebase added only the manager's unrelated lane contract. `src` and `tests`
tree objects are byte-identical; numerical validation was not repeated.
Full receipts/hashes: [VALIDATION.json](VALIDATION.json).

* [M CPU] 420 unique composed checks pass: own 37 tests, ensemble/station
  consumers and all nine mandatory shared modules. Initial sparse fixture skips
  were closed by running just those two checks with their tracked assets.
* [M CPU] Clean `git archive`: all 37 feature tests pass, including the recorded
  original CPU-WRF surface fixture, actual ensemble preparation/fake scheduler,
  CLI calls, chronological calibration and nonfinite refusals.
* [M CPU] Two committed mutants each produce one genuine assertion failure and
  zero setup errors: future-truth leakage in calibration; reversed physics loss
  ranking. The candidates without those mutations pass the same assertions.
* [M source] Twelve model/default-source tree entries are identical to main.
  Bootstrap change only adds two offline command names; `run` keeps its prior
  bootstrap condition and parsed arguments. Existing Swiss release jaxpr
  `2bf12cb8` is carried forward by this exact-source proof, not newly traced.
* [M CPU instrumental] Interleaved parser construction, n=80: baseline
  5.063 ms, candidate 5.587 ms; tracemalloc peaks 172795 / 215950 bytes. This
  is CPU parser overhead under shared-host conditions, not forecast timing.
  No new model launches, device temporaries, transfers or compilation.

The station objectives, normalization, support, date-block interval and MOS
formula are documented in [regional optimization](../../docs/regional_optimization.md).
The retained 1831-byte NPZ is an unmodified 2x2 crop from original CPU-WRF Swiss
surface output, with full source-file hashes. It qualifies reader integration;
synthetic station observations qualify formulas/selection logic. Neither is
evidence of regional forecast skill. Tests' GPU admission receipts are explicitly
synthetic; they do not qualify any real alternative physics configuration.

The small real GPU demonstration remains **OPEN**: this lane contract prohibits
GPU execution here. After independent CPU review, the manager must provide
source-bound GPU admissions, four genuine case dates and an approved local
station/observation set, then schedule the existing locked `optimize run` workflow.
An example two-configuration, two-train/two-test, 3-hour study budgets 18
case-hours (six forecasts); no observations or cases are invented for that gate.
This handoff is CPU-qualified readiness for a critic, not full final graduation.

Caches disabled (`GPUWRF_JAX_CACHE=0`), all CPU work pinned 8,24 nice19 with
`JAX_PLATFORMS=cpu`, QUIET guards and explicit timeouts. Sparse detached snapshot,
clean archive and pytest scratch were removed after their last dependent check.
Original CPU-WRF references are unchanged; small receipts remain in
`<USER_HOME>/wrf_gpu2_lanes/u2-optim/`.

## Critic F1/F2 correction

The initial receipts above remain bound to their original sources. The critic
found that a station with only the first three of six dates could be exported
without a held-out score. Fitting now refuses the entire model if any exported
station/variable lacks chronological held-out support; application also refuses
unqualified groups in legacy saved models. Bias and MOS regressions cover both
paths and keep qualified legacy stations usable. The serializer whitespace is
removed; its AST is unchanged. See [F1_FIX.json](F1_FIX.json).

[M CPU] The targeted correction gate passes 22 calibration/nonfinite/NetCDF-lock
checks, including the existing no-future-leakage test. Disabling the new support
check produces four genuine expected-exception assertion failures, zero setup
errors. The complete branch passes `git diff --check main...HEAD`. Fourteen
model/default/CLI source entries are byte-identical to the reviewed candidate;
the prior numerical receipts are reused only for unchanged work. The group
sets add offline CPU work; no model device allocations, launches, transfers or
compilation are added. Snapshot/pytest scratch are cleaned; GPU demo stays open.
