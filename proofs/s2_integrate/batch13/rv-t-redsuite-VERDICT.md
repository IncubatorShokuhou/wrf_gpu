# MERGE — rv-t-redsuite independent review

[M CPU/source] All ten assigned test-only commits pass the registered critic gate.
183 passed executions, 5 explicit skips, 3 unchanged inherited xfails; 2 product mutants and 1 present-wrong external input fail by genuine assertions, with zero setup errors.

| Commit | Verdict | Independent evidence |
|---|---|---|
| `de2e7089e` | MERGE | FAST0/1 each 4 PASS; land-value mutant AssertionError |
| `9a958cb46` | MERGE | 6 PASS with real reload predecessor and pristine C/D/deletion checks |
| `984f5070c` | MERGE | 8 PASS / 4 explicit SKIP; present-wrong path AssertionError, never SKIP |
| `1c5561af4` | MERGE | 29 PASS; original lifecycle assertions unchanged |
| `873cac346` | MERGE | 12 PASS; exact current admission/dispatch contracts |
| `50873bfc5` | MERGE | 12 PASS / 1 inherited SKIP; omitted-T2-write mutant AssertionError |
| `1b6cf9f93` | MERGE | 18 PASS on original WN3 operands; all byte/order assertions unchanged |
| `6122c751b` | MERGE | 8 PASS; pristine CPS rates/ULP/byte assertions unchanged |
| `b0d618512` | MERGE | 45 PASS / 3 inherited XFAIL; default dtype/device gates also PASS after analytics |
| `d185e9dce` | MERGE | 32 strict analytic PASS + 3 independent restoration PASS + 2 default-consumer PASS |

The analytic fp64 fixture is opt-in. Independent same-process checks prove REAL32 constructors before and after it, fp64 only while requested, exact classmethod restoration, and restoration of the prior JAX x64 setting. The unmodified-default dtype/device tests also pass afterward.

All 15 changed files are tests; source diffs and removed/changed numeric tolerance keywords are zero. Manual assertion review is in STATIC_REVIEW.md; original proof SHA and every committed test-file byte hash were verified. The lone new skip is only for the absent un-vendored original WRF path; present-but-wrong reaches an assertion.

Snapshot `37c4cf8c46c7129184f1af3df23bb0063f95a008` = base `09fce2d4f` plus exactly the ten reviewed repairs. R01 was carried forward by exact production and selected-test byte identity. Every arm used an independent 6GiB/no-swap RAM slot and fresh CPU process, private caches, timeout and QUIET guard. Peak measured RSS: 2937432 KiB (about 2.80 GiB); no swap reported.

Production runtime/performance/memory are unchanged by these test-only patches. No GPU, forecast-fidelity or forecast-speed qualification is claimed. The producer full-suite gate remains outside this verdict.

R02Q was an unstarted old-version queue waiter, canceled and resubmitted into FIFO; zero numerical work was interrupted. Finished caches, pytest temp trees and the detached snapshot are cleaned after their last consumer.

Full receipts and hashes: REVIEW.json and receipts/.

R09.log/xml are stored as lossless gzip; REVIEW.json binds both original and compressed SHA256. Original raw receipts remain in the lane artifact directory.
