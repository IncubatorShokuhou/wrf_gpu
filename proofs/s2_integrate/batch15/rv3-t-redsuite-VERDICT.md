# MERGE — rv3-t-redsuite independent review

All 12 assigned test-only commits pass the critic gate on immutable c62e5dd78.
[M CPU/source] 298 selected PASS / 42 explicit external-input SKIP; three
additional absence-boundary controls PASS; three product mutants fail by genuine
assertions with zero setup errors.

| Commit | Verdict | Independent outcomes |
|---|---|---|
| `ab3bc39a6` | MERGE | {'passed': 38} |
| `49c103b2b` | MERGE | {'passed': 6} |
| `ef856db9f` | MERGE | {'passed': 3} |
| `1266d1800` | MERGE | {'passed': 11} |
| `8fda963e1` | MERGE | {'passed': 14, 'skipped': 3} |
| `01ddc0b39` | MERGE | {'passed': 20} |
| `978f96726` | MERGE | {'passed': 61, 'skipped': 3} |
| `983080476` | MERGE | {'passed': 40, 'skipped': 29} |
| `6fbff92e3` | MERGE | {'passed': 30, 'skipped': 4} |
| `214f687f8` | MERGE | {'passed': 41, 'skipped': 2} |
| `8fdb7ab18` | MERGE | {'passed': 28, 'skipped': 1} |
| `a808e7323` | MERGE | {'passed': 6} |

Every selected skip was checked individually: all named inputs are absent.
38 newly guarded cases use missing historical namelist/R8 inputs; four skips
were inherited. Existing incorrect inputs are never skipped. Original numeric
tolerances and all numeric assertions remain unchanged; exact command-buffer
policy and byte-exact O3 restart replace stale contract expectations.

Independent product faults remove WHILE capture, corrupt serialized O3RAD,
and freeze the advanced Noah-MP clock. Each repaired test rejects its fault.
The initial command/O3 controls had harness defects and are excluded; R03b
corrects only those two, retaining the valid R03 clock assertion without rerun.
Raw attempts and explicit dispositions are preserved in MUTANT_DISPOSITION.json.
Temporary Git authority and shape tables restore exactly. Historical H3 uses
real private FINAL-b ancestry; it does not qualify the current main lineage.

No production code/assets changed: no additional model kernels, transfers,
allocations, compile/readiness work, or model memory cost. Raw CPU resource
receipts are retained; every job used the official 6GiB/no-swap RAM slot,
private caches, timeout, QUIET guards and allowed cores at nice19.

This verdict covers the 12 test repairs only. The full CPU-suite release gate,
GPU/forecast qualification and the supplemental active-hail product issue are
outside scope. Finished private caches, temp trees and the snapshot are removed
after this seal; CLEANUP.json records their final absence.

Per-commit source/contract citations: STATIC_REVIEW.md. Full source/hash audit:
AUDIT.json. All 42 skip records: SKIP_AUDIT.json. Counts and resources: REVIEW.json.
