# MERGE — NSSL test-double/import-order repair

To s2-integrate and manager: reviewed `403240f257..c4e3f0d47d4d3f4c3b080a8d4ab314f3e22fbf6c`.
Test commits to integrate: `c8ebdeb64b` (two mock dictionaries) and `f4fe2fad10` (radius-test import order); c4e adds the packet. Test-only scope, no production fix or scientific qualification change.

The added `rainncv` and `sr` keys match the real producer: `nssl2mom/driver.py:100` returns the total/frozen slots and SR, and :166 adds `rain_liquid`. The consuming `nssl_public_precipitation` reads both keys. The zero-valued test dictionaries remain internally consistent, retain their original array/dtype, and now allow the intended variant/cold-start assertions to run.

Whole-file AST comparison proves that removing only those two dictionary keywords restores each test file exactly to the base. Removing only the new module-level scan_adapters import similarly restores the radius test. Every assertion, test body, parameterization and marker remains unchanged; no skip/xfail is added (`RN00-source.json`). The real operational module imports scan_adapters before tracing, so the radius test now reproduces production initialization. The disclosed low-level fresh-JIT lazy-import hazard remains unchanged; this test repair does not claim to fix that separate API hazard.

Independent checks on the immutable candidate:

- **64 PASS**, 0 failures/errors, 57.20 s: both affected one-moment/no-hail files (`RN01-mocks.xml`).
- **2 PASS**, 0 failures/errors, 22.00 s: radius_handoff alone in a fresh process, including actual adapter/pristine-radii comparison (`RN02-radius.xml`). It was intentionally run separately so other tests could not pre-import scan_adapters and hide the defect.
- **2 genuine assertion kills**, 0 errors, .77 s: the independent process-local wrong-variant mutation fails the unchanged MP21 step1/step2 assertions (`RN03-mutant.json/xml/log`). No source file is mutated. The producer's missing-SR KeyErrors are schema-negative controls, not numerical assertion kills.

Source, model data and packaging are byte-identical to base 403240f257; the src Git subtree hash is recorded in RN00. No default identity re-trace, model numerical rerun or GPU work is needed. Runtime performance, compiled program, memory and restart schema are unchanged by construction. The producer's broader 536 PASS over 12 files is reused; the critic reran the three changed files only.

This closes the reviewed seven pre-existing test failures without waiving the final composed release-subset gate. s2-integrate may include the test-only repair in batch63/64 and rerun the final composed subset. No new GPU qualification is introduced.

Evidence: `<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/NSSL/`, durable copy `proofs/rv_t_calleraudit/NSSL/`. Heavy probes use official 6G/no-swap RAM slots, JAX CPU, cores8/9/12 nice19, QUIET guards and timeouts. Snapshot/pytest/cache are removed after the final arm; no persistent cache is used. Tracked logs/XML trim trailing whitespace; raw receipts remain unchanged externally.
