# FIX — rv4-t-redsuite final batch

`60675e705` needs one narrow test correction. `tests/test_m5_rrtmg_harness.py:34-35` skips when the executable compiler is configured through `WRF_PRISTINE_FC` or `WRF_ENV`, because its new guard checks only `FC` and PATH `gfortran`. The real unchanged builder supports those settings at `scripts/wrf_rrtmg_harness_build.sh:39,49,71-72` (also the `GPUWRF_WRF_ENV` alias).

R05 proves both cases independently: the real private builder selects the executable compiler and rejects deliberately invalid objects; the repaired test then incorrectly skips. Both counterexamples are genuine AssertionErrors with zero setup errors. The real positive WRF build passes with absolute FC. Actual absent objects/table/compiler explicitly skip; present invalid objects fail rather than skip.

Required fix: mirror/reuse the builder's full compiler resolution and precedence, preserve every existing assertion and all honest absence guards, and add regressions for configured compilers. No production or numerical change is needed. The remaining commits are MERGE; the table below records each disposition. `a367c1a57` was already integrated and `c62e5dd78` only archives the previously reviewed followup; their evidence is reused without a second review.

[M CPU/source] 85 passed executions, 1 legitimate compiler-absent skip; 3/3 product mutants killed by genuine assertions; 2 false-skip counterexamples; 0 setup errors. Every numerical tolerance/bound is retained. All1,249 hash checks and28 heavy-test rows reproduce their actual receipt/source/outcome/HWM/cap/no-swap evidence. No production-source changes in this review. R06 independently passes all13 affected C123/NSSL checks on accepted CAMMG/main9edb53e3a composition.

The debug-tap gate is explicitly trace/shape-only: its original four shape assertions remain, with additional all-leaf checks. Its retired runtime OOM is not passed runtime/fidelity coverage. Heavy peaks are measured caps/HWMs or explicit unknown/lower bounds; four unexecuted nodes remain unexecuted. The full producer CPU suite remains INCOMPLETE and mixed-source.

All tests used immutable commits, official6GiB/no-swap RAM slots, timeouts, QUIET guards, private caches/basetemp, JAX CPU and cores8,9,24,25 nice19. No GPU. Model performance/memory/compile behavior is unchanged by this test-only range (source identity), not newly benchmarked. CPU-review peak RSS3,540,408KiB (about3.38GiB). Finished scratch is cleaned after the last consumer.

| Commit | Verdict | Reason |
|---|---|---|
| `c62e5dd78` | MERGE | Historical followup proof packet only; its twelve test repairs already reviewed by rv3; no second review. |
| `20fb4b0db` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `6f9b3404f` | MERGE | Only private proof destination changes; original closure/units assertions pass. |
| `60675e705` | FIX | FIX: compiler guard ignores available WRF_PRISTINE_FC and WRF_ENV; two real-builder counterexamples. |
| `1b8c64017` | MERGE | Original four tap-shape asserts retained and strengthened; trace-only coverage explicitly disclosed, not runtime/fidelity. |
| `1a534d8e5` | MERGE | Only subprocess output destination changes; actual schema/tamper probes and four assertions preserved. |
| `cc8a20545` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `c9c6a796d` | MERGE | Declared mp default8 replaces incomplete mock; same no-route expectation; wrong-route product mutant killed. |
| `e3aa610ba` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `b2f71e5c4` | MERGE | Existing ri3d -> rimi contract at scan_adapters.py:292; all frozen WRF mass/thermal assertions retained; product mutant killed. |
| `a367c1a57` | MERGE | Already integrated as dd4424ad3; reuse exact repaired-function identity, no second review. |
| `bde7e89a3` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `a085a9b97` | MERGE | Existing60s fixture and MP0/CU0 isolate PBL; all original mutation assertions/bounds unchanged; actual caller passes. |
| `4b6e0dfbf` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `ab6fce68d` | MERGE | LW/SW _STATIC_FIELDS at rrtmg_lw.py:347 / rrtmg_sw.py:219; static metadata exact, dynamic gases preserved; mutant killed. |
| `092b8abb5` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `2bc272a65` | MERGE | Only private JSON_OUT; real B2 regression and unchanged absolute/relative gates pass. |
| `13672579f` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `e38b343ec` | MERGE | Original particle counts4/4/8/5 retained plus literal3/3/6/3 held diagnostics; all selected/unused shape/dtype/zero checks preserved. |
| `fd037345a` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `729557973` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `02fe5b8e3` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `5f4278aff` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `eec09ea7e` | MERGE | Matches operational_mode.py:7839-7840 required REAL zero radar seed; acceptedCAMMG composition13 strict checks independently pass. |
| `e537866a9` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
| `7539dc3e3` | MERGE | Historical source/command/XML/resource receipt hashes verified; no test or product behavior changed. |
