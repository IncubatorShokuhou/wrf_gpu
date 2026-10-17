# Additional verified test corrections

Range `c62e5dd78..1a534d8e5210f6b85595d9a5cc2bc81ced484293`:4testcommits plus1resource-proof commit. Selected results:22PASS and1disclosed legacy CPU-order XPASS. Original critic batch stays frozen.

| Commit | Summary |
|---|---|
| 20fb4b0db | proof: close heavy resource gates; SMS 5 pass, peak 12.525 GiB |
| 6f9b3404f | test: isolate conservation proof output; 2 pass, snapshot clean |
| 60675e705 | test: preserve RRTMG references and verify real fallback; 15 pass |
| 1b8c64017 | test: trace real pre-sanitize tap shape contract; 3 pass, 1.012 GiB |
| 1a534d8e5 | test: isolate schema dry-run output; 2 pass, historical proof intact |

Original command/XML/file hashes and exact scope limits: `EXTRA_FIXES_20261015.json`.

C048 debug tap test is explicitly a real-driver TRACE/STRUCTURAL gate, not coupled-runtime/numerical fidelity. Every original shape assert retained, all tap leaves checked; actual numerical invariant tests unchanged. Original6G/12G compile OOM receipts retained.
C046 old unconditional harness xfail removed only after --runxfail real private-build1PASS; all other conftest AST and CPU guards unchanged.
Legacy CPU-order-dependent SW1W/m2 quarantine remains;1XPASS disclosed, bound unchanged.
Generated historical proof/fixture writes captured privately and exact originals restored; retained reference files never committed as replacements.
Old/new immutable source receipts remain separate; final composed-source/full plan pending; supplemental active-hail product counterexample open.

Zero own production source changes; no added model launches/traffic/allocations. Test trace gate reduces >12GiB compiler failure to1.012GiB without claiming numerical runtime coverage; pure SMS operator original bounds PASS once at24G/HWM12.525GiB, permission spent.
