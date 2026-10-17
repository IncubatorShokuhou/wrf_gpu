# Verified test repairs (partial suite)

The table below is the first ten-commit batch, already independently reviewed.
The next35testcommits are sealed in [REVIEW_BATCH_20261014.md](REVIEW_BATCH_20261014.md);
the subsequent12verifiedtestcommits are in [FOLLOWUP_20261014.md](FOLLOWUP_20261014.md).
All initially dirty Python repairs are now committed; the full-suite gate remains incomplete. No production files, numerical tolerances, historical physics receipts,
or release defaults changed. Production source tree: `97b49b49c7f1a158807f16a41d84857c1a0e83c8`.

| Commit | Before | Required result [M CPU] |
|---|---|---|
| de2e7089e | 3 stale SSiB assertions | FAST0 and FAST1 each 4 PASS, strict missing-wind error / exact land field bytes |
| 9a958cb46 | 2 radiation order/cache failures | reload predecessor plus MAXRAND 6 PASS, original pristine C/D and deletion bounds |
| 984f5070c | 1 missing external source | 8 PASS, 4 explicit external-input SKIP; present inputs retain original checks |
| 1c5561af4 | 7 NVML shell unit failures | 29 PASS, only expected memory.free query stubbed; unexpected GPU queries refuse |
| 873cac346 | 2 superseded qualification pins | 12 PASS, paired FLG refusal and qualified open-top rhs_ph |
| 50873bfc5 | 2 incomplete history mocks | 12 PASS, 1 existing explicit SKIP; canonical defaults, real write boundary |
| 1b6cf9f93 | 18 incomplete lower-boundary mocks | 18 PASS, unchanged real WN3 operand byte checks and solver order |
| 6122c751b | 2 MYJ bare-State mocks | 8 PASS, original pristine CPS-rate/ULP/byte assertions plus held history values |
| b0d618512 | 4 stale State/backend assertions | 45 PASS, 3 inherited XFAIL; populated State identity, independent byte sum |
| d185e9dce | 5 legacy analytic storage failures | 32 PASS; explicit scoped fp64, original convergence and 1e-12 bounds |

`VERIFIED_REPAIRS.json` binds exact file bytes and original full XML SHA for
C003's five successful files. `VERIFIED_REPAIRS.xml` is an explicitly selected
subset of that original XML; C003's 18 lower-boundary failures were then repaired
and measured separately, without repeating its 61 passing cases.
`RECEIPTS.json` binds all original measured arm commands and XML hashes.

Merging these tests adds no model launches, allocations, traffic, synchronization,
or executable compilation. Cache isolation intentionally recompiles the affected
CPU oracle fixture; C001 stayed below 2.7 GiB RSS, no swap. All post-reboot test
processes use the main two-slot RAM wrapper and an actual 6 GiB/no-swap cgroup.
These are regression/caller checks, not a new forecast-fidelity or speed claim.

Overall full-suite gate remains **INCOMPLETE**. The historical monolithic RS02 run
was killed after 15510/16094 nodes; the final composed candidate is being checked
in fresh capped processes. Root-flow MP28/MP40 heavy singletons are deferred until
tomorrow after the ALISIOS nightly DONE signal under the manager's 12 GiB exception.
The independent SSiB optional-history product issue belongs to t-u10erestart.
