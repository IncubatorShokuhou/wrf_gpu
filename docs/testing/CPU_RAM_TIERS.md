# CPU RAM tiers for the SMS, KF and nested-fusion gates

These are CPU test-harness limits, not GPU memory or forecast-fidelity results.
Use fresh pytest processes through `scripts/with_ram_slot.sh`; an entire suite
in one process can retain JAX executables from earlier tests.

The ordinary 6G tier preserves the original SMS pristine-WRF inputs, all 17
checked outputs and tolerances. It compiles one output at a time and clears
JAX compilation caches after each output. This increases test compilation work
but avoids the original simultaneous multi-output compiler peak.

The KF production dispatch gate loads only real PROD d01, which is the only
domain it traces and asserts. It still tests both independent/composed flag
settings and every original dispatch/order/carry assertion. The shared loader's
default remains d01+d02 for tests of actual nesting. This is structural trace
coverage, not executed forecast fidelity. Omitting d02 also omits its unused
initial physics compilation.

The full seven-sibling fusion regression is marked `heavy`, preserving its
grids, ratio, three root steps, alarm, 8 fused / 1 fallback schedule and every
ordering/recompile-floor assertion. A one-leaf reduction also hit the official
6G cgroup limit before a numerical verdict, so it was discarded. Explicit
terminal finiteness checks prevent Python `max()` from discarding NaN deltas.
The analytic fixture is a scheduling regression, not a CPU-WRF oracle.

`heavy` tests are skipped by default with a reason: they need about 17 GiB and
must run with `--run-heavy` in a 24G RAM slot. Use pytest `-rs` to display it.
`--run-heavy` includes them;
`--run-heavy -m heavy` selects only that tier. `-m heavy` alone does not opt in.
An explicit skip is never numerical validation. The historical seven-leaf
case passed at 17.043 GiB RSS under a 24G/no-swap cap; 24G is a measured working
cap, not a proven minimum. Full receipts are in
`proofs/t_redsuite/HEAVY_CONTINUATION_20261015.{md,json}`.

Example singletons (lane t-heavytests has cores 2,3; use your current grant):

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 JAX_ENABLE_X64=true taskset -c 2,3 nice -n 19 scripts/with_ram_slot.sh --label t-heavytests --max 6G --timeout 1200 -- timeout 900 python -m pytest tests/dynamics/test_sms3dtke_full.py::test_sms_complete_compiled_pristine_operator --basetemp <USER_HOME>/wrf_gpu2_lanes/t-heavytests/pytest
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 JAX_ENABLE_X64=true taskset -c 2,3 nice -n 19 scripts/with_ram_slot.sh --label t-heavytests --max 24G --timeout 1200 -- timeout 900 python -m pytest --run-heavy -m heavy tests/test_v0234_b2_bounded_terminal_gate.py --basetemp <USER_HOME>/wrf_gpu2_lanes/t-heavytests/pytest
```

Wait and retry if QUIET is printed. The 24G tier needs enough shared host
headroom and the official RAM slot; the marker does not allocate memory or
raise a cap. Other historical heavy tests in the continuation inventory remain
separate follow-up work; this change does not promise that all test files or
arbitrary multi-file sessions fit 6G. No production source, runtime launches,
allocations, traffic, precision, GPU compile or cache key changes are made.
