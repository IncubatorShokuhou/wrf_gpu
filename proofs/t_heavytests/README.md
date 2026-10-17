# t-heavytests qualification

READY-FOR-CRITIC: CPU test-harness gate, with the allowed heavy-tier fallback.

| Gate | Current measured peak RSS GiB | Cap | Result |
|---|---:|---|---|
| Compiled SMS, all17 pristine-WRF outputs/original bounds | 2.957 | 6G/noSwap | PASS (337.76s process wall) |
| Real PROD d01 KF dispatch, both independent/composed settings | 4.840 | 6G/noSwap | 2PASS (86.91s process wall) |
| Full seven-sibling fused/eager terminal regression | 17.410 | 24G/noSwap | PASS (281.50s process wall) |
| Tier/loader/NetCDF/eager SMS consumers | 2.801 | 6G/noSwap | 16PASS,1heavy deselected |
| NaN/plusInf/minusInf state/carry guard + finite control | 0.173 | 6G/noSwap | 7PASS |

Both reduced harnesses reject genuine mutants: SMS momentum corruption and
removed KF RK coupling each cause one assertion failure with zero setup errors.
The one-leaf fusion reduction OOMed under6G before a verdict and was discarded;
its original scope journal is retained. The full seven-sibling test keeps every
original assertion and adds explicit terminal finiteness before the floor.

QUALIFICATION.json hashes every raw receipt and records measured commits. SMS
and KF checks reuse exact final file bytes. The final fusion guard extraction
preserves the paid body AST and inner finite loop AST; seven new host-only guard
checks qualify that extraction without repeating the expensive physical run.
The whole model source tree is BYTE-identical to main. All original gate
assertion expressions remain; no tolerances or fixture values changed.

Per-output SMS compilation trades more CPU compilation/cleared caches for lower
peak memory. Root-only KF avoids unused d02 initialization and preserves the
nested root ozone carry ABI. Default shared PROD loading still uses both domains.
No production launch/traffic/allocation/precision/GPU compile cost changes.
Other historical heavy tests remain unclassified by this narrow lane; this is
not an entire-suite6G or GPU forecast-fidelity claim. Snapshots/caches cleaned.

See ../../docs/testing/CPU_RAM_TIERS.md for ordinary/heavy commands. Reproduce
mutants with JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0
JAX_ENABLE_X64=true PYTHONPATH=src:. and GPUWRF_HEAVY_MUTANT=sms or kf;
add `-p proofs.t_heavytests.mutants` to the corresponding test invocation and
run via the official6G RAM slot, your assigned CPU affinity and timeout.
