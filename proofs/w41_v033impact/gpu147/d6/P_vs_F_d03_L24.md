# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-10-18T09:47:30.396010+00:00`

## Verdict

- verdict: `PASS`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in PSFC, Q2, QVAPOR, RAINNC, SST using same-state/tendency probes.

## Coverage

- domain `d03`; paired files `24`; leads `1`-`24` h
- CPU files `24`; GPU files `24`; unmatched CPU/GPU `0`/`0`
- variables CPU/GPU/common `377`/`377`/`377`
- compared numeric `14`; dynamic `14`; static/time-invariant `0`; time metadata `0`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `PSFC` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `Q2` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `QVAPOR` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `RAINNC` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `SST` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `T` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `T2` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `TSK` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `TSLB` | dynamic | 0 | 0 | 0 | 0 | 1 |
| `U` | dynamic | 0 | 0 | 0 | 0 | 1 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `PSFC` | 0 | 0 | NA | 0 |
| `Q2` | 0 | 0 | NA | 0 |
| `QVAPOR` | 0 | 0 | NA | 0 |
| `RAINNC` | 0 | 0 | NA | 0 |
| `SST` | 0 | 0 | NA | 0 |

## Top 5 Coverage Issues

- none

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in PSFC, Q2, QVAPOR, RAINNC, SST using same-state/tendency probes.
