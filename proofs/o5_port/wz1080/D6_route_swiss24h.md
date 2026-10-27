# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-10-24T21:21:59.640271+00:00`

## Verdict

- verdict: `PASS`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in QNICE, ACHFX, ACSWDNB, ACSWUPT, ACLWDNB using same-state/tendency probes.

## Coverage

- domain `d01`; paired files `25`; leads `0`-`24` h
- CPU files `25`; GPU files `25`; unmatched CPU/GPU `0`/`0`
- variables CPU/GPU/common `362`/`378`/`362`
- compared numeric `361`; dynamic `173`; static/time-invariant `187`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QNICE` | dynamic | 3733.410 | 7.849 | 901.872 | 8.471e+05 | 23 |
| `ACHFX` | dynamic | 1.998e+04 | -187.935 | 7.921e+04 | 4.266e+05 | 24 |
| `ACSWDNB` | dynamic | 3.536e+04 | -6941.304 | 1.271e+05 | 3.724e+05 | 17 |
| `ACSWUPT` | dynamic | 2.076e+04 | 2911.927 | 7.880e+04 | 2.308e+05 | 17 |
| `ACLWDNB` | dynamic | 2.556e+04 | 967.005 | 8.538e+04 | 1.521e+05 | 24 |
| `ACSWUPB` | dynamic | 1.742e+04 | -3513.552 | 7.425e+04 | 1.304e+05 | 17 |
| `ACGRDFLX` | dynamic | 8530.822 | -708.835 | 3.388e+04 | 1.116e+05 | 24 |
| `ACLHF` | dynamic | 5564.347 | -220.144 | 2.291e+04 | 8.904e+04 | 24 |
| `ACSWUPBC` | dynamic | 5447.348 | -476.218 | 2.580e+04 | 7.353e+04 | 17 |
| `ACSWUPTC` | dynamic | 4911.375 | -425.436 | 2.316e+04 | 6.624e+04 | 17 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `ACSWDNB` | 2897.029 | -888.452 | 0.941 | 4.976e+04 |
| `ACSWUPT` | 1684.745 | 377.535 | 0.941 | 2.904e+04 |
| `ACSWUPB` | 1454.857 | -450.788 | 0.765 | 2.486e+04 |
| `ACHFX` | 1429.941 | -41.935 | 0.542 | 3.124e+04 |
| `ACLWDNB` | 1275.325 | 126.692 | 0.875 | 3.517e+04 |

## Top 5 Coverage Issues

- `gpu_only` count `16`: `CLDFRA_BL`, `CON`, `DTAUX3D`, `DTAUY3D`, `DUSFCG`, `DVSFCG`, `OA1`, `OA2`

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in QNICE, ACHFX, ACSWDNB, ACSWUPT, ACLWDNB using same-state/tendency probes.
