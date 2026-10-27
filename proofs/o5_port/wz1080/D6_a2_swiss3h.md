# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-10-24T18:37:28.224155+00:00`

## Verdict

- verdict: `PASS`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in QNICE, ACLWDNB, ACHFX, ACLWUPT, ACGRDFLX using same-state/tendency probes.

## Coverage

- domain `d01`; paired files `4`; leads `0`-`3` h
- CPU files `25`; GPU files `4`; unmatched CPU/GPU `21`/`0`
- variables CPU/GPU/common `362`/`378`/`362`
- compared numeric `361`; dynamic `146`; static/time-invariant `214`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QNICE` | dynamic | 1795.714 | 11.638 | 12.658 | 4.554e+05 | 3 |
| `ACLWDNB` | dynamic | 7781.730 | 201.670 | 3.127e+04 | 6.451e+04 | 3 |
| `ACHFX` | dynamic | 3309.065 | 111.978 | 1.464e+04 | 4.620e+04 | 3 |
| `ACLWUPT` | dynamic | 2726.634 | 780.428 | 1.041e+04 | 2.797e+04 | 3 |
| `ACGRDFLX` | dynamic | 2028.971 | 8.293 | 8566.175 | 1.882e+04 | 3 |
| `ACLWUPB` | dynamic | 2099.168 | 1185.835 | 7222.925 | 1.667e+04 | 3 |
| `ACLHF` | dynamic | 1036.867 | 57.301 | 5205.913 | 1.636e+04 | 3 |
| `ACLWUPBC` | dynamic | 1948.265 | 1181.929 | 6434.850 | 1.531e+04 | 3 |
| `ACLWUPTC` | dynamic | 1202.859 | 970.545 | 2714.481 | 5676.000 | 3 |
| `SOILENERGY` | dynamic | 12.262 | 0.414 | 4.143 | 1027.211 | 2 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `ACLWDNB` | 3796.815 | 132.296 | 1.000 | 1.163e+04 |
| `ACHFX` | 1643.951 | 81.026 | 1.000 | 4957.423 |
| `ACLWUPT` | 1309.187 | 273.052 | 1.000 | 3994.013 |
| `ACLWUPB` | 948.115 | 465.704 | 1.000 | 3026.130 |
| `ACLWUPBC` | 858.735 | 463.506 | 1.000 | 2757.484 |

## Top 5 Coverage Issues

- `gpu_only` count `16`: `CLDFRA_BL`, `CON`, `DTAUX3D`, `DTAUY3D`, `DUSFCG`, `DVSFCG`, `OA1`, `OA2`

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in QNICE, ACLWDNB, ACHFX, ACLWUPT, ACGRDFLX using same-state/tendency probes.
