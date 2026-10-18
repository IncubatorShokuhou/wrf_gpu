# V0.14 Wrfout Grid Comparison Smoke

Generated UTC: `2026-10-17T12:19:54.419545+00:00`

## Verdict

- verdict: `PASS`
- tolerance manifest: `True`
- CPU-only: `True`; GPU used: `False`
- next: With static fields quiet, localize first dynamic divergence in QNICE, ACHFX, ACSWDNB, ACLWDNB, ACSWUPT using same-state/tendency probes.

## Coverage

- domain `d01`; paired files `25`; leads `0`-`24` h
- CPU files `25`; GPU files `25`; unmatched CPU/GPU `0`/`0`
- variables CPU/GPU/common `362`/`377`/`362`
- compared numeric `361`; dynamic `173`; static/time-invariant `187`; time metadata `2`

## Top 10 Field Differences

| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QNICE` | dynamic | 4201.618 | 0.512 | 1055.938 | 1.702e+06 | 14 |
| `ACHFX` | dynamic | 2.034e+04 | -294.058 | 7.856e+04 | 4.043e+05 | 24 |
| `ACSWDNB` | dynamic | 3.831e+04 | -8448.541 | 1.485e+05 | 2.554e+05 | 17 |
| `ACLWDNB` | dynamic | 2.672e+04 | 2109.521 | 9.150e+04 | 1.690e+05 | 24 |
| `ACSWUPT` | dynamic | 2.155e+04 | 3440.455 | 8.924e+04 | 1.676e+05 | 15 |
| `ACLHF` | dynamic | 6213.036 | -269.132 | 2.401e+04 | 1.597e+05 | 24 |
| `ACSWUPB` | dynamic | 1.993e+04 | -4390.283 | 9.312e+04 | 1.483e+05 | 17 |
| `ACGRDFLX` | dynamic | 8726.149 | -333.447 | 3.632e+04 | 1.213e+05 | 24 |
| `ACLWUPBC` | dynamic | 7183.436 | 976.720 | 2.798e+04 | 1.108e+05 | 24 |
| `ACLWUPB` | dynamic | 7851.884 | 1015.505 | 3.029e+04 | 1.107e+05 | 24 |

## Top 5 Drift Signals

| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |
| --- | ---: | ---: | ---: | ---: |
| `ACSWDNB` | 3159.096 | -1064.727 | 0.941 | 5.422e+04 |
| `ACSWUPB` | 1680.320 | -550.699 | 0.941 | 2.866e+04 |
| `ACSWUPT` | 1750.970 | 439.629 | 0.941 | 3.019e+04 |
| `ACLWDNB` | 1378.096 | 241.514 | 0.917 | 3.699e+04 |
| `ACHFX` | 1457.435 | -60.010 | 0.500 | 3.135e+04 |

## Top 5 Coverage Issues

- `gpu_only` count `15`: `CLDFRA_BL`, `CON`, `DTAUX3D`, `DTAUY3D`, `DUSFCG`, `DVSFCG`, `OA1`, `OA2`

## Next Debug Recommendation

With static fields quiet, localize first dynamic divergence in QNICE, ACHFX, ACSWDNB, ACLWDNB, ACSWUPT using same-state/tendency probes.
