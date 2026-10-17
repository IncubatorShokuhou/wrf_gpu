| Config | Qualification | Frames / CPU fields | Cold s/fc-h | VRAM MiB sampled | E41 |
|---|---|---:|---:|---:|---|
| bl9 | FAIL | 25 / 357 | 37.963 | 1726 | PASS |
| cu5 | PASS | 25 / 362 | 20.651 | 1716 | FAIL; kInput max 1713 HLO > 500 |
| cu93 | PASS | 25 / 362 | 18.444 | 1716 | PASS |
| km3 | BOTH_RUNS_FAILED | — / — | — | 1714 | FAIL; kInput max 853 HLO > 500 |
| km2 | PASS | 25 / 362 | 18.469 | 1714 | FAIL; kInput max 1287 HLO > 500 |
| sf3 | PASS | 25 / 209 | 12.718 | 1712 | PASS |
| ra3 | PASS | 25 / 362 | 24.926 | 6174 | FAIL; STACK 2144 B > 256; kInput max 2141 HLO > 500 |
| mp40 | FAIL | 25 / 377 | 20.911 | 2618 | FAIL; kInput max 2226 HLO > 500 |
| mp24 | FAIL | 25 / 361 | 22.839 | 3334 | FAIL; STACK 3864 B > 256 |
| mp28 | INCOMPLETE | 25 / 369 | 18.735 | 2868 | FAIL; STACK 928 B > 256; kInput max 1510 HLO > 500 |
| mp16 | FAIL | 25 / 363 | 30.259 | 1596 | PASS |
| mp26 | FAIL | 25 / 364 | 30.495 | 2620 | FAIL; STACK 832 B > 256; kInput max 1220 HLO > 500 |
| mp18 | INCOMPLETE | 25 / 372 | 25.884 | 2632 | FAIL; kInput max 728 HLO > 500 |
| mp18_cu5 | INCOMPLETE | 25 / 372 | 27.951 | 2640 | FAIL; kInput max 728 HLO > 500 |
| urban1 | PASS | 25 / 362 | 18.400 | 1714 | PASS |
| fdda1 | FAIL | 25 / 363 | 17.077 | 1712 | PASS |
| dfi3 | PASS | 25 / 362 | 29.867 | 1856 | PASS |
| moist0 | PASS | 25 / 362 | 18.597 | 1714 | PASS |

All numbers below are measured [M]. Incomplete coverage prevents qualification even when every bounded field passes.

| Config | First D6 failure | Missing CPU fields |
|---|---|---|
| bl9 | T2 h4: RMSE 1.7002 > 1.5 | — |
| km3 | No valid D6: CPU crashed in both Swiss/Madrid; Swiss GPU nonfinite u observed h1 | — |
| mp40 | RAINNC h13: RMSE 1.90303 > 1 | CCN1_GS, CCN2_GS, CCN3_GS, CCN4_GS, CCN5_GS, CCN6_GS, CCN7_GS, EFCG, EFIG, EFSG, MSKF_REFL_10CM, QNCLOUD, QNGRAUPEL, QNSNOW, WACT |
| mp24 | RAINNC h12: RMSE 1.36091 > 1 | QHAIL |
| mp28 | No bounded D6 failure; coverage incomplete | QNBCA, QNCLOUD, QNIFA, QNIFA2D, QNWFA, QNWFA2D, TAOD5502D |
| mp16 | RAINNC h11: RMSE 1.18838 > 1 | QNCCN, QNCLOUD |
| mp26 | RAINNC h11: RMSE 1.28284 > 1 | QHAIL, QNCCN, QNCLOUD |
| mp18 | No bounded D6 failure; coverage incomplete | QHAIL, QNCCN, QNDROP, QNGRAUPEL, QNHAIL, QNSNOW, QVGRAUPEL, QVHAIL, REFD_MAX, REFL_10CM |
| mp18_cu5 | No bounded D6 failure; coverage incomplete | QHAIL, QNCCN, QNDROP, QNGRAUPEL, QNHAIL, QNSNOW, QVGRAUPEL, QVHAIL, REFD_MAX, REFL_10CM |
| fdda1 | PSFC h1: RMSE 203.506 > 120 | HFX_FDDA |
