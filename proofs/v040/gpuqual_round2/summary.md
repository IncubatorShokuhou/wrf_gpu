| Config | D6 + coverage | Bounded D6 | Frames | Missing CPU fields | E41 | Cold concurrent s/fc-h | Sampled VRAM MiB |
|---|---|---|---:|---|---|---:|---:|
| mp16 | PASS | PASS | 25 | — | PASS | 34.16429532637509 | 1596 |
| mp24 | PASS | PASS | 25 | — | FAIL | 29.828799611714203 | 1648 |
| mp26 | PASS | PASS | 25 | — | FAIL | 39.551297469666075 | 1598 |
| mp40 | INCOMPLETE | PASS | 25 | CCN1_GS, CCN2_GS, CCN3_GS, CCN4_GS, CCN5_GS, CCN6_GS, CCN7_GS, EFCG, EFIG, EFSG, MSKF_REFL_10CM, WACT | FAIL | 21.600531321086844 | 1594 |
| bl9 | PASS | PASS | 25 | — | PASS | 45.93826120137237 | 1726 |
| fdda1 | PASS | PASS | 25 | — | PASS | 23.68337081274755 | 1714 |
| mp18 | INCOMPLETE | PASS | 25 | REFD_MAX, REFL_10CM | FAIL | 38.9527446514597 | 2634 |
| mp18_cu5 | INCOMPLETE | PASS | 25 | REFD_MAX, REFL_10CM | FAIL | 41.333081548829796 | 2638 |
| mp28 | INCOMPLETE | PASS | 25 | QNBCA, QNIFA2D, QNWFA2D, TAOD5502D | FAIL | 22.17535736937619 | 2870 |
| release_default | PASS | PASS | 25 | — | PASS | 21.729144634921493 | 1712 |

Every numerical bound is the unchanged frozen D6 bound, assessed at every hourly frame 0–24.
All required arms processed: True. All required D6 coverage gates pass: False.
Missing fields prevent full qualification. Unbounded fields are report-only.
Times include cold compilation and shared concurrent work; they are not S1 speed claims.
Raw F2 events remain separate evidence and do not become justified through a D6 pass.

An additional Swiss MP_RE=0 diagnostic is preserved separately. It is not the release-default qualification.
