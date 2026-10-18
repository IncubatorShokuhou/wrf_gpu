| arm | &stoch | D6 vs CPU-WRF (fields x frames) | stochastic fields / seeds | E41 kernels / STACK | census nonfinite / repaired | cold run s | steady s per forecast h | VRAM MiB |
|---|---|---|---|---|---|---|---|---|
| c00 | OFF reference (v-prep c00, 2026-10-16, tree 04cb09f8) | PASS (362 x 25; 0 fail, 0 nonfinite) | none written (OFF) | PASS 2459 / 16 B | 0 / 174717 | 521.5 | 0.913 | 5490 |
| st1 | SKEBS | PASS (365 x 25; 0 fail, 0 nonfinite) | PASS 3 fields, worst 5.2e-07; seeds 25/25 | PASS 2458 / 16 B | 0 / 176737 | 595.1 | 0.999 | 5564 |
| st2 | SPPT 15 km | PASS (363 x 25; 0 fail, 0 nonfinite) | PASS 1 fields, worst 6.2e-07; seeds 25/25 | PASS 2481 / 16 B | 0 / 172848 | 589.6 | 0.976 | 5564 |
| st3 | SPP (MYNN) 15 km | PASS (374 x 25; 0 fail, 0 nonfinite) | PASS 12 fields, worst 1.9e-06; seeds 25/25 | PASS 2486 / 16 B | 0 / 174445 | 588.4 | 1.005 | 5564 |
| st4 | all + RAND, vert 1, nens 7 (serial RNG) | INCOMPLETE (379 x 25; 0 fail, 0 nonfinite; CPU-only FIELD_CONV,FIELD_PBL,FIELD_SF) | FAIL 14 fields, worst 1.6e-06; seeds 25/25; missing on GPU FIELD_CONV,FIELD_PBL,FIELD_SF | PASS 2547 / 8 B | 0 / 174984 | 686.7 | 1.988 | 5566 |
| st4p | all + RAND, vert 1, nens 7 | PASS (379 x 25; 0 fail, 0 nonfinite) | PASS 17 fields, worst 1.6e-06; seeds 25/25 | PASS 2566 / 8 B | 0 / 174234 | 631.4 | 1.122 | 5566 |
