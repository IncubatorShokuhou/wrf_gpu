# Hail-mass final-source GPU requalification

Source `627a033284dc457a9949c2a32f2c111d0227e6df`. Original CPU-WRF Swiss 42×42×44, dt18s, full24h, hourly h0–24; original input and namelist hashes match. Frozen D6 bounds unchanged. No model edits by v-run.

| Config | D6 / coverage | E41 | Cold wall (s) | Sampled VRAM (MiB) |
|---|---|---|---:|---:|
| mp18 | PASS / complete | FAIL | 901.508 | 2714 |
| mp18_cu5 | PASS / complete | FAIL | 963.845 | 2720 |
| mp24 | PASS / complete | FAIL | 676.287 | 1600 |
| mp26 | PASS / complete | FAIL | 920.507 | 1598 |

[M] Each verdict uses every paired hourly frame and all original CPU inventory. See per-config grid_gate.json for first failure, missing fields and report-only values. E41 scans the real paid JIT+AOT artifacts before deleting caches; exact native PINs and GPU files remain external with hashes.

F2 negative phase-field visits remain OPEN. Finite output alone does not clear pre-mask events. Cold concurrent wall is qualification timing, not S1, a warm benchmark, or a speedup claim. No release clearance claimed.

The lock released between pairs; a real intervening v-prep cohort is bound in SECOND_PAIR_ORDER.json. Source snapshots and caches are disposable; original CPU references and previous science are preserved. Logs are lossless gzip, verified against originals.
