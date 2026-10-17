# Passing MYNN reference repeatability

Two passing composites each ran original WRF twice for 24 hours, serially through official 6 GiB/no-swap RAM slots. Each pair used the same executable and working directory, with private regular namelists. All four runs completed with 25 finite frames and sealed input hashes. Original executable SHA starts `52f914cb`; the only private binary changes are two reversible diagnostic OPEN paths.

| Case | MP / PBL | Fields × frames | Pair arrays / files | Arrays vs original reference | Zero replacement |
|---|---|---|---|---|---|
| c00 | 8 / 5 | 362 × 25 | BYTE / BYTE | BYTE | unnecessary |
| c17 | 24 / 5 | 361 × 25 | BYTE / BYTE | BYTE | unnecessary |

18,075 field/frame pairs compared; all 50 preserved canonical files independently match their original SHA manifests. Results qualify these two references only. c21–c24 zero-init references and original-vs-zero spread remain t-p3fix-owned under the manager MYNN policy.

All run outputs and receipts are retained under `<DATA_ROOT>/wrf_gpu2/scratch/v-prep/mynn-repeat-20261014/`. Temporary executables, runtime directories and copied MP tables were cleaned. No GPU execution or speed claim.
