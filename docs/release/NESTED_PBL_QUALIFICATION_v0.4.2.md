# Boundary layers inside nested domains (v0.4.2)

Until v0.4.1, the YSU, Shin-Hong, ACM2 and EEPS boundary layers were validated on single domains
only. In v0.4.2 all four also passed the 24-hour comparison with the original CPU-WRF **inside
three nested domains**, so they are now validated inside nests in this configuration.

## What was compared

- **Forecast:** the Tenerife set-up with three nested domains (9, 3 and 1 km), issued
  2026-07-07 18 UTC, first 24 hours from 2026-07-08 00 UTC. The four GPU forecasts are the
  ALISIOS G2 runs C2 (YSU), C3 (Shin-Hong), C4 (ACM2) and C7 (EEPS), all with the revised MM5
  surface layer (`sf_sfclay_physics=1`), made with `wrf_gpu` v0.4.1 (release bundle
  `24a7b24f`) and release defaults.
- **Reference:** for each scheme, the original CPU-WRF 4.7.1 (dmpar, 3 ranks) with the same
  namelist and the same input files. The namelists differ only in run length, restart interval
  and MPI tiling.
- **Check:** the frozen D6 tolerances on every hourly frame of every domain (25 frames each),
  scored with `scripts/wn3_score.py`. Every variable of the CPU-WRF history file must be present,
  finite and complete; 62 fields have bounds, all others are reported. The scorer also checks for
  GPU fields that stay constant where CPU-WRF varies and for missing global attributes.
- **Inputs and outputs are the ones compared:** the GPU run's input links and namelist match the
  files the CPU twin used, and all 300 GPU history files match the SHA-256 list published with
  the G2 runs.

## Results

| `bl_pbl_physics` | Scheme | Nested 24 h result (d01, d02, d03) | Largest error vs limit | T2 RMSE, worst hour (d01 / d02 / d03) |
| --- | --- | --- | --- | --- |
| 1 | YSU | **Pass** on all three domains | 0.024 (T2, d03, +21 h) | 0.025 / 0.015 / 0.036 K |
| 11 | Shin-Hong | **Pass** on all three domains | 0.059 (V10, d03, +24 h) | 0.021 / 0.013 / 0.043 K |
| 7 | ACM2 | **Pass** on all three domains | 0.069 (V10, d03, +21 h) | 0.019 / 0.022 / 0.062 K |
| 16 | EEPS | **Pass** on all three domains | 0.110 (V10, d03, +21 h) | 0.014 / 0.033 / 0.061 K |

On every domain: 0 of 62 bounded fields over their limit, no NaN or infinite values in either
model, every CPU-WRF variable present, no GPU field constant where CPU-WRF varies (first, middle
and last hour checked), all global attributes present on every hour.
"Largest error vs limit" is the highest ratio of a field's worst-hour RMSE to its D6 limit; the
largest is 0.165 m/s for EEPS V10 on the 1 km domain, against a limit of 1.5 m/s.

## Limits

- One forecast date (July) and one domain set-up. This shows the nested runs
  follow CPU-WRF closely; it is not a forecast-skill or station verification.
- Checked with the revised MM5 surface layer only. KEPS (`bl_pbl_physics=17`) and the other
  boundary layers remain validated on single domains only.
- The compiled-code quality check (E41) of these nested programs was not measured; the
  single-domain programs of all four schemes have a disclosed E41 failure since v0.4.0.

## Receipts

- Summary of all four schemes: [`proofs/w41_nestqual/NESTQUAL_SUMMARY.json`](../../proofs/w41_nestqual/NESTQUAL_SUMMARY.json)
- Per scheme (`C2`, `C3`, `C4`, `C7`): `NEST_GATE.json` (verdict and numbers), `GPU_STAGE.json`
  (GPU files, SHA checks, engine), `PREPARE.json` (namelist and inputs), `cpu_run.json` (CPU-WRF
  run) and the scorer reports `wn3_d0N.md` in [`proofs/w41_nestqual/`](../../proofs/w41_nestqual/)
