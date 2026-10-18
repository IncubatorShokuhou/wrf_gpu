# w41-stoch — stochastic physics GPU qualification (Swiss d01, 24 h) + ens presets

Lane w41-stoch, 2026-10-17. Labels: [M] measured, [I] inferred. GPU = RTX 5090 (cuda:0 asserted in every
receipt), official lock, one arm per lock session, systemd scope memory cap. Reference = ORIGINAL serial
CPU-WRF 4.7.1 (`<USER_HOME>/src/wrf_pristine/WRF/main/wrf.exe` sha 52f914cb, private copy with only the two
E104 debug OPEN literals redirected), run with the IDENTICAL namelist file, hence identical seeds.

## Cases

Base = v0.4.0 default Swiss composite v-prep c00 (mp8, cu0, MYNN bl5/sf5, Noah-MP, RRTMG; 43x43x45, 3 km,
dt 18 s, 24 h), inputs unchanged, only `&stoch` appended (`prepare_cases.py`). Gaussian length scales 15 km:
with the Registry defaults the original CPU-WRF run of v-prep c41 wrote NaN SPP spectral coefficients and a
constant clipped pattern (docs/UPSTREAM_WRF_BUGS.md). OFF reference = the v0.4.0 final-validation GPU run of
v-prep c00 (2026-10-16, tree 04cb09f8, same inputs, no `&stoch`); a same-tree OFF arm (st0) was prepared but not
run, to stay inside the 60 GPU-min grant (contract arms are st1-st4).

## Gates per arm

1. `proofs/v040/final_validation/score.py`: frozen D6 manifest 9df4fab8 over all 25 hourly frames, every
   original history field present on GPU, every frame's header/schema, raw finite audit of both arms.
2. `pattern_compare.py`: every WRF Registry.stoch history field (patterns, FIELD_*, SPP spectral
   coefficients) per frame vs CPU-WRF, pre-registered `8e-6 + 64 eps` of the field max (u2-stoch ST18/ST40),
   and the six ISEEDARR* seed arrays exact.
3. E41 cubin gate (`scripts/wave_cubin_gate.py`, JIT + AOT d01 programs).

## Results

| arm | &stoch | D6 vs CPU-WRF (fields x frames) | stochastic fields / seeds | E41 kernels / STACK | census nonfinite / repaired | cold run s | steady s per forecast h | VRAM MiB |
|---|---|---|---|---|---|---|---|---|
| c00 | OFF reference (v-prep c00, 2026-10-16, tree 04cb09f8) | PASS (362 x 25; 0 fail, 0 nonfinite) | none written (OFF) | PASS 2459 / 16 B | 0 / 174717 | 521.5 | 0.913 | 5490 |
| st1 | SKEBS | PASS (365 x 25; 0 fail, 0 nonfinite) | PASS 3 fields, worst 5.2e-07; seeds 25/25 | PASS 2458 / 16 B | 0 / 176737 | 595.1 | 0.999 | 5564 |
| st2 | SPPT 15 km | PASS (363 x 25; 0 fail, 0 nonfinite) | PASS 1 fields, worst 6.2e-07; seeds 25/25 | PASS 2481 / 16 B | 0 / 172848 | 589.6 | 0.976 | 5564 |
| st3 | SPP (MYNN) 15 km | PASS (374 x 25; 0 fail, 0 nonfinite) | PASS 12 fields, worst 1.9e-06; seeds 25/25 | PASS 2486 / 16 B | 0 / 174445 | 588.4 | 1.005 | 5564 |
| st4 | all + RAND, vert 1, nens 7 (serial RNG) | INCOMPLETE (379 x 25; 0 fail, 0 nonfinite; CPU-only FIELD_CONV,FIELD_PBL,FIELD_SF) | FAIL 14 fields, worst 1.6e-06; seeds 25/25; missing on GPU FIELD_CONV,FIELD_PBL,FIELD_SF | PASS 2547 / 8 B | 0 / 174984 | 686.7 | 1.988 | 5566 |
| st4p | all + RAND, vert 1, nens 7 | PASS (379 x 25; 0 fail, 0 nonfinite) | PASS 17 fields, worst 1.6e-06; seeds 25/25 | PASS 2566 / 8 B | 0 / 174234 | 631.4 | 1.122 | 5566 |

D6 (frozen manifest 9df4fab8) has numerical bounds for 62 variables; the 362-379 counts are the full
field inventory compared in every frame, of which the unbounded fields are REPORT_ONLY (FINDINGS D28). The
stochastic fields and seeds have their own gate (pattern column).

Source commit of st1/st2/st3/st4p: 07ef43fe6 (src == lane head src); st4: 0c43829f46 (pre writer fix, serial
RNG). Machine-readable: `SUMMARY.json` (summarize.py). Per arm: `cases/<arm>/{D6_d01,PATTERN_d01}.json`,
`e41/wave_gate.json`, `gpu_run.json`, `gpu/census.json` under <DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/.

Timing scope: cold single processes on a shared box, steady = median of the hour-2..24 file intervals;
not S1 benchmarks.

## Findings

- [M] Inventory gap closed: original CPU-WRF writes SPP's FIELD_CONV/FIELD_PBL/FIELD_SF (+0) whenever SPP is
  allocated; the writer emitted them only for cu_physics=1. All-frame CPU proof st3/st4
  (`FIELD_SPP_CPU_ALL_FRAMES.json`), fix c0deda2d8b (pre-rebase 6e25dd55d), GPU st3/st4p write them; st4 (pre-fix tree) is the only
  INCOMPLETE row and differs from st4p only by that host-writer change and the RNG kernel.
- [M] Performance: the WRF/GNU RNG was one single-thread Pallas program per pattern (7 serial calls per step
  with all switches). The exact parallel stream (ad067fd2d0, pre-rebase 58f7edd35) is bitwise identical to it on CPU and GPU
  (`RNG_GPU_PROBE.json`): 4.85 -> 0.36 ms per step on 43x43, 21.4 -> 0.49 ms on 71x121, 78.2 -> 1.12 ms on
  118x268; st4p steady 1.12 vs st4 1.99 s per forecast hour. All 14 stochastic pattern/coefficient/seed
  fields of st4 and st4p are byte-identical in every frame (GPU-vs-GPU regression only); the dynamics differ
  from hour 1 by program-level round-off and chaos, both pass D6 vs CPU-WRF.
- [M] Inventory: no CPU-WRF field missing on GPU in any frame of st1/st2/st3/st4p; schema mismatches 0. GPU-only
  fields: the 15 of the OFF program (GWDO statics/diagnostics, CLDFRA_BL, QC_BL, CON) plus the seven port diagnostics
  STOCH_CLIP_O..W. Header differences = the OFF program's three (TITLE, NTASKS_X/Y).
- [M] NF0: census nonfinite 0 in every arm; repaired 172,848-176,737 = hydrometeor non-negativity guards
  (dynamics.qc/qi/qr/qs/qg), the same guard classes as the OFF reference with similar, NOT equal counts (OFF
  174,717; arms -1.1 % .. +1.2 %) -> no new repair class from the stochastic schemes; F2 stays OPEN (default program).
- [M cross-session, indicative] Overhead vs the OFF reference 0.913 s per forecast hour: SKEBS +9 %, SPPT +7 %,
  SPP +10 %, all + RAND +23 % (~1 ms per 18 s step on 43x43x45); same-session noise of such walls is about 5 %
  (FINDINGS E191). VRAM +74 MiB (5564 vs 5490 sampled
  per-process peak, pool fraction .15); the proof's persistent_state_bytes (39.6 MB, equal in all arms)
  counts the State only, not the stochastic carry.
- [M] v-prep c46's E41 FAIL (two kInput transposes 569/502) is not stochastic: st4/st4p (all switches) PASS.
- [M] Release identities on the final source (07ef43fe6 == lane head src): Swiss 3356e249, PROD 196aed9f exact.

## Presets (`gpuwrf ens prepare --mode stochastic|multiphysics`)

CPU only: tests/test_ensemble_presets.py 27 PASS; 7/7 mutants killed by test assertions, 0 unexpected exceptions
(`MUTANTS.json`, snapshot 87ce8fc363; the first run's M2/M5 'kills' were a product ValueError and a KeyError, E202);
RNG 4/4 mutants killed by AssertionError (`MUTANTS_RNG.json`); tests/test_ensemble.py, test_optimize.py, test_perturb.py unchanged-consumer
runs PASS (offline-CLI 30 s timeouts under 100 % lane load reproduced on the base tree, then PASS on both).

## Open

Nested (WN3 3-nest) stochastic GPU qualification, station skill/spread of stochastic ensembles
(BERNER_PLAN_V050.md), stochastic restart on GPU (CPU-exact only).
