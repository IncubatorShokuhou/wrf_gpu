# w42-scorer: bounded-memory WN3 scoring (72 h and 9-nest without OOM)

Tooling only (CPU). No traced model code touched.

## Change
- `scripts/compare_wrfout_grid.py`: ONE CPU+GPU frame pair is open at a time. One pass builds the inventory, then
  the fields stream through in batches whose accumulator state fits `--field-batch-mib` (default 1024), one pass over
  the frames per batch. Lead order inside every field is unchanged. Spatial-split percentiles are cut out of the overall
  per-lead |diff| chunks, so each field keeps one float32 copy instead of five. The report schema and strings are
  unchanged. The new option is read with `getattr`, so Namespace callers (frozen proof scorers) keep working.
- `scripts/wn3_fast_compare.py`: no held handles any more (the 72 h OOM cause); thin entry point.
- `scripts/wn3_score.py`: `--domains` (default `d01 d02 d03`) and `--[no-]sequential-domains` (default sequential when
  `--hours > 24` or more than 3 domains). Per-domain timeout = 3600 s × ceil(frames / 25). Each compare's peak RSS and
  CPU (os.wait4) are printed to stderr and written to `resources.json`. Every domain's file set is checked before the
  first compare starts. `summary.json` and the other result files are unchanged.

## Evidence [M] (core 25 nice 19, 12G RAM slot, paused during foreign QUIET)
| case | scorer | tree peak RSS | largest compare | identity |
|---|---|---|---|---|
| 0227 24 h (v0.4.1 N1 vs guarded F20 CPU) | main d96c268a5e, parallel | 7678 MiB | d02 4401 | reference |
| same | c98cffc56c, parallel (24 h default) | 4193 MiB | d02 1629 | 10/10 files identical |
| same | 4f345e3215, `--sequential-domains` | 1708 MiB | d02 1570 | 10/10 identical |
| 0420 #146 72 h B vs C | 4f345e3215, sequential (default) | 2395 MiB | d02 2246 | 10/10 identical to d6_seq d6_B |
| 0420 #146 72 h F vs C | c98cffc56c, sequential (default) | 2322 MiB | d02 2174 | 10/10 identical to d6_seq d6_F |
| old d6_seq (0125 72 h, w41-swddirscreen) | main, one domain at a time | 15–17 GB per arm | – | – |

Identity = bytes after masking only the comparator's `generated_utc` / `elapsed_seconds`, the md "Generated UTC"
line, and directory aliases with symmetric hit counts (scorer tree root 4/4; hard-linked input dirs 441/441).
4f345e3215 → c98cffc56c changed only wn3_score reporting and `--domains`; the comparator and wrapper bytes are the
same. 0125 could not be re-scored, because its originals were deleted after delivery.

- Edge identity (synthetic case: NaN pairs, U/V staggered splits, missing and incompatible leads, non-numeric fields,
  CPU-only and GPU-only fields, a Times mismatch): old vs new identical in 4/4 configurations (1 and 4 passes, with and
  without splits).
- CPU, d02 24 h, interleaved A/B/A/B (A = main, B = new): 330 / 238 / 767 / 488 s. Pairwise B/A is 0.72 and 0.64, so
  the new path is not slower. Absolute values are noisy because the SMT sibling core 9 ran a CPU-WRF twin and the host
  load was about 49. Peak RSS 4402 → 1622 MiB.
- Mutants: 8 of 8 KILLED by an assertion failure, with 0 errors (`receipts/mutants_c98cffc56c.json`).
- Consumers (E199): the bounded, fast_compare, compare_fast, score_integrity and release_guards tests pass (51). The
  `wn3_integrity.py` output is identical old vs new (sha 643a88c5). Namespace + `gpuwrf.io.netcdf_lock.Dataset`
  callers are covered by a test. The CLI callers (rolling_cpu_gpu_validate, wn3_redelivery/post_arm, wz1080_matrix,
  v0234_ninenest_score) use unchanged arguments. `v0234_ninenest_static_preflight.py` pins an older comparator sha
  (5099e49d): this was already stale before this change and is left as is.

## Delta 1: non-finite reference values (cr-w42-scorer GATE4, E200; the gap also exists on old main)
`report_gate_failures` now fails every compared numeric field, bounded or not, with kind `nonfinite_cpu` per lead.
`nonfinite_gpu` already existed. `NONFINITE_REFERENCE_EXEMPT` is empty. If an exemption is ever used, it is listed per
domain in `summary.json` as `reference_nonfinite_exempted`.
- [M] The real reports contain no non-finite values: 0 reference and 0 candidate values over 374 fields × all leads ×
  d01–d03, for 0227 24 h and 0420 72 h B/F (`receipts/nonfinite_counts.json`).
- [M] The policy replay gives identical gate failures old vs new, equal to the stored summaries
  (`receipts/policy_replay_finite.json`). The end-to-end 24 h run is byte-identical (`receipts/identity_24h_9990de2f63.json`).
- 16 tests in `tests/test_wn3_score_finite_gate.py`; mutants 11/11 KILLED (`receipts/mutants_finite.json`).
