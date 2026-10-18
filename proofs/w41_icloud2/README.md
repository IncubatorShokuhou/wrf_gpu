# w41-icloud2 — GPU coupled-forecast qualification of `icloud=2` (`cal_cldfra2`)

The `icloud=2` port (WRF 4.7.1 `cal_cldfra2`, binary `qc+qi > 1e-6`, replacing the whole `icloud=1` block
including the `icloud_bl` MYNN merge) was merged in v0.4.1 as CPU/column-qualified (`proofs/w41_icloud`).
This lane runs the one missing gate: a 24 h coupled GPU forecast against an original CPU-WRF run of the same
namelist, frozen D6 on every domain, and the E41 cubin gate on the actual executables.

## Case and arms

- WN3 0227 Canary 3-nest (9/3/1 km), the `icloud=3` twin case of `proofs/w41_icloud`: the same seven model inputs
  (byte-identical SHAs) and the same namelist except `icloud = 2` (`stage_inputs.py` proves a one-line diff to the
  `icloud=3` twin namelist; the twin itself differs from the ALISIOS original only in `run_hours` 72→24,
  `restart_interval` 180→100000 and the `icloud` line).
- Inputs read only through `alisios_guard.facade_stage` (one guarded session, 8 authorized reads, 0 refused;
  guard days 20260227–20260303 ALLOW) into a private stage outside the ALISIOS roots.
- Original CPU-WRF: gen2 dmpar WRF 4.7.1 (`wrf` sha 1d0f5779, the `icloud=3` twin's executable), 4 ranks 2×2.
- GPU: release defaults, the `icloud=3` twin recipe (`gpu_arm.sh`), RTX 5090, one lock session.

## Tools

| file | role |
| --- | --- |
| `stage_inputs.py` | guarded staging + twin-identity checks (`--scan-only`, `--finalize` from the private stage) |
| `cpu_ref.sh` | original CPU-WRF reference via `../w41_icloud/cpu_reference.sh` in a RAM slot; reports WRF's own rc |
| `gpu_arm.sh` | GPU arm (lock, own 20G memcg, run + `gpu_cost2.py`), E41 scan before cache cleanup; posts GPU END on every path |
| `gpu_cost2.py` | per-call cost of `wrf_cal_cldfra2` at the three domain shapes |
| `post.sh` | frozen-D6 `score.py` per domain + `../w41_icloud/cldfra_report.py` (incl. `nonbinary_count`) + receipts |

## Results

Run 1, snapshot `2aec723cf5` (src = main `6933b344a3`) [M]:

- GPU: rc 0, 75 wrfout, all finite, 24 GPU-min, forecast wall 1092 s, warm 6.86 s per sim-hour; E41 PASS on all
  4 cache entries (d01 + fused d02/d03), max STACK 32 B, no conservative flags; `cal_cldfra2` 0.03 ms per call and
  domain, no scratch.
- CPU-WRF `icloud=2`: SUCCESS COMPLETE WRF, 75 wrfout, 9989 s on 4 cores.
- Frozen D6 (manifest 9df4fab8), 24 h, PASS d01/d02/d03: 25/25 frames each, 0 failures, 0 non-finite, all 375
  CPU history fields present. Worst-lead RMSE d03: T2 0.092 K, PSFC 3.5 Pa, U10/V10 0.27/0.31 m/s, RAINNC
  0.033 mm; d01/d02 T2 0.010/0.017 K.
- `CLDFRA` (no D6 bound): binary in BOTH arms (0 values other than 0/1 in any frame); +24 h cloudy share CPU/GPU
  0.68/0.68 % (d01), 1.11/1.10 % (d02), 4.10/4.04 % (d03), pattern correlation 0.94–0.96.

Run 2, snapshot `869bf9cd79` = merged main `93346f7979` (incl. the w41-swddir default-path SW fix and the
bdyrec record-count fix; default ids Swiss b977712fe325 / PROD 321aa651067f) + the catalog/docs flip — the
binding run, because run 1's program is not bit-identical to the shipped one [M]:

- GPU: rc 0, 75 wrfout, all finite, 26 GPU-min, forecast wall 1210 s, warm 7.14 s per sim-hour (host contended
  by this lane's concurrent CPU checks; fidelity run, timing not a claim); E41 PASS 4/4, max STACK 32 B, no flags.
- Same original CPU-WRF reference (unchanged executable, inputs and namelist).
- Frozen D6 24 h PASS d01/d02/d03: 25/25 frames, 0 failures, 0 non-finite, 375/375 fields. Worst-lead RMSE d03:
  T2 0.084 K, PSFC 3.69 Pa, U10/V10 0.24/0.29 m/s, RAINNC 0.029 mm; d01/d02 T2 0.010/0.015 K.
- `CLDFRA` binary in both arms; +24 h cloudy share 0.68/0.67, 1.11/1.11, 4.10/4.05 %, correlation 0.93–0.95.

Receipts: `<USER_HOME>/wrf_gpu2_lanes/w41-icloud2/` (`stage/`, `gpu_arm_run1/`, `wn3_icloud2_receipts_run1/`, `gpu_arm/` + `wn3_icloud2_receipts_run2/` = run 2).
