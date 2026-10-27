# rrtmg_precision_gate overlap fix — receipts (lane/o5-port, for branch lane/o5-port-radgate 5c153794e4)

Candidate = main c9ecb56056 + tests/v025/b_phys/{rrtmg_precision_gate.py (blob 4d0e3a2b), test_rrtmg_precision_gate_overlap.py
(blob b87fc364)}; tree 557fbde5. Box snapshot = `git archive` of the tree (wz1080:/home/wz/src/o5radgate_2679e7f, TREE_PROVENANCE).

- `radgate_oracle_cpu3.receipt.json` — gate `--mode oracle --backend cpu`, release defaults (MAXRAND 1 imported), box CPU: rc 0.
- `radgate_oracle_gpu2.receipt.json` — same gate file on the GTX 1080 Ti, CROSS-TREE (lane/o5-port dd51631c51 src = main b14a9801cb
  + facade; main's later src changes do not touch RRTMG except default-OFF constfold), GPUWRF_PRE_AMPERE_KERNEL_CALL=1: rc 0.
  Main itself cannot run on Pascal (no pre-Ampere route on main).
- `tests_mutants_cpu.txt` — pytest (5 new + 3 b_rad maxrand_re01) and mutants M1–M6 (`run_mutants.py`, restores the gate by sha).
- `clear_probe_cpu.txt` (`clear_probe.py`) — LW driver truth under both overlaps: all-sky fails only with max-random (47.98 W/m²),
  clear-sky passes under both.
