# RunPod 2× RTX 4090 runbook — does the wrf_gpu kernel decompose efficiently?

Lane o2-multigpu (Opus 5.5), 2026-10-10. Status: **experimental; CPU-virtual-rank qualified; no multi-GPU
hardware run yet; F2 census decomposition explicitly unsupported.** Spending/renting is the Owner's decision; this runbook only says what to run and what to expect.
Design: `.agent/decisions/ADR-O2-MULTIGPU-DECOMPOSITION.md`; debugging: `DEBUG_GUIDE.md`.

## 0. Machine
2× RTX 4090 (24 GB, Ada sm_89, PCIe Gen4 x16, **no NVLink, no GeForce P2P** → GPU↔GPU copies are host-staged),
≥ 16 vCPU, ≥ 64 GB RAM, ≥ 60 GB disk, NVIDIA driver ≥ 580 (CUDA 13 wheels; else `JAX_CUDA=12`), Python ≥ 3.11,
`nsight-systems-cli` optional (T5). A RunPod "2x RTX 4090" pod with the PyTorch/CUDA image is sufficient.

## 1. Setup (≈ 10 min)
```bash
git clone <repo> wrf_gpu && cd wrf_gpu && git checkout lane/o2-mgpufix   # or the merged main
bash scripts/runpod/setup.sh ~/wrfgpu_runpod      # venv, jax[cuda13]==0.10.*, -e install, WRF v4.7.1 tables, facts/
source ~/wrfgpu_runpod/env.sh
```
Check `facts/jax_devices.txt` shows 2 CUDA devices and "cross-device copy OK"; `facts/p2p_read.txt` (NS expected).
`facts/wrf_tables.txt` must confirm the pinned Noah-MP submodule, `run/MPTABLE.TBL` link, and all required
Noah-MP/RRTMG runtime tables. No CPU-WRF build is needed.

## 2. Test matrix (`bash scripts/runpod/test_matrix.sh ~/wrfgpu_runpod/results`; ≈ 1–2 h incl. compiles)
| id | what | pass criterion | expected [I] |
|---|---|---|---|
| T1 | single-GPU CLI baseline, Swiss 42×42, 6 h, cold then warm | rc 0, finite, 6 history frames | warm stepping ≈ 1–2 s/fc-h (5090: ≈ 1; 4090 ≈ 1.8× slower by HBM bandwidth) |
| T2 | ensemble: 4 identical Swiss members over 2 GPUs (`run_parallel_cases.sh --gpus all`) | all rc 0, 2 per GPU, identical history SHA across members | wall ≈ 2 member-walls ⇒ ≈ 1.9–2.0× throughput vs one GPU (ensemble efficiency ≥ 0.9) |
| T3a | decomposition correctness: Swiss mirror 4×2 (168×84), 2×1 split vs single GPU, 6 matched steps incl. radiation | strict finite + literal BITWISE; any mismatch returns nonzero and preserves full evidence | GPU release kernels unqualified — this IS the test; round-off mismatches need independent investigation |
| T3b | scaling: Swiss mirror 10×10 (420×420), 2×1, device_put/ppermute, k=1/2, live/all | strict finite + BITWISE; record matched-window speedup/efficiency | with shadow OFF, cost model [I]: single4090 ≈0.26s/step, P2 E≈0.74 at H38, payload≈126MB; enabled shadow needs a larger recomputed bound |
| T3c | same, 1×2 split | finite | ≈ T3b |
| T4 | 840×840 decomposed only (memory use case) | finite, no OOM | per-GPU ≈ 0.5–0.6 s/step |
| T5 | nsys of T3b (6 steps) | trace written | look for: per-tile step kernels overlapping in time on both GPUs; D2H/H2D of the halo message between steps; host gaps |
| T6 | PROD d01 (needs `PROD_CASE` inputs) 2×1 + mirror 4×6 | as T3a/T3b | PROD d01 itself is too small (redundancy 1.6) |

## 3. What to look at
- `T3*/bench_*.json`: `single_gpu.s_per_step`, `decomposed.s_per_step`, `speedup_vs_single`, `parallel_efficiency`,
  `transport_only_s_per_exchange`, `raw_copy_gpu0_to_gpu1_GBps`, `message_bytes`, `compare`.
- Decompose the time: `decomposed.s_per_step ≈ max(tile compute) + exchange`. Tile compute alone ≈ single-GPU
  s/step × (tile columns / domain columns) if per-column cost is size-independent; if the measured tile compute is
  higher, small tiles under-occupy the GPU (report the ratio). Exchange ≈ `transport_only_s_per_exchange / k`.
- Go/no-go for the kernel: efficiency on the 420×420 domain ≥ 0.7 at P=2 ⇒ deep-halo decomposition is viable for
  large domains; < 0.5 ⇒ look at exchange (PCIe) vs tile-compute inflation first (nsys) before concluding.
- Then rerun `scripts/runpod/cost_model.py --bcol <message_bytes/(halo*edge)>` with the measured per-column
  time to calibrate the P = 4/8/16/64 predictions (still [I] — only real multi-GPU nodes measure those).
- Model inputs [I]: 4090 per-column step time = RTX 5090 LW9 0.82 µs × 1792/1008 (HBM ratio); host-staged PCIe4
  12 GB/s effective × 2 hops, 15 µs latency; payload 7.9 kB/column measured (release carry, live leaves, CPU).
  Predictions for other domains/GPUs/fabrics: `python3 scripts/runpod/cost_model.py` (writes all rows).

## 4. Rules
Fresh output dirs per attempt (history files are never overwritten). Do not edit result files. Send back the whole
results directory. Never report a speedup without the matching single-GPU number from the same machine.
Both benchmark arms time the same start/stop/cadence from restored equivalent carries; compile/warmup is
separate. Finiteness is checked after timing even for T4/no-single. `matrix_status.json` records every arm;
the shell exits nonzero if any arm fails while still running later independent arms. T2 requires exactly four
successful member receipts and matching, nonempty history sets (at least six frames each).

To recalculate the added shadow traffic, redundant work and memory, use
`python3 scripts/runpod/cost_model.py --shadow-length-m 25000 --dx-m <actual-case-dx>`.
For dx100m the k1 bound is H290; clipping may make every tile span the whole domain, so these configurations
can have no compute scaling benefit. Predictions with an override below the bound are explicitly labelled.
For the 420×420/P2/4090 host-staged model, shadow-off H38 predicts E=0.744, 126MB/message and
1.91GiB/tile; dx100m/shadow25km H290 predicts E=0.345, 696MB/message and 3.24GiB/tile [I].
These are the same uncalibrated bandwidth/compute/memory assumptions; actual GPU figures remain unknown.
