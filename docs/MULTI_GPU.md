# Several GPUs (experimental)

Status: **experimental, validated on CPU virtual devices only; no multi-GPU hardware run yet.**
Real multi-GPU speed is unmeasured. Single-GPU runs are unchanged: nothing here is active unless you ask for it.

## Many forecasts / ensemble members: one GPU each (recommended)

```bash
scripts/run_parallel_cases.sh --gpus all --out-root runs/ens MEMBER_01 MEMBER_02 MEMBER_03 MEMBER_04 -- \
  --domains-from-namelist --hours 24
```

Every member runs in its own process on one GPU (`CUDA_VISIBLE_DEVICES`, PCI bus order). Each GPU is sized
and admitted separately (the memory plan is per GPU); host RAM is shared and admitted globally. Members on a
GPU that still has room start next to each other, as on a single GPU. One failed member never stops the
others. Options:
- `--gpus 0,1` selects GPUs by `nvidia-smi` index; `--max-per-gpu N` caps members per GPU.
- `--checkpoint-interval-steps N` gives every member its own checkpoint store (`<out>/<member>/checkpoint`).
- `--retry-failed K` re-runs a failed member up to K times, resuming from its newest checkpoint (otherwise from
  scratch; the failed history is kept under `failed-attempt-<k>/`, never overwritten).
- All selected GPUs must be the same model; do not set `CUDA_VISIBLE_DEVICES` together with `--gpus`.

This is the efficient way to use several GPUs for the domain sizes the model is normally run on.

## One large domain over several GPUs (domain decomposition)

```bash
python -m gpuwrf.cli run --input-dir CASE --output-dir runs/big --domain d01 --hours 6 --decompose 2x1
```

`--decompose PXxPY` splits a single native d01 domain (from `wrfinput`/`wrfbdy`) into PX × PY tiles, one per
visible GPU, like WRF's `nproc_x`/`nproc_y`. Columns stay whole. Each tile runs the normal model on its own
columns plus a wide halo (38 columns for 4 acoustic substeps: a safe bound; measured exact from 20). Halos are refreshed from the neighbouring tiles
after every step. Output, restart files and checkpoints are written from the combined domain exactly as for a
single-GPU run.

It pays off only for **large** domains: each tile recomputes its halo, so a tile of W owned columns costs as
much as W + 2 × halo along each split direction (only internal sides). Guidance [I]: ≥ 400 columns per tile along each split axis; smaller
domains should use one GPU (and ensembles use one GPU per member). Nested runs and replay runs are not
decomposed (refused with a message). Options: `,halo=H`, `,k=K` (exchange every K steps, wider halo),
`,transport=ppermute` (NCCL) or `device_put` (default), `,exchange=all` (exchange every field, for debugging).

## Validation and limits
- CPU virtual devices (`XLA_FLAGS=--xla_force_host_platform_device_count=P`): the real model step, decomposed
  against the single-device run (`proofs/o2_multigpu/`): bitwise identical on the 9 km Canary Islands domain (120×70) for halos of 20–25 columns;
  a compiled tile program of another width can differ from the full-domain program at floating-point round-off
  (~1e-15 relative, at the domain edge), never near the tile boundaries.
- Hardware qualification on 2× RTX 4090: `scripts/runpod/RUNBOOK.md` (not yet run).
- Design record: `.agent/decisions/ADR-O2-MULTIGPU-DECOMPOSITION.md`; predicted scaling: `scripts/runpod/cost_model.py`.
