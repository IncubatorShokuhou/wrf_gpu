# icloud=0 WN3 arm for the v0.4.2 bundle session (w42-icloud0 → w42-bundle042)

Manager 2026-10-19T06:05:46Z: the repeat of the `icloud=0` GPU forecast on the FINAL composed candidate runs once
inside w42-bundle042's GPU session and is scored there. This is the exact arm of the morning run (tree
`e0041902bd`, D6 PASS, see `README.md`), so the two runs are directly comparable.

## Inputs (do not edit, do not delete before the bundle arm is scored)

Case root `CASE=<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/wn3_icloud0`:

| item | path | sha256 |
| --- | --- | --- |
| namelist (WN3 0227 twin, `icloud = 0`, run_hours 24, restart_interval 100000, nproc 2×2) | `CASE/input/namelist.input` | `2c15e10b51d7…3f80` |
| model inputs (copies of w41-icloud2's guarded stage, 0 new ALISIOS reads) | `CASE/input/{wrfinput_d01..03, wrfbdy_d01, wrflowinp_d01..03}` | `CASE/STAGE.json` (verdict OK, equal to the icloud=3/2 twins) |
| original CPU-WRF 4.7.1 `icloud=0` reference (gen2 dmpar `wrf` sha `1d0f5779`, 4 ranks 2×2) | `CASE/cpu/wrfout_d0*` (75 files) | `CASE/CPU_OUTPUTS.json` `41f257b1…05b`, `CASE/cpu_run.json` `31534bec…cb7` |

## GPU arm

Tree = the bundle pin; it must bind `icloud=0` (`io/namelist_binding.py`: `if icloud not in (0, 2, 3)`) and carry the
default SWDDIR fix (`kernels/rad_sw_band_sums.py`: `dn * bn` twice; `GPUWRF_SWDDIR_V040_SPLIT` unset).

```bash
OUT=<bundle lane dir>/icloud0            # fresh, never reuse (E42)
python -m gpuwrf run --input-dir "$CASE/input" --output-dir "$OUT/gpu" --proof-dir "$OUT/gpu_proof" \
  --scratch-dir "$OUT/gpu_scratch" --hours 24 --domains-from-namelist --emit-initial-history
```

Morning env (proofs/w42_icloud0/gpu_arm.sh): `JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_RRTMG_MP_RE=1
GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0` (rich AOT modules so E41 sees the kernels), private
`GPUWRF_JAX_CACHE_DIR`/`JAX_COMPILATION_CACHE_DIR`/`GPUWRF_XLA_AUTOTUNE_CACHE_DIR`/`TRITON_CACHE_DIR`/`CUDA_CACHE_PATH`,
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30
GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=4`, own 20G memcg. If the bundle session uses its consumer/C-auto pool
instead, record the env; `icloud=0` is its own program (own cheap keys, cold compile or its own bundle entry).
Morning cost: 21 GPU-min incl. cold compile (first hour 855 s), warm k2–24 median 6.76 s per sim-hour (3 domains,
contended host, disclosure only).

E41 before any cache cleanup (E166/E177/E201):
`JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python scripts/wave_cubin_gate.py <jax_cache> <out> --run-proof "$OUT/gpu_proof/nested_pipeline_run.json"`
(morning: PASS on 4/4 entries = d01 + fused d02/d03 JIT and AOT, max STACK 32 B, no conservative flags).

## D6 (CPU, one domain at a time)

```bash
S=$OUT/score; mkdir -p "$S"
ln -s "$CASE/input" "$S/input"; ln -s "$CASE/cpu" "$S/cpu"; ln -s "$OUT/gpu" "$S/gpu"
cp "$CASE/CPU_OUTPUTS.json" "$CASE/cpu_run.json" "$S/"
for d in d01 d02 d03; do   # sequential, <= 12G RAM slot each
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<pin>/src python <pin>/proofs/v040/final_validation/score.py <pin> "$S" --domains $d
done
JAX_PLATFORMS=cpu python <pin>/proofs/w42_icloud0/icloud0_report.py "$S" "$S/ICLOUD0_report.json"
python3 <pin>/proofs/w42_icloud0/d6_summary.py "$S" "$S/D6_SUMMARY.json"
```

Acceptance: `D6_d0{1,2,3}.json` verdict PASS (frozen manifest `9df4fab8…`, 25 paired frames, 0 failures, 0
non-finite, coverage complete); `ICLOUD0_report.json` `cldfra_zero_both_arms_all_forecast_frames` true on all three
domains; E41 PASS. Compare `D6_SUMMARY.json` with the morning run (`fixtures/d6_summary_morning.json`).
