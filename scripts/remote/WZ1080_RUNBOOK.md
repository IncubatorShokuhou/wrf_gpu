# wz1080 test box — runbook (proposal for a kernel skill; manager owns the final text)

**What:** the Owner's second machine `ssh wz1080` (Fedora, GTX 1080 Ti = Pascal sm_61, 11 GB, older CPU).
Use it for **debugging, repros, CPU/GPU tests** so you don't have to wait for the local GPU/CPU. **Never** use it for
final gates, benchmarks, S1/speed claims or fidelity verdicts. Its numbers are regression/debug evidence only, and
must be labelled `wz1080` (different GPU, different compiler target, no cache sharing with the RTX 5090).
Box facts and the capability matrix: section "Capability matrix" below (filled after setup).

## Daily use (from any local worktree; until merged call `<USER_HOME>/src/wrf_gpu2_wt/wz1080/scripts/remote/...`)
The box is shared by all lanes while ALISIOS holds the local GPU (Owner 2026-10-15: all CPU+GPU ours until v0.4.0).
```bash
scripts/remote/wz1080_sync.sh <commit|branch>          # bundle over ssh (no credentials) -> ~/src/wrf_gpu2_wt/<sha12> on the box
scripts/remote/wz1080_run.sh --tree <sha> --label <lane> -- python -m pytest tests/test_x.py -q --basetemp '$WZ_RUN_DIR/pt'
scripts/remote/wz1080_run.sh --gpu --tree <sha> --label <lane> --timeout 1800 --env GPUWRF_FAST_DEFAULTS=0 -- python my_probe.py
scripts/remote/wz1080_run.sh --gpu --bg ... -- <long job>;  ... --status | --tail <label> | --kill <label>
```
- `--cpu` (default) forces `JAX_PLATFORMS=cpu`, hides CUDA, `nice` (E50: nothing touches the GPU outside the lock).
- `--gpu` runs under the box's lock `wz-gpu-lock` (`/tmp/wrf_gpu2_gpu.lock`, same holder/token/fd proof the product
  preflight checks); one GPU job at a time, **FIFO queue** (tickets in `/tmp/wz_gpu_queue`), manager priority =
  `echo <label> > /tmp/wrf_gpu2_gpu.next` on the box (< 1 h old, consumed on acquire). `--lock-wait` bounds the queue,
  `--timeout` the job (always set it). `--status` shows holder, queue and background jobs.
- Always pass `--tree <sha>` (your synced snapshot) and `--label <lane>`; never rely on the box's main clone.
- Trees are immutable detached snapshots with real commit SHAs (sync verifies the tree SHA). `--main` moves the box's
  main clone instead. The sync keeps the newest 10 snapshots (`--keep N`). `data/wrf_pristine` and the git-ignored test
  fixtures are added automatically.
- Env on the box (`~/src/wrf_gpu2_env/activate.sh`): venv (Python 3.13, jax/jaxlib 0.10.0 + CUDA-12 plugin),
  `PYTHONPATH=<tree>/src`, `GPUWRF_WRF_ROOT=~/src/wrf_pristine/WRF`, JAX cache `~/wrf_gpu2_cache/jax`,
  `GPUWRF_MIN_FREE_VRAM_GIB=4` (11 GB card), `FC` = conda-forge gfortran 14.3 if the toolchain is installed.
- Artifacts: `$WZ_RUN_DIR` = `~/wrf_gpu2_runs/<label>/` on the box. Delete them when done; the disk is the Owner's.
  Experiments with their own cache: `--env GPUWRF_JAX_CACHE_DIR=/home/wz/wrf_gpu2_runs/<label>/cache --env
  JAX_COMPILATION_CACHE_DIR=...same` (E23) and delete it after the last dependent arm; the shared default cache is
  `~/wrf_gpu2_cache/jax` (the box compiles for sm_61: never comparable to or shareable with RTX 5090 caches).
- References on the box: Swiss 24 h original CPU-WRF `~/src/wrf_gpu2_data/refs/swiss_rd11_cpu` (D6 with
  `scripts/compare_wrfout_grid.py --init 2023-01-15T00:00:00`).

## GPU arms on the box — exact recipe
`wz1080_run.sh --gpu --tree <sha> --label <lane> --timeout S --env GPUWRF_FAST_DEFAULTS=0 --env WZ_PALLAS_PORTABLE=1 -- <cmd>`
(legacy XLA path + box-only Pallas-interpret shim; the only practical full-model path on sm_61). The product refuses
GPUs below sm_80 (exit 69, `runtime/gpu_preflight.py`); the box's `activate.sh` sets `GPUWRF_ALLOW_UNSUPPORTED_GPU=1`,
so runs print "UNSUPPORTED, SLOW" and continue. Results are debug evidence only (not release paths, not speed).

## Pascal caveats (see matrix for measured status)
- CUDA 13 dropped Pascal → the box uses `jax-cuda12-plugin==0.10.0` (pip CUDA 12, ptxas 12.x) with the same jax 0.10.0.
- Pallas/Triton: XLA refuses Triton below Ampere (string in the plugin: "Triton support is only enabled for Ampere
  GPUs (compute capability 8.0) and up") → release fast paths (Pallas kernels) cannot compile; use
  `GPUWRF_FAST_DEFAULTS=0` (XLA legacy paths) or CPU interpret tests. Pallas-specific bugs need the RTX 5090.
- fp64 is 1/32 of fp32 (5090: 1/64) → legacy fp64 paths are slow but fine for debugging.
- The Owner may use the box: GPU jobs short, always `--timeout`, nothing long while the Owner works there.

## One-time setup (done by lane wz1080; idempotent)
`scripts/remote/wz1080_bootstrap.sh ssh tools inventory python jax-cpu activate data repo`, then `jax-cuda`
(after the inventory shows an R580-or-older driver), optional `toolchain` (gfortran/netCDF for pristine oracles).

## Capability matrix [M on wz1080, 2026-10-15, main a0e061fd9]
Box: Ryzen 5 5500 (6C/12T; SMT pairs N/N+6), 31 GB RAM, /home btrfs 830 GB free, GTX 1080 Ti 11 GB cc 6.1, driver 580.178.04,
Fedora 44, no sudo. Env: CPython 3.13.13, jax/jaxlib 0.10.0 + jax-cuda12-plugin 0.10.0 (ptxas 12.9). Core plan: GPU-lock
holder 0-2,6-8, lane CPU jobs 3-5,9-11 (`--env WZ_CPU_CPUS=` / `WZ_GPU_CPUS=` to unpin a timed run).

| capability | status |
|---|---|
| JAX XLA on GPU (jit, x64, while/cond command buffers, cuSPARSE gtsv) | OK (stream 332 GB/s, GEMM f32 8.7 / f64 0.38 TFLOPs, f32/f64 FMA 21×) |
| Pallas/Triton kernels | **IMPOSSIBLE**: XLA `FAILED_PRECONDITION: Triton … only enabled for … compute capability 8.0 and up` |
| release defaults (CLI) | **cannot run** (Pallas); `WZ_PALLAS_PORTABLE=1` shim compiles (~8 min) but > 360 s/fc-h on Swiss → impractical |
| legacy path `GPUWRF_FAST_DEFAULTS=0` alone | fails (legacy still has Pallas kernels) |
| **legacy + shim** (`--env GPUWRF_FAST_DEFAULTS=0 --env WZ_PALLAS_PORTABLE=1`) | **works**: Swiss d01 3 h cold 220 s/fc-h (compile ~9 min), warm 33.2 s/fc-h, D6 vs CPU-WRF PASS |
| original CPU-WRF (reference binary + env + tables) | works, bitwise vs RD11; Swiss np6 9.6 s/fc-h (`probes/swiss_cpuwrf.sh`) |
| CPU pytest, whole `tests/` (`JAX_PLATFORMS=cpu`, 1 process, 6 cores) | runs, 3 h 35 min (tree a0e061fd9): 16,444 pass / 213 fail / 393 error / 758 skip. Of the 606: 189 hard-coded `<USER_HOME>` paths, 277 git objects absent (tests pin non-main commits; the sync ships main history only), 23 `<DATA_ROOT>` inputs, 21 proof/Fortran build scripts, **50 Pallas-facade order leak** (whole-suite only, fixed test-side in 3905ed2c1b), 46 other (untriaged, old tree). Use the box for targeted files, not as a whole-suite verdict |

Good uses: CPU tests/oracles in parallel with the local box, legacy-path GPU repros, XLA/HLO/jaxpr questions, CPU-WRF runs
(gfortran 14.3 + OpenMPI in `~/micromamba/envs/wrf-build`, `FC` set). Not possible here: anything that needs a compiled
Pallas/Triton kernel (release fast paths, cubin/E41 gates, Pallas speed) → local RTX 5090.
