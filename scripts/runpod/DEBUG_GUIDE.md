# DEBUG_GUIDE — multi-GPU wrf_gpu on real hardware (for the on-site debugger)

Code: `src/gpuwrf/parallel/` (`decomposition.py` partition/slicing/exchange plan, `tile_driver.py` runtime +
transports, `synthetic.py` mirror tiling, `harness.py`), `scripts/parallel_cases.py` (`--gpus` ensemble mode),
ADR `.agent/decisions/ADR-O2-MULTIGPU-DECOMPOSITION.md`. CPU proofs: `proofs/o2_multigpu/`.

## 1. How the decomposition works (read first)
- Every GPU holds one **tile** = owned columns + a **deep halo** of `H` columns on each internal side, and runs
  the UNMODIFIED release step (`operational_mode._advance_chunk_fori`) on it. The step thinks every tile edge is
  a domain edge (it applies specified/relaxation zones, edge padding, degraded advection there). That garbage
  needs a selected-configuration dependency bound `H >= k*(R + S) + bdy_width + 1`:
  R = 24 + 1 + ns//2 + ns + 1 = 32 for ns=4; S = `int(shadow_length/dx+1)+1` when
  both slope_rad/topo_shading are enabled, else0. Radiation precedes dynamics, so the radii ADD.
  `configuration_halo()` gives H=38 with shadow off and H=290 for dx100m/shadow25km/k1.
  This is a candidate bound for the audited release path; GPU partition qualification is still pending.
- After every k steps each tile sends, to every tile whose halo overlaps its owned region, ONE uint8 message with
  all halo rectangles of all exchanged leaves (`exchange_plan`: `dst.range ∩ src.owned` per staggering; corner
  and multi-hop neighbours included). The receiver writes them into its halo at the start of its next program.
- Per tile and interval exactly ONE jitted program: `unpack → k steps → pack`. Halo leaves: `--exchange live`
  (default) = only carry leaves the step reads (jaxpr DCE, `release_step_live_leaves`); `all` = every field.
- True domain edges: each tile's `*_bdy` strips are re-sliced by its offset (W/E along y, S/N along x).
- CLI admission allows only a single leaf domain. `domain_tree.integrate` advances such a domain in
  output-alarm chunks, so full-global scatter/gather happens at those boundaries. `TileRuntime.advance`
  itself exchanges halos and carries traced device clocks; it adds no host control-scalar transfers per step.
  Tree/run caches use weak identity ownership (the actual DomainTree is frozen); reset/GC releases buffers.

## 2. Known risks on 2x RTX 4090
| risk | symptom | check / fix |
|---|---|---|
| GeForce has **no PCIe P2P** | GPU→GPU copies go D2H+H2D; `raw_copy_gpu0_to_gpu1_GBps` ≈ 5–12 | `nvidia-smi topo -p2p r` = NS. Expected; compare `--transport ppermute` (NCCL SHM path) vs `device_put` |
| NCCL hangs / slow init | `ppermute` run stalls at first exchange | `NCCL_DEBUG=INFO`; try `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0`; keep `device_put` as fallback |
| CUDA 13 wheels vs driver | `jax.devices()` shows CPU only / CUDA init error | `nvidia-smi` driver ≥ 580 for cuda13; else `JAX_CUDA=12 bash setup.sh` |
| VRAM preallocation | 2nd process / 2nd tile OOM | one process owns both GPUs; default JAX prealloc 75 % per GPU is fine for ONE process. For `run_parallel_cases.sh --gpus` the C-auto plan sizes each case |
| per-tile compile time | first interval takes minutes | 2 programs (one per tile) + the single-GPU baseline; second run hits the JAX cache (`GPUWRF_JAX_CACHE_DIR`) |
| ptxas ICE (E103/B48) | `C7907` during compile | `export CUDA_ROOT=<a 13.1 toolkit>` or report the fusion name |
| small tiles slower per column | decomposed s/step > model | GPU occupancy; check the 1x1 per-column time vs the tile size (cost model assumes linear) |
| mixed compute capabilities | refused by launcher | devices must have the same compute capability (`compile_cache._cuda_tag` keys by GPU 0); identical GPU models are not checked |

## 3. Bisecting a mismatch (decomposed vs single GPU)
`bench_*.json → compare.mismatch_rows` lists every differing leaf with `diff_x_range/diff_y_range`.
1. **Where?** Differences only near global x = owned boundary ± H → halo too small or a message missing.
   Rerun with `--halo 60` (larger). If it disappears: the dependency radius on this path exceeds the formula —
   find the minimal H that passes and report it (the formula lives in `tile_driver.required_halo`).
2. **Exchange set?** Rerun with `--exchange all`. If that fixes it, the live-leaf DCE missed a leaf (report which).
3. **Only at true domain edges?** Boundary strips: check `decomposition._slice_bdy` offsets (W/E use y offset).
4. **Tiny (≲1e-13 rel), not near the internal boundary?** Not a halo bug: per-shape compiled arithmetic. On CPU
   (measured, PROD d01): tiles 80/85 columns wide are bitwise, a 98-wide tile differs at ~1e-15 rel ONLY in the
   last 8 columns of the true domain edge (cause not isolated: elementwise transcendentals and fused arithmetic
   are bitwise at all widths). On GPU expect more of this (autotuned fusions per shape, E99). Check: the
   `--px 1 --py 1` decomposed run (tile == domain) must be bitwise vs the single-GPU run (same program); if not,
   the trip count is not traced (see tile_driver step_fn) or the compile differs.
5. **Halo width.** Default comes from the selected configuration, including shadow-before-dynamics. Measured on CPU (PROD d01,
   one ordinary step, legacy XLA path): exact from H = 20 (bitwise), round-off only at 15, localized errors at
   ≤ 10. These historical halo20 measurements do not qualify release GPU kernels. CLI refuses halos below
   the configuration bound. Benchmark `--unsafe-halo-probe --halo 20` is a negative diagnostic, suppresses
   speedup claims, and returns nonzero for ANY byte mismatch/nonfinite value. Do not reinterpret it as a pass.
6. **NaN** in a tile: run the tile's program alone (TileRuntime.steps[r]) after `jax.config.update("jax_debug_nans", True)`.
CPU reproduction of anything: `proofs/o2_multigpu/virtual_rank_check.py` (virtual CPU devices, same code path;
`XLA_FLAGS=--xla_force_host_platform_device_count=P JAX_PLATFORMS=cpu`).

Census-enabled decomposition is explicitly refused. Tile-reduced guard counts, first-event indices and
water budgets include halos and cannot be gathered correctly without model ownership masks. These lab
partition checks provide no F2/release qualification. Never disable census to claim a qualified forecast.

## 4. Where each halo/edge concern lives
| concern | code |
|---|---|
| owned/face ownership (last tile owns face nx) | `DecompositionSpec.owned_x/owned_y` |
| tile ranges, H clipping at true edges | `DecompositionSpec.build` |
| corner + multi-hop messages | `exchange_plan` (pairwise, all tiles) |
| message packing / bitcast to bytes | `tile_driver._pack/_unpack/_to_bytes` |
| transports | `DevicePutTransport`, `PpermuteTransport` (one round per neighbour offset) |
| static per-tile fields (terrain, metrics, Noah-MP, radiation, GWDO flat) | `slice_namelist` (+ audit refuses leftovers) |
| boundary strips | `_slice_bdy` |
| synthetic large domains | `synthetic.mirror_tile_case` |

## 5. What to send back
The whole results directory (`matrix.log`, `T*/bench_*.json`, `T2/*/receipt.json`, `facts/`, nsys `.nsys-rep`),
plus `pip list` and `nvidia-smi -q | head -80`. Never edit result files in place.
