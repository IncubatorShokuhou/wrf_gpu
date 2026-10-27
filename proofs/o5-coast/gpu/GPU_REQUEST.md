# o5-coast GPU request — root ring-0 WRF walk (GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk), WN3 0227 3-nest

- Launcher (immutable): `<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/coast_gpu_04cbe43e7e4bedce03ca9b19d1f15e5cb1d97e88.Lf17a691026f7ab40709f41744775946397a3ee89.sh`,
  sha256 `4a1b84101713d9b7cdd690e844c11e14ecd55c23433c8592b2e3a3c494bf212b` (`prepare.sh 04cbe43e7e f17a691026`: launcher +
  tools of commit f17a691026, src = the 04cbe43e7e export). Sequencer-compatible: takes its own GPU lock, then checker
  (`check_arm.py`) and a detached E41 (`scripts/wave_cubin_gate.py`, waits while `/tmp/wrf_gpu2_gpu_window` exists, RAM slot).
  Wrappers (immutable): CUDA mini `cuda_mini_o5coast_f17a6910.sh` sha `5bed26d9…` (= launcher `--mini`: Swiss d01 example,
  1 h, walk, cap 10 min; checker reports the d01 MU ring offset vs our Swiss CPU-WRF at +1 h) and extended production
  `prod_seed_o5coast_f17a6910.sh` sha `9f134b0e…` (MINI,WALK,SEED). Superseded: 0ea908e1, 1fb079d0.
- Source: immutable `git archive` export of lane commit `04cbe43e7e` at
  `<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/src_04cbe43e7e4bedce03ca9b19d1f15e5cb1d97e88` (960 MB, `EXPORT_SHA256SUMS`, read-only;
  `data/wrf_pristine` link). The launcher re-verifies the export and the case-input SHAs before any GPU work.
- Case: `<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/case0227_3nest` = symlinks to our own CPU twin inputs (w43-twin0227) with
  the original reference namelist; all 8 SHAs == teacher batch1 0227 (`case0227_3nest.sha256`). No ALISIOS roots.
- Invocation (manager window): `GA_START_UTC=… GA_END_UTC=… scripts/with_gpu_lock.sh --label o5-coast --timeout <wait> --
  bash <launcher> MINI,WALK` (optional third arm `REL` = released clock, GPU self-twin vs te_gpu).
- Child env (per arm, `env.txt` receipt): `JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1
  PYTHONPATH=<export>/src`, private `GPUWRF_JAX_CACHE_DIR=JAX_COMPILATION_CACHE_DIR=<out>/cache/jax`,
  `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `GPUWRF_XLA_AUTOTUNE_CACHE_DIR`, `TMPDIR` under `<out>/cache`, `OMP_NUM_THREADS=4`,
  `XLA_FLAGS`/`CUDA_ROOT` unset; arm clock `GPUWRF_ROOT_SPEC_STAGE_CLOCK` = walk (MINI, WALK) / 0 (REL).
- Arms and caps (hard cap 45 min per arm, refuses to start an arm that cannot finish before GA_END_UTC):
  MINI = CUDA mini, 3 h, cold compile into the private cache (expected ~12–15 min);
  WALK = 72 h, warm on MINI's cache (expected ~20–25 min); SEED optional = walk on `case0227_3nest_seed` (o5-9nest
  `perturb_theta.py` aa3e25cb: wrfinput T ±1 ulp of fl32 total θ, seed 148+N; launcher re-verifies src/dst SHAs), warm,
  ~25 min → GPU self-twin G = SEED − WALK; REL optional (released clock, cold, ~30 min).
- Scheduled (manager 2026-10-24 17:04Z): MINI+WALK(+REL) in the large 25.10 window; a ≤2-min CUDA mini is not possible
  because every walk step program is a cold compile.
- Disk: raw history streamed through `tools/stream_extract.py` (each published frame extracted to npz, verified finite,
  sha256 logged, then deleted) → peak ≈ cold cache (~3 GB) + a few frames + ~0.6 GB npz per arm; refuses below 370 GiB.
  Cache kept until the E41 cubin scan (`scripts/wave_cubin_gate.py <out>/cache/jax`) after the session, then deleted.
- CPU launch smoke (2026-10-24): `--dry-run --mini` and `--dry-run MINI,WALK,SEED,REL` → every phase started (checks incl. seed SHA chain, CLI `--dry-run` rc 0 per arm, extractor import, checker, E41 with the expected empty-cache refusal; receipts `gpu/dry_receipts/`)
  (plan `DRY_RUN_PLAN_04cbe43e7e.json`: nested_live d01–d03, 72 h, emit_initial_history), extractor import OK;
  extractor delete path smoke-tested on two copied d01 frames.
- Measurements after the session (CPU): d01/d02/d03 MU/PSFC lag fit vs our CPU twin (expect walk ≈ 0 vs released
  +47…+61 s rings 1–9), U10/V10 X_twin(walk) vs X_twin(te_gpu) by lead block and coastal mask, D6-style
  RMSE vs twin, nonfinite census.
