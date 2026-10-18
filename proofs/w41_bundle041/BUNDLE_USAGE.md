# wrf_gpu v0.4.1 sealed compile bundle — how to use it (w41-bundle041, 2026-10-18)

Valid ONLY with these exact absolute paths and this environment (AOT targets pin XLA_FLAGS incl. the autotune path;
JAX cache keys include the cache dir; Pallas kernels embed source paths in the HLO):

- P = <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1
- source checkout: P/src (clean, detached at the v0.4.1 source-freeze commit; never edit;
  data/wrf_pristine -> <USER_HOME>/src/wrf_pristine)
- cache: P/jax (+ P/autotune, P/triton, P/cuda beside it)
- environment: `source P/bundle.env` and set NO other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT (release
  defaults). Run under scripts/with_gpu_lock.sh.

## What is primed (release defaults, product CLI `python -m gpuwrf.cli run`)

| arm | input | CLI args |
|---|---|---|
| Swiss d01 | Swiss RD11 case | `--domains-from-namelist --hours 3` |
| PROD 2-domain | PROD s0_case_20260725 | `--domains-from-namelist --hours 3` |
| WN3 3-nest | WN3 20260227_18z_a1, 72 h inputs | `--domains-from-namelist --emit-initial-history --hours 72` |

New in v0.4.1:
- The nested programs are **date-blind**: any other issue date on the same WN3 grid and physics loads the same
  programs (the fused d02/d03 cascade takes its namelists as arguments; v0.4.0 baked Noah-MP TMN into the d02/d03
  program). The G2 C0 configuration (WN3 namelist, other dates, `--hours 48`) is served by the WN3 arm.
- The step programs no longer depend on the **record count** of wrfbdy/wrflowinp: 48 h and 72 h input files of one
  geometry hit the same programs.
- NOT primed: other namelist physics/nudging/domain setups (e.g. `gpuwrf optimize` variants, nudging namelists, other
  grids). Those compile on first use; run them through the consumer wrapper below so P is never written.
- Check before a block which programs a case would use (CPU only, no GPU): `PYTHONPATH=P/src/src python3 -m gpuwrf.cli
  aot-key --input-dir CASE [--compare-input-dir OTHER]` (CPU keys carry the CPU target: compare cases with each other).

## Consumer mode (shipped in P/tools; P is never written)

    scripts/with_gpu_lock.sh --label L -- \
      bash P/tools/bundle_consumer.sh --bundle P --layer <your dir outside P> -- <your run script>

The command sees P at the same path; every write lands in `<layer>/upper`. Last stdout line:
`BUNDLE_CONSUMER rc=.. compiled_new=N ...` — `compiled_new=0` means the run used only sealed programs. Reusing one
layer keeps your own extra programs warm. P/tools/SHA256SUMS pins the three tool files.

Before AND after every arm (CPU):

    JAX_PLATFORMS=cpu python3 P/tools/compile_bundle.py check --cache P/jax --manifest P/sealed/BUNDLE_MANIFEST.json

-> `BUNDLE_OK missing=0 changed=0 new=0`. The bundle removes codegen variance only; trajectories still need D6
against the original CPU-WRF.
