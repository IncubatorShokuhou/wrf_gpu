# wrf_gpu v0.4.3 sealed compile bundle — how to use it (w43-release, 2026-10-21)

Valid ONLY with these exact absolute paths and this environment (AOT targets pin XLA_FLAGS incl. the autotune path;
JAX cache keys include the cache dir; Pallas kernels embed source paths in the HLO):

- P = <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
- source checkout: P/src (clean, detached at the v0.4.3 source-freeze commit 888804310bb6, src tree 6f98d8d8558a; never
  edit; data/wrf_pristine -> <USER_HOME>/src/wrf_pristine)
- cache: P/jax (+ P/autotune, P/triton, P/cuda beside it)
- environment: `source P/bundle.env` and set NO other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT (release
  defaults). Run under scripts/with_gpu_lock.sh.

## What is primed (release defaults, product CLI `python -m gpuwrf.cli run`)

| arm | input | CLI args |
|---|---|---|
| Swiss d01 | Swiss RD11 case | `--domains-from-namelist --hours 3` |
| PROD 2-domain | PROD s0_case_20260725 | `--domains-from-namelist --hours 3` |
| WN3 3-nest | WN3 20260227_18z_a1, 72 h inputs | `--domains-from-namelist --emit-initial-history --hours 72` |
| WN3 3-nest `icloud = 0` | WN3 20260227_18z_a1 inputs + namelist `&physics icloud = 0` (w42-icloud0 case) | `--domains-from-namelist --emit-initial-history --hours 24` |

Carried over from v0.4.1 (unchanged): date-blind nested programs (any issue date on the same WN3 grid/physics loads the
same programs; the G2 C0 configuration — WN3 namelist, other dates, `--hours 48` — is served by the WN3 arm) and
record-count-blind step programs (48 h and 72 h input files of one geometry hit the same programs).

New in v0.4.3 (identity with the v0.4.2 bundle bfa2dc90, proof `proofs/w43_release/RELEASE_PLAN.md` section 1):
- Model source == v0.4.2 except the version line; the new host-suspend receipt code is outside every compiled program.
- Compiled with the v0.4.2 bundle's per-fusion autotune results copied in first (1,227 entries, 0 added). All 21 AOT
  programs (d01 4, d02 1, fused_d02 2, init LW 7, SW 7) have the same optimized HLO, native cubin (Riegeli record 2) and
  ELFs as v0.4.2; every release-check wrfout (716 files) is byte-identical to the v0.4.2 bundle receipts; a v0.4.2
  replay of 0227 72 h equals the v0.4.2 receipt (219/219). IDENTITY_VERDICT = IDENTICAL -> the v0.4.2 numerical
  validation (RW3 D6, `icloud = 0` D6, teacher tables computed on bfa2dc90) holds for this bundle.
- Programs and keys: same families as v0.4.2 (`icloud = 0` primed as its own programs, as in v0.4.2). The AOT dir is
  version-keyed (`aot/0.4.3-…`), so the v0.4.2 and v0.4.3 bundles do not share files.
- NOT primed: the opt-out flag, other namelist physics/nudging/domain setups (e.g. `gpuwrf optimize` variants, nudging
  namelists, other grids). Those compile on first use; run them through the consumer wrapper below so P is never written.
- Check before a block which programs a case would use (CPU only, no GPU, private temporary cache — never writes P, even
  with P/bundle.env sourced): `PYTHONPATH=P/src/src python3 -m gpuwrf.cli aot-key --input-dir CASE
  [--compare-input-dir OTHER]` (CPU keys carry the CPU target: compare cases with each other).

## Consumer mode (shipped in P/tools; P is never written)

    scripts/with_gpu_lock.sh --label L -- \
      bash P/tools/bundle_consumer.sh --bundle P --layer <your dir outside P> -- <your run script>

The command sees P at the same path; every write lands in `<layer>/upper`. Last stdout line:
`BUNDLE_CONSUMER rc=.. compiled_new=N ...` — `compiled_new=0` means the run used only sealed programs. Reusing one
layer keeps your own extra programs warm. P/tools/SHA256SUMS pins the three tool files.

Several WN3 cases on one GPU: `bash P/src/scripts/run_parallel_cases.sh <case args> --max-parallel 3 --host-gb-per-case 8`
inside the consumer (each case sizes its own C-auto pool). `--host-gb-per-case` is the host-RAM admission per case when no
measured sizing exists yet; warm WN3 cases peak at 6.2–6.35 GB VmHWM (x 1.1 = 7.0 GB), so 8 admits three cases, while
the 16 GB default needs 3 x 16 GB + the 8 GB reserve = 56 GB of MemAvailable plus the running cases' RSS before the third
case starts (often not there on a shared host). Measured with the v0.4.2 bundle (same compiled programs):
3 cases x 24 h in 448.3 s = 6.23 s per case-hour (host <DATA_ROOT> busy at the time, disclosed).

Before AND after every arm (CPU):

    JAX_PLATFORMS=cpu python3 P/tools/compile_bundle.py check --cache P/jax --manifest P/sealed/BUNDLE_MANIFEST.json

-> `BUNDLE_OK missing=0 changed=0 new=0`. The bundle removes codegen variance only; trajectories still need D6
against the original CPU-WRF.
