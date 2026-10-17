# Remaining t-icediag gates — authorized future singleton window

Status: blocked on the manager's 12 GiB CPU singleton window, **after** the
20261014_12z ALISIOS nightly verdict is DONE, on 2026-10-15. No GPU work is
requested. The user explicitly deferred operational_smoke after the 6 GiB OOM.
Jensen's previous standalone CPU interpreter also peaked at 9,039,808 KiB;
do not attempt that compile in a 6 GiB slot.

Use the final composed commit (rebase on the P3 fix if the manager merges one),
a detached snapshot with `data/wrf_pristine` linked, and the lane artifact root.
The manager must schedule this **alone**; `--max 12G` by itself does not exclude
another RAM slot holder. Do not raise the global resource limits or hold slots
between commands. Every command has the same-line QUIET guard.

Environment for each arm: `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 GPUWRF_CENSUS=1 GPUWRF_FAST_DEFAULTS=0`, source snapshot `src` on PYTHONPATH,
TMPDIR under `<USER_HOME>/wrf_gpu2_lanes/t-icediag/`. Run through
`scripts/with_ram_slot.sh --label t-icediag --max 12G --timeout 1800 --
taskset -c 16-19 nice -n 19 timeout 1800 ...`.

1. `python proofs/t_icediag/native_oracle.py --mp MP --out OUT.json`, separately
   for MP=50,51,52,53,55. These are CPU Pallas interpreter checks, not GPU
   qualification. P3 checks every retained operational output on three evolving
   original columns, with the pre-existing P348/P360 conditioning bounds.
   Jensen checks new diagnostics against the independent original input-ULP/FMA
   enclosure, and requires old outputs BYTE to the retained native receipt
   before reusing its disclosed experimental deviations. Read the JSON status;
   DISCLOSED_EXCESS/REGRESSION_REVIEW_REQUIRED is not a PASS or a widened bound.
2. `python proofs/t_icediag/coupled_smoke.py --case
   <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu --mp MP --steps 2 --out OUT`,
   separately for MP=50,51,52,53,55. This uses the actual Swiss loader and full
   CPU RK/physics path, new private regular namelist (never a source symlink),
   REAL32 diagnostics, finite carry, timestep handoff and full restart BYTE.
   The script refuses before importing JAX until the specified nightly is DONE.
   GUARD_REVIEW_REQUIRED needs a matched control under the standing policy.

The source/IO gates already passed: 433 distinct light tests; 525 finite
original diagnostic field-frames, 21 active independent writer crops; 75/75
original full CPU files SHA-exact; a genuine swapped DI/rho assertion mutant;
all five expanded native CUDA TTIR frontends; full Swiss/PROD default identities.
Use these source-bound receipts rather than repeating completed gates. New
P3/Jensen GPU execution, E41, runtime/VRAM and full-forecast D6 remain separate
release gates. P3/Jensen radar output is outside this lane's contract.
