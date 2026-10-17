# Soil/snow cycling qualification

`gpuwrf cycle` transplants exact-time, same-grid land state into a separate
wrfinput and records provenance. [Usage and field contract](../../docs/cycling.md).
The opt-in Noah-MP input hook retains evolved soil, snow and canopy state after
ordinary cold-input defaults. It changes no timestep or persistent carry schema.

## Critic correction: column consistency

The independent review found four internally hash-consistent but physically
invalid columns admitted by the original candidate. The correction validates
active/inactive snow geometry, strictly positive active snow/soil thicknesses,
soil thickness against DZS, layered SWE against SNOW and layered depth against
SNOWH. It runs on the host before publication and before any restore assignment,
without repairing data. [Pristine arithmetic bounds](COLUMN_BOUNDS.md) derive
the REAL rounding allowances; thin snow with ISNOW=0 remains supported.

The changed artifact passed 162 tests, then 18 selected checks covering an
explicit pre-mutation assertion and four new rounding controls: 166 unique
cycling tests on the same computational source. All four critic controls and
three additional error types are refused at both boundaries on literal and
original CPU-WRF copies, including resealed invalid markers. Empty/thin and
one/two/three-layer positive controls preserve bytes. Both admission-deletion
mutants are killed by genuine assertions, with zero setup errors.

Rebase checks passed 95 tests, including `cycle` and `perturb` CLI paths; another
five initializer/CLI checks passed after retaining main's two-way initialization
overrides. The transfer and device-restore operations remain AST-identical
apart from the added host checks. Unaffected prior forecast/shared/identity
validation is carried forward. [Correction seal](receipts/column_fix.json).

The complete corrected offline command measured 0.191–0.197 s across the three
Swiss frames, at 119.28 MiB peak RSS, with JAX still unloaded. Host column arrays
are the only added work; no GPU operations or carry fields were introduced.
This is an absolute measurement, not a matched speedup claim.
[Correction timing](receipts/column_timing.json).

## Original qualification and retained reference evidence

The scientific reference is original CPU-WRF, not a previous GPU port. The
[CPU receipt](receipts/original_cpu_receipt.json) identifies the executable,
namelist and retained Swiss hourly outputs. The mandatory tracked reference is
`tests/fixtures/cycling/swiss_cpu_land.npz`: 58,384 bytes, 204 unmodified arrays,
including snow, glacier and vegetated cells at hours 1–3. The adjacent manifest
records whole-file and per-array hashes. New-init templates explicitly retime
the shipped Swiss wrfinput; they are not independent new real.exe analyses.

The final composed candidate passed **330 tests** (CY13), including the 131
cycling/original-initializer tests and every affected fast shared consumer.
Those cover independent literal array copies, both REAL32/REAL64 loaders,
source/template NaN/Inf refusal, grid/category/layer mismatches, exact-time
selection, provenance tampering, inactive water/lake/glacier cells, atomic
publication, a JAX-dark CLI, and the tracked fixture from a clean git archive.
The complete shared physics smoke ran in CY04: 361 passes plus one own test
setup failure (a dummy state was accessed before the tampered field). That test
was corrected and passed in the final suites. No shared consumer failed.

Both genuine deletion mutants were caught by assertions, with zero setup
errors: removing the transfer breaks the literal TSK reference; removing the
loader restore breaks original CPU-WRF SH2O bytes. See [mutants](receipts/mutants.json).

One full Swiss operational CPU step (44 levels, 42×42 cells, dt=18 s) from
cycled state produced **zero nonfinite values and zero guards**; soil temperature
advanced by 0.003947281 K. This is a legacy CPU-path smoke (FAST_DEFAULTS=0),
not GPU qualification or a trajectory fidelity claim. The later device-select
restore preserved all 117 land/static leaves byte for byte in both precisions,
allowing that smoke to be carried forward without another numerical run.
The parity proof is regression evidence; the independent fidelity reference
remains CPU-WRF. [Smoke](receipts/coupled_smoke.json), [carry parity](receipts/carry_identity.json).

The final native-facade Swiss release jaxpr remains
`2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.
Main's late BEP/BEM merge changes opt-in urban paths; this lane's code and
Noah-MP input loader stayed byte-identical across the rebase. Main's existing
urban-off program-identity proofs, the fresh composed release identity, and
the final source binding justify reusing CY04's completed physics smoke.
[Source binding](receipts/final_source_binding.json), [identity](receipts/release_identity.json).

The complete offline operation measured 0.301/0.316/0.366 s across the three
Swiss frames and peaked at 118.86 MiB CPU RSS, without importing JAX. It includes
validation, double input hashing, full template copying, writing and fsync;
the OS page cache was uncontrolled. The restore uses initialization-only device
selects and source uploads; fallback buffers stay on device. The timestep code,
default forecast program and persistent carry schema stay unchanged. The opt-in
initializer may compile small selects; GPU init cost and forecast-skill benefit
remain unmeasured. [Timing](receipts/timing.json).

Reproduce the targeted checks from an immutable checkout, honoring QUIET:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src \
  taskset -c 12,13,28,29 nice -n 19 timeout 600 python -m pytest \
  tests/test_cycle.py tests/test_noahmp_swiss_glacier_init.py -q \
  --basetemp <USER_HOME>/wrf_gpu2_lanes/u2-cycling/pytest
```

Full external Swiss reference tests require the retained CPU files; the tracked
crop/clean-archive checks are mandatory without those files. All experiment rows
CY00–CY14 are appended to the main EXPERIMENTS ledger. Disposable snapshots,
pytest trees and temporary cases were removed; cache directories were NONE.
