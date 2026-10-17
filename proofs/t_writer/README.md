# Composite writer preflight

This gate compares the actual writer's initialized inventory with original
CPU-WRF headers. It advances **zero model steps**. It is schema coverage, not
forecast fidelity, numerical qualification, or a GPU/performance claim.

`audit.py` inspects every available CPU frame against the declared writer specs.
It is an upper bound: a declared spec does not establish an emitted variable.
`initialized_schema.py` loads the real resolved namelist and initial carry on
CPU, then exercises writer phases 0 and 1 with that same carry. The established
CPU harness interprets Pallas and substitutes Thomas for the deadlocking LAPACK
solver. Schema-only NetCDF files are deleted after reading their inventories.

`batch_schema.py SOURCE ROOT OUTPUT [CASE ...]` runs these arms sequentially,
with a 300-second bound per case and an explicit QUIET guard before every arm.
SOURCE must be a clean detached snapshot. Completed receipts are preserved on
resume. ROOT is the read-only v-prep directory containing MATRIX.json and case
inputs/original CPU references. Output and logs belong under the lane directory.
Use `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,24,25 nice -n 19`.

`compare_schema.py ROOT INVENTORIES RESULT.json SOURCE` reads **every available
original frame**, compares dimensions, variable names/dtypes/attributes and
global attributes, and authenticates the actual case/input files against the
original CPU receipt. Missing/unfinished references and loader refusals cannot
pass. GPU writer provenance/task counts are explicitly different. Extra writer
variables/dimensions/attributes are disclosed; coverage PASS is not exact
inventory identity. Numerical D6 remains separate.

The supplied `cpu_schema.json` freezes independently observed CPU metadata;
`generate.py AUDIT ROOT` regenerates the compact host metadata table. No table
from this lane enters a traced model program or changes restart formats.
`external_gates.json` records the manager's additional actual-GPU inventory
findings, including mp40's bounded D6 pass with twelve missing history fields.
Missing producer values are disclosed, never supplied as invented zeros.

NSSL17/18/22 now retains the existing oracle-qualified `radardd02` result
from S5, before unpacking. WRF's mandatory radar allocation, first-step bottom
level, history alarm and specified-boundary lifetimes govern publication.
ARW with NWP diagnostics disabled does not compute `REFD_MAX`; it remains an
explicit omission. The vertical-maximum routine in `module_diag_refl` is NMM-only.
The existing radar restart slot is reused; old NSSL checkpoints without the
newly required held radar are refused under the existing E78 checks.

`replay_headers.py` records per-case source/receipt hashes while combining the
complete TW08 census with final initialization probes for the18 configurations
affected by emission changes. It requires the final model source to differ from
the qualified diagnostic source composed with approved upstream main only in
the registered host writer/header functions and metadata table. Protected
initialization/model AST is checked. It replays global NetCDF attributes without
repeating model initialization. Older unaffected inventories keep their actual
source identity and an explicit reuse declaration. Schema reuse is not forecast
validation. Raw receipts are never rewritten.
