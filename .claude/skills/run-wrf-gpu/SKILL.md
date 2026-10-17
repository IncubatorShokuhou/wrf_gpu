---
name: run-wrf-gpu
description: Run or set up an existing WRF forecast case with wrf_gpu, including prepared inputs, GPU checks, output and restart. Use the operator runbook; not for developing the model.
---

# Run wrf_gpu for an end user

Read [AI_OPERATOR.md](../../../AI_OPERATOR.md) first. It is the authoritative
runbook. This skill follows the **released v0.4.0** documentation; use the
installed release's matching guide.
Explain installs, cold compilation and long integrations, then report actual
source, inputs, resolved options and output paths.

## Run the requested case

Identify `namelist.input`, prepared `wrfinput_d0*`, `wrfbdy_d01` and required
runtime tables/auxiliary files. Raw GRIB or `met_em` requires upstream
WPS/real.exe. Distinguish native initialization from CPU-history replay.
Resolve duration, domain count and products. Preserve requested physics; use
single-domain `--domain` or `--domains-from-namelist`/`--max-dom` as documented.
Use fresh output and disk-backed scratch. Explain unsupported selections;
do not drop required options, replace physics or bypass preflight to get a run.

Use the runbook's pinned Python/CUDA-JAX recipe, set `GPUWRF_WRF_ROOT`, and
confirm JAX reports a GPU. On the shared development host, device queries and
GPU commands require its official lease/lock; public installations do not
require that infrastructure. CPU imports/Pallas interpretation are not GPU
checks. A short Swiss smoke proves toolchain readiness, not forecast fidelity.

Compilation is expected for a new source, target, geometry or selected
constants. Report compile versus integration progress and actual memory
admission failures. Keep default fp32 WRF REAL carry, explicit WRF DOUBLE
islands, cadence and cache compatibility checks. Do not promise a cache hit,
fixed VRAM requirement or extrapolated speedup.

## Selection and qualification limits

Consult the [support matrix](../../../docs/release/SUPPORT_MATRIX_v0.4.0.md)
and [known issues](../../../KNOWN_ISSUES.md):

- Smagorinsky `km_opt=3` is GPU-refused. CAM-MG11 uses merged stock
  Triton3.6.x (`pip install "triton==3.6.*"`); disclose the separate offline STACK1240B FAIL and
  c89 D6/inventory/NF0 PASS at its registered source, native E41 UNKNOWN. Historical failures remain.
- `sf_surface_mosaic=1` is refused. Crop/irrigation branches need geographically
  valid maps; a column oracle does not qualify absent inputs.
- DFI1/2/3 run in supported single-domain native cases; nested DFI remains
  refused. Preserve DFL timestamps and the runbook's forcing/filter restrictions.
- Where single-domain-only options remain admitted in nests, report **nested
  not validated**. Actual combination refusals still apply.
- Bounded D6, full variable inventory, E41, F2 and station skill are separate.
  REPORT_ONLY/missing fields are not passes. Finite output does not justify
  intermediate negatives or repairs.

New optional state and held diagnostics can make older active checkpoints
incompatible. Follow the [upgrade note](../../../docs/release/UPGRADE_v0.4.0.md);
never edit authenticated checkpoints to bypass refusal. Incorrect historical
precipitation accumulations require a fresh forecast for corrected totals.

## Deliver useful output

Use the installed command's help for syntax. v0.4 workflows include `post`,
`export`, `quickstart`, `tslist-export`, `stream-check`, doctor/provenance,
station verification, ensemble statistics and regional optimisation/cycling.
Definitions/evidence are linked in the operator runbook. Resolve only missing
preferences affecting the requested output; do not add an unrequested campaign.

Inspect completion, dimensions/variables and finiteness; retain source/input
hashes and deliver actual file paths. Original CPU-WRF/pristine WRF is fidelity
truth; older GPU output is regression evidence. Report failed/incomplete gates
plainly. The frozen default identities are Swiss3356e249/PROD196aed9f.
The final ledger records66 accepted scientific scopes (65full24h+c45sixhour),
with E41 33PASS/32FAIL-disclosed/1UNKNOWN(c89). This is scientific validation;
source and tags are published; GitHub release assets await owner authentication. Consult the
[complete user guide](../../../docs/release/USER_GUIDE_v0.4.0.md) and
[final source/receipt ledger](../../../proofs/r_docs040/FINAL_RELEASE_LEDGER.json).
