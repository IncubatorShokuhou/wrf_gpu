# Original-WRF limitations retained by v-prep

These are upstream-reference limitations, not GPU fidelity passes. The manager
owns the release support matrix. No original source body was changed here.

## Ferrier hires/advection, MP15

Owner decision, MAIN MAILBOX 2026-10-14T12:59:16Z: exclude MP15 from the coupled
coverage gate, as for km_opt3. Label **experimental, column-oracle-qualified;
no coupled reference: original CPU-WRF 4.7.1 SIGSEGV at first output**.

The authentic Swiss c63 fails before its first timestep. Removing its nondefault
Noah options still fails. Both the canonical original executable (SHA52f914cb…)
and the unchanged-body compiler-zero executable (SHA79273167…) reproduce the
failure. GDB reaches `output_wrf_ -> open_w_dataset -> open_hist ->
med_before_solve_io`; the failing pointer is `0x3f8000003f800000`, two REAL4 ones,
loaded from the domain structure. The corrupting write is unknown. This evidence
does not establish that suppressing an output field would make the trajectory
safe. No output-mask workaround is admitted; the bounded investigation stops.

Receipts: `<USER_HOME>/wrf_gpu2_lanes/v-prep/c63/cpu_run.json`,
`p15/probe.json`, `p15g/probe.json`, `p15c/probe.json`, `p15d/cpu.log`.
The first p15 preparation inherited run_days1 despite a planned36s stop; that
fault occurred before stepping. The corrected p15g/p15c/p15d namelists explicitly
set run_days0 and36s. None is a successful forecast or GPU qualification.

## Morrison aerosol global-initialization compiler failure

The private global-zero initialization build failed in the original Morrison
aerosol module because its bare SAVE declaration conflicts with automatic
arrays under that compiler profile. The ordinary canonical build succeeds.
The zero build retains that inactive module's exact canonical object/interface;
all400 original source bodies are unchanged and hash-bound. MP40 itself uses
the ordinary canonical executable. This is a compiler-profile limitation,
not a patched upstream algorithm or a GPU result.

Receipts: `ZERO_BUILD.json`, `ZERO_BUILD_PIN.json` alongside this file; external
build logs under `<USER_HOME>/wrf_gpu2_lanes/v-prep/`. Failed global-init compile
564.866s; successful bounded repair291.711s [M CPU, shared nice19].

## km_opt3

**Experimental, column-oracle-qualified; no coupled reference: original CPU-WRF
crashes (SIGSEGV ~00:04) on Swiss and Madrid with km_opt3.** Owner excludes it
from the full-coverage GPU gate. The fresh compiler-zero c48 also fails (rc139,
one frame,29.6s, near00:04:12). Historical receipts remain under
`proofs/v040/gpuqual/km3`; c48 is retained externally. No further variant or GPU
arm is needed for this gate.
