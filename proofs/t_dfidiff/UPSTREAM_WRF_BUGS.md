# DFL consumes diffusion save slots without saving the original options

WRF 4.7.1 `main/module_wrf_top.F:2143–2161` selects DFL (`dfi_opt=1`) by calling
`wrf_dfi_fwd_init` directly. Unlike DDFI/TDFI, it never calls `dfi_bck_init`.
That backward initializer would save `grid%diff_opt` / `grid%km_opt` into the
`*_dfi` fields (`share/dfi.F:189–190`). Their Registry defaults are instead
0 / 1 (`Registry/Registry.EM_COMMON:2847,2851`) when DFL omits the options.

`share/dfi.F:324–327` unconditionally copies these fields into **both** the grid
and model namelist configuration. `dfi_fst_init` does not restore the original
diffusion settings. Each `share/solve_interface.F:29` refreshes `config_flags`
from the model configuration and passes it to `solve_em`. The diffusion guards
in `dyn_em/module_em.F:802,1380` consequently see `diff_opt=0` during the forecast.
This affects dynamics, as well as the history header.

Measured independent solver-entry diagnostic: stage 2 (forward) and stage 3
(forecast), own steps 0/1/2, each report grid **0/1** and solver **0/1**. The
paired uninstrumented control and print-only executable produce 5 files × 362
byte-identical, finite variable arrays. Original c49's retained 24 history
frames all report 0/1. See `receipts/TD01.json` and `receipts/TD01_dump.log`.
The private executables only replace the mediation wrapper and relocate the
two existing diagnostic OPEN paths; pristine numerical routine bodies and
original reference files are untouched. No `fort.98` was emitted; neither
run log contains an out-of-bounds diagnostic.

The port already preserves this WRF quirk: `gpuwrf.dfi.driver12` writes the DFI
options (defaults 0/1) into **both** forward and forecast stage namelists. For
explicit `diff_opt_dfi=1, km_opt_dfi=4`, both stages use 1/4. DDFI/TDFI save the
original options through the backward phase and retain their existing behavior.
Changing DFL to forecast with the original 1/4 would diverge from WRF.

This is a deterministic upstream control-flow quirk, not an uninitialized
read; compiler initialization sensitivity is inapplicable here. No numerical
port correction or tolerance change is proposed.
