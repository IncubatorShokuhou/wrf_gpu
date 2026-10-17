# NWP operator boundary

Reference: pristine `WRF/phys/module_diag_nwp.F`, full subroutine and its
original `WGAMMA/GAMMLN`; `WRF/dyn_em/module_diffusion_em.F::cal_helicity`.
Source SHA256 values, compiler flags, full operands and sequential results are
in `nwp_oracle.json`. The generator imports no gpuwrf code. This is operator
qualification, not a coupled forecast or GPU performance result.

`module_first_rk_step_part2.F:535–567` calls helicity before RK. The diagnostics
driver calls NWP after RK. Fixed-dt NWP resets at
`MOD(itimestep-1,idump)==0`, including `UP_HELI_MAX` computed by that earlier
call, then accumulates current W and graupel. W_MEAN is normalized only at
`MOD(itimestep,idump)==0`. W_DN_MAX is negative. W extrema sample z faces
where P8W > 40000 Pa. The column mean includes ZNW in the closed [0.5,0.8]
interval; sums retain the original bottom-to-top order. REFD_MAX samples
REFL_10CM at the bottom level, not a column maximum.

For mp=8/28 the original `module_diagnostics_driver.F:462–510` omits NG_CURR;
the diagnostic hail operator therefore diagnoses its own graupel number
distribution from Qg. This differs from the transient Ng used by the radar
operator at microphysics exit. Its default REAL density parameter is 500
kg/m3, even though Thompson itself uses a different distribution parameter.
Hail bin positions, lambda and cumulative concentration are original DOUBLE
islands. Parentheses evaluated in REAL before DOUBLE assignment remain REAL.

NWP package Registry fields are WSPD10MAX, W_UP_MAX, W_DN_MAX, UP_HELI_MAX,
W_MEAN, GRPL_MAX, HAIL_MAXK1/HAIL_MAX2D; REFD_MAX is a separate always-declared
field sampled by the operator. UH is a restart-only field used by helicity.
T2MIN/T2MAX belong to `output_diagnostics==1`, with different cadence and
additional averages/extrema; they are not implied by `nwp_diagnostics==1`.

The helicity API owns a whole specified/nested, nonperiodic grid, with
`ide=nx+1,jde=ny+1` as in WRF. UH's western/southern mass edges retain prior
values; the smoother updates indices 1 through nx-2/ny-2 (zero based).
UP_HELI_MAX's western/southern edges copy the next cell in, while the last
eastern/northern owned mass cell lies outside the smoothing range. This
asymmetry and the restart UH leaf are deliberate source behavior. ZX/ZY use
split PHB and PH differences, following compute_diff_metrics, rather than
differencing a reconstructed total geopotential.

Cost when enabled: three Pallas calls per own step (helicity column,
helicity smoother, NWP/hail column) plus small pointwise metric assembly if
the caller does not provide retained metrics. The ten REAL surface leaves
require 40 bytes per owned horizontal cell. Helicity's temporary use-column
mask is one 4-byte surface array; its UH output is carried. The column
programs keep profile accumulators and hail-bin iteration state in registers,
without allocating bins x levels x domain arrays. Actual GPU launch count,
memory peak, compile/readiness and timings remain unmeasured in this CPU wave.
