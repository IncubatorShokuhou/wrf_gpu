# Two-way nesting

**Status in v0.4.0: Validated 24 h on one two-domain configuration** (two-way feedback, `smooth_option=2`). Other nest set-ups are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured. Switch it on with the namelist setting `feedback=1` or with `gpuwrf run --feedback`.

Set `feedback=1` in `&domains` or pass `gpuwrf run --feedback` to feed a child's
interior back to its parent after each subcycle. Omitted `feedback` stays off.
The API configuration can explicitly set `feedback=False` to override a case.
Single-domain execution records these controls but has no child feedback.

`smooth_option=0` copies the area/face averages without smoothing; `1` applies
WRF's 1-2-1 filter; `2` applies its smooth/desmooth filter. Omitted
`smooth_option` uses WRF's default, 2. `spec_zone` excludes the prescribed nest
boundary from averaging. Odd refinement ratios greater than 1 are supported.
Admission is limited to two domains, B=1, Thompson (8), MYNN/MYNN-SL (5),
Noah-MP (4), RRTMG (4/4), cumulus 0/1, `hypsometric_opt=2`, `sst_update=0`,
and inactive urban/lake/nudging and advected MYNN TKE. Other combinations
are refused before model initialization. Deeper hierarchies are not supported
yet. Solar interpolation, online NWP/radar diagnostics and precipitation buckets
are not validated with two-way feedback and are refused when feedback is enabled.

The implementation uses WRF's ordered REAL accumulation, area averages for
mass and vertically staggered fields, and face averages for U/V. Registry
selection includes dynamics, moisture, supported number/scalar fields, PB/PHB/MUB,
and unsmoothed microphysics heating and precipitation accumulators. Perturbation
pressure and nonadvected QKE are held. WRF point copies cover soil/snow/canopy
carry; T2/Q2/U10/V10/PSFC and independent TH2 use unsmoothed averages.
Initialization includes WRF's pre-child-start feedback, parent start_domain,
terrain/map/Coriolis/rotation fields and parent land/radiation/GWDO setup.
Independent resident ALB is fed back rather than reconstructed from averaged
PHB/MUB. Packed radiation history and static land pools keep up-fed SWDNB/LWDNB
separate from SWDOWN/GLW forcing; per-step sums consume the up-fed fluxes.
Lead-zero and later output retain those resident feedback auxiliaries.

The tracked fixtures are generated from verbatim pristine WRF
`copy_fcn`, `sm121`, and `smdsm`, with 63 cases and three feedback epochs per
case. They include active nonuniform fields and real input crops from a two-domain
9/3 km Canary Islands case. These check the operators, not a coupled forecast. Another 48 verbatim point-copy
cases cover REAL/INTEGER and mass/U/V staggering; six pristine EOS cases
exercise supplied ALB under both hypsometric branches and all smooth options.
The CPU EOS check emulates explicit rounding and the original CPU math
library. That Canary Islands case is used only for operator checks.

A short CPU coupled check passes against the original CPU WRF on a real child
crop of a three-domain Tenerife case: 120×70 parent, 36×30 child, 44 levels,
dt 54/18 s, two parent and six child steps, ten frames. All 312 registered
field/frame limits pass, with zero nonfinite values or guard events. The
196/201 reported fields include diagnostics without limits; they are not all
independent fidelity checks. Initial P/PH/QKE differences are listed in the
qualification record. The CPU harness emulates GPU rounding and uses a
Thomas tridiagonal solver shim. On a shared CPU,
initialization took 444 s and the run 3,592 s; these are not GPU performance
numbers.

The off path returns the original state and does not invoke feedback kernels.
The on path keeps each leaf's averaging and filter in one JIT executable,
without compiling the full State feedback into one program. Option 2
requires four directional updates per field versus two for option 1. Resident
bases, heating and precipitation accumulators add feedback work; the old
perturbation-pressure and QKE feedback work is removed.
This changes launch count, traffic, temporary memory and compile cost only for
two-way runs. GPU costs have not been measured; the CPU numbers below are diagnostics only.
The 41-field packed CPU API measured 107 us versus 13.57 ms for sequential
dispatch (126.7x), with 83,392 bytes of executable temporaries; operands/results
were bit-identical. Both restart formats preserve ALB and packed history exactly;
active two-way resume refuses missing groups.
Active restart headers bind the complete initialized auxiliary field list,
domain shapes and WRF REAL storage. Missing or malformed groups are refused
before device restoration. Both portable restart APIs accept `feedback_schema`
to check the current initialized template against the saved header. Legacy
inactive restarts remain readable; single-domain feedback selects no active
child feedback and does not require these groups.

The reproducible oracle builder and coupled-reference tools are in
`proofs/u2_twoway/`. They preserve original reference inputs and executables.
The complete field disposition, measured limits and evidence hashes are in
`proofs/u2_twoway/QUALIFICATION.json`.
