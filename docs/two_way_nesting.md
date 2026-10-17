# Experimental two-way nesting

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
are refused before model initialization. Deeper hierarchies need separate
static-field lifetime qualification. Solar interpolation, online NWP/radar
diagnostics and precipitation buckets require separate two-way qualification
and are refused when feedback is enabled.

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
case. They include active nonuniform fields and real PROD input crops. These
qualify the operators, not a coupled forecast. Another 48 verbatim point-copy
cases cover REAL/INTEGER and mass/U/V staggering; six pristine EOS cases
exercise supplied ALB under both hypsometric branches and all smooth options.
The CPU EOS gate emulates explicit rounding and the original CPU libm;
GPU libdevice execution remains unmeasured. PROD is a WN2 input set and is
used only for operator debugging.

The short CPU coupled gate passes against original CPU-WRF on a real WN3
Tenerife child crop: 120×70 parent, 36×30 child, 44 levels, dt 54/18 s,
two parent and six child steps, ten frames. All 312 registered field/frame
bounds pass, with zero nonfinite values or pre-mask guard events. The
196/201 reported fields include unbounded diagnostics; they are not all
independent fidelity gates. Initial P/PH/QKE differences remain disclosed in
the qualification receipt. The CPU harness emulates PTX rounding and uses a
Thomas solver shim. Initialization took 444 s and the cold run reached its
complete numerical report at 3,592 s; its 3,600 s timeout occurred during
teardown. These are shared CPU harness costs, not GPU performance numbers.
GPU 24-hour qualification remains pending (KI-11).

The off path returns the original state and does not invoke feedback kernels.
The on path keeps each leaf's averaging and filter in one JIT executable,
without compiling the full State feedback into one program. Option 2
requires four directional updates per field versus two for option 1. Resident
bases, heating and precipitation accumulators add feedback work; the old
perturbation-pressure and QKE feedback work is removed.
This changes launch count, traffic, temporary memory and compile cost only for
two-way runs. GPU costs remain unmeasured; CPU measurements are diagnostics.
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
