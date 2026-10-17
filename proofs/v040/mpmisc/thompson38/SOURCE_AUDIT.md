# Thompson mp38 source and caller qualification

The independent reference is unmodified WRF 4.7.1
`phys/module_mp_thompson.F`, SHA256
`fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802`.
H66 at `beabc51c6` qualifies the composed column: all 6,160 prognostic values
and 77 public surface values are literal REAL/no-FMA BYTE. Three evolving
original own-steps also match all 1,680 column and 21 surface values BYTE,
including held emissions, theta projection and nonzero precipitation carry.
Direct evolving public theta also matches 120/120 values BYTE.
A supplementary direct caller dump also matches public theta 440/440 BYTE;
43 projected T_OUT cells were non-injective, so projection alone was insufficient.
No candidate physics was rerun for that supplemental comparison.
Compiled differences are reported separately in [COMPILED_DEVIATIONS.md](COMPILED_DEVIATIONS.md).
Parent integration receipts now pass: AOT collision tests 10 PASS, shared
checks 404 PASS plus the corrected-pin delta 59 PASS, and a genuine Swiss
CPU two-step option-38 smoke with zero nonfinite values, guard repairs or
pre-mask events. Approved Swiss f672a573 and PROD 6fc715a2 identities match.
The smoke uses the retained wide CPU facade; its three conditional volume
prognostics are FLOAT64 and five held surfaces are REAL32. It does not
replace the separate REAL32 column proof or establish native GPU carry.

The production `physics_init.F:4565` caller enables BOTH hail (`NG`) and
aerosol (`NWFA2D/NWFA/NIFA`) switches. The old v018 hail-only reference omits
aerosols and is unsuitable for production mp38 qualification. The corrected
oracle gives initialization a two-wide memory tile while retaining one
active column. `physics_init.F:988-1024` excludes option 38 from optional
microphysics effective radii: the qualified caller flags are 000.

The supported scope is `wif_input_opt=1`, `aer_init_opt=0`; original WRF
sets working NBCA and final NBCA to zero in this branch. Black-carbon option
2 and other initialization modifiers require separate qualification.
The near-999e3 stress pair includes one negative incoming NI1D level per
column; this edge is disclosed, and its raw baseline is preserved exactly.
The remaining levels genuinely distinguish the original explicit two-gamma
snow-number gate from the incorrect Field zeroth moment.

The implementation reuses existing Thompson/Aero equation code objects
through private bindings and an optional hail/aerosol context. Fifteen shared
functions specialize to the pre-hook default AST exactly; mp8/mp28 paths
retain their original arithmetic. There is no mp8/mp28 substitution for hail.
The nine density planes are 50,100,200,300,400,500,600,700,800 kg/m3.
Five original DOUBLE collision channels occupy 674,697,960 bytes. Six
original freezing records add 254,944,800 bytes; their every DOUBLE bit is
preserved. The exact initialized cloud coefficients add 720 bytes.
Legacy freezing/cloud coefficient assets had measured discrepancies and
are overridden only for option 38.

Original working `rg/ng` are per volume; `rb` and prognostic `qvolg` are per
kg dry air. The volume flux is `vtg*rb`; its floor uses `rg/rho/800`.
The column preserves original cumulative REAL tendencies, frozen ice/snow/
graupel working moments through saturation/evaporation, and the original
post-evaporation density chronology. DOUBLE slopes and rates round at the
original REAL assignment sites. The cloud-slope reciprocal has an explicit
source-kind regression, including a missing-DOUBLE-island control.

The initialization-only host helper matches 243 original profile/emission
values BYTE. NWFA and NIFA missing-profile checks are independent; ORHO is
declared but not read by this pristine initialization routine. Positive
held emission input is retained. The wrapper adds held NWFA2D/NIFA2D after
physics. Public temperature uses both REAL rounding points, `T/pii` then
`theta*pii`. Accumulated RAINNC uses the previous value FIRST in the original
left-associated sum; SNOWNC similarly adds snow then ice.

Qualification includes all 33 original rate groups in the active-regime
census, independently authored threshold witnesses, eleven compiler-init
sensitivity baselines, and a whole-equation legacy-freezing-table mutant
rejected by the original-reference gate. NaN and missing-field controls
also fail the gate. This is CPU source/oracle qualification; GPU performance,
SASS/E41 and full coupled D6 remain pending. The production program is the
shared JAX column composition, not a Pallas column implementation.
