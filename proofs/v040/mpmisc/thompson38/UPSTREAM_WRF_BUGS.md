# Original-source initialization and caller checks

No pristine WRF executable body was modified. Original modules use
compiler-defined zero initialization with FP traps off, as production WRF.
A sensitivity build changes only defaults to real NaN, integer 17 and
logical true. All six initial caller dumps match BYTE
(`init_sensitivity.json`); the additional five fresh/threshold baselines
also match all public inputs, outputs and scalars BYTE
(`fresh_init_sensitivity.json`). This covers eleven tested regimes and
does not assert the absence of every undefined read in WRF.

The staged v018 hail-only caller omits aerosol init switches by design;
it is not the production mp38 reference. Its empty `its:ite-1` aerosol slice
is corrected by a two-wide initialization tile while retaining one active
column. Original mp38 optional-radius flags are 000 under
`physics_init.F:988-1024`; that caller correction preserves all non-radius
bytes in the six original columns (`original_build.json`).

The mp8/28 collision allocation has one plane and historically reads density
index 5 outside that plane. Option 38 allocates all nine planes and uses the
actual original table; it does not inherit the mp8 address workaround.

The near-999e3 threshold fixture has one negative incoming ice-number level
in each column. Original WRF keeps that NI1D baseline while flooring or
reconstructing its private working ni. Clamping the baseline early changed
REAL tendency cancellation and produced a two-ULP counterexample. The port
preserves the original chronology; this constructed input scope is disclosed.

The legacy project's freezing and cloud-coefficient assets differed from
initialized original values; these are port asset defects, not changes to
pristine WRF. Option 38 uses exact original records and REAL coefficients.
The standalone cold-start helper matches original profiles/emissions BYTE;
ORHO is declared but unused in this original initialization body.
