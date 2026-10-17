# Pristine initialization sensitivity

No pristine WRF body was edited. The lane-local full FDDAGD oracle uses
compiler-defined `-finit-real=zero -finit-integer=0 -finit-logical=false`,
with floating point traps off as in production WRF.

SF09 rebuilt the same source with `-finit-real=snan -finit-integer=7
-finit-logical=true` (gfortran accepts named real initializers, not numeric
`42`). Every array in all 20 REAL4/REAL8 Swiss fixtures is byte-identical:
7,115,160 values, including inputs, analysis pairs, all tendency lifetime
stages, and SDA outputs. No observed output depends on an uninitialized
read. Unused extended-tile storage is not treated as an output.

The wrapper initializes model fields and external INOUT tendencies, as the
original caller does. In particular surface OLD is **not** overwritten by
model U10/V10/TH2/Q2: the corresponding block is commented out in original
`phys/module_fddagd_driver.F:275–291`. Only three dimensional OLD analyses
are overwritten on step 1.

Receipt: `initialization_sensitivity.json`; zero-init source hashes are in
`savepoints/MANIFEST.json`. These are component checks, not a long forecast
qualification or a proof of every upstream WRF configuration.
