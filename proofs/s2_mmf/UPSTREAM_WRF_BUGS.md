# Original WRF MMF details

No pristine routine body is edited. Wrappers use compiler-defined zero
initialization with traps off, as required by the wave policy. All selected
soil-table readers and actual caller operations are preserved.

`SOILWATER` does not assign RUNSUB in the MMF branch; the thin caller initializes
it to zero. The driver zeros local RECH and DEEPRECH at
`module_sf_noahmpdrv.F:842-843`, then accumulates them into grid fields at
1366-1367. The port mirrors these local deltas and retains persistent grid
recharge between calls. SHALLOWWATERTABLE's deep branch does not assign its
local RECH output; the existing local-zero caller convention supplies zero.
Zero-versus-signaling-NaN compiler initialization sensitivity is recorded in
the final native-bound receipt. This is separate from changing caller inputs.

`module_sf_noahmp_groundwater.F:522` adds `maxwatdw/DZS` to resolved SMC on a
falling water table. The port preserves that original sign. It does not replace
it with an apparently more physical subtraction.

The test-only literal executor evaluates actual port bodies using NumPy REAL4
arithmetic and original scalar libm expf/powf. Compiled JAX CPU intrinsics and
FMA reassociation are evaluated separately against independent original-WRF
input-ULP and compiler-spread bounds. No tolerances are fitted to the port.
