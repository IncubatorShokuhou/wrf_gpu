# Original WRF 4.7.1 NTU behavior

Pristine source SHA256: `c864483fa572d42ab1dc51cc538f8062dc2f582369f597261e8c850c49ce34dd`.
Original routine bodies are unchanged in every oracle build.

## Aerosol initialization reads an unused top interface

`NTU_INIT` initializes `P_PHY`, `RHO`, and `DZ8W` only over `KTS:KTE-1`
(`module_mp_ntu.F:282–296`), then passes `KTE` unchanged to `INIT_AEROSOL`
(:298–300). `INIT_AEROSOL` reads the extra row (:1616–1649), including
`P(I,KTE,J)` in the PBL search. With compiler initialization to zero this
sets `K_PBL=KTE` before the existing `KTE-2` cap. With initialization to
infinity the last initialized mass-level pressure determines the PBL.

NTU03 compared zero/infinity initialization: 12 original 12-level, two-step
columns remain finite, but 239/1428 arrays differ, including the initial
aerosols. This test changes forecast input, so it does **not** establish
that the timestep itself depends on uninitialized scratch. A subsequent
comparison freezes every MP input byte before varying compiler policy.

The port must mirror zero only at this unused interface, under the standing
compiler-initialization policy. This materially affects initialization and
must be disclosed for experimental admission. Never patch the pristine body
or loosen any numerical tolerance.

## Timestep scratch sensitivity

NTU04 fixes every input byte to NTU01 before executing MP_NTU under infinity
initialization. All 1428 arrays are finite. Exactly 30 forecast output arrays
change, all in the evaporation column (case 10); every initial state/input
array remains byte-identical. Graupel number differs substantially, so this
is a material upstream behavior, not a tolerance issue. Its exact read site
remains under investigation; zero local initialization is the reference policy.

## Original ice area tendency can be NaN before its literal zero maximum

Print-only original scalar observations, independently byte-identical to the
unobserved scalar outputs, expose an intermediate NaN at :5444 (`AIMcsi`).
`NIMcsi` may be negative in the original Hallett–Mossop calculation, and the
REAL cube root of the negative area expression is NaN. Original :6833 writes
`AI1D = MAX(0., AI1D + ... )`, yielding exactly zero on the four affected
mixed-phase scalar samples. NumPy `maximum` propagates that NaN instead.

The literal port reproduces this **existing source operator** using `fmax`
only when the original MAX has an explicit zero literal. No new floor,
rate repair, or tolerance change is added. This behavior needs disclosure
with experimental admission and any pre-mask event accounting. The original
variable-operand MAX NaN behavior differs; it is not changed globally.

## Public surface precipitation differs from accumulated totals

Original `MP_NTU:1720` zeros private CLODNCV/ICENCV and public RAINNCV/SNOWNCV
independently. `NTU_MICRO:2573–2578` accumulates all six phases in RAINNC and
ice+snow in SNOWNC, but leaves public RAINNCV as rain-only and SNOWNCV as
snow-only. The public slots pass unchanged to `module_surface_driver.F:1569`
(RAINBL) and :3283 (Noah-MP), then `module_sf_noahmpdrv.F:779–783` reads them.
The port preserves these public instantaneous arguments, including separate
hail, while retaining all six disjoint buckets and the original accumulated
totals for history. Summing all phases into the land's public RAINNCV would
silently repair upstream behavior and would violate the original caller.
