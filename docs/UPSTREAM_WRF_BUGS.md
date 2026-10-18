# Upstream WRF bugs affecting experimental ports

| Scheme | Original WRF issue | Port/reference disposition |
|---|---|---|
| Goddard radiation, LW/SW 5 | Cloudy fast-overcast SW reads undefined `taux` (`module_ra_goddard.F:5315`) | Original oracle uses compiler-defined zero initialization; port mirrors zero only at that read. Sensitivity is material; option 5 is experimental. |
| Goddard radiation, LW/SW 5 | Exact-equator climatology reads profile 11 outside the 10-profile table (`:2491`) | Exact-equator input is refused before forecasting. |
| Stochastic SPPT/SPP/RAND Gaussian patterns | A length scale far above the domain size underflows the REAL normalization `ZGAMMAN` (`dyn_em/module_stoch.F:739-768`); coefficients become NaN and the clipped pattern is the constant +cutoff | The port's spectral setup refuses the configuration at initialization; `gpuwrf ens prepare` refuses it before writing members. |

[Source hashes, reproduction and measured sensitivity](../proofs/s2_goddardrad/UPSTREAM_WRF_BUGS.md).

## NUDAPT frontal-area direction reads undefined output winds

Pristine WRF 4.7.1 commit `f52c197ed39d12e087d02c50f412d90d418f6186`,
`phys/module_sf_urban.F:710`, computes `WDR=(180/PI)*ATAN2(U10,V10)` before
assigning the `INTENT(OUT)` arguments U10 and V10. Its Noah-MP caller
(`module_sf_noahmpdrv.F:noahmp_urban`) passes uninitialized local
`U10_URB` and `V10_URB`. This affects directional NUDAPT frontal-area inputs;
the ordinary table-only and LCZ-only paths never execute that block.

The manager authorized a compiler-defined oracle on 2026-10-13, matching the
policy used for the k-epsilon port. We compile the verbatim caller with
`-finit-real=zero -finit-integer=0 -finit-logical=false` and FP traps off; ATAN2(0,0) selects LF_URB(1). The GPU port mirrors that
selection for its one-cell zero-init oracle, and prepares geometry once during input loading.
It does not substitute lowest-level winds or previous-step winds.

Sensitivity is material: the same 25 LCZ columns over 30 steps, with only
caller-local initialization changed from zero to positive infinity, change
HFX by up to 9.862823 W m-2, U10 by 1.536792 m s-1, V10 by 1.170660 m s-1,
TSK by 0.711914 K and UST by 0.203611 m s-1. Both arms have finite post-step
outputs. This is qualification against the explicitly initialized reference,
not parity with an arbitrary compiler's uninitialized-memory behavior.

Reproduction: `proofs/v040/urban_lcz/probe_reference.py`; source and driver
hashes plus the measured differential are retained in the lane receipt.
The reproducible one-cell LCZ/NUDAPT oracle is
`proofs/v040/urban_lcz/build_oracle.py`, with source hashes, precision flags
and frozen tolerances in its manifest. Original WRF sources remain unmodified.


The LF(1) convention has **one-cell zero-init parity only**. Compiler
initialization zeros caller locals at procedure entry; it does not reset
U10_URB/V10_URB before each tile cell. The verbatim `noahmp_urban` loop reuses
those locals, so later cells can read the preceding cell's assigned wind
outputs at the undefined WDR read. Fixed pointwise LF(1) does not reproduce
that multi-cell dependency, even with the same compiler initialization flags.
Source anchors: `module_sf_noahmpdrv.F:3230,3380–3382,3510,3578` and
`module_sf_urban.F:427–428,709,1675–1676`.

The critic's sealed two-cell pristine experiment uses identical array inputs
and state in both cells,25 regimes × 30 steps, anisotropic LF=(.1,.3,.5,.7),
and unmodified WRF bodies. [M] Maximum cell differences are HFX 6.273635864 W/m2,
U10 0.992087841 m/s, TSK 0.301879883 K, V10 0.863518000 m/s and UST 0.090088338 m/s.
All five exceed the existing one-cell REAL4 bounds, including differences at
step 1; scored values remain finite. The first cell is BYTE with the five frozen
one-cell fields. An isotropic LF=(.1,.1,.1,.1) control makes all five differences
exactly zero. These are diagnostic differences, not a whole-tile PASS.

The deterministic LF(1) path remains experimental under the standing policy;
coupled original CPU-WRF/D6 qualification is pending. LCZ without morphology
is unaffected. The four one-cell oracle results and frozen tolerances remain
unchanged. Retained [multi-cell diagnostic and reproduction sources](../proofs/v040/urban_lcz/evidence/multicell/README.md),
[result](../proofs/v040/urban_lcz/evidence/multicell/RV03-multicell-final.json)
and [seal](../proofs/v040/urban_lcz/evidence/multicell/SEAL.json) preserve the
anisotropic savepoints, isotropic control, verbatim caller and source hashes.

## Stochastic Gaussian patterns collapse to a constant for unresolved length scales

Pristine WRF 4.7.1 `dyn_em/module_stoch.F` (`SETUP_RAND_PERTURB`, lines 739-768)
normalizes Gaussian SPPT/SPP/RAND spectra by `ZGAMMAN`, a REAL sum of
`exp(-4 pi^2 L^2 rho^2)` over the resolved wavenumbers. When the length scale `L`
is far larger than the domain, every term underflows, `ZGAMMAN = 0`, the
amplitude constant is infinite and the spectral coefficients become NaN. WRF
does not stop: the clip `MAX(MIN(pattern, thresh), -thresh)` (for example line
1341) turns NaN into `+thresh`, a constant, systematic perturbation. Measured on
the original CPU-WRF 24 h Swiss run with the Registry defaults (v0.4.0
validation case c41, 43x43 at 3 km): `SPFORCC3`/`SPFORCS3` are NaN in all 1,849
cells of every frame, `PATTERN_SPP_PBL` is 0.3 everywhere (700 km default,
cutoff 2 x 0.15) and `RAND_PERT` is 0.09 everywhere (500 km default); the 150 km
SPPT pattern `RSTOCH` is normal. The port's REAL32 setup raises "WRF spectral
band has no finite nonzero normalization" at stochastic initialization instead
of running a constant pattern. On that grid the limit lies between 200 and
250 km; the 9 km WN3/PROD d01 grids resolve all Registry defaults. Use a
smaller `lengthscale_*` for small domains (see [stochastic.md](stochastic.md)).
