# EEPS and KEPS PBL

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. **Since v0.4.2, EEPS (`bl_pbl_physics=16`) is also validated inside nested domains:** it passed the 24-hour comparison on all three domains of a Tenerife 9/3/1 km forecast with `sf_sfclay_physics=1` ([details](release/NESTED_PBL_QUALIFICATION_v0.4.2.md)). KEPS inside nests and other combinations are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

`bl_pbl_physics=16` selects EEPS; `17` selects KEPS. WRF 4.7.1 has no
`bl_pbl_physics=6` package: MYNN's third-order closure is option 5 with
`bl_mynn_closure=3.0`.

| PBL | Supported surface pairing | CPU reference | Forecast status |
| --- | --- | --- | --- |
| 16 EEPS | `sf_sfclay_physics=1` | 30 full carried REAL32 columns | Validated 24 h (one tested configuration) |
| 17 KEPS | `sf_sfclay_physics=1` | 30 full carried REAL32 columns | Validated 24 h (one tested configuration) |

The port currently supports revised MM5 only. WRF's check_a_mundo has no
additional EEPS/KEPS-specific mandatory surface pairing; this is a restriction
of the port. Both CLI pipelines bind the requested PBL and
refuse unsupported surface pairings. The operational path requires flux-form
scalar advection and raw RK physics tendencies (`rad_rk_tendf=1`).
Land coupling currently supports no-LSM and Noah-MP (`sf_surface_physics=0/4`);
other land schemes fail the honour check. Both paths retain the actual surface
driver's history outputs rather than re-solving MYNN at output time.

The reference bodies are unchanged WRF 4.7.1. Private builds use compiler-defined
initialization (`-finit-real=zero -finit-integer=0 -finit-logical=false`) with
floating point traps off. Explicit zeros at four undefined-read sites mirror
this reference: KEPS GET_PBLH:822 `ttke`, surface_bl_pra_ri:1068 `radsum`,
`c_nsquare`'s unwritten top `dtmdz`, and EEPS eeps2d:748 `radsum`.
Changing initialization materially changes both schemes' output: examples are
EEPS TKE by 0.10875146 m²/s² and KEPS exchange heat coefficient by 4.586938 m²/s.
See [sensitivity evidence](../proofs/s2_keps/initialization_sensitivity.json).
Because of these upstream WRF bugs, results depend on how memory is
initialized; the port does not claim compiler-independent WRF behavior.

The scalar packages are EEPS `pek_adv,pep_adv` and KEPS
`tke_adv,diss_adv,tpe_adv`. KEPS disables their feedback into the closure as in
WRF, while the dynamical scalar loop still transports them. Live children carry
corresponding boundary records. Physical TKE/epsilon, KEPS TPE/Prandtl number,
held exchange/flux diagnostics, and the PBL call counter use REAL32/int32
versioned restart fields. Incomplete legacy resumes fail before stepping.
History reads held physical fields; it does not run PBL physics again.

The frozen full-column comparison uses 16 REAL32 ulps with propagated input
rounding for tendencies/fluxes, requires finite values on both sides, and covers
stable, unstable, neutral, stratocumulus, mixed ice and carried restart profiles
at 16/44 levels. CPU reference checks disable algebraic division reassociation
and FMA contraction. This column check does not cover GPU precision, long
forecast drift or performance; the 24-hour forecast comparison is separate.

A three-step CPU run on a Swiss case is finite for both options. EEPS reports no
guard repairs. KEPS reports one QC roundoff repair (-1.29e-31 kg/kg) at the
north boundary; unchanged WRF PD and RK routines reproduce a negative value
from the same captured operands. The port passes the fixed positive-definite (PD) limits;
its local error is 2.20 times WRF's own spread with and without fused
multiply-add. It does not pass the stricter check against one times that spread;
this is a documented deviation. The counter and independent operator evidence
are in [the qualification package](../proofs/s2_keps/README.md).
