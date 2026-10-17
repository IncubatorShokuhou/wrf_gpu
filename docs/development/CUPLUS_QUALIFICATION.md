# Historical component-only CuPlus qualification

This document records the early component-only stage, before the complete
cu=7/10/11 family was implemented and admitted. Its fail-closed status and
missing-work table below describe that historical stage. For the current
experimental CPU qualification, supported pairings, coupled/restart evidence
and pending GPU/D6 limits, see [CUMULUS_CUPLUS.md](../CUMULUS_CUPLUS.md) and
[QUALIFICATION.json](../../proofs/v040/cuplus/QUALIFICATION.json).

At that stage, `s2-cuplus` provided standalone REAL32 KF-CuP components in
`gpuwrf.physics.cumulus_kfcup_support`, and cu_physics=7, 10 and 11 remained
fail-closed. A qualified helper alone did not qualify a complete scheme.

The component fixtures use exact original WRF 4.7.1 procedures, the original
`module_model_constants`, and an exact block from `module_radiation_driver.F`.
The generator records source checksums, extraction bounds, compiler flags and
caller conventions. Bounds-checked and uninstrumented outputs must agree.
Fixtures, their checksum and the driver are tracked in `proofs/v040/cuplus`.

The 45 constructed cases cover the joint frequency distribution (including
signed slopes, limits, even/odd bins and a one-bin zero-sigma case), deep,
shallow and absent cloud fraction, lifetime/fraction bounds, parcel velocity
fallback, saturation deficit, thermodynamic helpers, and 18 combinations of
radiation cloud flags, convection regime and forced cloud fraction. These
inputs activate the components directly. They do **not** establish deep or
shallow activation in a complete convection driver. The original JFD signed
overflow branch is retained but is outside this finite-case qualification.

`cupCloudFraction` never assigns its `cldfraTend` output. Several other
diagnostics are defined only in its shallow branch. The Python boundary returns
only the outputs the selected original branch defines. The active shallow
formula integrates liquid alone; its ice term is commented out in WRF.

The remaining work at that stage was source-backed:

| Scheme | Original requirements and missing work |
| --- | --- |
| MSKF, cu=11 | `check_a_mundo:1373–1415` rejects km_opt=5, forces icloud=1 and cu_rad_feedback; aercu_opt>0 also requires mp=40 and aerosol input. `physics_init:1116` sets icloud_cu=2 even with no aerosol. Port full scale-aware column, CMT and driver/carry; preserve its explicit r8 islands. |
| KF-CuP, cu=10 | `check_a_mundo:2290–2300` sets cu_diag=1. Port the parcel ensemble and convection driver, surface/PBL statistics, cadence, carries/restart, both pipelines and honour checks. The qualified radiation block supplies cloud fraction and condensate mediation only. |
| CAM-ZM, cu=7 | `cumulus_driver:1332–1385` requires ice and the number-tendency arguments. `camzm_driver:get_tpert:918–1018` accepts CAM-UW PBL (bl=9), or MYJ PBL with MYJ surface layer (bl=2/sf=2). MG is not an unconditional requirement there. Port the r8 convection, evaporation, transport and REAL caller conversions. |

Full graduation then required pristine active driver columns, CPU coupled
smokes, operational wiring, restart completeness, unchanged release identity,
and later GPU/D6 qualification. The component module had no production import
at that stage. Complete drivers and resident carries are now wired for the
documented experimental profiles; GPU performance, VRAM, E41 and D6 remain
pending. Immutable historical receipts retain their original bytes and hashes.
