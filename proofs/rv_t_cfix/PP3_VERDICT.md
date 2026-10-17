# MERGE — t-cfix-phyprep series 3

Reviewed source at `cfe70e98e6`, series-3 implementation `fecaa223a5`, base
`1b1e31e575`. CPU scope: F4/F15/F16 thermodynamic staging of API-only slab,
Noah-classic and Pleim-Xiu land paths. CLI admission is unchanged.

[M CPU/source] PQ01: all 22 phyprep tests and five changed Grell consumers PASS.
PQ02 reuses the independently rebuilt original-WRF REAL4/8 fixture with exact
fixture/builder/shared-helper identity and current pristine source/constants SHA
checks. PQ03b: three actual two-step API operational callers PASS, verifying
grid forwarding and refreshed staged temperature/pressure after mass/moisture
change. Sentinel solvers and isolated RK make this caller evidence, not a
numerical dycore/forecast oracle. PQ06: all three unmocked land operational
smokes PASS (8.23 s). PQ05: independent TH2=theta_m deletion produces one genuine
numerical assertion, zero errors/skips (3.94 s).

PQ04 both default native-facade CPU traces EXACT:
Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.

Source matches pristine surface_driver/noahdrv/pxlsm: slab gets dry T3D,
hydrostatic P3D and surface face pressure; Noah-classic SFCTMP is dry T,
SFCPRS is the mean of the first hydrostatic faces, TH2 uses the exact noahdrv
822–825 Exner formula, and ZLVL uses g=9.81 geometry; PX gets dry TA1/THETA1,
surface face pressure and g=9.81 height. The three operational call sites pass
the real GridSpec. Existing flux-handle conversion and PX initial air temperature
now use dry thermodynamics. Kernel internals and restart schema are unchanged.
No tolerance was widened.

Efficiency [I/U]: affected API paths add hydrostatic/density staging and retain
their existing legacy wide calculations. Actual GPU launches, live temporaries,
peak memory and readiness are unmeasured; there is no default-program delta.
No speed claim follows from source complexity. No new baked inputs or carry fields.

The first reviewer API attempts lacked held radiation; those runtime fixture
exceptions are logged and excluded. Correctly seeded repeats pass. Existing
precipitation/SW/specific-humidity findings belong to the separate F24 work and
are not cleared by this thermodynamic-staging verdict. The producer's unrelated
external-reader-key red is also outside this scope.

Next: merge the scoped series and retain the existing API/CLI qualification
labels; no current composite exercises these API-only LSMs. Future admitted GPU
paths require their coupled qualification. No GPU or producer edits; MAIN logs,
cache off, snapshot cleaned. Artifacts `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PP23/`.
