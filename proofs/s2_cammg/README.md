# CAM-MG CPU qualification evidence

`CONTROL_PLAN.json` froze individual REAL4-input ULP controls and original
FMA off/on controls before the native candidate was executed. The original
WRF bodies were unchanged; both original executables used byte-identical
initialized namespaces. `fixtures/compiled_enclosure.json` records 116,256
original calls and its NPZ hash. `ORIGINAL_SOURCE.json` pins pristine sources.

The full literal public caller matched all 52 fields byte for byte on 48
calls. `NATIVE_CPU_GATE.json` records all 27 modified Pallas outputs inside
that independent enclosure, with zero nonfinite values or original fatal
status. `SOURCE_REUSE.json` proves the current scientific bodies and assets
preserve that qualification. Generated line annotations now name original
WRF source lines; their relocation preserved AST identity.

`NATIVE_PREFLIGHT.json` is CPU-only CUDA/sm120 TTIR lowering at 44 physical
levels, with no fp64 arithmetic outside the original CAM DOUBLE island.
It is not GPU execution, an E41 cubin audit, or a forecast performance result.
`PHYSICAL_MUTANT.json` records two genuine original-public assertion failures
when MG total ground precipitation is removed.

Runtime code: `column.py` coordinates the original public caller;
`micro.py`, `macro.py`, `aerosol.py` and helpers preserve the numerical bodies;
`_native.py` and `kernels/phys_cammg.py` lower one column per Pallas program.
`adapter.py` supplies original REAL operands and returns transported fields;
`binding.py`, `registry.py`, `restart.py` and `radiation.py` hold the caller
state and mediate the existing pipeline seams. Modal initialization is a
tracked runtime asset with a selected AOT provider and collision regression.

CPU continuation uses authentic Swiss input with only mp_physics changed.
Its explicit CPU policy is FAST_DEFAULTS=0, CENSUS=1, RRTMG column tiles128,
6 GiB/no-swap scope, and cache clearing between phases. These execution and
memory choices do not alter model operands, physics functions or cadence.
Full coupled WRF trajectory/D6 and actual GPU resource/roofline gates remain
pending manager scheduling.

Shared test contract updates add mp11 to the accepted selector set, append
its 20 optional State fields, supply four new namelist defaults in the fake
binder object, and advance the PhysicsStepSpec inventory to147. Numerical
assertions and tolerances are unchanged.

The first critic rejected the PBL caller seam despite the accepted numerical
body gates. `PBL_CALLER_AUDIT.json` pins the original shared-array call list:
11 PBLs write EXCH_H; seven can write shared TKE_PBL. Shin-Hong writes TKE
only for `shinhong_tke_diag=1`; Registry default 0 holds its cold REAL4
`EPSQ2L/2=0.005` value. The flag is bound for every Shin-Hong run, through both
surface paths and JIT/static keys, independently of microphysics selection.
CAM-MG consumes the shared held/diagnosed profile. Nondefault diagnostic-1
deviation is an accepted, disclosed v0.4.0 caveat (manager decision); the
investigation is preserved on `lane/s2-cammg-diag-v041` for v0.4.1.
(Superseded: lane t-shinhong corrected the kernel and caller before v0.4.0,
op-by-op BYTE vs original WRF incl. `shinhong_tke_diag=1`, `proofs/t_shinhong/`;
history TKE_PBL/EL_PBL wiring: `proofs/w41_shinhongtke/`.) The current
caller resubmission leaves the numerical kernel unchanged. Nonwriters retain their held
values. New BL5/BL1 producer-to-consumer tests reproduce original C14 step2
Nc/QNDROP/WSEDL byte values and fail on the old dropped-field route. Additional
tests pass through the actual post-RK diagnostic restore, preserve unassigned
top memory, count one PBL invocation and retain CAM REAL dtype/object identity
inside legacy precision wrappers. These are caller gates, not a new numerical
qualification of the unchanged PBL or CAM-MG kernels.
