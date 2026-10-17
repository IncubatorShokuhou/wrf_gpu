# TEMF qualification work

The first boundary is WRF's full `temfsfclay` surface entry point, with the
`surface_driver.F` arguments and `module_model_constants` REAL parameters.
The comparison gate is registered before the first oracle execution: all 20
outputs within 8 binary32 ULP of each record's output-field magnitude, with
zero and nonfinite values checked separately. The cases cover stable,
neutral, unstable, capped Richardson number, zero shear, energy/velocity
floors, heat-flux signs, and wet/partly dry columns over three carried calls.

Regenerate with `build_surface_oracle.py <new lane artifact build directory>`.
The compiler runs on a private copy of the unmodified WRF surface module.
No shared pristine build or reference output is modified.

The full `temf2d` PBL reference is now qualified separately on 24 structured
stable/neutral/dry/moist-convective columns over three carried isolated PBL
calls. All 57,744 scalar outputs are bit-exact to the unmodified WRF REAL
routine with CPU `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp`.
CPU XLA FMA/reassociation effects without those settings are retained as
failed receipts TF01b/TF02b; no tolerance was changed. Source-state ULP/dt
conditioning for tendency subtraction was registered before the PBL run.

The operational pair is opt-in by namelist 10/10 and has a compact REAL
Registry carry, mandatory pairing, and explicit restart schema. Surface
physics receives hydrostatic pressure; PBL receives nonhydrostatic pressure
and the post-surface TSK copied to TSKOLD inside WRF's PBL driver. Noah-MP
receives the TEMF surface diagnostics explicitly, preserving its own CHXY/
CMXY coefficients. It does not re-solve MYNN-SL on the TEMF path.

The reference column uses JAX level recurrences. GPU compile, contraction,
launch/memory costs, an efficient GPU column kernel, and coupled forecast
fidelity remain **pending**. CPU results never establish GPU fidelity or
performance. Default configurations have no TEMF leaves or step work.

The final CPU qualification receipts are in `receipts/validation.json`.
The surface comparison passes all 1,920 outputs at at most 6 ULP; the PBL
comparison checks actual uint32 representations and finds zero differing
values among 57,744 outputs. Active reference regimes include 36 thermal
records, 18 convective-cloud records, and 12 with cloud top above dry thermal
height. Removing WRF's INTEGER taper produces 438 failing field/record
comparisons, one assertion failure, and no setup errors.

The real Swiss 42x42x44, dt=18 s case completes three coupled CPU steps with
TEMF/Noah-MP, zero nonfinite values, and zero census guards. That stable case
does not activate thermals; the pristine-column tests supply the active-cloud
coverage. All required shared consumers pass (253 tests); the composed host
checks pass another 85 tests, including the daily held-history path. Pickle
restart continuation uses evolved nonzero carry, and both restart formats
reject a missing active TEMF schema marker. The clean git archive loads both
tracked fixtures. The release-default program retains its full `2bf12cb8`
hash; this establishes regression identity, not WRF fidelity.

Memory and performance scope: the Swiss TEMF carry contains 4,177,152 logical
bytes, all floating leaves REAL32. The 72-column CPU PBL executable reports
941,696 temporary bytes and a measured 3.158 s compilation in TF09; surface
temporaries are 1,152 bytes and compilation 0.320 s. These are CPU component
measurements, not peak VRAM, GPU readiness, launch counts, or forecast timing.
No GPU executable was compiled in this lane. Full receipts and source hashes
permit reuse of completed numerical validation after the main-branch merge.

Native synchronized checkpoints bind the TEMF pair and domain dimensions in
`nested_identity(...)["temf_config"]` for each domain. Public `RestartStore`
save/read admission validates the existing TEMF schema marker, every Registry
and held field, domain shape, REAL32/int32 dtypes, and finiteness on the already
materialized host payload. Save refuses before publication; read refuses before
device restoration. Active TEMF cannot use an opaque namelist digest without
the explicit configuration binding. Legacy unbound inactive stores and bound
inactive stores still accept absent TEMF markers. The generic finite scan skips
TEMF after its complete validator runs, avoiding a duplicate scan. This is a
checkpoint-boundary correction and adds no timestep work or device transfers.
