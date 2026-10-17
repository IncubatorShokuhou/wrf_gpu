# c40 SSiB MP_RE caller fix

SSiB passed the raw namelist `use_mp_re` to the shared RRTMG column builder.
The builder's default radiation selection is the paired RRTMG path, whereas
c40 uses RRTMG-LW and Dudhia-SW. With the CLI MP_RE flag enabled, this falsely
required radius fields that WRF deliberately leaves inactive for this pair.

Original WRF behaviour is sealed in `WRF_SOURCE.json`:

- `module_physics_init.F:985–1027`: qualified LW **and** qualified SW are
  required to set `has_reqc`, `has_reqi`, `has_reqs`; Dudhia fails that condition.
- Initial radius backgrounds are written only for nonzero flags (:1267ff).
- Thompson calculates effective radii only when all three flags are nonzero
  (`module_mp_thompson.F:1466–1477`).
- RRTMG-LW receives the fields and flags, then uses its flag-zero fallbacks
  (`module_radiation_driver.F:2113`; `module_ra_rrtmg_lw.F:12178ff`).
- Dudhia has no `re_cloud`, `re_ice` or `re_snow` interface.

The fix passes `mp_re_active(namelist)` and the explicit `mp_physics` identity,
matching the standard operational refresh. No radii are invented or seeded.
The column equations, other caller arguments, carry and restart remain intact.
No new tables, assets or baked inputs enter the production program.

Measured CPU-only checks:

- Actual c40 baseline: held-radius refusal, 15.56 s.
- Fixed full CLI history segment: native CUDA lower PASS, 72 steps / radiation
  cadence 12; 55.615 s, 2,888,500 KiB peak RSS, 128 Triton custom calls.
- Three new tests pass: two actual radiation refreshes through Dudhia and
  RRTMG-LW with unseeded radii for both `use_mp_re` values, plus selection and
  MP-identity forwarding. This is caller plumbing, not forecast fidelity.
- Restoring the original hook causes the actual held-radius refusal: one
  genuine failure, zero setup errors.
- Both official default identities are checked in the sealed receipts.

Existing-consumer limits are preserved: FAST1 gives 98 passes / 15 failures;
unchanged-main control reproduces the same 15 failed nodes. Standard CPU FAST0
gives 101 passes / 12 failures, all in that pre-existing set. The failures
concern missing `history.u10e` in restart/history cases, with additional wind
message and object-identity assertions under FAST1. They are not recorded as
passes or repaired by this radiation fix. Required contracts, MP refusal pins
and NetCDF-lock checks pass. See the exact failure inventories in the packet.

Resource consequences: the predicate and MP identity are Python-static values.
No device operation, field allocation, kernel call, transfer, or new carry leaf
is added by resolving them [source/I]. Lowering resource use is measured above;
GPU timing, VRAM, executable E41 and full forecast D6 remain unmeasured here.

`QUALIFICATION.json` binds original-WRF excerpts, source identity, lossless
receipts, the pre-existing failure inventory and the default hashes. MR01–MR10
are logged in MAIN. Use the lane's QUIET, CPU-core and 6 GiB / zero-swap controls
with the three proof scripts; no GPU is required.
