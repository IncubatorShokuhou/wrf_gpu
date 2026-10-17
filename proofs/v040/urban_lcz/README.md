# SLUCM LCZ, morphology and anthropogenic heat

Full worker deliverable is `sf_urban_physics=1` with Noah-MP4. Canonical WRF4.7.1
LCZ51–61 map to eleven URBPARM_LCZ classes; ordinary ISURBAN maps to class5 when
LCZ is selected. BEP/BEM LCZ is explicitly refused (manager scope decision).
See [input contract](../../../docs/urban_lcz.md) and
[undefined pristine locals](UPSTREAM_WRF_BUGS.md).

## Scientific gate

`build_oracle.py` compiles unmodified pristine WRF urban code and the verbatim
Noah-MP urban caller with compiler zero/false initialization. Four frozen
REAL4/REAL8 default/heated fixtures each exercise25 separately called one-cell
regimes for30 steps, all
11 classes, active/disabled morphology and street-width fallback. Pre-registered
limits are REAL8 error <=1e-10 times reference max, and REAL4 <=3 times the
WRF REAL4/REAL8 spread +4 ULPs. All four pass (LCZ10;107 total suite passes).
`build_heat_clock.py` independently compiles pristine solar geometry and literal
surface/PBL AHE statements. Clock/month and routing tests pass. Material
zero-vs-inf pristine initialization sensitivity is disclosed; NUDAPT remains
experimental. LF(1) parity is one-cell zero-init only: WRF reuses undefined
U10/V10 between tile cells and compiler initialization does not reset each cell.
The [sealed critic multi-cell diagnostic](evidence/multicell/README.md) reports
HFX 6.273636 W/m2, U10 0.992088 m/s and TSK 0.301880 K differences for two identical
array-input/state cells. Five first-cell fields are BYTE with the frozen oracle,
and an isotropic control has zero differences. This is not whole-tile parity;
coupled original CPU-WRF/D6 qualification is pending. The four one-cell gates
and their tolerances are unchanged; LCZ without morphology is unaffected.
The pristine checkout and original Swiss meteorology are preserved.

`evidence/numerical_reuse.json` seals unchanged kernel bytes and numerical
helper ASTs. Numerical tests are carried forward through metadata/caller edits,
with changed interfaces tested separately. The shared consumer cohort is320
passed nodes (LCZ14 first77 before timeout; LCZ15 remaining243). Additional
changed-seam, NetCDF lock, AOT key, cross-bundle and restart checks pass; see the
final qualification receipt. Counts are separate cohorts, not a deduplicated sum.

## Coupled and packaging gate

`prepare_swiss.py` copies real Swiss weather into an isolated synthetic city
patch with12 LCZ cells, canonical WRF morphology names, consistent LANDUSEF,
monthly AHE25 W/m2, hourly AHE40 W/m2 and active table AH/ALH. This is integration
smoke, not original-WRF trajectory validation. The generator and all inputs are
sealed in `evidence/city_input_seal.json`.

`coupled_smoke.py 1` and `2` execute two actual native-loader CPU-interpreted
production steps and compare every evolved carry leaf after public restart.
The actual production device commitment is applied to restored carries. CPU
Pallas interpretation, RN32 emulation and the CPU Thomas solver are declared;
these are not GPU executable or speed claims. Original receipts remain alongside
final composed receipts. The earlier monthly raw resume timing included an
extra placement compile and must not be called warm timing.

Two real source mutations (LCZ ISURBAN class5 ->2 and deletion of outer clock
forwarding) are killed by genuine assertions, with no setup errors. The clean
archive gate loads tracked fixtures without an external pristine checkout.
OFF identity uses the accepted `proofs/s1_port/release_identity.py` native-facade
trace, through a wrapper that redirects scratch to this lane. Frozen Swiss
2bf12cb8 and PROD d01 6fc715a2 are required. PROD is WN2 iteration evidence only.

## Remaining release qualification and follow-up

GPU timing, launch count, VRAM, E41 and full forecast D6 for the new inputs remain
unmeasured. Existing ordinary SLUCM GPU qualification is distinct. Geometry is
prepared once on the host; all forcing selection and clock updates remain in
the device step. Input carry size and CPU interpretation timing are disclosed
in the input documentation and receipts.

A separate BEP/BEM LCZ lane must group per-type geometry, prognostic material
state, surface/radiation carries and PBL sources, restore their type-index
mapping, and qualify original-WRF eleven-class mixed-city callers. The current
multilayer bundle's single default UTYPE2 cannot represent this. Doctor,
namelist binding and catalog preserve this refusal while ordinary urban2/3
remain admitted and restartable.


## Reviewed rebase for integration

[REBASE_QUALIFICATION.json](REBASE_QUALIFICATION.json) binds the approved98f06745b
feature after rebase onto main2eb72fac1 (measured source26bf7472717d). Main's
Goddard/feedback/grid-FDDA ordering and precision checks are preserved, with
LCZ additions appended in conflict blocks. Requested suites produced470 passes,
11 skips and2 expected failures. Five older M7 failures reproduce with identical
test IDs and failure signatures on the exact plain main base; they are disclosed
in the retained control, not hidden by skips or changed assertions. The native
release trace matches the full frozen2bf12cb8 hash. LCZ numerical/input/schema
files are BYTE with the approved source. Original column results, one-cell LF1
experimental disclosure and sealed multi-cell diagnostic remain unchanged.
