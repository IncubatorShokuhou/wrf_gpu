# Noah-MP option qualification

All defined natural-soil WRF 4.7.1 Noah-MP controls are implemented, bound in the
native and daily pipelines, admitted, and covered by guarded restart. The
admitted values and pairing restrictions are in [the option guide](../../docs/noahmp-options.md).
The lane supplies 50 PhysicsStepSpec variants. Main supplies the separately
qualified glacier options and MMF run5; its MMF model and binder are unchanged.
The manager deferred Noah-classic mosaic to v0.4.x. Noah-MP ignores that WRF
control. Both paths refuse it with their specific reasons. GECROS crop2 keeps
main's upstream-broken refusal. Soil4 and other undefined source choices refuse.

## Original-source evidence

Twenty tracked numeric fixtures come from untouched pristine WRF 4.7.1 routine
or caller blocks, with source hashes, compiler flags, units and input bounds in
the fixtures, [index](FIXTURE_INDEX.json) and [manifest](fixture_manifest.yaml).
The pristine commit is `f52c197ed39d12e087d02c50f412d90d418f6186`.
Each builder uses a private build directory; upstream source is never edited.
The binding compiler initialization policy and Dynamic VIC sensitivity are in
[UPSTREAM_WRF_BUGS.md](UPSTREAM_WRF_BUGS.md).

The final native profile adapter was checked once against the affected frozen
original references: 9,912/9,912 literal REAL4 outputs are byte-identical across
carbon, crop, groundwater, equilibrium water, prognostic water, runoff,
advanced water and the three new original-water branches. The separate actual
surface-cap blocks add 240/240 byte-identical outputs and 8 active caps across
48 savepoints, with 2,016 independently varied input-ULP variants. Earlier
snow, irrigation, SST, surface resistance, phenology, canopy, thermal, soil and
crop-phenology evidence is retained in receipts rather than rerun.

Literal operation-order parity and compiled floating-point precision are
separate checks. Independent input-ULP bounds and nominal tolerances remain
frozen. Some compiled outputs exceed those bounds slightly; the measured
excesses remain disclosed in [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) and the
receipts. This does not establish coupled forecast D6 fidelity. The unchanged
default water3/inf1 path retains its previously observed source-order gap;
the three newly implemented branches have the 264/264 byte proof.

The post-MMF source audit specializes only absent-MMF host routing and proves
four functions retain every other AST node. Twenty-seven physics files are
byte-identical to the pre-MMF candidate. The native STC3 FSNO wiring repair
uses an existing block operand; whole-Noah-step canonical jaxprs for STC1 and
STC2 native, and STC3 vector, are exact before/after. The STC3 native smoke ran
after the repair. These are validation-reuse proofs, not fidelity comparisons.

## Composed CPU integration and restart

Four three-step Swiss arms use real geometry and forcing, with four explicitly
declared crop/irrigation cells and active tile-drain soil in the relevant arm.
All have zero nonfinite values, zero strict guard events, and byte-identical
step3 after a step2 pickle save/resume. They execute the real coupled loader
and stepping path; the retained CPU Thomas implementation is declared in the
harness. All other release fp32 and fused-radiation entry flags are recorded.

| Arm | Selected coverage | Restart leaves | Cold CPU whole arm |
|---|---|---:|---:|
| A | crop1/dveg5, run1, soil3, RAD1/SFC2/STC3, all mapped irrigation, SST skin | 200 | 606.845 s |
| B | dveg8, run2, soil2, Jarvis/SFC2/STC2, micro irrigation; refreshed after cold LAI fix | 185 | 593.696 s |
| C | dynamic carbon dveg6, run8/infdv3, STC2, flood irrigation | 191 | 557.398 s |
| D | run3/tile2, soil2, glacier2, native STC3, sprinkler irrigation | 186 | 616.660 s |

Arm A has 4 events for each irrigation method and crop PGS0→1 with cold GDD0;
it does not claim active crop growth in that short Swiss window. Active crop
growth/stages are covered by the original multistep crop oracle. Arm B has
4 micro events totaling 0.0828 mm; C has 4 flood events totaling 91.1561 mm;
D drains 0.945203 mm and applies 0.183382 mm sprinkler water. These constructed
map interventions are declared coverage, not operational forecast validation.
All supported carry groups also pass pickle, checkpoint, native NetCDF and
NativeRestartStore guards; missing groups, changed selection/table/map identity,
nonfinite values, shape and dtype mismatches fail closed.

## Cache, default identity and consumer gates

[AOT_INPUT_AUDIT.md](AOT_INPUT_AUDIT.md) accounts for all six new external
readers. Resolved baked scalar controls use main's keyed-provider mechanism.
Selected soil/runoff/tile data use its existing content-keyed static holder.
Carbon/crop/irrigation arrays stay runtime children; content-aware function
identity keys any fused closure that bakes them. Regression checks require
different actual table values to produce different keys, HLO and outputs.
The standalone restart identity binds actual selected tables/maps as well.

The pre-critic shared-consumer gate passed 406 unique checks: 404 initial passes,
then 81 whole binding/archive passes replace the obsolete dveg2 refusal with
an undefined-choice refusal and add a positive dveg2 admission check. The
separate MMF/own-carry composition gate passes 185 unique checks. Contracts,
legacy MP fail-closed, D31, NativeRestartStore, all owned carry formats, key
collisions, full native STC3 caller and active caps are green.

The [R1–R4 corrections](CRITIC_FIXES.md) add requested-selector authentication,
refusal of active groups in both reduced checkpoint readers, and WRF cold
LAI/XSAI initialization for DVEG7/8/9 only. Their affected gate passes 187 checks
with eight explicit compiled XFAILs and zero failures/errors. The XFAILs retain
the original precision bound violations and are not numerical PASS claims.
All 24 advanced-water literal modes remain BYTE. The B smoke above is refreshed;
A/C/D and unchanged numerical/native evidence are carried forward with source proof.

Default Swiss identity is
`f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD d01 is
`6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
Full results, fixture manifest/archive checks and the genuine crop-deletion
mutant are recorded in [QUALIFICATION.json](QUALIFICATION.json) and receipts.
Identity probes use the official
native-facade algorithm with only its scratch directory redirected to this
lane. They do not execute a GPU forecast. The genuine mutant deletes actual
crop evolution and is killed by one numerical AssertionError with no errors.

To reproduce component precision, use `precision_gate.py TREE precision_bounds.json
OUT.json` for the early snow/irrigation/skin gates and `changed_precision.py TREE
OUTPUT_DIRECTORY` for the final eight changed families. `surface_caps_gate.py`,
the fixture builders, source-reuse scripts and manifest/archive checks are also
tracked. Reference-building scripts write only to the supplied lane directory.

## Performance and memory

Inactive selections retain the release program and carry prefix. Active
prognostic carry memory is 12 B/cell for SST skin, 48 for irrigation, 24 for
dynamic carbon or 36 for crop, 12 for SIMGM groundwater, and 4 for tile drain.
These are carry bytes, not total option memory. Bound parameter arrays and
cold seeds also remain resident: crop parameters alone contain 19 REAL scalar
maps, twelve eight-stage REAL maps, 15 natural-carbon maps and an INTEGER crop
category, totaling 524 B/cell by schema before aliasing or compiler elimination
[I]. The canopy input temporary described below is additional. Whole-run peak
memory remains unmeasured [U].
There is no new per-step host transfer in these source paths [I]; nsys evidence
is unavailable in this CPU-only wave [U].

Deviceless sm120 compilation of the real native phases gives:

| Phase | Compiler time | Registers | Stack / local |
|---|---:|---:|---:|
| BATS | 3.639 s | 64 | 0 / 0 B |
| carbon | 1.255 s | 64 | 0 / 0 B |
| crop | 23.655 s | 192 | 0 / 0 B |
| run1 | 32.476 s | 180 | 0 / 0 B |
| run2 | 29.727 s | 168 | 0 / 0 B |
| run8 | 38.881 s | 185 | 0 / 0 B |
| tile1 | 43.964 s | 188 | 0 / 0 B |
| tile2 | 50.371 s | 205 | 0 / 0 B |
| RAD1 | 0.221 s cached binary readiness, cold cost unknown | 72 | 32 / 0 B |
| actual SFC2/Jarvis/STC3 canopy | 32.943 s | 168 | 0 / 0 B |

Each new carbon/crop/water phase has one native column call. The six fine soil
substeps use a native loop: the initial unrolled run1 compile timed out at
240 s; the retained loop compiles in 32.476 s. Static component cubin scans have
no nonleaf callee over 1,000 SASS instructions. This is a component prescreen,
not the whole executable E41 gate. The actual canopy has 196 packed REAL4 input
planes (784 B/cell temporary); final whole-run peak memory/traffic is unmeasured.
These costs and register pressure are material opt-in overhead, not speedups.

The run1 source has only WRF's explicit DOUBLE S_NODE island. RAD1 source jaxpr
has no fp64 outputs, but CUDA's generic tanf/cosf argument-reduction slow path
contains six fp64 PTX operations, disclosed without changing original trig
math. Other inspected new phases have zero fp64 PTX operations. No whole-GPU
S2, S1, launches, VRAM, cached readiness, station or full-forecast D6 claim is
made. Those qualification gates remain with the release validation work.
