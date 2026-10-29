# F1/F2 delta on 3293124406

Model `src/` and fixtures are byte-identical to the reviewed source. Original-WRF
recapture, cold moments and default E222/E224 evidence carry forward. No forecast
or physics capture is repeated for these test/launcher changes.

F1: the three compiled oracle/deletion tests launch `check_compiled.py` in a
fresh CPU process with explicit release REAL flags, JIT enabled and
`--xla_cpu_max_isa=AVX`, matching the documented no-FMA original-WRF condition.
The parent pytest process needs no shell flags. The negative child first passes
the healthy full compiled gate, then deletes only the divisor barrier and requires
a genuine frozen-bound assertion. Bounds and native GPU gates are unchanged.

F2: the actual launcher builds six owned cache/temp controls per PDF and unique
run; each PDF starts a fresh child before JAX imports. Native persistence is on,
CPU persistence off. Native export uses the very `Compiled` object returned by the
measured lower/compile, and project E41 scans that retained rich executable.
Unknown/INCOMPLETE fails. Native artifacts are retained separately from expendable
caches. The [plan](GPU_PLAN.md) gives caps, cleanup and remaining qualification.

CPU prefix tests inspect the actual CPU environment and native command prefix,
stopping before any real lock/GPU call. Private simulated exports test object and
path wiring only; they are not native evidence. Actual sequencer CPU smoke exercises
the full numerical child bodies and private-path/configuration checks. Actual GPU
export/E41, speed/memory and coupled RW1–RW4 remain pending.

Performance: model work, native arithmetic, carry, launches and memory are unchanged
by source identity. F1 adds isolated test-process startup. F2 adds two bounded
controller children, private persistence and a required postcompile artifact scan;
these are qualification-arm costs, not forecast timestep costs. Native compile/cache
sizes and export/scan overhead remain unmeasured until the booked native arm.

Validation: 68 unique tests pass across the two overlapping ordinary-root runs
(parent FAST0, XLA_FLAGS empty); F1's three isolated compiled checks pass, including
healthy-before-mutant rejection. The only initial failure was an omitted launcher
source file in the slim archive; after adding that proven dependency, all12 artifact
checks pass (0.42s). Full immutable-archive collection is21,247 nodes/0 errors.
The refreshed actual night_seq_v9 CPU step is rc0/GPU0, with six private controls
per PDF, different PDF cache roots and all disposable directories removed. Source
and fixture equality to3293124406 is checked explicitly. Small receipts and the
new20-minute plan are in `receipts/f1_f2/`; the15 original sealed receipts are retained.
