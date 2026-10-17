# Pristine-WRF post-RK negative-value investigation

This lane measures original WRF 4.7.1 on the exact Swiss initial, boundary and
namelist inputs used by v-run's release-default (Thompson) and WDM6 forecasts.
It changes no gpuwrf model code, limiter, guard, threshold, or release gate.
The final numerical disposition is in REPORT.md when both runs and their E26
checks have completed. Interim output is not a qualification.

## Observer and identity checks

`prepare_build.py` operates only in the private lane copy of the retained
original WRF build. It adds read-only calls after `END DO Runge_Kutta_loop` in
`solve_em.f90`, before microphysics and its cleanup. Existing diagnostic paths
are redirected into the lane. Removing those calls and restoring the paths
recovers the original preprocessed source byte-for-byte (FN05). The static
archive has 404 original objects: 403 remain byte-identical, only `solve_em.o`
is rebuilt, and the observer is one new object (FN06).

`observer.f90` receives arrays with `intent(in)` and writes only its own buffer
and files. It records every negative value and its i,j,k coordinates, and
counts every nonfinite value for each active field at each of 4,800 steps.
The buffer is 1,241,856 bytes per call for the 42x42x44 mass grid. The stream
format is big-endian int32: `(step,count,nonfinite_count)`, followed by `count`
records `(i0,j0,k0,float32_value_bits)`. The horizontal outer ring includes all
vertical levels. A missing species is absent, not an observed zero.

`analyze.py` requires all 25 hourly frames and every original output variable
to be byte-identical to the uninstrumented serial reference. The default
reference is v-prep/c00: its namelist differs only in whitespace and equivalent
290./290.0 and 5000./5000.0 spellings, and both input files are SHA-identical.
The older RD11 scientific reference used four MPI ranks; it is not substituted
for the serial observer identity check. WDM6 uses s1-gpuqual/mp16/cpu.

The reader checks finite values, negative predicates, indices, record lengths
and step sequence. `verify_reader.py` tests known bytes and five invalid-file
controls; these are diagnostic parser tests, not synthetic physics evidence.
CPU quantiles are retained for all, ring and interior negative values. The GPU
receipt has aggregate counts, region counts and extrema but no quantiles; the
report must not invent a GPU distribution or claim a same-operand replay.

## Source mechanism and scope

Original `dyn_em/solve_em.F:1934-1955` and `1997-2018` call
`rk_update_scalar_pd` on the last RK stage for both moisture and number scalars
when the selected advection option is 1. The routine
`dyn_em/module_em.F:1877-1911` adds the complete frozen scalar tendency to the
old scalar with the old dry mass in both numerator and denominator, then clears
the consumed tendency. It does not impose a nonnegative lower bound.

Original `dyn_em/module_advect_em.F:7733-7774` constructs the low-order mass from
that old field and scales outgoing antidiffusive fluxes. A sufficiently negative
pre-added source can make the old field and low-order mass negative; zeroing
outgoing corrections does not itself restore a nonnegative old state. The
positive-definite advection option therefore does not promise positivity after
arbitrary negative source pre-adds. This is a source explanation, not proof
that diffusion caused every event in the measured trajectories.

The port mirrors those pre-add and update operators in
`runtime/operational_mode.py::_rk_update_scalar_pd` and `_rk_scan_step`. Both
WRF and the port apply the stage's flow-dependent scalar boundary conditions
before the observed end-of-RK seam. Prior hunt-mp HMP15 evidence localizes three
historical QC operands to the frozen-source pre-add; that historical evidence
is cited only for mechanism, never transferred as current-run amplitude proof.

This experiment can establish whether the reported intermediate population
and magnitude classes also occur in pristine WRF under matched forecast work.
It cannot certify every GPU event from identical per-step input operands, or
close F2 for other cases, schemes, nests, or unobserved phases. Full-forecast D6
receipts remain separately owned by v-run. No output clipping is introduced.

## Resources and reproducibility

The two original serial forecasts run at nice 19 on the granted contract cores
16 and 17. CPU work sets `JAX_PLATFORMS=cpu`; no GPU command is used. Build and
forecast commands are bounded by timeouts and honor QUIET. There is no persistent
JAX cache. This is diagnostic instrumentation only: the product's launches,
allocations, traffic, synchronization and compile path are unchanged by source
identity; no GPU timing or speedup is claimed.

External raw evidence is under `<USER_HOME>/wrf_gpu2_lanes/t-f2neg/`.
The final packet binds the source, executable, inputs, reference provenance,
E26 field hashes, streams, population summaries and same-step/location queries.
The private build and disposable caches are removed after their final consumer.
