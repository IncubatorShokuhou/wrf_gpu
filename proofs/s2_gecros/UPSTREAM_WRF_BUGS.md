# GECROS reference viability — BLOCKED

Pristine WRF 4.7.1 at `f52c197ed39d12e087d02c50f412d90d418f6186` has two
independent blockers for the lane's active growing-season oracle. This is a
source and CPU reference diagnostic, not port qualification. `opt_crop=2` must
remain refused. The production port and pristine bodies have not been changed.

## Module parameters are not assigned by the initializer

`phys/module_sf_gecros.F:90–98` declares module variables `YGO`, `CFO`, `LNCMIN`,
`CLVI`, `CRTI`, `NLVI`, `NRTI`, `HTI`, and `RDI`. The comment says they are
computed in the driver, but `phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F:1855`
imports only constants. Its initializer declares local copies at line 1984 and
assigns those locals at lines 2243–2255. `gecros_init` initializes the state
array from arguments; it does not assign any of these module variables.

GC01 compiles the complete, byte-identical pristine crop module and the
verbatim caller initialization expressions and `gecros_init` routine. With
compiler-defined zero initialization and production-style FP traps off, all
nine module parameters remain zero. The direct active-column diagnostic then
evaluates `WSO = CSO / CFO` at crop-module line 476 and stores it in
`STATE_GECROS(45)` at line 934. `YGO` is also a denominator at lines 777 and 798.

Measured results, REAL4, `-ffp-contract=off`, `-fcheck=all`:

| Arm | Seedling | Vegetative | Grain fill | Cold/night seedling |
|---|---|---|---|---|
| pristine caller, `-O0` | yield NaN | yield NaN | yield +Inf | yield NaN |
| pristine caller, `-O2` | yield NaN | yield NaN | yield +Inf | yield NaN |
| wrapper-only all-one parameters, either optimization | finite | finite | finite | finite |
| wrapper-only derived parameters, either optimization | finite | finite | finite | finite |

The zero-arm flux outputs are finite, but that does not satisfy the no-NaN carry
gate. Nonzero controls change only wrapper assignments; they establish
sensitivity and **are not pristine WRF truth**. The four states/forcings are
constructed active boundary inputs, not censused forecast columns. None of
these results establishes a valid growing-season trajectory.

## Cold initialization never reaches the active crop call

`gecros_init` sets `STATE_GECROS(40:42)` (drilled, emerged, harvested) to -1 at
driver lines 2864–2866. The only GECROS timestep call is inside
`IF (GECROS1D(41)>0 .AND. GECROS1D(42)<0)` at
`phys/noahmp/src/module_sf_noahmplsm.F:3958–3995`, inside the `ITER==1` block.
`EMERG` would update slot 41 at lines 11000–11021, but a whole-tree case-insensitive
search finds only its definition/end (and the generated `.f90` copy), no caller.
The other writers of slot 41 are initialization and harvest reinitialization,
both setting it to -1. A cold-start active growing season therefore cannot be
obtained through this caller. A manually activated state, as in GC01, bypasses
this separate lifecycle problem and exposes the nonfinite-yield problem.

## Reproduction and disposition

The measured script is `proofs/s2_gecros/pristine_probe.py` at `7846b84f0`.
Run from a detached snapshot, obeying QUIET and the CPU allocation:

```sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8 nice -n 19 timeout 300 \
  python proofs/s2_gecros/pristine_probe.py \
  <USER_HOME>/src/wrf_pristine/WRF \
  <USER_HOME>/wrf_gpu2_lanes/s2-gecros/GC01/build \
  <USER_HOME>/wrf_gpu2_lanes/s2-gecros/GC01/result.json
```

The report records compiler, exact flags, source/routine/wrapper hashes and all
60 state + 6 flux IEEE bit patterns. Receipt SHA-256:
`f85354ff5e1fbff83272d940753368a2bd8e70fefcab919187c93df34c3330de`.
`GC01_summary.json` retains the compact results and identity proof in git;
full evidence is `<USER_HOME>/wrf_gpu2_lanes/s2-gecros/GC01/result.json`.
The three target pristine files are clean against their recorded source commit.

The standing policy requires mirroring upstream zero initialization and permits
experimental disclosure of material sensitivity. Here mirroring zeros produces
NaN/Inf in a required crop carry, while following the untouched cold caller
never activates the crop. Disclosure alone cannot satisfy both the full active
oracle and no-NaN gates. The manager decision on 2026-10-14 is to keep
`opt_crop=2` REFUSED because no functional pristine WRF oracle exists. The
catalog and doctor explain these upstream defects and point to this evidence.
READY-FOR-CRITIC covers the refusal and evidence only; it does not qualify a
GECROS implementation.

Performance/memory: the change adds a host-side catalog refusal and its tests;
doctor consumes the same catalog reason. It adds zero timestep launches,
temporaries, traffic, synchronization, or model compile cost. No GPU process was
started. Model sources are byte-identical to main at composition; release identity
evidence is carried forward by exact model-source identity. Disposable builds
and detached snapshots are removed after evidence extraction; there is no
retained compile cache.
