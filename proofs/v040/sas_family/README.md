# SAS and old-KF qualification evidence

`QUALIFICATION.json` describes the complete 4/14/94/95/96/99 lane scope.
Original WRF 4.7.1 modules are the scientific reference. CPU coupled smokes,
default-program identity, and byte comparisons between port revisions have
separate purposes; none is a GPU or full-forecast fidelity claim.

The fixture manifests record original source/input/output hashes and compiler
flags. Native 14/96 fixtures contain 455 columns of 44 levels. OSAS95 retains
the actual original random draws at two cadences. Old KF99 includes five
soundings, and `kf99_variants.json` records seven additional phase/NCA witnesses.
`regime_census.json` distinguishes observed coverage from exhaustive coverage.

Rebuild tools operate in a private lane artifact directory, never in pristine
WRF. `native_batch.py build WORK 14` (or 96), `build_sas94.sh`,
`build_osas95.py`, and `build_kf99.py` stage the corresponding original modules.
The probe and phase-observer scripts retain raw failures as well as passes.
Original source observers must reproduce complete original stdout BYTE before
their intermediate values are interpreted. OSAS additionally replays the
recorded draws, since a fresh original SYSTEM_CLOCK run is not reproducible.

Run CPU tools with `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, pinned to an authorized
core at nice 19, with a timeout and the repository QUIET guard. The five
`tests/test_v040_{sas94,native_sas,osas95,osas_rng,kf99}.py` files cover the
scientific fixtures, caller packing, cadence, deterministic RNG and restart.
Shared contracts, CLI consumers and NetCDF locking are also required.

Native contraction deviations are explicit in the manifests,
`kf99_conditioning.json`, the catalog, and `docs/SAS_FAMILY_SUPPORT.md`.
Their original frozen bounds are not reported as passed. No tolerance file or
product ISA setting changed; Owner decisions and the standing conditioning
policy apply, and final coupled D6 remains binding.

`native_hlo_final.json` records zero optimized-HLO float64 instructions for
native 14/96/99 and byte-exact outputs versus the pre-fix port. The only precision
fix types exact zero/one selectors. 94/95 retain original WRF DOUBLE internals.
These are CPU resource receipts; GPU launches, VRAM, E41 and roofline remain
unmeasured. `cpu_coupled.json` records the actual resolved work of every smoke.


The existing critic's two required fixes are sealed in `CRITIC_FIXES.json`.
`build_boundary.py WORK` stages unchanged original boundary-consumer bodies;
`fixtures/boundary_manifest.json` authenticates four ownership regimes. The
new boundary and evidence tests exercise original ranges and counted NaN/Inf
rejections. `rescore_saved.py LANE OUTPUT` checks paid finite scores against
prior gates without rerunning physics. `cpu_coupled_boundary_fixed.json` holds
final affected coupling receipts. The raw census has 66 entries (22 by 3);
earlier 69-entry summaries were incorrect. Raw receipts remain unchanged.
