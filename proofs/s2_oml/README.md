# OML1D and hurricane-flux qualification

This package qualifies the opt-in CPU port of `sf_ocean_physics=1` and
`isftcflx=1/2` on surface layers 1, 5 and 91. GPU execution and coupled
CPU-WRF forecast deviation remain pending. The executable smoke uses real
Swiss geometry/meteorology with two explicitly synthetic warm-water cells;
it is a finite/carry/caller check, not forecast-fidelity evidence.

`fixtures/manifest.json` and `flux_manifest.json` bind the original WRF 4.7.1
source hashes, compiler options and active regimes. Generator scripts copy
the original source bodies unchanged into private build directories. The
OML literal path compares 24 columns × seven fields byte for byte. Initialization
tests cover all three HML0 modes. Surface-layer fixtures include six warm
option/scheme combinations and 16 MYNN cold-start columns. Compiled tolerances
come solely from pristine input-ULP envelopes and pristine compiler-contraction
spread. See `UPSTREAM_WRF_BUGS.md` for initialization sensitivity and arithmetic
disclosures. No candidate output participates in tolerance construction.

The integration tests cover both real loaders, namelist binding/honour checks,
the current LH and absorbed-shortwave caller operands, initial-array cache-key
stability, seven ocean history fields, and pickle/checkpoint/NetCDF restart.
Active missing fields/LH and changed configurations fail closed. The registered
`PhysicsStepSpec` entries identify the ocean and hurricane selectors separately.

`SOURCE_REUSE.json` records exact file identity for reused compiled-component
receipts. Seven CPU component programs have zero fp64 arithmetic and float32
physical array outputs. A legacy weak scalar `SurfaceFluxes.t_skin=-1` is static
metadata. CPU compilation/memory receipts describe components only; they do not
measure GPU launches, traffic, readiness, throughput or peak VRAM. New physical
carry is 32 bytes/cell. The namelist retains up to seven initialization arrays
(28 bytes/cell); enabling previously absent history can retain six more arrays
(24 bytes/cell). The default zero-selector program retains its native identity.

`QUALIFICATION.json` links the immutable source commits, retained measured
receipts, deletion mutant and archive check. The mutant removes ocean evolution
and is killed by a genuine numerical assertion, with no setup errors. Exported
tests include the repository's existing `data` assets as well as these fixtures.
All measurements use CPU affinity, nice 19, no persistent JAX cache and a timeout;
no GPU work was performed. Disposable snapshots/builds are removed after use.
