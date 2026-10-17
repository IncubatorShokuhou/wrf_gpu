"""Multi-device runtime (experimental, opt-in): horizontal domain decomposition of one domain.

Nothing here is imported by the default single-GPU path. ``decomposition`` is pure NumPy/JAX slicing
(partition specs, tile carries/namelists, the halo exchange plan); ``tile_driver`` runs the unmodified
single-domain step program per tile and exchanges halos between devices. See
``.agent/decisions/ADR-O2-MULTIGPU-DECOMPOSITION.md``.
"""
