# Density convention clarification

This addendum supersedes only the density suspicion in `MP_CALLER_AUDIT.md/json`.
The nine missing theta-boundary findings and all dispatcher coverage remain unchanged.

The actual pristine post-RK microphysics chain is `solve_em.F:3750` (call
`moist_physics_prep_em`) followed by `solve_em.F:3809` (call `microphysics_driver`).
That preparer sets `rho=1/(al+alb)` at `module_big_step_utilities_em.F:5542`:
dry-air density. The earlier `phy_prep` at line 4856 uses `(1+qv)/ALT` for
surface/PBL/cumulus physics. They are different boundaries.

Therefore, using a dry-density diagnostic in the generic legacy microphysics
adapters is consistent in convention. **Do not add a `(1+qv)` multiplier there
based on the original secondary suspicion.** Exact diagnosed-ALT versus EOS
rounding is outside this source audit.

The explicit moist-density helpers in NSSL (`physics/nssl2mom/adapter.py:61`),
Ferrier (`physics/ferrier/adapter.py:21–24`) and Goddard 4-ice
(`coupling/goddard4ice_adapter.py:24–27`) warrant a separate check of their
public-wrapper-to-kernel mapping against this post-RK call chain. This is a
source observation, not a proven numerical failure or authorization to edit them.
Only Lin was changed in this lane.
