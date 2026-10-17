# Host snow/soil column admission

Registered before running the correction tests. These allowances follow the
pristine WRF 4.7.1 REAL operations; they are not calibrated to port errors.
The validator compares source values without clipping or rebuilding them, and
runs before publication and before the first restoration assignment.

`module_sf_noahmplsm.F:826–833` derives active thicknesses from adjacent ZSNSO
interfaces. `:6478–6486` explicitly zeroes inactive snow mass, temperature and
geometry; `:6500–6505` accumulates SWE; `:6508–6533` reconstructs geometry, fixed
soil thicknesses and bulk layered snow depth. `module_sf_noahmpdrv.F:689–692`
constructs ZSOIL from DZS. `:2339–2439` supplies the initialization variant.

Let `u` be half the floating format's epsilon and
`gamma(n) = n*u/(1-n*u)`. For mixed storage, take the coarsest format among the
operands of that relation. Comparisons use float64 host arrays and add their
own evaluation-rounding allowance. There is no absolute epsilon floor.

* Inactive SNICE, SNLIQ, TSNO and ZSNSO slots must be exactly zero (either sign
  of zero is accepted). Active snow and all four soil thicknesses must be
  strictly positive. Each active snow layer must have positive total mass.
* For soil layer `k`, let `S_k=sum(DZS[0:k+1])`, `P_k=S_(k-1)` (or zero), and
  `z_k` its stored interface. ZSOIL needs at most four REAL additions, then
  its endpoint subtraction contributes one rounding. Its thickness's error
  against DZS is bounded by `gamma(5)*(S_k+P_k)`: endpoint errors and one
  subtraction are covered by the composed five-operation bound. The final
  `ZSNSO_k=fl(ZSNSO_(k-1)-thickness_k)` contributes at most
  `gamma(1)*abs(z_k)` when expressed in terms of the stored result. Thus compare
  the recovered thickness to DZS with the sum of those two bounds, plus
  `gamma_host(2)*(abs(z_k)+abs(z_prev)+DZS_k)` for host evaluation. Strict
  positivity is checked independently, so zero-thickness layers never pass.
* Layered mass is the host sum of the six stored SNICE/SNLIQ values. WRF's
  SNOWWATER accumulation uses at most six REAL additions, giving an error
  bounded by `gamma(6)*layered_mass`. For SNOW_INIT, the three nonnegative
  thicknesses partition SNODEP. Its deepest branch forms two differences;
  both intermediate results are between zero and SNODEP, so the partition
  error is bounded by `gamma(2)*SNODEP`. The half-depth branches need no larger
  bound. The common SWE/SNODEP division and each mass multiplication contribute
  at most another two rounding factors; the literal multiplication by `1.00`
  is exact. Therefore the exact sum of stored initial masses differs from SWE
  by at most `gamma(4)*SWE`. The implemented conservative bound
  `gamma(8)*(layered_mass+SNOW)` dominates both the initialization and ordinary
  accumulation bounds (and includes margin for the literal products/three-layer
  sums). Add `gamma_host(6)*layered_mass` for host evaluation of three mass pairs
  and their sum. No observed model discrepancy enters this allowance.
* Layered SNOWH and the bottom snow interface come from sums of at most three
  thicknesses each. Their combined six-operation bound is
  `gamma(6)*(abs(ZSNSO[2])+SNOWH)` plus the two-operation host allowance.
  SNOW_INIT's deepest branch instead forms up to two differences followed by
  three accumulations; it also fits this bound.

ISNOW=0 is **not** required to have zero bulk SWE/depth. WRF permits shallow snow
without explicit snow layers; its three snow slots are zero and soil geometry
starts at the soil surface, without a bulk-SNOWH offset. No 0.025 m admission
threshold is imposed: snowfall creates a layer at that threshold, but the later
SNOWH2O update can change the shallow bulk depth before the next creation check.
SNOW and SNOWH must vanish together, as in COMBINE/SNOWH2O (noahmplsm.F:6725–6730,
7153–7173). No density or snow-layer-count threshold is inferred from observed
Swiss output.

The original transfer, initialized carry, timestep and default program are
unchanged for admitted data. The additional work is bounded host column reads,
float64 comparison arrays and masks; it adds no GPU work, transfers or persistent
carry. Correction timing is measured separately on the changed admission path.
