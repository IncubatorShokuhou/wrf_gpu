Pristine WRF 4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.

`DYNAMIC_VIC` (`module_sf_noahmplsm.F:8261,8347`) reads `YD` before assignment on its low-capacity branches. The binding wave policy defines those reads with compiler initialization to REAL4 zero, FP traps off, without editing any pristine routine. `RR1(0)=0` then selects the documented iteration-2 branches; their alternative iteration-30/3 paths are unreachable under this policy. The port records zero at these exact sites and retains the original 20-iteration convergence criterion and arithmetic.

LAND29 compared the identical pristine bodies with `-finit-real=zero` and `-finit-real=nan`: all outputs were finite, but 52/480 values changed, maximum absolute difference 0.0026633054949343204. This is materially initialization-sensitive. `opt_run=8` is experimental with that disclosure; it has source-order qualification under the defined zero policy, not generic compiler-independent WRF forecast qualification. Final D6 against CPU-WRF remains pending. `runoff_oracle.json` contains both initialization receipts, source/table hashes and the original compiler flags.

`opt_run=8,opt_infdv=0` leaves `FSUR` unassigned (`:8209-8215`); require an explicit defined infiltration choice 1/2/3. The released default run=3 keeps infdv=0 unchanged.

`TRANSFER_MP_PARAMETERS:1704-1707` skips `FRZX` assignment for soil category 14. The optional layered-soil binding mirrors zero at that site. The driver's preceding non-XICE category-14 fallback (`:911-914`) replaces all four categories with 7; ordinary modeled soil therefore uses the assigned branch.

Liu crop's `PGSXY` is not initialized by `NOAHMP_INIT`; preserve WRF's allocated INTEGER zero before the first PHENOLOGY call, then let original GROWING_GDD assign stages 1..8. Crop categories over water/barren/glacier/disabled urban are reset to zero by NOAHMP_INIT, so CARBON_CROP's unassigned early-return PGS is unreachable for an active bound crop.

GECROS (`opt_crop=2`) remains scientifically refused per the manager's upstream audit; retain the exact main catalog reason and `proofs/s2_gecros/UPSTREAM_WRF_BUGS.md` reference.
