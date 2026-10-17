# c01: surface-driver-off history

The original CPU-WRF c01 run holds PSFC, T2, Q2, U10, V10 and TH2 at their
initialized values throughout all 25 frames of the 24-hour forecast. The complete
read-only reference audit is `CN01-held-fields.json` in the lane artifact directory.
`held_surface.npz` freezes a real terrain crop from the original CPU input and
history at hours 0, 1, 2 and 24; `held_surface.json` records full-file hashes,
executable provenance and input namelist hashes. It is independent of GPU output.

Pristine `phys/module_surface_driver.F:1490` returns when
`sf_sfclay_physics == 0`, before `PSFC(I,J)=p8w(I,kts,J)` at line 1988 and before
the surface-layer and land-model diagnostic writes. Dynamical pressure still
evolves. Recomputing PSFC from the evolved atmospheric state for history therefore
compares a different quantity with the CPU's held PSFC. This explains the prior
133.192 Pa PSFC RMSE at hour 2 without identifying a dynamics error.

The fix reads these six host arrays once from wrfinput, preserves them over
output-time diagnostic merges when the surface layer is disabled, and preserves
explicit TH2 in the final writer. It covers native/nested output, daily output,
and sub-hour auxiliary streams. Skin temperature and radiation retain their
own producers. Model-step functions and all seven model source trees are unchanged.
The loaded arrays are host metadata and never become compiled constants or carry
leaves, so they add no baked input to the AOT key or restart schema.

The focused tests write real NetCDF files against the original-WRF fixture,
exercise synchronous and asynchronous daily/auxiliary output, reject NaN/Inf and
wrong shapes, and retain active-surface precedence. Deleting the preservation
produces a PSFC AssertionError with zero setup errors. The frozen Swiss and PROD
release program identities remain f672a573 and 6fc715a2.

Measured host cost on this Swiss input: six arrays, 42,336 bytes retained;
median output merge 490 ns with sfclay off and 203 ns with it active. No new
device operation, timestep allocation or host/device transfer is introduced.
This is a host microbenchmark, not a whole-forecast speed claim.

The actual c01 GPU arm and E41/D6 receipts are recorded separately in the final
lane handoff. Soil schema and other writer inventory work belongs to t-writer;
unbounded fields or incomplete schema coverage are never called numerically qualified.
