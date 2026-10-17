GFDL `ZENITH` calendar state (`module_ra_gfdleta.F:1548–1558`)

The local `MONTH` array is DATA initialized and therefore implicitly saved.
A call with an initial year divisible by four sets February to 29. Later calls
with another initial year never reset it to 28. This affects a test harness that
cycles through distinct initial years in one process; ordinary forecasts retain
one initial year. The scalar support port takes that initial year explicitly.
The native support oracle orders non-leap initial years before leap initial years
to reproduce independent fresh WRF processes without editing any native body.
FG18 initially exposed this: the first year groups were BYTE, subsequent dates
after February differed by one day after the intervening leap-year group.

This is a calendar state issue, not an undefined read. No zero-initialization
policy or physical-body patch is used to hide it. Qualification of a forecast
that changes its initial year within one process requires explicit treatment
of this original WRF saved state.
