# CuPlus critic fixes

The review of `3e7c7bd0b` found a real daily lifecycle gap: all four CuPlus
profiles selected the State-only forecast callback. The existing full-carry
stepper is now activated for cu=7/10/11, preserving W0AVG, held cloud/rate
fields, countdowns and the model clock across hourly/auxhist boundaries.
Its existing full-carry checkpoint branch and CAM entropy guard now execute.

Actual daily host-loop tests cover all four pairings, hourly and 15-minute
auxhist segments, nonzero held fields, a checkpoint at step 200 with byte-exact
full-carry restoration, one initialization and monotonic steps through 400.
CAM's sticky fatal flag raises at both hourly and auxhist boundaries. The CU0
control retains its State-only route. Model chunks and output IO are recorded
stubs; the dispatcher, stepper, clock, guard and restart codec are production.
These are host lifecycle checks, not new scientific forecast measurements.

The early component document is explicitly historical and links the current
experimental qualification. Stale parcel/CPS/CMT admission headers are corrected.
The extra production EOF blank is removed. All 11 relevant kernel/helper
executable ASTs and all 24 immutable receipt hashes match the reviewed source;
the original scientific evidence is reused. That pre-glacier fix-delta trace
matches `2bf12cb8…`; its receipt remains unchanged. The subsequent approved
glacier merge uses Swiss `f672a5739cb9…` and unchanged PROD d01 `6fc715a2…`,
recorded separately in `POST_GLACIER_IDENTITY.json`.

The fix adds one host selector predicate and reuses existing carry/stepper code.
It adds no physics solve or timestep-kernel implementation, and no in-step host
transfer. Existing
per-option carry, temporary-memory and CPU cost receipts remain applicable;
GPU speed, peak VRAM, readiness, E41 and full forecast D6 remain pending.

Two existing daily tests also exposed main's direct ocean-selector attribute
access on legacy dictionary fixtures. The neighboring safe-access convention
restores those defaults; typed operational namelists select the same route.

Whitespace checks exclude only the hash-bound pristine timescale excerpt
`mskf_timescale/original_timescale.f90` and the retained historical proposal
`CAMUW_CUMULUS_SEAM.patch`. Their original bytes are preserved; production code
passes `git diff --check`. Exact test/source receipts are in `CRITIC_FIXES.json`.
