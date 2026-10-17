# Same-critic qualification delta (manager 2026-10-14 23:2xZ)

UPDATE 23:34Z: manager explicitly opens 12 GiB singletons NOW through
2026-10-15 19:15Z; ALISIOS 20261014_12z DONE. All five v013 nodes run now,
each alone through with_ram_slot --max 12G. No GPU work assigned to this lane.
CPU plan changed: use core16 only, rather than the old16-19. Keep the already
queued first P3 column arm at6G; use12G for known large compiles, with the proof
early-stop at10.25GiB (--max-gib12 --limit-gib10.25). Slot count is manager4.
The historical constraints below explain the first arm; this update governs.

The sealed QUALIFICATION.json remains the original partial packet. The newest
manager instruction supersedes HEAVY_PLAN's resource deferral for column and
Swiss checks: run all five expanded native numerical projections and all five
actual Swiss two-step/full-restart checks NOW, each in an official <=6 GiB slot.
Only the five affected standard v013 nodes remain tomorrow's exclusive 12 GiB
singletons after ALISIOS nightly DONE. Return the delta to rv-t-icediag.

Use a committed detached snapshot rebased on main, CPU16-19 nice19, JAX CPU,
GPUWRF_JAX_CACHE=0, FAST_DEFAULTS=0, CENSUS=1, OMP/OPENBLAS threads1. The usual
same-line QUIET guard precedes each heavy command and is repeated after the RAM
queue. All arms have a timeout. ram_guard.py terminates the owned child process
group at 4.75 GiB sampled cgroup memory, before the official 6 GiB OOM limit.
Such a stop is a RESOURCE result, never scientific FAIL or PASS.

Numerical bounds and original fixtures remain unchanged. P3 requires all
retained native outputs within the existing P348/P360 bounds. Jensen's original
ULP/FMA enclosure remains strict; old operational outputs must be BYTE before
reusing its prior disclosed deviations. DISCLOSED_EXCESS requires the same
critic's assessment. Save actual arrays so any failures can be localized rather
than rerunning a compiler unnecessarily.

Swiss uses the real RD11 CPU case, full RK/physics, two actual own steps and
full restart BYTE. Cache release between phases follows the already measured
s2-cammg proof; no model operand/function/cadence changes. Confirm same-step
producer diagnostics BYTE in the carry interior, dtype/finite/guards and the
actual producer timestep sequence. Any guards need a matched control.

No GPU, E41, VRAM, forecast D6 or new speed claim is made by this CPU delta.
