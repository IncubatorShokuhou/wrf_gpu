# X67 attribution: pre-registered plan (CPU preparation; GPU NOT BOOKED)

Hypothesis: one of RC items TOPO, QSQ/EL cold-start seed or E84 solar/ozone
division changes explains d03 RAINNC RMSE at lead 67 rising from ~0.50 mm
(v0.4.3) to 1.0600553865983122 mm (v0.4.4). Reference is original CPU-WRF,
not an older GPU run. A3 interpretation and causal attribution are separate.

Freeze at measured RC `286e2442a9dbf812c43b60a18196a011018f2d22` rather than
current main (which contains later V05PRE changes). [SOURCE_VARIANTS.json](../../scripts/rw3_rainfloor/SOURCE_VARIANTS.json)
pins three immutable private commits and source trees. Each is RC plus exactly
one reviewed source-only reverse patch: NO_TOPO (item d0da043c86), NO_QSQ
(e7c89892a5), NO_E84 (5424ce9959). The forward patches are archived beside the
manifest. Private indexes applied all three reversals cleanly; no shared source
files were changed. Commits are retained at `refs/experiments/v45-rainfloor/<arm>`.

Run each arm once, in NO_TOPO / NO_QSQ / NO_E84 order, with unchanged WN3 0227
facade inputs, all three nests, dt 54/18/6 s, release defaults, initial + hourly
history, 72 h (73 frames/domain). Existing v0.4.4 baseline metadata and CPU-twin
measurements are reused; there is no redundant baseline forecast. Each arm has
its own cold 3 h setup (timeout 1800 s) and warm N72 (timeout 720 s). N72 cap is
1.57 × the measured 459 s, allowing host/startup variation. The cold cost is
reported separately. Run the actual GPU executable's E41 cubin/HLO gate after
cold setup; manager records the pin-bound `COLD_GATE_PASS.json` before warm.
Any timeout, nonfinite, backend mismatch or E41 failure stops that arm; never
score partial output or silently retry. Unknown E41 is not PASS.

After each N72, pipeline CPU scoring against the preserved original CPU-WRF
0227 reference (facade stage only): `scripts/wn3_score.py --hours 72 --rw3-floor`
from this lane's reviewed scorer. Preserve raw D6, every variable/lead,
integrity, and the separate A3 report. Extract X67, RAINNC RMSE at leads 60–72
and hourly increments. The two endpoints are ~0.50 and 1.0600553865983122 mm;
report the signed recovery fraction `(1.0600553865983122-X67)/(1.0600553865983122-X43)`
using the exact archived X43 when staging the reference. A fraction ≥0.75 is a
pre-registered “dominant contributor” diagnostic, **not** a fidelity tolerance.
No individual reversal reaching it means interactions remain open; measure a
combined reversal or commit bisection next, after manager review. Do not turn a
diagnostic reversal into a release fix without pristine-WRF causality/gates.

Report CLI bench JSON wall/stepping s/fc-h, cold/readiness cost, peak VRAM/RSS
and compiler memory alongside X67. These historical diagnostic reversals are
not product performance improvements; any proposed fix still needs unchanged
resource gates and RW1–RW4 on the final composed candidate.

Launcher: [entry.sh](../../scripts/rw3_rainfloor/entry.sh) accepts `<ARM> <cold|warm>` and is intended for
manager `night_seq_v9.sh`/v10. It requires GA_START/GA_END (both checked) and an
explicit manager booking JSON via `V45_RAINFLOOR_BOOKING`: `approved_by=manager`,
`disk_approved=true`, `case=<facade copy>`, and `arms[ARM]` containing `pin`,
`snapshot` (clean detached worktree), `run_root`. A lease alone cannot launch it.
Every GPU child goes through the official lock, has a private cache, and asserts
the GPU backend in the measured process after product allocator setup. No
environment dumps or profiling reports are produced.

DISK-PLAN (manager approval pending): sequential arms only. Runtime snapshot
~2.5 GiB worst case, private cache estimate ≤4 GiB, N72 history ≤23 GiB,
receipts ≤0.1 GiB; allow **30 GiB** per arm plus the ALISIOS **365 GiB** floor.
No new runtime data on <DATA_ROOT> is authorized by this plan. Manager must book
disk and a snapshot path; current CPU preparation uses only source in this
worktree and <3 GiB under the lane's /home directory. Hash histories, finish
their CPU scoring, then delete that exact disposable history/cache/snapshot
before the next arm; preserve original CPU references and small receipts.
Do not run while QUIET. CPU smoke uses an immediately deleted `src` archive,
the exact duration/CLI options and CPU `--dry-run`; receipt is **PLAN_ONLY**,
GPU numerical/E41/performance gates **NOT_APPLICABLE**. A real first-root
smoke needs a separately booked >8 GiB scope if this WN3 configuration needs it.
