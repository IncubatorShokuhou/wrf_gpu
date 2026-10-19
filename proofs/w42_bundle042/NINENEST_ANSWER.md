# 9-nest arm in the v0.4.2 bundle session? (manager PLAN QUESTION 2026-10-19T07:16:36Z)

**Answer: no, it does not fit the ≤ 120 GPU-min session.** A cold 9-nest priming arm needs about 33–53 GPU-min [I]
on top of the ~110 GPU-min already planned. ALISIOS should prime the 9-nest in its own consumer layer over the sealed
v0.4.2 bundle (their QUAL_PLAN already budgets an N16 cold priming with ≤ 45 min compile in their 20.10 08–13Z slot).

Labels: [M] measured, [I] inferred, [U] unknown. ALISIOS reads: none of model inputs or CPU-WRF outputs (only their
plan documents in `MW/nightly-engine-148/`, cross messages, and the w41-v033impact guarded geometry receipt).

## Receipts

`grep -i 'ninenest|9-nest|9nest|W9N'` over FINDINGS / EXPERIMENTS / INBOX / MAILBOX / lane notes finds **no v0.4.x
9-nest GPU run**. What exists:

- v0.21.0, an old tree: cold compile of the root step ~6157 s (CANARY-BENCHMARK-LEDGER). Not transferable.
- v0.3.3 ALISIOS nightly: ~30 GPU-min per 72 h night (lease request of 2026-10-12). Their RUNBOOK (S28) says 30–75 min
  per run.
- w41-v033impact: the W9N geometry and the code path, from guarded namelist reads. Not run end to end.

So every 9-nest number below is scaled from v0.4.1 WN3 measurements.

## Geometry [M]

Source: `<USER_HOME>/wrf_gpu2_lanes/w41-v033impact/TMN_LAND_INVARIANCE.json`. alisios_mgr confirmed W9N == WNN9.

- `max_dom` 9. `parent_id` 1,1,2,2,2,2,2,2,2. Ratio 3. dx 9 / 3 / 1 km.
- `e_we` 121,268,112,58,55,58,73,94,73. `e_sn` 71,118,94,73,52,55,79,112,73.
- d01, d02 and d03 have exactly the WN3 dims. Six more 1 km leaves hang under d02, so the fused d02 program carries
  **7 leaf children instead of 1**.
- The leaves hold 41,382 cells, 4.0× WN3 d03.
- dt-weighted work is 474,555 against WN3's 195,024, i.e. 2.43×. This matches the ALISIOS QUAL_PLAN figures.

## Estimates for v0.4.x [I], scaled from v0.4.1 receipts [M]

| item | WN3 3-nest on v0.4.1 [M] | 9-nest on v0.4.x [I] |
|---|---|---|
| cold compile / load overhead | 628 s for d01 + fused (d02 + 1 leaf); first hour ready at 625 s (A3); v0.4.0 775 s | fused d02 + 7 leaves ≈ 4× the fused HLO → **25–45 min**, plus d01 (0 if its key equals WN3's, else ~3–4 min) |
| warm stepping | 6.61 s/sim-h steady (A3 72 h) | **19–31 s/sim-h** (leaves scaled by cells (4.0×) up to latency-bound per leaf (7×)) → 72 h ≈ 25–40 min |
| cold priming arm, `--hours 3` (E187: hours divisible by 3) | — | **≈ 33–53 GPU-min** |
| cold priming arm, 72 h | 18:27 | ≈ 55–90 GPU-min |
| VRAM per PID | 5,304 MiB peak (A3) | resident state ×1.62 (81,021 vs 49,962 cells), leaf temps reused one after another → **≈ 6.5–8.6 GiB** |
| host RSS, 72 h | 12.4 GiB (A3, 219 files) | ≈ 15–20 GiB (657 files) → needs a ≥ 20G memcg |

Keys [U]. The WN3 d01 key `5a03c654` differs from the PROD d01 key `cabfc644` on the same grid, so keys depend on the
options (e.g. `sst_update`), not on dims alone. Whether the 9-nest d01 reuses the WN3 d01 executable can be checked on
CPU with `gpuwrf aot-key` (read-only on v0.4.2, 0 GPU-min) once a guarded 9-nest input exists. The fused 9-nest program
is new in any case.

## Fit

The session plan is ≈ 110 GPU-min:

- v0.4.1 prime / seal / replays: 61 GPU-min actual. Minus B1 and B7 (16.5): ≈ 44.5.
- icloud=0 cold arm: 17–21 (w42-icloud0 morning: 21).
- B8: ≈ 1.3.
- RW3T interleaved N/O including N72 / O72: ≈ 44.

Adding 33–53 for the 9-nest gives 143–163 GPU-min, over the 120 cap. Dropping the RW3T A/B arms (contract RW3 timing)
would still not reliably make room, and the window cannot extend: hard end 18:30Z, ALISIOS nightly from 19:15Z.

## Scorer [M]

- `scripts/wn3_score.py` on main `1f7c1dc3b3` has sha `7a5b51d6…6e3362`, equal to ALISIOS S41.
- The D6 manifest is `9df4fab8…`, equal to S42.
- `--domains d01 … d09` runs the domains **sequentially by default** for more than 3 domains. An explicit
  `--sequential-domains` does the same.
- Any `--hours` other than 24 is labelled `"<H> h probe, NOT the release gate"`.
- The script lives in `scripts/`, not `src/`, so it leaves the release identities alone.

**CPU self-test** (`tools/ninenest_scorer_selftest.py`; receipts `receipts/NINENEST_SCORER_SELFTEST_h{3,24}.json`):

- Setup: an immutable archive of main `1f7c1dc3b3` and the real CLI, unchanged. Synthetic files only: the 9-nest
  horizontal mass shapes, 4 levels, the 14 D6-mandatory fields, statics and WRF globals.
- Result: **PASS 7/7 at 3 h and at 24 h** (24 h = 25 frames × 9 domains, 1:40 wall, 47 CPU-s, peak 264 MB).
- Clean case: all 9 domains pass D6 and output integrity, on 25/25 paired frames each.
- Scheduling: domain mode `sequential`, 9 compares, never more than 1 compare child alive.
- d07 GPU T2 + 3 K: only d07 fails (`T2 frame_tolerance`); the other 8 pass.
- One d09 GPU frame missing: `ValueError` before the first compare starts, and no domain JSON is written.

Real 72 h memory [M w42-scorer, WN3 72 h]: one domain at a time, largest compare d02 at 2.2–2.4 GB. The 9-nest leaves
are no larger than d03, so a 9-nest 72 h score should peak at ≈ 2.4 GB [I] and take ≈ 2,000–2,600 CPU-s [I]. WN3 72 h
takes 1,221–1,440 CPU-s.
