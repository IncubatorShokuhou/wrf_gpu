# A6 answer for ALISIOS: `gpuwrf optimize` on the GPU (draft by lane w41-optimize, for alisios_mgr via the manager)

Your two questions, answered from the v0.4.0 FINAL validation and the optimizer changes on lane/w41-optimize.

## 1. Which physics options are released for optimize?

Released are **exact configurations, not single options**: the seven-option tuple (mp, pbl, sfclay, lsm, lw, sw, cu)
per domain that passed frozen bounded D6 against original CPU-WRF in the v0.4.0 FINAL set (66 composites plus the
release-default runs). Any other combination is refused by the tool. `gpuwrf optimize released` prints the list
(`--max-dom 1 --e41 PASS --json` for machines); it ships with the package (`gpuwrf/optimize/released_v0.4.0.json`).

- 47 configurations: 44 single-domain, 2 two-domain, 1 three-domain.
- Single-domain with **E41 PASS** (17): default Thompson/MYNN-SL/MYNN/Noah-MP/RRTMG with cu 0, 1 (KF, your d01
  default), 10 (modified KF, PDF trigger) or 11 (multi-scale KF); RUC land (lsm 3, needs six soil layers); Pleim-Xiu
  surface layer without PBL scheme; Goddard LW/SW 5/5 (qualified by a 6 h gate only); microphysics 5 (Ferrier),
  7 (Goddard 4-ice), 50-53 (P3 variants), 24 (WSM7, only with Grell-Devenyi cu 93); the WSM3 + Grell-Freitas +
  GFS PBL/surface-layer set; two idealised references (all physics off; RRTMG LW 31 only).
- 26 more single-domain tuples passed D6 but have **E41 FAIL** (compiled-kernel resource findings: may be slower or
  less robust, not a numerical failure), one is E41 UNKNOWN (CAM 5.1). A study can demand `"e41": "PASS"`.
- Not released although D6 passed on an earlier source: CAM radiation 3/3 (c28) and Zhang-McFarlane cu 7 (c29) are
  not in the FINAL checkpoint, so optimize refuses them.
- Where they were measured: mostly the 3 km Swiss case; KF (cu 1) on Madrid composites; cu 0 also on c60, a d01-only
  run of the WN3 Atlantic root grid with sst_update. On your Canary d01 a tuple meets your namelist (sst_update, GWDO,
  slope radiation, ...): the plan lists, per configuration, every setting where your case differs from the closest
  qualifying run. That is disclosure; your rung scores are the evidence for the Canaries.

## 2. Can d01-only members share the GPU?

Yes, measured on the RTX 5090 (lane w41-optimize GPU smoke, 2026-10-17): two d01-only members on the WN3 grid
(120x70x44, 0227 d01 with sst_update, release physics with cu 0 and with KF cu 1) ran **in parallel** at a
1.0 GiB pool, each peaking at **1.71-1.74 GiB per process** (nvidia-smi, incl. CUDA context), host RAM 6.4 GB while
compiling, 631-638 s for 3 h incl. the cold compile. The optimizer plans 1.25 GiB pool + 1 GiB context = 2.25 GiB per
member -> **about 12 members next to a 4 GiB desktop** on the 32 GB card. Host RAM binds first while compiling
(~6.5 GB per cold member, ~3 GB warm); the optimizer feeds the launcher the measured peaks after the first batch.
Every rung sends all its members at once, in two waves: one member per not-yet-compiled configuration first, then
the other dates load that program (our smoke showed two dates of one new configuration otherwise compile the same
program twice in parallel). Nested studies keep the launcher's C-auto sizing.

Practical rules for the tournament:
- One compile per configuration, reused by all dates **if all case inputs have the same wrfbdy/wrflowinp record
  count** (v0.4.0 programs take every record as a shape; w41-bdyrec removes that in v0.4.1). Generate every date's
  inputs over the same window length; the plan reports `program_groups`.
- 8 configurations x 2 rungs, e.g. rung 1 = 8 configs x 2 dates (16 members: 8 cold compiles, then 8 warm), rung 2 =
  best 4 x 2 more dates (8 warm members), then the winner on the held-out dates; 24 h members cost
  (2 + 16 + 8) x 24 = 624 case-hours. Every dispatched member is charged, also a failed one and its retry, so set
  `budget_case_hours` with headroom (e.g. 696 = 624 + three 24 h retries); the budget is part of the study and cannot
  be raised on resume.
- `--chunk-minutes 45` sizes every batch to the time left in the GPU block and stops (exit 3) when nothing more fits;
  `--resume` continues in the same directory, reusing every finished member (also members that finished inside an
  interrupted batch). Batches are sized with the parallelism the launcher actually admits: until the first members are
  measured the optimizer assumes one at a time, so the first block of a new study starts small; give
  `--member-wall-s` (a measured member wall) to size it better. Case input files are bound by content: changing a
  wrfinput/wrfbdy/wrflowinp of a running study makes `--resume` refuse it. Smoke: a
  study paused after rung 1 resumed later on the GPU and finished rung 2 + the held-out dates in 2 GPU-minutes,
  every member loading an already compiled program (about 40 s per 2-member batch of 3 h Swiss members).
- `--bundle <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0` runs members with the sealed bundle in a private layer; the
  bundle is never written (smoke: bundle check after the run BUNDLE_OK new=0; the release-default Swiss member
  loaded the bundle's program warm, 41-49 s per 3 h; KF compiled into the layer). Only the release defaults are
  primed there (WN3 3-nest and Swiss d01): a d01-only WN3 configuration compiles once into the layer and is then
  reused by every later rung and resume.
- Receipts: `rungs/rung-NN.json` per rung (scores, survivors, member wall/host/VRAM peaks), `ledger.json`, `result.json`.
