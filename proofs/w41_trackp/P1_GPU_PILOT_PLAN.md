# Track P — P1 GPU plan: smoke + two-coefficient roughness pilot (pre-registered, for the validation window)

Owner lane w41-trackp, 2026-10-17. Design: `.agent/decisions/DESIGN-SURFACE-PARAM-ESTIMATION.md` + Astra critique
(PURSUE-MODIFIED: offline two-coefficient roughness pilot first, held-out gate, stop on no signal).
Everything below is written BEFORE any station value is read; the split, grid, scores and gates are frozen here.

## A. GPU smoke of the traced controls (≈ 25 GPU-min, one lock session) — prerequisite for B
Tree: lane/w41-trackp at the reviewed commit, release defaults (FAST_DEFAULTS), capped prealloc as release.
1. Cold compile WN3 0227 3-nest with PRIOR controls on d01–d03 (`gpuwrf surface-params basis` + `expand` with all
   multipliers 1; 3 h). Receipt: compile seconds, E41 wave cubin gate (`compile/cold/wave_cubin_gate.py` on the
   JCT cache incl. the fused d02+d03 AOT program, E201) BEFORE cache cleanup (E166/E177).
2. Same process tree, warm: (a) OFF (no option) 3 h, (b) PRIOR 3 h, (c) A = forest z0 ×1.25 3 h, (d) B = shrub
   z0 ×0.8 3 h — fresh processes on ONE cache. Gates: (b),(c),(d) load the SAME executables (cheap keys equal, 0
   compile events, `compile_total` 0 in their receipts); (b) wrfout vs (a): byte-identical [expected; CPU proof
   shows bitwise ON-prior == OFF over 2 steps] — if not byte-identical, report max |Δ| per field and the first
   differing step (constant-folded table gathers vs runtime operands) and decide with the manager before B;
   (c) vs (a) differs only through land/surface fields first (ZNT, UST, HFX, LH) and stays finite.
3. Cost: warm step wall OFF vs PRIOR (n = 10 interleaved roots, E191 ±5 % noise → nsys device time per root);
   per-PID VRAM; extra executable inputs (11 small fields per domain). Expected ≈ 0 extra kernels (fields replace
   constant-folded gathers; any extra convert/copy is reported).
4. Two cases in one process/VRAM: the B=2 batched path on PROD d01 (`GPUWRF_BATCH_ENSEMBLE=2`,
   `batch_surface_controls=(A, B)`), 3 h — lanes differ, each lane ≈ its solo run (vmap ≠ bitwise is allowed;
   relative leaf difference reported), VRAM per lane vs solo (the historic B=4 28.8 GiB temp → subchunk = 1 or 2).

## B. Roughness pilot (≈ 2.9 GPU-h at the v0.3.2 WN3 aggregate rate 5.507 s per case-hour [M historic];
current v0.4.0 throughput [U] — re-estimate from A.3 before starting)
- Cases: the 18 WN3 3-nest PASS dates (alisios_mgr 2026-10-02: 0102, 0118a2, 0120, 0123, 0218, 0220, 0225a2,
  0227, 0303a2, 0407, 0408, 0502, 0503, 0601, 0608, 0613, 0614, 0615; all 18z, all before 2026-09-15).
  Guard scan (`alisios_guard.drop_deny_days`, CALIBRATION_READ) first; a DENY date is removed, never read.
- Split (frozen rule, applied to the ALLOW list in date order): training = every third date starting with the 1st
  (6 dates), test = the other 12. Fewer than 15 ALLOW dates → pilot INCONCLUSIVE (report, stop).
- Controls: two log multipliers on MPTABLE `Z0MVT`: F = MODIS forest 1–5, S = shrub/savanna/grass/crop 6–10, 12,
  14 (all domains). Grid {0.8, 1.0, 1.25}² = 9 settings incl. the prior (1, 1). Same IC/BC, no stochastic physics.
- Runs: 9 settings × 6 training dates × 24 h + (winner, prior) × 12 test dates × 24 h = 1,872 case-hours,
  4 processes per GPU (C-auto), `gpuwrf ens` scheduler, one compiled program for all settings (Track P).
  Disk: score from station series, delete full histories after scoring (≈ 7.5 GB per case-day).
- Station operator (step 0, before scoring): `gpuwrf verify` land/sea stencil, elevation + sensor height,
  earth-relative U/V, hourly sampling, ALISIOS QC flags; representation error and temporal correlation via
  event-block bootstrap (blocks = dates). Audit the known coastal WS10 excess against original CPU-WRF first.
- Selection (training only, once): minimum mean vector-wind RMSE over the 6 dates subject to T2 RMSE ≤ prior + 0.1 K.
- Gate (test dates only, once): vector-wind RMSE ≥ 5 % better than the prior with the event-block 95 % CI excluding
  0; T2 RMSE worse ≤ 0.1 K; no pre-registered regime (trade wind, calima/east, DANA/convective, night stable) wind
  RMSE worse > 2 %; better than a training-only pooled wind bias correction (`gpuwrf calibrate fit --method bias`).
  Spatial blocks: repeat the gate on held-out station blocks (north/south/east/west Tenerife); report both.
- Fidelity: one original CPU-WRF 24 h run (ALISIOS CPU window, 4 cores, ~2.5 h) on one test date with the
  winner's EDITED MPTABLE vs GPU with the winner's controls → frozen D6 + full-inventory scorer (this is the
  forecast-level proof of the table-edit equivalence; CPU proof is byte-level per Noah-MP step).
- Stop rule: gate failed or no signal → no expansion to EKI on real stations; report the negative result.

## C. After a passing pilot (not in this window)
Manager decision 2026-10-17T16:54Z: the contract's 'guarded real stations on ALLOW days' step is deferred to
this phase (alisios_guard is on main since 39993e4df8); v0.4.1 ships the EKI scaffold + OSSE only.
Guarded real-station EKI (`gpuwrf surface-params eki`) on the training dates only, 16 members × 6 dates × 3
iterations ≈ 10.6 GPU-h [I], same held-out gate; then spatial modes only with new evidence.
Data rule for every real-station step: ALISIOS station observations only for valid times before 2026-07-01
(ObsDB extract rule) and only on guard-ALLOW days, read inside `alisios_guard`; PROD 2026-07-25 is after that
cutoff, so PROD runs are model-side evidence only (smoke, identity), never station calibration/scoring.
Every new obs read is reported in MAILBOX (`w41-trackp: ALISIOS OBS READS …`).
Model inputs count too (rule C1745, kernel 3a997be576): the WN3 case dirs (wrfinput/wrfbdy/wrflowinp/namelist/tables,
realpath counts, also via staging symlinks) and any CPU-WRF reference outputs are opened only inside
`alisios_guard.guarded_session` / `alisios.guarded_read`; preferred: one guarded copy per ALLOW date outside the
ALISIOS roots, then every pilot member and the basis/expand steps read that copy.
