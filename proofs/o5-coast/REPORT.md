# o5-coast — coastal 10 m wind: port difference or CPU-WRF-shared? (Opus frontrunner, 2026-10-24)

Status: CPU part critic-MERGE (cr-o5-coast VERDICT 2026-10-24T18:51Z on ff92d6f834, <USER_HOME>/wrf_gpu2_lanes/cr-o5-coast/VERDICT.md); GPU/RW gates pending (25.10 window). Labels: [M] measured, [I] inferred, [U] unknown. Reference = original CPU-WRF 4.7.1.
Data rule (manager 14:57Z/15:01Z): nothing under <DATA_ROOT>/alisios/**; only our own CPU twin 0227 (w43-twin0227, 4×1,
same input SHAs as the GPU runs), our GPU outputs (te_gpu v0.4.2 extracts, F20 v0.4.0 raw) and our own Swiss CPU-WRF
(release-docs RD11). Teacher scorer receipts used as aggregate statistics only.

## 1. Open items this lane owns
1. Coastal WS10 excess: ALISIOS R32-67 0115 stations (GPU v0.3.3 W0115 − CPU R0115 WS10 RMSE +.066/+.057 m/s on
   coastal scopes A/B, τ37–58; FINAL_BE0 B′ +.035 SYSTEMATIC) + "V10 h42–44 event under IC R".
2. Coastal 72 h twin floor (w43-coastwind left it PENDING).
3. Wind τ>40 h HELD (TEACHER_QUAL §3: E-C ERA5 d03 U10@68 X/C 2.48, d02 U10@35 X/C 2.06).

## 2. Results so far
| id | question | result | label |
|---|---|---|---|
| OC-A1 | systematic coastal WS10 bias GPU v0.4.2 − CPU (0227 d03, 72 h) | bias ≤ \|0.04\| m/s on open sea, coast-sea, coast-land, urban, barren, grass, … in every lead block; pooled RMSE grows .15 → .32 → .77 (coast-sea) | M |
| OC-A2 | U10/V10 X/C by lead (0227 WN3, E-C ERA5; teacher receipts) | τ1–24 X/C 0.88–1.22; τ25–72 1.07–2.01; d02 X>C at 32/32 leads in both cases | M |
| OC-A3 | surface layer / roughness / U10 diagnostic (GPU v0.4.0 vs CPU twin, d03 τ1–24) | WS10/WS_L0, (UST/WS_L0)², bulk CH per surface class × Rib agree ≤ 0.1 % near neutral; code audit: MYNN-SL, Noah-MP ZNT, masks, U10 formula IDENTICAL on the release WN3 path | M |
| OC-A4 | d01 dry-mass (MU/PSFC) GPU − CPU | coherent domain-mean offset (0227: −0.38 Pa at τ1 vs CPU twin 0.002) that tracks the boundary mass tendency: GPU leads by +47…+61 s in rings 1–9 (dt 54 s), interior +25 s, d02 ~+30 s, d03 0; Swiss (dt 18 s) +16…+19 s → lead ∝ dt; #148 9-nest (o5-9nest, accepted 18:43Z) rings 1–9 +53…+65 s, absent in their CPU−CPU twin control (0420) → GPU-specific | M |

## 3. Cause (source) — root ring-0 boundary clock one step ahead
`runtime/operational_mode._acoustic_core_state_from_prep`, root (non-nested) specified branch:
`lead_stage = lead_seconds + dt_rk` for the ring-0 mass/θ/ph/tangential pins and `lead_seconds` for the normal u/v work
targets, while `lead_seconds = step_index·dt` (:10942) is the END of the step (= WRF `dtbc` after `solve_em.F:378`).
WRF walks the spec zone from t_n by `dts_rk·btend` per acoustic substep (`spec_bdyupdate`), ending stage k at
t_n + dt_rk. The port pins every substep to the absolute value at t_{n+1} + dt_rk. The end-of-step re-pin
(`apply_lateral_boundaries(dry_spec_only)`) restores ring 0 at output, which hides it in ring-0 diagnostics. Introduced
with v014 (144847a942, docstring "step-start lead"). Nested children use the WRF walk (58f20858f4). Relaxation target
(`_specified_bdy_relax`, lead = WRF dtbc) is correct. Affects every real-data root (single-domain native and nested).

## 4. Fix — exact WRF walk, opt-in (merged DEFAULT OFF, manager 2026-10-24 18:51Z; flip after the GPU walk arm)
`GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk` (flip = one `FAST_PATH_DEFAULTS` line on lane/o5-coast-flip; `0`/unset = released
clock; `1`/`2` = stage-end / stage-midpoint absolute pins, diagnostics). `coupling.root_spec_walk.root_walk_rates`
(moved out of `boundary_apply` so no line sits above its Pallas kernel, E109) re-couples every wrfbdy record with its
own mass exactly as the loader decoupled it (`MT_k = MUB + MU_B(k)`; u/v with `calc_mu_staggered` faces and
msfuy/msfvx; θm − T0 with c1h/c2h; ph′ with c1f/c2f; mu uncoupled) and returns WRF's `spec_bdy_dry` tendencies
`(rec[l+1]−rec[l])/interval` with WRF's bracket and corner ownership. The acoustic kernels (legacy `acoustic.py` and
native `dyn_acoustic_fp32.py`) add `dts·btend` to the pre-substep ring like the live-nest walk: u/v ring incl. the
tangential sides, mu/muts/θ work, `spec_bdyupdate_ph`; muave is never pinned (WRF never pins it); w keeps
`zero_grad_bdy`. The end-of-step spec pass is unchanged (re-anchors ring 0 at t_{n+1}; second-order coupled vs
decoupled interpolation difference to WRF's `spec_bdy_final` remains, no timing error).

## 5. CPU proof (pristine WRF)
| check | result | label |
|---|---|---|
| C1 rates vs real.exe: Swiss wrfbdy (shipped example), production loader, 5 step-end leads incl. a step ending on a record | MU/T/PH/U/V ring rates == file `*_BT*` ≤ 1.1e-6 rel (f32 leaf rounding); a step ending at 10800 s uses the old interval, 10818 s the new (WRF dtbc bracket) | M |
| tests/test_o5_root_spec_walk.py (oracle `scripts/o5_root_spec_walk_oracle.py`) | 8/8: coupled btend restored on synthetic records decoupled like the loader; ring after every substep of every RK stage == bdy(t_n + m·dts); released clock one step ahead at every stage end; default/debug switch; production wiring | M |
| deletion sensitivity (results/MUTANTS.txt) | 6/6 non-equivalent mutants killed by AssertionError, 0 setup errors (face averaging, re-coupling, map factor, wiring, default, kernel branch); 1 equivalent mutant documented | M |
| regressions | test_fast_defaults (post-LW9 list), default-value suites v031–v033/lw11/lw12, nested-cadence RCA suites: green after the two pinned-text updates | M |
| Swiss d01 CPU-JAX legacy probe, base | reproduces the lead (τ1 ring MU −0.17…−0.31 Pa; lag +9…+14 s at dt 18) | M |
| OFF identity (`proofs/s1_port/release_identity.py`, CPU, release defaults with GPUWRF_ROOT_SPEC_STAGE_CLOCK=0) | Swiss b977712f… EXACT, PROD 321aa651… EXACT (= documented v0.4.2/v0.4.3 identities). Scope (E222): the change touches no Pallas kernel body (only XLA-level boundary adapters and Python branching), so the probe's Pallas blind spot cannot hide an OFF difference | M |
| OFF identity, critic independent (E222-safe) | canonical + full-constant native StableHLO exact: Swiss 51b58b7d / PROD e1766f50, true Pallas census 157 calls / 62+61 bodies | M |
| default (walk) identity | Swiss 47a20298…, PROD dc2adbcb… (intended identity change; WN3 likewise) | M |
| Swiss d01 CPU-JAX legacy probes at +1 h vs our Swiss CPU-WRF (both arms OOM at the 12 GiB cap after the first output, so τ1 only) | ring-mean MU (Pa) r1/r2/r3/r5/r8/interior: base −0.22/−0.20/−0.17/−0.23/−0.31/−0.26 → walk +0.14/+0.12/+0.15/+0.08/+0.01/+0.04; walk − base = +0.36/+0.32/+0.32/+0.32/+0.32/+0.30 ≈ one 18 s step of the local boundary mass tendency (−51…−63 Pa/h → 0.26–0.32 Pa) everywhere; PSFC mean −0.52 → −0.23 Pa. The +0.12…+0.15 left in relax rings 1–3 is a legacy-path relax-zone difference already present in the base (bump of +0.08…+0.14 vs ring 8), absent on the release path (GPU v0.4.0 base r1 −0.30 ≈ r8 −0.31) → the CUDA mini (release path, Swiss +1 h) is the decisive release-path check | M |

## 6. Contract items — status and recommendations
1. **0115 coastal WS10 excess (ALISIOS R32-67, GPU v0.3.3 W0115 − CPU R0115 +.06 m/s on coastal scopes, SYSTEMATIC).**
   Not a surface-layer/roughness/U10-diagnostic port difference [M A3 + code audit]. The root boundary clock (one step
   ahead since v014, hence also in v0.3.3) is a systematic, domain-coherent GPU–CPU difference [M A4, Swiss, #148 via
   o5-9nest] and the only port defect found on this path; its size on station WS10 is [U] until a walk run exists.
   Recommendation: after the walk is default, ALISIOS repeats the 0115 station check (R32-67 protocol, same yardstick).
2. **Coastal 72 h twin floor.** Domain level [M A2]: τ1–24 X/C 0.88–1.22, τ25–72 1.07–2.01 (U10/V10, 0227 WN3 and
   E-C ERA5). Per coastal mask GPU−CPU-twin [M A1]: coast-sea pooled WS10 RMSE .15/.32/.77 m/s (τ1–24/25–47/48–72),
   bias ≤ |0.04|. A per-mask CPU–CPU floor needs the 0227 CPU reference (admission) — instead the GPU session adds the
   SEED arm (o5-9nest ±1-ulp seed, same definition as #148) → GPU self-twin G per mask and lead, no admission needed.
3. **Wind τ > 40 h HELD.** Keep HELD. Proposed release criterion for ALISIOS/manager: with the walk default, the WALK arm
   (and SEED for G) on 0227 gives U10/V10 X_twin per lead block and coastal mask; qualify τ > 40 h if X_twin(walk)
   ≤ max(4/3·C_teacher, G) on every block (C from the teacher receipts, G from SEED), otherwise keep HELD and report the
   residual as realization spread vs systematic with the measured numbers.

## 7. Pending
PROD clock-0 identity + walk identities and the Swiss walk lag (CPU chain queued for RAM slots); GPU CUDA mini + MINI/WALK
(+SEED) in the next window (`gpu/GPU_REQUEST.md`); #148 lag signature already measured by o5-9nest (rings 1–9 +53…+65 s),
decided by the walk arm.
