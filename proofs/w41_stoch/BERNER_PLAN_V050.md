# v0.5.0 plan (not implemented): stochastic vs multiphysics ensembles on the Canary WN3 cases

Lane w41-stoch, 2026-10-17. Plan only; nothing here has been run. Labels: [M] measured, [I] inferred, [U] unknown.

## Question

Berner et al. (2011, MWR 139, 1972-1995) compared a stochastic (SKEBS) and a multiphysics representation of
model error in a WRF mesoscale ensemble. Repeat that comparison for the operational ALISIOS geometry: does a
GPU-cheap stochastic ensemble (SKEBS + SPPT [+ SPP]) give station-verified probabilistic skill comparable to, or
better than, a multiphysics ensemble, and does combining both help, for Tenerife 3-nest forecasts?

## Design

| arm | members | model error | IC/BC |
|---|---|---|---|
| CNTL | N | none | perturbed (decision D1) |
| STOCH | N | SKEBS + SPPT (+ SPP-MYNN, decision D5), `nens` = 1..N | same as CNTL |
| PHYS | N | N rows of the qualified physics table (decision D6) | same as CNTL |
| PHYS+SKEBS | N | rows + SKEBS, distinct `nens` | same as CNTL |

N >= 10 per arm (Berner used 10 multiphysics members); the table has 12 rows today, 7 of them E41-PASS.
Members come from `gpuwrf ens prepare --mode stochastic|multiphysics` (this lane); CNTL needs an IC/BC
source (D1). Same cases, same lead times, same output stream in every arm.

Cases: the WN3 3-nest Tenerife set (33 case dirs under `<DATA_ROOT>/server/work/src/alisios/wrf_gen/`, dates
2026-01-02 .. 2026-06-17, all before the 2026-09-15 cut-off). Pick 10-18 spanning the regimes alisios_mgr uses
for training (Calima/east, DANA, Feb-Apr trade-wind; D24 ordering). Never a DENY day.

Verification (existing tools, no new metric code): `gpuwrf ens verify` against ALISIOS stations, read ONLY
through `gpuwrf.assimilation.letkf.alisios_guard` (guard_call scan -> drop DENY -> admit -> READ_SPY), reads
reported in MAILBOX. Variables T2, 10 m wind, gust, precip rate; land-aware bilinear, lapse-rate as for the
deterministic station gates. Scores: CRPS (fair), spread-skill ratio, reliability (threshold events), rank
histograms; paired case-block bootstrap of arm differences (STOCH - PHYS etc.) by lead time and variable.

## GPU-hour sizing

Basis [M]: WN3 3-nest, 3 cases in parallel on one RTX 5090, LW9 tree: 971.5 s for 72 case-hours =
13.49 s per case-sim-hour whole run (D28); solo warm 14.4 s per sim-hour (D26). Host-side output for the
full 375-variable stream is 0.30-0.40 GB per case-hour (A68).
Stochastic overhead [M Swiss only, this lane, `GPU_QUALIFICATION.md`]: with all switches the step cost was
dominated by the serial WRF/GNU RNG (4.85 ms/step on 43x43, 21.4 ms/step on a 71x121 d01 ~ 1.4 s/fc-h);
the exact parallel stream cuts that to 0.36 / 0.49 ms per step, bitwise identical. Swiss steady time with
all switches 1.12 s per forecast hour (st4p) vs 0.91 for the OFF reference run (c00, other session: +23 %,
~1 ms per step); SKEBS, SPPT and SPP alone +7 to +10 %. WN3 overhead [I] small:
the Fourier work scales with the 9 km d01 grid only; children receive the parent pattern.

Compile [I]: one executable serves every member and case that share the physics options and the set of active
stochastic switches (seeds/`nens` live in the device carry; B41 makes case data traced). Each physics row is its own
program, and the same row with SKEBS active is another one. Programs: CNTL 1 (default physics, stochastic off; may be
the cached release program) + STOCH 1 + PHYS one per row + PHYS+SKEBS one per row = 2R+2 cold 3-nest compiles for R
distinct rows: 22 at N=10 (R=10). The table has 12 rows today, so N=20 needs repeated rows (R<=12, 26 programs) or new
qualified rows. At ~10-15 min per cold 3-nest compile, cached across cases afterwards: 22 programs ~3.7-5.5 GPU-h,
26 ~4.3-6.5 GPU-h. A deviceless CPU compile with validated autotune results (FINDINGS B12k) could move most of this off
the GPU [U for 3-nest programs].

| scope | member-runs (4 arms x N x cases) | forecast GPU-h at 13.5 s/case-h | + compiles | total [I] |
|---|---|---|---|---|
| 10 cases, N=10, 24 h | 400 | 36 | 3.7-5.5 (22 programs) | ~40-42 GPU-h |
| 18 cases, N=10, 24 h | 720 | 65 | 3.7-5.5 | ~69-71 GPU-h |
| 10 cases, N=10, 72 h | 400 | 108 | 3.7-5.5 | ~112-114 GPU-h |
| 10 cases, N=20, 24 h (12 rows reused) | 800 | 72 | 4.3-6.5 (26 programs) | ~76-79 GPU-h |

Add the stochastic overhead factor to the STOCH and PHYS+SKEBS halves once measured on WN3. One RTX 5090 at
~20 h/day of free GPU gives the 10-case/24 h scope in ~2 days; the ALISIOS GPU schedule decides (D8).

Storage: full history for 400 x 24 case-hours ~ 2.9-3.8 TB [I] — does not fit (<DATA_ROOT> ~0.9 TB free).
Use station time series (tslist) plus a small regional stream (T2, U10/V10, RAINNC, hourly), keep full
history only for one member per arm (D9).

## Prerequisites (each is a separate measured gate)

1. Stochastic physics on the nested WN3 path: 24 h GPU + original CPU-WRF reference with identical seeds on one
   WN3 case (3-nest, 4-rank dmpar WRF), frozen D6 per domain + pattern/seed comparison as in this lane's Swiss
   arms. Today the nest forcing path is CPU-oracle-qualified only.
2. Multiphysics rows on WN3: each chosen row needs its own 3-nest GPU + CPU-WRF 24 h D6 (the table is
   Swiss-d01-qualified). CPU-WRF 4 ranks ~385 s per sim-hour -> ~2.6 h on 4 cores per row and case [I].
3. Seed-stream independence across members on WN3 dates is checked by the preset at preparation time;
   additionally confirm on GPU that two members' pattern fields differ and each matches CPU-WRF for one member.
4. An IC/BC perturbation source that is better than the current K0 uniform wind scaling (D1).

## Decisions needed (owner / alisios_mgr)

- D1 IC/BC uncertainty: none (pure model-error comparison), `gpuwrf perturb` uniform wind scaling (unbalanced),
  lagged initializations, or real global-ensemble members (GEFS/ENS through WPS/real.exe, outside this port).
- D2 case list and regime balance (10 vs 18 cases), and which cases are binding vs exploratory.
- D3 lead time: 24 h (fits) vs 72 h (3x cost, matches the v0.3.3 problem-day horizon).
- D4 N per arm: 10 (Berner) vs 20.
- D5 stochastic settings: WRF defaults vs tuned amplitudes; SPP-MYNN on or off; SPPT/SPP length scales on the
  9 km d01 (the model's spectral setup resolves all Registry defaults on the 121x71 x 9 km grid; small domains need smaller scales).
- D6 multiphysics rows (and whether to pay the WN3 requalification for E41-FAIL rows).
- D7 verification set: variables, stations, observation days (guard), significance method.
- D8 GPU allocation windows relative to the ALISIOS queue.
- D9 output policy (station series + reduced stream; which member keeps full history).
- D10 CPU-WRF reference policy: one reference member per arm and case (feasible) vs a full CPU ensemble
  (~400 member-runs x ~2.6 h x 4 cores ~ 4,000 core-h: not feasible on this host).
