# Ensemble design fixes for the A1 LETKF (lane w41-da-ens, v0.4.1) — evidence

Scope: the R3 design defects (`proofs/a1_da/R3_SPREAD.md`, `BA_DEBIASED.md`; DA-F11/F12): spread lost
40–75 % per 6 h, bias-dominated ρ, RTPS attenuating the inflation, ρ escalating cycle to cycle, co-located
cross-variable noise at N = 10, shared LBC. All new code is opt-in, host NumPy only, no forecast-program file
touched. Labels: [M] measured here, [I] inferred.

## What was built (src/gpuwrf/assimilation/letkf)

| R3 item | Module | What |
|---|---|---|
| 1 bias-aware calibration | `inflation.ObsBias` | online scalar-Kalman obs-minus-background offset per group/station; optional correction of the obs values |
| 2 var × height inflation | `inflation.AdaptiveInflation` | Miyoshi-2011 Gaussian fusion of debiased per-(obs group × band) estimates into ρ(T/QV/U/V × 4 bands), persisted across cycles; sampling variance replaces the hard clamp |
| 4 posterior maintenance | `spread.rtps/rtpp/posterior_inflation/additive_inflation`, `spread.make_maintenance` | RTPS toward the inflated prior (state-space inflation, or `target: inflated`), RTPP, seeded centred additive inflation (one weight matrix for all fields, log-space moisture) |
| 3 variable localisation | `varloc` → core `analyse(var_loc=)` | `surface_split` / `strict` coupling tables in the core `localisation.variables` format |
| 5 balanced perturbations | `perturb` | Gaussian combinations of model states incl. MU, `rebalance_with_mu` (real.exe recursion, ΔMU), forecast/spun-up deviations, divergence/vorticity diagnostic |
| 6 perturbed LBC | `lbc_ens` | per-member wrfbdy with record-0 = IC deviation, time-evolving deviations, `lbc_perturb` encoder |
| 7 N ≥ 20 | `ensemble.perturbed_ic_members(combination=, balanced=)` | N members from M archive states (N may exceed M), MU-balanced |
| product wiring | `inflation.validate_config_section / calibrate / analysis_hooks / read_plan / chain_section` | da-core API (`config.inflation`, `api.analyse_case`), da-cycle chaining |

## Lorenz-96 OSSE [M] (`l96_ens.py`, `results/L96_ENS.json`)

**Seeds 1–3 are development seeds**: a first pass with other settings (α 0.8, additive 0.15) was run on them before
the sweep, and the gates were revised after it. The qualifying result is the **prospective run** on untouched
seeds 31–33 at the end of this section.

40 variables, N = 20, σo = 1, imperfect model (truth gets model error q = 0.6 on odd / 0.15 on even variables per
cycle: the two halves need different inflation), +0.8 obs bias on the odd half. 3 seeds × 1500 cycles, tail 3/4.
α = 0.3 and additive = 0.02 were frozen from a declared development sweep on other seeds
(`results/L96_SWEEP.json`, seeds 11/12, 1000 cycles: RTPS 0.8 on top of the adaptive ρ is overdispersive,
spread/RMSE 1.45; α ≤ 0.3 gives 0.92–1.05).

| config | RMSE | RMSE odd / even | spread | spread/RMSE |
|---|---|---|---|---|
| R3 baseline: global E[d²] ρ (bias in) in the transform, RTPS 0.8 raw | 0.752 | 0.879 / 0.589 | 0.861 | 1.15 |
| R3 + ObsBias correction | 0.648 | 0.729 / 0.547 | 0.813 | 1.25 |
| adaptive var-group ρ + ObsBias + RTPS 0.3 | 0.620 | 0.711 / 0.504 | 0.613 | 0.99 |
| **+ additive 0.02 (new design)** | **0.618** | 0.708 / 0.503 | 0.620 | **1.00** |

Gates (all seeds): spread/RMSE ∈ [0.7, 1.3] ✓; RMSE below R3 ✓ (−17.8 %); spread/RMSE ∈ [0.5, 2] in cycles
3–10 after the climatological start ✓.

The table above is estimator **v1** of `ObsBias`. After the Swiss twin diagnosis (below) the gain includes the
ensemble's coherent forecast-error variance (**v2**, the default). Rerun with v2 (`results/L96_ENS_v2.json`):
NEW RMSE 0.610, spread/RMSE 1.025; R3 + ObsBias 0.642; all 3 gates ✓. The v1 table reproduces bitwise with
`ObsBias(var0=4, q=0.05, coherent=False)` (`L96_OBSBIAS=v1`).

**Held-out confirmation** [M] (`results/L96_ENS_v2_heldout.json`, seeds 21/22/23, never used before). The final
configuration (estimator v2, α 0.3, additive 0.02) and the three gates are those of `l96_ens.py` in commit
16d3eb76d3 (12:28:47Z). The output file was written at 12:29:33Z and opened afterwards; the script is unchanged
since. The freeze was not separately announced before the run.

| config | RMSE | spread/RMSE |
|---|---|---|
| R3 design | 0.754 | 1.16 |
| R3 + ObsBias v2 | 0.643 | 1.29 |
| adaptive + ObsBias v2 + RTPS 0.3 | 0.611 | 1.02 |
| **new design (+ additive 0.02)** | **0.611** | **1.03** |

Gates on every held-out seed: spread/RMSE ∈ [0.7, 1.3] ✓ (1.02–1.04); RMSE below R3 ✓ (0.606–0.615 vs
0.750–0.755); spread/RMSE ∈ [0.5, 2] in cycles 3–10 ✓. Supporting evidence only (the freeze was not announced).

**Prospective confirmation** [M] (`results/L96_ENS_v2_prospective.json`, untouched seeds 31/32/33). Configuration,
estimator and gates were registered in MAILBOX at 2026-10-17T12:36:39Z (code 028256559d, `l96_ens.py` unchanged
since 16d3eb76d3) before the run:

| config | RMSE | spread/RMSE |
|---|---|---|
| R3 design | 0.751 | 1.15 |
| R3 + ObsBias v2 | 0.641 | 1.27 |
| adaptive + ObsBias v2 + RTPS 0.3 | 0.608 | 1.02 |
| **new design (+ additive 0.02)** | **0.608** | **1.03** |

**Gates PASS on every seed**: spread/RMSE 1.04 / 1.03 / 1.02 ∈ [0.7, 1.3]; RMSE 0.600 / 0.613 / 0.611 below R3
0.747 / 0.749 / 0.757; spread/RMSE ∈ [0.5, 2] in cycles 3–10.

## Estimator v2: what the real model showed that L96 did not [M]

In the Swiss twin (A1, cycle 0), estimator v1 learned the forecast error that is coherent over the whole 126-km
domain as an observation bias in a single cycle (gain ≈ 1 because the uncertainty of the cycle-mean innovation
was s_d²/n only):

| group (truth) | v1 03Z → 06Z | v2 (replay of the same cycles) |
|---|---|---|
| V10 (unbiased) | −2.07 → −1.49 m/s | −0.18 → −0.23 |
| sonde V (unbiased) | −1.95 → −2.01 | −0.13 → −0.23 |
| U10 (+0.8 station bias) | +1.37 → +1.35 | +0.68 → +0.83 |
| T2 (−1.0 K station bias) | +0.49 → +0.14 | +0.09 → +0.09 |

v2: estimator variance = variance over members of the group-mean H(x) + s_d²/n; prior variance and random
walk are relative to the group's σo² (0.25, 0.01). A cycle-varying coherent flow error and a constant station
bias can only be separated over many cycles: the T2 bias (−1.0 K) is masked by a coherent cold forecast error of
≈ −1.5 K at 03Z and is not identifiable in 2 cycles — `obs_bias` is for long cycling, not for a 12-h twin.
Adaptive inflation now counts profiles (station|site|launch), not levels: 2 radiosondes are no longer 40
independent samples (upper-level T ρ 2.01 → 1.31 in the replay).

## Real-model twin OSSE, Swiss example case [M] (`swiss_osse.py`, `osse_tables.py`, `results/SWISS_OSSE.json`)

Setup: the shipped non-ALISIOS example `examples/switzerland_d01` (42 × 42 × 44, dx 3 km, 2023-01-15 00Z),
**original CPU-WRF** as the member engine (1 rank per member), N = 20. Perturbations: Gaussian combinations of the
original CPU-WRF reference run at lags 0–9 h (scale 0.5); truth = an independent draw of the same construction
with its own perturbed LBC, MU included. Observations from the truth run: 2 synthetic sondes (T/QV/U/V every 2nd
level up to 250 hPa) + 30 surface stations (T2 with a −1.0 K and U10 with a +0.8 m/s station bias, Q2, V10).
Analyses at 03, 06, 09Z, 3-h cycles, forecast to 12Z. Interior = ≥ 8 rows from the edge, rms over the
0–1/1–2/2–5 km bands, U/V on mass points.

- **A0 = the R3 design**: IC U/V/T/QV only, MU frozen, tapered; shared wrfbdy; innovation ρ (E[d²], bias in) in
  the transform; RTPS 0.8 to the raw prior; no additive; no variable localisation.
- **A1 = the new design** (pre-registered arm): IC incl. MU (`rebalance_with_mu`), untapered + matched perturbed
  wrfbdy per member (`lbc_ens`); adaptive debiased var × band ρ in state space (estimator v1) + surface obs-bias
  correction; RTPS 0.3 toward the inflated prior + additive 0.1 (lagged pool at t_a); `surface_split`.

Gate, pre-registered at 10:13Z after cycle 0 (cycles 1–3 unseen), evaluated on A1:
G1 prior spread/RMSE ∈ [0.5, 2.0] at 06, 09, 12Z; G2 prior spread at 06/09/12Z ≥ 0.5 × that at 03Z (per variable);
G3 written-member spread = posterior spread (± 1 %); G4 mean posterior RMSE A1 < A0.

**A0**

| t (prior) | prior spread T/U/V | prior RMSE T/U/V | posterior spread | posterior RMSE | written spread | 1st-h spread retention | analysis wall s / peak RSS MB |
|---|---|---|---|---|---|---|---|
| 03Z | 0.07 / 0.22 / 0.14 | 0.91 / 1.86 / 2.55 | 0.11 / 0.37 / 0.24 | 0.89 / 1.67 / 2.47 | 0.11 / 0.37 / 0.24 | 0.62 / 0.77 / 0.36 | 31 / 1789 |
| 06Z | 0.02 / 0.08 / 0.05 | 0.78 / 2.10 / 2.05 | 0.06 / 0.25 / 0.15 | 0.77 / 2.52 / 1.89 | 0.06 / 0.25 / 0.15 | 0.48 / 0.52 / 0.63 | 41 / 1891 |
| 09Z | 0.01 / 0.05 / 0.04 | 0.65 / 1.54 / 1.86 | 0.07 / 0.24 / 0.19 | 0.66 / 1.55 / 1.69 | 0.07 / 0.24 / 0.19 | 0.63 / 0.54 / 0.52 | 62 / 1913 |
| 12Z | 0.02 / 0.04 / 0.03 | 0.66 / 2.09 / 2.34 | – | – | – | 0.43 / 0.40 / 0.37 | – |

**A1**

| t (prior) | prior spread T/U/V | prior RMSE T/U/V | posterior spread | posterior RMSE | written spread | 1st-h spread retention | analysis wall s / peak RSS MB |
|---|---|---|---|---|---|---|---|
| 03Z | 1.28 / 3.29 / 3.94 | 1.26 / 1.94 / 3.24 | 0.69 / 1.82 / 1.96 | 0.58 / 1.23 / 1.26 | 0.69 / 1.82 / 1.96 | 1.54 / 2.02 / 1.94 | 103 / 2591 |
| 06Z | 1.47 / 3.59 / 4.77 | 1.08 / 1.98 / 2.60 | 0.81 / 2.05 / 2.27 | 0.45 / 1.08 / 0.93 | 0.81 / 2.05 / 2.27 | 1.96 / 1.99 / 2.20 | 185 / 2711 |
| 09Z | 1.29 / 3.06 / 3.64 | 0.71 / 1.45 / 1.52 | 0.78 / 1.83 / 1.86 | 0.53 / 1.32 / 1.16 | 0.78 / 1.83 / 1.86 | 1.77 / 1.79 / 1.98 | 163 / 2715 |
| 12Z | 1.10 / 2.91 / 2.26 | 0.61 / 1.66 / 1.95 | – | – | – | 1.49 / 1.51 / 1.59 | – |

"1st-h spread retention" = spread +1 h / +0 h of the forecast that ends at that row's time (the 03Z row is the
first hour after the initial perturbations).

**Verdict (pre-registered): G1 FAIL, G2 PASS, G3 PASS, G4 PASS.**

- G1: A1 spread/RMSE 1.68 / **2.12** / 1.60 at 06/09/12Z — overdispersive, outside the band at 09Z.
  A0: 0.029 / 0.025 / 0.020.
- G2: A1's prior spread stays at 0.57–1.21 × its 03Z value (lowest: V at 12Z; T 1.28 → 1.47 / 1.29 / 1.10). A0
  falls to 0.14–0.36 × its 03Z value (T 0.07 → 0.02 / 0.01 / 0.02). The R3 collapse is gone; the R3 ρ escalation is reproduced in A0:
  ρ 182 → 670 → 1653, and its posterior spread is still only 0.06–0.11 K for T, because RTPS 0.8 relaxes toward
  the raw prior.
- G3: written members = posterior spread to 4 digits in every cycle. The inflation reaches the members.
- G4: mean posterior RMSE (T/U/V) A1 0.949 vs A0 1.568 (−39 %). The 12Z forecast RMSE is A1 0.61 / 1.66 / 1.95
  vs A0 0.66 / 2.09 / 2.34.

Two defects of the code A1 ran, found after the arm (critic cr-w41-da-ens), both fixed since:
- Estimator v1 of `ObsBias` (see above).
- `inflate_state` did not carry the T/QV inflation increment into P/PH. The analysis rebalances only
  analysis-minus-prior, so A1's written members kept a hydrostatic inconsistency of the size of the inflation
  increment (critic synthetic test: up to 794 m²/s² PH at ρ = 4). A1's ρ was at most 2.0 (T above 5 km).
  Fixed in e159e7b904.

So A1 validates the design direction, not the final code. The final code is tested by the prospective arm F1
(below); the post-hoc A1b was never run.

Why A1 is overdispersive [M + I]:
- Spread and ensemble-mean error grow together after every analysis (A1 06Z: T posterior 0.81 / 0.45 → +1 h
  1.35 / 1.27), so the growth is mostly real error growth. The ratio stays above 1 because the posterior is
  already overdispersive: 1.2–2.4 at 03/06/09Z, with ρ ≥ 1 (no deflation), RTPS 0.3 and additive 0.1 on top.
- Spread is larger in the interior than at the lateral edge (`results/EDGE_PROFILE.json`; A1 06Z T 1.15 at the
  edge vs 1.74 at ≥ 16 cells). So it is not the boundary simply re-injecting spread.
- With one truth realization the ratio is a single sample. The 03Z prior (before any analysis) is already
  1.0 / 1.7 / 1.2 (T/U/V). Calibrating the absolute level needs several truths, or deflation (ρ < 1) once the
  posterior spread exceeds the error.

Cost: A1 analysis 103–185 s, peak RSS 2.6–2.7 GB (A0: 31–62 s, 1.8–1.9 GB) for N = 20 on 42 × 42 × 44, CPU,
under a lane-core load of 60–80.

## Balance experiment [M] (`swiss_osse.py balance`, `results/BALANCE.json`)

N = 10 members, the same combination weights, all tapered, shared wrfbdy: only the balance of the initial
perturbation differs. First-hour spread retention (+1 h / +0 h, interior, rms over the 0–1/1–2/2–5 km bands):

| arm | IC construction | initial spread T / U / V | retention T / U / V |
|---|---|---|---|
| B0 | U/V/T/QV perturbed, MU frozen (R3 archive style) | 0.57 / 1.06 / 1.60 | 0.64 / 0.81 / 0.37 |
| B1 | + MU perturbed, `rebalance_with_mu` (MU deviation 43 Pa rms) | 0.57 / 1.06 / 1.60 | 0.64 / 0.81 / 0.37 |
| B2 | spun-up: B1's 3-h forecast deviations recentred on the 03Z reference | 0.07 / 0.23 / 0.15 | 0.51 / 0.57 / 0.64 |

B0 and B1 agree to three digits: in this small domain the first-hour spread loss is not caused by the mass
imbalance of the perturbation, and the MU-consistent construction (oracle-exact, `test_perturb.py`) brings no
measurable retention gain. The spun-up deviations are model-balanced but eight times smaller and decay too. The
spread-maintaining factor in A1/F1 was the per-member perturbed LBC (spread grew 1.5–2× in the first hour).

## Prospective Swiss arm F1 [M] — FAIL (`results/SWISS_OSSE.json`, `results/swiss_F1_report.json`)

Registered at 2026-10-17T12:36:39Z before it ran (critic item 4): the final code (estimator v2, rebalanced state
inflation, path/band fixes; source = critic-MERGEd c2ec3906ec), a NEW truth (seed 778, run 495 s / 608 MB under a
1 GiB cap) with new observation-noise seeds, the A1 configuration; A1's 03Z prior ensemble reused (identical member
inputs, deterministic engine) and re-scored against the new truth; analyses at 03/06/09Z, forecast to 12Z. Scored
with the corrected gate (registered times per cycle, explicit 03Z baseline, incomplete reports refused).

**F1** (PROSPECTIVE, new truth)

| t (prior) | prior spread T/U/V | prior RMSE T/U/V | posterior spread | posterior RMSE | written spread | 1st-h spread retention | analysis wall s / peak RSS MB |
|---|---|---|---|---|---|---|---|
| 03Z | 1.28 / 3.29 / 3.94 | 0.83 / 1.47 / 1.93 | 0.69 / 1.82 / 1.96 | 0.42 / 1.15 / 0.93 | 0.69 / 1.82 / 1.96 | – | 35 / 2195 |
| 06Z | 1.47 / 3.61 / 4.77 | 0.64 / 1.91 / 1.42 | 0.79 / 2.03 / 2.26 | 0.38 / 1.11 / 0.79 | 0.79 / 2.03 / 2.26 | 2.01 / 1.98 / 2.19 | 39 / 2359 |
| 09Z | 1.29 / 3.06 / 3.64 | 0.57 / 1.75 / 1.29 | 0.74 / 1.83 / 1.86 | 0.42 / 1.30 / 0.80 | 0.74 / 1.83 / 1.86 | 1.88 / 1.74 / 1.95 | 32 / 2363 |
| 12Z | 1.10 / 2.90 / 2.24 | 0.38 / 1.40 / 0.89 | – | – | – | 1.56 / 1.51 / 1.58 | – |

**Verdict: FAIL.**
- G1 FAIL: prior spread/RMSE 2.59 / 2.32 / 2.53 at 06/09/12Z, outside [0.5, 2.0]
  (overdispersive).
- G2 PASS: no compounding decay.
- G3 PASS: written members carry the posterior spread.
- G4′ PASS: mean posterior RMSE 0.810 vs prior 1.312 (−38 %).

The 03Z prior, before any analysis, is already 1.5 / 2.2 / 2.0 × the error of this truth (A1's truth: 1.0 / 1.7 /
1.2). The overdispersion therefore comes mainly from the initial/boundary perturbation amplitude (lagged-pool
combinations at scale 0.5), not from the inflation [I]. Nothing was retuned. A follow-up should calibrate the
perturbation amplitude and allow deflation (ρ < 1), then test prospectively on further truths. The post-hoc A1b
was replaced by F1 and never run.

## Variable localisation demo [M] (`varloc_demo.py`, `results/VARLOC_DEMO.json`)

The R3 (4) 0228 mechanism in isolation: temperature and wind perturbations drawn independently (true cross
covariance 0, so the coupling table is right by construction), 20 co-located stations observing T2 (σo 1 K) and
U10 (1.5 m/s), product `analyse` path, 40 random trials, interior lowest-level RMSE of the analysis mean vs truth:

| N | T: background / T2 only / T2+U10 / T2+U10 `surface_split` | T2+U10 worse than T2-only | U: background / no varloc / `surface_split` |
|---|---|---|---|
| 10 | 1.039 / 1.026 / **1.044** / 1.026 | 75 % of trials | 1.601 / **1.650** / 1.607 |
| 20 | 1.015 / 0.967 / **0.999** / 0.967 | 92.5 % | 1.575 / 1.569 / 1.525 |

Without variable localisation the co-located obs of the other variable make the analysis worse than the
background at N = 10 (both T and U), as in 0228 (T2 OmB 0.444 → OmA 0.520). With `surface_split` the T analysis
equals the T2-only analysis bitwise in every trial and the wind analysis gains. Doubling N reduces but does not
remove the cross-variable damage.

## Cost and memory

No forecast-program file is touched: 0 kernels, 0 device memory, 0 compile cost; the default `gpuwrf da` path
(no `inflation` section) is the probe path (`calibrate` delegates to `api.calibrate_scale`; default maintenance is
`core.rtps`). Host costs of the opt-ins:
- variable localisation: one LETKF weight set per coupling group; weights are n_pts·N²·8 B per group [I]:
  ALISIOS d01 (stride_h 2: 92,400 points) 0.30 GB at N = 20, 1.18 GB at N = 40; `surface_split` = 2 groups.
- adaptive inflation / obs bias: O(n_obs) statistics + one state-space rescale of the members (≈ one extra member
  pass); state files < 10 kB.
- additive inflation: one (N × M) weight draw per analysis, one `einsum` per pooled field; pool npz = M samples
  of the pooled fields in float32 (Swiss, M = 10, U/V/T/QV: ≈ 13 MB).
- N > archive pool: `combination` costs one (N × M) einsum per field at generation time.
Measured Swiss N = 20 analysis wall/peak RSS: see the twin-OSSE table.

## Limits

- The Swiss twin OSSE is one 126-km winter case over 12 h with ONE truth realization, synthetic observations and
  the original CPU-WRF as the member engine: spread/RMSE is a single sample (03Z prior, before any analysis,
  already 1.0 / 1.7 / 1.2 for T/U/V). The pre-registered calibration gate G1 failed (overdispersive, 2.12 at 09Z);
  absolute calibration needs several truths and/or deflation (adaptive ρ < 1), not tested here. The Canary development cases were not
  re-run (no cheap CPU-WRF member engine at that size under the current core plan; the GPU member driver is
  w41-da-cycle's).
- `obs_bias` removes the mean innovation of a group whatever its source: with a genuine MODEL bias it stops the
  analysis from correcting that bias [I]. It is meant for station/representativeness offsets (R3: T2 −1.9 K at
  stations below their model terrain).
- Adaptive inflation needs observations in a band; bands without observations keep their prior value.
- Perturbed LBC: `lbc_perturb` keeps MU frozen in the boundary file, so the MU part of the perturbation stays
  tapered and the outer rows keep a small mass–wind imbalance.
