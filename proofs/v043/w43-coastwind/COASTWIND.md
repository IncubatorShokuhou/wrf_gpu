# w43-coastwind — coastal near-surface wind at El Médano (V043 P1, ALISIOS fix list item 3)

Status: stage 1 (A+C) MERGE (critic DELTA2 2a9dbc0968) + addendum CW-A04; stage 2 = READY-FOR-CRITIC DELTA2 (Q-B GPU arm
CW-B02 measured in the 2026-10-21 08:30–09:18Z real slot, launcher v2 sha 8d10f474; critic VERDICT_STAGE2 FIX 6b9ad8d1 →
claim scoped to one config W9N 0316, history frames 0–3 h).

## Verdicts (machine-readable: analysis/COASTWIND.json)
| item | verdict | basis |
|---|---|---|
| A engine at the coast | **No evidence of a systematic coastal port wind bias in the F20 cases** [M]; CPU floor (72 h) and matched v0.4.2 72 h check PENDING | F20 v0.4.0 GPU vs original CPU-WRF, same inputs: mean GPU−CPU WS10 bias ≤ 0.007 m/s on every mask/point; pooled RMSE and maxima over leads 1–22 are of the same size as a CPU-vs-CPU decomposition twin (twin 3.06 m/s at the same NW Teno cell at τ22, GPU 3.65 at τ22; the 3.95 m/s GPU maximum at τ23 lies outside the scored twin leads) |
| B max_dom 9 vs 3 changes d03 | **WRF-inherent in original CPU-WRF** [M CPU; I mechanism]; **GPU v0.4.2 is max_dom-invariant on d01–d03** [M, one date W9N 0316, 0–3 h history frames only] — NOT the same kind; the v0.3.3 differences ALISIOS reported are not reproduced by v0.4.2 (their cause on v0.3.3 [U]) | CPU-WRF replicate bitwise; max_dom 3→9 changes d02/d03 at round-off (d01 bitwise); WRF couples/uncouples the parent in place once per child. GPU v0.4.2 bundle, same inputs, max_dom 3 vs 9: d01/d02/d03 377/377 variables byte-equal in all 4 frames (0–3 h) |
| C coastal deficit vs truth | **NOT_A_BUG as a port conclusion** [M]; conditional verification error remains real | all-hours bias vs T1 positive for the sea cell and every bilinear operator (sea +1.05, ALISIOS water point +0.68) and negative at the urban cell (−0.79); negative values appear in observation-selected strong-wind samples (ALISIOS point −0.49 vs T1, −1.32 vs T2), against T2, at the urban cell and in 4/33 issues for the sea cell (8/33 bilinear); ALISIOS confirms their figures are surf-day conditioned |

Labels: [M] measured, [I] inferred, [U] unknown. Reference = original CPU-WRF 4.7.1 (dmpar exe sha 1d0f5779…).

## Station mapping and model points (applies to every question)
ALISIOS point set (WT2 `POINTS.parquet`, schema teacher_extract.v2) folds `elmedano:ELMEDANO_T1_DESHIELDED`,
`elmedano:ELMEDANO_T2_ONWATER` and `cabezo:CABEZO_SURF` into one point, 28.04120 N / 16.53460 W, mapped to d03 grid
(y 19, x 55) at 605 m [M]. That cell is **sea** (LANDMASK 0, MODIS 17). The station sits almost exactly between the
sea cell centre (≈490 m S) and the **urban** cell (20,55) centre (≈510 m N; LANDMASK 1, MODIS 13 "Urban and Built-Up",
HGT 22 m). `windguru:14924` (El Médano Kite Club, 28.04245 N / 16.53493 W) is nearest to the urban cell (20,55).
Model points used below, always labelled: **sea (19,55)**, **urban (20,55)**, **vec-bil T1** (bilinear U/V then speed at the
T1 coordinate; first draft, WT2 only), **scal-bil T1** (bilinear of native scalar speed at T1; the F20 operator),
**scal-bil ALISIOS** (their surf-monitor water point 28.04105162 N / 16.53422754 W, scalar speed). Bilinear weights at T1:
(19,55) .330, (19,56) .183, (20,55) .313, (20,56) .174; at the ALISIOS point .321/.208/.285/.185 [M] — both include the
urban cell with ~0.3 weight. vec-bil vs scal-bil at T1 differ by −0.019 m/s on average, up to 1.46 m/s [M].
ZNT is not in the WN3 history stream (absent in all 144 paired files), so roughness is only seen through the drag proxy
(UST/WS10)² [M].

## Q-A ENGINE — does GPU differ from CPU-WRF at the coast? (analysis/QA_FIX.json, qa_fix.py; first-draft QA_F20_v040.json)
F20 v0.4.0 GPU vs original CPU-WRF, identical inputs, WN3 0227/0502/0614, 24 h (CW-A01, corrected). Statistic per mask:
mean bias; **pooled** space–time RMSE (√ of the mean over all cells × leads); the first draft reported the **mean of hourly
spatial RMSEs** instead (labelled "mean-hourly" here); max |Δ| with lead/cell/day-night [M]:
| case · domain · mask | bias | pooled RMSE | mean-hourly RMSE | max abs (lead, night/day) |
|---|---|---|---|---|
| 0227 d03 Médano box 7×7 | −0.0013 | 0.0752 | 0.0433 | 0.818 (τ24, night) |
| 0227 d03 sea (19,55) | +0.0068 | 0.0252 | 0.0165 | 0.066 (τ22, night) |
| 0227 d03 urban (20,55) | +0.0052 | 0.0258 | 0.0182 | 0.069 (τ24, night) |
| 0227 d03 coast-sea | +0.0009 | 0.1720 | 0.1304 | **3.947** (τ23, night; 28.288 N 16.896 W) |
| 0227 d03 coast-land | −0.0042 | 0.1309 | 0.1032 | 2.057 (τ22, night) |
| 0227 d03 open sea | −0.0002 | 0.1196 | 0.0844 | 6.233 (τ23, night) |
| 0502 d03 sea / urban / box | −0.0001 / −0.0015 / −0.0005 | 0.0068 / 0.0057 / 0.0077 | 0.0052 / 0.0036 / 0.0072 | 0.015 / 0.019 / 0.036 |
| 0614 d03 sea / urban / box | +0.0070 / +0.0061 / +0.0069 | 0.0133 / 0.0107 / 0.0127 | 0.0097 / 0.0078 / 0.0101 | 0.040 / 0.030 / 0.054 |
| d03 coast-sea 0502 / 0614 | −0.0022 / +0.0044 | 0.0221 / 0.0371 | 0.0208 / 0.0343 | 0.295 / 0.312 |
(d02 rows in QA_FIX.json; coast-sea pooled ≤ 0.030, max ≤ 0.62; coast-land max 3.15 in 0227 τ5.)

**The 3.947 m/s coastal-sea maximum (critic request) [M]:** 0227 d03 τ23 (2026-02-28 23 UTC, night) at 28.288 N /
16.896 W, HGT 2 m, a coastal sea cell (LANDMASK 0) at the NW Teno corner of Tenerife; CPU 16.0 vs GPU 20.0 m/s; W max 4.3/5.3 m/s within
±5 cells, hourly rain ≤ 0.08 mm [M]; 155 cells > 1 m/s in 11 patches — the event patch has 39 cells, the largest patch 46
(critic CR-D01) — RMSE outside the ±5-cell window 0.26 m/s [M]. Corner-jet / non-convective / round-off-sensitivity
mechanism is an inference [I]. **Control:** the original CPU-WRF twin (w43-twin0227, read-only) differs from the reference in MPI decomposition (4×1 vs
2×2) and in the IO setting restart_interval 180 → 100000 (no restart writes); it has its own coastal-sea maximum of
**3.059 m/s at the same cell** (τ22; GPU at that lead 3.649), and over the scored leads τ1–22 the GPU−CPU pooled RMSE is
1.0–1.4× the CPU−CPU one on every mask. τ23 (the 3.947 GPU maximum) is outside these scored leads (QA_FIX_t24.json adds τ23–24 when the twin frames exist):
| d03 mask, τ1–22 | CPU twin − CPU pooled / max | GPU − CPU pooled / max |
|---|---|---|
| coast-sea | 0.121 / 3.06 | 0.135 / 3.65 |
| coast-land | 0.102 / 1.96 | 0.111 / 2.06 |
| open sea | 0.076 / 3.96 | 0.090 / 4.53 |
| Médano box | 0.036 / 0.16 | 0.042 / 0.22 |
| sea (19,55) / urban (20,55) | 0.023 / 0.052, 0.021 / 0.055 | 0.023 / 0.066, 0.022 / 0.054 |
→ [I] the large pointwise differences sit at a flow-sensitive location where a CPU-only perturbation (different
decomposition) also produces m/s differences at the same cell; no GPU-specific signature is visible in τ1–22. This is one case and 22 leads;
the formal floor (A3 rule, 72 h) is w43-teacher's, and the twin's τ23–72 plus the v0.4.2 72 h pair are PENDING.
Original CPU-WRF is therefore also **not bitwise across MPI decompositions** (non-zero CPU floor) [M].

Earlier evidence kept, re-scoped: WT1 GPU v0.3.3 vs WT2 CPU on 0225/0303 (72 h, CW-A02) has a different moisture
initialization (RH2 v8.5 vs legacy) and **cannot isolate an engine effect**; it is context only. The "GPU == CPU to 0.01
m/s" statement applies only to the aggregate truth-comparison statistics in Q-C (F20 rows), not to fields or hourly points.
The ALISIOS deficit is **not** "2 orders of magnitude larger than any GPU−CPU difference": pointwise maxima reach 3.9 m/s;
it is about 100× the mean GPU−CPU bias at El Médano. Scoped conclusion: no evidence of a systematic coastal port wind
bias in these F20 cases.

## Q-B CONFIG — why does max_dom 9 vs 3 change d03 in a one-way nest?
Facts from ALISIOS (surrogate s0_9nest_post `SURROGATE_9NEST_P1_FINDING_20261011`, `REPLICA_P1B`) [M, theirs]:
inputs byte-identical (wrfinput d01–d03, wrfbdy, wrflowinp), namelist d01–d03 identical, only max_dom; same GPU engine
v0.3.3; a second identical 3-nest run is bitwise equal to the teacher (0.000000); 9-nest vs 3-nest differs already in
**d02** (0.008–0.08 m/s early, up to 0.65 m/s at τ25–48 in the DANA case 0316) and d03 (up to 0.94 m/s).
Stage-1 mechanism guess [I, from FINDINGS A67; **not confirmed on v0.4.2 by CW-B02 below**]: the GPU nest step runs the child domains as one fused XLA program (d02 + its leaves);
adding leaves changes module-global fusion/rounding decisions, so d02 differs at f32 round-off after one step, and the
difference grows chaotically (largest in unstable/DANA regimes). That is a program-composition (reproducibility)
property, not a physics change and not run-to-run nondeterminism.
**CW-B01 [M] CPU-WRF itself is max_dom-sensitive.** Original CPU-WRF dmpar (sha 1d0f5779), W9N
20260316_12z inputs (guarded stage S02), namelist identical except max_dom, feedback 0, decomposition 1×3 in all runs,
CPU-only OMPI env (libcuda 0 per rank), 1 h:
| pair | d01 +1 h | d02 +1 h | d03 +1 h |
|---|---|---|---|
| n3 vs n3 replicate | 375/375 byte-equal | 375/375 | 375/375 |
| n3 vs n9 | 375/375 byte-equal | 257/375 equal; U10 max 0.018 m/s, MU 0.14 Pa, 30.4k of 31.2k cells (not only leaf footprints) | 260/375; U10 max 0.019 m/s |
COSZEN identical, SWDNT 1 ulp (3e-5 W/m²) → solar geometry/cadence equal; the difference is round-off that then grows.

**Mechanism in WRF source [I, source-verified, not yet measured by a patched build]:** `frame/module_integrate.F:410-423`
calls `med_nest_force(parent, kid)` once per child after every parent step; `share/mediation_force_domain.F:111-123`
couples the **parent's own state in place** (`couple_or_uncouple_em(parent,.true.)`: ph_2, w_2, t_2, moist, scalar, u_2,
v_2 multiplied by column mass, `dyn_em/couple_or_uncouple_em.F:274+`) and `:191-202` uncouples it again (multiply by the
reciprocal). x·μ·(1/μ) ≠ x in REAL, so d02's prognostic state takes one rounding round trip per child per d02 step:
1 with max_dom 3 (d03), 7 with max_dom 9 (d03–d09). d01 has one child in both configurations → bitwise equal, as
measured. With feedback 0 nothing else couples siblings; physics module SAVE state (MYNN, MYNN-SL, Noah-MP, RRTMG,
Thompson, radiation driver/topo shading) was checked and carries no cross-domain state.
Deciding measurement for the mechanism (optional, CPU only): a WRF build that calls the parent couple/uncouple 6 extra
times in `med_force_domain` with max_dom 3 should reproduce the max_dom 9 d02 bit for bit.

**Port:** gpuwrf's nest force-down works on uncoupled fields and does not round-trip the parent state [I, source], so the
WRF mechanism above has no GPU counterpart; the remaining possible route was program composition (fused child cascade,
FINDINGS A67).

**CW-B02 [M] GPU v0.4.2 is max_dom-invariant on d01–d03 for W9N 0316, history frames 0–3 h.** Released v0.4.2 bundle (sha bfa2dc90) through the bundle
consumer + `run_parallel_cases --hours 3`, W9N 20260316_12z inputs (guarded stage S02), launcher
`analysis/maxdom/gpu_maxdom_v2.sh` (sha 8d10f474, commit 0d481f007f), one GPU lock session inside the 08:30–09:18Z real slot (waiter exit seen 09:08:03Z real).
Inputs: wrfinput d01–d03, wrfbdy_d01, wrflowinp d01–d03 sha256-identical in both arms; namelist differs only in
`max_dom` (3 vs 9; m9 adds wrfinput/wrflowinp d04–d09). Both arms rc 0, bundle check before/after BUNDLE_OK, no
hibernation in the window. m3 hit the bundle (compiled_new 0, 60.3 s wall); m9 missed the fused child key and compiled its own 9-nest program into
the lane layer (fused key 6894af7f vs m3 bundle key ac024b4c; d01 same bundle key in both; compiled_new 4127 = new cache
files — 2,925 JIT entries, 1,150 autotune records, 26 AOT meta + 26 AOT executables — not 4,127 program compilations;
2,154 s wall incl. cold compile — no timing, warm-cache or RW1 claim), so the two arms ran **different compiled programs**. `analysis/maxdom/cmp_maxdom.py` (same mutant-checked byte compare as CW-B01) →
`analysis/maxdom/GPU_MAXDOM_m3_vs_m9.json`:
| pair (GPU v0.4.2) | d01 | d02 | d03 |
|---|---|---|---|
| m3 vs m9, frames 0/1/2/3 h | 377/377 byte-equal ×4 | 377/377 ×4 | 377/377 ×4 |
No missing frame; d04–d09 written in m9 (4 frames each). So for this one date and configuration, over the 0–3 h history frames, the extra
leaves change neither the d01–d03 numbers nor their output. Hourly output equality is not proof of complete carry
equality; other dates, later leads/72 h and other forcing branches are **not qualified** by this check.
Context, not a floor [M]: GPU m3 vs the CPU-WRF n3 run (1×3 decomposition) at +1 h: d03 U10 max |Δ| 0.777 m/s, d02
0.353, d01 0.134 (`analysis/maxdom/CPU_n3_vs_GPU_m3.json`) — the engine difference at +1 h is ~40× the CPU max_dom
effect (0.019).

**Reading.** The max_dom sensitivity is a property of original CPU-WRF (in-place parent couple/uncouple per child), at
round-off level that then grows. GPU v0.4.2 does not have it: max_dom 3 and 9 give the same d01–d03. The 9-vs-3
differences ALISIOS measured were on GPU v0.3.3 (`SURROGATE_9NEST_P1_FINDING_20261011`); v0.4.2 does not reproduce them
on the same 0316 inputs over 0–3 h, so their v0.3.3 cause is [U] — this measurement neither explains nor generally closes
that historical result (a composition effect in that release is the candidate
[I]; not pursued — no v0.3.x fixes). Consequences for ALISIOS: (1) for W9N 0316 over 0–3 h, GPU v0.4.2 9-nest d03 equals 3-nest d03; general
9/3-nest pooling of GPU d03 (other dates, later leads, 72 h teacher labels) is **not qualified** by this check;
(2) CPU-WRF teacher labels from 9-nest vs 3-nest runs are different realizations (round-off seeded, growing in
convective/DANA regimes) → compare/pool CPU teachers only at the same max_dom, or treat the difference like a CPU twin
floor; (3) a GPU 3-nest vs a CPU 9-nest teacher carries that CPU realization spread in addition to the engine difference.

## Q-C PHYSICS-AS-WRF
- [M] CPU-WRF around El Médano (F20 24 h, 3 cases): the urban cell (20,55) is 25–35 % slower than the sea cell (19,55)
  (0227 11.1 vs 15.1, 0502 4.5 vs 5.9, 0614 6.7 vs 9.1 m/s); drag proxy (UST/WS10)² .028 vs .002. Noah-MP MPTABLE MODIS
  class 13 Z0MVT = 1.00 m (HVT 15 m) with sf_urban_physics = 0; barren (16) ~0, water Charnock. This is a neighbouring-cell
  contrast, **not** a measured effect of changing land use (no sensitivity arm was run; magnitude of such a change [U]).
- [M] WT2 CPU 33 issues × 72 h mean WS10: sea 6.92, urban 5.07, vec-bil T1 6.46, 7×7 sea mean 7.46, box max 8.63
  (analysis/QC_WT2_sector_hour.json: urban/sea 0.66–0.77 in every 30° sector).
- [M] CPU-WRF config variant (w41-topowind, 0227 24 h: YSU + revised MM5 sfclay + topo_wind 2/1/1): urban cell 7.9 vs
  11.1 m/s default, sea 12.9 vs 15.1 (lower).

### What the El Médano "truth" is (ALISIOS admission metadata + read-only ALISIOS code) [M source]
- `ELMEDANO_T1_DESHIELDED` is **not a separate sensor**: T1 = Cabezo raw wind + Δ(30° sector × Cabezo speed bin) from the
  frozen `config/elmedano/elmedano_correction_model_v1.json` (applier `src/alisios/getter/modules/elmedano_correction.py`);
  the Δ table was fitted with target = **max(Cabezo, windguru WG)**. Cabezo: ~5 m AGL on a building, shielded (station
  meta: "under-reads wind (shielding)"); exact mast height unknown.
- `ELMEDANO_T2_ONWATER` = T1 × marine uplift 1.08–1.12 (by sector).
- Trade sector (60°): Δ = 0.16–1.08 m/s for Cabezo 2–10 m/s; sectors 150–240° 1.1–3.7 m/s.

### Truth comparison [M] (A-V043-COASTWIND-T1-TRUTH-EXPORT, guarded read R01; obs qc 0, mean of samples within ±10 min of the model valid time; analysis/QC_TRUTH*.json)
WT2 CPU-WRF, 33 issues × 72 h, n = 2359 T1 matches. Bias model − obs (m/s); obs-selected surf = T1 ≥ 15 kt and 10–18 local;
fcst-selected surf = the operator's own value ≥ 15 kt and 10–18 local:
| operator → truth | all hours | 10–18 L | obs-selected surf (n 439) | fcst-selected surf |
|---|---|---|---|---|
| sea (19,55) → T1 | **+1.045** | +0.174 | +0.027 | +0.920 (n 403) |
| sea (19,55) → T2 | +0.554 | −0.464 | −0.798 | +0.104 |
| scal-bil ALISIOS → T1 | **+0.677** | −0.249 | **−0.492** | +0.447 (n 376) |
| scal-bil ALISIOS → T2 | +0.186 | −0.887 | −1.318 | −0.376 |
| scal-bil T1 → T1 | +0.610 | −0.307 | −0.570 | +0.362 (n 374) |
| vec-bil T1 → T1 (first draft) | +0.591 | −0.318 | −0.577 | +0.355 (n 374) |
| urban (20,55) → T1 | −0.792 | −1.599 | −2.292 | −1.002 (n 228) |
| sea → Cabezo raw / windguru | +1.77 / +1.86 | | | |
On the common sample where T1, T2, Cabezo and WG all exist the sea-cell signs are unchanged (obs-selected +0.023,
fcst-selected +0.944 vs T1 — numbers from the critic's independent reconstruction CR-C01b; this lane's own common-sample
all-hours value is +1.07, QC_TRUTH.json), so sensor availability does not explain the sign change. Per issue, all-hours sea → T1 bias is
negative in 4 of 33 issues (0423 −1.03, 0531 −0.16, 0615 −0.80, 0617 −1.16); with the bilinear operators 8 of 33; at the
urban cell 26 of 33. By observed-speed bin the sea-cell bias falls from +1.8 (T1 < 4 m/s) to ≈ 0 at 8–12 m/s (−0.6 vs T2);
by forecast-speed bin it rises from −0.5 (< 4 m/s) to +3.9 (≥ 12 m/s); by observed direction NNE +2.4, ENE +0.07, E −0.6.
F20 GPU and CPU rows give identical aggregate truth statistics to 0.01 m/s (aggregates only; pointwise see Q-A).

**Reading:** a conditional-bias finding consistent with sample selection. Over all hours the sea cell and every bilinear
operator are positive vs T1 (too windy, mostly weak wind and NNE/N) while the urban cell is negative (−0.79); negative
values like those ALISIOS reported appear in observation-selected strong-wind samples, against T2, at the urban /
land-weighted points, and in individual issues (sea cell 4/33, bilinear 8/33). This sample (Jan–June, ±10-min matching) does not reproduce ALISIOS's
October surf episode or their hourly observation operator (no October truth read), so it does not show that the October
event deficit was spurious; it shows that the deficit is what conditional verification of an otherwise positively biased
point forecast produces, and that the port adds no detectable bias in the F20 cases (aggregate truth statistics GPU vs CPU identical to 0.01 m/s;
mean GPU−CPU WS10 bias ≤ 0.007 m/s; pointwise differences up to 3.9 m/s, see Q-A). Verdict: NOT_A_BUG as a port conclusion; the
conditional verification error itself remains meaningful for ALISIOS (strongest observed winds under-forecast, weak winds
over-forecast at this point).
Options for ALISIOS (unvalidated product/configuration choices, none tested here): score per wind class and with
forecast-conditioned or full samples; choose the model point by exposure; apply any on-water uplift consistently to model
and truth; a land-use sensitivity (urban cells at El Médano → barren/grass) would need its own WRF arm before any claim.

### ALISIOS answer (cross TO_WRF_GPU2_COASTWIND_ANSWER_20261021.md, 2026-10-20T23:07Z)
Their "0.7–2 m/s below" comes from (1) surf-monitor-w1: 9-nest v0.3.3 nightly, leads 18–48, scalar-bilinear water point
28.04105 N / 16.53423 W (the "scal-bil ALISIOS" operator above), T1, 09–18 UTC on three observed surf days 15.–17.10.2026
(Δ −2.24 / −1.05 / −2.18) and (2) surf-hybrid-v1: 7 July–August days, −0.66 m/s; their all-hours fit pool implies raw ≈
+0.66 above T1 (our same operator over Jan–June: +0.68). Both are conditioned on observed surf situations. ALISIOS accepts
NOT_A_BUG pending the critic; consequence for them is metric-side (assess per wind class).

### Addendum (after stage-1 MERGE): CPU-twin control over the full τ1–24 [M] (analysis/QA_FIX_t24.json, corrected qa_fix.py)
Twin frames τ23–24 became available; same pairing as above (0227, twin 4×1 + restart_interval 100000 vs reference 2×2):
| d03 mask, τ1–24 | CPU twin − CPU pooled / max (lead) | GPU − CPU pooled / max (lead) |
|---|---|---|
| coast-sea | 0.138 / 3.059 (τ22) | 0.172 / 3.947 (τ23) — same cell (47,20) |
| coast-land | 0.121 / 1.963 (τ21) | 0.131 / 2.057 (τ22) |
| open sea | 0.098 / 5.583 (τ23) | 0.120 / 6.233 (τ23) — same cell (38,19) |
| Médano box | 0.052 / 0.328 (τ23) | 0.075 / 0.818 (τ24) |
| sea (19,55) | 0.049 / 0.199 (τ24) | 0.025 / 0.066 (τ22) |
| urban (20,55) | 0.041 / 0.144 (τ23) | 0.026 / 0.069 (τ24) |
d02 coast-sea/coast-land/open sea: twin 0.028/0.050/0.023, GPU 0.030/0.070/0.027 (pooled). GPU/twin pooled ratio 0.5–1.4
(GPU smaller than the twin at the El Médano cells, larger by ≤ 1.44× on the coastal/box masks); the open-sea maxima of
both pairs sit at the same cell and lead. Event patch 39 / largest 46 cells (GPU), twin 17 / 27. One case, 24 h; the 72 h
A3 floor stays with w43-teacher.
