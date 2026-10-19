# D6-R — regime-local add-on gate (spec only; w42-swdfid 2026-10-19, implementer: w42-scorer)

Requested by ALISIOS (#146 decision, 2026-10-19 01:56Z): a local slope/cloud check next to D6, as a scorer option
(masked regime statistics against a CPU-WRF noise floor). Added on manager request: the precipitation add-on P
(03:22Z, from the 0420 RAINNC anatomy) and the land afternoon residual R4 as a tracked statistic (03:26Z, #146 closed).
Revised after cr-w42-swdfid VERDICT 4fb6bacd (R3 pooling and attribution, R4 quantities and window, degenerate
floors). None of this changes the frozen D6 manifest or its verdict. This file is the specification; the AN01–AN08
scripts under `<USER_HOME>/wrf_gpu2_lanes/w42-swdfid/` are the reference arithmetic.

## Why D6 alone is not enough, and why a plain regime RMS is not either

- D6 pools the whole domain. 151 cloudy south-slope cells of a 112×94 d03 are invisible in it.
- At 36–39 h lead, the per-cell regime RMS is dominated by where the clouds are. In #146 (WN3 0420, d03 south slopes,
  04-22 12–15Z):
  - SWNORM RMS was F−C 233, B−C 125 and B′−C 154 W/m² (B′ = a second cold compile of B).
  - F's gap sat in the flat (cloud) term (RMS 169–257 W/m² per hour), not in the slope term where the fix acts
    (RMS 4–15 W/m²). The split (R2) is exact but carries the interaction in the flat term, so the two RMS values are
    not additive shares.
  - F−B exceeded the compile-noise spread B−B′ in only 16 of 33 daylight hours.
  - A second realization of the fix (F′, same source, fresh cold cache) did not repeat F's focus hours: RMS F′−C
    SWNORM 140/170/172/173 W/m² and T2 .42/.44/.51/.57 K (F 170/230/253/269 and .48/.59/.68/.84; B′ 149/133/186/143
    and .39/.45/.61/.52).

  A single run therefore cannot separate a code-path error from weather noise.
- The gate needs four things:
  1. a noise floor from at least three equivalent runs;
  2. statistics that are robust to cloud displacement;
  3. physics-path diagnostics that are less sensitive to cloud position, confirmed by a same-input oracle before any
     FAIL;
  4. persistence over hours and over days.

## Inputs

- Candidate X (GPU run) and reference C (original CPU-WRF, same inputs and namelist), hourly history.
- Equivalent members (noise floor): at least two further runs X′, X″ of the candidate's own code with the same inputs
  (same source tree; so at least 3 members, 3 pairs).
  - Options: a second cold compile (E99), or the same tree and cache with a documented 1-ulp initial-condition
    perturbation (T at one interior cell, written into a copied wrfinput; there is no CLI switch for this today).
  - Neither is guaranteed to diverge: a cold compile can come out bitwise identical, and a 1-ulp perturbation can be
    rounded or damped away. A member counts only after an observed non-degenerate control: it differs from X beyond
    round-off in the scored field at the scored hours or leads. Bit-identical or degenerate members are listed and
    excluded.
  - Floors use same-code, same-input pairs only. Cross-code pairs (e.g. F−B, a fix against the previous release) are
    descriptive and never a floor.
  - With fewer than 3 valid members, every floor-based result is INCOMPLETE: values are reported, no FLAG, EXCESS or
    NOISE label is assigned.
- Fields: SWNORM, SWDOWN, SWDNT, CLDFRA, T2, RAINNC (+ RAINC when cu_physics > 0), QCLOUD, QICE, MU, MUB, C1H, C2H, DNW
  (optional LH, HFX, TSK), plus the static fields HGT, XLAT, XLONG, LANDMASK, SINALPHA, COSALPHA, MAPFAC_MX and
  MAPFAC_MY. Take radt, dx and the shadow length from the namelist or global attributes.
- Column water paths use WRF's hybrid dry-mass layer weight −(C1H·(MU+MUB) + C2H)·DNW/g with g = 9.81:
  - LWP from QCLOUD;
  - IWP from QICE, separately.

  QCLOUD + QICE is a combined cloud-water path, not LWP. The simple eta weight (MU+MUB)·(−DNW)/g is wrong on WN3 grids
  (C1H/C2H nontrivial; up to 54 % upper-level layer-mass difference, cr-w42-swdfid 03:20Z).
- Do not use the GPU history COSZEN for daylight or geometry.
  - Up to v0.4.1 the GPU writer evaluated it at the output instant (`io/wrfout_writer.py` `_compute_coszen(lead)`).
  - WRF writes the held radiation-call value, calc_coszen at the last call + radt/2. The 0420 CPU history matches that
    to 2.4e-7.
  - lane/w42-swdfid makes the product writer emit the held value. The scorer still computes the held coszen itself
    (version-independent), using WRF radconst/calc_coszen at the last call time (step start of the last radiation
    step, every STEPRA = nint(radt·60/dt) steps from step 1) and solar time call + radt/2.
  - Always use that STEPRA/own-step recipe. The shortcut call T − radt and solar time T − radt/2 applies only when
    cadence and output are exactly aligned (e.g. WN3 d02/d03); it does not apply to WN3 d01 (dt = 54 s).

## Masks and classes

- Static masks from the reference frame 0:
  - land (LANDMASK > .5);
  - slope from WRF start_em (HGT with map factors and rotation): flat < 2°, sloped ≥ 10°;
  - aspect sectors N/E/S/W of 90° each, on sloped cells only.

  A mask with fewer than 50 cells is reported but not judged.
- Daylight: held coszen > 0.1.
- Cloud class per cell and hour, taken from the reference C only. This avoids selecting cells on the candidate's own
  error. kt = SWDOWN/max(SWDNT, 1 W/m²) and CLDFRA colmax is the column maximum.
  - clear: CLDFRA colmax < .01 and kt_C > .55;
  - partly: .25 ≤ kt_C < .55;
  - overcast: kt_C < .25.

## Metrics

Granularity:
- R1 and R2: per domain × mask × class × daylight hour;
- R3: pooled, as defined below;
- R4: per case-day;
- P: per lead.

**R1 — regime bias against the noise floor.**
- b = mean(X − C) over the cells, for T2, SWNORM and SWDOWN.
- n = max over the valid equivalent members X′ of |mean(X′ − C) − mean(X − C)|, i.e. the spread of the bias among
  same-code runs. It needs at least 3 valid members; otherwise, or if n = 0, R1 is INCOMPLETE.
- Also report RMS(X − C), RMS(X′ − C) and n_cells.

**R2 — SWNORM term split** (r = SWNORM/SWDOWN, cells with SWDOWN > 5 W/m²):
- flat = (SWDOWN_X − SWDOWN_C)·r_X, which is cloud amount and position;
- slope = SWDOWN_C·(r_X − r_C), which is the TOPO_RAD_ADJ factor.

flat + slope = SWNORM_X − SWNORM_C exactly; the flat term carries the interaction (it uses r_X). Report RMS and mean of
each term (AN01 `decompose.py`); never present the RMS values as additive variance shares.

**R3 — physics-path diagnostics (pooled; FLAG only, see the verdict rule).**

R3 is pooled per domain and candidate. That pooling reproduces the calibration (AN03 `rtest.py`, AN05 `effect.py`):
- Cells: land (LANDMASK > .5, reference frame 0).
- Hours: every hourly frame of the run (0–72 h for WN3) with reference held coszen > 0.1. The reference held coszen is
  the CPU-WRF history COSZEN, or the computed held value; the GPU history COSZEN is never used.
- r = SWNORM/SWDOWN where SWDOWN > 5 W/m² in both runs; kt = SWDOWN/max(SWDNT, 1 W/m²).
- n is the number of pooled (cell, hour) samples. Per-hour and per-aspect breakdowns are reported and never judged.

R3a, clear-sky slope factor:
- Selection: CLDFRA colmax < .01 and kt > .55 in both X and C.
- Report the signed mean and RMS of Δr = r_X − r_C, and the p99 of |Δr|.
- #146 F: mean 7.0e-6, p99 4.5e-4 on n = 15,611. The RMS (.0032–.0033) is set by a < 1 % tail that every arm shares
  (B and B′ too), so it is reported and not judged.

R3b, diffuse fraction at matched clearness:
- G = csza_slp/csza at the exact held time from the STEPRA/own-step recipe in Inputs, with
  G = 0 where shadowed. Use the WRF TOPO_RAD_ADJ/toposhad transcriptions with the domain dx and the namelist shadow
  length (AN04 `dinvert.py`).
- Invert d = (r − G)/(1 − G) where |1 − G| ≥ .2.
- Select matched samples: finite d in both runs, |kt_X − kt_C| < .02, .25 < kt_C < .6, and CLDFRA colmax > .3 in both.
- Statistic: the pooled mean(d_X − d_C), with n.
- #146: F +.002 (n 293); v0.4.0 B −.029 (n 314); B′ −.025 (n 350).

Both R3 statistics stay meteorology-dependent, so they are diagnostics and not proofs:
- clear-sky r also contains the atmospheric clear-sky diffuse fraction;
- R3b conditions on post-treatment cloud transmission (matched kt).

A threshold excursion is therefore a FLAG. It is attributed to the code path only after a same-input check (see the
verdict rule).

**R4 — land afternoon residual (tracked statistic; never a verdict on its own).**
- Window: the registered calibration window is 12–17 UTC on the Canary WN3 cases (6 hourly frames). Domain-centre
  longitude there is L ≈ −16.2°E (0420: d02 −15.8, d03 −16.6), so the window starts at about 10.9 h local mean solar
  time (LMST = UTC + L/15 h). Other cases use the same solar-time window: 6 hourly frames from
  h0 = round(10.92 − L/15) UTC to h0 + 5, with L the domain-centre longitude in °E. For the WN3 gate cases h0 = 12,
  i.e. exactly 12–17 UTC; Switzerland (L ≈ 8°E) gives 10–15 UTC.
- Per case-day and domain: A_X = mean over the window frames of the land-mean (LANDMASK > .5) X − C for:
  - T2;
  - kt (cells with SWDNT > 50 W/m²);
  - LWP (QCLOUD) and IWP (QICE), hybrid weight.
- Report A for every member and the same-code realization spread (max − min over at least 3 valid members, else
  INCOMPLETE). When an older-release pair exists in the same screen, also report the deviation from that pair's mean
  (the #146 F′ rule statistic, `fprime_rule.py`; cross-code, descriptive).
- Track the sign across case-days and releases. Send a FLAG to the fidelity owners (no gate effect) when |A_T2|
  exceeds the same-code realization spread with the same sign on at least 3 case-days.
- Keep it separate from D29 (night-time land cold bias): different hours.
- #146 calibration (0420, 12–17 UTC; cells d02 04-22, d02 04-23, d03 04-22, d03 04-23; land cells d02 1573,
  d03 2034). All descriptive: two members per code, so INCOMPLETE under this rule.
  - Registered F′ statistic D_F′ = F′ − (B+B′)/2: −.017/−.023/−.019/−.009 K, mean −.017 K. This is the tracked
    "−0.02 K land cold residual". The pre-registered verdict was INCONCLUSIVE (4/4 negative, 38 % of F's sum).
  - Fix pair − v0.4.0 pair (AN08b): T2 −.017/−.027/−.043/−.035 K (mean −.030); kt −.003/−.005/−.009/−.012;
    LWP +1.9/+7.3/+8.3/+14.9 g/m²; IWP 0.00 in every arm and cell (no ice). Within-pair T2 half-spreads: v0.4.0
    .003/.001/.003/.018 K, fix .0002/.005/.024/.026 K.
  - Against C:
    - T2: fix pair −.005/−.040/−.021/−.035 K, v0.4.0 pair +.012/−.013/+.022/−.001 K;
    - LWP: fix pair −0.3/+4.0/−3.1/+1.7 g/m², v0.4.0 pair −2.3/−3.3/−11.4/−13.2 g/m² (C 87/73/257/185 g/m²).

    These numbers are not attributed to any mechanism.

## Verdict rule (pre-register before any candidate output exists)

- **FLAG (physics path)** if, on any domain:
  - R3a: |mean Δr| > 1e-3, or p99 |Δr| > 5e-3, with n ≥ 1000;
  - R3b: |mean Δd| > .01 with n ≥ 100.
- **FLAG (regime)** if, for the same (mask, class, field):
  - |b| > n (R1, valid only with at least 3 members) with the same sign for at least 3 consecutive daylight hours,
    and
  - this happens on at least 2 distinct case-days of the gate set (WN3 3-case set: 9 days).

  Report the R2 split of the window with it. In #146 the window met the hour criterion on one day only (04-22, 13–15Z;
  12Z had the opposite sign), and the slope-term means were −1…−4 against flat-term means −20…−74 W/m². No FLAG.
- **FAIL (physics)** only after a FLAG and a failed same-input confirmation:
  - the confirmation runs the pristine-WRF caller/operator oracle on the candidate's own column inputs of the flagged
    domain and hours;
  - the template is `tests/v025/w41_swddir` (`swddir_oracle_runner.py`: RRTMG_SWRAD direct/diffuse with frozen
    bounds), plus the TOPO_RAD_ADJ/toposhad transcriptions for r;
  - it is FAIL only when that oracle misses its frozen bounds. A FLAG whose confirmation passes stays a FLAG for the
    consumer check (stations, ALISIOS).
  - #146: v0.4.0 (B, B′) exceeds the R3b threshold and F does not. The same-input oracle (w41-swddir) independently
    confirms the v0.4.0 direct-beam defect, so under this rule v0.4.0 would be FAIL and the fix is not flagged.
- **INCOMPLETE** for any floor-based metric with fewer than 3 valid same-code members or a zero floor.
- **REPORT always**: per-cell RMS (dominated by displacement), the R2 split, R4, the member list with validity and
  source SHAs, cell counts.

## P — accumulated precipitation against a GPU realization floor (add-on; frozen D6 unchanged)

D6 keeps its frozen RAINNC bound (RMSE ≤ 1 mm at every lead). P reports two things next to it:
- whether the candidate's precipitation error is larger than the error between same-code GPU realizations;
- whether totals or heavy-cell counts are biased against CPU-WRF.

Why: the 0420 anatomy (w41-swddirscreen `post/20260420_18z/RAINNC_ANATOMY.json`, sha 417889c4bc90…; leads 1–72,
pairs B−C F−C B′−C F′−C B′−B F′−F) shows the following on d03 (convective, cloudy):
- per-cell RMSE between two same-code GPU realizations is 0.85–0.92 × the GPU−C RMSE at every lead:
  - 24 h: B−B′ 1.91, F′−F 2.21 mm vs B−C 2.13, F−C 2.40, B′−C 2.16, F′−C 2.44 mm;
  - 72 h: 3.38, 3.54 mm vs 4.00, 4.11, 3.67, 4.07 mm.
- B−B′ alone exceeds the 1 mm D6 bound from lead 18 h (F′−F from 17 h, GPU−C from 17 h). At those leads the frozen
  bound is below the same-code realization spread. On d02 every pair stays below 1 mm.
- The domain totals are one-sided:
  - d03 domain-mean 72 h accumulation is C 4.509 mm vs GPU 4.118–4.241 mm (−6…−9 %, all four GPU runs below C, GPU
    spread ±1.5 %);
  - cells ≥ 10 mm: C 1454 vs GPU 1217–1271 (−13…−16 %); ≥ 25 mm: C 106 vs 69–97; max C 62.8 vs 40.4–58.2 mm;
  - at 24 h the GPU counts bracket C (≥ 10 mm: C 467 vs 375–506); bias² ≈ 1 % of the MSE.
- Where the error sits on d03:
  - at 24 h the top-20 cells hold 9.7 % of the B−C SSE (B−B′ 12 %), so the error is spread out;
  - land (20 % of cells) holds 34 % of the B−C SSE (B−B′ 44 %);
  - at 72 h, sea holds 78 % of the SSE for both B−C and B−B′.

Inputs:
- accumulated RAINNC (+ RAINC when cu_physics > 0) since the run start;
- the candidate X and at least 2 further valid same-code, same-input members (Inputs above);
- the CPU set: {C}, or {C, C2} where C2 is the same pristine WRF build with a different MPI decomposition (#149);
- LANDMASK and HGT.

Metrics: per domain, at every 6-h lead and at 24/48/72 h; also split land/sea and by HGT band (0–50, 50–1000,
> 1000 m).
- **P1 — realization-floor ratio.**
  - ρ = RMSE(X − C) / f, where f = max over the same-code member pairs of RMSE(Xi − Xj).
  - f = 0 (all valid pairs bit-identical) gives INCOMPLETE, with no ratio. Never divide by zero or assign a noise label.
  - Report the full RMSE table (every member−C pair, every same-code pair, and cross-code pairs as descriptive rows),
    ρ, and the anatomy shares (top-20 cells, land/sea).
- **P2 — domain-mean accumulation bias.** For each member, the domain mean (and land mean). Δ = mean over the GPU
  members − mean over the CPU set. Spreads: s_G = max − min over the GPU members; s_C = max − min over the CPU set
  (0 with one member).
- **P3 — heavy-cell counts.** Number of cells ≥ 1, ≥ 10 and ≥ 25 mm, and the domain max, per member; Δ, s_G and s_C as
  in P2.

Verdicts. P qualifies the D6 RAINNC result and never changes it.
- P1, per lead. It is judged only at leads where RMSE(X − C) exceeds the D6 RAINNC bound (1 mm), i.e. where D6 misses;
  elsewhere it is REPORT only. With small errors the floors are tiny and ratios meaningless: on 0420 d02 the
  same-code ratio reaches 1.54 although every d02 error stays below .78 mm.
  - **NOISE** if ρ ≤ 1.25: the D6 RAINNC miss is within 1.25 × the same-code realization floor;
  - **EXCESS** if ρ > 1.25 at two consecutive judged 6-h leads. This triggers a consumer check (stations, ALISIOS),
    not a FAIL;
  - **INCOMPLETE** with fewer than 3 valid same-code members or f = 0.
  - 0420 calibration (descriptive, two members per code, so INCOMPLETE under this rule): same-code floors (B−B′ for B
    and B′, F′−F for F and F′). d03 is judged at leads 17–72 h with ρ .93–1.20 (would be NOISE); d02 is never judged.
- P2/P3, per metric × lead (24/48/72 h): the #149 rule registered by w41-swddirscreen (`c2_rule.py`, sha 78f5f4e5…),
  applied two-sided:
  - **SYSTEMATIC** if every CPU member lies beyond every GPU member on the same side, and |Δ| > s_C + s_G;
  - **NOT_SYSTEMATIC** otherwise;
  - **INCOMPLETE** with a single CPU member or fewer than 3 GPU members: report Δ and s_G, make no claim.
  - The registered #149 instance for 0420 uses all four GPU runs {B, B′, F, F′} (two codes) as the GPU set. That is
    ALISIOS's frozen choice for that case and is not redefined here.
  - 0420 with C only: INCOMPLETE. For reference, 72 h domain-mean Δ = −.32 mm (−7.2 %) with s_G = .12 mm; ≥ 10 mm
    cells Δ ≈ −183…−237 with s_G = 54. C2 is running (#149).

## Output and hygiene

- One JSON per case and domain:
  - masks with cell counts and the held-geometry recipe;
  - per-hour rows for R1 and R2, the pooled R3 rows with n, per case-day rows for R4;
  - per-lead P rows (RMSE table, ρ, shares, means, counts) and the P verdicts;
  - the member list (same-code, cross-code) with validity (non-degenerate control) and source SHAs;
  - the verdicts, with the rule SHA frozen before scoring.
- E200: reject non-finite values in every candidate, member and reference before any max, ratio, floor or count, and
  carry NaN/+Inf/−Inf regression tests for both candidate and reference.
- Score domains sequentially (72 h WN3, memory).
- E184: attest the measured process inputs.
- E199: run every in-repo consumer of the shared tool.
- Cost: CPU only, about 5–10 min per 72 h domain on 2 cores (AN04: d03 72 frames ≈ 6 min).

## Calibration on #146 (0420, d03), for the implementer's tests

R3 below is pooled as defined above (d03 land, all frames with reference held coszen > 0.1).

| check | F (v0.4.0 + fix) | B (v0.4.0) | B′ |
|---|---|---|---|
| R3a clear mean Δr / p99 \|Δr\| (n) | 7.0e-6 / 4.5e-4 (15,611) | 9.1e-6 / 4.3e-4 (15,577) | 7.0e-6 / 3.6e-4 (15,698) |
| R3b mean Δd (n) | +.002 (293) | −.029 (314) | −.025 (350) |
| R2 slope-term RMS, 04-22 12–15Z | 4 / 7 / 11 / 15 W/m² | 4 / 4 / 4 / 3 | 6 / 4 / 5 / 7 |
| R2 flat-term RMS, 04-22 12–15Z | 169 / 224 / 244 / 257 | 122 / 141 / 127 / 98 | 146 / 130 / 182 / 139 |

P calibration (0420 d03, RAINNC_ANATOMY.json 417889c4; the floors are same-code, the cross-code columns descriptive):

| lead | RMSE B−B′ / F′−F | RMSE B−C / F−C / B′−C / F′−C | ρ B / B′ / F / F′ | domain mean C vs GPU | cells ≥ 10 mm C vs GPU |
|---|---|---|---|---|---|
| 24 h | 1.91 / 2.21 mm | 2.13 / 2.40 / 2.16 / 2.44 mm | 1.12 / 1.13 / 1.09 / 1.11 | 2.135 vs 2.035–2.111 mm | 467 vs 375–506 |
| 48 h | 2.73 / 2.94 mm | 3.16 / 3.46 / 3.00 / 3.35 mm | 1.16 / 1.10 / 1.18 / 1.14 | 3.509 vs 3.147–3.285 mm | not in the anatomy |
| 72 h | 3.38 / 3.54 mm | 4.00 / 4.11 / 3.67 / 4.07 mm | 1.18 / 1.09 / 1.16 / 1.15 | 4.509 vs 4.118–4.241 mm | 1454 vs 1217–1271 |

Recommended unit tests:
- synthetic r = d + (1 − d)G fields with a known Δd: R3b recovers it;
- a mutant that drops the shadow mask raises R3a;
- an injected cloud displacement moves R1 and R2-flat but not R3a;
- pooling: the R3 n reproduces the calibration counts on the #146 arms;
- the hybrid weight: a mutant with the simple eta weight changes R4 LWP on a WN3 frame; LWP uses QCLOUD only;
- floors: two bit-identical members give INCOMPLETE (no division by zero, no NOISE label), and so do 2 valid members;
- P: a pure displacement of a rain cell raises ρ but leaves P2 unchanged; a multiplicative bias of the candidate's
  accumulation moves P2/P3; with C2 := a GPU member the P2/P3 verdict is NOT_SYSTEMATIC; with one CPU member it is
  INCOMPLETE;
- E200: NaN, +Inf and −Inf in any candidate, member or reference are refused before every ratio, max or count.
