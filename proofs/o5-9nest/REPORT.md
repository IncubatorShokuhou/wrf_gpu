# #148 — v0.4.2 9-nest D6 FAILs: per-row class (a) GPU fidelity / (b) CPU realization floor / (c) integrity (lane o5-9nest)

**Rev 2.1, 2026-10-24.** Opus 5.5 worker. Rev 2.1 corrects a lead-labelling error of mine in the 0420 twin analysis (R5/R7, see "Correction") and adds the R0 CPU-CPU control. Rev 2 applies critic cr-o5-9nest verdict FIX (`<USER_HOME>/wrf_gpu2_lanes/cr-o5-9nest/VERDICT.md`,
sha cdc35276) F1–F6 to rev 1 (a830358986).

Context:
- Continues lane w43-148fid (REPORT rev2 + sibling test, critic MERGE: cause INCONCLUSIVE, no engine bug localised, fused-sibling
  cross-talk not observed).
- Labels: [M] measured, [I] inferred, [U] unknown.
- **No threshold, manifest or scope change.** The frozen D6 FAILs stay FAILs; every floor below is a PROPOSAL for ALISIOS.
- No GPU run. ALISIOS reads go only through admission **A-O5-9NEST-148** (sha `c0a2eb96…`) with the sealed GuardedRead facade
  (receipts `<USER_HOME>/wrf_gpu2_lanes/o5-9nest/receipts/`, 0 refused targets; one setup refusal of my own unlisted npz open,
  disclosed). Provenance details are in the "Provenance" section.

The case:
- issue `20261018_12z`, WRF start 18.10. 18Z; τ = hours since start;
- CPU-WRF 4.7.1 (1d0f5779) 2×2 reference a2/a3 vs GPU v0.4.2 warm72 (bundle bfa2dc90; the v0.4.3 programs are identical);
- tree: d01 9 km → d02 3 km → seven 1 km leaves d03–d09, feedback 0;
- **no CPU-vs-CPU twin of #148 exists** (ALISIOS). The only CPU decomposition twin with W/U10/V10 is #149 0420 (C 2×2 vs C2 4×1,
  3 nests, 72 h), admitted as order-of-magnitude context. ALISIOS granted a CPU window tonight for an own twin (see P1).

## Verdict

- **No row is decided.**
- **A known WRF-fidelity difference of the GPU default path is present in this run [M].** Lane o5-coast found it (MAILBOX
  15:27Z): the root d01 specified-boundary ring is pinned one RK step ahead of WRF (`operational_mode.py:3281–3325` in v0.4.2;
  opt-in fix `GPUWRF_ROOT_SPEC_STAGE_CLOCK` on lane/o5-coast).
- **The #148 GPU run shows its signature (R0; fit [M], attribution [I])**, inherited by d02 and the leaves. This is a coherent, deterministic d01→d02→leaf
  difference from step 1. Its contribution to the D6 breaches is [U]. Therefore:
  - **(a) is not excluded for any D6 row**, and especially not for the wind and W rows 5, 8, 9, 10, 13, 14, 15;
  - the rev 1 statements "no engine defect was found" and "fix candidates: none needed" are **withdrawn**.
- What the data show [M]:
  - The GPU fields are consistent with CPU-WRF in variance, grid-scale variance share and pooled convective intensity (R8):
    no difference was detected.
  - These tests cannot see timing or phase errors such as the root clock (critic shift mutant: all statistics 1.000 at r 0.41).
  - The binomial intensity test only detects large shifts.
- Against the only same-resolution transferred CPU floor (0420, lead-matched, R7):
  - leaf rain rows 4, 7, 11, 12 lie within it at most breach leads;
  - rows 1, 2, 3, 5, 6, 8, 9, 10, 13, 14, 15 lie beyond it at most or all breach leads.
  
  0420 is a weaker convective regime, so "beyond the transferred floor" does not mean (a), and "within" does not mean (b).
- **Integrity rows 16–20 [M]:**
  - **Rows 16–17:** reduced to a trace-level difference of the parent d02 at the same frame. The GPU d02 is exactly 0 on the
    interpolation support, while the CPU d02 has 5.7e−12 kg/kg. This is the G-INT signature one level up; passed as a lead
    to w44-gint. It is not a writer, producer or nest-boundary defect (the GPU ring-0 path is WRF-faithful, C1).
  - **Row 18:** a parent realization difference. Ring-0 graupel traces alternate between the runs.
  - **Rows 19–20:** interior traces; their fields are active in both runs.
- **Deciders:**
  1. o5-coast's root-clock fix arm on WN3 0227 (X_ON/C vs X_OFF/C); if that moves X/C, a #148 GPU arm with the fix;
  2. tonight's own CPU twin (P1) for the noise floor;
  3. w44-gint for rows 16, 17, 19, 20.

## Per-row table

X = GPU−CPU RMSE. All 15 D6 rows reproduce the ALISIOS frozen table exactly (breach count, max X, lead of max; `rows148.py`).

"Lead-matched" = the breach leads where the pattern correlation r_X(τ) ≥ r_C(τ) of the 0420 CPU-CPU twin at the same lead and
resolution (d02 ↔ 0420 d02; 1 km leaves ↔ 0420 d03) (`results/leadmatched.json`, rev 2.1 alignment).

| # | stage | dom | field | failing τ | max X @τ (limit) | lead-matched vs 0420 | class (rev 2) | decider |
|---|---|---|---|---|---|---|---|---|
| 1 | 24h | d02 | RAINNC | 17–24 | 1.70 @24 (1) | 0/8 | undecided — beyond the same-resolution transferred floor; (a) not excluded | own twin P1 |
| 2 | 24h | d04 | RAINNC | 10–24 | 4.54 @18 (1) | 3/15 | undecided — beyond the transferred floor at most leads | P1 |
| 3 | 24h | d05 | RAINNC | 10–24 | 2.57 @23 (1) | 2/15 | undecided — beyond at most leads | P1 |
| 4 | 24h | d06 | RAINNC | 20–24 | 14.40 @24 (1) | 3/5 | undecided — within at most leads; #149 link | P1; o5-heavyrain |
| 5 | 24h | d06 | W | 22–24 | 0.564 @24 (0.3) | 0/3 | undecided — beyond; (a) not excluded | root-clock fix arm + P1 |
| 6 | 72h | d02 | RAINNC | 17–72 | 5.95 @72 (1) | 0/56 | undecided — beyond the same-resolution floor; (a) not excluded | P1 |
| 7 | 72h | d04 | RAINNC | 10–72 | 7.71 @72 (1) | 37/63 | undecided — within at most leads | P1 |
| 8 | 72h | d04 | U10 | 59–63 | 2.28 @59 (1.5) | 0/4 | undecided — beyond; (a) not excluded | root-clock fix arm + P2 |
| 9 | 72h | d04 | V10 | 59 | 1.59 @59 (1.5) | 0/1 | undecided — beyond; (a) not excluded | fix arm + P2 |
| 10 | 72h | d04 | W | 55–66 | 0.558 @59 (0.3) | 0/4 | undecided — beyond; (a) not excluded | fix arm + P1 |
| 11 | 72h | d05 | RAINNC | 10–72 | 2.60 @42 (1) | 46/63 | undecided — within at most leads | P1 |
| 12 | 72h | d06 | RAINNC | 20–72 | 28.74 @42 (1) | 39/53 | undecided — within at most leads; #149 link | P1; o5-heavyrain |
| 13 | 72h | d06 | U10 | 25–34 | 1.90 @32 (1.5) | 0/6 | undecided — beyond; (a) not excluded | fix arm + P2 |
| 14 | 72h | d06 | V10 | 25–34 | 1.76 @33 (1.5) | 0/7 | undecided — beyond; (a) not excluded | fix arm + P2 |
| 15 | 72h | d06 | W | 22–35 | 0.626 @34 (0.3) | 0/14 | undecided — beyond; (a) not excluded | fix arm + P1 |
| 16 | integ24 | d06 | QRAIN τ12 | ring 0, 8 cells | CPU ≤ 6.27e−13 kg/kg | — | (c) reduced to a d02 trace-level difference of G-INT type [M] | w44-gint |
| 17 | integ24 | d06 | QNRAIN τ12 | ring 0, 8 cells | CPU ≤ 1.99e−3 #/kg | — | (c) as 16 [M] | w44-gint |
| 18 | integ72 | d04 | QGRAUP τ36 | ring 0, 8 cells | CPU ≤ 2.97e−12 kg/kg | — | (c) parent realization [M] (ring-0 graupel alternates between runs) | none |
| 19 | integ24 | d06 | GRAUPELNC τ24 | interior, 2 cells | CPU ≤ 4.48e−10 mm | — | (c) trace; later GPU surface-graupel onset is a lead | w44-gint / o5-heavyrain |
| 20 | integ72 | d06 | QRAINXY τ72 | interior, 1 cell | CPU ≤ 2.84e−9 mm/s | — | (c) trace, realization [M] | w44-gint |

## Evidence

**R0 — a lag signature matching the root boundary-clock defect is present in the #148 GPU run (lag fit [M]; attribution to the root clock [I])** (`lag148.py`; `results/lag148_tau{6,12,24}.json`;
method of o5-coast a4_mu_lag.py, PSFC only because MU is not in the extracts).
- The fit is mean_ring(GPU−CPU)(τ) = lag · mean_ring(dPSFC_cpu/dt), per ring.
- **d01, τ1–12:** ring 0 −0.2 s (the end-of-step re-pin hides it there); rings 1–9 +53…+65 s (r 0.76–0.89), i.e. ≈ one root
  dt (54 s); interior +54 s (r 0.79).
- **d01, τ1–24:** ring 0 −0.2 s; rings 1–9 +42…+69 s (r 0.67–0.83).
- **d02, inherited:** interior +47 s (τ1–12, r 0.76); rings 13–39 s at τ1–24.
- **Leaves:** d06 +40…+46 s on all rings at τ1–12 (r 0.55–0.66); the leaf fits are noisy at later leads.
- The τ1–6 window (5 points) gives unstable fits (d01 −15…−18 s, r 0.35–0.62) and is not used.
- **Amplitude:** mean |ΔPSFC| ≈ 0.2–0.5 Pa at τ1–12, far below the D6 limit of 120 Pa. Its effect on the convective
  divergence is [U].
- This matches o5-coast's WN3 0227 signature: ring 0 exactly 0, rings 1–9 ≈ +1 root dt, d02 inherited.
- **Control [M]** (`lag_0420.py`, `results/lag_0420_cpu_cpu.json`, 52 guarded reads): the same fit on the 0420 CPU-vs-CPU
  decomposition pair (C2 4×1 − C 2×2, τ1–12) gives d01 rings 0–9 −0.9…+0.8 s and interior −1.0 s, with mean |ΔPSFC|
  0.003–0.016 Pa; d02 −0.9…+0.9 s.
- So the lag is GPU-specific. In τ1–12 the GPU−CPU d01 seed (0.2–0.3 Pa, coherent) is ≈ 20–100× the decomposition seed (a
  different case and day: order of magnitude only).
- It is the coherent, deterministic part of the GPU−CPU seed in this run.

**R1 — anatomy (w43-148fid, critic MERGE) [M].**
- Error growth in each leaf coincides with its own convective evolution.
- Interior > nest ring.
- W squared error is 79–93 % within 3 km of ≥ 1 mm/h cells.
- CPU restart is bitwise continuous.
- Fused siblings are independent (CPU testbed 624/624, GPU PROD 1046/1046).

**R2 — the CPU realization floor exceeds D6 RAINNC on a convective day [M].**
- #149 C2 (0420): d03 RAINNC CPU-CPU RMSE 1.20/2.23/3.31/4.20 mm at τ17/24/48/72; GPU−CPU on that day 1.15/2.13/3.16/4.00.
- Own-twin cases 0227 / E-C (w43-teacher scorer JSONs, `floor/xc_table.json`): RAINNC X/C median 0.90–1.11 at τ12–72.
- Normalized: the CPU-twin RAINNC RMSE/mean is 0.65–1.04 on 1 km d03 (0227, 0420) and ≈ 2.1 on 0420 d02. #148 X/mean is
  d04 0.80–0.95, d06 0.82–0.85, d05 1.2–1.3, **d02 2.2–3.4 (above the 0420 d02 value)**.

**R3 — W, own-twin cases [M].** W X/C median 1.08–1.21 (max 2.09).

**R4 — #149 link (d06 rain) [M].**
- On 0420, cells with ≥ 10 mm were systematically fewer on the GPU (−12..−15 %).
- #148:
  - d06 heavy cells 0.80 (τ24) / 0.95 (τ48/72) of CPU;
  - d06 domain mean 0.85 (τ72);
  - d04 1.14;
  - pooled leaves 0.91/0.98/1.02.
- The cause of #149 stays with o5-heavyrain.

**R5 — wind/W excess over decomposition twins is general [M]; its cause is open [U].**
- On the TEACHER-OK own-twin cases, GPU-vs-CPU wind exceeds 4/3 of the decomposition-twin distance at 23–77 % of leads.
- GPU-vs-CPU r is below CPU-vs-CPU r at almost every lead for W/U10/V10 (0227 d02: W 3/61, U10 5/61, V10 1/61 leads with
  r_X ≥ r_C; E-C d02: 8/61, 4/61, 4/61; d03: 11–26/61; critic recount from `results/scalefree.json`).
- All three CPU decomposition twins (0420, 0227, E-C) diverge from τ1. Rev 2's "0420 bit-identical through τ6" was a lead-labelling
  error (see Correction).
- Two open explanations:
  - (i) the GPU's arithmetic is a larger, global round-off seed than a decomposition change;
  - (ii) the root boundary-clock defect (R0), a coherent seed that is not WRF's. On d01 it alone is ≈ 20–100× the decomposition
    seed at τ1–12 (R0 control).
- The decider is o5-coast's fix arm (X_ON/C vs X_OFF/C on 0227). The #148 wind/W excess is the same general signature.

**R6 — CPU floor for W/U10/V10 on a convective day (0420) [M]** (`results/cpu_twin_0420.json`, 438 guarded reads).
- d03 max C: W 0.16, U10 0.87, V10 1.02, RAINNC 4.20 mm.
- Minimum r over τ12–72:

  | field | d02 | d03 |
  |---|---|---|
  | W | — | 0.33 |
  | U10 | — | 0.89 |
  | V10 | — | 0.85 |
  | RAINNC | 0.85 | 0.69 |

- 0420 is less convective than the #148 leaves; magnitudes do not transfer.

**R7 — lead-matched, same-resolution comparison [M]** (table column; `results/leadmatched.json`).
- d02 is compared with 0420 d02, the leaves with 0420 d03, at the same lead (rev 2.1 alignment).
- **Rev 1 compared d02 with the 0420 d03 minimum (0.69). That was the wrong resolution (critic F2).** Against the d02 minimum
  0.85 over τ12–72, the d02 rows reach 2/8 and 9/56 of breach leads (25 % / 16 %), and lead-matched 0/8 and 0/56.
- rmse/std is above the 0420 value at every d02 breach lead.
- For rows 5, 9, 10, 13, 14, 15, no breach lead is within the transferred floor.

**R8 — distribution checks [M]; what they can and cannot show** (`stats_compare.py`, `intensity_symmetry.py`).
- σ GPU/CPU median 0.996–1.000 for W/U10/V10 on d04/d06.
- Grid-scale variance share: W 0.177 (CPU) vs 0.181/0.182 (GPU).
- W breach hours are dominated by single intense cells present in one run at that hour: d04 τ59 GPU 22.2 vs CPU 6.3 m/s; τ66
  CPU 23.4 vs GPU 6.2.
- Pooled over 41 active hours, GPU maxW > CPU maxW in 23/41 (p 0.53); p99.9|W| ratio 1.02; rain p99 ratio 1.02.
- **Limits:**
  - these statistics are blind to timing/phase errors (critic shift mutant: σ, max, p99 and grid-scale all 1.000 at r 0.41);
  - the 41 hours are serially dependent (37 of them d04/d06, mostly consecutive), so the binomial test rejects only for
    k ≤ 13 or ≥ 28, i.e. it detects only a ≥ 0.32 shift in the GPU>CPU fraction.
- Hence "no intensity bias **detected**", and R8 says nothing about R0.

**C1 — the GPU nest ring-0 code path is WRF-faithful under the v0.4.2 release defaults [M source; critic verified against
pristine `share/interp_fcn.F` bdy_interp1 and `share/sint.F`].**
- `GPUWRF_NEST_SCALAR_SPEC_FINAL=1` (v0.4.2 `_fast_defaults.py:126`) re-pins every nest moist/scalar spec zone at the end of
  each step, like WRF `spec_bdy_final` (`coupling/boundary_apply.py:2542`, BD95).
- That pin runs on the live-nest branch (`boundary_apply.py:333`), which is default ON (`integration/nested_pipeline.py:597`).
- Microphysics skips the spec zone (`runtime/operational_mode.py:8785`).
- There is no positivity floor under strict guards (`:11180`, `:1469–1470`).
- The force-down uses two records (`nesting/boundary_construction.py:708–720`).
- So at an hourly frame the leaf ring 0 is the SINT-interpolated parent value (shipped v0.4.3 source line numbers).

**C2 — parent/child trace check [M]** (`ring0_trace.py`, `results/ring0_trace.json`).
- Necessary condition "leaf ring-0 trace ⇒ parent trace on the SINT support (±2 parent cells)": 0 violations in either run.
- The converse fails in both runs, so the test is necessary-condition only. Examples: CPU d06 QRAIN τ9, 11, 52, 58; GPU τ11, 62.
- **Rows 16–17:**
  - at d06 τ12 the GPU d02 support has 0 nonzero cells, the CPU d02 support 8 (max 5.73e−12 kg/kg);
  - one hour earlier both parents had traces of 2.6–4.0e−10.
  - So the difference is itself a trace-level parent difference of the G-INT type → lead to w44-gint.
- **Row 18 (d04 QGRAUP):**
  - the GPU parent support is 0 over τ32–38, while the CPU had up to 2.03e−8 kg/kg at τ33;
  - ring-0 graupel traces alternate (GPU at τ30–31 and τ39, CPU at τ32–33 and τ36–37) → realization reading.

**C3 — interior single-cell traces [M]** (`rows148.py`).
- **Row 19:**
  - graupel aloft appears in both runs before τ24 (CPU from τ22, peak 2.4 g/kg; GPU from τ20, peak 0.82 g/kg);
  - first surface graupel CPU τ22, GPU τ30; nonzero frames 51 vs 43 → lead to w44-gint / o5-heavyrain.
- **Row 20:** CPU canopy drip in one cell, with no rain anywhere in d06 in that hour; QRAINXY is nonzero in 55/73 frames in
  both runs.

## Proposals for ALISIOS (their decision; we change no gate)

- **P0 — decide the root-clock contribution first.**
  - o5-coast's GPU WN3 0227 arm with `GPUWRF_ROOT_SPEC_STAGE_CLOCK` ON vs OFF, against the 0227 CPU twin (X_ON/C vs X_OFF/C).
  - If it moves X/C, a #148 GPU 9-nest arm with the fix (≈ 35 min GPU), scored like this report: X_fix per row vs X.
  - The fix changes default identities, which is a manager/Owner decision.
- **P1 — own CPU floor (GO for tonight, cores 5, 6, 7 + SMT 21, 22, 23).**
  - The **n18k5 twin**: d01, d02, d04, d05, d06 renumbered 1–5, nproc 3×2; covers all 15 rows.
  - Cost [I, 2-point timing fit t ≈ 106 s + 1.316 s per thousand cell-steps on 4 physical cores, SMT pair ≈ 1.5 cores from
    a2/a3]: ≈ 314 s per simulated hour → 72 h in ≈ 6.3 h; output ≈ 5.2 GB.
  - Renumbering is valid (`share/input_wrf.F:440–497` checks only dims, dx/dy and nest starts).
  - Tooling: `twin/make_twin_namelist.py`, `twin/run_twin148.py`, `twin/score_twin148.py` (frozen A3 form, missing twin
    leads = INCOMPLETE), `twin/night_twin148.sh`, `twin/stop_twin.sh`.
  - The twin separates noise (b) from anything systematic. It cannot tell (i) from (ii) in R5 by itself; combined with P0 it
    can.
- **P2 — wind rows.**
  - A decomposition twin may be too weak a seed for GPU-vs-CPU wind (R5).
  - A global-seed CPU twin (±1 ulp of the total θ in every wrfinput cell, `perturb_theta.py`) gives the round-off floor for a
    global seed. It cannot see a systematic defect, so it must be paired with P0.
- **P3 — ring-0 integrity rows:** judge a leaf's ring 0 against its parent on the SINT support at the same frame (as done
  here). Ring 0 stays in the integrity check: it is where nest-boundary bugs show (B36 class).
- **P4 — interior traces (19, 20) and the parent traces behind 16–17:** the general rule comes from w44-gint.

## Correction (rev 2.1)

- The first 0420 analysis (`cpu_twin_0420.py`, runs T0420/T0420b) used the issue time 2026-04-20 18Z as τ0. The WRF start is
  2026-04-21 00Z (issue + 6 h; first wrfout `2026-04-21_00:00:00`). All 0420 lead labels were therefore 6 h too large.
- Fixed: T0 = 2026-04-21 00Z in `cpu_twin_0420.py` and `lag_0420.py`. `results/cpu_twin_0420.json` was relabelled by −6 h (same
  files and values; `results/cpu_twin_0420.RELABEL_NOTE.json`), and `scalefree.json` / `leadmatched.json` were recomputed.
- Check: with the correct alignment, the 0420 CPU-CPU RAINNC RMSE is d03 1.20/2.23/3.31/4.20 mm at τ17/24/48/72 and d02
  0.17/0.56/0.67 at τ24/48/72, identical to ALISIOS's own C2 table.
- Effects:
  - the retracted claim "0420 bit-identical through τ6";
  - the lead-matched column (rev 2 printed an unshifted/+6 h pair; the +6 h value was the correctly aligned one, now the only
    value; row 8 0/4);
  - the R6 minima/medians over τ12–72 are numerically unchanged (W 0.33/0.70, U10 0.89/0.96, V10 0.85/0.96, RAINNC 0.69/0.78
    on d03; RAINNC 0.85/0.92 on d02).
- Classes and verdict are unchanged.

## Provenance (critic F6)

- **Derived extracts.** The w43-148fid derived npz (made under A-148/ADD1) are verified bit-identical to the
  A-O5-9NEST-148-admitted wrfouts:
  - S01b: d02/d04/d05/d06, 108/108 arrays, 12/12 sha256;
  - S02: d03/d07/d08/d09, 72/72 arrays, 8/8 sha256; this covers all leaves used by the R8 pooled intensity.
- **0227 and E-C numbers.** `floor/xc_table.json`, `floor/curves_0227_era5.json` and the 0227/E-C r values in
  `results/scalefree.json` come from the w43-teacher scorer JSONs (`<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score`, produced
  under the w43-teacher admission), not from A-O5-9NEST-148. `floor/twin0227_rain_means.json` is from our own 0227 CPU twin
  outputs.
- **Use of model-vs-model RMSE/r — confirmed by ALISIOS (F6 evidence).** cross
  `TO_WRF_GPU2_ANSWERS_o5-9nest_SCOPE_AND_149_HR07b_20261024.md` §1 (alisios_mgr 2026-10-24T17:47:55Z, sha256 `de2b60c0…`):
  `no_score` means skill scoring against observations/truth and product model selection; GPU↔CPU-WRF and CPU↔CPU-WRF
  (decomposition twin) fidelity metrics on the admitted paths are engine diagnosis covered by CALIBRATION_READ. Conditions
  (all met here): no observation, truth or score-store paths; reads only via guard_call.admit; results only in lane/proofs.
  Between the manager's hold (16:26Z) and this answer no new metric on admission data was computed.

## Reproduction

All scripts are in `proofs/o5-9nest/` and run CPU-only (`JAX_PLATFORMS=cpu`, nice 19). `<npz>` is the w43-148fid extract dir
(`<DATA_ROOT>/wrf_gpu2/worktrees/w43-148fid/analysis/data`).

| script | arguments | output |
|---|---|---|
| `gread148.py` | — | guarded-read helper |
| `spotcheck_npz.py` | `[samples label]` | S01b/S02 receipts |
| `rows148.py` | `<npz>` | `results/rows148.json` |
| `ring0_trace.py` | `<npz>` | `results/ring0_trace.json` |
| `cpu_twin_0420.py` | — | `results/cpu_twin_0420.json` |
| `scalefree.py` | `<npz>` | `results/scalefree.json` |
| `stats_compare.py` | `<npz>` | `results/stats_compare.json` |
| `intensity_symmetry.py` | `<npz>` | `results/intensity_symmetry.json` |
| `lag148.py` | `<npz> 12` | `results/lag148_tau12.json` |
| `leadmatched.py` | — | `results/leadmatched.{json,txt}` |
| `floor/xc_table.py` | — | `floor/xc_table.json` |

## Limits

- One case and one realization per model; no #148 CPU twin yet.
- The transfer from 0420/0227/E-C is regime-dependent [I].
- R8 statistics cannot see timing/phase errors.
- The R0 fit uses PSFC on hourly frames (MU not extracted); the fit is unstable for short windows.
- Column maxima in C2 drop level information.
- No observation truth and no skill score.
