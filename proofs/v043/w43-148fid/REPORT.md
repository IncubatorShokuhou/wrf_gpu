# #148 — v0.4.2 9-nest D6 fidelity FAIL: analysis (lane w43-148fid; Q1–Q6 CPU only, sibling phase 2 on GPU)

2026-10-20, Opus 5.5 worker. Case `20261018_12z` (WRF start 2026-10-18 18Z, τ = hours since start), GPU v0.4.2 warm72
vs the original CPU-WRF 4.7.1 reference `cpu_reference_a2_a3`. Labels: [M] measured, [I] inferred, [U] unknown.
The Q1–Q6 analysis and the N18 twin cost used no GPU run; the only GPU runs are the separately authorized PROD
sibling-independence phase 2 (own section below, manager leases). No threshold/manifest/scope change, no
retiming/renaming. Every ALISIOS read went through the
guard: `guard_call.admit` with admissions A-148 (`b0c718cd…`) and ADD1 (`dfd73f98…`) (receipts in `receipts/R0*`,
MAILBOX GUARDED READS lines 10:35Z and 11:21Z).

**Revision 2** (critic cr-w43-148fid verdict PARTIAL/FIX on f3890578ca, `<DATA_ROOT>/wrf_gpu2/worktrees/cr-w43-148fid/analysis/VERDICT.md`):
all seven required corrections applied (overall cause INCONCLUSIVE; qualified pre-convection table; association
wording + registration-correct footprint numbers; mask-radius table; no #149 exclusion; corrected displacement
list; Q5 split by field/unit). The measurements themselves were reproduced by the critic (raw RMSE 876 checks
0.0 diff; restart 375 × 9 bytes; analysis mutants killed).

## Verdict

**Overall cause: INCONCLUSIVE. No ENGINE-BUG is localised.** Convective realization (round-off-level differences
amplified by convection) is a plausible hypothesis that fits the error anatomy. The anatomy does not exclude a
defect that acts only under active convection. Fused-sibling cross-talk was NOT observed: the sibling-independence
test passed on CPU (dynamics-only testbed) and on the GPU (real PROD d01 + 3 × d02, release defaults, ordinary and
radiation phases); see the sibling section for its limits. The N18 CPU-WRF twin remains the open discriminator.
The frozen D6 failures remain failures.

| Q | verdict | key evidence |
|---|---|---|
| Q1 timeline | INCONCLUSIVE; divergence is ASSOCIATED with local convective evolution | Error growth in each leaf coincides with its own rain evolution (not top-down from d01); interior > ring; pre-convection errors are small vs D6 and broadly comparable to a released WN3 case (54-row table: ratios 0.18–411, median 1.12) — not equality |
| Q2 CPU restart | Restart bitwise continuous; d01 name offset = CPU-REF-ARTEFACT | a2 vs a3 at τ40: 375/375 variables byte-identical on all 9 domains; all 24 h FAILs (τ10–24) predate τ39 |
| Q3 W/U10/V10 | W localised at rain cells; U10/V10 INCONCLUSIVE | W: 79 % (d04) / 93 % (d06) median SSE in a 3 km mask covering 26 % / 51 % of the area; outside-W radius-dependent (d04 0.301 at 0 km … 0.168 at 5 km). d04 τ59 U10 outside the mask 2.213 m/s > 1.5 |
| Q4 RAINNC | INCONCLUSIVE; no #149 exclusion | Totals at 72 h mixed sign (0.85–1.14); leaf-only heavy cells ≥ 10 mm at τ24 0.908; d06 τ24 0.799, τ72 domain mean 0.850 |
| Q5 integrity zeros | Consistent with realization (trace values); raw integrity verdicts stand | No permanently dead field: d06 QRAIN/QNRAIN nonzero in 65/73 GPU frames, GRAUPELNC 43, QRAINXY 55; reversals GPU>0/CPU=0 exist. Does not exclude an intermittent producer defect |
| Q6 noise floor | an N18 CPU twin WOULD decide the A3 floor rule | 0420 C2−C (decomposition only) d03 RAINNC 1.20/2.23/3.31/4.20 mm at 17/24/48/72 h > 1 mm; no CPU−CPU W/U10/V10 exists anywhere |
| schemes | default suite only; 7-sibling fused geometry is new | see below |

## Data and reproduction

- Extraction `analysis/extract.py <dom> <label>` (GuardedRead; 146 files/domain; d01 72 pairs) → `analysis/data/dXX.npz`
  (untracked, 2.3 GB, delete after critic). Analysis `analysis/analyze.py d02 … d09` → `q_dXX.json`;
  `q1_footprint.py`, `q2_restart.py`, `q3_radius.py`, `sha_sample.py`, `build_summaries.py` → `Q1.json … Q5.json`,
  `Q3_radius.json`. CPU: `taskset -c 8,9,12 nice -n 19`, `scripts/with_ram_slot.sh --max 6G`, `JAX_PLATFORMS=cpu`.
- **Validation of the extraction [M]:** my per-lead RMSE equals the ALISIOS scorer's `by_lead.rmse` exactly (max |diff|
  0.0) for W, U10, V10, RAINNC, T2 on d02–d09 and all 73 leads. 18/18 re-hashed τ24 files match the admitted SHA-256.
  The critic reproduced this independently (876 checks, 1,752 arrays).
- Rings: ring 0 = WRF spec zone (spec_zone 1), rings 1–4 = relax zone (spec_bdy_width 5), interior ≥ 5.

## Q1 — timeline and hierarchy [M]

Tree: d01 9 km (dt 54) → d02 3 km (18) → d03…d09 1 km leaves (6), all children of d02; feedback 0 (one-way), so d01
and d02 do not see the leaves in either model.

- **d01**: small for 24 h (T2 ≤ 0.065 K, U10 ≤ 0.058, W ≤ 0.0056 m/s). The 9 km domain uses KF; no explicit convection.
- **Pre-convection error vs a released WN3 case — qualified comparison.** The full matched table (d01–d03 × W, U10,
  V10, RAINNC, T2, T × τ1–3 = 54 rows, `receipts/pre_convection_54rows_from_CR04.json`, source CR04) has
  N18/0420 ratios from 0.18 to 411 (median 1.12; 7 rows > 2, 6 rows < 0.5). Examples:

  | field/domain/lead | WN3 0420 B | N18 v0.4.2 | ratio |
  |---|---:|---:|---:|
  | d01 V10 τ1 (m/s) | 0.00322694 | 0.01937244 | 6.003 |
  | d02 W τ3 (m/s) | 0.00070155 | 0.00074535 | 1.062 |
  | d03 W τ3 (m/s) | 0.00982260 | 0.01783424 | 1.816 |
  | d03 U10 τ1 (m/s) | 0.00998872 | 0.00992501 | 0.994 |
  | d02 RAINNC τ1 (mm) | 9.29e-9 | 3.82e-6 | 411 |

  These errors are small relative to D6 and broadly comparable in many fields, **not equal**. The cases differ in
  weather, season, program version and start hour (baseline τ1 = 2026-04-21 01Z, start 00Z; N18 τ1 = 2026-10-18 19Z,
  start 18Z). No same-input 3-nest/9-nest comparison exists, so "no extra early error from the new fused program" is
  **not** established.
- **Association with local convective evolution** (first lead with W > 0.03 m/s | first lead with ≥ 1 mm/h hourly
  rain in either run): d07 6|3, d03 5|4, d08 5|4, d02 9|6, d04 9|9, d05 9|9, d09 9|10, d06 16|4. d06 reaches 3.79 mm/h
  at τ4 but W exceeds 0.03 only at τ16, when its hourly rain is 0.008 mm; heavier convection (≥ 5 mm/h) starts at τ18.
  So there is no single common onset threshold. The sibling contrast argues against one large simultaneous
  contamination event: at τ6 d03 hourly rain is 7.19 mm with W RMSE 0.0588 m/s, while almost-dry d04 (0.0017 mm)
  has 0.0024 m/s and fully dry d09 0.0074 m/s. These dry-leaf errors are small but are not demonstrated round-off
  bounds, and they are not an intervention on a sibling's input; they do not exclude cross-talk.
- **Interior larger than the nest boundary.** W at the d06 peak (τ24): ring0 0.098, relax 0.18, interior 0.68 m/s. The
  scorer's all-lead splits agree on every failing domain: frame (5 cells) < interior (d06 W 0.085 vs 0.24, RAINNC 9.4
  vs 26.3 mm). Ring-0 RAINNC is exactly 0 in both models (WRF skips microphysics in the spec zone; B40 is fixed in v0.4.2).
- **Shared parent/leaf pattern, no measured inherited fraction.** With the WRF mass-cell registration (child mass
  index i ↦ parent (i_parent_start−1) + i/3 − 1/3, `nesting/interp.py:25–37`; critic CR04b), the linearly
  interpolated parent error and the leaf error share a pattern but leave a large residual:

  | domain/lead, U10 | parent-interp RMS | leaf RMS | spatial corr | residual after fitted parent pattern |
  |---|---:|---:|---:|---:|
  | d04 τ59 | 1.583 | 2.280 | 0.613 | 1.775 |
  | d06 τ24 | 0.846 | 1.312 | 0.564 | 1.081 |
  | d04 τ10 | 0.463 | 0.690 | 0.725 | 0.476 |
  | d05 τ10 | 0.430 | 0.680 | 0.608 | 0.539 |

  The interpolation is a diagnostic, not a replay of the boundary forcing; RMS ratios are not an inherited
  fraction. (My first footprint table, `q1_parent_footprint.json`, used node-aligned slicing and is superseded.)
- **No systematic drift.** Per-lead biases are small next to RMSE, and their sign agrees in only 0.50–0.76 of the leads
  (e.g. d06 T2 max bias +0.20 K vs RMSE 0.50; d04 PSFC +13 vs 14.5 Pa at τ47). This does not exclude a regime-specific
  (active-convection) bias.

## Q2 — CPU restart era [M]

- Restart continuity: CPU a2 (uninterrupted, ran to τ40) vs a3 (restarted from the τ39 wrfrst) at τ40. On all
  9 domains **375/375 variables are byte-identical** (`Q2.json`; critic CR03 reproduced dtype, shape and bytes,
  0 nonfinite). GPU−CPU at τ40 is the same against a2 and a3.
- All 24 h FAIL frames (d02 τ17–24, d04/d05 τ10–24, d06 τ20–24) predate τ39. The 72 h extension failures are
  unaffected by the restart.
- d01 "+54 s" names: after the restart, CPU d01 history times are :18/:36/:54 (e.g. 12:00:54) instead of :00/:18/:36.
  3600 s is not a multiple of the 54 s d01 step, and the history alarm phase restarts at τ39. This is a CPU output-timing
  artefact (CPU-REF-ARTEFACT); it only causes the d01 72 h coverage UNRESOLVED and does not change the model state.

## Q3 — W/U10/V10 versus convective cells [M]

Mask: cells with an hourly RAINNC increment ≥ 1 mm/h in either run (this or the previous hour), dilated by r km.
The mask is a diagnostic of my choice; it does not change the binding full-domain D6 gate.

- **W is strongly localised at rain cells, but "outside W passes" depends on the radius** (`Q3_radius.json`; d04
  reproduced exactly by the critic):

  | domain | r = 0 km | 1 km | 2 km | 3 km | 5 km |
  |---|---:|---:|---:|---:|---:|
  | d04 max outside-W RMSE (m/s) | 0.301 | 0.225 | 0.191 | 0.180 | 0.168 |
  | d05 | 0.141 | 0.126 | 0.111 | 0.103 | 0.098 |
  | d06 | 0.239 | 0.226 | 0.205 | 0.170 | 0.146 |
  | d04 median W SSE / area fraction (frames W > 0.1) | 0.63 / 0.13 | 0.74 / 0.17 | 0.76 / 0.22 | 0.79 / 0.26 | 0.84 / 0.33 |
  | d06 median W SSE / area fraction | 0.87 / 0.34 | 0.90 / 0.40 | 0.92 / 0.46 | 0.93 / 0.51 | 0.97 / 0.61 |

  d04 τ59 (3 km): the mask holds 26.6 % of the area and 92.4 % of the W squared error; outside W 0.180. d06 τ24: 53.7 %
  of the area, 97.4 % of the W SSE (enrichment 1.81). d06 τ24 max |ΔW| 13.9 m/s, MAE 0.19 m/s. Localisation is not
  proof of unbiased dynamics: a convection-only defect would also look like this.
- **U10/V10 stay INCONCLUSIVE.** d04 τ59 U10 outside the 3 km mask 2.213 m/s > 1.5 (full domain 2.280); the mask holds
  only 30.9 % of the U10 squared error. d06 (τ20–35): 70–97 % inside, but the mask covers 30–60 % of this 54 × 57 km
  leaf; outside U10 0.4–1.2, V10 0.5–1.4 m/s. Max outside-U10 at r = 0…5 km: d04 2.21–2.27, d06 1.40–1.15, d05 1.18–1.20.
  Whether CPU-WRF itself varies this much at these leads is [U]; the N18 twin would decide.
- **Terrain**: W squared error on steep cells (|∇h| > 0.1) reaches 30–50 % of the total on 9 % (d06) / 17 % (d04)
  of the area at some leads (orographic convection over the islands); at other leads the steep share is below its
  area share.

## Q4 — RAINNC [M]

- Domain means at 72 h, CPU/GPU (mm): d02 2.07/2.18 (1.055), d03 0.72/0.73, d04 8.12/9.23 (1.136),
  d05 2.24/2.32 (1.036), **d06 34.88/29.67 (0.850)**, d07 0.24/0.24, d08 0.07/0.06, d09 0.10/0.12. Mixed signs across
  domains do **not** exclude a regime-specific engine error.
- Heavy cells (≥ 10 mm), CPU/GPU. Pooled d02–d09 counts mix 3 km parent cells with the overlapping 1 km leaves:
  τ24 unweighted 1887/1776 (0.941); nominal cell-area weighted 4767/4888 km² (1.025, still overlapping); **leaf-only
  1527/1387 (0.908)**. τ48 leaf-only 2878/2805 (0.975), τ72 3233/3281 (1.015). **d06 alone: τ24 528/422 (0.799),
  τ48/72 1809/1725 (0.954)**. The pooled counts are **not** used to exclude #149 or a localised rain deficit; the exact
  0420 pattern need not repeat, and a non-repeat is weaker than exoneration.
- Intensity: maxima d02 221/179, d04 59/60, d06 241/332 mm. d06 p99 GPU/CPU 1.03–1.08. Max hourly rates comparable
  (d06 81/88 mm/h at τ24).
- Displacement (best integer rigid shift ≤ 10 cells, on the common support with 11 rings removed — d06 only 36.4 % of
  its area; descriptive only; nonwrapping recheck by the critic, CR05, reproduces all 20 results):
  d02/d03/d07/d08 0 km at τ12–72; **d04 2.236 (τ12), 1.414 (τ24–48), 3.606 km (τ72)**; **d05 5.0 km (τ12)**, 0 km later;
  d06 0/1.414/2.0/2.0/2.0 km (τ12/24/36/48/72); **d09 3.162 (τ12, τ24), 2.236 km (τ36)**, 0 later. Best-shift RMSE is
  0–25 % below the no-shift value (d05 τ12 55 %). This cannot characterise all-domain displacement.
- Graupel/snow: GRAUPELNC domain means ≤ 0.003 mm, SNOWNC 0.

## Q5 — integrity zeros [M]

Raw integrity verdicts stand. Observation: every flagged field is active at other leads in the GPU run.

| domain/field/lead | CPU nonzero voxels/cells | CPU max | localisation |
|---|---:|---:|---|
| d06 QRAIN τ12 | 8 | 6.26677e−13 kg/kg | all ring 0 |
| d06 QNRAIN τ12 | 8 | 0.00198536 #/kg | all ring 0 |
| d04 QGRAUP τ36 | 8 | 2.97275e−12 kg/kg | all ring 0 |
| d06 GRAUPELNC τ24 | 2 | 4.48392e−10 mm | both interior |
| d06 QRAINXY τ72 | 1 | 2.84208e−9 mm/s | interior |

- Mechanisms, by field: pristine `module_mp_thompson.F:183` R1 = 1e−12 kg/kg; :1878, :1900–1901 and :4040–4044 zero
  **qr and its associated nr when qr ≤ R1** (R1 is a mass threshold, not a number threshold; QNRAIN is in #/kg,
  Registry.EM_COMMON:534). `module_microphysics_driver.F:809–838` excludes the spec ring. Observation: all CPU nonzero
  cells of the flagged d06 QRAIN/QNRAIN (τ12) and d04 QGRAUP (τ36) frames lie on ring 0, i.e. trace values in the
  zone that microphysics skips and parent forcing fills; this localisation is consistent with parent forcing. Only the
  QRAIN maximum (6.27e−13 kg/kg) is below R1; the QGRAUP maximum (2.97e−12 kg/kg) is above R1, and QNRAIN (#/kg) is
  not comparable with R1. GRAUPELNC (accumulated mm) and QRAINXY (Noah rate, mm/s;
  `noahmp/noahmp_driver.py:417`, pristine `module_sf_noahmplsm.F:1547` QRAIN = QDRIPR + QTHROR) are separate mechanisms
  (trace interior precipitation in CPU only).
- GPU nonzero frames: d06 QRAIN/QNRAIN 65/73, GRAUPELNC 43, QRAINXY 55; d04 QGRAUP 23. Reversals (GPU > 0, CPU = 0):
  d04 QGRAUP τ30/31/39/65, d06 QRAIN/QNRAIN τ52, QRAINXY τ65 — the one-sided rule does not flag these.
- This excludes a permanently dead field/writer mapping, **not** every intermittent producer defect. Label:
  consistent with realization.

## Q6 — realization noise floor

- Existing CPU−CPU evidence (#149 C2, 0420, WN3 3-nest, decomposition 4×1 vs 2×2 only): d03 RAINNC RMSE
  1.20/2.23/3.31/4.20 mm at 17/24/48/72 h, above the 1 mm D6 limit. The GPU−CPU B arm on that day was
  1.15/2.13/3.16/4.00 mm. No CPU−CPU W/U10/V10 score exists (C2 was scored only for RAINNC; originals deleted).
- 0420 GPU−CPU d03 W/U10 peaked at 0.16/1.06 m/s; this case's leaves are more convective (d06 RAINNC 35 mm domain
  mean). Floor magnitudes do not transfer between days [I].
- **An N18 CPU twin would decide the A3 floor rule** for this case/domain/field/lead: the same 9-nest namelist and inputs
  with another decomposition, 72 h on ALISIOS cores. A comparable twin spread would not by itself prove every GPU
  operator correct. ALISIOS owns the decision and resources. Cost: section Q6b (pending its timing admission).
- Further discriminators proposed by the critic (not run here): fused-sibling independence (in progress, separate
  section) and same-input one-step pristine oracles on active d04/d06 convective columns (Thompson incl.
  sedimentation, MYNN/surface-layer cold-pool forcing, W damping / sixth-order diffusion), which need caller
  savepoints not present in hourly wrfout.

## Scheme set vs the v0.4.2 support matrix [M]

- Namelist on all 9 domains: mp 8, ra 4/4 (radt 30), sfclay 5, Noah-MP 4, MYNN 5, KF on d01 only (cudt 5), gwd on d01,
  diff_opt 1 / km_opt 4, diff_6th 2 (0.12), damp_opt 3, w_damping 1, hybrid 2, epssm 0.5, sst_update 1, slope_rad 1,
  topo_shading 1, icloud 1, feedback 0. a2/a3 differ only in restart/start/run_hours.
- Every scheme is in the default suite (support matrix: Thompson 8, MYNN 5, MYNN-SL 5, KF 1 "Default suite; Composite
  D6 PASS24h"; `sst_update/slope_rad/topo_shading` c60 PASS). No experimental or refused option.
- Geometry: 7 sibling 1 km leaves in one fused d02 program is new. The support matrix (line 146) lists the nine-domain
  grid as "inferred from inputs + code path, not run end to end"; this run is its first end-to-end GPU forecast. d01
  loaded the sealed WN3 d01 executable (key a1662307); the fused d02 key 23cabda7 was primed by ALISIOS.
- `ADMISSION`-listed `key_compare3/namelist.input` is a max_dom=3 key probe, not the run namelist; the run config is
  `warm72/proof/nested_pipeline_run.json` (9 domains).

## Limits

- One case, one GPU realization and one CPU realization. The overall cause is INCONCLUSIVE; realization is a hypothesis
  consistent with the anatomy, not a proof.
- The rain-cell mask (≥ 1 mm/h, r km) is my own diagnostic, not a gate; its outside-W result depends on r.
- No observation truth, no skill score. D6 unchanged.

## Q6b — N18 CPU twin cost (admission ADD2 `8b0499c2…`, receipts `receipts/R07_twin/`, `Q6b_twin_cost.json`)

Source [M]: rank-0 `rsl.error.0000` "Timing for main" of d01 (inclusive of all 8 nests, E30) for the CPU reference
arms a2 (τ0–40) and a3 (τ39–72), plus the run configs. Original CPU-WRF 4.7.1 dmpar `1d0f5779…`, 4 MPI ranks,
`nproc_x = nproc_y = 2`, nice 5, CPU-only OpenMPI env (E219 set, `CPU_ONLY_ENV.json`), memcg MemoryMax 24 GiB.

| placement (measured) | cores | s per sim-hour (d01 main) | 72 h wall | core-hours |
|---|---|---:|---:|---:|
| a2 τ0–35, ranks on 0,1,2,4 = **4 physical cores** | 4 | median **731** (p10 697, p90 784) | **14.6 h** | 58.5 |
| a2 τ37–39 + a3 τ39–72, ranks on 4,5,20,21 = **2 physical cores + their SMT siblings** (N/N+16) | 2 phys (4 HT) | median **964** (892–1066) | **19.3 h** | 38.6 phys |
| τ36 (night re-pin hour) | — | 4442 (outlier, not used) | — | — |

History writes add ≈ 5.2 s per sim-hour (separate "Timing for Writing", negligible).

Twin options [I]:
- **4 cores, 4×1 or 1×4** (a decomposition different from the 2×2 reference, like #149 C2 4×1): ≈ 2×2 time ±10 %
  (patch aspect changes halo volume) → **≈ 13–16 h wall, ≈ 53–64 core-h** on 4 physical cores. On 2 physical + SMT:
  ≈ 19 h.
- **8 cores, 4×2 or 2×4**: no 8-rank measurement exists for this grid. With strong-scaling efficiency 0.65–0.85 for
  4 → 8 ranks (assumed; the 1 km leaves 55–112 × 52–112 cells give 8-rank patches of ~14–28 × 26–56 cells, halo-heavy):
  **≈ 430–560 s per sim-hour → ≈ 8.6–11.2 h wall, ≈ 69–90 core-h**. This needs 8 *physical* cores (8 distinct N, no
  N/N+16 pairs); 8 ranks on 4 physical cores + SMT siblings are expected to give no gain over 4 ranks [I] (the a3
  numbers show SMT sharing costs ≈ 32 % vs physical cores at 4 ranks).
- Pin one rank per physical core (taskset/`--bind-to core` on distinct N; never N and N+16 together), CPU-only env
  as above, memcg ≥ 24 GiB as for the reference.
- The decomposition change is the twin's only difference (namelist `nproc_x/nproc_y`), as in #149 C2.

## Fused-sibling independence (critic discriminator; ALISIOS priority 1)

Question: in the fused one-way cascade (`domain_tree._build_fused_cascade_program`, the program that runs d02 + 7
leaves of the 9-nest), can one sibling's values change another sibling's result? With feedback 0 there is no WRF
dataflow between siblings, so any dependence is a structural defect.

**Phase 1 [M, CPU] — PASS on the dynamics-only fallback** (`analysis/sibling/sibling_test.py --testbed`, receipt
`analysis/sibling/receipts/phase1_testbed_log.json`):
- Product builder unchanged; parent 48×42×8 (3 km) + three **different** 1 km siblings (24×21, 18×15, 21×27 at
  different parent starts), flat neutral state with a 2 K warm bubble and 5/2 m/s flow; one parent step with
  3 child substeps (ordinary phase).
- Arms from the same fixed parent carry and child carries: A unperturbed; B = middle sibling perturbed (every float
  carry leaf ×(1+1e−2 N(0,1)), θ/u/v + N(0,1)); C = first sibling perturbed.
- Gate: parent and every unperturbed sibling, same position, B vs A and C vs A: **624/624 leaf checks byte-identical**.
  The perturbation propagated (38 leaves of the perturbed child differ).
- Isolated single-child program vs the fused target sibling: **104/104 leaves bitwise identical** (max |Δ| 0.0).
- Mutant (builder coupled so child k's forced θ gets 1e−3 × mean θ of child k−1): **killed, 75 gate failures**.
- Realism limit: no physics and no radiation (`run_physics=False` in the testbed), XLA:CPU only. The real-physics CPU
  arms could not run on the shared box: the XLA:CPU compile of the fused real program was OOM-killed at 20 GB (PROD
  d02 copies, n = 3 and n = 2) and at 12 GB (three real-structured PROD-d02 crops 60×60/54×48/57×63, legacy and
  release); receipts kept under `analysis/data/*OOM*` (untracked) and in MAILBOX.
- A CPU replay cannot exclude a GPU-only buffer/aliasing defect. Phase 2 tests the real program on the GPU.

**Phase 2 design**: real PROD d01 + 3 × d02 (44×117×267), release defaults (product
kernels), ordinary and radiation phases, same gate; launcher `analysis/sibling/gpu_phase2.sh` (sha `e9c200cf…`),
script sha `e2d3febb…`. CPU lower-only dry test: traces, 117 s, host RSS 7.06 GiB, StableHLO 1.27 M lines.

**Phase 2 run 1 (2026-10-21 07:32–07:52Z real, RTX 5090, 3 × d02): NO VERDICT, timed out (rc 124).** [M] Asserted
GPU (`cuda:0`), lock acquired at once, REM 1315 s → 3 copies, inner timeout 1183 s. Load 121.7 s, ORD_A at 570.6 s
(one 4 min 40 s XLA compile of `jit_fused_jit`), ORD_B at 983.0 s (412 s for one arm), then timeout before ORD_C;
the gate was never evaluated and no arm outputs were kept. Cause [M CPU]: a harness defect, not the product and not
stepping. The v1 perturbation rebuilt each perturbed leaf with `jnp.asarray(numpy)`, which yields an UNCOMMITTED
array, while the loaded carries are committed (`device_put` at init). 168 of 170 leaves of each perturbed child
changed only in committedness (shape, dtype, weak_type, sharding, treedef equal). JAX's jit cache key includes
committedness, so arms B and C each got a new cache entry and re-lowered + recompiled the whole fused cascade
(cached jaxpr, new backend compile); RAD would have reused the three entries. Toy repro: the same `fused_jit` called
with an uncommitted and then a committed array logs one trace and two backend compiles.
Fix (`sibling_test.py` sha `89530466…`): values-only perturbation (same draw formula per leaf; each new leaf placed
with the original leaf's sharding and committedness; weak-typed leaves left unperturbed, none in these carries); an
assertion that every perturbed child has identical leaf signatures before any arm; per-stamp counts of jaxpr traces
and `fused_jit` backend compiles in log.json; `--abort-on-recompile` stops with rc 3 at a second fused compile.
RNG stream (disclosure, critic cr-w43-148fid 18:59Z): the same commit also changed the call site from ONE shared
generator `default_rng(148)` consumed sequentially by the perturbed siblings (first sibling, then the middle one) to
a fresh `default_rng(148 + i)` per sibling. The first sibling (index 0, seed 148, first consumer in both versions)
gets the same draws as in run 1; the middle sibling (index 1) gets DIFFERENT draws (seed 149 instead of the
continuation of stream 148). The perturbation recipe and magnitude are unchanged; the gate does not depend on the
particular draws, so this is disclosed, not re-run.
CPU check on the real PROD release carries (3 × d02, `--reuse-check`, cheap proxy jit over the exact `fused_jit`
argument tuples incl. record windows and clock bases): fixed perturbation 6/6 arm calls (ORD+RAD × A/B/C) → 1 trace,
1 cache entry; v1 control → 3 cache entries (A, B, C). Run 2 launcher `analysis/sibling/gpu_phase2b.sh` (new file,
sha `b5be5dda…`), output `analysis/data/sibling_gpu3/`. Expected GPU time ≈ 11 min (load 2 + one trace/lower/compile
7.5 + six arms ≲ 1.5) [I from run 1], hard cap inside the granted window.

**Phase 2 run 2 (2026-10-21 15:20:14–15:29:28Z real, RTX 5090, 3 × d02): PASS.** [M] Launcher `gpu_phase2b.sh`
(`b5be5dda…`), script `89530466…`, asserted GPU (`cuda:0`, platform gpu), lock acquired at once, REM 3246 s →
3 copies; rc 0; receipt `analysis/sibling/receipts/phase2_gpu_run2_log.json`.
- One compile for all six arms: `fused_jit` traced once and backend-compiled once (366.4 s), no further trace or
  compile from ORD_A to the end; the perturbed children had identical leaf signatures to the loaded carries
  (170/170 leaves, 0 differing in shape/dtype/weak_type/committed/sharding).
- Timings: load 116.3 s, ORD_A at 535.1 s (incl. the compile), each further arm 1–2 s, end 551.3 s; GPU lock
  held 9 min 14 s (estimate ≈ 11). Host peak RSS 12.81 GiB.
- **ORD** (parent step 2, child steps 4–6, no radiation): parent + unperturbed siblings, B vs A and C vs A,
  **1046/1046 leaf checks byte-identical**; the perturbation propagated (112 leaves of the perturbed child differ).
- **RAD** (parent step 34, child steps 100–102, radiation in both the parent step and the child steps; 8 radiation
  diagnostic leaves of child 0 change): **1046/1046 byte-identical**; 110 perturbed-child leaves differ.
- Sanity metric `child0_step_maxabs_change` ≈ 1.0e9 in both phases: max |Δ| over all float leaves of child 0
  across the step, dominated by one large-valued leaf (which leaf is not recorded [U]); it is not part of the gate.

Limits: three siblings with the SAME geometry (PROD d02 copies) and one parent step per phase; different sibling
geometries were covered only in the CPU dynamics-only testbed; no isolated-program or mutant arm on the GPU (both
validated in phase 1 on the same script file); a single process and XLA build. Within these limits the fused
cascade shows no sibling cross-talk on the real release program, ordinary and radiation steps.
