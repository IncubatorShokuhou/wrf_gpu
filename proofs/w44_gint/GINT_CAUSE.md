# G-INT trace-zero: cause per item class (lane w44-gint, rev 4: rev 2 after critic cr-w44-gint FIX F1–F4 + GI14 + GI15)

Question (ALISIOS, v0.4.3 teacher qualification): in 4/8 cases the frozen G-INT flags fields where the GPU is EXACTLY 0.0 and
CPU-WRF has a trace or light value (`proofs/v043/w43-teacher/GINT_ITEMS.md`: 19 REAL FAIL items, 10 annex-explained). Prior
lane w43-conv (`lane/w43-conv:analysis/TRACE_ZERO.md`) refuted an extra final Thompson mass floor; cause stayed OPEN.
The existing 4/8 G-INT FAIL verdicts and the frozen rules stay unchanged; nothing here is a NOT_A_BUG closure.

Labels: [M] measured, [I] inferred, [U] unknown. Port source = the v0.4.2 bundle the runs used
(`<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2/src/src/gpuwrf`, = v0.4.3 programs); truth = pristine WRF 4.7.1 + original CPU-WRF.

## 0. Result

1. **CPU-WRF itself exhibits the G-INT trace-zero signature** [M]. The frozen all-frame G-INT of a CPU-WRF twin (same
   executable and inputs, only the MPI decomposition changed) against its own reference gives registry FAIL-new-class with
   8 NEW items on 0227 (the GPU: 4 on the same case), 6 of them on the same domain/field/frame as GPU-degenerate frames (GI10).
2. **8 REAL items occur identically CPU-vs-CPU** [M, direct frozen integrity / raw values]: an additional CPU-WRF
   configuration of the same case has the field exactly zero at the item frame — **all 6 items of 0502 d01** (the 4×1
   configuration has QICE/QNICE @05-03_07, GI12; the original-decomposition 2×2 configuration has QGRAUP @05-04_11, @05-04_12,
   @05-05_04 and QRAINXY @05-05_08, GI14; same flagged columns and CPU maxima as the GPU items) and 0218 d01 QRAIN and QNRAIN
   @02-19_06 (GI13). 2 more REAL items (0302 d02 CANLIQ/CANWAT @03-04_15) have twin statistics consistent with zero; **exact
   zero is unverified** (critic F2 counterexample; twin raw deleted).
3. **No additional zeroing predicate found** [M, source] in the audited paths: root lateral boundary handling of the
   hydrometeors, the default Thompson kernel's exit/activity predicates, the Noah-MP canopy/ground-rain chain (GI02).
4. **The causes of the other 9 REAL items are OPEN.** Observations: each examined item is the first/last active hour of an
   intermittent event or a single-hour blip (GI03/GI08/GI13), and for 4 of them (0218 d01 RAINNC, 0302 d02 canopy@03-04_14)
   the own-case CPU run(s) keep the event. Two hypotheses are not separated by the CPU evidence:
   - **H1 trajectory variability**: CPU-WRF configurations of one case differ among themselves by as much as the GPU differs
     from CPU-WRF (0502 d01: the 2×2 configuration's T RMSE at +36 h 1.64e-2 K vs the GPU's 1.82e-2 K, the 4×1 configuration
     7.1e-3 K; GI12/GI14), and such differences move event edges by 1–2 h;
   - **H2 the known v0.4.2 root boundary-clock defect** (lane o5-coast, fix candidate `lane/o5-coast d79dd91c7a`, coupled gate
     pending): `operational_mode.py:3281` pins root ring 0 at `lead_stage = lead_seconds + dt_rk` while `:10913` makes
     `lead_seconds` the step END, so ring-0 mass/θ/ph/tangential winds run one root step ahead of WRF (pristine
     `solve_em.F:378` advances `dtbc` before the solve); measured signature: d01 ring 1–9 MU lead +47…+61 s on WN3 0227.
     It is present in the teacher programs and is an **unexcluded contributor** to the OPEN items, especially the lateral-zone
     0218 d01 west relax-zone first rain. No claim is made that it causes those items. (The 0502 d01 lateral-zone items do not
     need it: CPU-WRF configurations without the defect produce them identically, GI12/GI14.)
5. **Proposed GPU experiment (not requested by this lane; the manager decides):** run cases with OPEN items with the
   o5-coast walk clock and with clock 0 (o5-coast's immutable build; arms differ only by `GPUWRF_ROOT_SPEC_STAGE_CLOCK`;
   clock 0 is identity-exact with the released program on Swiss/PROD per o5-coast), plus one 1-ulp member of clock 0, and score each with the frozen G-INT.
   - **Interpretation (critic N1): outcomes are evidence to weigh, not causal decisions.** An item that disappears under the
     walk is H2-consistent but the walk also changes a chaotic trajectory (H1 also predicts some items to move); an item that
     flips between clock-0 members shows sensitivity and does not exclude a coexisting operator defect; an item stable in a
     few members does not prove a systematic defect. Results are weighed together with the known clock correction.
   - **Coverage (critic N2):** 0218 18 h + 0302 39 h (≈ 40 GPU-min appended to o5-coast's walk window) cover **5 of the 9 OPEN
     REAL items** (0218 d01 RAINNC@+6 h, d03 QCLOUD@+16 h; 0302 d02 CANLIQ/CANWAT/FWET@03-04_14). Not covered, they stay OPEN:
     0502 d03 RAINNC@+31 h, 0220 d03 QCLOUD@+46 h, 0218 d02 QRAINXY@+63 h, 0218 d03 QRAINXY@+58 h (they would need
     0502 ≥ 33 h, 0220 ≥ 48 h and 0218 ≥ 66 h members; FULL history ≈ 0.28 GB per sim-hour per member, so a DISK-PLAN first).
   - **References:** the ERA5 0218 and 0302 CPU references were deleted; their PRECONDITIONS keep all 219 frame hashes. A rerun
     in the original configuration regenerates the 0218 reference byte for byte (GI15); 0302 is expected to behave the same
     [I]. Regeneration costs ≈ 1.7 h (0218 18 h) and ≈ 3.6 h (0302 39 h) on 4 uncontended cores [I].

## 1. What the G-INT rule measures

The frozen rule (`wn3_score.output_integrity`, all-frames copy) flags a field/frame when the candidate field is CONSTANT over the
whole domain (e.g. all 0) while the reference varies; it is one-directional (CPU 0 / GPU > 0 is invisible). The annex explains
an item only if the CPU amount at the flagged columns is below a per-field cap (e.g. QICE/QGRAUP 1e-10, QRAINXY 1e-7) at ≤ 200
columns. For an intermittent field the domain-wide activity is a few cells for an hour or two around onset/decay, so any
trajectory difference that moves an onset or decay by one history interval can produce a flagged frame with a CPU amount that
is not small.

## 2. Evidence

| id | evidence | result | lbl |
|---|---|---|---|
| GI01 | CPU-WRF twin0227 (same exe 1d0f5779, same inputs, 4×1 vs 2×2) vs its reference, frozen scorer, 3 frames | twin d03 QGRAUP@2026-03-03_00 exactly 0.0 where the reference has max 6.57752e-12 — the same item the GPU has | M |
| GI02a | source audit root d01 lateral BC of QC/QR/QI/QS/QG/Ni/Nr (WRF solve_em/module_bc/microphysics_driver vs port) | no stricter rule: flow_dep_bdy after every RK stage with stage-averaged ru_m/rv_m (port `operational_mode.py:3680`), MP result discarded in the spec zone (`:8756`, = WRF its=ids+spec_zone), hydrometeors never relaxed toward wrfbdy (QV only), no edge clamp. **Does not cover the ring-0 mass/θ/ph clock (H2), which is a separate, known defect** | M (source) |
| GI02b | default Thompson kernel (NATIVE_REAL + FULL_COLUMN + EARLY_EXIT + MIXED_PHASE_WRF) exit/activity predicates vs mp_thompson | none stricter than WRF; condensation (`_condensation_branch`, = WRF :3401) never gated; nucleation activity 0.2 / 0 (WRF 0.25 / eps, looser); autoconversion 0.01e-3 identical; final floor R1 = WRF | M (source) |
| GI02c | Noah-MP canopy/ground rain (PRECIP_HEAT, CANWATER, QRAINXY) | identical: `CANLIQ<=1e-6 -> 0`, `CANICE<=1e-6 -> 0` (WRF noahmplsm.F:6346/6355 = port water_hydro.py:186/192); FWET 0 iff canopy 0; QRAINXY = RAIN − QINTR > 0 iff rain in the previous step. Does not validate the upstream rain trajectory | M (source) |
| GI03 | GPU side from the retained teacher extracts (full d03 LWP/RAINNC per lead, both sides) | 0218 d03 QCLOUD@+16 h: CPU 2-column cloud blip for one hour (LWP 6e-4 kg/m²), both runs cloud-free +10…+20 h otherwise, both cloudy from +21 h (5 vs 3 columns); 0220 d03 QCLOUD@+46 h: both clouds dissipate at +42 h, CPU re-forms 1 column at +46 h, GPU 1 column at +47 h, both ≈22 columns at +48 h; 0502 d03 RAINNC@+31 h: CPU first drizzle +31 h (2 cells, 1.86e-6 mm), GPU +32 h (1 cell, 2.7e-5 mm) | M |
| GI05 | one-sided whole-field-zero census on the retained extracts (d01/d02 CROPS + full d03; only LWP, IWP, RAINNC, rain increment; leads 1–72), 8 GPU-vs-CPU pairs + twin0227 vs its reference (`results/GI05_onset_census.json`) | twin0227: 5 flagged-direction frames vs the GPU's 3, all 3 GPU frames recur in the twin at the same lead/cells/max; pooled GPU 22 flagged-direction vs 49 mirror frames. **Diagnostic only**: 4 fields, cropped d01/d02, not the G-INT population; a global mirror surplus cannot exclude a field-local defect | M |
| GI06 | twin "all-zero" at GPU-degenerate frames INFERRED from retained difference statistics (mean(ref) = mae_GPU, mean(twin) = bias_twin + mean(ref)) (`results/GI06_twin_degenerate.json`) | inferred all-zero at 6/11 (0227) and 8/14 (0302) frames. **Inference only** (critic F2: a 1e-30 REAL32 value passes it). Calibration where direct evidence exists: on 0227 it agrees with the direct frozen integrity of GI10 in 11/11 testable frames (`results/GI06b_inference_vs_direct_0227.json`); the 0302 entries stay unverified | I |
| GI07 | frozen D6 per-lead difference stats 0502 d01 (GPU raw deleted) | QICE: GPU and CPU correlated r 0.94–0.98 at +3…+5 h, at +6 h r 0.0006 with \|bias\| = mae (GPU ice essentially gone), CPU ice gone by +8 h; QGRAUP: GPU all-zero at +35/+36 h, both have graupel +37…+39 h (r 0.04–0.16), GPU all-zero again at +52 h after both had graupel +49…+51 h | M |
| GI08 | reference-side anatomy of every 0502/0220 item ±3 h, WT2 CPU-WRF read IN PLACE through guarded sessions (A-O5-W44-GINT) (`results/GI08_maps_*`) | every flagged frame is the first or last active hour of the reference's own event: 0502 QICE/QNICE@05-03_07 = last hour of a decaying ice cloud (321 → 135 → 105 → 93 columns, then 8e-43, then 0), all 93 columns within 6 cells of the lateral edge (44 in ring 0), levels 16–31, ice supersaturation ≤ 0.19; QGRAUP@05-04_11 = reference onset hour (1 → 82 → 199 columns); QGRAUP@05-05_04, QRAINXY@05-05_08 = last hour; d03 RAINNC@05-04_07 = first hour; 0220 d03 QCLOUD@02-22_22 = first hour of a level-0 fog onset (1 → 3 → 23 columns); 0220 QRAINXY = last drizzle hours (1.6e-12) | M |
| GI09 | staging-free frozen G-INT: frozen tools run unchanged with `netCDF4.Dataset` replaced by a read cache; reference cached inside guarded sessions (`tools/gint_cached.py`) | outputs byte-identical to the plain frozen chain on a 9-frame test with 3 planted items (`results/GI09_equivalence.json`) | M |
| GI10 | frozen all-frame G-INT of twin0227 against its own reference, direct (219 reference frames read in place) (`results/GI10_twin0227_gint/`) | registry FAIL-new-class, 8 NEW items (d01 QNICE@03-01_13, d02 QGRAUP@03-02_21, d03 SNOW/SNOWC/SNOWH/TAUSS@03-02_21, d03 QGRAUP@03-02_23 and @03-03_00; all annex-explained), 13 degenerate frames; the GPU on the same case: 4 NEW items, 12 degenerate frames; 6 degenerate frames identical (d01 QNICE@03-01_13, d03 ECAN/EVC/QRAINXY@03-02_18, d03 QGRAUP@03-02_23, @03-03_00) | M |
| GI11 | root-only CPU-WRF configuration (max_dom 1, feedback 0, 4×1; same exe, E219 env) vs the 3-nest twin0227 d01 | NOT bit-identical (equal at t0, 106 fields differ from +1 h; Thompson tables byte-identical; cause U). Its departure has the size of the decomposition floor (T RMSE 2.4e-4 / 3.6e-3 / 4.3e-3 K vs twin-vs-ref 2.6e-4 / 3.6e-3 / 4.5e-3 K at +1/+3/+5 h). Root-only runs are therefore **additional d01 CPU-WRF configurations**, not a decomposition-only floor, and are not transferred to d02/d03 | M |
| GI12 | root-only CPU-WRF configuration 0502 d01 4×1 72 h vs the WT2 reference (read in place): frozen d01 G-INT, item series, per-lead stats (`results/GI12_*`) | REAL FAIL d01 QICE/QNICE@05-03_07 identical to the GPU items (same 93/101 flagged columns, same CPU max; direct values re-verified by the critic); nothing else flagged; it keeps the 4 graupel/QRAINXY events. Relative to THIS configuration the GPU is farther from the reference (X/C RMSE T 1.2–2.7, U 1.4–3.7, QVAPOR 1.1–4.0 at +3…+60 h), but the second configuration (GI14) is as far as the GPU | M |
| GI14 | second root-only CPU-WRF configuration 0502 d01 in the ORIGINAL 2×2 decomposition (72 of 73 frames; its timeout fired after the hibernation resume) vs the WT2 reference (read in place): frozen d01 G-INT + per-lead stats (`results/GI14_*`) | REAL FAIL QGRAUP@05-04_11 (1 column, 1.88e-6), @05-04_12 (82, 5.63e-6), @05-05_04 (34, 7.31e-10), QRAINXY@05-05_08 (2, 4.77e-7) = IDENTICAL to the GPU items, plus its own QNICE@05-05_18 (REAL) and QGRAUP@05-04_22 (explained); together with GI12 the two CPU configurations reproduce all 6 GPU d01 items frame-exact. This configuration is as far from the reference as the GPU (T RMSE +36 h 1.64e-2 vs GPU 1.82e-2 K) and its graupel is all-zero at +35/+36 h and uncorrelated at +37/+38 h (r 0.015/0.09; GPU 0.04–0.16) → the GI12 X/C 2–4 was a one-configuration artefact | M |
| GI13 | two root-only CPU-WRF configurations of 0218 d01, 18 h (2×2, 4×1; the 3-nest ERA5 reference itself was deleted, its item facts survive in the teacher receipts; the root-only 2×2 run matches only its t0 file hash; the 3-nest rerun of GI15 matches the reference trajectory) (`results/GI13_0218_d01/`) | at 02-19_06 the 2×2 configuration has QRAIN ≡ 0 and QNRAIN ≡ 0 like the GPU (reference 1.39e-27 / 8.79e-20); the 4×1 configuration has 1.4e-6 / 1.48; both have the 5-cell first RAINNC at the west lateral edge (5.7e-3 / 7.7e-3 mm vs reference 5.9e-3); the GPU rains 1 h later (r 0.77 at +7 h, 0.98 at +8 h) | M |
| GI15 | 0218 ERA5 3-nest CPU-WRF rerun in the ORIGINAL reference configuration (2×2, max_dom 3; only run_hours 72→18 and restart_interval 180→100000 differ; same exe 1d0f5779, same input hashes, E219 env, libcuda 0) vs the deleted teacher reference's PRECONDITIONS `cpu_sha` (`results/GI15_0218_3n_repro/`, `runs/e0218_3n_2x2/`) | **9/9 history files byte-identical** (d01/d02/d03 at +0/+1/+2 h): the deleted reference is reproducible bit for bit, and CPU-WRF 4.7.1 here is run-to-run deterministic in a fixed decomposition (this supports reading the GI10 twin differences as caused by the decomposition change, not by run-to-run nondeterminism; [I] for 0227). The planned 4×1 3-nest twin to +18 h (for 0218 d01 RAINNC@+6 h and d03 QCLOUD@+16 h) was **stopped by the producer at +2 h**: the shared lane cores were oversubscribed (26 → 44 min per sim-hour), so the item frames were out of reach of the declared deadline; those items stay OPEN | M |

CPU-run provenance (executable, consumed inputs/namelists, all namelist differences, rank checks): `runs/PROVENANCE.md`.

## 3. Per-item classification

Classes (causal explanations are hypotheses unless the class says verified):
- **CPU-exact (verified)**: a CPU-WRF run (decomposition twin or additional d01 configuration) has the field exactly zero at
  the same domain/field/frame, verified on raw values or the direct frozen integrity. The item does not need a port defect.
- **CPU-consistent (unverified)**: only the retained difference statistics are consistent with an all-zero twin (GI06).
- **OPEN-shift**: the GPU event edge is 1–2 h shifted; the own-case CPU run(s) keep the event. H1 or H2 (§0).
- **OPEN-onset**: first/last event hour with the GPU field active within ±1 h (observed); no own-case CPU run for that domain.
- **OPEN**: no GPU-side observation of the event, or no matched candidate event.

| case | item | G-INT | class | evidence |
|---|---|---|---|---|
| 0227 | d01 QNICE@03-01_13 | explained | CPU-exact (verified) | GI10 direct |
| 0227 | d03 CANICE@03-01_11 | explained | OPEN-shift | the twin is NOT degenerate at this frame (GI10 direct): it keeps canopy ice |
| 0227 | d03 QGRAUP@03-02_23 | explained | CPU-exact (verified) | GI10 direct |
| 0227 | d03 QGRAUP@03-03_00 | explained | CPU-exact (verified) | GI10 direct, GI01 |
| 0502 | d01 QICE@05-03_07 | REAL | **CPU-exact (verified)** | GI12: the d01 CPU configuration is exactly zero at this frame, same 93 flagged columns / CPU max 1.59e-8; GI08 last hour of a decaying edge ice cloud |
| 0502 | d01 QNICE@05-03_07 | REAL | **CPU-exact (verified)** | GI12: same 101 columns, 126 /kg |
| 0502 | d01 QGRAUP@05-04_11 | REAL | **CPU-exact (verified)** | GI14: the 2×2 CPU configuration is all-zero at this frame (same 1 flagged column, 1.88e-6); reference onset hour (GI08) |
| 0502 | d01 QGRAUP@05-04_12 | REAL | **CPU-exact (verified)** | GI14: same 82 columns, 5.63e-6 |
| 0502 | d01 QGRAUP@05-05_04 | REAL | **CPU-exact (verified)** | GI14: same 34 columns, 7.31e-10; last hour of a graupel tail (GI08) |
| 0502 | d01 QRAINXY@05-05_08 | REAL | **CPU-exact (verified)** | GI14: same 2 columns, 4.77e-7; last hour of a ground-rain event (GI08) |
| 0502 | d03 RAINNC@05-04_07 | REAL | OPEN-onset | GI03: CPU first drizzle +31 h, GPU +32 h |
| 0220 | d02 QRAINXY@02-21_13/14 | explained | OPEN | last drizzle hours 1.6e-12 (GI08); GPU side not observed |
| 0220 | d03 QCLOUD@02-22_22 | REAL | OPEN-onset | GI03/GI08: fog onset CPU +46 h, GPU +47 h |
| 0120 | d03 CANICE@01-23_19 | explained | OPEN | not examined beyond the annex |
| 0302 | d02 CANICE@03-03_11 | explained | CPU-consistent (unverified) | GI06 inferred |
| 0302 | d02 CANLIQ/CANWAT/FWET@03-04_14 | REAL | OPEN-shift (H1/H2) | own 3-nest twin keeps canopy water (GI06 inferred); GPU local 1-h rain 0.0012 vs CPU 0.055 mm (R4 watch) |
| 0302 | d02 CANLIQ/CANWAT@03-04_15 | REAL | CPU-consistent (unverified) | GI06 inferred twin ≡ 0; exact zero unverified (twin raw deleted) |
| 0302 | d02 FWET@03-04_15 | explained | CPU-consistent (unverified) | GI06 inferred |
| 0218 | d01 QRAIN@02-19_06, QNRAIN@02-19_06 | REAL | **CPU-exact (verified)** | GI13: the 2×2 d01 CPU configuration has QRAIN ≡ 0 / QNRAIN ≡ 0 (reference 1.39e-27 / 8.79e-20); the 4×1 configuration 1.4e-6 / 1.48 |
| 0218 | d01 RAINNC@02-19_06 | REAL | OPEN-shift (H1/H2; west relax zone) | GI13: 5-cell first rain at +6 h in both CPU configurations, GPU 1 h later; the 3-nest decomposition twin was stopped at +2 h (GI15), not decided |
| 0218 | d03 QCLOUD@02-19_16 | REAL | OPEN | CPU-only single-hour 2-column blip; no matched GPU event (next GPU cloud +21 h, GI03); 3-nest twin not reached (GI15), not decided |
| 0218 | d02 QRAINXY@02-21_15, d03 QRAINXY@02-21_10 | REAL | OPEN | QRAINXY is not in the extracts; GPU side not observed |
| 0218 | d03 QRAINXY@02-21_22 | explained | OPEN | not examined beyond the annex |

REAL items: CPU-exact (verified) 8 · CPU-consistent (unverified) 2 · OPEN-shift 4 · OPEN-onset 2 · OPEN 3 → 19.

## 4. Suggestions for ALISIOS (diagnostic; no gate change by wrf_gpu2; any acceptance rule must be preregistered)

1. **Publish a CPU floor item list.** Run the same frozen G-INT on every available CPU twin (twin vs its reference) with the
   direct integrity (as GI10) and list its items beside the GPU's; report a GPU item as "also CPU-vs-CPU" only on direct
   evidence (raw values or frozen integrity), never on difference statistics.
2. **Report the mirror direction** (candidate active where the reference is all-zero) beside the flagged one, per field and
   domain over the full G-INT field population, as a diagnostic.
3. **Onset/decay labels stay diagnostic.** If ALISIOS wants an acceptance path for event-edge items, it should require a
   matched candidate event (same field, overlapping columns/area, within a stated window) and be preregistered and calibrated
   against mutants that (a) erase a complete event, (b) shift every event by a systematic phase error, (c) zero a field; the
   rule must kill (a)–(c) and pass CPU-vs-CPU twins. (The rev-1 single-hour-blip clause is withdrawn: it admits an erased event.)
4. **Count rules need calibration.** A pooled flagged-count rule needs a matched field/domain/coverage denominator and an
   event-count tolerance derived from several CPU twins; the D6 RMSE factor 4/3 does not transfer to Bernoulli event counts.
5. Unchanged: the trace caps, the REAL default, and CPU-WRF as the only reference.

## 5. Limits

- 9 REAL items have OPEN causes (§0.4); the CPU evidence cannot separate H1 from H2 for them. The decider is the GPU walk / clock-0 /
  1-ulp A/B of §0.5 (tools prepared and CPU-dry-tested: `gpu/gpu_members.sh`, `tools/perturb_t.py`; not run by this lane).
- A CPU port A/B of the clock fix on WN3 d01 is not practical here: o5-coast's CPU-JAX legacy probes hit the 12 GiB RAM cap
  after the first output even on Swiss d01.
- d02/d03 items have no own-case CPU run beyond twin0227/0302: a 3-nest run costs ≈ 6.7 h per 72 h on 4 free cores, but on
  2026-10-24 the shared lane cores ran 26–44 min per sim-hour (GI15), so the 0218 3-nest twin pair (2 × 18 h) was stopped at
  +2 h. Finishing it needs ≈ 3.5 h on 4 uncontended cores (2 × 18 h at 6.7 h per 72 h) [I]; it would show whether a CPU
  decomposition twin in the true configuration reproduces 0218 d01 RAINNC and d03 QCLOUD.
- The GPU raw wrfouts of the teacher runs were deleted before this lane started; GPU-side facts come only from retained
  receipts (frozen integrity, D6 per-lead stats, extracts).
- Reads of ALISIOS data: admission A-O5-W44-GINT, in place through the sealed guarded facade (receipts `receipts/`); only the
  CPU-WRF run inputs (0.23 GB) were staged.
