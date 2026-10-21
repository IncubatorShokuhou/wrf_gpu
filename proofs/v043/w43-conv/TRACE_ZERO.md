# w43-conv: G-INT "GPU exact 0 where CPU-WRF has trace" — no extra final MP floor found; cause OPEN

Manager request 2026-10-21T16:45Z. Items: 0502 d01 QICE/QNICE/QGRAUP (+QRAINXY, d03 RAINNC); 0220 d03 QCLOUD @+46 h;
era5_20240302 d02 CANLIQ/CANWAT/FWET after a light-rain difference. Hypothesis: a small-value zeroing threshold in our
Thompson/qc path that WRF does not apply. CPU-only analysis, release source b3e9b92a8c (= v0.4.2), no product edit.
GPU raw wrfouts of these three cases were already deleted by w43-teacher; the evidence uses CPU-WRF refs + teacher
extracts (|GPU−CPU| per lead) + the still-present GPU v0.4.2 output of era5_20240218.

## Status (after critic cr-w43-conv PARTIAL, 2026-10-21T17:36Z; manager line 17:46Z)

**No extra final MP floor found (matches pristine R1=1e-12), onset-compatible, cause OPEN, no engine bug localized.**
This is NOT a NOT_A_BUG closure of the items. Only the specific hypothesis "an additional final Thompson mass cutoff
removes traces above R1" is refuted [M]. The causes of the individual flagged frames (0502 boundary/edge ice and graupel,
0220 warm-cloud onset, ERA5 canopy after a local rain difference) remain OPEN: the evidence below is compatible with
onset/decay/placement timing but does not prove it. The PREREG G-INT FAIL verdicts stay FAIL; no annex or threshold change.
Critic verdict: <USER_HOME>/wrf_gpu2_lanes/cr-w43-conv/TRACE_ZERO_VERDICT.md (sha256 61c650bb…40fc).

## Findings

1. **Thompson floor is WRF's.** Port `_finish` (physics/thompson_column.py:999–1033) zeroes qc/qr/qi/qs/qg only at
   `<= R1 = 1e-12` — identical to pristine module_mp_thompson.F:4007–4060 (Ni/Nr zeroed only with their mass species).
   CPU peaks of the flagged frames: QICE 1.59e-8, QCLOUD 8.87e-8, QGRAUP 1.88e-6 / 5.63e-6 (05-04_11/12) and only
   **7.31e-10** for the late QGRAUP tail at 05-05_04 — the smallest is still ~731×R1, so this floor cannot remove any of
   them. Case namelists: mp_zero_out unset (=0, no WRF zeroing either). This checks the final zero predicates only, not
   sources/sinks, number balance or coupled-step operations.
2. **GPU keeps traces far below the flagged values** (era5_20240218 v0.4.2 GPU output, 18 frames d01–d03): thousands of
   cells per frame with QCLOUD/QICE/QRAIN/QSNOW in [1e-12, 1e-8] and values down to 1e-20…1e-45 (advected sub-R1 tails).
   (No QGRAUP in these 18 frames → no graupel-retention evidence.) Physical sinks and WRF-style inflow boundaries can
   still legitimately remove values above R1; the claim is limited to the final mass-floor predicates.
3. **Symmetry census (7 cases × 3 domains × 72 h, `analysis/trace_onset/`):** frames where the GPU field is identically 0
   but CPU is not (= what G-INT flags) vs frames where CPU is identically 0 but GPU is not (CPU 0 and |GPU−CPU| > 0 at that
   lead — the G-INT rule cannot see this direction):
   | family | both active | both 0 | GPU-only | CPU-only |
   |---|---|---|---|---|
   | hydrometeors Q*/QN* | 5659 | 5019 | **29** | **24** |
   | RAINNC (accumulated) | 1452 | 67 | 13 | 1 |
   | canopy/land (CAN*, FWET, Q*XY) | 4304 | 4825 | **35** | **34** |
   Above per-field magnitude floors: GPU-only 25, CPU-only 19. A blanket zeroing threshold would make GPU-only ≈ 0; balanced
   counts do NOT prove the absence of a bug (pooled correlated fields/cases/magnitudes; a conditional discrepancy can coexist
   with GPU-only frames elsewhere). Mirror cases
   exist and are larger than the flagged ones: 0227 d03 03-02_16 GPU canopy 5.8e-3 mm / FWET 0.38 with CPU exactly 0;
   0502 d01 05-05_08 GPU QICE 1.1e-7 / QNICE 99 / QSNOW 1.5e-7 with CPU 0 (mirror of 05-03_07); 0220 d03 QCLOUD GPU-only at
   +4 h (1.3e-6) and +13 h (4.6e-7) (mirror of +46 h); 0220 d03 RAINNC GPU starts earlier in 13 frames.
4. **Per item (CPU-WRF ref cells):**
   - 0220 d03 QCLOUD @02-22_22: ONE cell (k0, j57, i26), 8.87e-8; leads 42–45 both 0, from +47 h both cloudy → cloud onset ≤ 1 h apart.
   - 0502 d01 QICE/QNICE @05-03_07: decay tail of an ice cloud leaving the EAST edge (i 113–119, k 16–31, median edge
     distance 0, only 100 cells > 1e-10, max 1.59e-8; 05-03_05 max 2.5e-7, 06 5.5e-6) — GPU's ice left ≤ 1 h earlier.
   - 0502 d01 QGRAUP @05-04_11/12, 05-05_04: graupel onset at the NW corner (j 55–69, i 0–34, edge distance ≤ 10); GPU
     graupel present from 05-04_13 on (diff bias at 36–45 h of both signs). 05-05_04 = late CPU-only edge tail, peak 7.31e-10.
     Mechanism (upstream state, sources/sinks, transport, boundary coupling) OPEN.
   - 0502 d01 QRAINXY 2 cells, d03 RAINNC 2 cells 1.86e-6 mm: drizzle onset.
   - era5 d02 canopy: local 1 h rain CPU 0.055/0.117 vs GPU 0.0012/0.0054 mm at 4–6 cells (45×/22× lower), d02 domain-mean 72 h
     RAINNC bias −0.0004 mm (RMSE 0.083). A near-zero domain mean cannot separate displacement from a local amount error, and
     the MP floor does not cover canopy fields → cause OPEN. GPU canopy interception responds elsewhere: era5_20240218 GPU, vegetated land
     with 1 h rain 0.001–0.15 mm holds CANWAT > 0 in 67–93 % of cells (d02 159/227, d03 948/1225).
5. Boundary note: the 0502 d01 ice/graupel cells lie in the lateral zone; the port's root `flow_dep_bdy`
   (coupling/boundary_apply.py:539, have_bcs_moist=False path) matches share/module_bc.F (outflow interior copy, inflow 0).
   Residual differences: port sign test on instantaneous u/v at end of step vs WRF ru_m/rv_m per RK stage (B43 flag
   GPUWRF_ROOT_SCALAR_BDY_RK1) — only matters when the edge normal flow is near 0; not shown to cause these items [I].

## Census caveat (E200, critic controls)
`analysis/trace_onset/census.py` has no explicit finite checks. Exact-script synthetic controls (critic TZ_CENSUS_CONTROLS.json)
show silent misclassification: CPU-zero frame with NaN diff → both0, with ±Inf diff → GPU_ONLY, CPU NaN/Inf array → both.
The critic's independent finite/raw-pair validation shows this did not affect 0502 and 0220 (84/84 records exact); the other
five cases were not finite-validated from raw. Reuse the census only after adding explicit finite/raw-pair validation.

## Possible follow-ups (manager/teacher decision; no threshold change by me)
A symmetric degeneracy check (also report CPU-degenerate frames) would make both directions visible. Any annex/onset-class
treatment is the teacher's/manager's call; this report does not justify one. B60 (Thompson cold-path gaps,
e.g. png_scw subtracted from Ns, thompson_column.py:1731) is still open in v0.4.2 but cannot zero a field and is not linked here.

Limits: GPU raw of the three cases deleted → GPU-side magnitudes only as |diff| at CPU-zero frames (exact there), no GPU
cell maps for the flagged frames; onset timing differences themselves are free-run divergence, not explained per cell.
