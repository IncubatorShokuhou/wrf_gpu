# w43-teacher — PRE-REGISTERED teacher-qualification rule v2 (3-nest 9/3/1 km, 72 h)

Chronology (actual; v3 = v2 + critic delta fix 5 wording only, see §5 N72): v1 committed 72aa2f5b4e 2026-10-20T22:08:16Z, manager-signed with 3 notes 22:08:49Z, critic REFUSED
(cr-w43-teacher PREREG_REVIEW_72aa2f5b4e.md, 5 fixes). v2 = all 5 fixes + the manager notes; written 2026-10-20 ~22:3xZ (see
commit time). No new GPU, CPU-twin or ERA5 value has been opened. Known in advance and disclosed: the v0.4.2 RW3 72 h scoring of
WN3 0227 (bundle N72 vs CPU-WRF: d03 breaches RAINNC τ69–72 + W τ69, FLOOR-LIMITED under frozen A3 with ±1-ulp C rows; integrity
raw FAIL d03 QGRAUP t72 = registered exception). Nothing below is tuned to those numbers.

## 1. Reference, tools (frozen, pinned by sha256)
- Truth = original CPU-WRF 4.7.1 dmpar (gen2 build, wrf sha 1d0f5779…), same inputs + namelist as the GPU arm. GPU-vs-GPU never
  counts as fidelity.
- D6 scorer: `scripts/wn3_score.py --hours 72` (7a5b51d6…) → `wn3_fast_compare.py` (905fc13b…) → `compare_wrfout_grid.py`;
  manifest `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json` 9df4fab8…; d01–d03 sequential; every frame τ0–τ72.
- All-frame integrity (frozen G-INT): REGISTRY_ALLFRAMES.md 2a11c0cb…, integrity copy wn3_integrity.py 413299de… (+ its scorer
  module cbc95179…) with `--all-frames`, integ_classes.py d3cfb72e…, r4_eval.py 73deaf41…, classify_all.sh 738be63c…, annex
  post-pass annex_postpass.py 3b03e7cb… (G-INT (i)–(iv) + A1 + A2). Byte-exact copies recovered from git (commits 757a8cc7c4,
  e1f677468e, 7badd17017, ab6aa5b37d) into `analysis/frozen/`; executed unchanged, only I/O roots redirected (CPU dir = guarded
  stage copy, tool dir = analysis/frozen); a tool that cannot run unchanged on a case → that case-domain is INCOMPLETE (manager).
- Metrics per (case, domain d, field f, lead τ, manifest metric m): X = GPU−CPU value (scorer `field_summaries[f].by_lead`),
  L = D6 limit, C = the same metric for CPU-twin − CPU-reference from the SAME scorer with the twin as `--gpu-dir`.

## 2. Verdict per (case, domain, field) — the 62 fields with a D6 bound
- **strict24 (binding, no annex, per A3 "the 24 h D6 gate stays strict")**: X ≤ L for every τ0–τ24 (τ24 included) and m.
  Taken from the same 72 h per-lead rows (no extra run). Any strict24 breach = FAIL, whatever C is.
- **PASS**: strict24 holds and X ≤ L for every τ25–τ72 and m; finite pair fraction 1; 73 paired leads.
- **FLOOR-LIMITED**: strict24 holds, at least one τ25–τ72 breach X > L, and EVERY such breach has X ≤ k·C(d,f,τ,m), **k = 4/3**
  (= frozen A3 ratio clause C/X ≥ 0.75), C from the case's OWN valid CPU twin (§5). The frozen alternative "C > L ⇒ floor-limited
  for any X" is not used for the verdict (rule X ≤ max(L, k·C) is a strict subset of A3's floor); frozen-A3 reading disclosed beside.
- **FAIL**: any strict24 breach; any τ25–72 breach with X > max(L, k·C); any breach on a case without its own valid twin (no floor
  transfer between cases; A3 precedent 0408); any non-finite value, incomplete/inexact lead coverage, missing required field or
  incompatible metadata (never floor-limited).
- Nature of the result: a **deterministic qualification against ONE decomposition twin**, without statistical confidence or
  false-admission guarantees — an unusually dispersive twin can inflate C and admit a real GPU error, a quiet twin can reject a
  comparable realisation. k and the twin are fixed now; the twin is never reselected or rerun after X/C has been seen.

## 3. Integrity (binding) and unbounded fields
- Preconditions: every one of the 73 × 3 GPU and CPU frames present with exact names/Times, all values finite, field
  shapes/dtypes/metadata compatible. Not trusted from the scorer boolean alone.
- G-INT all frames: frozen tools of §1 on all 73 paired frames per domain; NEW items explained only by the annex with ALL its
  conditions: (i) field family {QICE,QSNOW,QGRAUP,QNICE,QNSNOW} ∪ {SNOW,SNOWC,SNOWH,TAUSS,QSNOWXY,ACSNOM} ∪
  {CANLIQ,CANWAT,CANICE,FWET,ECAN,EVC,QRAINXY}; (ii) GPU exactly 0.0 at every flagged cell; (iii) CPU max ≤ cap (mixing ratios
  1e-10 kg/kg, QNICE/QNSNOW 10 /kg, SNOW 1e-3 kg/m², SNOWH 1e-5 m, SNOWC 1e-3, QSNOWXY 1e-6, CANLIQ/CANWAT/CANICE 1e-4 mm, FWET
  0.05, EVC 0.01 W/m², ECAN 1e-6, QRAINXY 1e-7; A2: ACSNOM CPU per-frame increment ≤ 1e-3 kg/m² AND value ≤ 2.5e-2 kg/m²; TAUSS
  ≤ 5e-2); (iv) ≤ 200 flagged cells per flag-frame. Missing / incompatible / non-finite data or metadata is never annex-explained.
  Anything else = integrity FAIL. Also report the scorer's own `output_integrity`.
- The ≈313 fields without a D6 bound: integrity as above is binding; X/C at τ ≥ 6 (C > 0) reported, WATCH if X > k·C at ≥ 25 % of
  those leads (not a gate).

## 4. Cases (binding table) and deterministic ERA5 selection
| driving | case | CPU reference | CPU twin (C) | GPU arm |
|---|---|---|---|---|
| WN3 | 20260227_18z | WT2 wg_20260227_18z_a1 (nproc 2×2) | w43-twin0227 (4×1) | v0.4.2 bundle rerun in the GPU window (manager 22:08:49Z note 1, §5) |
| WN3 | 20260502_18z | WT2 wg_20260502_18z_a1 | none | v0.4.2 bundle |
| ERA5 | E-C | new CPU-WRF 72 h (2×2 or 3×1) | new CPU twin (different nproc) | v0.4.2 bundle |
| ERA5 | E-T | new CPU-WRF 72 h | none | v0.4.2 bundle |
- Regime metadata artifact = `rd-e2-massdays/TRANCHE1.json` as staged under the admission; its sha256 is recorded in
  `analysis/ERA5_SELECTION.json` together with the selected issue IDs BEFORE any forecast/twin value of any case is opened.
- Candidates = TRANCHE1 issues ordered by ascending issue UTC; complete = all 9 files of `era5_inputs/<issue>/` (namelist,
  INPUT_MANIFEST, wrfinput d01–d03, wrfbdy_d01, wrflowinp d01–d03) admitted and staged with sha256 equal to INPUT_MANIFEST where it
  lists one. E-C = first complete issue whose regime label is exactly the DANA-proxy class; E-T = first complete issue (≠ E-C)
  whose label is exactly the strong-trade class. Exact label strings are copied from the artifact into ERA5_SELECTION.json.
- If TRANCHE1 has no unambiguous regime field, no DANA-proxy or no strong-trade issue, or none complete: ERA5 coverage =
  **INCOMPLETE**; I post it and wait for a manager decision BEFORE any value. No arbitrary day is labelled convective. Disclosed:
  ALISIOS labels are proxies, not confirmed convection.
- Secondary (reported, delivered to ALISIOS for THEIR frozen twin rule 89f4b37c; not in the binding table): WN3 0614, 0220, 0608,
  0120 (WT2 CPU refs) on the same bundle, if GPU time allows after the binding arms.

## 5. Provenance bindings (any failure → that case has no verdict: INCOMPLETE)
- GPU: sealed v0.4.2 bundle <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2, bundle_sha256 bfa2dc90…, tuple commit b979188128e0 /
  src_tree b3e9b92a8c (BUNDLE.json); `sha256sum -c` of the bundle SHA256SUMS PASS before and after the session; consumer mode
  (P never written), source_head + device (asserted GPU) + env receipts; `compiled_new = 0` required for WN3 arms; for ERA5 arms
  compiled_new is recorded (> 0 = disclosed first-use compile of the same sealed source, not a different program claim).
- 0227 N72 (manager note 1): raw N72/A3 wrfout of RW3 were deleted 19.10 (only mtimes/rc, d6_N72 d01/d02/d03.json sha
  2c25a11f…/295da348…/3e9445f3… and RW3T_BYTES 219/219 survive; no per-file SHA list) → 0227 is RE-RUN from the sealed bundle.
  The retained d6_N72 numeric statistics are compared exactly as summary-statistics regression evidence only; old-vs-new output
  byte identity cannot be verified (UNKNOWN) because the old raw files/per-file hashes were not retained. The fresh sealed-bundle
  rerun supplies the raw qualification arm and is independently scored against CPU-WRF. Any statistics mismatch is reported as a
  FINDING and disclosed. The separate genuine RW1 bundle replay evidence (RW3T_BYTES, N72 == A3 219/219 at the time) keeps its
  original scope.
- Inputs: GPU and CPU runs of one case use SHA-identical wrfinput/wrfbdy/wrflowinp and namelist (CPU namelist may differ only in
  nproc_x/nproc_y/restart_interval; recorded diff); sha256 of the CPU build's run/ tables and Thompson .dat recorded.
- Outputs: per-file sha256 + Times for all 219 frames of every GPU, CPU-reference and twin run, recorded before scoring.
- Twin validity: rc 0 + SUCCESS COMPLETE WRF, identical executable sha, identical inputs/tables, namelist diff ⊆
  {nproc_x, nproc_y, restart_interval}, one continuous run (no restart/data substitution), exact τ0–72 file names and Times,
  all fields finite and compatible with the reference. An invalid twin supplies no C (case treated as no-twin).

## 6. Table and routing
- Case-domain TEACHER-OK iff every bounded field is PASS or FLOOR-LIMITED (strict24 included) and integrity PASS. Table: case ×
  domain × field class (T2/Q2 · 10 m wind · 3-D U/V/W · T/θ · PSFC/P/PH/MU · RAINNC/precip · hydrometeors · radiation/surface
  fluxes · land) → worst verdict in the class with X, L, C, X/C at the worst lead; τ blocks 0–23 / 24–47 / 48–72 separately.
- Every FAIL → w43-conv (convection/rain), w43-coastwind (10 m wind), w43-robust (runtime / HELD τ>40) or a new lane via the manager.
