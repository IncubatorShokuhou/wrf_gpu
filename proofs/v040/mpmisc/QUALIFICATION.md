# s2-mpmisc qualification evidence

The complete lane gate is met for Ferrier5/15, Goddard7 and Thompson38.
All four column ports are CPU-oracle-qualified under the standing
compiler-deviation policy and admitted experimentally. Real-input execution,
required shared consumers, baked-input keys, archive and default identities pass.
All measurements below are CPU-only. No GPU execution, GPU performance,
GPU E41/VRAM or forecast D6 qualification is claimed.

Reviewer corrections are recorded in thompson38/REVIEW_FIXES.md and
review_fixes_cpu.json. Commit4da8ed1fb removes the mp38 inherited cloud-DZ
floor and retains raw vapor through every cumulative REAL reconstruction.
MP60 passes72 affected checks; MP62 repeats all original Thompson literal,
compiled and evolving caller gates. All original values remain BYTE and
the115 disclosed compiled excesses are unchanged. The previous source seal
below describes the pre-review candidate; final corrected source is4da8ed1fb.
MP63/64 repeat Swissf672a573/PROD6fc715a2 exactly; MP65 passes all12 final
source/key regressions. MP66 repeats real Swiss two-step execution with zero
nonfinite/guards/pre-mask events and zero guard water changes. Its CPU
compile+first293.815 s/next3.553 s and peak18119564 KiB are disclosed alongside
the earlier smoke in REVIEW_FIXES.md, without a causal speed comparison.
MP67 checks the corrected clean archivee396b760e:92 assets,
738789201 compressed bytes, all original asset/CRC/finite/packaging gates
pass4.999 s. Later handoff edits change only Markdown.

| Option | Independent pristine-WRF column evidence [M] | Real Swiss CPU execution smoke [M] |
|---|---|---|
| Ferrier5 | 36 literal REAL/no-FMA columns BYTE; active dt/GID/multistep/rime stress; independently derived compiler bounds with disclosed small excesses | MP41: two official own steps, nonfinite0, guards0, load63.237 s, cold192.165 s, next0.573 s; peak7842344 KiB |
| Ferrier15 | Same 36-column suite includes distinct advected-rime option and fresh stress | MP44: two official own steps, nonfinite0, guards0, load62.247 s, cold197.128 s, next0.423 s; peak10303628 KiB |
| Goddard7 | 18 columns x36 original public groups literal BYTE; 5760 compiled prognostics within independent 14-input-ULP/FMA envelopes; diagnostic excesses disclosed | MP45: two official own steps, nonfinite0, guards0, load63.096 s, cold179.628 s, next0.929 s; peak9650668 KiB |
| Thompson38 | Eleven original hail+aerosol columns, 6160 physical/77 surface values BYTE; evolving three own-steps1680/21 BYTE; direct public theta440/440 and evolving theta120/120 BYTE; 33/33 process census. Independent compiled threshold bounds pass;115/6237 ordinary/fresh values exceed strict intervals, disclosed. Cold profiles/emission243/243 BYTE | MP56: two official own steps, nonfinite0, guards0, pre-mask events0, load44.417 s, cold233.826 s, next2.347 s; peak18128108 KiB |

The smoke harness changes only mp_physics in authentic Swiss RD11 inputs,
42x42x44, dt18, MYNN5/SF5/Noah-MP4/RRTMG4/4/CU0. It executes the official
CPU `_advance_chunk`, normal JIT, FAST0 and census enabled, with no dispatch
patch. These short runs establish execution, not CPU-WRF forecast agreement
or a GPU speed estimate. MP41/44 report tiny post-RK qc/qi negatives (34 each,
first -7.08e-24/-8.69e-22 kg/kg); MP45 reports113 qi negatives (first
-2.39e-12 kg/kg). No guard repairs or guard water changes occur. All pre-mask
events remain disclosed in receipts; full coupled D6 is pending.
MP56 executes the actual Thompson38 adapter on the final composed source and
has no pre-mask events or guard water changes. FAST0 retains the existing
wide CPU prognostic carry: qvolg/nwfa/nifa are float64, while the five held
surface fields are float32. This smoke does not prove whole-model fp32 carry;
the independent column ABI proof covers its nineteen REAL32 inputs/outputs.

The Goddard caller retains original PHYS_TOT history using only one REAL4
sum output, rather than six intermediate rate arrays. An independent tiny
Fortran program executes the verbatim original microphysics-driver statement:
4320 qualified input operands and720 totals are BYTE. Previous native outputs
retain exact equation ASTs; the new38-output ABI lowers to CUDA/sm120 IR in
0.351 s, without GPU execution. This adds4 B/cell and no column-kernel launch
or scratch. Serial level-point reductions in Goddard still have quadratic
level-count arithmetic; GPU efficiency is unknown. Radiation5/5 with mp7 is
refused by the separate radiation admission; the measured coupled pair is4/4.

Restart gates refuse missing hail state, active Ferrier/Goddard carries and
malformed shape/storage. Thompson38 preparation holds QNWFA2D/QNIFA2D,
instantaneous SR and original RAINNC/SNOWNC (20 B/surface), and emits after
column physics as pristine mp_gt_driver does. Original totals retain the
REAL addition order starting with the previous total; component buckets
remain disjoint. MP48's adversarial rounding test rejects pre-summing or
DOUBLE accumulation. Full/subset history uses the original totals directly.
Its original host initialization costs0.148 s at42x42x40 with no timestep
launch/transfer. Independent CCN/IN presence checks preserve supplied
positive aerosols and emissions. All three restart formats preserve these
surface fields BYTE. Nonzero unimplemented black-carbon inputs are refused.
The scoped option is CPU-oracle-qualified; GPU execution and forecast D6 remain
pending. MP41/44/45 execution receipts precede the approved glacier composition;
their physics kernels remain byte-identical. The final mp38 smoke includes it.

Final shared evidence: MP53 ran the eight required consumer files,
tests/contracts/, tests/test_v018_mp_family_fail_closed.py, the NetCDF lock
gate and owned carry/caller/restart/history/EPS checks:404 passed in666.43 s,
with two stale test pins failing. Test-only commit8c0575488 replaces the old
unsupported5 fixture with95 and classifies historical mp38 as an oracle
reference. MP54 reran both affected files and all contracts:59 passed1.48 s.
Production source is byte-identical to tested aca64db61 through839fa50a8;
the404 unaffected checks are carried forward. MP48b also passed all88 checks
including adversarial original REAL accumulation and all three restart formats.

MP50 verifies keyed providers for actual Ferrier tables/constants, Goddard
constants/tables and all Thompson38 NPZ/static/private bound inputs:10 checks
pass27.04 s, including nine genuine data-only mutations with unchanged
function identity but different full keys, HLO and compiled scalar outputs.
MP51 measures CPU key metadata cost:38 first16.298 s, repeat1.014 s, loaded
baseline8 0.163 s; peak2441088 KiB. This is outside the timestep. GPU readiness
impact is unknown. Thompson38 adds929643480 bytes of source-faithful table
data and20 B/surface held carry; legacy table duplication remains possible.
Its measured CPU temporary-buffer count excludes those constants. Goddard
quadratic level arithmetic and actual GPU memory/launch costs remain open.

The final source is rebased onto main6877bbb31, including the approved glacier
fix0821ebc0a and AOT-key mechanism76a72bdd1. MP57/58 use the official
proofs/s1_port/release_identity.py with only an owned temporary-directory
wrapper, on immutable8c0575488. Both full native-facade identities are exact:
Swiss f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d
(39.7 s CPU load+trace) and PROD d01
6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b
(64.3 s). These are trace identities, not executable-readiness measurements.
The historical MP39 Swiss2bf12cb8 predates the glacier fix. PROD WN2 inputs
are used only for code identity, without a forecast-fidelity claim.

Ferrier details: COMPILER_DEVIATIONS.md. Goddard details:
goddard4ice/COMPILED_DEVIATIONS.md, CALLER_AUDIT.md and
PHYS_TOT_SOURCE_PROOF.json. Thompson evidence: thompson38/NOTES.md,
SOURCE_AUDIT.md, COMPILED_DEVIATIONS.md and original_theta_byte.json plus
original_multistep_theta_source.json. Small compiled excesses remain disclosed
under the standing policy; strict interval agreement is not claimed for them.
Artifacts: <USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/MP41/receipt_mp5.json,
MP44/receipt_mp15.json, MP45/receipt_mp7.json, MP56/receipt_mp38.json,
MP57_swiss_identity.json, MP58_prod_identity.json and MP55_archive.json.
MP55 streams immutable839fa50a8:91 source/reference assets,
738779074 compressed bytes, all NPZ CRC/dtype/finite, strict JSON,
required asset and package-data checks pass4.225 s. No archive was extracted.
Disposable input views, snapshots, original table builds and caches were
removed after their last consumers. The default path allocates no new arrays.
All experiments are logged in MAIN. Independent critic and manager merge
are the next step; GPU execution, E41, VRAM, roofline and forecast D6 remain
pending owner qualification.
