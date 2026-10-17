# MERGE — Lin prerequisite through 1612acc49

Reviewed exactly `1612acc495b890832147622557f70edff2bf47ae`, source tree
`9bea5a29cb47e42873a4575d8dbefa0523e620ce`, against common base `6e8b447e2`.
All nine inherited commits and all 19 changed files are included. The Lin
temperature handoff is correct; no source fix is requested for this range.
Later t-c03rain changes are outside this verdict.

Independent evidence [M, CPU]:

- Targeted producer tests: **75 pass, 2 fail, 1 skip, 1 XPASS**, 38.85 s.
  Both failures are stale history-test `_Carry` mocks missing `nwp_diagnostics`.
  Independently rerunning both on immutable common base reproduces the same
  failures (1.96 s, zero setup errors). The affected runtime function and both
  test functions are AST-identical across base and candidate. No numerical
  failure is waived. The skip is an unavailable original surface wrapper;
  XPASS is a non-strict existing mark, not additional qualification.
- New independent real-solver probe: **120/120 field-column-step checks pass**,
  5.26 s, two consecutive actual production-dispatcher calls on six active
  40-level profiles. Original REAL4 Lin bodies are copied unchanged; literal
  moist prep/finish assignments are extracted from pristine WRF. Maximum
  theta error **6.103515625e-5 K**, vapor **2.60770320892334e-8 kg/kg**,
  total precipitation **2.877472301543449e-7 mm**, within the established Lin
  bands. All returned carry values are finite. Diagnostics/geometry/density
  are matched at the caller boundary: this isolates temperature wiring and
  does not certify their complete WRF parity or a coupled trajectory.
- Two independent **source deletion mutants**, one per conversion, each
  produce **one genuine numeric assertion failure and zero setup errors**.
  Removing prep gives dry-theta error 9.374603 K; removing finish gives
  theta error 9.334595 K against the existing 0.01 K band. Tests are unchanged
  from the successful independent probe.
- Rebuilt the terrain oracle privately: **all 12 original output dumps are
  byte exact** (six regimes at HT=0/1500 m); original bodies and caller hashes
  match the tracked manifest. The committed six-column terrain fixture is
  HT=1500 m; independent rebuilding verifies both heights.
- Full native-facade default jaxpr identities independently reproduced:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
  These are CPU trace identities, not GPU readiness measurements.

Inherited-commit audit (E51):

| Commit | Disposition of every inherited change |
|---|---|
| fc37e0126 | Lin optional HT API and operational HT forwarding; original terrain builder/fixture/manifest/comparison and rain-attribution tools; actual MYJSFC special-wind retention, optional history leaves, operational and writer plumbing; tests and notes. Reviewed; terrain source and numerical provenance reproduced. |
| 33af853d9 | Eta output-only barrier, HLO attribution tool, pickle/NetCDF special-wind restart tests. Barrier is removed in the final candidate; tool is read-only; restart tests independently pass. |
| 3276e1d1d | GPU launcher, frozen D6 scorer, notes. Reviewed official inherited-FD/backend/source/input admissions, timeout ownership, immutable reference seals, finite checks and unchanged frozen tolerance hash. No GPU run by critic. |
| 8575b3d94 | Evidence notes and API docstring/return annotation. No arithmetic change; recorded 343 shared passes/3 skips audited as producer evidence, not independently rerun. |
| 501e61d52 | Preserve optional MYJSFC leaves through Noah-MP reseeding and history writes; static-tree test. Reviewed and independently tested. |
| eced813d7 | None-aware seed test preserves seven default leaves. Reviewed and independently tested; no weakened numeric assertion. |
| 6c2f7c9e4 | Initial-history special winds and initial test. Superseded initialization audited with next commit. |
| 955b523cc | Initialize U10E/V10E from U10/V10, overriding input special winds exactly as pristine physics_init; replay proof tool and tests. No rotation: these are equivalent-roughness winds. Independent initialization/writer/restart tests pass. Replay tool preserves raw input and labels its scope. |
| 1612acc49 | Lin dry-theta prep and original moist finish; real caller test; remove ineffective Eta barrier. Independent original-body caller and separate deletion mutants pass the preregistered gates. |

Wiring: pristine microphysics_driver's LINSCHEME passes TH, dry-air mixing-ratio
QV, total pressure P in Pa, Exner PII, and HT. The candidate now passes dry TH
and restores theta_m using OLD dry theta/vapor and NEW dry theta/vapor through
the established WRF three-term finish. It preserves the existing fp64 dispatcher
island and geometry/density conventions; these are not newly qualified as WRF
REAL arithmetic. No new repair, clamp, persistent thermal field, or baked asset
is introduced. The independent test uses the actual production dispatcher;
the producer's prescribed-output seam test is also deletion-sensitive.

Performance/memory: default program identity supports unchanged default
arithmetic. On MP2, dry theta already feeds density, so compiler reuse is
possible [I]; moist finish adds elementwise arithmetic and live temporaries.
For SF2 with existing in-step history, two REAL surface fields add **14,112 B**
at 42x42 [I logical storage]; with history disabled, the new SF2 seeding also
allocates the other history fields. There are no new timestep host transfers
or synchronizations in the source. Actual launch/layout/temporary-memory and
warm speed deltas remain **[U]** under this CPU-only critic contract.

Producer evidence is source-bound and retained, not rerun by this critic:
the exact candidate's c03 24 h receipt reports **25/25 frames, D6 PASS, zero
nonfinite or bounded failures**, cold shared-CPU/I/O wall **887.759 s** and
sampled process VRAM **5862 MiB**. Its actual E41 remains **FAIL**: six STACK
flags (maximum 2456 B), including new Lin STACK392/kInput917 HLO. This is an
opt-in scheme disclosure under current wave policy; release-default E41 gates
remain binding. This verdict makes no S1, roofline, warm-speed or release claim.
Later SF4 work and final t-c03rain READY have separate review scope.

Artifacts under `<USER_HOME>/wrf_gpu2_lanes/rv-t-c03rain-lin/`: RV01 tests;
RV02 identities; RV03 pristine build metadata; RV04D actual caller and JSON;
RV05 terrain reproduction; RV06 ancestry/source audit; RV07 mutant XML/logs;
RV08 baseline reproduction; RV09 bound producer/baseline audit; CLEANUP.json.
Early RV04 A/B/C runs were invalid reviewer harness attempts (pytest import
path, then optional None accumulators); preserved and logged, never counted
as physical failures or mutant kills. No GPU was used. Private original builds,
snapshot, and pytest temporary data are removed after their last dependent arm.
