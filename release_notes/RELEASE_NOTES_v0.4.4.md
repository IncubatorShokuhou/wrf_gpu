# v0.4.4 — initialization, terrain shading and regional tuning

**27 October 2026.** The 72-hour release gate is **PASS by the Owner-approved
manager decision**, with two disclosed exceptions: d03 rain at leads 67–72 is
CPU-twin-floor-limited, and the final graupel trace is the registered v0.4.2
exception class. Raw exceedances and the original raw FAIL report remain visible;
no threshold or frozen rule was changed.

[Support matrix](../docs/release/SUPPORT_MATRIX_v0.4.4.md) ·
[Known issues](../KNOWN_ISSUES.md) ·
[v0.4.3 notes](RELEASE_NOTES_v0.4.3.md)

## What is included

- **Terrain shading in nested forecasts.** Each nest keeps its own shadow height
  and follows WRF's separate shadow-update clock. The slope correction and terrain
  shadow have distinct roles. Cases already using both `slope_rad=1` and
  `topo_shading=1` intentionally change. The Swiss case also has other initialization
  and clock changes; shading being inactive is not a whole-program identity claim.
- **MYNN fresh-start state.** Retain the QSQ moisture-variance and EL mixing-length
  results from initialization instead of losing them before the first MYNN call.
  The selected original-WRF helper also corrects QKE's snow treatment. QKE, EL and
  QSQ are all part of this initialization change; it is not merely a copied
  diagnostic. Restart/later-step restoration keeps its existing authority.
- **Solar and ozone clock arithmetic.** Protect 14 quotient families (nine solar,
  five ozone) with round-to-nearest division and REAL32 rounding. Dynamic sites
  use the reviewed binary64 division island and round back to REAL32. This does
  not make every model division or transcendental expression bit-identical to WRF.
- **Parameter admission.** Diffusion/damping runtime overrides, including `c_s`,
  are refused after their GPU qualification failed. The nine new Noah-MP controls
  (`smcmax`, `dksat`, `bexp`, `psisat`, `hvt`, `hvb`, `z0sno`, `snow_emis`, `rsmin`)
  are also refused. Existing Track-P controls such as `z0mvt` are separate and remain.
- **Experimental MYNN coefficients.** `mynn_parameters` passes the original-WRF
  component and short two-domain GPU root gates: default bytes identical,
  changed values reuse the same selected programs with no new native blobs, and
  actual program resource checks pass. It is not qualified for three-nest D6,
  stations or 24–72-hour drift. The CLI help's pending-GPU wording refers to that
  full coupled qualification, not the already passed component check.
- **Reader registry and versioning.** The reader registry now covers the added
  host readers. Package, runtime-data, checkpoint and bundle identities change;
  the reviewed registry correction adds no numerical operators. The package,
  executable namespace and release metadata use version 0.4.4.
- **Regional experiment documentation.** The [regional guide](../docs/regional_optimization.md)
  links input overrides, surface controls, diagnostics, ensembles and checkpoint
  branches, with each tool's actual scope. Implemented experimental tools are
  not collectively upgraded to full forecast qualification by this release check.

Scope and original-WRF evidence: [terrain](../proofs/w44_topo/TOPO.md),
[MYNN initialization](../proofs/w44_qsqseed/QSQSEED.md),
[clock repair](../proofs/w44_e84fix/E84_FIX.md),
[final composition](../proofs/w44_rc/FINAL_GATE_TABLE.json),
[MYNN parameter scope](../proofs/w44_docs/receipts/DOC44_MYNN_SCOPE.json).

## What is not included

The constant-folding default flip, root-walk default flip and command-buffer
capture default flip are OUT. The retained root-walk behavior still has the
reported ring-timing lead; the proposed flip missed four frozen coastal-wind
checks and was not accepted. Its timing improvement is not a fidelity pass.

Diffusion/damping overrides and the nine new Noah-MP controls are not available
in v0.4.4. A later release would require successful qualification; there is no
date or commitment. Thompson's four admitted coefficients are microphysics,
not PBL mixing, and remain experimental at their component/short-root scope;
ice/rain fall-law and autoconversion extensions remain refused.

Deviceless compilation followed by GPU loading is unavailable. CPU variant-queue
planning and ordinary GPU priming remain supported. Internal low-level AOT
fingerprint opt-outs/legacy metadata handling are a separate hardening item;
the supported user-callers' strict-boundary audit is not a GPU-readiness proof.

## Program identities changed

The five native-first comparisons against v0.4.3 are DIFFERENT. These are
full-constant compiler-input SHA256 prefixes, not output checksums and not a
fidelity comparison. The final registry-only composition carries the exact
reviewed compiler inputs/native records. Full values and attribution are in
[the identity table](../proofs/w44_rc/IDENTITY_TABLE.json) and
[attribution](../proofs/w44_rc/IDENTITY_ATTRIBUTION.json).

| Program | v0.4.3 | v0.4.4 |
|---|---|---|
| Swiss Alps, d01 | `f8619ec597fb6afb` | `192c5436b10af1c9` |
| Canary Islands, d01 | `47c10960663133b0` | `10423ba3a36c6927` |
| Canary Islands, d02 | `5e5c3f0026db288d` | `f49f24f2d321477f` |
| Tenerife, d01 | `01afa1d54fdbb42c` | `160e5dd002e7d348` |
| Tenerife, fused d02/d03 | `15824b9c288577ba` | `437e6974cbc448c1` |

All existing native kernel bodies, launch settings and call/grid records match
under the measured native-body comparison. Terrain-enabled domains add the
shadow/solar kernels. Pure-XLA changes are attributed to the reviewed QSQ/EL
and solar/ozone source changes; that causal assignment is source-based, not a
separate ablation for every item. Source-location differences remain disclosed.
Do not carry v0.4.3's blanket byte-identical-program/output claim into v0.4.4.

## Release checks

The model source is `a22dea711202a48d00c86eafa34391ff2cb60fe2` on RC
`286e2442a9`. Documentation composition must leave that source and tests unchanged.

| Check | Result and scope |
|---|---|
| Final CPU suite | 4,209 passed, 34 skipped; zero unresolved source failures after classification |
| Release defaults | 102 passed, one expected failure |
| Reader-registry checks | 56 passed, including 22 additional package checks |
| Unavailable checks | Two old deviceless-only test files are absent on both baseline and candidate: not applicable, not passed |
| Compiler/native carry | Five canonical full-constant compiler inputs and native records exactly match the reviewed pre-registry composition; existing prescreens carry, not a new GPU measurement |
| RW1 warm cache | PASS for the recorded short replays: zero new compilation, byte-exact paired replay outputs |
| RW2 cross-date warm | PASS for the recorded additional dates/input-time variants on the shared grid/cache |
| RW4 bundle replay | PASS for the recorded three dates through the sealed bundle |
| RW3 fidelity/integrity | **PASS by approved decision**, with six floor-limited rain checks and the registered graupel-trace exception below |
| RW3 speed/memory | PASS from three same-session 24h pairs; O72 not run |

The two cache-test failures reproduced on the exact baseline with persistent
cache disabled and passed on both sources with the test cache enabled. A new
reader-registry coverage failure was fixed by the reviewed metadata-only change.
Test bodies and scientific tolerances were not weakened. See
[CPU classification](../proofs/w44_rc/FINAL_GATE_TABLE.json).

### RW3: 72 hours against original CPU-WRF

All 219 hourly frames (73 per domain) are paired with original CPU-WRF for the
27 February 2026 Tenerife three-nest case. Each domain has 374 comparable history
fields; only fields with frozen limits participate in bounded D6. No required
field/global or GPU nonfinite mismatch is reported.

| Domain | Frames | Binding scientific result | Output integrity |
|---|---:|---|---|
| d01 |73| PASS | PASS |
| d02 |73| PASS | PASS |
| d03 |73| PASS with six FLOOR-LIMITED rain checks | PASS with the registered final-frame QGRAUP exception |

**Rain disclosure.** d03 RAINNC raw RMSE exceeds 1.0 mm at leads 67–72. At67h,
X increased from 0.50055 mm in v0.4.3 (and v0.4.2) to 1.06006 mm in v0.4.4;
it was 0.75682 mm in v0.4.1 and 0.95209 mm in the earlier v0.3.3 comparison.
This is not an improvement in raw rain error.

The original CPU P0227-vs-R metric series retains C67=0.8905983272943488 mm.
Its provenance matches the same pair used for the frozen floor, with all six
overlapping frozen values reproduced byte-exact and independent confirmation.
This is archived-metadata recovery, not a new raw CPU-pair computation.
Under the unchanged rule `C > limit OR C/X >= 0.75`, C/X=0.840143 at 67h
qualifies that exceedance as FLOOR-LIMITED, alongside leads 68–72. X67 is larger
than C67; the result passes the declared floor rule, not the raw1.0 mm bound.
The approved binding result is 0 REAL FAIL and 6 FLOOR-LIMITED.

**Graupel disclosure.** At d03t72 the GPU QGRAUP is exactly zero and CPU-WRF's
maximum is 6.5775e-12 kg/kg. The exact frame/values match the registered
v0.4.2 exception class. The Owner-approved manager decision explicitly applies
that exception here; it is not a general waiver for zero-valued trace fields.

The [separate approved decision](../proofs/w44_docs/receipts/MANAGER_DECISION_RW3_20261027.md)
records RW3 PASS. The original `RW3_VERDICT.json` and raw D6 failures are
unchanged; the disposition is recorded separately. Tenerife programs change
through terrain shading, QSQ initialization and E84 clock work; no per-item
causal proof attributes the hour67 rain change to terrain shading alone.

### Matched release-day speed and memory

Three interleaved same-session 24-hour pairs compare the candidate (N) with
v0.4.3 (O), on the same case and host, with zero new compilation. The registered
stepping statistic is the d03 segment median in seconds per forecast hour.

| Metric | v0.4.3 O | v0.4.4 N | N/O and interpretation |
|---|---:|---:|---|
| Stepping | 5.92033 | 5.91100 | 0.9984 (0.998 rounded); no resolved speed change, PASS |
| Process start to last output | 7.28923 | 7.18373 | 0.9855; disclosed whole-run statistic, not a speedup claim |
| Sampled own-process VRAM peak | baseline | candidate | pooled ratio 0.99557 (0.996 rounded), PASS |
| Host peak RSS | baseline | candidate | pooled ratio 0.99162 (0.992 rounded), PASS |

The stepping repeat spread is 2.79%; the registered ratio-plus-spread test stays
below the 1.10 STOP threshold. The strict 20/50 core-watch shadow has no counted
arms; the contracted 100/200 watch is the one that passed. No stronger isolation
claim follows.

The v0.4.4 72-hour process took 459.276 s (6.37884 s per forecast hour), with
5,426 MiB sampled process VRAM peak and 6.04254 GiB host RSS. **O72 was not run:**
the same-session 72-hour v0.4.3 arm was skipped by the disk-floor admission.
The speed comparison uses the three same-session 24-hour pairs, not a fabricated
72-hour pair or an old-session substitute.

### Earlier v0.3.2 comparison: inconclusive

The 23 October matched three-hour Canary Islands probe measured mean stepping
2.904 s per forecast hour for v0.3.2 and 3.131 for v0.4.3, with the same explicit
allocator/session/harness. The v0.3.2 repeats differed by 8.71%, above the
registered 5% noise limit; the overall headline is invalid. These observations
cannot support a resolved relative-speed claim. They are not the historical
3.3 versus 4.8 s whole-process measurements, which used different conditions.
[Original measured packet](../proofs/w44_perf/receipts/GPU_BENCH_OCT23.json).

## Known issues and limits

The [RC diagnostic ledger](../proofs/w44_rc/KNOWN_ISSUES_RC.md) remains binding:

- A large kInput transpose fusion belongs to the component root graph without
  writer/output cadence. It has not been demonstrated as a canonical default
  or general Jarvis defect; do not identify it by name alone or invent zero
  launch time for an unexecuted fusion. Canonical v0.4.3 rich programs had no
  kInput above 500 (maximum 408). The diagnostic policy does not waive fidelity.
- The root-walk/ring-timing issue, command-buffer partition limitation and
  rejected constant-folding flip remain unresolved/deferred as stated above.
- Thirteen ranked clock/quotient families remain unchanged. Ordinary physical
  rate/unit divisions and unarchived evolving operands are not cleared by the
  14-family repair; pre-existing solar transcendental/FMA residuals remain.
- The nine new LSM controls remain refused, independently of older `z0mvt`.
  Internal AOT fingerprint bypass/legacy-blob behavior is not a supported
  deviceless-to-GPU loading route.
- The long-run heavy-rain, nine-domain and teacher-qualification limitations
  recorded in v0.4.3 are not erased by short tool/component checks. No new
  station-skill or general many-nest qualification is claimed.

## Evidence

A compact source-bound snapshot of the numbers is in
[release facts](../proofs/w44_docs/receipts/V044_RELEASE_FACTS.json). The authoritative
owner report is `receipts/RW3D_20261027/RW3_VERDICT.json` under the RC artifact
root; its SHA256, the CPU/identity tables and original performance packet hashes
are recorded there. The raw source report remains distinct from the separate approved decision above.
