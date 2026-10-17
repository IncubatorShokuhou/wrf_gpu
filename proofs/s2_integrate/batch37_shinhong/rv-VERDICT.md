# MERGE-WITH-FIXES — lane/t-shinhong

Reviewed `44b7b226b55f991d8a5aa26da207ff049dfb5be9` against `897d2b882`.
Reviewer: Codex, lane rv-t-shinhong, 2026-10-15. CPU only; no GPU work or agents.

The physics changes and WRF caller mapping are supported by independent evidence.
One test correctness fix is required before merge. No model-source revision or
repeat of completed numerical/default/GPU evidence is requested for that fix.

## Required fix

`tests/v040/test_shinhong_wrf_oracle.py:109` and
`proofs/t_shinhong/check_envelope.py:33` define normal cells with
`(abs(reference) >= FLT_MIN) & (abs(candidate) >= FLT_MIN)`. A zero candidate
therefore disables comparison against a normal reference. My in-memory mutant
erased u/v/theta/qv/qc/qi/EL/EXCH_H compiled outputs, retained TKE, and PASSED the
compiled gate (`RV04_compiled.json`). This is a gate false PASS, not a finding
that the submitted model has zero outputs.

1. Reject nonfinite reference, candidate and envelope values before comparisons.
2. Compare when either reference OR candidate is normal. Restrict any subnormal
   exclusion to pairs where both are subnormal/zero, with the documented small
   absolute difference; a normal candidate against zero truth must also be checked.
3. Add a regression that injects zero compiled driving outputs against nonzero
   normal references and requires an AssertionError. Also cover the reciprocal
   zero-reference/normal-candidate case. Keep the existing exact two-cell excess
   disclosure and quantum bound; do not widen the physics tolerance.

The Registry-default mode0 branch currently checks held TKE and then skips all
compiled driving fields. Use the identical-input mode1-cold sibling's independently
derived bounds for those driving fields, or derive mode0 bounds; keep TKE held.
This is test coverage only, not a request for another coupled model run.

## Independent evidence

- RV01: **34 PASS, 0 failure/error**, 47.36 s on the exact immutable target:
  oracle, conditional TKE caller and all selected-surface contracts.
- RV02: rebuilt the unmodified original `module_bl_shinhong.F`, SHA256
  `99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56`, with
  the reviewed public wrapper. All **36 tracked rows and all ten public output
  channels BYTE** against the frozen reference; zero/inf initialization BYTE;
  zero/inf/FMA builds all finite. This is independent original-WRF execution.
- RV03: independently reproduced release jaxpr hashes **Swiss f672a573 / PROD
  6fc715a2**, full hashes in RV03_swiss.json/RV03_prod.json, exact target.
- RV04: with the OR comparison, the actual candidate still has only the **two
  disclosed synthetic v tendency cells**, rows4/5 level9, ~3.275e-9 m/s² outside
  the bound. The two cells omitted by AND are inside the bounds. Maximum
  subnormal-pair difference 8.577e-39. This supports the standing small disclosed
  compiled-rounding policy, not a claim that the strict envelope fully passes.
- RV05: independent physical mutant deleting pblflg from the local scale gate
  **KILLED by genuine AssertionError**, row0 u maxabs1.458e-7, zero setup errors,
  no source-file mutations.
- RV07: **1 PASS**, 6.97 s; nonzero RQC/RQI stub through the ACTUAL bl11 adapter
  and operational caller. Entry qc/qi restored; raw rates mass-coupled once as
  WRF calculate_phy_tend, no dt or extra Euler increment in the RK source.
- Producer's 409 shared PASS receipt inspected and reused; immutable target's
  clean worktree materialized and loaded the tracked fixture successfully.

Source audit: original pbl_driver1301–1329 maps QC3D/QI3D/CORF and FM/FHH as
reviewed. Original ndiff=3 moisture solve returns RQC/RQIBLTEN; TTEND includes
PI and public RTH divides PI; all three scale factors use pblflg; parked
mixlen/prodq2/vdifq interfaces and akq previous interface match the pristine
indices. Operational raw source restoration/mass coupling inspected through
RK scalar sc_tend formation. Other PBL numerical functions are unchanged;
the shared adapter change is confined to bl11. No new State/restart leaves.

## Performance and memory

RV06, CPU component on 1,764 replicated real 44-level columns, diag0, forced
REAL4 for both parent and candidate; lab diagnostics only:

| metric | parent | candidate |
|---|---:|---:|
| compile seconds | 1.221 | 1.397 |
| CPU warm median milliseconds | 6.647 | 4.199 |
| temporary bytes | 16,163,488 | 14,794,144 |
| output bytes | 2,201,560 | 2,822,504 |
| optimized CPU HLO while ops | 10 | 8 |

Two cloud rate outputs add 620,944 bytes including output tuple bookkeeping;
temporary storage falls 1,369,344 bytes. The shared moisture RHS and stacked
u/v solve remove one paired Thomas traversal in diag0. Sequential sums unroll
adds without adding a scan. No extra host/device transfer is introduced in
source. These CPU numbers do not establish GPU launches, speed or peak VRAM;
those remain UNKNOWN until the affected program is compiled/measured.

## Integration scope

GPU/coupled-forecast qualification remains pending. **v-prep c07 needs its
24 h GPU rerun and D6 against original CPU-WRF** after integration; no default
Swiss/PROD numerical rerun is justified by the exact identities. The bl1 YSU
sibling gap is outside this review and has a separate owner. The former eager
TKE mismatch is closed; compiled CPU deviations and GPU limits stay disclosed.

Artifacts in `<USER_HOME>/wrf_gpu2_lanes/rv-t-shinhong/`: RV01.xml,
RV02_oracle.json, RV03_{swiss,prod}.json, RV04_compiled.json, RV05_mutant.json,
RV06_cost.json, RV07.xml and their scripts/logs. Temporary source/build/test
directories are removed after the last dependent check.
