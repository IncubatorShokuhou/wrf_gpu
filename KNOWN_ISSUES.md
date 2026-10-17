# Known issues — v0.4.0 draft and v0.3.4 history

**Default path frozen — 2026-10-16 11:59:18Z.** Main `5c15881ede`
(`403240f257` adds proof only) carries the caller-audit repairs and device-only
C5′ MYNN default. Release identities are Swiss **3356e249** and PROD d01
**196aed9f**. F20 plus C5′ original-CPU 24h default gates are closed; the
FINAL composite checkpoint is **66/66 accepted-scope scientific D6 PASS (65 full24h + c45 accepted6h)** at00:26Z on17October.
Final publication follows final review and the release chain. Historical f672a573/6fc715a2 receipts remain pre-F20 baseline evidence.


## Current frozen default and composite checkpoint

Swiss **3356e249** / PROD d01 **196aed9f** are frozen on5c15881ede; full
hashes and source bindings are in the [support matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md#frozen-defaults-and-results-so-far).
F20 covers five24h default cases. C5′ reruns Swiss/PROD/WN3-0227 with150
paired domain-frames, bounded D6/inventory/NF0 and actual E41 PASS3/3.
Host transfers/synchronization/callback counts equal F20; matched nsys −1.4%
is not S1 or ABAB. [Default closure](proofs/s2_integrate/batch62/gate_WN3_closed/GATE_CLOSURE.json).

[66/66 accepted-scope scientific D6 PASS](docs/release/FINAL_COMPOSITES_v0.4.0.md)
are recorded at00:26Z on17October with their measured source. All66 scientific scopes pass:65 full24h plus c45 accepted6h. The [class-A/default fix list](docs/release/FIXES_v0.4.0.md)
records batches38–62. The raw original failures retained below are historical.

## c11 resolved compiler stall and residual resources

The former mp16 WDM6/cu96 SAS-family/bl17 KEPS compile on5c15881ede reached
its60-minute cap without a stepping executable; that INCOMPLETE/no-model-result
receipt remains historical. The BYTE fusion repair merged in batch65; actual
changed-source **b823c8ca87 full24h now passes25frames, complete inventory,
zero bounded misses and NF0**. The measured process wall is900.005s and sampled
process VRAM5386MiB [M], with once/second sampling that can miss brief peaks.
This is not S1 or a new whole-run speedup claim.

Residual actual E41 remains **FAIL**: maximum STACK512B and kInput1353
optimized HLO instructions. Exact sites and SHA bindings are retained in the
[E41 inventory](proofs/r_docs040/E41_OPTIN_DISCLOSURE.json). The old failed
compile's host RSS/cgroup measurements are not a current minimum-RAM claim;
the successful run used a48GiB host-memory cap, not a measured requirement.
[Historical triage and final qualification](proofs/r_docs040/RELEASE_CHECKPOINT.json).

## v0.4.0 draft qualification limits

**DRAFT.** These limits apply to the development candidate and the exact
source-specific receipts below. Final release approval is **TBD-VALIDATION**.
The [support matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md),
[historical frozen snapshot](proofs/r_docs040/EVIDENCE_SNAPSHOT.json) and
[post-merge reconciliation](proofs/r_docs040/RECONCILIATION.json) distinguish
bounded D6 results, field coverage, E41 resources and pending final-source work.
The v0.3.4 statements retained below are dated historical records.

### E41 and incomplete forecast coverage

The manager's 2026-10-16 04:36Z decision keeps E41 release-blocking for the
default WN3, PROD and Swiss F20 programs. Opt-in composites require strict
original-CPU-WRF D6, complete inventory and zero nonfinite output values (NF0).
An opt-in resource failure is disclosed as **E41 FAIL (perf/robustness), D6 PASS**
only when those scientific gates pass. Raw E41 verdicts remain FAIL; incomplete
or failed D6 is never promoted. The frozen C5′ successor has
closed the default E41 gate; this opt-in disposition did not waive it. STACK, non-leaf callee and kInput thresholds
remain unchanged. Exact executable properties and unknown runtime impact remain in
the [complete provisional E41 ledger](docs/release/E41_LEDGER_v0.4.0.md);
source attribution for older named executables remains in the
[supplement](docs/release/E41_OPT_IN_v0.4.md). This is not a waiver for
release defaults or an isolated spill-cost measurement.

Goddard radiation 5/5 has an accepted six-hour scope: seven frames, complete
346-field inventory, zero bounded misses/nonfinite output and E41 PASS [M].
The 24-hour attempt has only h0–23; h24 was cut by the run cap. The tile-2048
repair supersedes the old slowdown report. It is not a full 24-hour PASS.
[Accepted receipt and source](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

### Source-bound opt-in E41 disclosure (00:26Z final)

This table is generated from v-prep's final `FINAL_E41_DISCLOSURES.json`
at `2026-10-17T00:30:45.643194+00:00`: **32 FINAL resource FAILs**, with
33 PASS and c89 UNKNOWN counted separately. The [portable exact-content
projection](proofs/r_docs040/FINAL_E41_DISCLOSURES.json) retains every actual STACK site, kInput fusion,
non-leaf/recompute callee, ELF/JIT/AOT origin and receipt SHA. All rows below
are measured registered FINAL sources (F); the three historical-only failures
in the earlier provisional inventory do not enter the final counts. Raw
E41 FAIL stays FAIL despite scientific D6 PASS. Missing scan data is UNKNOWN,
never zero. STACK is bytes; kInput is optimized HLO; callees are SASS. Callee
columns give site count and independent maxima; exact sites remain in JSON.
Options are MP/CU/PBL/SFCLAY/LSM/LW/SW.

| Case / source scope | Options | Raw D6 / scientific gate | Max STACK B | kInput >500 (HLO) | Non-leaf/recompute sites (max SASS / max nested calls) |
| --- | --- | --- | --- | --- | --- |
| c02 / `04cb09f816` F | 1/1/1/1/4/1/1 | PASS / PASS [E41 disclosed] | 16 | 549, 570, 586, 632, 659, 661 | — |
| c03 / `04cb09f816` F | 2/2/2/2/4/4/2 | PASS / PASS [E41 disclosed] | 2456 | 540, 598 | — |
| c05 / `04cb09f816` F | 4/5/7/1/4/4/4 | PASS / PASS [E41 disclosed] | 856 | 785, 954 | 1 site; 14418 / 72 |
| c06 / `db78f9bd8c` F | 6/6/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 40 | 536, 598, 722 | — |
| c07 / `04cb09f816` F | 9/16/11/1/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 538, 615, 622, 643, 679 | — |
| c08 / `04cb09f816` F | 10/14/12/1/4/4/4 | PASS / PASS [E41 disclosed] | 408 | 714, 904, 973, 1320, 1600 | — |
| c09 / `04cb09f816` F | 13/94/99/1/4/4/4 | PASS / PASS [E41 disclosed] | 424 | 626, 1383 | — |
| c10 / `04cb09f816` F | 14/95/16/1/4/4/4 | PASS / PASS [E41 disclosed] | 536 | 1463 | — |
| c11 / `b823c8ca87` F | 16/96/17/1/4/4/4 | PASS / PASS [E41 disclosed] | 512 | 1353 | — |
| c12 / `04cb09f816` F | 17/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 1336 | 562, 660, 724, 835, 1041, 1064, 1128, 1172 | — |
| c13 / `04cb09f816` F | 18/5/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 681, 973, 1405, 2096 | — |
| c14 / `04cb09f816` F | 19/99/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 900, 1034, 1096 | — |
| c15 / `04cb09f816` F | 21/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 208 | 660, 1235, 1287 | — |
| c16 / `04cb09f816` F | 22/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 640, 641, 649, 856, 944, 1060 | — |
| c18 / `5c15881ede` F | 26/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 904 | 618, 1211 | — |
| c19 / `5c15881ede` F | 28/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 741, 1439 | — |
| c20 / `5c15881ede` F | 40/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 820, 832, 2227 | — |
| c26 / `2ed8c04c26` F | 8/0/4/4/4/4/4 | PASS / PASS [E41 disclosed] | 7232 | — | — |
| c27 / `2ed8c04c26` F | 8/0/10/10/4/4/4 | PASS / PASS [E41 disclosed] | 32 | 529, 531, 543 | — |
| c40 / `36fcfaa64d` F | 8/0/1/1/8/4/1 | PASS / PASS [E41 disclosed] | 384 | 1369 | — |
| c42 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 572 | — |
| c46 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |
| c55 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 80 | 503, 645, 820, 956 | — |
| c56 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 516, 544, 566, 586, 687, 759, 937, 1015, 1179, 2477 | — |
| c57 / `2ed8c04c26` F | 8/0/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 509, 578, 643, 708, 712, 759, 797, 830, 992, 1015, 1253, 1707, 2566 | 15 sites; 9612 / 2 |
| c61 / `04cb09f816` F | 97/0/0/91/4/4/4 | PASS / PASS [E41 disclosed] | 328 | — | — |
| c65 / `5c15881ede` F | 38/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 496 | 638, 644, 1089, 1134, 1314 | — |
| c83 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 887 | — |
| c85 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 384 | — | — |
| c87 / `04cb09f816` F | 55/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 5520 | — | — |
| c88 / `5c15881ede` F | 56/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 3624 | 894 | — |
| c90 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |

For c03, STACK2456B is pre-existing since the f4fd predecessor; final c03 also
has kInput540/598 HLO. For c05, STACK856B, kInput954/785 HLO and a14,418-SASS
non-leaf callee with72 nested calls are a recomputation signature. This does
not measure its runtime share; c05 runtime attribution is a v0.4.1 follow-up.
Earlier c05 raw D6/resource failures remain preserved.

The producer separately lists c89 as resource UNKNOWN despite D6/inventory/NF0 PASS under
its attempt-selection scope. Missing flags are UNKNOWN, never zero; historical
reuse/successor receipts in the older ledger retain their exact-source validity.
Neither list approves an unmeasured final executable. CAM-MG's separate offline
native STACK1240B scan must not be replaced by an XLA glue-only PASS.

The JSON retains the historical five-case F20 packet at sourcee7e37390cc:
bounded D6 PASS but raw E41 FAIL. That old failure remains failed. The frozen
C5′ successor now closes current default D6/E413/3 and transfer gates; its
identity/default qualification is in the release checkpoint. The old
`default_F20` field is not a current default blocker. F2 remains separate and
these profiling/component results make no S1 claim.

### GPU refusals and pending affected runs

CAM-MG `mp_physics=11` is GPU-admitted through the merged stock-Triton 3.6.x
route (batch45), with explicit compiler/device preflight. Its original c89
GPU attempt failed frozen RAINNC D6, terminated with only15/25 hourly frames
and lacked DLF2. A later source9a40424b0 run has25frames, zero bounded misses
and NF0, but raw D6 remains INCOMPLETE because DLF2 was missing. The merged
DLF2 writer preserves WRF's initialized +0 while shallow convection is refused;
registered-source c89 now passes full24h D6/inventory/NF0; native E41
remains UNKNOWN while the separate offline STACK1240B scan is FAIL.
[Current receipt](proofs/r_docs040/RELEASE_CHECKPOINT.json).
[CAM-MG detail](docs/CAMMGMP.md), [raw receipt/decision](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
The localized defect is in jaxlib/XLA's bundled Triton pipeline: a minimal
reproducer lost live input loads and wrote zeros, while stock Triton retained
them and passed the small actual-GPU activity gate. This does not by itself
qualify the full native CAM-MG forecast. The retained localization packet is
[proofs/t_rainnc/LOCALIZATION.json](proofs/t_rainnc/LOCALIZATION.json).


`km_opt=3` is GPU-refused by the integrated pre-JAX/pre-device guard.
The Swiss GPU forecast became nonfinite in U at
step200 (one hour); attempted original-CPU Swiss/Madrid references crashed.
Both classes of failure remain preserved; column evidence is not a successful
coupled reference. [Original failed pair](proofs/v040/gpuqual/km3/D6.json),
[decision](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

The c05 WSM5/ACM2/Grell composite on de374a3c0 retains RAINNC RMSE1.108mm,
above the frozen1mm bound at h16–24 [M]. The paid ACM2 condensate-caller successor5131ddffb now has a full24h
original-CPU bounded D6/coverage PASS over25frames, with E41FAIL and F2OPEN.
The earlier failed trajectory remains failed. The merged final-composition
source04cb09f816 c05 also passes strict24h D6, full inventory and NF0, while
E41 remains FAIL. [Composed-source receipt](proofs/r_docs040/RECONCILIATION.json),
[earlier successor receipt](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
Jensen c87 and final-source NTU c88 now pass full24h D6/inventory/NF0 with
E41FAIL. Older QHAIL-incomplete output remains historical. The merged writers do not relabel old receipts.
[Current registered-source table](docs/release/FINAL_COMPOSITES_v0.4.0.md).

Surface mosaic (`sf_surface_mosaic=1`) is refused: Noah-classic needs missing
per-tile binding/carry/restart/aggregation infrastructure, while original WRF
ignores this switch for Noah-MP. Valid crop/irrigation/groundwater maps do not
qualify mosaic. [Admission rule](src/gpuwrf/io/noahmp_options.py).

Where advanced single-domain-only options remain admitted in nests, report
**nested not validated**. Actual nested/pairing refusals still apply. Neither
an original single-domain D6 result nor a finite nest smoke provides arbitrary
nested fidelity. [Scope](docs/release/SUPPORT_MATRIX_v0.4.0.md).

### Intermediate negative moisture and number values

Original WRF reproduces the magnitude and predominantly interior location
classes of large post-RK negatives in matched Swiss default/WDM6 runs. These
are not purely roundoff. This accepted narrow known issue does not justify
every GPU amplitude or repair. Residual GPU/original extremum ratios are
1.140410 for default QI, 1.394381 for default QG, 1.004246 for WDM QS and
1.164901 for WDM Nr [M]. WDM snow/graupel visit-count differences also remain
OPEN; counts are repeated phase-field visits, not unique events. No extra
clip or new acceptance band was introduced.
[Original-WRF comparison and disposition](proofs/t_f2neg/REPORT.md).

The final hail quartet retains negative-event justification as OPEN. Its
observed nonfinite counters are zero, but qh is absent from the counted field
inventory (Nh is present); this is not full hail pre-mask clearance.
[Final-source quartet](proofs/v040/gpuqual_qhmass/QUALIFICATION.json).
NTU's separate case-scoped source-diffusion exception is described below;
neither exception is global F2 clearance.

### Original MYNN undefined initialization and P3 precision

Pristine WRF 4.7.1 passes local `rmol` into MYNN cold initialization before its
first assignment. A Valgrind/native-PC source witness locates the mixing-length
read. The port already uses zero. Some original MYNN/P3 references therefore
depend materially on an undefined value. Designated comparisons use unchanged
original bodies compiled under the documented ZERO convention; raw-original
results and initialization spread remain preserved. Blanket compiler
initialization can affect other undefined sites too. Frozen D6 bounds do not
change. [Upstream source proof](proofs/t_p3fix/UPSTREAM_WRF_BUGS.md),
[original/ZERO reference disposition](proofs/t_p3fix/DISPOSITION.md).

The retained P3 native column replay has 240 strict deviations; eight values
remain outside the independently reproduced original 44-level FMA envelope
(six NR, one QC, one BIR) [M]. They are accepted as a disclosed small deviation,
not relabelled as a strict column pass. MP52's compiler repair produced an original-CPU 24-hour bounded D6 PASS and
actual E41 PASS at its measured source. Newer P3 50–53 receipts also close
full history inventory; the earlier missing-field reports remain historical.
Final composed-source approval is **TBD-VALIDATION**. [P3 numerical disposition](proofs/t_p3fix/DISPOSITION.md),
[MP52 successor](proofs/t_p3compile/QUALIFICATION.json),
[later radar retention](proofs/t_writer_p3_radar/QUALIFICATION.json).

### WDM5 and Jensen compiled residuals

WDM5 Nn/Nc miss the canonical two-percent number band in the theta-handoff
columns; original REAL4 versus REAL8 itself differs by 9.8852928e7 #/kg [M].
The retained fp64 island is precision-sensitive. This is not a number-gate
PASS or a new tolerance. [Two-call original-kind evidence](proofs/t_thetafix/README.md).

Jensen's existing compiled CPU column implementation has 335 values outside
its independent enclosure across 17 finite columns [M]; those remain small
experimental deviations, not strict enclosure passes.
[Original column qualification](proofs/s2_mpmore/jensen55/STATUS55.json).
Its expanded ice diagnostics preserve previous operational outputs,
but 165 values across 17 original columns exceed the unchanged input-ULP/FMA
enclosure, with maximum relative excess 1.43e-5 [M]. The small diagnostic
deviation is explicitly disclosed; full coupled GPU qualification remains
**TBD-VALIDATION**. [Definitions and independent disposition](docs/physics/ice_diagnostics.md).

### Exclusions and other open work

- MP15 and `km_opt=3` are column-qualified experimental paths, excluded from
  coupled coverage because original CPU-WRF crashes on the attempted cases.
  They are not successful reference forecasts. Crop/tile/map-selected
  irrigation branches without geographically valid NCAR maps lack 3-D
  validation. [Reference/exclusion receipts](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- Class-D producers and WRF-constant fields are merged (batch40/41), with
  admitted initialized/held branches and active refusals listed below. CPU
  producer proofs do not qualify final GPU trajectories or repair old missing
  inventories. [Composition](proofs/s2_integrate/batch40/QUALIFICATION.json),
  [independent caller review](proofs/rv_t_classd/composition_review/VERDICT_COMPOSITION.md).
- The earlier Shin-Hong diagnostic deviation has a later caller/kernel
  correction with source44b7b226b c07 full24h D6/coverage PASS and E41FAIL;
  final-source identity and the disclosed native quantum residual remain separate.
  GBM mixing diagnostics retain small strict residuals. Their investigation
  is separate from finite output or bounded forecast comparisons.
  [CAM-MG/PBL caller audit](proofs/s2_cammg/README.md),
  [GBM repair qualification](proofs/t_c02nan/NAN_REPAIRS_QUALIFICATION.json).
- MP28's startup emission-height correction follows the measured mismatch
  between WRF interface height and the port's mass-averaged height. Its
  CPU producer evidence does not establish a new full GPU trajectory PASS.
  [Measured mismatch](proofs/v040/gpuqual_mp28_followup/summary.md),
  [subsequent fix](proofs/t_writer_mp28_emission/QUALIFICATION.json).
- Final composed-candidate gates, station campaigns, full CPU-suite closure
  and real multi-GPU scaling remain **TBD-VALIDATION**. Heavy tests have a
  measured separate CI tier; the test/source/attempt list is bound as
  `heavy_tests` in the [snapshot](proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
  Historical v0.3.3 coastal-wind/rain/cirrus deviations below remain disclosed.

## v0.3.4 historical release status

**v0.3.4 (fix release).** wrf_gpu ≤ 0.3.3 silently ran the default physics suite when `namelist.input` selected another microphysics, PBL, surface-layer or radiation scheme or other damping/filter values. Re-run such forecasts; the binding fix and full key list are described below. Experimental opt-in schemes added in v0.3.4 are matched to unmodified-WRF oracles on CPU and most were GPU-smoked (three finite steps); none is a validated forecast configuration. CAM radiation (`ra_lw/ra_sw_physics=3`) still fails the static kernel stack check (2,144 B spill; fix planned for 0.4.0). Per-scheme gaps: [support matrix](docs/release/SUPPORT_MATRIX_v0.3.4.md). The default program is unchanged, so the v0.3.3 deviations below still apply.

Open v0.3.3 follow-ups (not addressed in v0.3.4): coastal 10 m wind-speed excess; nine x-major loop-layout invariants on unpinned paths (E41 hardening); the amplitude of WRF-consistent negative-moisture visits versus original WRF is unproven; the ≤4,096-column stability firewall cost should become a layout-level fix.

## CAM-MG GPU precipitation

`mp_physics=11` (CAM-MG) runs on GPU again. Earlier v0.4.0 builds refused it
because XLA's bundled Triton (both the in-graph route and the PJRT plugin
compiler) miscompiled the native column and returned zero `RAINNCV`/`SNOWNCV`.
The column is now compiled to PTX by **stock Triton 3.6** and launched through
jaxlib's `triton_kernel_call`; other kernels keep XLA's route. GPU mp11 runs
therefore need `pip install "triton==3.6.*"`; other Triton versions, a lowering
without a visible CUDA device, or a missing Triton install are refused with an
explicit error rather than silently compiled. The native kernel uses 255
registers and a 1240-byte stack frame (E41 STACK flag, disclosed for this
opt-in scheme); the first mp11 call of each fresh process assembles the 5.2 MB
PTX in-process (about 26 s per grid shape; not cached across processes).
Coupled 24 h GPU qualification against CPU-WRF is pending.

<a id="v040-history-fields-without-a-producer"></a>
## v0.4.0 history producers and WRF-constant fields

The nine CPU-WRF history fields previously omitted here are now written. For the
configurations the port admits, WRF itself writes the values below; the WRF branches
that would make them vary remain unported and are **refused at load**, never run
silently. Original CPU-WRF proofs cover every cell of all 25 frames
([proofs/t_classd](proofs/t_classd/)).

| Fields | Configuration | Value written (WRF source) | Still refused |
| --- | --- | --- | --- |
| `QGR_URB4D` | BEP+BEM `sf_urban_physics=3`; c57 | Held `QGR_URB3D`: Noah-MP-initialized top-layer `SMOIS` (`urban_var_init`), reset to 0 on BEP/BEM columns every call | Green roofs (`GR_FLAG`/`GR_FRAC_ROOF`) |
| `TGR_URB4D`, `DRAINGR_URB3D`, `EP_PV_URB3D` | BEP+BEM; c57 | +0: initialized and reset to 0; PV power is `pv_frac_roof`-weighted | Green roofs, photovoltaic roofs (`PV_FRAC_ROOF`) |
| `QNBCA` | Thompson aerosol-aware `mp_physics=28/38`; c19/c65 | +0: `mp_gt_driver` zeroes black carbon unless `wif_input_opt=2`; nonzero `QNBCA` input is refused | `wif_input_opt=2` black-carbon aerosol |
| `TAOD5502D` | `mp_physics=28/38`; c19/c65 | +0: assigned only under `aer_opt=3` with RRTMG SW | `aer_opt=3` aerosol-radiation coupling |
| `MSKF_REFL_10CM` | Morrison aerosol `mp_physics=40`; c20 | +0 with `do_radar_ref=0` (WRF assigns it only under `do_radar_ref=1`). With `do_radar_ref=1` it equals the computed `REFL_10CM`: its MSKF cumulus hydrometeor arrays exist only for `aercu_opt>0` and are unallocated in WRF otherwise ([upstream note](proofs/t_classd/UPSTREAM_WRF_BUGS.md)) | `aercu_opt>0` (MSKF + CESM aerosol) |

One-moment NSSL (`mp_physics=19/21`) `REFL_10CM` is computed with WRF's one-moment
`radardd02` branch (byte-exact against pristine WRF REAL4 column output). Morrison
`mp_physics=10/40` with `do_radar_ref=1` (and `nwp_diagnostics=0`) now computes
`REFL_10CM` with WRF's `refl10cm_hm` at history times (pristine REAL4 column oracle:
byte-exact on all 13,376 levels, melting layers included).

The Morrison history reconstruction has84 column-frames at h10–14 differing
by up to15.45dBZ. Missing exact MP-exit operands is an attribution hypothesis,
not a demonstrated cause or coupled PASS. The pristine oracle on the
reconstructed operands equals the port byte for byte. Preserve this distinction
from the routine-boundary oracle and final trajectory gate.
[Class-D evidence and limits](proofs/t_classd/README.md).

Jensen55 now retains its actual pre-sedimentation `REFL_10CM` every MP call;
it is not reconstructed at output time. Its routine-boundary CPU oracle is
byte-exact on5,632 recorded dBZ values, with the disclosed one-ULP input-recipe
qualification. NTU56 now writes prognostic `QHAIL`; original CPU c88 has no
REFL_10CM field. Jensen ice diagnostics are implemented with the residuals
disclosed above. CAM-MG `DLF2` is WRF's initialized +0 only in the admitted
shallow-off branch. NSSL/Jensen `REFD_MAX` is initialized +0 only under the
documented inactive diagnostic controls. These repairs do not relabel old
INCOMPLETE outputs or establish affected-source GPU cost/fidelity.
[Jensen/NTU review](proofs/rv_t_classd/jensen_review/VERDICT_JENSEN.md),
[DLF2 source/CPU proof](proofs/t_cammgws/README.md#dlf2-history-field-mp11-2026-10-16),
[exact inactive branches](src/gpuwrf/io/producer_history.py).

For Noah-MP (`sf_surface_physics=4`) with `grid_sfdda=2`, `HFX_BOTH` and
`QFX_BOTH` retain WRF's initialized zeros: their flux calculation belongs to
classic Noah. The [c78 original-CPU proof](proofs/t_writer_fields/FASDAS_NOAHMP_CPU_ALL_FRAMES.json)
covers every cell in all 25 frames. Composite c53 has only one CPU reference
frame and remains **INCOMPLETE**; no 24-hour qualification is inferred for it.

## NTU56 development: case-scoped F2 source-diffusion undershoots

The opt-in experimental NTU scheme (`mp_physics=56`) has post-RK ice mass
and ice number undershoots in the real Swiss two-step CPU continuation.
The complete legacy-path census records 93 qi and 127 Ni visits, with minima
−1.3227e−10 kg/kg and −954.597/kg, respectively, about 1.5% of the affected
column maxima. These exceed the tested 1e−12 relative round-off criterion;
**they are not round-off and are absent from the matched default control.**

Pristine WRF's complete sixth-order horizontal diffusion routine reproduces
the negative class on the captured full-grid operands. The captured settings
are `diff_6th_opt=2`, `diff_6th_factor=0.12`, `diff_6th_slopeopt=0`: the source
uses the RK1 dt/3 tendency, frozen for the final dt update. No NTU source-term
sign, units or clipping discrepancy was found. No equation, diffusion
coefficient, input, tolerance or clip was changed to suppress the events.

The manager accepted this **case-scoped source-diffusion F2 known issue** on
2026-10-14 at 22:44:27Z for v0.4.0; a full original RK-chain comparison is
not required for this disposition. Its scope is the captured single-domain,
44-level Swiss CPU continuation at dt=18 s, CCNTY=2, FAST0; selecting the
existing native PD transport path retains the class. This is not global F2
clearance or complete original-CPU/GPU forecast qualification. The later old-source
c88 GPU run has25frames with zero bounded misses and NF0, but raw D6 remains
INCOMPLETE for QHAIL and E41 FAIL (STACK3496B). The QHAIL writer and NTU
performance repairs are now merged; final affected-source D6, resources and
performance/VRAM qualification remain pending.
[Final-source integration](proofs/s2_integrate/batch57/QUALIFICATION.json).
See the [qualification and sealed replay inputs](proofs/s2_ntu/README.md)
and [measured F2 ledger](proofs/s2_ntu/QUALIFICATION.json).

## Noah-MP glacier points in shipped versions through v0.3.4

Glacier initialization and undefined history fields were handled, but the
timestep incorrectly ran natural-soil Noah-MP at `IVGTYP == ISICE` points.
WRF calls the separate `NOAHMP_GLACIER` model there. The shipped Swiss example
contains 22 affected points; all 54 domains of the 18 WN3 validation cases and
both PROD domains contain zero. Earlier Swiss forecast receipts do not qualify
the missing glacier timestep.

The development fix adds glacier energy, ice phase change, snow layers and
water balance for `opt_gla=1/2`, with the previous-call `QSNOWXY` retained in
restart state. It intentionally changes the Swiss program. Glacier CPU oracle
and short coupled qualification are separate from the final Swiss 24-hour
comparison against CPU-WRF. The new c00 source-specific 24-hour comparison
now passes bounded D6, complete coverage and E41; final-RC carry-forward remains
TBD-VALIDATION ([receipt](proofs/r_docs040/EVIDENCE_SNAPSHOT.json)). `opt_snf=5` on a glacier
domain is refused because the pristine glacier precipitation partition has
no branch for that choice.

The development glacier history writes `QRAINXY=0` at ice points, matching
WRF's defined per-point hold on the qualified cold-initialized path. The
pristine driver loads that point's `QRAINXY` before the glacier branch and
writes it back unchanged; `NOAHMP_INIT` initializes it to zero. Arbitrary
nonzero externally seeded `QRAINXY` is outside this qualification. The glacier
water budget uses its own rain partition.

Development fix in v0.4.0: WDM5/6 (`mp_physics=14/16`, including shipped WDM6)
omitted cloud/CCN number (`Nc/Nn`) transport, and Morrison (`mp_physics=10`)
omitted snow/graupel number (`Ns/Ng`) transport. WDM rain number (`Nr`) could
already use the materialized Ni/Nr path; the fix transports Nr exactly once
and removes inactive WDM Ni. The fix transports those scalars with WRF's
scalar advection option, reads `QNSNOW/QNGRAUPEL/QNCLOUD/QNCCN` from inputs,
and applies WRF's `flow_dep_bdy_qnn` with bound `ccn_conc` inflow. WDM7 (`26`)
uses the same repaired number path plus its existing hail transport and a
documented fp64 microphysics island. Missing prognostic numbers in active
restarts are refused. These schemes are restricted to single domains until
parent-to-child forcing is qualified. CPU operator evidence is separate from
GPU/coupled-forecast qualification. Later scheme-specific status is recorded in
the [v0.4.0 matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md); the default
Thompson release configuration is unchanged.

## Non-Thompson precipitation and WDM caller defects

These defects are confirmed in **v0.3.0–v0.3.4 and pre-fix v0.4.0 development
builds**, whenever the affected adapter actually ran. In v0.3.3 and earlier,
the CLI often ignored non-default physics selections and ran Thompson instead;
those selections alone do not establish exposure. API/custom-driver runs and
v0.3.4 CLI runs must be assessed by the scheme that actually executed. WSM7,
WDM7 and Morrison-aerosol were refused by the v0.3.4 CLI but became runnable in
v0.4.0 development.

- **Frozen precipitation counted twice in `RAINNC`.** The generic adapter
  stored WRF's all-phase `RAINNCV` as liquid rain, then the writer added the
  overlapping frozen channels again. This affects Purdue–Lin (`mp_physics=2`),
  WSM3/5/6 (`3/4/6`), Morrison (`10`), SBU-YLin (`13`), WDM5/6 (`14/16`),
  WSM7/WDM7 (`24/26`), Morrison-aerosol (`40`) and Goddard (`97`) when frozen
  fallout occurs. Kessler has no frozen increment; Thompson (`8`) and
  Thompson-aerosol (`28`) use separate adapters and are unaffected by this
  bookkeeping defect.
- **Shipped WDM6 (`mp_physics=16`) and WDM7 (`26`) thermodynamic coupling was
  wrong.** Both passed prognostic moist theta to a scheme expecting dry theta
  and omitted the conversion back to moist theta after microphysics. WDM6
  also passed water mask `0` instead of WRF's `2`, selecting the land
  autoconversion threshold over water. The development fix restores the WRF
  dry-theta preparation, moist-theta finish and WDM6 water mask.
- **WSM5/WSM6 (`mp_physics=4/6`) and WDM5/6/7 (`14/16/26`) radiation
  coupling and caller operands were not WRF's.** With RRTMG and `use_mp_re=1`
  (the WRF default) WRF `physics_init` sets `has_reqc/i/s=1`, so RRTMG reads
  the scheme's effective radii and treats snow as its own optical species. The
  port never activated this for these schemes: RRTMG used fallback radii and
  folded snow into the ice path, making snow layers too opaque in the longwave
  (Swiss hour-1 OLR bias c05/c06 −3.5/−3.2, WDM c10/c11/c18 −3.2/−3.1/−4.0
  W m⁻² against original CPU-WRF; radius-coupled schemes ≈0). The WSM/WDM
  callers also passed the Thompson-formula density (R 287.04) instead of the
  post-RK dry `1/(al+alb)` and `DELZ` with g 9.80665 instead of WRF's 9.81.
  The development fix (lane t-wsmdz) activates the WSM5/6 and WDM5/6/7 radii
  and passes WRF's post-RK `DEN`/`DELZ` (WSM3 included); 24-hour GPU
  re-qualification of the affected composites is pending.
- **NSSL (`mp_physics=17/18/19/21/22`), Ferrier (`5/15`) and Goddard 4-ice
  (`7`) microphysics received the wrong air density.** WRF's microphysics
  caller passes the post-RK dry density `1/(al+alb)` that `moist_physics_prep_em`
  forms immediately before `microphysics_driver`; the port passed the earlier
  `phy_prep` total `(1+qv)/ALT`, about 0.3–1.5 % too large in moist layers (NSSL
  also clamped `DZ` at 1 m, which WRF does not). Recomputing original CPU-WRF
  NSSL `REFL_10CM` (composites c14/c15) from its own history reproduces WRF to
  7.6e-6 dBZ only with the dry density (0.043 dBZ off with the old one). The
  development fix (lane t-classd) passes WRF's post-RK `RHO`/`DZ8W` for all
  three families; 24-hour GPU re-qualification of c12–c16 and c62–c64 is pending.
- **WSM3 (`mp_physics=3`) reached RRTMG with its cold ice/snow staged as
  liquid.** WSM3 stores ice in `qc` and snow in `qr`; WRF's RRTMG LW/SW drivers
  restage them as ice/snow below 273.15 K (`module_ra_rrtmg_lw.F:12075-12083`,
  `module_ra_rrtmg_sw.F:10676-10684`) and, with the RRTMG LW/SW pair and
  `use_mp_re=1`, read the WSM3 radii. The port did neither (Swiss c04, RRTMG LW
  + Dudhia SW, hour-1 OLR bias −12.9 W m⁻²). The development fix adds the phase
  view and the WSM3 radius selection; the c04 hour-1 OLR bias on the original
  CPU-WRF state drops to −3.1 W m⁻², the same residual the other radius-less
  RRTMG schemes show (RRTMG `has_req=0` path, separate item).

The v0.4.0 development fix has independent CPU oracle and one-hour CPU-WRF
comparison evidence. Later receipts establish source-specific full 24-hour
bounded D6/coverage for WDM6 and the corrected WSM7/WDM7 quartet; Morrison-aerosol
coverage remains incomplete and F2 negative-event justification stays OPEN.
See the [current matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md) and
[quartet receipt](proofs/v040/gpuqual_qhmass/QUALIFICATION.json). Substantial negative
moisture/number candidates are not justified as blanket roundoff; this CPU
defect fix does not establish GPU forecast qualification or close F2. See the
[CPU composition evidence](proofs/t_mprain/COMPOSITION.json) and
[re-qualification plan](proofs/t_mprain/REQUALIFICATION.json).

**Accumulations in checkpoints written by older affected versions are not
repaired or migrated.** Resuming preserves the existing erroneous totals;
correcting earlier accumulated precipitation requires a fresh forecast.

## v0.3.4 restart schema gaps

In v0.3.4, the NetCDF restart API refuses a populated experimental CAM-UW PBL
carry (`bl_pbl_physics=9`) because its inventory lacks that group. The older
State-only pickle checkpoint reader also refuses current default checkpoints
with a `missing non-additive leaves` error. These are loud errors, not silent
loss of state; the CLI's authenticated full-carry checkpoint store is separate
and its release-default continuation was verified byte for byte.

The development fix adds the CAM-UW NetCDF group and allows inactive optional
State fields in the older reader. Held CAM-UW carry, MYNN WSPD and effective
radii are preserved exactly when populated. Active configurations with missing
held state are refused (E78), including Morrison-aerosol radii independently of
the Thompson radiation flag. CAM-UW files predating the new group/schema marker
can resume only with CAM-UW inactive; regenerate active checkpoints. A legacy
NetCDF file without saved PBL selection requires the caller's resolved inactive
namelist (`read_wrfrst_carry(path, namelist=...)`); an unknown selection is refused.

## v0.3.3 disclosed deviations (unchanged in v0.3.4)

All eight original-CPU scoring readers are closed. The manager accepts the disclosed deviations under the Owner rule; raw failed checks remain failed.

- 0115 strict24 accumulated rain RMSE 1.042–1.045 mm versus 1 mm; fresh same-IC replica .845–.886 mm passes, demonstrating trajectory sensitivity without replacing the failed primary.
- 0115 late V10: primary h44 1.692 m/s; fresh replica h42 1.725/h44 1.623 versus1.5. Frozen A3 REAL FAIL rows retained. Coastal station wind-speed RMSE +1.8%; T2/RH2/WD improve.
- 0408 trace graupel: CPU2.46e-10kg/kg/GPU0 in5boundary cells. 0115 cirrus lifetime: primaryd02h60 QICE2.28e-7kg/kg/QNICE2.4e4/kg in50columns; freshd03h63 QICE2.14e-8/QNICE4.5e4 in112columns, GPU0. Both raw integrity annexFAILs disclosed.
- Unclipped WRF-order source/transport may briefly produce negative moisture: qc−3.84e-6,qv−4.20e-6kg/kg;634857censusvisits,0nonfinite/newclass. This is not a claim of pure round-off.
- Swiss≤4096-column stability firewall:+12%kernels/~8%ordinary-root time, notwholewall; larger WN/Mon programs unchanged.

[All final plots](docs/release/evidence/v033/final_w3/index.html) · [Raw reports, scope and ledger](docs/release/TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-eight-arm-results).

## ≤ v0.3.4: `moist_adv_opt = 0` or an omitted `moist_adv_opt` ran without moisture transport (fixed after v0.3.4, lane o2-moistadv)

In wrf_gpu **0.3.0 – 0.3.4** (and the internal 0.23.4+ builds) the RK step transported water vapour and every
condensate (`QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`, plus scheme extras) **only when
`moist_adv_opt` was 1 or 2**. WRF's option 0 (`ORIGINAL`) is the ordinary 5th/3rd-order flux transport without
the positive-definite limiter; the port treated it as *no* dycore moisture transport. Both CLI drivers also bound
an **omitted** `moist_adv_opt` / `scalar_adv_opt` as 0, while the WRF Registry default is 1.

**Who is affected:** CLI runs whose `namelist.input` set `moist_adv_opt = 0` **or did not set `moist_adv_opt` at
all**. In 0.3.0 – 0.3.2 (and in 0.3.3 / 0.3.4 with `GPUWRF_FAST_DEFAULTS=0`) such runs completed with moisture
moved only by physics and lateral boundaries — re-run them. With the 0.3.3 / 0.3.4 release defaults the same
namelists stopped after model load with `NotImplementedError: GPUWRF_PHYS_TEND_RK_WRF: physics sc_tend for
non-advected [...]` — no completed forecast (an initial lead-zero frame may exist). A namelist that omitted
`scalar_adv_opt` (or set 0) ran Thompson `QNICE`/`QNRAIN` with the plain stencil instead of WRF's default
positive-definite one (on nests with both keys 0 also from stage-start operands). Namelists that set both keys to
1 or 2 — the release namelists (Swiss, PROD, WN3), the WRF `em_real` template and the shipped examples — are **not
affected by that transport fix**; its historical pre-glacier Swiss identity
was `2bf12cb8`. The pre-F20 candidate baseline has Swiss `f672a573` and PROD d01
`6fc715a2`, with separate glacier/source qualification
([identity and receipts](proofs/r_docs040/EVIDENCE_SNAPSHOT.json)).

**Fix:** option 0 now runs WRF's plain transport on every RK stage (pristine `advect_scalar` parity on real
Swiss operands, `tests/test_o2_moistadv_original.py`), an omitted key binds the Registry default 1 in both
drivers, and `cu_physics = 6/16` no longer needs `moist_adv_opt ≠ 0`. Omitted-key audit of every bound option:
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md#cli-binding-v034). Besides the already-documented
substitutions (omitted scheme selectors run the release suite with a warning, `sf_surface_physics` and the replay
driver's fixed harness values, omitted `diff_opt`/`km_opt`), the only other mismatch was an omitted `radt` (WRF:
radiation every step; the port: 30 min), which now prints a warning — set `radt` explicitly.

## ≤ v0.3.3: `gpuwrf run` silently ignored non-default physics options (fixed in v0.3.4)

In wrf_gpu **0.3.3 and earlier**, both CLI drivers (the native single-root / live-nested
driver and the CPU-history replay driver) **never read** `mp_physics`, `bl_pbl_physics`,
`sf_sfclay_physics`, `ra_lw_physics` or `ra_sw_physics` from `namelist.input`, nor the
`&dynamics` knobs `epssm`, `damp_opt`, `zdamp`, `dampcoef`, `w_damping`, `diff_6th_opt` and
`diff_6th_factor` (the replay driver also ignored `cu_physics`, `radt`, `cudt`, `top_lid`).
The namelist check accepted the requested scheme and the forecast then **ran
Thompson / MYNN / MYNN surface layer / RRTMG** with epssm 0.5, damp_opt 3, zdamp 5000,
dampcoef 0.2, w_damping 1 and the 6th-order filter (2, 0.12). The validator also refused
`icloud_bl=1`, the value that actually runs, and accepted `icloud_bl=0` and
`bl_mynn_mixlength=2`, which never ran.

**Who is affected:** only runs whose namelist selected something else. The release
configuration (Thompson / MYNN / Noah-MP / RRTMG / KF with those dynamics values) asked for
exactly what ran and is unchanged; its traced program is byte-identical in v0.3.4. **Re-run
any ≤ 0.3.3 forecast that selected a different microphysics, PBL, surface-layer or radiation
scheme or different damping/filter values:** its output came from the default suite.

**v0.3.4:** every scheme and dynamics option is bound per domain. Before any compute,
`gpuwrf run` **refuses** any explicit `&physics` / `&dynamics` / `&noah_mp` value the
selected driver would not run. That includes values that differ from the WRF Registry
default of a key the port does not bind, per-domain selections WRF itself rejects or
overrides (WRF runs the innermost domain's `mp_physics` everywhere), and land options the
native driver cannot build (it supports only 0 and 4). The CPU-history replay driver warns
loudly that its fixed 10 s step, 10 sound steps and hourly CPU land replay replace
`time_step`, `time_step_sound`, `sf_surface_physics` and `sst_update`. An omitted `&dynamics`
knob now runs the WRF Registry default (`epssm` 0.1, `w_damping` 0, `diff_6th_opt` 0; ≤ 0.3.3
hard-wired 0.5 / 1 / 2).

**Exposed by the binding (v0.3.4 CLI probe on the real Swiss case, release defaults):** most
non-Thompson microphysics (`mp=1/2/3/4/6/10/13/14/16/97`), PBL `bl=7/8/12` and GSFC shortwave
`ra_sw=2` could not even be traced under the REAL32 release carry; ≤ 0.3.3 hid this by running
the default scheme. They now run as an explicit fp64 island (scheme computed in fp64, stored
REAL): CPU-traced, not GPU- or forecast-qualified. `mp=24` (WSM7), `26` (WDM7), `28`
(aerosol-aware Thompson) and `40` (Morrison-aerosol) are refused: the release root boundary has no
`qh` / `nwfa,nifa` / `Ns,Ng,Nc` record. Constant-K diffusion (`diff_opt=2`, `km_opt=1`) ran as *no*
explicit diffusion through the CLI and is now refused, with every other `(diff_opt, km_opt)` pair
that has no operational path. The 3-D TKE (`km_opt=2`) and SMS-3DTKE (`km_opt=5`) closures are
unqualified v0.22 scaffolds that produce NaN under the release REAL carry and are refused in v0.3.4;
`diff_opt=1/km_opt=4` (release) and `diff_opt=2/km_opt=3` (3-D Smagorinsky) run.
Per-code CLI support matrix:
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md#cli-binding-v034).

## Historical issue records (source-dated)

# Known issues and validation limits — v0.3.3

## v0.3.3 CF/b6 wave — historical release HOLD

[Exact current eight-arm evidence](docs/release/CURRENT_W3_EVIDENCE.md) retains
strict0115 d03 RAINNC h21–24 FAIL, hard72 0408 d02 RAINNC h39–72 FAIL and two
0408 QGRAUP REAL FAILs. Current original-CPU/station evidence has0227 8/8 CLEAR
and0115 WS10 A/B DAMAGE+REPLICATED, SYSTEMATIC (2/16); other14 CLEAR.
Monica72/Swiss24 D6 and all-frame integrity annex pass, without clearing the
other cases. Replica CANICE/SNOWENERGY raw flags remain disclosed separately.
Six arms executed CF; Monica/Swiss executed b6. No publication or fullE41 PASS.

## Historical W2 record — source-dated

Reviewed W2 source/defaults candidate is `e83ec1fa7` (`src/` tree
`fb50726fee1e`). Its external gates remain open; source MERGE is not release
acceptance. Original 69d8a9a7d evidence is retained as **pre-W2**.

- **W0115 D6+floor annex: FAIL.** Of 89 raw d03 rows, 57 are floor-limited
  and 32 remain REAL: U10 h38–52 (15), V10 h37–58 (17). Raw wind maxima
  2.38/3.04 m/s; 54 RAINNC h19–72 rows are floor-limited. The strict
  24-hour window is unchanged. Y0 is a 219/219-byte-identical pinned replay,
  not an independent realization; distinct ZR/ZP checks are pending.
- **W0115 station WS10: OPEN.** Non-data-chosen whole-run RMSE excess
  +0.027 m/s versus S 0.018; CI [0.015,0.042]. Late h48–72 is CLEAR.
  T2/RH2*/WD10 are CLEAR; data-chosen scopes are separately disclosed.
- **W0408:** raw D6 PASS, all 73 frames/domain. Two d02 QGRAUP exceptions
  have identical cells/values to pre-W2, so the manager's boundary-trace
  judgement applies. Canopy trace CPU 1.65e-6 mm/GPU zero lies at WRF's
  1e-6 zeroing threshold; interception is active and snow-burial key inert.
- **0227 afternoon WS10: NOT REPRODUCED.** Fresh XC +0.122 versus S
  0.452; realization-class disposition, no named code cause.
- **W2 Monica/Swiss:** Monica diagnostic available with E41 HARD FAIL;
  W2 Swiss cancelled. Old identity/benchmarks stay on
  69d8a9a7d. The new [W2 bands](docs/release/evidence/v033/w2/index.html)
  cover 657 paired domain-hours. This source-dated candidate was held.
- **W2 Monica E41: HARD FAIL [M].** Two-domain fused d01 contains three
  STACK results 664/520/648 B, above 256 B. Their small measured runtime
  share is not a waiver. W3 must compose a reviewed fix and revalidate
  every executable; its attribution and source are pending.

- **W2 0227 B/T2: one-realization class [M].** R32-65 is 8/8 CLEAR, but
  gate A is DAMAGE on IC R (+0.0148 vs S 0.0109); IC P is −0.0038,
  NOT REPLICATED → CLEAR under the frozen pair rule. One realization per
  IC; W3 requires a fresh reading, not an inherited clearance.

## Archived pre-W2 findings

- **BD92(c): KNOWN RESIDUAL**, not zero/fixed. Binding d01 diagnostic maximum
  0.0136 K; the non-binding 0227 d03 h67 −0.0596 K departure remains. Clear-night
  cooling aloft is a v0.3.4 roadmap item.
- **0227 model-side cloud/wind defect: fixed in the final default set [M].**
  MYNN step-one/two, F1'', R32 and whole-run gates pass. Observational R32-60 is DAMAGE [M] in A/WD10, B/T2 and B/RH2*;
  the station defect remains OPEN and tag/push is held.
- **MYNN predictor floor: OPEN.** WRF applies plume `s_aw` floors in TKE
  prediction; the held port omits that path. The upper variance-face extent
  also differs. Default-off fixes are under pristine-oracle review.
- **SWDOWN history on slopes: OPEN.** With slope_rad=1 the held port writes
  slope-normal SWNORM as SWDOWN; WRF restores horizontal flux after the LSM.
  This output-only convention fix is being validated; forcing is unchanged.
- **72 h D6:** 0227 d03 retains five raw failures. Original CPU-WRF's IC twin
  breaches the same fields/leads with larger errors; frozen floor annex accepts
  them. Other final cases are raw clean. Strict 24 h limits are unchanged.
- **Integrity:** all frames are scanned. Trace annex and two manager judgements
  explain recorded flags without changing caps: frozen-dew canopy placement
  (CPU twin reproduces one of two cells) and parent-boundary graupel traces.
  These are disclosed exceptions, not blanket clean-output parity.
- **Native code:** hard stack coverage passes; conservative CALL/kInput flags
  remain, runtime-bound. Compact AOT kInput metadata is UNKNOWN, not zero.
- **Scope:** the development Noah-MP glacier runtime is CPU oracle and short
  coupled-smoke qualified; GPU and Swiss 24-hour qualification remain pending.
  Universal physics-option parity and MPI/multi-GPU domain decomposition
  remain unimplemented/unvalidated.
  Original CPU-WRF restart compatibility is not claimed; GPU checkpoint/resume
  and WRF-style history are separate capabilities.
- Existing speed/energy and data-centre extrapolations remain explicitly
  v0.3.2 evidence. No new whole-machine energy or four-core Swiss CPU energy
  measurement is claimed.

[Current validation and exact annexes](docs/release/V0.3.3.md),
[per-hour statistical bands](docs/release/evidence/v033/final33v/index.html),
and the [verbatim candidate history](docs/release/evidence/v033/CANDIDATE_HISTORY.txt)
keep source scopes and old failures visible. Candidate history is not the final
release verdict.

## W2 source and publication dependencies

The [W2 source inventory](docs/release/V033_W2_FIXES.md) covers urban soil
(overrides/FRZX order), SWDOWN history, MYNN predictor floors, CANWATER
ELAI/ESAI and the **land-only** post-LSM Q2 cap. The old RC k0 warm PBL bias
masked the urban cold term; preserving that compensator is not a fix.
A/WD10 is NOT REPLICATED (roadmap); replicated B/T2/B/RH2* remain the active
fix target in the archived reading. Eight W2 keys are now reviewed defaults;
full release acceptance remains separate. Approved
[history cleanup](docs/release/HISTORY_CLEANUP.md) requires sealed maps,
full-object scans and verified remote refs; credential rotation is unconfirmed.


W2 qualification update (2026-10-07 10:18Z, source471): the distinct ZR realization leaves two REAL V10 rows at h42/h44; ZP has zero REAL rows. Frozen Z class is PROPERTY, IC-R-specific, and the original W0115 FAIL is retained. R32-67 v3 confirms SYSTEMATIC station WS10 damage in A/B/C (13 CLEAR, 3 SYSTEMATIC, 0 OPEN); B′ remains CLEAR. Earlier OPEN statements above are dated readings, superseded by this result. W3 must apply the affected original-CPU and station gates on its final source; no W2 waiver or inherited clearance.

## Noah-MP nondefault controls: compiled precision and qualification

The u2-land v0.4 development controls are opt-in and CPU original-source qualified; Whole-executable GPU resource/E41, performance and full coupled-forecast D6 remain pending. CPU deviceless component resource scans and short coupled integration/restart smokes are recorded separately. Literal REAL4 operation-order gates match pristine WRF byte for byte. Compiled differences are compared separately with independently frozen input-ULP envelopes, without widening nominal tolerances. The following small excesses are disclosed; final D6 decides forecast acceptance.

| Boundary | Compiled envelope excess | Practical scale |
|---|---|---|
| Existing Jordan snow default | 3/765 values, maximum5.96e-8 | New BATS/SNF2–5 choices all within; old default program retained |
| BTR/RSF surface choices | 64/480 outside | Maximum raw RSURF difference0.00390625s/m at large resistance; at most7ULP |
| Monthly phenology | 9/288 outside, maximum5.96e-8 excess | Input-LAI7–9 and dynamic2/5/6 gates all within |
| RAD1 geometry | 2/432 outside, maximum2.032e-43 | Subnormal WGAP only |
| SFCDIF2 canopy drag | 9/5400 outside, maximum7.45e-9 | All Jarvis values within |
| Dynamic carbon trajectories | 29/1920 outside, maximum1.526e-5 excess | Maximum raw pool difference0.00024414g/m2 |
| Equilibrium water table | 6/192 outside, maximum4.547e-13 | RUNSUB in mm/s |
| Prognostic groundwater alone | 98/1440 outside, maximum1.4305e-6 excess | Maximum raw WT difference0.00048828125mm |
| Complete prognostic soil water | 20/240 outside, maximum2.6226e-6 excess | ZWT/QIN/RUNSUB |
| Saxton–Rawls pedotransfer | 28/576 outside, maximum4.768e-7 excess | Maximum raw coefficient difference1.907e-6 |
| VIC/XAJ/dynamic VIC surface | 12/480 outside, maximum5.684e-14 excess | VIC/XAJ all within |
| Full runoff/tile soil solver | 46/2496 outside, maximum8.2784e-7 excess | Maximum raw difference1.5423e-6; active Hooghoudt both permeability modes all within |

Irrigation, SST skin, thermal/freezing, Liu crop and crop phenology gates are all within their independent compiled bounds. Receipts and source/input hashes are under `proofs/u2_land` and the lane's retained `LAND*_precision.json` artifacts. These are component and caller results, not GPU-vs-GPU fidelity claims.

Dynamic VIC has a separate material upstream initialization sensitivity: original unassigned YD reads change52/480 outputs under NaN versus zero compiler initialization, maximum0.0026633055. The port mirrors the binding zero policy at those exact sites and labels run8 experimental. See [upstream defects](proofs/u2_land/UPSTREAM_WRF_BUGS.md). Liu crop requires valid active crop calendar/season-GDD geography; crop fractions in a crop-disabled wrfinput do not establish that those maps exist. Missing active maps are refused. The approved independent glacier model supports opt_gla1/2; glacier SNF5 is undefined in WRF and refused. MMF run5 is a separate lane, and GECROS2 retains main's scientific refusal.

Native RAD1 uses REAL4 source operations (source jaxpr fp64 outputs0). CUDA libdevice tanf/cosf emits6 fp64 conversion/multiply instructions in its generic slow range-reduction branch; finite canopy angles are bounded below π/2, so that branch is unreachable in the qualified regime. The original trig expressions are retained. Component resources:72registers,STACK32B,LOCAL0; runtime GPU execution remains pending. Other new native phases haveSTACK0/LOCAL0; RUN1 preserves the explicit WRF S_NODE DOUBLE island.
