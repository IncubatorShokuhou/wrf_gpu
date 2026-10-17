# v0.4.0 — expanded physics and forecast tools

**Release date: 2026-10-17.**

**Default path frozen — 2026-10-16 11:59:18Z.** Main `5c15881ede`
(`403240f257` adds proof only) carries the caller-audit repairs and device-only
C5′ MYNN default. Release identities are Swiss **3356e249** and PROD d01
**196aed9f**. F20 plus C5′ original-CPU 24h default gates are closed; the
FINAL composite checkpoint is **66/66 accepted-scope scientific D6 PASS (65 full24h + c45 accepted6h)** at00:26Z on17October.
Source and annotated tags are published. Historical f672a573/6fc715a2 receipts remain pre-F20 baseline evidence.


**Released source — assets awaiting owner authentication.** Defaults and their
identities are frozen; all66 registered scientific scopes pass.
The numbers below are a sealed00:26Z checkpoint; c11 now has a changed-source full24h PASS.

## Published source and asset distribution

- Private annotated `v0.4.0` tag object:
  `485deb27d1ee3520b2e86a3fa8360b5e3e144281` →
  `7bb5ae669b2e4b863d6bafdd659a03fea026004f`.
- Public annotated `v0.4.0` tag object:
  `4d91b7dff2fdabb6cde15232b002df9c8c786b74` →
  [`99a9e8e6571c361a7230cac8ba14dbeda16b62eb`](https://github.com/wrf-gpu/wrf_gpu/tree/99a9e8e6571c361a7230cac8ba14dbeda16b62eb).
- The prepared [GitHub release v0.4.0](https://github.com/wrf-gpu/wrf_gpu/releases/tag/v0.4.0)
  distribution contains **20 assets plus `SHA256SUMS`**, whose SHA-256 is
  `95a772ffb1ad23ca38ee2370945293210f3b3586ef3dbfe7c32314d227190b4d`.

**The GitHub release object and asset upload are blocked on owner
authentication.** The source and tags above are already published; the assets
are prepared and checksum-verified but are not yet downloadable from that
release. Runtime tables larger than the repository limit are fetched from
`releases/download/v0.4.0`; public remote-table fetches fail until the owner
creates the release and uploads its assets. A matching prefilled local table
cache remains usable. No credentials are included in the source or these notes.

## New physics and model capabilities

The v0.4 development series expands the WRF 4.7.1 physics coverage across
microphysics, cumulus, boundary layer, radiation and land/surface schemes. It
does **not** support every WRF option or every combination. The
[per-option support matrix](SUPPORT_MATRIX_v0.4.0.md) distinguishes
default configurations, source-specific GPU validation, experimental paths and
refusals; the [scope inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json)
also records API-only, CPU-only and deferred options.

- **Microphysics:** NSSL variants, Milbrandt–Yau, Ferrier/Ferrier–Aligo,
  Goddard four-ice, Thompson hail, CAM-MG, P3 variants, Jensen–ISHMAEL and NTU.
  Species transport, root boundaries, history and restart support depend on
  the selected scheme. Extra-species nesting and some diagnostic producers
  remain restricted. Evidence: [integration packet](../../proofs/s2_integrate/batch15/QUALIFICATION.json),
  [current selector inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- **PBL/surface and cumulus:** QNSE and TEMF pairs, EEPS/KEPS,
  SAS-family options, old Kain–Fritsch, CAM-ZM, KF-CuP and MSKF. WRF pairing
  restrictions still apply; SAS code 4 remains refused by the ARW CLI.
  Evidence: [selector and qualification rows](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- **Radiation and land:** CAM resource repairs, Goddard radiation,
  CPU-only FLG/GFDL, RUC CLI inputs, SSiB, lake and ocean mixed-layer paths;
  expanded Noah-MP options, glacier physics, irrigation, crop and
  groundwater paths within their prepared-input limits. `sf_surface_mosaic=1` is refused. Evidence:
  [land qualification](../../proofs/u2_land/QUALIFICATION.json),
  [SSiB scope](../../proofs/s2_ssib/README.md),
  [radiation scope](../../proofs/s2_flggfdl/README.md),
  [surface/ocean qualification](../../proofs/s2_oml/QUALIFICATION.json).
- **Optional model features:** DFI modes, analysis/observation/spectral/surface
  nudging, online weather diagnostics, station time series, regional streams,
  stochastic forcing, Fitch turbines, passive tracers, urban LCZ and two-way
  nesting. These are separately selected and qualified features; their
  implementation does not establish arbitrary composite forecast fidelity.
  Evidence: [feature rows and receipts](SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows).

## Forecast tools beyond WRF output

| Workflow | Delivered scope and evidence | Limit |
| --- | --- | --- |
| Forecast products | `gpuwrf post`: cloud, precipitation, wind/height, thermal, instability, helicity and freezing products; [definitions](../post.md), CPU fixture/source receipts in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) | Product assumptions and supported precipitation schemes apply; this is post-processing qualification. |
| Application products | Wind/PV power, radiation decomposition, icing, fire-weather, frost and fog/visibility products; [definitions](../post_apps.md), `apps` receipt in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) | Diagnostic risk/power estimates; no new operational forecast-skill claim. |
| Station verification | Observation ingestion, matched forecast/observation scores and reports; [guide](../station_verification.md), `stations` receipt in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) | A verification engine is not evidence that the newly added schemes improve station skill. |
| Ensembles | Probabilities, quantiles, spread/skill, CRPS/reliability and physics/lagged/multi-date preparation; [guide](../ensemble.md), `ensemble` source/CPU receipts in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) | Final workflow GPU qualification: **TBD-VALIDATION**. Multi-GPU scaling remains unmeasured. |
| Case preparation | Case doctor, reproducibility manifests, open-data onboarding and custom terrain/land-use checks; [doctor](../doctor.md), [onboarding receipt](../../proofs/u1_onboard/receipt.json), `doctor` receipt in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) | Capacity advice is an estimate; container/region GPU end-to-end coverage: **TBD-VALIDATION**. |
| Export and radar | CF-NetCDF/Zarr/GRIB2 subsets and cropping; `formats` CPU qualification in the [snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). Synthetic radar products: [qualification](../../proofs/u2_radar/QUALIFICATION.json) | Optional dependencies and documented scattering assumptions apply; no new radar-data assimilation. |
| Regional optimisation and cycling | Regional physics-selection/bias-correction framework and soil/snow cycling; [optimisation qualification](../../proofs/u2_optim/QUALIFICATION.md), [cycling evidence](../../proofs/u2_cycling/README.md) | Held-out forecast-skill campaigns and a GPU optimisation demonstration: **TBD-VALIDATION**. |

## Frozen defaults and results so far

| Program | Frozen release identity |
| --- | --- |
| Swiss | `3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57` |
| PROD d01 | `196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e` |

F20 compared WN3-0227/0502/0614, PROD and Swiss with original CPU-WRF over
24h:300 paired domain-frames and bounded D6 PASS [M]. The device-only C5′
replacement was then measured on WN3-0227 (75frames), PROD (50) and Swiss
(25): full D6/inventory/NF0 and actual E41 PASS3/3. Its composed main source is
byte-identical to approved353a85f8a. The WN3-0502/0614 measurements remain
F20-source evidence; no extra C5′ run is claimed for them.

C5′ host-transfer counts/bytes, synchronization and callbacks equal F20.
The accepted matched nsys PROD root interval is58.05 versus58.89ms (−1.4% [M]).
The discarded ABAB recipe has zero valid samples; this is not warm ABAB,
whole-run S1, energy, or an inferred peak-memory improvement.
[Default closure and identity proof](../../proofs/s2_integrate/batch62/gate_WN3_closed/GATE_CLOSURE.json),
[sealed F20/C5′ receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json).

At the00:26Z final checkpoint, **66/66 registered FINAL composites pass their accepted
scientific scope [M]**. The raw aggregate is33 PASS,32 FAIL
and1 INCOMPLETE because raw resource verdicts remain distinct. c89 has full
D6/inventory/NF0 but native E41 UNKNOWN; its separate offline STACK1240B scan
remains disclosed. The66 accepted scopes comprise65 full24h cases plus c45 accepted6h.
Every measured source, option tuple and incomplete row is in
the [66-case FINAL table](FINAL_COMPOSITES_v0.4.0.md). All measured sources and source-bound reuse/rerun receipts remain explicit. F2 remains
separate; unbounded fields are REPORT_ONLY.

[Caller-audit/class-A and default fixes, batches38–62](FIXES_v0.4.0.md),
[per-executable E41 disclosures](E41_LEDGER_v0.4.0.md).

## Fixes found during qualification

- **Moisture and number transport:** repair omitted WDM cloud/CCN and Morrison
  snow/graupel number transport, keeping rain-number transport single-counted.
  Plain moisture advection and omitted-key WRF defaults were also repaired.
  [Issue record](../../KNOWN_ISSUES.md#non-thompson-precipitation-and-wdm-caller-defects),
  [development support evidence](SUPPORT_MATRIX_v0.3.4.md).
- **Microphysics caller and precipitation:** pass dry thermodynamic inputs
  where WRF expects them and restore moist theta afterward; correct WDM water
  masks and overlapping frozen-precipitation accounting. Existing incorrect
  accumulated totals cannot be repaired by resuming an old checkpoint.
  [Caller evidence](../../proofs/t_thetafix/README.md),
  [precipitation composition](../../proofs/t_mprain/COMPOSITION.json).
- **Hail mass coupling:** the corrected source `627a03328` was rerun for NSSL,
  NSSL+Grell, WSM7 and WDM7. Each has full hourly 24-hour bounded D6 and complete
  CPU history inventory, but each failed E41 at that precursor source and retained open F2 events. Later
  source-specific results are in the provisional support matrix.
  [Final quartet receipt](../../proofs/v040/gpuqual_qhmass/QUALIFICATION.json).
- **Nonfinite and compiler failures:** repaired original-caller number
  writeback/initialization paths and bounded the QNSE/P3 native compiler
  structures. MP52 now has an actual 24-hour GPU run and E41 PASS at its
  measured source; newer full P3 50–53 inventory/D6 receipts are in the provisional ledger.
  The disclosed native precision deviations remain; c21–c24 now pass the
  registered FINAL D6/inventory/NF0 and E41 gates on5c15881ede.
  [MP52 receipt](../../proofs/t_p3compile/QUALIFICATION.json),
  [QNSE successor evidence](E41_OPT_IN_v0.4.md#c26-qnse-actual-native-resources).
- **History and restart:** retain additional actual scheme diagnostics,
  strengthen active-state schema checks and repair host writer routing.
  [Writer/P3 retention](../../proofs/t_writer_p3_radar/QUALIFICATION.json),
  [ice diagnostic definitions](../physics/ice_diagnostics.md),
  [restart upgrade guidance](UPGRADE_v0.4.0.md).

## Refusals and remaining repair gates

CAM-MG (`mp_physics=11`) uses the merged stock-Triton3.6.x route with
explicit compiler/device preflight. Its native STACK1240B remains E41 FAIL;
registered-source c89 full24h D6/inventory/NF0 now passes, while native
E41 is UNKNOWN and the separate offline STACK scan remains FAIL. Its original RAINNC D6
failure and later DLF2-only incomplete inventory remain preserved. The DLF2
writer now retains WRF's initialized +0 in the admitted shallow-off branch.
Smagorinsky (`km_opt=3`) is GPU-refused by the integrated guard after a
nonfinite Swiss GPU run and failed original-CPU references.
[CAM-MG evidence](../../proofs/t_cammgws/README.md),
[GPU admission integration](../../proofs/s2_integrate/batch45/QUALIFICATION.json).

The earlier WSM5/ACM2/Grell c05 RAINNC failure remains failed. Final-composition
source04cb09f816 passes strict24h D6/full inventory/NF0 with E41 FAIL.
Class-D initialized/held/radar producers are now merged, with active green-roof,
PV and aerosol branches refused. Jensen radar and NTU QHAIL writers are merged;
c87 now passes full24h D6/inventory/NF0 with E41FAIL; c88 now passes full24h D6/inventory/NF0 with E41FAIL; c11 now passes changed-source full24h with E41FAIL. Old incomplete inventories remain historical.
CPU producer proofs do not qualify a GPU trajectory or its resource cost.
[Producer scope and limits](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer).

Default WN3/PROD/Swiss F20 programs must pass E41. Under the manager's04:36Z
policy, opt-in composites require strict D6, full inventory and NF0, and disclose
**E41 FAIL (perf/robustness), D6 PASS** when those gates pass. E41 thresholds
and raw verdicts stay unchanged; F2 remains separately justified. The
[E41 inventory](E41_LEDGER_v0.4.0.md) preserves source scope.

**Nested not validated:** an admitted option with single-domain-only evidence
does not acquire nested qualification. This applies to the listed advanced
surface/PBL/cumulus paths until matched evidence exists; actual pairing and
nested refusals remain authoritative. Surface mosaic remains refused.
[Support matrix](SUPPORT_MATRIX_v0.4.0.md).

## What the validation establishes

The frozen D6 manifest bounds only a subset of inventoried history fields.
Unbounded fields are **REPORT_ONLY**; absent fields make coverage incomplete.
Finiteness, D6, E41 resources, pre-mask F2 accounting, restart and station skill
are separate checks. All numbers below are measured [M] from the cited receipts.

- The new Swiss default glacier configuration has a source-specific full
  24-hour original-CPU-WRF D6/coverage and E41 PASS (c00). WN3 has a source-specific
  full 24-hour comparison over 75 frames across three domains and E41 PASS.
  The historical GPU byte-regression comparison failed separately and is not
  rewritten as a pass. [Source-bound original-CPU receipts](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
- The historical source627 hail quartet has 25 frames per configuration, 62 bounded fields,
  complete inventories and zero raw nonfinite output values. E41 is FAIL on those exact executables; later source-specific scans
  can differ. No global F2 clearance follows. [Quartet summary](../../proofs/v040/gpuqual_qhmass/summary.md).
- Goddard radiation 5/5 is **experimental, validated for a full six hours**:
  seven frames, 346 inventoried fields, no bounded misses or raw nonfinite
  values, and E41 PASS. Its 24-hour attempt has 24 of 25 frames (h0–23);
  h24 was cut by the run cap. The tile-2048 repair supersedes the earlier
  slowdown report. [Accepted six-hour receipt and source binding](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

Registered scientific validation is complete:66 accepted scopes,67 domain
inventories and1656 paired frames. Station campaigns and unrun optional
workflow regimes remain unqualified; GitHub release-object creation and asset
upload await owner authentication.
This release makes no v0.4 speedup, warm S1, roofline or energy claim. Shared cold
qualification walls are not S1 benchmarks. [Qualification scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

## Limits and upgrade

E41 opt-in failures, F2 negatives, original MYNN initialization sensitivity,
P3/Jensen/WDM5 precision residuals, history diagnostic limitations and excluded
MP15/km_opt=3/geographic crop branches remain disclosed in
[KNOWN_ISSUES](../../KNOWN_ISSUES.md#v040-draft-qualification-limits).
Read the [upgrade note](UPGRADE_v0.4.0.md) before using old
checkpoints or selecting new options.

**Pre-F20 baseline identities:** Swiss f672a573 / PROD d01 6fc715a2.
The frozen replacements are Swiss3356e249 / PROD196aed9f (full hashes above).
These identify source programs; the published source commits and tag objects
are recorded above.
The prior Swiss2bf12cb8 is historical.
[Identity/source binding](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

**Frozen program identities: Swiss3356e249 / PROD196aed9f.** Documentation
follow-ups use plain fast-forward commits; the published v0.4.0 tag targets
remain unchanged.
