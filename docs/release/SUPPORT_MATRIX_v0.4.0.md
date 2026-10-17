# Scheme support matrix — v0.4.0

**Default path frozen — 2026-10-16 11:59:18Z.** Main `5c15881ede`
(`403240f257` adds proof only) carries the caller-audit repairs and device-only
C5′ MYNN default. Release identities are Swiss **3356e249** and PROD d01
**196aed9f**. F20 plus C5′ original-CPU 24h default gates are closed; the
FINAL composite checkpoint is **66/66 accepted-scope scientific D6 PASS (65 full24h + c45 accepted6h)** at00:26Z on17October.
Source and tags are published; GitHub release assets await owner authentication. Historical f672a573/6fc715a2 receipts remain pre-F20 baseline evidence.


**Released v0.4.0 — all66 registered accepted scientific scopes are complete;
source and tags are published; GitHub release assets await owner authentication.**

Current identities/default closure and registered-source composite results
are below. The [release checkpoint](../../proofs/r_docs040/RELEASE_CHECKPOINT.json)
seals the producer files and exact input/report SHA bindings. Older
[evidence](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) is historical;
the final ledger is complete; source and tags are published, with GitHub release assets awaiting owner authentication.

## GPU requirements

The release Pallas/Triton fast path requires **NVIDIA Ampere or newer,
compute capability ≥8.0**. **Pascal and Turing are unsupported.** This is a
backend prerequisite, not a guarantee of VRAM capacity, per-card validation
or a measured speedup. [Requirement and measured witness](../../proofs/r_docs040/GPU_REQUIREMENTS.json).

On a GTX 1080 Ti (Pascal, capability 6.1), Triton refused the release defaults.
A box-only legacy interpreted path passed the frozen bounded D6 comparison
over three Swiss forecast hours; warm whole-run GPU time was 33.16 s/forecast-hour
versus 9.61 for original CPU-WRF on the same six-core Ryzen 5 5500, **3.45× slower [M]**. These are
debug-box clocks including initialization/output, not release S1, a 24h gate
or evidence of a supported fallback. No Turing run is claimed.
[Source, clocks and scope](../../proofs/r_docs040/GPU_REQUIREMENTS.json).

## Reading the validation classes

- **Default-qualified:** the exact frozen default suite passed the F20/C5′ gates below. A scheme used by that suite does not qualify every combination containing its code.
- **Composite D6 PASS:** the cited original-CPU forecast passed frozen bounded D6, complete registered inventory and NF0. The clock is explicit; c45 is accepted6h, while65 other accepted cases cover24h. Unbounded fields remain REPORT_ONLY.
- **E41-disclosed:** composite science may pass while raw native STACK/non-leaf/kInput resources FAIL. The [per-executable list](E41_LEDGER_v0.4.0.md) retains every measured failure. An UNKNOWN scan stays unknown.
- **CPU-only / API-only:** implementation/oracle scope is explicitly narrower than the native GPU CLI. Backend refusal and API evidence are separate.
- **Refused:** the operational CLI rejects the named selection or pairing. A recognized Registry code is not operational support.
- **INCOMPLETE:** historical partial/failed-to-compile receipts keep that status. c11 now has changed-source full24h PASS; no initial-only output was used to close its gate.

Default WN3/PROD/Swiss programs retain hard E41 gates. The manager's opt-in
rule is strict original-CPU D6 + complete inventory + NF0 with raw resource
failures disclosed. F2 event justification remains separate for every row.

E41 limits remain STACK ≤256 B, non-leaf callee ≤1,000 SASS and kInput transpose ≤500 optimized HLO instructions. See the [exact executable supplement](E41_OPT_IN_v0.4.md); these are not interchangeable units or a runtime slowdown measurement.

The [per-composite resource table](../../KNOWN_ISSUES.md#source-bound-opt-in-e41-disclosure-0026z-final)
and [complete site/receipt inventory](../../proofs/r_docs040/FINAL_E41_DISCLOSURES.json)
cover every FAIL in the00:26Z final producer checkpoint, with final-cohort versus
historical sources explicit. Final-source c03/c05 are **E41 FAIL
(perf/robustness), D6 PASS**; default C5′ E41 is now PASS3/3 at the frozen source. Older raw
INCOMPLETE cases keep that status. These are provisional observations,
not final release counts or arbitrary-combination approval.

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

**Nested not validated:** single-domain evidence does not qualify arbitrary
nests. Retain the documented restrictions for advanced surface/PBL/cumulus
paths; actual pairing and admission refusals remain authoritative.

## Physics selectors

Each row cites one exact composite, not arbitrary combinations. Counts are
measured [M] from the provisional receipt; unbounded fields are REPORT_ONLY.
All66 registered scientific scopes now pass; resource/other disclosed limits remain explicit.


### mp_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no microphysics | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Kessler warm-rain | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Lin et al. (Purdue-Lin) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | WSM3 simple-ice | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | WSM5 | Composite D6 PASS24h (E41-disclosed) | c05/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Ferrier (new Eta), HRW | Composite D6 PASS24h | c62/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 6 | WSM6 graupel | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Goddard 4-ice | Composite D6 PASS24h | c64/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 8 | Thompson | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 9 | Milbrandt-Yau 2-moment | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 10 | Morrison 2-moment | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | CAM 5.1 microphysics | Composite D6 PASS24h (E41 UNKNOWN) | c89/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41UNKNOWN. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 13 | SBU-YLin | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 14 | WDM5 | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 15 | Ferrier-Aligo advected rime | Experimental column-only; excluded | Original CPU-WRF crashes at first output under both initialization conventions. [Exclusion](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 16 | WDM6 | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 17 | NSSL (legacy) | Composite D6 PASS24h (E41-disclosed) | c12/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 18 | NSSL 2-moment 4-ice w/ predicted CCN | Composite D6 PASS24h (E41-disclosed) | c13/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 19 | NSSL one-moment with hail | Composite D6 PASS24h (E41-disclosed) | c14/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 21 | NSSL one-moment without hail | Composite D6 PASS24h (E41-disclosed) | c15/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 22 | NSSL two-moment without hail | Composite D6 PASS24h (E41-disclosed) | c16/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 24 | WSM7 (separate hail/graupel) | Composite D6 PASS24h | c17/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 26 | WDM7 (separate hail/graupel) | Composite D6 PASS24h (E41-disclosed) | c18/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 27 | UDM 7-class | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 28 | aerosol-aware Thompson (water/ice-friendly) | Composite D6 PASS24h (E41-disclosed) | c19/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 29 | RCON (Thompson aerosol-aware, liquid-phase mods) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 30 | HUJI spectral-bin (fast) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 32 | HUJI spectral-bin (full) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 38 | Thompson w/ 2-moment graupel/hail | Composite D6 PASS24h (E41-disclosed) | c65/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 40 | Morrison 2-moment w/ CESM-NCSU RCP4.5 aerosol | Composite D6 PASS24h (E41-disclosed) | c20/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 50 | P3 1-ice, 1-moment cloud water | Composite D6 PASS24h | c21/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 51 | P3 1-ice + double-moment cloud water | Composite D6 PASS24h | c22/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 52 | P3 2-ice + double-moment cloud water | Composite D6 PASS24h | c23/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 53 | P3 1-ice 3-moment + double-moment cloud water | Composite D6 PASS24h | c24/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 55 | Jensen-ISHMAEL | Composite D6 PASS24h (E41-disclosed) | c87/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 56 | NTU multi-moment | Composite D6 PASS24h (E41-disclosed) | c88/5c15881ede: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 95 | Ferrier (old Eta) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 96 | Madwrf | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 97 | Goddard GCE | Composite D6 PASS24h (E41-disclosed) | c61/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### cu_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no cumulus | Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Kain-Fritsch (new Eta) | Default suite; Composite D6 PASS24h | c36/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Betts-Miller-Janjic (BMJ) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | Grell-Freitas ensemble | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | Scale-aware GFS SAS | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 5 | Grell 3D ensemble | Composite D6 PASS24h (E41-disclosed) | c13/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 6 | Modified Tiedtke | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Zhang-McFarlane (CAM5) | Experimental:E41; validated GPU24h D6 | c29/7468605609: rawD6 PASS,frames25,E41FAIL. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 10 | Modified Kain-Fritsch (PDF trigger) | Composite D6 PASS24h | c30/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | Multi-scale Kain-Fritsch (MSKF) | Composite D6 PASS24h | c31/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 14 | KIM Simplified Arakawa-Schubert (KSAS) | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 16 | New Tiedtke | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 93 | Grell-Devenyi ensemble | Composite D6 PASS24h | c17/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 94 | 2015 GFS SAS (HWRF) | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 95 | previous GFS SAS (HWRF) | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 96 | previous new GFS SAS (YSU) | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | previous Kain-Fritsch | Composite D6 PASS24h (E41-disclosed) | c14/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### bl_pbl_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no boundary-layer | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | YSU | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Mellor-Yamada-Janjic (MYJ) TKE | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | Hybrid EDMF GFS (ACM-GFS) | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | QNSE-EDMF (Quasi-Normal Scale Elimination) | Composite D6 PASS24h (E41-disclosed) | c26/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | MYNN TKE | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | ACM2 (Pleim) | Composite D6 PASS24h (E41-disclosed) | c05/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 8 | Bougeault-Lacarrere (BouLac) | Composite D6 PASS24h (E41-disclosed) | c06/db78f9bd8c: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 9 | UW (CAM5) | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 10 | TEMF (Total Energy Mass Flux) | Composite D6 PASS24h (E41-disclosed) | c27/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 11 | Shin-Hong scale-aware | Composite D6 PASS24h (E41-disclosed) | c07/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 12 | Grenier-Bretherton-McCaa | Composite D6 PASS24h (E41-disclosed) | c08/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 16 | TKE + TKE-dissipation (epsilon) | Composite D6 PASS24h (E41-disclosed) | c10/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 17 | TKE + TKE-dissipation + TPE | Composite D6 PASS24h (E41-disclosed) | c11/b823c8ca87: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | MRF | Composite D6 PASS24h (E41-disclosed) | c09/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### sf_sfclay_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no surface-layer | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | revised MM5 Monin-Obukhov | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Monin-Obukhov (Janjic Eta) | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | NCEP GFS surface layer | Composite D6 PASS24h | c04/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | QNSE surface layer | Composite D6 PASS24h (E41-disclosed) | c26/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | MYNN surface layer | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | Pleim-Xiu surface layer | Composite D6 PASS24h | c43/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 10 | TEMF surface layer | Composite D6 PASS24h (E41-disclosed) | c27/2ed8c04c26: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 91 | old MM5 surface layer | Composite D6 PASS24h (E41-disclosed) | c61/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### sf_surface_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no surface temp prediction | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | thermal diffusion (5-layer slab) | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 2 | Unified Noah LSM | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 3 | RUC LSM | Composite D6 PASS24h | c33/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 4 | Noah-MP LSM | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | CLM4 (Community Land Model v4) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 6 | CTSM (Community Terrestrial Systems Model) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 7 | Pleim-Xiu LSM | Refused by native CLI; API-only | Requires explicit static/land bundles. [Scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 8 | SSiB (Simplified Simple Biosphere) | Composite D6 PASS24h (E41-disclosed) | c40/36fcfaa64d: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |

### ra_lw_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no longwave radiation | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | RRTM | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | CAM | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 4 | RRTMG | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Goddard longwave | Composite D6 PASS: accepted6h | c45/5c15881ede: D6PASS,6h/7frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | FLG (UCLA) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |
| 14 | RRTMG-K (KIAPS) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 24 | fast RRTMG (GPU/MIC) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 31 | Earth Held-Suarez forcing | Composite D6 PASS24h | c39/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 99 | GFDL (Eta) longwave (semi-supported) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |

### ra_sw_physics

| Code | WRF scheme | Provisional status | Evidence / limitation |
| --- | --- | --- | --- |
| 0 | no shortwave radiation | Composite D6 PASS24h | c01/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 1 | Dudhia | Composite D6 PASS24h (E41-disclosed) | c02/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 2 | Goddard shortwave | Composite D6 PASS24h (E41-disclosed) | c03/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41FAIL. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 3 | CAM | Validated GPU24h D6 | c28/f5d90ca0d3: rawD6 PASS,frames25,E41PASS. [Source/report](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| 4 | RRTMG | Default suite; Composite D6 PASS24h | c00/04cb09f816: D6PASS,24h/25frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 5 | Goddard shortwave (new) | Composite D6 PASS: accepted6h | c45/5c15881ede: D6PASS,6h/7frames/full inventory/NF0,E41PASS. [Final result](FINAL_COMPOSITES_v0.4.0.md), [SHA/receipts](../../proofs/r_docs040/RELEASE_CHECKPOINT.json). |
| 7 | FLG (UCLA) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |
| 14 | RRTMG-K (KIAPS) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 24 | fast RRTMG (GPU/MIC) | Refused | Outside admitted native CLI inventory; reference/API/build scope may exist. [Catalog](../../src/gpuwrf/io/scheme_catalog.py). |
| 99 | GFDL (Eta) (semi-supported) | Refused on GPU; experimental CPU-only | Native GPU kernel pending. [Qualification](../../proofs/s2_flggfdl/README.md). |

## Other options and workflows

These feature rows describe admitted implementation scope and the specific evidence available. A physics-row PASS does not automatically qualify these additions. Selection/inputs, active-regime coverage and feature-specific comparisons remain required.

| Option | Status / scope | Receipt |
| --- | --- | --- |
| `moist_adv_opt=0/1/2`, `scalar_adv_opt=0/1/2`; `diff_opt`, `km_opt=0/1/4` within supported pairs | WRF transport/default binding repaired; exact combination qualification TBD-VALIDATION. Constant-K paths without native CLI support refused. | [Catalog](../../src/gpuwrf/io/scheme_catalog.py), [binding](../namelist-compatibility.md), [source/scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `diff_opt=2/km_opt=2` LES TKE; `km_opt=5` SMS-3DTKE | Experimental: E41; c38/c37 full24h bounded D6/coverage PASS at dd619e480; extra-scalar/nested/PBL restrictions apply. | [Ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [SMS qualification](../../proofs/s1_km5/QUALIFICATION.md) |
| `km_opt=3` | GPU-refused by the integrated CLI validator before JAX/device work. Historical Swiss GPU nonfinite at1h; Swiss/Madrid original CPU references crashed. CPU/reference scope remains experimental. | [Exclusion](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `dfi_opt=1/2/3` | c49/50/51 source5c15881ede DFI1/2/3 full24h D6/coverage+E41PASS. DFL has its natural24frames, not padded25. Single-domain/forcing restrictions remain; see [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [DFI oracle/evidence](../../proofs/s1_dfi12/), [scope and source](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `grid_fdda=1/2`, `obs_nudge_opt=1` | `grid_fdda=1`: c77/c78; `grid_fdda=2`: c79; all source5c15881ede full24h scientific PASS/E41PASS. `obs_nudge_opt=1`: c55/5c15881ede full24h scientific PASS/E41FAIL disclosed. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Spectral](../../proofs/s1_specnudge/), [observation](../../proofs/s1_obsnudge/), [ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `grid_sfdda=1/2` | c77/5c15881ede scientific PASS, E41PASS; c78/5c15881ede scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Surface qualification](../../proofs/s2_sfdda/), [actual admission ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `sf_urban_physics=1` | c36/04cb09f816 scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Source/input receipt](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `sf_urban_physics=2/3`, `use_wudapt_lcz`, anthropogenic heat/morphology | c56/04cb09f816 scientific PASS, E41FAIL; c57/2ed8c04c26 scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Class-D initialized/held producers are merged; active green-roof/PV refused. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Urban/LCZ evidence](../../proofs/v040/urban_lcz/), [missing-field disclosure](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer) |
| `sf_lake_physics=1` | c83/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Genuine NCAR lake cells/default-depth mode; nested not validated. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Lake evidence](../../proofs/s1_lake/), [configuration](../lake-model.md) |
| `sf_ocean_physics=1`, `isftcflx=1/2` | c84/5c15881ede scientific PASS, E41PASS; c85/04cb09f816 scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. PWP refused; nested not validated. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Qualification](../../proofs/s2_oml/QUALIFICATION.json) |
| Noah-MP `dveg=1–9`, `opt_run=1–8`, `opt_soil=1–3`, `opt_irr=0–3`, other admitted `opt_*`; `sst_skin` | Experimental branch-specific qualification: c68/c69/c70 source5c15881ede select dveg7/8/9, opt_run8 and opt_infdv1/2/3 respectively, full24h scientific PASS/E41PASS. No blanket options pass. Crop/tile/map-selected irrigation needs valid geography and lacks3D validation; sf_surface_mosaic=1 is refused. | [Land qualification](../../proofs/u2_land/QUALIFICATION.json), [exact selector coverage/exclusions](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [MMF inputs](../../proofs/s2_mmf/INPUT_REQUIREMENTS.md) |
| `sst_update=1`, `slope_rad=1`, `topo_shading=1` | c60/5c15881ede scientific PASS, E41PASS. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Selector inventory and WN3 source](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `nwp_diagnostics=1`, `prec_acc_dt`, `do_radar_ref=1`, WRF-Solar diagnostics | Experimental online products; frozen field bounds apply only where present, radar can be REPORT_ONLY. Feature-specific final gate TBD-VALIDATION. | [Online diagnostics](../../proofs/u2_diag/QUALIFICATION.json) |
| tslist, selected station profiles | Implemented CPU-oracle sampling/export; final composite/regime gate TBD-VALIDATION. | [Original tslist oracle](../../proofs/u2_tslist/oracle/manifest.json) |
| Regional streams / window statistics | Experimental; instant/mean/min/max definitions and restart restrictions explicit. | [Qualification](../../proofs/u2_streams/QUALIFICATION.json) |
| `skebs`, `sppt`, `spp`, `spp_conv/pbl/lsm`, `rand_perturb`, `stoch_force_opt` | Experimental stochastic/statistical checks; per-branch final GPU qualification TBD-VALIDATION. | [Qualification](../../proofs/u2_stoch/QUALIFICATION.md) |
| `windfarm_opt=1` Fitch | c46/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Manifest](../../proofs/u2_windfarm/manifest.json), [scope](../wind-farms.md) |
| `tracer_opt=2`, `gpuwrf_tracer_file` | c90/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Tracer values remain REPORT_ONLY; WRF-Chem and two-way tracer feedback refused. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Original active reference/ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [component qualification](../../proofs/u2_tracers/QUALIFICATION.json) |
| `feedback=1`, `smooth_option=2` | c42/5c15881ede scientific PASS, E41FAIL. Exact inputs/pairings and feature-specific limits apply. Raw historical CLAT failure retained; arbitrary nested combinations remain unqualified. [FINAL rows](FINAL_COMPOSITES_v0.4.0.md). | [Two-way CPU evidence](../../proofs/u2_twoway/QUALIFICATION.json), [actual GPU ledger](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| Multi-GPU independent ensembles / domain decomposition | Launcher and CPU virtual-rank evidence; real GPU scaling/efficiency TBD-VALIDATION. | [v0.4 scope/source inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json), [multi-GPU runbook](../../scripts/runpod/RUNBOOK.md) |
| Moving/vortex-following nests, adaptive time step, unsupported WRF-Chem/Fire/Hydro/wave/DA paths, HUJI30/32, CLM4/CTSM, RRTMG-K14/fast24 | Refused/out of this native CLI release scope; recognized reference/build paths do not establish operational support. | [Catalog](../../src/gpuwrf/io/scheme_catalog.py), [scope inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |

## Noah-MP selector admission

This enumerates every selector in the backend-dark `SUPPORTED` table, including
selectors whose only admitted value is their default. Values are admission
ranges, not every-combination scientific approval. **Default-qualified** means
the frozen default tuple; **experimental CPU/oracle/branch-specific composite
scope** applies to nondefault values, with the exact selected GPU cases in the
[FINAL ledger](FINAL_COMPOSITES_v0.4.0.md) and the original
[land qualification](../../proofs/u2_land/QUALIFICATION.json).

| Selector | Default | Admitted scalar values | Validation class / limits |
| --- | --- | --- | --- |
| `dveg` | 4 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_crs` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_btr` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_run` | 3 | 1, 2, 3, 4, 5, 6, 7, 8 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_sfc` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_frz` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_inf` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_rad` | 3 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_alb` | 2 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_snf` | 1 | 1, 2, 3, 4, 5 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_tbot` | 2 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_stc` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_gla` | 1 | 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_rsf` | 1 | 1, 2, 3, 4 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_soil` | 1 | 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_pedo` | 1 | 1 | Default tuple only; other values refused; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_crop` | 0 | 0, 1 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_irr` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_irrm` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_infdv` | 0 | 0, 1, 2, 3 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |
| `opt_tdrn` | 0 | 0, 1, 2 | Default-qualified tuple; nondefault experimental branch scope; [exact admission/pairings](../../src/gpuwrf/io/noahmp_options.py). |

Prepared geography and actual MPTABLE parameters remain required for the
selected branches. BATS albedo needs its seven table coefficients; dynamic
VIC, tile drainage, crop/irrigation and MMF retain their explicit pairing/input
rules. Surface mosaic and active unsupported urban/aerosol branches stay
refused. No absent geography or output field becomes an initialized zero pass.

## Operator commands and workflow classes

These are the actual v0.4.0 command names; every listed help/argument path was
[checked against the CLI](../../proofs/r_docs040/CLI_GUIDE_AUDIT.json). Help
verification establishes the interface, not model fidelity or a new forecast
skill result. Forecast-launching workflows inherit the scheme/composite
classes above. [Usage and input schemas](USER_GUIDE_v0.4.0.md).

| Command | Class and delivered scope |
| --- | --- |
| `run` | GPU forecast; default-qualified or per-composite/E41-disclosed/INCOMPLETE/refused as selected. |
| `namelist-support` | CPU/backend-dark admission catalog; scientific classes remain in this matrix. |
| `doctor`, `manifest`, `reproduce` | CPU offline case/provenance tools; [source/fixture scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). No forecast launched by reproduce. |
| `post` | CPU offline forecast/application products; [definitions and qualification](../post.md). |
| `verify` | CPU offline station verification with explicit optional observation fetching; [scope](../station_verification.md). |
| `ens products/verify/prepare/launch` | CPU products/statistics/planning; launch schedules admitted forecasts. [Workflow evidence](../ensemble.md); real multi-GPU scaling unqualified. |
| `optimize identity/plan/score/run`, `calibrate fit/apply` | CPU/source-qualified regional planning/scoring/calibration; run schedules forecasts. [Held-out study rules](../regional_optimization.md); tools do not establish new skill. |
| `perturb` | CPU prepared-case perturbations; [member preparation scope](../ensemble.md). |
| `export` | CPU offline NetCDF/Zarr/GRIB2 output with optional dependencies; [source/fixture evidence](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| `tslist-export`, `stream-check` | CPU offline station/stream interfaces; [source inventory](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). |
| `cycle` | CPU same-grid land/snow cycling; [qualification](../cycling.md). Distinct from atmospheric DA and crash restart. |
| `quickstart` | CPU onboarding plan/WPS-real preparation; optional GPU run inherits forecast gates. [Inputs/source rules](../onboarding.md). No separate onboard/cycling executable. |

## Qualification limits that apply across the matrix

- F2: zero raw output NaN/Inf does not justify negative pre-mask events or repairs. Final hail receipts do not count qh in their census. [Quartet](../../proofs/v040/gpuqual_qhmass/QUALIFICATION.json).
- Original MYNN rmol cold-start undefined read: use the admitted compiler-ZERO original reference where designated, disclose raw-original spread, retain frozen bounds. [Source/reference evidence](../../proofs/t_p3fix/UPSTREAM_WRF_BUGS.md).
- P3 native strict/FMA residuals, Jensen diagnostic residuals, WDM5 number-kind spread and NTU source-diffusion undershoots stay visible. [Known issues](../../KNOWN_ISSUES.md#v040-draft-qualification-limits).
- The nine formerly omitted class-D fields now have merged WRF-defined producers or held constants; unsupported active branches are refused. Missing fields would still make coverage INCOMPLETE. [Producer evidence](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer).
- The66 registered scientific scopes are complete. Station skill, new warm S1, portability/real multi-GPU performance and unrun optional regimes remain unqualified; source and tags are published; GitHub release assets await owner authentication. No every-combination or later-source PASS is inferred from these rows. [Snapshot scope](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).
