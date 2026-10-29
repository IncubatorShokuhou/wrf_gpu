# V045 MYNN options — CPU gate ready for independent review

The [F1/F2 follow-up](F1_F2_FIX.md) supersedes the ambient test-mode and cache/launcher
packaging below: explicit isolated CPU conditions, per-PDF private caches, retained
native export/E41 wiring,68 unique ordinary-CPU passes and21,247-node clean collection.
The reviewed model source and original numerical receipts are unchanged.

Runtime source tested at `a1b8fe2e14bb47877868e141705cac4748752685`, src tree
`336d5638bffc705ded69b3115636137957fcbe07`. GPU execution and coupled qualification
remain pending. No native speed or forecast-skill claim.

`icloud_bl=0` and `bl_mynn_edmf=0` were already bound on base `3b93841d70`.
The first two lane commits audit those implementations; they add no duplicate
runtime paths. The new implementation is `bl_mynn_cloudpdf=0/1` in both native
and legacy column routes, with optional previous-call moments, fresh initialization,
RK propagation, static binding and restart selection/missing-field refusal.
Defaults retain CASE(2). Non-MYNN, cycling, stochastic and MYNN-history combinations
with the new PDFs are refused. Multiple per-domain PDF values are refused because
WRF declares one shared scalar.

| Gate | Evidence [M unless stated] |
| --- | --- |
| Original WRF operator | Fresh recapture of `mym_condensation` from unedited pristine objects on 384 real WN3 night/Swiss day columns, 44 levels, both PDFs. QC/QI/CF/SGM bytes equal the frozen original reference. Full compiled QC/QI/CF satisfy the unchanged bounds; sigma has its separate operator check. `receipts/RECAPTURE.json`. |
| Cold moments | 32 real columns, original five-pass WRF QKE/EL/QSQ/TSQ/COV witness. Unchanged .2% + 1e-12 check for the selected extension; release REAL/no-FMA settings are explicit. `check_init.py`, final CI, `receipts/INIT_METRICS.json`. |
| Binding / lifecycle | Final 56 tests pass, including actual runtime source/legacy/EDMF-off forwarding, selected allocation, column/RK return, fresh/restart authority and checkpoint/pickle/NetCDF round trips plus selection refusal. `receipts/MO12.xml`. |
| Deletion sensitivity | Three moment lifecycle deletions, three mathematical operand/precision mutants and the compiled reciprocal-fold deletion are rejected by assertion gates. No tolerance change. |
| Default identity | Native-first, full constants: canonical+constant hashes, d01/force-down/d02 raw HLO and all Triton bodies including locations IDENTICAL to base; BLIND empty. Measured at `316a71d643`; source-proof carry to the final runtime is limited to four restart-only IO files and the ESAT helper used only by new PDFs. The entire default SGS prefix and other hot files are byte-identical. `receipts/OFF_COMPARE.txt`, `DEFAULT_SOURCE_REUSE.json`. This is regression evidence, not WRF fidelity. |
| Archive gate | 21,235 tests collected, zero errors, immutable slim archive with 66 SHA-bound import-support files. No exclusions or fabricated skips. |
| Exact-call launcher | Actual `night_seq_v9.sh SEQ_SMOKE=1`, same path/args/caps, component step rc0, GPU processes0. Receipt is parsed; outer sequencer rc alone is insufficient. `receipts/SEQ_SMOKE.json`. |

The manager explicitly admitted the existing WN3/Swiss evidence instead of a new
WN2 PROD capture (2026-10-27 23:08Z). Cases are labelled as WN3/Swiss. The affected
operator was freshly recaptured after the ESAT source change; the exact-source
receipt is scoped accordingly.

The sequencer smoke exposed a real compiled-only failure missed by the unfused
oracle: constant division in ESAT's liquid/ice blend became reciprocal multiplication.
Both PDFs had 10 CF, 2 QC and 2 QI misses. One opaque REAL divisor fixes it without
adding fp64 arithmetic: worst CF error 8.67248e-6 → 1.19209e-7; worst frozen-bound
ratio 14.544 → .496705. The compiled CI and deletion mutant now protect this site.
The default CASE(2) does not call this helper. GPU rounding is an explicit native gate.

Performance/memory: default mathematical work and array arguments are unchanged
by the source/IR proof [I]. Selected logical storage is two REAL profiles for case0,
one for case1 (`8/4 * nz*ny*nx` bytes); d01 is 2.82/1.41 MiB. Initializer outputs are
added only for selected fields, using the existing bounded tiles. Snapshot/checkpoint
payloads can grow. No new host transfer or callback is introduced inside a timestep.
The opaque scalar may affect native fusion/launches, so actual native cost is unknown.
The CPU-only component smoke measured 0.454/0.445 ms per 384-column call, scratch
397,824/331,776 bytes; these are not GPU or full-MYNN numbers. Case0's existing
DOUBLE variance island is WRF-declared; case1 and the ESAT fix use REAL arithmetic.

The pre-registered [GPU plan](GPU_PLAN.md) requires component original-WRF gates,
actual E41, matched full-root launch/transfer/memory/timing and final RW1–RW4 on the
composed candidate. The current launcher is component-only and does not close these.
Recreate the source-pinned archive before any future plan use; temporary archives,
pytest directories and caches have been removed. No GPU lease was used.
