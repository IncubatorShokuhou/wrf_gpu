# MERGE — lane/u2-land d94eb98ae

Same critic, fix-delta closure of R1–R4 from the original verdict.
Candidate: `d94eb98aefd1405c8c0b3ad49047fc061b567e56`.
Review scope remains CPU experimental component/caller qualification. GPU execution, whole-executable E41, speed, peak memory, full-forecast D6 and station gates remain pending.

Independent affected tests: **74 PASS, eight explicit known compiled XFAIL, zero failures/errors/unintended skips**, 665.42 s on CPU 2/nice 19. JUnit, log, summary.

- **R1 closed:** all five changed requested selections and stateless DVEG 8→9 with identical table hashes refuse through the public checkpoint API; matching full-carry restores remain byte-exact. Existing table/map validation stays in place.
- **R2 closed:** both reduced readers refuse all five active option groups and identify the full-carry reader. Legacy/inactive reads pass.
- **R3 closed:** three actual Swiss loader tests cover DVEG 7/8/9, eight natural LAI threshold cases and four nonvegetated categories each. Cold LAI/XSAI now match WRF initialization. Evolved subthreshold LAI remains unclamped and restart preserves it.
- **R4 closed:** crop REAL/cache scope and the release-prefix check pass. All 24 advanced-water literal modes retain 2,496/2,496 BYTE parity. Sixteen compiled modes meet unchanged original bounds; eight remain XFAIL only after their raw excess indices and maximum exactly match the published receipt. No tolerance or frozen bound was widened, and XFAIL is not a fidelity PASS.

Independent source audit confirms exactly three production-file changes: checkpoint readers, the added cold initializer, and its loader branch. All other production sources are byte-identical to the reviewed candidate. Removing only the new option-specific loader branch restores the complete old loader AST; removing only the new helper restores the existing initializer module AST. Checkpoint writers/restoration functions are unchanged. All 45 final source seals, 53 original receipts, eight fix receipts, 20 original references and the ULP auxiliary authenticate.

The refreshed B smoke is bound to the final source: three steps, zero nonfinite values/guards, 185 restart leaves BYTE. Producer Swiss `f672a573…` and PROD d01 `6fc715a2…` official identity receipts authenticate; these are trace/code-identity results. Their receipts, unchanged hot code and default-loader AST proof justify reuse. Original A/C/D, numerical/native and genuine crop-deletion evidence are carried forward; no duplicate forecast or mutant run was needed.

Performance/memory assessment: reader checks execute only at checkpoint reads; the NumPy LAI/XSAI replacements execute once during selected initialization. No new carry fields, timestep operations or host transfers were introduced in the hot path. Prognostic bytes are now explicitly distinguished from total option memory, including 524 B/cell crop parameter maps by schema and the additional 784 B/cell canopy pack. GPU cost remains unmeasured. The unpaired CPU B-arm time does not establish a speed regression or speedup.

Reviewed lane untouched. Original verdict/evidence preserved. Critic snapshot, pytest data and temporary scratch removed. **R1–R4 are closed; the manager may merge this candidate within the stated qualification scope.**
