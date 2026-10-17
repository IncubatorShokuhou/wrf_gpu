MERGE — lane/t-writer-clat e80905aec (product 30af97a7e)

Reviewed immutable e80905aec against f737666b6. This closes the host-writer CLAT retention defect. It does not qualify a new forecast.

WRF authority: pristine Registry/Registry.EM_COMMON:1848 marks REAL CLAT as a history field with u=(copy_fcnm). Generated inc/nest_feedbackup_interp.inc:3455-3469 calls copy_fcnm with grid%clat and ngrid%clat. share/mediation_integrate.F:784 and 1043-1044 perform initial feedback when enabled. The existing input-only feedback producer and nested writer constructor/callback already deliver the changed parent field. The full builder discarded it by assigning CLAT=XLAT. Source hashes are sealed in REVIEW.json.

Independent CPU checks:
- 72 passed, 3 explicit missing-reference skips, zero failures/errors (75 collected, 16.037 s); includes all 22 new tests and NetCDF process locking.
- Actual parent-function deletion mutant: 12 genuine CLAT byte-comparison assertion failures, 2 healthy fallback controls, zero setup errors; 8 selected tests deselected. Independently parsed every failure signature.
- Existing input-feedback producer republished through the fixed writer: all 25 original parent plus 25 child CPU-WRF CLAT frames BYTE, max error 0; all 50 original whole-file hashes freshly checked. A second critic control rejects nonfinite values in every original/published CLAT frame before BYTE comparison. CPU history is only a comparison input.
- No-feedback controls on both real c42 input geometries: all fields BYTE against the actual old writer in 8 full/compatibility/selected variants. Full and full-compatibility CLAT also BYTE vs original input. Selected-without-supplied-CLAT retains its inherited absence; selected compatibility retains its inherited default value. These last two behaviors are not new regressions and are disclosed for a separate follow-up.
- Only prepare_wrfout_payload changes executable source. The other 60 writer functions are AST-identical after ignoring docstrings; all other src, data, scripts and packaging are BYTE. Existing feedback/Registry operators are BYTE to VP162. All 17 producer raw/tracked seals and the approved default-identity receipt chain freshly verify.

Default programs: approved FULL Swiss f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d and PROD d01 6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b reused through exact-source identity, including release_identity.py and the upstream probe. No numerical/default trace repeat was needed for this host-only delta (AGENTS validation reuse); no new GPU, E41, or forecast claim.

Cost: independent 120x70x44 host payload medians 3.217 -> 2.756 ms, overlapping ranges; sequential samples do not establish a speedup. Traced peak 17,147,340 -> 17,147,244 bytes; 48 other output fields BYTE. Same NumPy producer object retained, no new device kernel/transfer or timestep allocation; the existing batched output boundary already materializes this producer. CPU peak RSS <=1,255,092 KiB for successful arms. Compile/readiness unchanged by source identity.

Scope limits: c42 capped GPU stream still needs its separately scheduled completion and full D6 scoring. Three missing-reference skips are unchanged and unwaived. Deferred-output tests drain the real snapshot in-process and do not establish thread scheduling. The inherited selected-stream CLAT fallback issue is separate. No tolerances, CPU references, or model sources were edited.

Own admission mistakes are retained without credit: audit count/line expectations corrected to the paid XML; no-feedback probe first assumed CLAT existed in selected output, then assumed its inherited compatibility default equaled input. Final controls measure these inherited behaviors and pass the required old/new BYTE gate. No product fix or repeated healthy/mutant/science arm followed those harness corrections.

Cleanup: one detached snapshot, c42 publication scratch, both pytest scratch directories removed after the last dependent arm; cache NONE, no GPU, no jobs. Report and raw receipts retained under <USER_HOME>/wrf_gpu2_lanes/rv-t-writer-clat. Next: manager/s2-integrate merges e80905aec, then v-prep completes the existing c42 forecast/D6 separately.
