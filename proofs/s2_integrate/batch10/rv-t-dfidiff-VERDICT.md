# MERGE — lane/t-dfidiff

Reviewed candidate `16ac7c8c81fe13b9f027497ac562a300d9da72ea` against
`12f41b29893f61e142b6666203192dd9e0e1306c`. No required fixes.

WRF 4.7.1 DFL uses `diff_opt=0, km_opt=1` in both forward filtering and the
forecast when the DFI-specific controls are omitted. This affects the solver
as well as the header. The existing port already matches; this branch adds
proofs, fixtures, and tests, with no production source change.

- [M source] `main/module_wrf_top.F:2145` selects forward initialization without
  the backward phase. `share/dfi.F:189–190` saves the original diffusion only
  in backward initialization; `:324–327` copies the DFI fields into both grid
  and model configuration. Registry defaults are 0/1 at
  `Registry/Registry.EM_COMMON:2847,2851`. `dfi_fst_init` has no diffusion reset.
  `share/solve_interface.F:29` refreshes solver configuration every call;
  `dyn_em/module_em.F:802,1380` consumes it. DDFI/TDFI call backward
  initialization and therefore preserve original diffusion. The port applies
  `_dfl_diffusion` only to DFL; other modes and all runtime sources are unchanged.
- [M reused original CPU evidence] Six solver rows cover forward stage 2 and
  forecast stage 3, steps 0/1/2, all grid and solver pairs 0/1. All 24 original
  c49 history headers report 0/1. The paired original-control/print-only run has
  5×362=1,810 finite BYTE arrays, and all five whole-file SHA pairs match.
  The audit independently reconstructed the print-only wrapper and matched its
  SHA, checked 10 pristine source/configuration hashes, nine original link-input
  hashes, original executable and case-input hashes, and all 11 packaged
  receipts. Both logs complete successfully without OOB diagnostics; neither
  arm emitted fort.98. Completed numerical work was reused, not repeated.
- [M independent CPU] RD01: 41 PASS, one inherited stale assertion, zero errors,
  8.19 s. All 15 registered TD03 cases PASS, including omitted/explicit DFI
  diffusion, forward/forecast/resume, and DDFI prepass routing. All three
  NetCDF process-lock cases PASS. The one failure is
  `test_catalog_reports_cpu_scope_and_pending_gpu[3]`: its old assertion expects
  GPU qualification pending, while the catalog now records completed TDFI
  qualification. Both the test and catalog blobs are identical to the base;
  this is a pre-existing test-maintenance issue, not a waived candidate failure.
- [M independent mutant] RD02 replaces `_dfl_diffusion` with the original
  namelist 1/4. Omitted-DFI case fails at the intended routing AssertionError;
  explicit 1/4 control passes. One genuine kill, one PASS, zero setup errors,
  1.84 s. No production file was modified.
- [M identity reuse] Candidate/base/measured `e60716432` have identical `src`,
  `data`, and `scripts` Git trees. The hash-bound batch8 qualification and
  original release-identity receipts confirm Swiss `f672a573` and PROD d01
  `6fc715a2`. The probe source SHA also matches. No redundant traces were run.
  Clean `git archive` yields the exact original c49 namelist fixture.

The quirk is recorded in the candidate's
`proofs/t_dfidiff/UPSTREAM_WRF_BUGS.md`. The writer must use the post-prepass
forecast namelist. Raw/staged wrfinput attributes may still contain 1/4 and
do not establish forecast configuration. No model fix or additional GPU
rerun follows from this proof. Planned c49 GPU/D6 qualification remains open.

Performance/memory: no production source, assets, or scripts changed. No added
forecast launches, transfers, carry fields, temporary allocations, cache-key
inputs, compilation, or readiness work. CPU review only; no GPU speed claim.

Evidence: `RD01_tests.log/xml`, `RD02_mutant.py/log/xml/json`, and
`RD05_audit.py/log/json` under `<USER_HOME>/wrf_gpu2_lanes/rv-t-dfidiff/`.
RD03 incorrectly assumed the identity receipt was SI83; RD04 selected a
different test subset by prefix. Both audit-harness failures are retained and
logged. RD05 binds the actual QUALIFICATION SHA and exact registered XML
node set; no model or test rerun was needed. Disposable snapshot and pytest
scratch are removed; caches NONE.
