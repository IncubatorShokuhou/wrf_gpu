# MERGE — t-writer-psfc-hail b43c721e7

Reviewed exact head `b43c721e7dccd9b6dd96385c45fe158f8b41ee55`, product
`e39edfcf57507b90f40b444c03d80c766362d766`, src tree
`c830026e15e01d76d531e93709b2d920ac7f9f98`, against approved parent `9d1cad351`.
Independent critic: lane rv-t-writer-psfc, 2026-10-15. No required fixes.
This admits the CPU-qualified writer correction; GPU/full-forecast gates remain open.

Both full and PSFC-only builders call one host hydrostatic fallback. It delegates
membership and accumulation order to the same `total_moist_mass` as the dycore;
Registry history aliases normalize qv/qc/qr/qi/qs/qg/qh/qi2/qi3. Missing inactive
core slots use scalar zero. Numbers, volumes, aerosols and rime partitions do not
load the air. Selected-source batching includes all nine mass fields, including
rain, and retains one output-boundary device_get. There are no new timestep
kernels, prognostic leaves or restart changes.

Pristine `module_big_step_utilities_em.F:4940–4958` sums every active moist
member in the downward p_hyd_w recurrence. `module_surface_driver.F:1988`
uses its bottom interface as PSFC. Resident C1H/C2H/DNW/p_top and dry column
mass now drive both fallback routes. Held producer PSFC still wins, preserving
WRF's step-start diagnostic; metric-free compatibility extrapolation is retained.
A direct post-step fallback is not proof of the operational diagnostic's timing.

- [M CPU] Independent targeted suite: **95 PASS, 3 unavailable-reference SKIP,
  0 FAIL, 0 ERROR**, 22.28 s; peak RSS 1,765,464 KiB, zero swap. This repeats
  the producer's 92 checks and adds three NetCDF lock consumers. All 26 new
  writer tests pass, including BYTE comparisons against unchanged compiled
  pristine REAL4 phy_prep on active nonuniform columns, aliases, batching,
  producer overrides, compatibility fallback and hail NetCDF publication.
- [M CPU] Deleting hail from the actual writer helper gives **four genuine
  numerical assertion failures and four passing controls, zero setup errors**.
  Both routes and alias forms fail against the unchanged pristine reference.
- [M CPU] Fresh official native-facade identities: Swiss **f672a573**, PROD d01
  **6fc715a2**, exact frozen full hashes. These are regression evidence.
- [M source] Only one product file changed. Three existing functions changed,
  one helper was added, and 57 other functions are AST-identical. Dry pressure
  arithmetic, dtype selection and Kahan/plain vertical reductions are
  AST-identical. Every other product file is BYTE-identical to the approved
  parent; final src/tests/data/scripts match the healthy product commit.
  All 20 sealed producer hashes, two inherited program receipt hashes and
  independently compiled pristine source/body/helper hashes verify. Mutant XML
  transport decompresses BYTE to its original raw receipt.

[M CPU] Independent seven-sample host check at 44×117×267 reproduces all
**55 no-hail full output fields BYTE**. Full output median 11.171 → 11.224 ms
is unchanged within noise; traced NumPy/Python peak drops 80.738 → 61.180 MB.
The producer's full-writer speedup is not independently established. Correct
PSFC-only integration costs 1.946 → 6.162 ms and 34.371 → 55.681 MB peak
(+4.216 ms, +21.311 MB). With hail: full 11.702 → 11.392 ms; selected
2.022 → 5.929 ms. These are controlled output-time measurements, not S1/GPU
speed or forecast memory. Corrected selected-stream PSFC intentionally changes
even without hail when resident hybrid metrics replace the old extrapolation.

[I source] The output-only helper adds no device arithmetic, timestep transfer
or persistent carry. Active optional mass volumes join the existing output
transfer batch. The selected fallback's extra host reduction and temporary
storage account for its small measured cost; it eagerly evaluates before a held
PSFC override, as the full writer already did. No new external baked assets or
keying inputs are introduced. Hot-program identities match; fresh GPU compile,
cached readiness, peak VRAM and E41 were not measured here.

Upstream affected hail GPU/D6/E41 qualification, station/24–72 h/global F2
gates and the three unavailable reference checks remain explicitly open.
Evidence: `REVIEW.json`, `receipts/`, `audit.py`, `identity.py`; MAIN rows RP00–RP07.
