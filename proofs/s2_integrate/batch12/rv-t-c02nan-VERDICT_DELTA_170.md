# MERGE — 170b15cacab17dde72b9f64c24d85a13472d893f

Manager-requested delta review against `5c0e7c11ed1a6d0ed075febb054f4dcca6b1f081` only. The original `0150b853f` FIX verdict and its evidence remain preserved.

The required GBM UST writeback is complete. The critic's unchanged numerical assertion now passes: retained real c08 column 0 returns **State.ustar = kernel UST = pristine REAL4 UST = 0.10000000149011612 m/s**, replacing the old 0.013878672383725643 m/s. All kernel rates remain finite. The new test preserves incoming REAL4 values with `_cast=False`, checks the independently verified original reference and asserts the carry dtype; no tolerance was added or relaxed.

Independent focused suite: **19 passed, 25.19 s**, including the 17 SF1 tests and two existing YSU tests. Independent deletion of the UST output writeback makes the new regression fail by numerical array-equality assertion, with zero setup errors. It retains the correct kernel UST and restores only the discarded State writeback, so the failure is specific to the fix.

Delta seal: exactly three paths changed: `scan_adapters.py`, the focused test, and its tracked real-column fixture. Removing the single `ustar` keyword from the new GBM result reproduces the entire old model AST exactly. All other source files are byte unchanged. The fixture inputs exactly match the previously retained real c08 column, and its reference exactly matches the critic's independently executed unchanged-pristine GBM output. No GBM cold-init or microphysics research changes are included.

The earlier independent c02 full-24h rescore (**25 frames, 318 numeric/62 bounded, PASS, zero NF**), default Swiss f672/PROD6fc identities, YSU oracle/mutant and satellite checks carry forward by the exact source/selector proof: c02 uses BL1 and the release defaults use BL5; neither executes the changed BL12 function. No duplicated GPU or default numerical run was performed.

Performance/memory: the change reuses an existing GBM 2-D UST result and existing State USTAR slot; no second physics solve, host transfer, new State field or new table input. Shape/dtype conversion may affect the changed GBM artifact's layout/launches [U]; this review makes no measured speed/peak-memory claim. Default/c02 programs are unaffected [source/selector proof]. Existing six c02 E41 kInput flags and GBM's 18 compiler/input-ULP mixing residual failures remain disclosed, without waiver. c08/c64 microphysics and the reported GFS/MRF UST follow-ups are outside this explicitly requested delta.

Verdict applies to the SF1 source merge, not overall release qualification. CPU-only, one sequential job, MemoryMax 6G/MemorySwapMax 0, pinned 16–19/nice19; no GPU. Evidence: `delta_source_seal.json`, `delta_ust_probe.json`, `delta_tests.xml`, `delta_mutant.xml`, logs and scripts under `<USER_HOME>/wrf_gpu2_lanes/rv-t-c02nan/`. Temporary snapshot and pytest directory removed after the last consumer.
