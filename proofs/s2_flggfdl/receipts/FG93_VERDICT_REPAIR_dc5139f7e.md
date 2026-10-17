# MERGE-WITH-FIXES — lane/s2-flggfdl dc5139f7e

Reviewed `dc5139f7ec7d70479a14d8d4e5e527c82d84ec44` as the fix delta from rejected `0ae2af11d`. Original verdict remains immutable. This verdict accepts the caller correction and the limited experimental CPU-only implementation, **conditional on the two disclosure fixes below before merge**. It does not grant strict compiled fidelity, D6, GPU or release qualification.

## Findings closed

**P1 is closed.** Both solver ABIs now consume the existing merged `lw.qc`/`lw.qi`, with `_rrtmg_pbl_cloud_kwargs` controlling non-MYNN selection. Independent rerun: all eight new seam cases PASS. Restoring the old raw-input mapping in memory causes four genuine numerical assertion failures, with four non-MYNN controls still passing and zero setup errors (`repair_seam_mutant.xml`, 3.41 s).

Independent affected tests: **89 PASS / 17 XFAIL**, 34.93 s, zero errors (`repair_tests.xml`). This includes literal column checks, the eight seams, all 18 hard compiled-regression cases, corruption controls and contraction-gate finite checks. It is a targeted 106-case rerun, not a claim to rerun all producer/shared evidence.

The XFAIL mechanism does **not hide a new zero-output regression**: independently replacing actual compiled IR outputs with zeros leaves the strict diagnostic XFAIL but causes its separate mandatory regression case to fail with a numerical AssertionError (1 FAIL / 1 XFAIL, zero setup errors, 0.32 s; `repair_mutant.xml`). The regression fixture is correctly labelled JAX-vs-JAX, not WRF truth. Narrow bands, strict XPASS behavior and `raises=AssertionError` are preferable to a module-wide waiver; they do not by themselves prove a compiler-bound pass.

## Independent assessment of the 17 deviations

I recomputed the already-preregistered matched-profile/field bound directly from the sealed pristine WRF no-FMA and FMA fixtures, independently of JAX values: native profile Linf spread + original 16 ULP + 3e-6 allowance. The current frozen compiled artifact is exercised by the hard regression tests. Fixture/source identities are unchanged.

- All 17 expected failures are genuine strict deviations: **3,254 values** outside the original strict allowance.
- Seven bands fit the independent source allowance: **SW5/6; IR11/12/13/14/18**.
- Ten bands retain **285 values outside even that allowance**: **SW1–4; IR7/9/10/15/16/17**. Worst source-bound ratio **12.8554** (IR16); maximum absolute band error **0.0013961792 W/m²**.
- Full physical-field evidence carried from the unchanged solver also records an LW TOA error **0.0096130371 W/m²**, versus independent native FMA spread **0.0011901855 W/m²**, and maximum heating error **4.4481567e-7 K/s**. These remain deviations, not source-envelope passes.

Evidence and all per-band numbers: `repair_bound.json`; reproducible host-only calculation: `repair_bound.py`. Native fixture SHA `1b1500bc…`, independent FMA fixture SHA `7803935b…`, compiled regression SHA `148f2396…` (full hashes in the report).

**Policy judgment:** the seven source-bounded cases qualify as compiler-spread evidence. The other ten can only use the standing policy's **small documented excess, experimental/D6-pending** route. Literal no-FMA native BYTE evidence and the unchanged solver support that limited disposition; absolute errors are small, and mandatory regression/finite checks preserve sensitivity. It would be false to call all 17 independently FMA-bounded or to use the frozen JAX result as a physical tolerance. No new solver defect is established by this repair review; the remaining numerical deviations are not strict fidelity passes.

## Exact fixes required before merge

1. **Put quantitative deviations in the catalog notes.** The four FLG/GFDL entries in `src/gpuwrf/io/scheme_catalog.py` are unchanged and still say only “small compiled deviations disclosed.” FLG entries must state the 17 strict failures, the 7 source-bounded / 10 source-envelope-excess distinction, 285 remaining values, maximum band error 0.00139618 W/m², and the full-field 0.00961304 W/m² / 4.44816e-7 K/s residuals, explicitly pending D6. GFDL entries must retain their own already-disclosed 0.0181 W/m² / 7.52e-8 K/s residuals, not substitute FLG numbers. Link the detailed per-band disposition rather than presenting a generic compiler-contraction PASS.
2. **Commit and reference the independent per-band source disposition.** Preserve the exact native/compiled fixture hashes, original bound, per-band strict counts, source-bound counts and maximum errors in a tracked receipt (the critic's `repair_bound.json` is sufficient data). Link this receipt from the XFAIL explanation/documentation and catalog. Explicitly label the ten residual bands as outside the native envelope and allowed only as small experimental deviations. Keep the existing strict assertions and mandatory hard regression/finite controls; no widened native bounds, hidden failures or numerical source changes are required for these disclosure fixes.

These are metadata/evidence changes; no forecast, solver, compile or identity rerun is needed if product and test executable code remain byte-identical. The producer/manager must complete them before merge.

## Reuse and resource assessment

The sole changed product file is the opt-in paired radiation caller; product tree `67cccc25edd57c0016dc10301c0e92c0a0e839eb` matches the repair receipt. Solver/table/IO/default runtime code is unchanged. Earlier independent Swiss f672a573 / PROD 6fc715a2 identities and three-format producer restart evidence carry forward. Both changed native REAL coupled receipts report three finite steps, two solves, held-between-solves BYTE state and zero guards; they were inspected, not redundantly rerun here.

The fix reuses the existing merge. Producer changed-artifact evidence: FLG CPU scratch **5,307,528 bytes (+931,328)**, GFDL **3,515,632 bytes (unchanged)**, compile **34.15/42.06 s**; no causal timing comparison because environments differ. Newly live SGS inputs explain the extra dependencies. FLG f64 compute=0; GFDL=9 within native DOUBLE islands. No GPU/VRAM/forecast performance claim.

Fix-delta `git diff --check` passes. Reviewed lane untouched; all own snapshot/pytest directories removed after the final consumer. Old verdict and receipts remain intact.
