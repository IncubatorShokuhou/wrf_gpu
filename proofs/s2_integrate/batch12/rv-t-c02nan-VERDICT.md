# FIX — SF1 coupling candidate 5c0e7c11e

Reviewer: lane rv-t-c02nan, Codex GPT Sol, 2026-10-14. Scope is the seven-file SF1 surface/Noah/PBL handoff only. No GPU was run by this critic; c08/c64 microphysics and the later GBM cold initializer are excluded.

## Required fix

**P1: retain GBM's reconstructed UST in State.ustar.** At candidate `src/gpuwrf/coupling/scan_adapters.py:1895–1900`, `gbm_pbl_adapter` retains QC, TKE and PBLH but drops `out['ust']`. Pristine `module_bl_gbmpbl.F:539–541` reconstructs UST using raw PSIM and GZ1OZ0, then writes it through its INOUT caller argument at line 835. The JAX GBM kernel already returns the correct result; the output adapter discards it.

Independent unchanged-pristine REAL4 execution on retained real c08 column 0 returned **0.10000000149011612 m/s**, byte equal to the candidate kernel's UST. The adapter returned its incoming **0.013878672383725643 m/s**, a factor 7.205 smaller. All kernel rates were finite. The output-seam assertion failed genuinely. This omission existed before this patch; it still prevents the explicitly required GBM UST caller contract from being complete. This is not evidence of the c08 microphysics NaN cause.

Write the already-computed GBM UST back in the adapter, with the existing 2-D shape and carry dtype, and add a deletion-sensitive assertion for this output. Apply it to the intended GBM route, without changing another PBL's UST contract. No second surface solve or new physics calculation is needed. Reuse c02/default evidence if the BL12-only change is proven program-identical there; check the changed GBM output and its affected qualification once. This critic did not edit producer source.

## Accepted evidence

- [M CPU] 18/18 independent key tests passed: all 16 new SF1 contracts plus existing YSU fixtures. An independently rerun raw-PSI mutant caused one numerical AssertionError, zero setup errors, with 32 nonfinite theta entries in the JAX output.
- [M original WRF] Private pristine YSU rebuild reproduced all six corrected active-neutral tendency/exchange arrays byte for byte. Source checksums match the producer's staged pristine bodies. Original WRF with raw zero PSI remained finite but changed scalar tendencies/exchange; the port's zero-PSI mutant is nonfinite. Do not claim original WRF itself produced NaN in that control.
- [M source] WRF pbl_driver maps YSU/Shin-Hong PSIM=fm, PSIH=fhh (1227/1313); GBM maps raw psim/psih, TSKOLD, GZ1OZ0 (2102–2103). Candidate input routing matches these; GBM reconstructs UST internally. GFS/MRF retain raw PSI; ACM2/BouLac do not consume PSI. The selected SF1 result is passed once into Noah-MP and the PBL; default MYNN and existing epsilon producer arithmetic are unchanged. Hurricane diagnostics also feed Noah without a duplicate solve. Urban YSU receives resistance aliases.
- [M CPU] Another 18/18 tests passed for pristine hurricane fluxes and revised-MM5 urban surface outputs.
- [M CPU tracing] Independent default native-facade identities exactly match Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` and PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
- [M existing GPU, independent CPU rescore] TCN23 c02 source commit, clean detached-source assertion, launcher SHA, actual input SHAs, device assertion, 4800-step proof and both retained native PIN SHAs verified. The frozen scorer was rerun on private read-only links to original CPU/GPU outputs: **25/25 frames, 318 numeric fields, 62 bounded fields PASS, zero nonfinite/incomplete/failures**. Every numeric field result equals the original report. Unbounded fields remain REPORT_ONLY. Numerical configuration in the run proof is c02; GPU history attributes still advertise default scheme IDs, a disclosed writer issue outside this source scope.

## Limits and cost

TCN23 wall 523.868864 s and sampled process VRAM 5392 MiB are cold shared validation measurements, not S1 or a matched performance comparison. Default jaxpr identity implies no default-program cost change [I]. The selected route removes duplicate surface solves [source]; net runtime/launch benefit is unmeasured [U]. Actual c02 E41 is **FAIL**, six kInput HLO sizes 570/549/661/659/586/632, native STACK 8 B, no nonleaf >1k, coverage 2/2; all remain disclosed, no waiver. The new 632-HLO SF1 lookup is not called free. The UST repair should reuse one existing 2-D result [I]; changed layout/launch cost must be assessed on the changed artifact, not inferred from linear complexity.

GBM's existing strict O0 33-element and compiler/input-ULP 18-element mixing residual failures remain FAIL, with all primary rates/TKE/PBLH in band. They were not softened or relabelled by this review. c02 causal evidence includes the integrated MP1 theta cofactor; no exclusive SF1-cause claim and no original failed-trajectory capture claim. c08/c64 remain separate open work.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-c02nan/`: `audit_summary.json`, `key_tests.xml`, `mutant.xml`, `satellite_tests.xml`, `oracle_audit.json`, `gbm_ust_probe.json`, `gbm_original_ust.json`, default identity JSONs, private `c02_recheck/` scorer reports, logs and reproduction scripts. Snapshot/private builds are removed after their last consumer; no foreign references, caches or native PINs were changed.
