# F17 repair efc9441e8 — MERGE, CPU scope

Reviewed delta `2daaa8e6d..efc9441e8`, implementation `2c06e1f895`, plus SSiB test mock `a59b120f79` and QNSE applicability receipt. Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/PMF1/`.

The prior F17 caller blocker is closed. Legacy bl12 now refreshes/holds radiation on the entry State before GBM, and the later radiation slot reuses that rate. This matches pristine `module_first_rk_step_part1.F:264` before PBL `:1113` and `module_pbl_driver.F:2099` supplying current RTHRATEN to GBMPBL. No producer/model edits by this critic.

- **[M CPU] Five healthy checks, zero failures/errors/skips:** both new producer refresh/hold gates, a private copy of the original critic's failing LW1/SW0 case, the changed SSiB+YSU consumer, and an independent two-operational-step refresh→hold check. Refresh computes one SW + one LW call on entry-state theta/qv/ustar/HFX/TSK, then GBM receives −7e−5 K/s. The next step performs no radiation solve and GBM consumes the held rate. Column/RK numerics are isolated in these lifetime tests; no forecast claim.
- **[M] One independent deletion control:** restoring the old bl1-only condition causes an assertion failure, zero setup errors/skips. No snapshot source was mutated; the deletion is in-process.
- **Exact-source reuse:** only the early-refresh condition changes in model source; all other changes are comments. The whole operational module AST is identical after specializing that condition to False for default bl5. GBM source and original numerical fixture hashes equal the prior independently rebuilt public GBMPBL qualification. Retain the prior REAL4/DOUBLE column results and numerical gates; no duplicate oracle compilation.
- **Default identities carried forward by that exact static proof:** Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`, PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`. These were independently measured on the prior composed source; no new default trace was needed. The producer also records these exact identities on the repaired tree. No MYNN/default-path change.

Efficiency: the existing legacy radiation solve moves before PBL; measured caller counts show no duplicate SW/LW evaluation, and zero radiation calls on a held step. No new State/carry leaf or GBM solver. The earlier ACC implementation remains BYTE-identical. Changed GPU buffer lifetimes, peak VRAM, whole-step timing and readiness remain **[U]**; the previous live-checkout component timing remains a producer observation, not a qualified performance result. c08 still needs the composed F17 GPU/D6/E41/resource qualification.

## QNSE SICE remains OPEN

This successor supplies applicability evidence, not an implementation. Pristine `module_bl_qnsepbl.F:488–499` reads SICE in the land-point QFC1/latent-heat correction; the port still supplies zero. An independent read-only census verifies **86 present composite wrfinputs**, including c26, have known finite SEAICE/XICE arrays with min=max=0. Array hashes and paths are retained in `PF04-seaice.json`. Thus SICE=0 is correct for those inspected inputs; this does **not** establish a universal admission refusal or sea-ice support.

The producer explains that the broader sea-ice surface path is also unimplemented. Keep that limitation and the prior scoped F23 verdict: Noah opt_sfc=2 / RUC / SSiB caller hunks are approved; QNSE sea ice is not fixed. A general initial-SEAICE refusal is a proposed loading/product change for the manager/owner, not present in this candidate. It must distinguish unsupported sea ice from the supported inland glacier path.

Integration must preserve both early-refresh conditions if composing with the cumulus branch: `(bl in (1,12) or cu in CU_RTHRATEN_SCHEMES) and diagnostics is None`. Do not overwrite one fix with the other. This F17 approval does not change the separate cumulus FIX verdict.

The prior PM3 sealed reports and restored raw failing receipt bytes were checked unchanged. The critic test was copied into this lane's new artifact directory before execution; old verdict evidence was not overwritten. Snapshot, pytest scratch and caches removed after the last dependent arm. Original reference/case files unchanged; no GPU execution.
