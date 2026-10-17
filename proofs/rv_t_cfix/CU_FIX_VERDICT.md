# Cumulus FIX round bf2e8c36ae — MERGE (scoped qualification)

Both prior class-A review blockers are closed: the unchanged registered F7 gate passes on the release path, and Tiedtke cu6 no longer captures or dry-converts unused RTHFTEN. Scope: review against main c360fdf6ee, using the prior CU1 review of b64ed0c878 for unchanged findings. No independent GPU run by this critic.

**[M producer GPU, independently reanalyzed on CPU] F7 frozen gate:**

| Step1, all levels, 5-point lateral ring excluded | Recomputed | Frozen gate |
|---|---:|---:|
| RTHFTEN relative RMS vs original CPU-WRF |0.000720872460655758|≤0.10 PASS|
| RQVFTEN relative RMS vs original CPU-WRF |0.000455284374972408|≤0.10 PASS|
| No-dry RTHFTEN error / correct error |59.1195805732×|≥3× PASS|
| Zero RTHFTEN error / correct error |1387.20793841×|≥3× PASS|
| Zero RQVFTEN error / correct error |2196.42942954×|≥3× PASS|

Raw release_fields.npz SHA `0c61fd8a1a756362d12c7608f291faea96204eeef6c31ec0d584b9ca664ba4ee` matches the committed checksum. All retained fields are finite;8 embedded original-CPU step1 fields match the real NetCDF values exactly. All7 original input/history hashes match this critic's prior verification, retaining the original executable/dump-path-only patch provenance. The actual GPU script matches the committed script BYTE, JSON matches the packet, official lock/lease log identifies RTX5090 and12G RAM slot, source54cdee223 dirty=0, platform assertion is GPU, release defaults unset=1, dt18/cadence33, arm rc0. The old legacy2.879446706× MISS is kept as history; no gate changed and no dynamics-attribution theory is needed to approve this result. The no-dry/zero controls are the same frozen **algebraic output controls**, not executed mutant forecasts. Steps2/3 are producer context, not independently raw-reanalyzed gated results.

**[M CPU source binding]:** the critic freshly traced the final candidate's official native-facade c05 program and obtained full `481eb074d699b3866a690aad9ceae284abcee8ac2cfefc34a8b42a1a52b0957d`, exactly the producer GPU-arm/final-program identity. Resolved work is mp4/bl7/sf1/cu5/Noah-MP, LW4/SW4, dt18/radiation33. Fresh full default IDs also match: Swiss `7379c7d77e94517f07e7bf37763cc1409d1287a13e50684ed9d624caafaec55e`, PROD d01 `6487f5b039a074f02f63d140f1e9212b7ede832d04bf5450e44ad2b3f2c34e73`. These are CPU traces holding the native facade, not GPU benchmark executions.

**[M CPU]191 healthy checks,0 failures/errors/skips:**152 current forcing/RK/GF-caller/restart gates,6 independent lifetime/trace probes (five real adapters across two calls plus the old cu6 critic),33 composed GBM-radiation/phy_prep/NetCDF-lock consumers. Two independent in-process deletions each produce exactly1 assertion failure,0 errors/skips: restoring the cu6 theta-capture path, and restoring cu6 RTHFTEN dry conversion. Snapshot source unchanged.

The field-specific flags match pristine module_em.F:654–665 and BSU:5291–5331: cu6 sets/decouples QV only, retains held TH unchanged; cu16 control captures TH and performs the original conversion. The actual cu6 RK trace records no extra theta-forcing-advection call. The independent two-call harness now injects held TH for cu6 and expects that initial held value unchanged, matching the original caller; other four schemes retain the original dry-literal expectation. These are adapter/forcing lifetime tests with isolated RK/column numerics, not complete forecasts.

Unchanged evidence is reused by BYTE proof: two GF/KF numerical modules and32 savepoints equal the previously independently rebuilt pristine public calls. The original executable manifest is also unchanged. Oracle shell ROOT corrections alter paths, not physics. Final src/tests equal the producer's paid f3ea0c0f3 qualification tree; overlapping owner suites are not added to the191 independent count. Previously reviewed F10/F11/F9/F14/F18 CPU scopes carry forward.

Efficiency: cu6 eliminates the extra final-stage TH-forcing-advection trace and its TH decouple/dry-conversion arithmetic. It retains the shared two-leaf forcing schema; no new carry/State leaf, solve, synchronization or host transfer is introduced by this repair. The original forcing pair still declares8*nz*ny*nx bytes (2,956,800B at PROD d01 shape if selected), not a measured peak allocation. Actual GPU CSE, launch reduction, temporary lifetimes, changed-option wall/readiness and peak remain unmeasured; the short3-step diagnostic arm is not an operational speed benchmark.

**E41 remains UNKNOWN:** the producer's scanner invocation failed and its cache was deleted. The manager explicitly permits this for this review (MAILBOX02:51); the c05 composite rerun must perform E41. This is not an E41 or full GPU-release approval.

Other disclosed residuals remain: r2 HFX/QFX from kinematic handles, r4 CUMULUS_CARRY_MEMBERS documentation, and KPBL fallback exact only for MYNN outside published GFS rates. GF evaluates all columns and masks outputs; no compute-skip claim. Compose with the approved surface branch preserving selected-surface handoffs and raw MYJ/GFS RK qc/qi rates. Keep the early-radiation union bl(1,12) OR CU_RTHRATEN_SCHEMES; shared consumers pass here. c25 stays retired; affected c02–11/c13/c14/c17 GPU composites/D6/resources remain pending.

Artifacts `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/CUF1/VERDICT.md`, REVIEW_NUMBERS.json and retained3MB raw fields. All CF arms MAIN logged; snapshot/pytest/cache removed. Original/producer references untouched; no GPU/model edits.
