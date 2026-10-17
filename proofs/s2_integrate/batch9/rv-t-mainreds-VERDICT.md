# MERGE — lane/t-mainreds

Reviewed immutable commit `12bf21669ff3972ad5019b4ff436c2095b4fe120` against `95bf2d90c`. Critic: rv-t-mainreds, Codex GPT Sol, 2026-10-14. No required fixes.

## Independent gates [M CPU]

- RV01: **228 PASS**, 0 failures/errors, 100.20 s. Includes all **15 assigned regressions** and full affected doctor, DFI admission, ensemble, nest admission, portability, memory-audit, producer regression, Jensen evidence and NetCDF-lock modules.
- RV05: **17 PASS**, 8.60 s. Fourteen real admission/offline subprocess paths reject any backend initialization, NVIDIA query or CUDA runtime library load; three deliberate activity controls prove the guard catches each. Existing doctor/ensemble guards also prohibit importing JAX. Ordinary run bootstrap may import JAX configuration and optional CUDA/plugin namespaces without initializing a backend.
- RV06: restoring only the eager coupling initializer produces **1 genuine assertion failure, 0 setup errors** (`heavy import during configuration admission: jax`). Snapshot source was never mutated.
- RV07: **5 PASS**, 33.75 s: import-time environment census/registry, import-resolved constant keys and external-reader disposition. No import-time constants or key rules were removed.
- RV08 Swiss: full hash `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` **EXACT** (55.6 s).
- RV09 PROD d01: full hash `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` **EXACT** (82.2 s).
- RV11: **2 PASS**, 3.35 s. Audit geometry matches actual EEPS16/KEPS17 initializer union, scalar int32 step and actual two-tracer volume/boundary/budget initialization. RV01 also confirms **172/172 State slots**, precision and tracer-count scaling; no inventory reduction.

Both release identities use the official native-facade CPU trace probe, with only its temporary directory relocated to this lane. They are program-regression evidence, not WRF fidelity, GPU execution or benchmark evidence.

## Source and evidence integrity

RV04 independently verifies all **11** kernel changes are exactly one Pallas import substitution, preserving every numerical body and source line count. The facade returns the original CUDA Pallas callable; nine lazy coupling exports retain their original objects/API. Full candidate `src` is byte-identical to the source used by the producer's shared-consumer run.

All **25 producer artifact hashes** match. Producer MR04/MR10 XML contains **19 genuine assertion failures, 0 setup errors**; RV06 independently reproduces the eager-import kill. Jensen's entire original proof tree, immutable native receipt, observations and **335 disclosed bound excesses** are unchanged. The migration guard accepts only the precise import undo before checking the original source SHA; any additional source edit still fails.

The NVML changes preserve measured rc 0/7/124 on malformed/failed telemetry, explicitly disclose UNKNOWN and preserve bounded draining for valid low-memory telemetry. All shell tests use private NVIDIA/sleep stubs; no vendor query was executed.

## Performance, memory and limits

[I] No new timestep launches, intermediate arrays, carry fields, transfers or synchronization: numerical kernels and callable identity are unchanged, as are both full default traces. Facade selection is cached host capability lookup at trace/factory construction; coupling exports resolve once. The audit changes static accounting only, including an explicitly reported tracer count (default one); it is not live VRAM. GPU timing, compiled-artifact performance and readiness were not measured or claimed; critic contract forbids GPU work.

The producer's 630 shared checks comprise 401 observed passing progress dots with reconstructed node IDs and 229 explicit XML passes. That documented limitation remains; this critic does not claim an independent rerun of the whole 850-check producer cohort. Existing release-wide E41/D6/F2 qualification is unaffected.

RV02/RV03 were discarded overstrict critic import guards (optional namespace imports are not backend initialization); RV10 corrected a critic assumption that both mutually exclusive epsilon families initialize under KEPS17. Their raw failure receipts remain, separately from final passing gates. No product fix or scientific tolerance change was made by this critic.

Snapshot, pytest scratch and helper bytecode have been removed. Persistent JAX/XLA caches: NONE. Small raw logs, XML, JSON and reproducible helpers are retained in the lane directory and `proofs/rv_t_mainreds/`.
