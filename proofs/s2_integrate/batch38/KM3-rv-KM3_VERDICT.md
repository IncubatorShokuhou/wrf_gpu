# MERGE — km_opt=3 GPU refusal c4af1a242

Same critic, separate item. Candidate c4af1a242b687058640f711d5a00569ea23fce6a; separate source change e0952523d50c57c04c6288b8a8d53d529e5bffd4. Scope: CLI before JAX import, API before device query/prefetch/allocation/compile, single/nested/resume and any input/domain, CPU unchanged.

The change passes the admission gate. `km3_config.selection_errors` rejects km_opt=3 for non-CPU targets, including per-domain lists; CPU and other km values remain admitted. The existing shared host admission now combines the independent MP11 and KM3 predicates. Namelist validation and binding also refuse KM3. Actual single/nested/public/private/daily/resume paths run the guard before device/model/DFI initialization; configured/environment ensemble paths use the retained all-input admission. CLI retains last-option semantics, actual-case/override/ensemble admission, and strict pre-JAX refusal. The refusal clearly identifies km_opt=3, CPU-only status, the failed hour-1 attempt, and a safe default selection.

[M CPU] KR01 independently reruns 288 tests: 288 PASS, zero FAIL/ERROR/SKIP, 6.67 s. This includes the producer KM3/CAM-MG entry controls, CLI/binding/validator/catalog consumers, and 20 independently authored KM3 checks covering d01/d02/d03, single/nested, fresh/resume, prefetch on/off, second batch input, daily initialization, repeated/equals/abbreviated CLI options in both directions. No GPU was initialized.

[M CPU] KR02 independently deletes only the KM3 predicate: 17 genuine failures (initialization sentinels, selection assertions and DID NOT RAISE), three healthy controls PASS, zero setup errors, 2.19 s. KM3 refusal is deletion-sensitive.

[M source] KR03 verifies all 18 producer receipt SHA values, exact src/tests/scripts trees versus measured e0952523d, and byte identity of dynamics/physics/kernels/runtime/integration/nesting/coupling/contracts before/after the separate KM3 change. CPU admission returns the same successful result with no numerical change. The producer's 367 shared PASS/one existing missing-corpus SKIP remains producer evidence; the critic independently ran the narrower 288-test selection.

Default IDs: independent FULL Swiss f672a573 and PROD 6fc715a2 are EXACT; complete hashes and timings are sealed in KR04/KR05. These are CPU trace/regression identities, not forecast fidelity.

The retained original D6 receipt reports BOTH_RUNS_FAILED, hour-1 non-finite d01 U, and no full-24-hour comparison. The refusal truthfully makes no CPU-WRF qualification claim. KNOWN_ISSUES wording is proposed in the producer packet for the manager/documentation owner; the catalog disclosure and referenced failure receipt are present.

Performance/memory [I]: zero numerical kernel/timestep changes, new launches, device temporaries/transfers, state fields or baked key inputs. KM3 reuses the already parsed admission input and adds only a host selection lookup/list. No additional namelist IO relative to the MP11 admission mechanism. No GPU speed, VRAM or scientific-fidelity claim.

Integration requirement: KM3 refusal is mandatory independently of the MP11 fallback. If a CAM-MG fix replaces the MP11 predicate, retain `_gpu_selection_errors`, KM3 validation and all shared entry hooks. Do not delete cammg_config wholesale merely because its filename names CAM-MG. The critic made no producer edits and no merges.
