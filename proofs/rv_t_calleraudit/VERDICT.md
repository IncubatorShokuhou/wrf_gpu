# MERGE — t-calleraudit F1

Reviewed immutable `052f16d2b69e9f0c1299e6ae0049c742e112a765`, product commit `93dfdac853`.
Scope: microphysics public precipitation slots → retained previous-step carry → Noah-MP/RUC/SSiB interface.
No product edits, no GPU use. Independent critic gate is green for this CPU/source-qualified fix.

## Independent evidence [M CPU/source]

- **55 PASS**, 0 failures/errors, 22.39 s: all 37 F1 tests, `test_noahmp_precipitation.py`, and NSSL cumulus caller tests (`RV01-tests.xml`).
- **32 PASS**, 0 failures/errors, 15.69 s: independent two-step checks for every precipitation-producing admitted MP code (`RV06-all-caller.xml`). These execute the actual operational step, post-RK dispatch, carry refresh, State-to-column view and Noah forcing assembler. Dynamics and land solver are sentinels. Generic adapters remain real with mocked leaf kernels; explicit adapters expose sentinel amounts. This is caller-wiring evidence, not kernel parity or a forecast.
- First surface call sees zero amounts; second sees previous-call RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV divided by 18 s, within the existing REAL comparison tolerance. Carry leaves are fp32 and finite. Kessler uses public RAINNCV=5 rather than accumulated sedimentation=7. NSSL/Milbrandt retain hail and authoritative total.
- **13/13 genuine numeric assertion kills**, 0 errors, 19 deselected, 6.93 s: in-memory restoration of the old seed set makes all generic MP second-step precipitation zero (`RV07-mutant-final.json/xml/log`). Production files are unchanged. Earlier RV05 independently produced the same 13 kills before the explicit-case test expansion.
- Independent pristine Registry/CASE census covers **33/33 admitted codes**, including dry mp0; aliases 17/19/21/22 are explicitly tied to NSSL18, not misrepresented as pristine WRF options (`RV04-source.json`). Generic public-slot filter matches the original driver's actual keyword arguments, including SBU-YLin rain-only and WSM3/5/WDM5 without graupel. Existing NTU rain-only public-slot quirk is preserved.
- Re-run default release identities are EXACT: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` (`RV02-swiss.json`, `RV02-prod.json`). These are program identity evidence, not WRF fidelity comparisons.
- mp8/5/7/15/38 dispatcher branch ASTs equal the product parent; their existing producer implementations and precipitation conversion are unchanged. The seed condition is logically identical on those selections. The two whole-program identities above measure mp8; the other retained selections rely on this exact-source proof, with no redundant numerical runs.

## Original WRF semantics and restart

`module_surface_driver.F:1568–1570` adds previous RAINCV/RAINNCV/RAINSHV to RAINBL; `:3283–3284` passes explicit MP slots. `module_sf_noahmpdrv.F:779–783` divides each amount by DT. `module_mp_kessler.F:134–147` overwrites RAINNCV on each sedimentation substep while adding every substep to RAINNC. The candidate uses instantaneous producer amounts rather than rounded accumulator differences, retains the established spec-zone mask and step order, and adds no physical clipping. Pristine driver/Registry hashes are in RV04.

No new carry schema fields are introduced; an existing optional group is now populated for additional schemes. The native CLI checkpoint identity hashes every model Python source (`restart_store.py:70–82`), so pre-fix native checkpoints cannot silently resume with the old absent carry. Existing precipitation restart checks passed in RV01. No claim of importing arbitrary old public interchange checkpoints is made.

## Performance, memory and qualification boundary

Default programs are unchanged. Newly seeded schemes retain five fp32 surface arrays: **20 bytes per horizontal column [I static payload]**, excluding allocator overhead and possible buffer sharing. Their side outputs reuse instantaneous kernel amounts; the added work consists of surface reshapes/conversions, slot selection and the existing precipitation-rate conversion. There is no new vertical solve, timestep host/device transfer, synchronization, table asset or host callback. Actual extra GPU launches, peak allocation/fusion effects, readiness and elapsed time are **[U]**; the producer's word “negligible” is a qualitative estimate, not a measured speed result. Existing correctness/performance gates remain binding at the affected composite reruns.

GPU/coupled forecast qualification remains pending for c02–c25, c61, c87–c89; use original CPU-WRF references. F2 convective-rain wiring is a separate unchanged defect. RUC still refuses non-Thompson MPs; this review does not expand its admission. No threshold changes or other required product fixes were found in F1.

Evidence and reproducible critic helpers: `<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/`; durable copy under `proofs/rv_t_calleraudit/` on the critic branch. Recreate a detached snapshot at the reviewed SHA and install `data/wrf_pristine` before rerunning the helpers. Private snapshot and disposable pytest artifacts are removed after their last dependent arm; no persistent JAX cache was used.

Tracked mutant log/XML copies trim trailing whitespace for git diff hygiene; original raw receipts remain unchanged in the lane artifact directory.
