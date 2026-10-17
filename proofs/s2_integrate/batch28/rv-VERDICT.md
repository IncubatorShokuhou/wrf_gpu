# MERGE — t-rainnc-callers

Reviewed candidate `e213b65c43e11cf1ed8bde32ef9d7e146eb940eb`, product commit
`bc92bb817c9d537f45a7c91ed758df03235fc626`, against base `77e825223`.
Independent CPU review only; no required source fixes.

| Gate | Independent evidence [M] |
| --- | --- |
| Pristine caller oracle | Rebuilt original WRF REAL32/DOUBLE statements and constants: 42/42 values and all fixture metadata exact. |
| Actual caller checks | 9 passed, 5.53 s. Tiedtke tests intercept the actual column boundary; density test verifies producer selection, not a new ALT oracle. |
| Exact-parent deletion | CU6: 5 numerical AssertionErrors; ACM2: 4. Zero other failures or setup errors. |
| ACM2 wrapper coverage | 16 clock checks pass across DOUBLE/REAL carry and SF0/SF1 routes, at own steps 1, 2, 3, 200. |
| Actual operational consumers | 3 affected smoke tests plus 2 existing ALT/density reconstruction checks pass, 31.11 s. |
| Default programs | Full native-facade CPU traces: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`. Both frozen hashes exact. |
| Existing evidence | 24 receipt hashes verified, including exact decompressed hashes; 243 shared and 3 affected smoke PASS cases confirmed from genuine XML. Both product files byte-identical to measured `c7fd806cc`. |
| Scope | AST comparison confirms only the four declared adapter/wrapper functions and `_physics_step_forcing` changed; unrelated scheme definitions unchanged. |

WRF source independently checked: `module_first_rk_step_part1.F:1377` passes
`grid%t_phy` and `grid%rho` to cumulus; `module_cumulus_driver.F:1397` forwards
these to CU6. `module_big_step_utilities_em.F:4834–4856` produces dry temperature
and total density; its original `conv_t_tendf_to_moist` body at 6674–6707 supplies
the conversion used after physics (`module_first_rk_step_part2.F:1009`).
`module_pbl_driver.F:1593` passes `XTIME=itimestep`, despite the ACM module's
minutes comment. WRF initializes the step counter to zero and increments before
physics; the production port supplies `(step_index-1)*own_dt` to physics, so
`1+round(lead/own_dt)` is the correct own-domain step. `XTIME>1` disables the
relevant corrections on the **first** own step, not the first two.
Theta remains in the RK non-dry increment replacement list.

Performance/memory: retained producer CPU component bench shows +0.17354 ms at
44×117×267, zero compiled temporary bytes, unchanged output bytes; new resident
arguments account for the density reconstruction. ACM clock adds four scalar
JAX operations. No new carry/schema or host transfer is introduced [I source].
Full GPU launch, traffic, memory and latency costs remain [U]; default program
identity is measured, not a speed claim.

Limits: original c05/c06 RAINNC failures remain binding pending manager GPU
requalification; no forecast-causality claim or tolerance change. Existing
Tiedtke pressure/interface/eta approximations and legacy DOUBLE solver remain.
CAM-MG is outside this patch. Full shared-smoke OOM is disclosed, not counted
as PASS. Reviewer RC05 initially had eight REAL fixture TypeErrors before model
execution; correcting State traversal to JAX pytree traversal gave 8/8 passes
in RC05b, reusing the already passing eight DOUBLE cases.

Raw independent evidence and reviewer harness are preserved beside this file.
