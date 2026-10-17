# MERGE — lane/s2-ntu

Reviewed candidate `a4f52286dcf554b076c81997f3d8db0df27a95fe` against base `e2ffb6858`. Reviewer: rv-s2-ntu, 2026-10-14T23:23:51.922683+00:00.
Verdict applies to the experimental, opt-in, single-domain CPU-qualified feature. GPU/E41, GPU resources and full original-CPU coupled D6 remain pending. No product files were changed by this review.

| Criterion | Independent evidence |
|---|---|
| Pristine WRF oracle | Fresh REAL4/REAL8 builds of unmodified `module_mp_ntu.F` reproduce all 1,428 frozen arrays per kind. Port matches all 8,568 public values per kind over 12 regimes and two evolving calls; finite and all 28 prognostics active. |
| Caller and transport | WRF `phy_prep` dry theta, unmodified vapor and Exner checked independently; dry-theta/vapor/returned moist-theta BYTE, Exner max difference zero. Registry inventory is seven moist and 21 scalar species; native level reversal/staggered W, scalar advection/diffusion and boundary mapping audited. |
| Required JAX execution | Paid NTU54 XML/log and committed receipt are genuine: compiled JAX body, 1 PASS in 101.53 s, 6.87 GiB RSS under its authorized 12 GiB exception. All 18 audited module/test/helper files are AST-identical; common caller differs only in inactive NSSL radar code. Reused under the validation-reuse rule, with no nighttime 12 GiB rerun. |
| Restart, history, keys, interfaces | 13 own and 223 shared tests PASS; complete held-state fail-closed restart, three writer modes and original metadata, actual table-provider and CCNTY key collisions, PhysicsStepSpec, both pipeline bindings and NetCDF lock consumers. |
| Mutation | Deleting only the RAINNC increment produces one genuine numerical AssertionError, 0 setup errors, difference 0.03842036 mm. Initial CPU-guard setup failure is retained separately and is not counted as a kill. |
| Native frontend | Actual 42×42×44 Swiss mp56 history program lowers for CUDA/sm120 on CPU: 199 Pallas calls, 0 interpreted, CompileOnlyPyClient, 48,662,675-byte StableHLO. Whole preflight 47.01 s, CPU RSS 2.48 GiB; no GPU compilation/execution or E41 claim. |
| Defaults | Fresh canonical native-facade CPU traces reproduce FULL Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d` and PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`. Only the probe scratch-directory literal is redirected. |

The case-scoped F2 disclosure is supported: a fresh original full-grid sixth-order diffusion replay reproduces all four retained row payloads, including REAL4 ice mass −1.341928046e−10 kg/kg and ice number −946.7988329/kg. It uses the captured RK1 dt=6 s tendency and final dt=18 s. This reproduces the large source-driven class, not every visit or the full original RK chain. The failed 1e−12 round-off predicate stays failed; manager MAILBOX decision 2026-10-14T22:44:27Z explicitly accepts this bounded known issue. No additional clip or source repair was added; native MAX/fmax sites mirror original WRF MAX(0, …).

The 313 compiled CPU values outside the fixed original enclosure remain disclosed deviations, not an enclosure PASS. Upstream initialization/scratch sensitivity and source NaN/MAX behavior are explicit. No GPU or forecast-accuracy claim is admitted by this verdict.

Resource assessment: selected NTU storage adds about 5.06 MiB [I] at 42×42×44 before transport scratch; default options allocate none of these arrays. Cold initialization host transfers occur outside stepping. Native adaptive loops are column-local; actual launches, spills, VRAM, readiness and roofline remain [U]. The inherited shared hail-density/acoustic/PSFC reducer omission reported by t-redsuite also applies to NTU's qh [I source]; it remains a coupled-qualification risk and was not fixed in this lane.

Temporary review builds, input symlinks, pytest scratch and snapshots are removed after their last consumer. Retained evidence and reproduction scripts are bound by `FINAL-SEAL.json`. Manager integration through s2-integrate is the next step.
