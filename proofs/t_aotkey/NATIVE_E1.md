# Native E1 diagnostic (original source 86e630370)

This is cache/execution regression evidence, not CPU-WRF fidelity evidence. The original ALISIOS E1 and case trees were mounted read-only for measured GPU probes. No fingerprint, HLO or ABI guard was bypassed. Model source was a clean detached snapshot of the actual E1 commit. Every arm used the official lock and driver bridge with a 20-minute maximum GPU timeout (the cached step-1 arm used seven minutes).

The repaired source-level key collisions are a separate established defect, fixed in 8b200ddd7 and merged by the manager in 76a72bdd1. The original repaired census covered 23 readers, including QNSE sf=4, GFS cu=94/95 and newly merged CuP/MSKF cu=10/11 providers. After rebasing onto main `7cd23caef`, the composed census covers 29 readers; nine targeted census/import-environment/provider full-key/HLO/output checks pass (AK52, 27.78 s). The temporary CPU snapshot and pytest cache were removed immediately. This does not by itself identify the cause of all historical E1 trajectories.

The native probe loads all three real nests to initialize the original root carry, including optional ozone. It then executes d01 with identical copied runtime inputs, the original altered roughness tables and original namelist. The complete lowered HLO digest matches retained Z: `01fa01a98274d40f78bdf67f97b2c57d46f96791c37bb5491176d6a83c826e7a`. Hardware/version/raw-flag target fingerprints and input/output ABI match.

| Comparison [M GPU] | Step 1 | Step 67 (3618 s) |
|---|---:|---:|
| Fresh compiled / after export / own AOT reload | 183/183 byte exact, finite | 183/183 byte exact, finite |
| Fresh direct compile / export-enabled compile, shared paid cache | 183/183 byte exact, finite | 183/183 byte exact, finite |
| Fresh native / retained Z native | 26/183 leaves differ, finite | 98/183 leaves differ, finite |
| Current original tune directory pinned / own reload | 183/183 byte exact, finite | 183/183 byte exact, finite |
| Pinned current original tunes / retained Z native | Same first-step magnitudes as fresh | 98/183 leaves differ, finite |

At step 1, maximum differences are theta 0.0001220703125 K (relative to leaf infinity norm 2.45e-7), u/v 0.0016604/0.0016606 m/s (4.62e-5/9.83e-5), w 0.000144236 m/s (5.10e-4), and total pressure 2.8671875 Pa (2.81e-5). Surface/T2/Noah land/radiation fields are byte exact. All 26 leaves and both infinity/RMS-relative norms are recorded in `AK45_step1_magnitudes.json`. A derived near-zero heating tendency has relative difference 1 but absolute difference only 1.695e-6 K/s. The proposed all-fields <=1e-6 classifier does not pass; these are also not O(1) bulk physical differences. At step 67, T2 differs by up to 0.80114746 K and skin temperature by 1.82418823 K.

All differing leaves at step 1 are listed below. Absolute deltas use each field’s native units; relative delta is `max(abs(A-B)) / max(max(abs(A)), max(abs(B)))`, which avoids pointwise division by near-zero cells. The separately recorded RMS-relative norm is available in the receipt.

| Carry leaf | Max absolute delta | Relative to leaf infinity norm |
|---|---:|---:|
| `state.u` | 0.00166040659 | 4.62243692e-05 |
| `state.v` | 0.0016605854 | 9.83146845e-05 |
| `state.w` | 0.00014423579 | 0.000510125728 |
| `state.theta` | 0.000122070312 | 2.44889628e-07 |
| `state.qv` | 1.39698386e-08 | 1.15305365e-06 |
| `state.p_total` | 2.8671875 | 2.8122823e-05 |
| `state.p_perturbation` | 2.8671875 | 0.00133606128 |
| `state.ph_total` | 0.03125 | 1.53521397e-07 |
| `state.ph_perturbation` | 0.0201416016 | 2.41674249e-06 |
| `state.mu_total` | 0.0234375 | 2.41436217e-07 |
| `state.mu_perturbation` | 0.0223388672 | 1.15508149e-05 |
| `t_2ave` | 3.05175781e-05 | 1.01724691e-07 |
| `ww` | 0.000984914601 | 0.000199198816 |
| `mudf` | 0.000751465559 | 0.000284695244 |
| `muave` | 0.0252771378 | 0.000350510856 |
| `muts` | 0.0234375 | 2.41436217e-07 |
| `ph_tend` | 45.144043 | 9.90120297e-05 |
| `u_save` | 0.000431060791 | 1.20003819e-05 |
| `v_save` | 0.000430345535 | 2.54785277e-05 |
| `w_save` | 5.99324703e-05 | 0.000279823521 |
| `t_save` | 6.10351562e-05 | 1.22445676e-07 |
| `ph_save` | 0.017578125 | 2.11201419e-06 |
| `mu_save` | 0.00708007812 | 3.67748553e-06 |
| `ww_save` | 0.000223912299 | 5.21242076e-05 |
| `rthraten` | 1.45519152e-11 | 5.7944397e-08 |
| `h_diabatic` | 1.69542102e-06 | 1 |

Native realization differs even though lowered HLO matches. Retained Z has ELF hash `52c19c9518e14d70127c4f7e2c9d250bd695cc76b5e1c6a00cb3a73cb12feb22`, fresh `e80c3e1757bc6016e4435e155310af63bdd78f03a179d38ac922b0f5c8fc3ed1`, pinned `9a8575cd34c5b0cc26e34632bc3db7252212120b30d5d80711075f80a206fda8`. Of 1112 kernels, 1033 fresh and 1038 pinned kernel code sections match retained Z allowing for renumbering. The vertical 44-level reduction `input_reduce_fusion_15` uses 152 SASS instructions and SHFL.DOWN in retained Z versus 872 instructions and SHFL.BFLY in both fresh variants. The rich fresh HLO identifies a weighted sum of horizontal mass divergence. This proves an arithmetic-reduction implementation difference, supporting a round-off seed; it does not alone prove which change causes the complete forecast drift.

Toolchain discriminator: all three native ELF `.note.nv.tkinfo` sections record ptxas **13.2.78**, build `cuda_13.2.r13.2/compiler.37668154_0`, sm_120a and matching compiler flags. Their actual serialized CompileOptions record the same pip `nvidia/cu13` CUDA data path; JAX/jaxlib versions are both 0.10.0. A CUDA 13.1 versus 13.2 mismatch is ruled out. The original metadata does not seal a JAX shared-library build hash, so version equality is the available evidence.

CompileOptions discriminator: field 15 is `ExecutableBuildOptionsProto.device_memory_size`, retained absent/default 0 versus fresh 8,416,215,040 bytes. Apart from this field and the effective per-fusion autotune path, full serialized options are byte equal. In the version-pinned XLA source, zero falls back to 80% of physical device memory; nonzero supplies the scheduler/rematerialization base budget, from which input/output memory and a slop factor are applied. This is a known confound, not a proof of a memory bug. The exact-options replay (AK49, actual GPU 13:30–13:36Z) requested the retained options, but the backend filled zero to 8,416,215,040 bytes. Its effective options are byte equal to fresh after normalizing only the tune path, and its ELF is byte-identical to the previous pinned compile. It again produces 183/183 exact own reloads at steps 1 and 67, with the same 26/98 differences from retained Z. This is a negative reproduction and does **not** eliminate the historical memory-budget confound. The retained serialized zero does not establish the budget actually used by its original compile.

The original tune directory contains 794 files; the fresh directory 505, all shared, with 110 changed configurations. It is mutable, not sealed against the retained executable. Pinning its current contents does not reproduce retained Z. Both rich original JIT cache files present at the time of inspection contain the *pinned-new* ELF, rather than retained Z; they cannot serve as evidence for retained Z's original optimized HLO. Native Riegeli record 2 contains raw CUDA ELF, not retained PTX. CPU E41 scans, including AK54 on the final replay executable, find no STACK/callee failure, but the full gate is **INCOMPLETE** because the stepping AOT stores slim HLO metadata; no release-level E41 PASS is claimed.

ALISIOS guidance: use the merged content-key repair and keep the original Z references immutable. For byte-exact E2/cache A/B controls, use one sealed compiled bundle containing the six initialization programs and the stepping programs for all active nests; do not independently cold-compile the two arms. Record exact source, input/table content, JAX/jaxlib, CUDA_ROOT/ptxas build and SHA, effective CompileOptions and native blob hashes, and ship the sealed tune cache with that AOT bundle. Copying today's mutable autotune directory alone is not sufficient, as measured here. Comparisons between independent cold binaries are regression/round-off evidence; acceptance of either trajectory still requires original CPU-WRF D6 gates. The root-only one-hour probe does not authorize an unconditional full-coupled default-path fidelity claim.

Primary XLA sources: [field 15 schema](https://github.com/openxla/xla/blob/b6f37ab/xla/pjrt/proto/compile_options.proto#L131), [scheduler memory budget](https://github.com/openxla/xla/blob/b6f37ab/xla/service/gpu/gpu_hlo_schedule.cc#L899), [native unroll field](https://github.com/openxla/xla/blob/b6f37ab/xla/service/gpu/backend_configs.proto#L390). Measured receipts and offline scans are under `<USER_HOME>/wrf_gpu2_lanes/t-aotkey/`; experiments AK32–AK54 are appended to the main ledger, including setup failures and negative interventions.

Decision: **E99-style variation between native FP32 reduction implementations is supported [I]; a complete historical E1 root cause, and WRF faithfulness of both trajectories, remain unproven [U].** The key collisions are independently fixed and merged [M]. Same-compiled-artifact round-trip determinism is established [M]; a changed toolchain is ruled out [M]; copying the mutable current tunes does not reconstruct the retained realization [M]. Two pinned compiles reproduce the exact same new ELF [M], but the backend prevents replaying serialized memory zero as an effective zero through this API. No new wrong-program/key defect is demonstrated by the same-input matched-HLO native probes. A historical per-binary tune/HLO seal is absent, limiting stronger attribution.

Cleanup: all own native experiment cache/CUDA/Triton directories, copied tune pins, detached model/test snapshots and pytest caches were removed after their final consumer. Compact receipts and scans remain. Both protected ALISIOS original executable SHAs were rechecked unchanged: d01 `f6469f7827a086d3508edd85c7c369fe1e39596e0c5d10e996f9d8fcd1e19c2f`; fused_d02 `9fcc26fb8ea042065c1ed5755100ae4e25201ca94e0c84081cfc0b67140d1f96`. No GPU work remains queued. No runtime/model arithmetic was added for this diagnostic; no GPU speed or peak-VRAM improvement is claimed. Pinning an already compiled bundle adds no timestep kernels or intermediate device buffers [I]; cold regeneration cost remains separate.
