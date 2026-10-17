# MERGE — lane/t-nsslni

Reviewed immutable `b53563750617b26e21c2aad4dc64259053f3f90d` against `1d791420da602b80f3d89ef9da26fc0b70693abe`. No fixes required.

The source premise is correct: pristine WRF 4.7.1 normalizes mp=19/21 to one-moment NSSL; Registry `nssl2mconc` enables QNI only in two-moment mode. MYNN writes scalar_tend(P_QNI) only with Flag_qni, and CAM inserts it only for an active P_QNI. Original NSSL reads/writes CCI only at ipconc>=5. Thus there is no direct Ni update to preserve for mp=19/21. This patch removes their spurious PBL source while retaining active mp=17/18/22 and the strict non-advected-source consumer refusal.

- [M CPU] 73 independent targeted tests PASS (12.82 s): Registry ownership, actual MYNN/CAM dispatcher, two original MP-call preservation gates on six real Swiss columns / 44 levels / dt18, with 27 nonzero CCI values. Original-source, input, fixture and submitted receipt hashes verified. Scope is two microphysics calls, not two full forecast timesteps.
- [M CPU] Independently repeated old-gate mutant: four genuine assertion failures; deleted active sources: six genuine assertion failures. Zero setup errors. Active-source deletion covers mp=8/18/22; mp17 is independently covered by the ownership/consumer unit tests.
- [M CPU CUDA IR] Exact c14 MP19/CU99 and c15 MP21/CU0 CLI inputs, direct AOT argument pytree with dynamic namelist/clock/int32 start, native Pallas/Triton, product tridiagonal primitive: 2/2 lower PASS (58.72/55.70 s). 161/154 Triton IR sites are not device-op counts. No CUDA executable compilation or GPU execution claimed.
- [M CPU] Independently repeated official default identity: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`. Both EXACT. Official runner only relocates temporary scratch to this critic lane.
- [M source] Complete Registry membership checked for codes 0–199. All other model blobs BYTE; op.py AST differs only by the helper and two producer predicates. All measured/final model trees equal `d87ad5662692c4a20156719be3ff8cdd0fce12d6`. scan_adapters.py untouched. No new baked asset or namelist input.
- Producer shared-consumer evidence reused after SHA/source audit: 78 completed PASS + 257 resumed PASS = 335 unique, disjoint nodes; no duplicate numerical validation.

Performance/memory: static Python selection adds no carry, array, intermediate allocation, kernel, synchronization or transfer. Inactive configurations lose erroneous source arithmetic; defaults have exact jaxpr identity. This is source/graph evidence, not a measured GPU speed or VRAM claim. No model executable was cold-compiled, so no new E41 cubin gate is due here.

Next: manager merges this candidate; v-prep reruns c14/c15 GPU/E41/D6 on the composed tree. Existing scheme/forecast qualification limits remain. Full evidence: REVIEW_RECEIPT.json and RV01–RV05 artifacts beside this verdict.
