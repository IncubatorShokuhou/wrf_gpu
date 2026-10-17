# MERGE — Shin-Hong caller delta 523fde0ef

Candidate `523fde0efdc8d4ddaa8fe20ac3abeb65494b8d38`; delta against `c17b9fb92e2602b588031e82ed1e15d63ef8349b`. Same critic: rv-s2-cammg. This closes the remaining caller finding from the previous FIX verdict for this successor candidate.

The selected `shinhong_tke_diag` value now reaches both runtime surface paths for every Shin-Hong run. Registry default 0 is explicit; the flag participates in static namelist aux/JIT identity and survives serialization. Cold initialization writes original REAL4 `0.005` to mass levels, retaining the unused CAM top-memory element. CAM's shared TKE is the authoritative input, independent of transported QKE and without an input floor. Mode 0 preserves the held shared field; mode 1 forwards the existing diagnostic. Those fields remain intact through the actual post-RK restore and checkpoint round trip.

## Independent checks

- **96 PASS in 9.24 s:** 14 caller/cold-init/static-key/restart cases, 80 namelist-binding checks, and two independently constructed deletion-witness cases.
- The deletion witnesses force the old implicit mode 1 for requested mode 0, for MP8 and MP11. Both produce genuine `AssertionError` with the original-source held/conditional-TKE assertion; zero setup errors. Control cases cover MP8/11 × SF0/1 × diag0/1.
- **14 CAM science/asset blobs BYTE-identical to c17.** Shin-Hong numerical AST and physics-registry executable AST are unchanged. Only the requested adapter, cold-init, static aux and runtime forwarding functions change; no parked diagnostic science is included.
- Final `src/tests/pyproject` blobs equal the producer's validated `da2340959`. CM77/CM90 caller, initialization, serialization, binding and restart source/test reuse dependencies were independently compared to the actual commits.
- Fresh producer full default hashes verified: Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`; PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.

Prior accepted original 48×52 BYTE, native 48×27 original-enclosure, shared-consumer, corrected Swiss two-step/restart and whole-CUDA-lowering proofs carry forward by their source/scope seals. They were not repeated. The earlier MYNN EXCH_H/RK-preservation/support-matrix findings remain closed.

## Disclosure and cost

The manager's direct instruction for this review accepts the pre-existing nondefault `diag=1` deviation as a disclosed v0.4.0 caveat. That disclosure remains in the documentation; numerical follow-up is parked for v0.4.1 and excluded from this candidate. This review does not reopen that accepted caveat or approve GPU/forecast fidelity.

No extra PBL solve, persistent field or timestep host transfer is introduced. The cold seed fills existing arrays once; the runtime flag is static, and the existing shared fields are reused. Actual GPU stores/layout/launches/traffic, VRAM/readiness, E41/roofline and full forecast D6 remain pending/unmeasured.

All independent JAX checks ran on immutable 523fde0ef through the official FIFO RAM-slot wrapper, **6 GiB/no swap**, CPU16–19 nice19, `JAX_PLATFORMS=cpu`, cache NONE and a 300 s child timeout. Measured peak RSS **1,508,240 KiB (1.44 GiB)**; process wall **10.59 s**. No GPU used and no producer files edited. RC15–16 are logged in main; the snapshot and pytest files are cleaned after the final consumer.
