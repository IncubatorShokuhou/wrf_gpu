# FIX — CAM-MG EXCH_H/TKE delta c17b9fb92

Candidate: `c17b9fb92e2602b588031e82ed1e15d63ef8349b`, reviewed against rejected `23fc64fea`. Same critic, rv-s2-cammg. Only the caller correction and affected evidence were reviewed; earlier paid scientific proofs carry forward.

## Remaining P1: Shin-Hong forwards the wrong TKE mode

The delta directly retains `out['tke']` in `scan_adapters.shinhong_pbl_adapter` for CAM-MG. That adapter omits `shinhong_tke_diag`, so `physics/bl_shinhong.py:729` selects its default **1**. WRF's Registry default is **0** (`Registry.EM_COMMON:2475`), passed through `module_first_rk_step_part1.F:1148` and `module_pbl_driver.F:1318`. Pristine Shin-Hong writes shared TKE only inside the diagnostic-1 block (`module_bl_shinhong.F:1478–1612`); diagnostic 0 preserves the shared input. Its cold initialization also sets shared TKE to `EPSQ2L/2 = 0.005` (`:2277`), whereas the new CAM cold-binding fallback retains zero when that input variable is absent.

Consequently the directly forwarded diagnostic-1 TKE is not the actual WRF-default shared operand. CAM activation uses this operand. The existing port explicitly discloses unqualified Shin-Hong TKE in its module header; forwarding it does not establish its fidelity. The old rejected generic QKE/2 route was also incorrect: this is an incompletely corrected caller, with no claim of a measured numerical regression against the old port. Source hashes and the omitted-keyword/default witness are in `delta-shinhong-default.json`.

**Exact fix:** for the currently admitted WRF default, initialize and preserve the actual shared Shin-Hong TKE, and prevent diagnostic-1 output or transported QKE from replacing it. Keep nondefault diagnostic mode refused until its caller is bound and qualified. Alternatively refuse only `mp=11/BL=11` pending that work. Add a source-grounded conditional-producer/held-state gate. Preserve WRF's lack of mandatory mp11/bl9/ra3 pairing; no broad PBL rewrite or tolerance change is needed.

## Closed findings and accepted evidence

- Original MYNN EXCH_H omission is corrected: existing `out.kh` reaches the already allocated CAM carry; current coefficients survive actual post-RK restoration. Nonwriters retain shared TKE, normal producers preserve unused top memory, and CAM REAL fields avoid legacy fp64 widening.
- Support-matrix row is present; uppercase/lowercase cold TKE aliases and nonfinite refusal are covered.
- Independent affected CAM rerun: **13 PASS, 18 deselected, 33.11 s**. This includes original-WRF Nc/QNDROP/WSEDL handoffs for BL5/BL1, actual RK restore including SF1, cold aliases, five scan producers/single-call behavior, and REAL-carry alias preservation. Stubbed producer tests cannot detect the real Shin-Hong default mismatch.
- Independent source audit: **14 science/asset blobs BYTE-identical to 23fc**; pristine source SHAs and 11 heat/7 TKE call-keyword inventories checked. The BL11 keyword inventory alone does not establish its conditional write/default mode.
- Final `src/tests/pyproject` blobs match the producer's validated `035b1aa42` exactly. Portable selected program, input carry and nested constants match across rebase; corrected whole CUDA lowering SHA is `3961271b…`; both full default hashes remain Swiss `f672a573…` and PROD `6fc715a2…`.
- Producer's affected Swiss receipt: two current-PBL-to-MP KH/TKE handoffs BYTE, 75,852 nonzero KH cells per step (max 614.115/596.776), MYNN shared TKE held zero, NF/fatal/guards 0, full restart BYTE, peak RSS 4.39 GiB. Source/program seals support reuse without another continuation.
- Carry forward earlier independently checked original 48×52 BYTE, native 48×27 original-enclosure values, and independent physical mutant RC05 on the unchanged science body. Producer's added handoff-deletion receipt reports two original Nc assertions and zero setup failures. Accepted shared/SF1 gates are reused; no broad suite or physics-column repetition.

Performance/memory: reuse of existing PBL outputs/carry introduces no extra PBL solve, persistent field or host synchronization. Additional GPU stores/layout/traffic/launches remain unmeasured; actual GPU/E41/roofline/D6 are pending. Preserving default-mode held TKE is also the smaller correction.

Resource disclosure: the initial broad 30-check FIFO waiter was cancelled before acquisition (0 log bytes, no tests started). The 13-check rerun used a requested hard **900 MiB/no-swap** scope as a light-work probe; measured peak RSS was **1,205,548 KiB (1.15 GiB)**, above the approximately 1 GB light-work threshold despite that cgroup limit. This measured RSS is recorded rather than labelled below 1 GB. Future JAX checks require FIFO. No GPU used; cache NONE. Only review files were written.

Verdict applies to this exact candidate. MYNN/YSU correction and evidence reuse are accepted; the source-proved Shin-Hong TKE default/held-state issue prevents MERGE until fixed or narrowly refused.
