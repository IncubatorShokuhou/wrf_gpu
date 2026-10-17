# MYNN references

Inventory: 2026-10-15T00:00:32.000621+00:00. 43 active composites and 11 excluded controls.

D6 evidence is historical and source-bound. `INCOMPLETE` can mean missing writer fields, a compile failure, or a stopped forecast; it is not a full numerical PASS.

Manager policy: a MYNN D6 numerical FAIL uses original WRF zero-init build `9a67a93d` as reference of record, with original-vs-zero spread disclosed. Original references remain intact and D6 bounds remain frozen. t-p3fix owns c21–c24 replacement references and rescoring. c00 and c17 receive two full original-build repeats here; any non-repeatable case receives a separate zero-init reference.

| Case | MP / CU | Recorded case D6 (paired frames; bounded misses) | Separate zero-init rescore | Missing recorded fields | Scope |
|---|---|---|---|---|---|
| c00 | 8 / 0 | d01 PASS (25; 0) | — | — | active |
| c12 | 17 / 0 | d01 INCOMPLETE (25; 0) | — | REFD_MAX, REFL_10CM | active |
| c13 | 18 / 5 | d01 INCOMPLETE (25; 0) | — | REFD_MAX, REFL_10CM | active |
| c14 | 19 / 99 | d01 INCOMPLETE (25; 0) | — | REFD_MAX, REFL_10CM | active |
| c15 | 21 / 0 | d01 INCOMPLETE (25; 0) | — | REFD_MAX, REFL_10CM | active |
| c16 | 22 / 0 | d01 INCOMPLETE (25; 0) | — | REFD_MAX, REFL_10CM | active |
| c17 | 24 / 93 | d01 PASS (25; 0) | — | — | active |
| c18 | 26 / 0 | d01 PASS (25; 0) | — | — | active |
| c19 | 28 / 0 | d01 INCOMPLETE (25; 0) | — | QNBCA, QNIFA2D, QNWFA2D, TAOD5502D | active |
| c20 | 40 / 0 | d01 INCOMPLETE (25; 0) | — | CCN1_GS, CCN2_GS, CCN3_GS, CCN4_GS, CCN5_GS, CCN6_GS, CCN7_GS, EFCG, EFIG, EFSG, MSKF_REFL_10CM, WACT | active |
| c21 | 50 / 0 | d01 FAIL (25; 92) | INCOMPLETE (25; 0) | D_ICE, QIB, QIR, REFL_10CM, RHO_ICE, V_ICE | active |
| c22 | 51 / 0 | d01 FAIL (25; 142) | INCOMPLETE (25; 0) | D_ICE, QIB, QIR, QNCLOUD, REFL_10CM, RHO_ICE, V_ICE | active |
| c23 | 52 / 0 | d01 INCOMPLETE (1; 0) | INCOMPLETE (1; 0) | D_ICE, D_ICE2, QIB, QIB2, QICE2, QIR, QIR2, QNCLOUD, QNICE2, REFL_10CM, RHO_ICE, RHO_ICE2, V_ICE, V_ICE2 | active |
| c24 | 53 / 0 | d01 UNRUN (0; 0) | INCOMPLETE (0; unmeasured) | — | active |
| c30 | 8 / 10 | d01 UNRUN (0; 0) | — | — | active |
| c31 | 8 / 11 | d01 UNRUN (0; 0) | — | — | active |
| c32 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: REFERENCE_INIT_CONTROL_ONLY |
| c33 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c36 | 8 / 1 | d01 UNRUN (0; 0) | — | — | active |
| c41 | 8 / 1 | d01 UNRUN (0; 0) | — | — | excluded: DEFAULT_GAUSSIAN_CONTROL_ONLY |
| c42 | 8 / 0 | d01 UNRUN (0; 0); d02 UNRUN (0; 0) | — | — | active |
| c44 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: LAKEMASK and LAKE_DEPTH all-zero; no actual lake cells; superseded by authentic-lowland c66 |
| c45 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c46 | 8 / 1 | d01 UNRUN (0; 0) | — | — | active |
| c47 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c48 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: experimental, column-oracle-qualified; no coupled reference: original CPU-WRF crashes (SIGSEGV ~00:04) on Swiss and Madrid with km_opt=3 |
| c49 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c50 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c51 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c52 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Original WRF refused old analysis TITLE; superseded by c77 |
| c53 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Original WRF refused old analysis TITLE; superseded by c78 |
| c54 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Original WRF refused old analysis TITLE; superseded by c79 |
| c55 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c58 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Original sf_ocean input-header refusal; replaced by original-real sf_ocean1 c84 |
| c60 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c62 | 5 / 0 | d01 INCOMPLETE (1; 0) | — | — | active |
| c63 | 15 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Owner12:59Z: MP15 experimental, column-oracle-qualified; upstream prehistory SIGSEGV; no coupled reference |
| c64 | 7 / 0 | d01 INCOMPLETE (1; 0) | — | QHAIL | active |
| c65 | 38 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c66 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Original WRF requires lake depth or explicit default mode; superseded by c83 use_lakedepth0/default50m |
| c67 | 8 / 0 | d01 UNRUN (0; 0) | — | — | excluded: Snow5 packed into actual Madrid c73; retained separate lowland CPU control |
| c68 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c69 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c70 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c77 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c78 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c79 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c71 | 8 / 1 | d01 UNRUN (0; 0) | — | — | active |
| c72 | 8 / 1 | d01 UNRUN (0; 0) | — | — | active |
| c73 | 8 / 1 | d01 UNRUN (0; 0) | — | — | active |
| c83 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c84 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c86 | 8 / 0 | d01 UNRUN (0; 0) | — | — | active |
| c87 | 55 / 0 | d01 INCOMPLETE (1; 0) | — | D_ICE, D_ICE2, D_ICE3, ICE_TYPE, ICE_TYPE2, ICE_TYPE3, PHI_ICE, PHI_ICE2, PHI_ICE3, REFL_10CM, RHO_ICE, RHO_ICE2, RHO_ICE3, V_ICE, V_ICE2, V_ICE3 | active |
