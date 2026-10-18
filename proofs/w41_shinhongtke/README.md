# Shin-Hong (`bl_pbl_physics=11`) TKE vs original WRF — lane w41-shinhongtke

## 1. The contract premise (diag=1 kernel TKE deviation) is already fixed
The "max abs .019 / rel .56" deviation came from the s2-cammg probe (2026-10-14). Lane
t-shinhong then corrected the kernel and caller (index/PI/qc-qi/CORF/RQC-RQI/pblflg/literal
REAL4 order; `proofs/t_shinhong/`), merged in 668f27f69 and contained in v0.4.0. On current
main (incl. the w41-topowind CTOPO hunks) its gates pass unchanged: 20/20
(`tests/v040/test_shinhong_wrf_oracle.py`, `test_shinhong_tke_caller.py`). The
`proofs/s2_cammg/README.md` sentence that still called it an accepted caveat is annotated.

## 2. What was still wrong: history TKE_PBL / EL_PBL [M]
`C07_TKE_PBL_HISTORY.json` (`c07_tke_history.py`, existing Swiss c07 runs, not ALISIOS data):
original CPU-WRF writes the shared `tke_pbl` array — REAL 0.005 (`EPSQ2L/2`, shinhonginit,
`module_bl_shinhong.F:2270-2277`) on all 44 mass levels from lead 0 on, 0 at the unused Z top.
The GPU history held 0 everywhere in both 24 h runs (v-prep c07 src 44b7b226b, w41-topowind
c07tw1 src 922e4de092; bias -0.00489 = -0.005·44/45, REPORT_ONLY so never gated). Cause: the
writer had no bl=11 source for `TKE_PBL` (the port holds the array in `State.qke`, which
only appeared as the superset variable `QKE`: 0 at t0 from the wrfinput overlay, 0.005 later),
so the generic zero fallback was written. With `shinhong_tke_diag=1` the diagnosed TKE and
`EL_PBL` (`:1604`, the adapter never published it) would both have been lost from history.

## 3. Fix (c5ba00f8d8)
- `io/wrfout_writer._shinhong_tke_history`: for bl=11 the existing `TKE_PBL` slot is filled
  from `State.qke` (or `State.cam_tke_pbl` incl. its top memory under CAM-MG) with the Z top 0;
  nonfinite refused. Host-only; the variable inventory is unchanged.
- `coupling/scan_adapters.shinhong_pbl_adapter`: `shinhong_tke_diag=1` publishes the kernel's
  `el_pbl` (zero at kts, `:673/:1604`; `el_pbl` is already a post-RK replace leaf and a restart
  leaf). `diag=0` keeps the held Registry zero, so the diag=0 adapter jaxpr is IDENTICAL to main
  (`ADAPTER_IDENTITY.json`) — C3/c07 (diag=0) step programs do not change; only diag=1 adds one
  (ncol,nz) REAL output store.

Gates [M CPU]: `tests/v041/test_shinhong_history.py` 11 PASS — cold seed + zero top, diagnosed
TKE/EL, CAM verbatim, YSU/MYNN untouched, nonfinite refusal, lead-zero frame through the
production per-domain writer (t0 diagnostics overlay QKE=0, TKE_PBL must still be 0.005), actual
operational caller + post-RK replace for diag 0/1, real kernel EL. `MUTANTS.json`: genuine
AssertionError kills, 0 setup errors (M1-M7 history; DX1-DX2 below). The existing caller stub
gained the kernel's `el_pbl` return key (test-only).

## 4. Scale-aware coverage at the G2/WN3 nest resolutions [M CPU]
t-shinhong's real columns are 3 km only. `build_dx_oracle.py` reruns the same 36 columns ×
modes through the SHA-checked pristine builds at dx = 1 km and 9 km (`DX_COVERAGE.json`):
literal op-by-op port BYTE on TKE/EL/EXCH_H/u/v/θ/qv/PBLH/KPBL (qc/qi differences only in
cells < FLT_MIN, E117); compiled XLA:CPU 0 normal cells outside WRF's ±1-ULP input envelope
(+2 output ULPs, 67,216 pristine calls per dx); init zero == inf. Approximate clamp census:
pthl/pthnl pmax active, pmin never. Tracked fixture `fixtures/shinhong_dx_oracle.npz` (48 rows)
+ `tests/v041/test_shinhong_dx_oracle.py` (strict, no disclosed cells). Mutants DX1/DX2 (drop the
pthl/pthnl pmax clamp) are killed by it — and also by the 3 km v040 test, so the fixture adds
resolution coverage, not a unique kill.

## 5. Same writer gap in five more PBLs (manager 22:24:56Z) [M]
v0.4.0 v-prep composites: GPU `TKE_PBL` ≡ 0 for MYJ (c03), QNSE (c26), BouLac (c06, c57), CAM-UW
(c28, c29, c89), GBM (c08); only KEPS (c11) was right. The port holds WRF's shared array in
`State.qke` (MYJ/QNSE q² = 2·TKE, WRF `TKE_MYJ = 0.5·Q2K`; BouLac/GBM TKE) or in
`carry.camuw_pbl.tke_pbl` (CAM-UW interfaces, nz+1). Commit 2af4c561aa: `_pbl_tke_history`
(factor table, zero Z top) + the nested writer passes the CAM-UW carry as a diagnostic; host-only,
no traced code changed. `PBL_TKE_HISTORY_PINS.json` (`pbl_tke_history_pins.py`): the old runs
wrote the post-RK held qke as the superset `QKE`, so factor·QKE is exactly the new writer output;
vs original CPU-WRF `TKE_PBL` over 24 leads: MYJ r ≥ .992 (ratio .985–1.006), QNSE r ≥ .999,
BouLac r ≥ .998, GBM r ≥ .990, Shin-Hong BYTE 24/24; CPU Z tops all 0.
Not fixed: `EL_PBL` of MYJ/BouLac/GBM is not held in State (adapter change = step-program change).

## 6. Cold TKE seeds (manager 22:30:53Z) [M CPU]
The loader seeds `State.qke` with MYNN's cold-start profile for every PBL (wrfinput QKE = 0); only
Shin-Hong/QNSE/GBM re-seeded. Commit ecac4bc22b applies the original `.not.restart` inits on kts:ktf in
`_initial_carry_for_run`: MYJ `TKE_MYJ = EPSQ2 = .2` (qke .4), BouLac 1e-4, CAM-UW `epsq2 = .2`
(carry only; the CAM-UW step never reads it), `cam_tke_pbl[:-1]` under CAM-MG. Lead-0 `TKE_PBL`
is BYTE vs the original CPU-WRF bytes of c03/c06/c57/c28/c29/c89 (0x3e4ccccd / 0x38d1b717); YSU and
MYNN initial qke untouched. Leaf shapes/dtypes unchanged → step programs unchanged; initial data
changes for MYJ/BouLac (prognostic TKE, first hour), CAM-UW history only.

Release identities: on the GPU-measured tree 280a0126fe Swiss `3356e24924a9…7c57` / PROD `196aed9f9973…911e`
(= v0.4.0, EXACT); after rebasing onto main 8d5a7f42f2 (w41-bdyrec-product changed the defaults by design)
`RELEASE_IDENTITY_{swiss,prod}.json` give Swiss `e1102c204a79…` / PROD `9b676bec1609…` = main's new identities —
this lane leaves the default programs unchanged. bdyrec is numerically BYTE vs v0.4.0 and orthogonal to these
hunks, so the GPU results on 280a0126fe carry forward (validation-reuse rule).

## 7. GPU qualification (one lock session 23:49:37Z–00:36:22Z, 46.8 GPU-min) [M GPU]
`GPU_ARMS.json` (`collect_gpu_arms.py`), final tree 280a0126fe, release-default CLI, asserted RTX 5090,
one shared cache (`E41_WAVE_GATE.json`). D6 = `scripts/compare_wrfout_grid.py` with the frozen manifest
vs ORIGINAL CPU-WRF, every frame; `tke_compare.py` for TKE_PBL/EL_PBL (no D6 bound exists).

| arm | what | wall | D6 bounded fails / CPU-only vars | TKE_PBL lead 0 | TKE_PBL r (leads ≥1) | EL_PBL |
|---|---|---|---|---|---|---|
| c03 | MYJ 24 h (cold seed) | 888 s | 0 / 0 | BYTE | .992–.998 | GPU 0 (disclosed) |
| c06 | BouLac 24 h (cold seed) | 521 s | 0 / 0 | BYTE | .9993–.99999 | GPU 0 (disclosed) |
| c07sd1 | Shin-Hong diag=1 24 h vs new CPU-WRF | 895 s | 0 / 0 | BYTE | .993–.9996 | r .951–.999, lead 0 BYTE |
| c28 | CAM-UW 6 h (seed history-only) | 479 s | 0 / 0 | BYTE | .978–.995 | not in WRF package |

All tops 0, 0 nonfinite. E41 overall FAIL = the v0.4.0 composite values per scheme (pre-existing, opt-in
disclosure): conservative kInput>500 flags on the rich JIT entries; hard STACK 2456 B only in c03 MYJ
(`loop_select_fusion_716` 2456, `loop_exponential_maximum_multiply_select_fusion` 2032, identical in the
v0.4.0 c03 gate); c06 40 B, c28 96 B, c07sd1 16 B = their v0.4.0 values. c57/c29/c89/c26/c08 carried forward on
the lead-0 pins + QKE evidence (§5). Cache kept for the critic at
`<DATA_ROOT>/wrf_gpu2/scratch/w41-shinhongtke/cache_w41sht` (moved from /home, path-independent scan).
Perf: no new TIMESTEP transfer, solve, scan, carry leaf or table; seeds are init-time fills; diag=1 adds one
(ncol,nz) f32 step output (EL_PBL, existing State leaf); walls ≈ v0.4.0 cold walls. Output-time cost (outside the
timestep, critic 01:46Z): TKE_PBL is one host (nz+1,ny,nx) f32 buffer per frame. bl 9 adds carry.camuw_pbl.tke_pbl
to the writer diagnostics -> one more (nz+1,ny,nx) f32 leaf in the batched wrfout device_get per frame (~0.318 MB
Swiss). bl 2/4/8/12 reuse State.qke, already fetched when the stream writes QKE (FULL); a subset stream requesting
TKE_PBL without QKE fetches State.qke (nz,ny,nx) once per frame outside that batch. Default MYNN: unchanged.
