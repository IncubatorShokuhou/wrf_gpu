# F7 end-to-end pristine-WRF oracle on real c05 (G3, cu=5) — receipt

## FINAL (release path, RTX 5090): pre-registered gate PASS
`f7_c05_oracle_release_rtx5090.json` — `f7_c05_release.py` via `run_c05_release.sh` (official lock, lease t-cfix-cu,
manager-approved short arm; detached snapshot 54cdee223, release defaults, asserted `platform == gpu`, private cache).
**Source binding:** the traced c05 release program of 54cdee223 is identical to the final lane tree's (sha 481eb074…,
`proofs/t_cfix_cu/identity/identity2.json`, official probe with O1_CASE = c05), so these numbers bind to the final tree.

| step 1, interior | rel RMS / ratio | pre-registered gate |
|---|---|---|
| RTHFTEN (port vs pristine WRF) | 0.000721 | <= 0.10 PASS |
| RQVFTEN | 0.000455 | <= 0.10 PASS |
| mutant no dry conversion (RTHFTEN) | 0.0426 = **59.1x** | >= 3x PASS |
| mutant zero RTHFTEN / RQVFTEN | 1.0 = 1387x / 2196x | >= 3x PASS |

Steps 2/3 (not gated): RTHFTEN .0027/.0036, RQVFTEN .0008/.0016, no-dry ratio 17.3x/12.6x.
Attribution (`attribution_release_rtx5090.txt`): end-of-step-1 w differs 5.9e-5 m/s rms (legacy arm: 1.4e-3), dry theta
5e-5 K; msfty^0, LS scale 1.00007; no convective columns at step 1.
**E41 cubin gate for this arm: UNKNOWN** — `wave_cubin_gate.py` was invoked without PYTHONPATH to the tree (ImportError,
not a cubin verdict) and the arm script then deleted the private cache unconditionally (E166/E177 repeat, logged to
INBOX). The c05 program's E41 scan is carried by the c05 composite GPU rerun (v-prep) or a re-run on request.

## Earlier arm (wz1080 legacy path, debug evidence) — superseded, kept for the record


**What:** the port's held WRF forcing `carry.cu_forcing = (RTHFTEN, RQVFTEN)` after own steps 1..3 vs the same fields
written by pristine serial WRF 4.7.1 on the same wrfinput/wrfbdy/namelist (history every 18 s step).

**Reference (CPU-WRF):** `<USER_HOME>/src/wrf_pristine/WRF/main/wrf.exe` (sha 52f914cb…) copied privately with only the
two hard-coded E104 dump-path strings byte-patched to a lane dir (`CPU_executable.json`; numerics untouched).
Namelist = c05 original with run_seconds 54, `history_interval_s = 18` and `iofields_cfix.txt` adding
RTHFTEN/RQVFTEN (+ the other cumulus/PBL rates) to the history stream (`pristine_namelist_vs_c05orig.diff`).
Inputs and history file SHA-256: `inputs.sha256` (lane scratch `<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-cu/{c05port,c05w}`).

**Port arm:** `f7_c05_compare_gpu.py` (this dir) on the wz1080 box (GTX 1080 Ti, sm_61), tree 791fdcde2 (product tree
e0212cbe4 = this branch's F7 code before the rebase; F7 files unchanged by the rebase), `GPUWRF_FAST_DEFAULTS=0` +
`WZ_PALLAS_PORTABLE=1` (legacy XLA path; the box cannot compile Pallas). Asserted `platform == gpu`. F7 capture /
phy_prep_part2 finish are the same code on the release path (advance_stage). **wz1080 debug evidence, not a release
verdict**: the release-default path is covered by the composite GPU reruns (c05 c13 c17 …). Local CPU attempts:
whole-step CPU XLA compile memcg-OOM at 12G (release and legacy), eager mode cannot run the REAL32 ACM2 island.

**Pre-registered gate (notes 2026-10-15 19:25Z, before any result):** step 1, interior (5-point ring excluded):
relative RMS port vs pristine <= 0.10 for RTHFTEN and RQVFTEN, AND each mutant (no dry conversion; zero forcing)
>= 3x the correct arm's error. Steps 2-3 reported, not gated.

| step 1, interior | rel RMS | vs gate |
|---|---|---|
| RTHFTEN (port) | 0.0160 | <= 0.10 PASS |
| RQVFTEN (port) | 0.0293 | <= 0.10 PASS |
| mutant zero RTHFTEN / RQVFTEN | 1.0 / 1.0 | 62x / 34x >= 3x PASS |
| mutant no dry conversion (RTHFTEN) | 0.0461 | **2.88x < 3x MISS** |

Steps 2/3 (not gated): RTHFTEN 0.0146/0.0206, RQVFTEN 0.0290/0.0332; no-dry mutant 3.31x/2.43x.

**Verdict: accuracy gate PASS; one pre-registered discrimination criterion narrowly MISSES (2.88x).** The use_theta_m
dry conversion is a few-percent correction, comparable to the 1.6 % end-to-end residual of a legacy-path GPU
trajectory vs serial CPU-WRF, so this arm cannot separate it at 3x. The conversion itself is pinned independently
by the REAL4 literal of the pristine phy_prep_part2 statements (tests/test_t_cfix_cu_g3_forcing.py, <= 1 ulp, plus
mutant "F7 no dry conversion" killed). Gate not relaxed.
Note: `state_theta_rel` in the JSON compares the port's moist theta_m with history `T` (dry) — the ~0.7 K offset
is the theta_m/theta difference, not a model error (unused by the gate).
