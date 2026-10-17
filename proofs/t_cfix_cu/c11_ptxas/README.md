# c11 cold-compile ptxas pathology: attribution + WDM6 small-grid firewall (2026-10-16)

## Symptom
c11 = Swiss 42x42x44, mp16 WDM6 / cu96 NSAS / bl17 KEPS / sf1 / Noah-MP / RRTMG, main 5c15881ede (src == main
0e645e8c1c). Cold compile hung > 57 min in ONE ptxas job (`input_and_compare_convert_maximum_select_transpose_fusion`,
ptxas RSS 25.9 GiB; 12G/24G scope OOMs earlier; v-prep `C11_COLD_COMPILER_TRIAGE.json`, preserved PTX
`<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/C11_COLD_TRIAGE_5c15881ede/input_and_compare_fusion.ptx`, sha 551173c2).

## Method (CPU only, GPU 0)
`dl_dump.py` = v-prep `lower_only_case.py` (sha 9d034639, lowering unchanged: actual c11 inputs, CPU loader, native
CUDA trace, captured sm_120a target) + deviceless `lowered.compile(xla_gpu_autotune_level=0)` with an XLA dump of
the optimized HLO; a watcher records every ptxas child (elapsed, peak RSS, kept input PTX). Env = v-prep GPU arm
flags (FAST_DEFAULTS=1, CENSUS=1, CENSUS_SLIM=0, AOT_SLIM_MODULE=0, RRTMG_MP_RE=1). Lane cores, nice 19, RAM slot.
`attribute_fusion.py`/`frames.py`: fusion -> source lines via the HLO stack-frame tables; `scan_fusions.py`: all
fusions + E41 HLO unit (kInput transpose fusion > 500 instr); `canon_diff.py`: multiset of canonicalized fused
computations (names/metadata stripped).

## Attribution [M]
The deviceless optimized HLO contains the same fusion name; its callees match the preserved PTX `.func` clones.
* 1613 instructions, kInput, 23 outputs f64/f32/pred[44,1764], 20 params; source = WDM6 (`microphysics_wdm6.py`
  `_wdm6_minor_loop`, `_slope_wdm6`, `_qsat_fpvs`, `_diffac`, `_viscos`, `_venfac` ... via
  `scan_adapters.wdm6_adapter` -> `_apply_post_rk_microphysics`). Not cumulus.
* Hero = transpose of Nn: f32[44,42,42]{0,1,2} (column-major; output of the root `_root_flow_dep` fence,
  operational_mode.py:5747) -> post-RK `where(step==1, ccn_conc, Nn)` (:8590) -> WDM6 clamp (wdm6:596) ->
  f64[44,1764]. The WDM6 column body fused in as the transpose epilogue -> PTX 24 MB, 102 cloned callees,
  59k mul/76k fma f64 (recomputation, the A19/E38 signature).
* Small-grid e41-class (Swiss 1764 columns <= SMALL_GRID_COLUMNS 4096).

## New since F9? Bisect, same recipe [M]
| tree | that fusion |
|---|---|
| de374a3c00 (c11 last compiled; GPU module compile 3m43s, E41 FAIL) | 1516 instr, 23 out |
| 6478f6d528 = pre-batch56 | 1516 |
| c91b4fd710 = post-batch56 (F9 cumulus RK coupling) | 1613 |
| 5c15881ede | 1613 |
The fusion PRE-EXISTS; batch56 shifted whole-module fusion decisions (+97 WDM6 instructions; `microphysics_wdm6.py`
unchanged since v060) past a ptxas cliff. The cause is the layout seam, not the F9 arithmetic.

## Fix (adc3fe84d9)
`wdm6_adapter`: `_small_grid_firewall(mp(state.Nn), (ny, nx))` — one optimization_barrier on the Nn column view
(e41-class precedent `kernels/layout_pin.small_grid_firewall`; grids above 4096 columns trace nothing). Variants
tried deviceless (`variants.sh`, experiment-only commit a62dd2d6b3): barrier on the 3-D inputs does NOT help
(1602-instr kInput transpose fusion remains: layout propagates through the barrier, E40); Nn column view,
row-major pin + barrier, all column views: all clear the gate; the single Nn view is the smallest change.

## Results [M CPU]
| | before (5c15) | after (adc3fe84d9) |
|---|---|---|
| E41 HLO kInput-transpose > 500 (whole module) | 1 (1613) | 0 |
| fused computations canonically identical | — | 3309 / 3314 (changed: 5 -> 6, all WDM6) |
| WDM6 largest fusion | kInput 1613, 23 out | kLoop 1519, 46 out (+ kInput 430, + 14-instr Nn transpose) |
| deviceless module compile | never finished | 181 s (autotune 0, 8 lane cores) |
| ptxas in-situ (XLA child) | > 57 min, 25.9 GiB, killed at cap (v-prep) | 11 s, 0.71 GiB (one module PTX, 2662 kernels) |
| ptxas offline, same binary/argv, taskset core 29 | TIMEOUT at 1800 s (rc 124), 24.69 GiB peak (flat from ~200 s), no cubin | 12.3 s, 0.73 GiB, cubin written |
| native E41 (cuobjdump) | n/a (no cubin) | non-leaf > 1k SASS: 0; STACK > 256: 3 pre-existing (below) |
The PTX of the committed fix is byte-identical to the tested variant (sha 921072460f).

Pre-existing STACK > 256 B (fusions canonically identical in the unfixed module, so not caused by the fix): WDM6
`_precip_out` reduce 512 B, WDM6/den_delz reduce 416 B, sfclay_revised_mm5 `_table_lookup` 400 B. c11 E41 was
already FAIL at de374; c11 stays opt-in E41-disclosed.

Release identity (`identity6`, official probe): Swiss 3356e249 / PROD 196aed9f EXACT (= main defaults); c11
583a7c87 (main) -> 8af8e66f (fix), intended.

## Values
The firewall is an identity op. CPU: `test_firewall_is_value_identity_on_cpu` bitwise. GPU BYTE versus the unfixed
program is NOT measurable: the unfixed program does not compile (ptxas > 57 min). XLA's PTX uses explicit `.rn`
on every mul/add (ptxas never contracts them); FMAs come from LLVM within one kernel, so the WDM6 repartition
(5 -> 6 fusions) can change f64 rounding where a mul->add edge crosses the new kernel boundary [U, not measured].
Fidelity gate = c11 D6 vs original CPU-WRF on GPU. Optional GPU BYTE proxy: pre-batch56 tree +/- the same
firewall (both compile), carry/wrfout compare.

## Tests + mutants
`tests/test_t_cfix_cu_wdm6_small_grid_firewall.py` (4): Nn argument is exactly the firewall output (spy), one
traced barrier on small grids; Nn operand values equal an independent column view; large grid (64x65) traces no
barrier; CPU bitwise identity. `mutants.py` -> `mutants_result.json`: 5/5 genuine AssertionError kills (removed,
wrong operand, unconditional barrier, operand perturbed, firewall not identity).

## Not covered / open
* WDM5 (mp14) and WDM7 (mp26) adapters share the column-major Nn pattern; their FINAL composites compiled, so
  they are unchanged here (latent; same fix if a cliff appears).
* GPU timing of the WDM6 repartition unmeasured (perf effect expected small: one extra 14-instr transpose kernel).
* Cost: no new arithmetic; +1 small kernel per MP call on small grids; deviceless temps unchanged [I].

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/c11/` (HLO dumps gz, scans, canon diff, ptxas logs, cubin scan).
