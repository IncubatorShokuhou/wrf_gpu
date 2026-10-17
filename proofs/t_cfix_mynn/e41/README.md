# F20 E41 PROD d01 repair (manager 03:55Z, 2026-10-16)

v-prep F20 FINAL on e7e37390cc: D6 5/5 PASS, but E41 FAIL on PROD d01 (jit + AOT): two kInput transpose
fusions with 803 and 694 optimized-HLO instructions (> 500). WN3, Swiss and PROD d02 passed.

## Attribution [M]
Source: the pinned GPU rich JIT blob
`<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/PROD/e41/pins/1_jit__advance_chunk_fori-feb70…`
(`extract_hlo.py`, then stack frames from HloModuleProto field 17 via `fusion_sources.py` / `where_funcs.py`).

- 803 = Noah-MP surface step (480 instr: phasechange, water_hydro/_infil, layer helpers) multi-output-fused
  with MYNN column prep (`step_mynn_pbl_column_with_pblh`, f32[8400,1]) plus one transposed pred root.
- 694 = Noah-MP energy/soil_thermo (592 instr) plus three transposed f32[120,70] roots that feed
  dycore/MYNN transposing fusions.
- No F20 op is inside either fusion. The F20 limit view sits in its own transposing fusions (295/322 HLO).
  F20 tipped a multistable small-grid multi-output-fusion choice (A22 / e41-class). PROD d01 has
  120x70 = 8400 columns, above `small_grid_firewall`'s 4096.

## Deviceless A/B (`dl_prod_d01.py`, E156 recipe; receipts in `receipts/`)
Setup: CPU init through the portable facade, then native Triton trace and CUDA lowering of the stepping
program, then a CompileOnly compile for the run's captured sm_120 target (b753abdd, autotune 0). Env as
the GPU run: FAST1 CENSUS1 CENSUS_SLIM0 AOT_SLIM0 RRTMG_MP_RE1. Cores 0,1, RAM slot, no GPU.

| arm | tree | transpose fusions | > 500 HLO | max | STACK | kernels |
|---|---|---|---|---|---|---|
| calibration | e7e37390cc (GPU-measured F20) | 663 | 803, 694 (= GPU) | 803 | 152 (= GPU) | 2414 (GPU 2418) |
| pre-F20 | 9a6faf3048 (src diff = F20's 3 files) | 744 | none | 407 | 152 | 2470 |
| main | 04cb09f816 | 663 | 803, 694 | 803 | 152 | 2414 |
| bisect: limit view off | diag f6400b18d0 | 744 | none | 407 | 152 | 2470 |
| bisect: mass-average w | diag e68d16c608 | 663 | 803, 694 | 803 | 152 | 2414 |
| C1a output-only fence on the handles | 103f855305 | 663 | 803, 694 | 803 | 152 | 2414 |
| C2a input fence | diag 1ee76b20be | 724 | 612 (sfclay psi + Noah precip_heat + KF) | 612 | 152 | 2459 |
| C2b input + output fence | diag 34a5ba6f69 | 724 | 612 | 612 | 152 | 2459 |
| C3 native (z,y,x) level-0 operands | 6b08d1ea14 | 650 | 1085, 563 | 1085 | 184 | 2394 |
| C4 forced PBL-entry fence | diag 24bbefdfd7 | 841 | 661 (sfclay + Noah) | 661 | 32 | 2498 |
| **C5 re-formation under lax.cond** | **ffabfefaf4** | 752 | **none** | 407 | 152 | 2480 |

Final head 4fab1d93a4 (C5 with the pre-F20 PSFC seam kept; `receipts/DL_c5b_*`), every stepping
program the GPU gate scans:

| program | transpose fusions | > 500 | max | STACK | kernels (GPU F20) | deviceless compile s (F20/main) |
|---|---|---|---|---|---|---|
| PROD d01 | 752 | none | 407 | 152 | 2480 (2418) | 178.7 (178.9 / 182.6) |
| PROD d02 | 129 | none | 360 | 32 | 2039 (2031) | 294.0 |
| WN3-0227 d01 | 756 | none | 407 | 152 | 2486 (2488) | 189.9 |
| WN3-0227 fused_d02 (d02+d03) | 299 | none | 400 | 32 | 2823 (2819) | 545.7 |
| Swiss d01 | 655 | none | 364 | 16 | 2453 (2459) | 190.5 |

Kernel counts are static (the cond branch's ~10 kernels are compiled but never execute on PROD/WN3).
Compile seconds are single samples on 2 shared cores (C5 vs final fused_d02 481 / 546 s = noise band).

The deviceless compile is a prescreen. Calibration reproduced the GPU verdict exactly here, but the
actual GPU executable scan stays authoritative (v-prep reruns it).

## Fix (C3 + C5, `coupling/physics_couplers.py`)
`_mynn_driver_limited_handles`:
- The bound test `beyond` (four 2-D compares on State hfx/qfx) stays in the step graph.
- The F90:859-876 re-formation runs in `lax.cond(jnp.any(beyond), reform, pass-through)`. Its kts
  operands (phy_prep rho, qv, p_hyd, pi_phy) are read inside the branch at level 0 of the native
  (z,y,x) fields: the same values as the column view's `[..., 0]`, without its layout-pinned
  (`GPUWRF_LAYOUT_PIN=cols`) 3-D transposes.
- `_surface_fluxes_from_state` keeps the pre-F20 PSFC seam `_surface_column_view(state, grid).psfc`
  (pinned by b-phys BP113 `test_actual_producer_passes_psfc_and_previous_el`). The branch computes its own
  `p_hyd`, so the hydrostatic profile is traced twice (in the jaxpr, about +1.65k lines in total with the
  branch's rho/Exner/driver statements) but executes only when a limit fires.

Values:
- No column beyond the limits (always on PROD/WN3: 48 CPU-WRF frames per domain, HFX -250..770 W/m2,
  QFX ≤ 2.3e-4, MY10): returns the shared handles, identical to F20's all-false select.
- Some column beyond: the same statements, operand values and selects. Op by op this is BYTE vs the
  eager reference. A compiled branch on XLA:CPU can differ by ≤ 1 REAL4 ulp there because of divide
  reassociation (E130/E186). F20's in-jit code was compiled as well.

## Identity / jaxpr [M CPU, official probe, lane-local LANE dir]
- Main 04cb09f816 reproduced: Swiss 7379c7d7…, PROD d01 6487f5b0….
- Final 4fab1d93a4: Swiss `a6166b20e08453f7a6d72167807c22610bf0453aebe17fc663d48633fff40cef`,
  PROD d01 `4e1879c2b1f1581867fa6d3862a0712a7477433880c4deb6a193574818d20865`.
- Anonymized jaxpr diff vs main (`*_jaxpr_diff_vs_main04cb.txt`): +1673 / -19 lines. These are the
  cond and its branch (hydrostatic p_hyd, phy_prep rho, Exner, driver re-formation, selects) minus F20's
  in-graph view slices and selects.

## Tests / mutants [M CPU]
- Final head: F20 module + consumer set 538 PASS (incl. b-phys test_mynn_closure_carries, which failed on
  the intermediate C5 PSFC refactor and is the reason the PSFC seam was restored).
- `tests/v025/t_cfix_mynn`: 31 PASS. New: `test_limit_reformation_is_traced_only_inside_the_cond`
  (structural, recursive jaxpr: no pow/div/log/transpose/select_n in the step graph, all inside the
  true branch) and `test_inside_the_limits_the_mynn_view_returns_the_shared_handles` (BYTE).
- Mutants: 8/8 killed by assertions, 0 setup errors (`mutants_e41_c5.json`; final head `mutants_c5b.json`). New ones: M7 (re-formation
  back in the step graph) and M8 (`all` instead of `any`).

## Perf [I/U]
Per d01 step, the executed work drops to pre-F20 plus one (ny,nx) bound-test/any-reduce and one
conditional node. F20's always-on view ops and their 3-D transposes are gone. The branch never executes
on PROD/WN3. The step already holds 5 conditionals of the same class (CONDITIONAL command buffers on by
default). Host syncs and warm per-root time stay [U] until v-prep's GPU rerun.

## C5 host-transfer FAIL → C7 branch-free (manager 09:17Z, 2026-10-16)
v-prep's C5 GPU gate on batch60 b09884fb87: E41 PASS on all programs, but hour 3 of PROD (66 root steps) had
+198 one-byte DtoH copies and +198 `cuStreamSynchronize` vs the F20 baseline (3 per root). Warm root time was
53.87 ms vs 52.06 ms (+3.5 %).

Attribution [M GPU HLO from v-prep's C5 pins, `extract_hlo.py`; I for the capture mechanism]:
- The new instruction is `%cond.242` (d01) / `%cond.215` (d02). The true branch is 4 fusions; the false branch is
  3 copies. Neither branch contains a loop.
- 198 = 66 roots x 3 d02 steps per root: one host predicate read per d02 step.
- d02 advances `n_steps=3` inside a host `WhileThunk`. That body is not CUDA-graph captured (WHILE command buffers
  are off, A42), so a `conditional` in it reads its predicate on the host every step.
- d01 (`n_steps=1`, loop removed by XLA) gets a CONDITIONAL graph node instead.
- Swiss (`n_steps` = output cadence) and the WN3 fused child loop would pay the same. No `lax.cond` in the step
  is device-only.

C7 (`_mynn_driver_limited_handles`, `_wrf_phy_prep_rho_from_state(kts_only=True)`):
- No control flow: the limited handles come from `jnp.where(beyond, re-formed, shared)`.
- The kts operands are formed as `(ny, nx)` arithmetic from kts slices of the native `(z, y, x)` State fields:
  phy_prep rho from the ph faces kts/kts+1, qv kts, and pi_phy from p kts. No 3-D intermediate is involved: not
  the full-column rho/Exner that is shared with the MYNN column prep (C3: 1085/563), and not the cols-pinned
  view (F20: 803/694).
- The values are the same statements on the same REAL4 operands as C5. In-range columns (always the case on
  PROD/WN3) get the shared handles, BYTE by construction (select).
- Cost [I]: about a dozen small `(ny, nx)` kernels per step that run every step. Compare: C5's branch never ran,
  but its predicate cost a host sync each d02 step.

Deviceless gate on final 43c7fd4dda (snapshot c7b, same recipe as above). Control flow is counted in the
optimized HLO as `conditional`/`while`/`custom-call`. The F20 column comes from the GPU-pinned F20 executables
(`<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/*/e41/pins`):

| program | transpose fusions | > 500 | max | STACK | kernels (GPU F20) | cond/while/cc C7 | F20 GPU | C5 GPU |
|---|---|---|---|---|---|---|---|---|
| PROD d01 | 743 | none | 441 | 32 | 2482 (2418) | 5/12/137 | 5/12/137 | 6/12/137 |
| PROD d02 | 129 | none | 360 | 32 | 2032 (2031) | 4/11/135 | 4/11/135 | 5/11/135 |
| WN3-0227 d01 | 750 | none | 441 | 32 | 2491 (2488) | 5/12/137 | 5/12/137 | 6/12/137 (deviceless) |
| WN3-0227 fused_d02 (d02+d03) | 299 | none | 400 | 32 | 2817 (2819) | 8/20/297 | 8/20/297 | 10/20/297 (deviceless) |
| Swiss d01 | 658 | none | 389 | 16 | 2458 (2459) | 4/10/144 | 4/10/144 | 5/10/144 (deviceless) |

- Every program has F20's `conditional`/`while` count. C5 added one per domain, which is the host predicate
  v-prep measured. The 1-root nsys count is still a GPU arm [U].
- The PROD/WN3 d01 maximum of 441 is the sfclay ψ-function fusion (`surface_layer.py` `_psi{m,h}_{stable,unstable}_full`
  + `_table_lookup`, `fusion_sources.py`); pre-F20 it was 406. No F20 code is inside it. It is a small-grid fusion
  reshuffle with 59 instructions of margin to the 500 gate.
- STACK falls from 152 B (pre-F20/F20/C5) to 32 B on both d01 programs.
- Deviceless compile seconds (single samples, 2 shared cores): d01 182.5, d02 295.1, WN3 d01 183.2,
  fused 465.3, Swiss 172.9. C5 was 178.7/294.0/189.9/545.7/190.5.

C7 status when the head was frozen: F20 module 31 PASS on the worktree; quick set 50 PASS on snapshot c7b
(F20 module + b-phys closure carries + fltv + lake wiring). The consumer set, mutants M1–M10 and the release
identities were still running on c7b. Their receipts go to `<USER_HOME>/wrf_gpu2_lanes/t-cfix-mynn/e41/`
(`c7b_chain.log`, `c7b_consumers.log`, `mutants_c7b.json`, `identity/e41_c7b_*`) and to MAILBOX, without moving
this head. HLO dumps: `<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-mynn/e41_hlo/DL_c7*` and `.../c5gpu`, `.../f20gpu`.
