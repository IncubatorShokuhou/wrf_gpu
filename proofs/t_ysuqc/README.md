# YSU PBL (`bl_pbl_physics=1`) vs pristine WRF 4.7.1 — lane t-ysuqc

## Oracle
- Source: unmodified `phys/module_bl_ysu.F` (`46ad9d21…`), `phys/physics_mmm/bl_ysu.F90` (`b2c9f11f…`),
  `phys/ccpp_kind_types.F` (`0cc7b2d7…`), lane-dir copies (E104), `-cpp -ffree-form`, built with
  `ysu_oracle_wrapper.f90`: a thin public `ysu(...)` call mirroring `module_pbl_driver.F:1208-1251`
  (QC3D/QI3D with RQCBLTEN/RQIBLTEN, FLAG_QC/FLAG_QI, PSIM=fm/PSIH=fhh, CTOPO/CTOPO2 = 1 as
  `start_em.F:1581` sets for topo_wind=0, RTHRATEN, YSU_TOPDOWN_PBLMIX, UOCE/VOCE, flag_bep false,
  module_model_constants values). Builds (`build_oracle.py`): `zero` -O0 no-FMA init zero (reference),
  `inf` init inf, `fma` -O2 -mfma (87 FMA instructions).
- Columns (census E154): 109 real 44-level CPU-WRF columns, both top-down modes = 218 cases:
  c02 (Swiss 2023-01-15, bl=1, mp1 Kessler: cloud water, FLAG_QI false), c40 (2026-10-13, bl=1, mp8,
  cloud-free), c07 (Swiss, mp9: cloud ice up to 1.1e-4) and c85 (2026-10-13, mp8: marine cloud). Picks:
  max/min HFX, max low cloud water, max low / column cloud ice, max PBLH, max cloud at the PBL top,
  2 random. RTHRATEN is not in wrfout: a deterministic real-structured profile (clear -2e-5 K/s,
  cloud top -4e-4 K/s, daytime in-cloud +1e-5 K/s). 14 columns depend on the top-down switch,
  19 rows are cloud-top-entrainment candidates, 68 have in-cloud Ri, 56 nonzero RQIBLTEN.
- Uninitialised reads: `zero` vs `inf` BYTE on every consumed output; only RQIBLTEN differs when
  FLAG_QI=.false. (`UPSTREAM_WRF_BUGS.md`; WRF never consumes it then; the port returns no qi rate).

## Deviations found and fixed (`src/gpuwrf/physics/pbl_ysu.py`, callers)
| # | deviation | WRF | class |
|---|---|---|---|
| 1 | no QC3D/QI3D input; RQCBLTEN/RQIBLTEN never produced | `bl_ysu.F90:390-421, 1186-1220` | caller (E11) + forecast-driving |
| 2 | `ysu_topdown_pblmix` ignored (port always 0; Registry default 1, all bl=1 composites): theta-li PBL enhancement, which can raise KPBL and set pblflg even for stable columns | `:691, 733-753` | forecast-driving |
| 3 | cloud-top radiative entrainment missing (RTHRATEN radsum, ent_eff, wstar3_2, hgamt2, xkzm(kpbl-1)=0, wscalek2) | `:840-899, 946, 962-966` | forecast-driving |
| 4 | in-cloud Richardson number (imvdif=1) missing above the PBL | `:993-1001` | forecast-driving |
| 5 | CTOPO is always present in the driver call: momentum surface diagonal `1+fric*vconvlim+ctopo*fric*(1-vconvlim)` needs tke_ysu/el_ysu, get_pblh and vconv | `:1256-1319, 1586-1693` | ULP-level |
| 6 | RTHBLTEN = TTEND/PI with TTEND = rate·PI (wrapper) | `module_bl_ysu.F:463`, `:1103` | ULP-level |
| 7 | literal REAL4 order: reciprocal-pivot `tridin_ysu`, ad/f assembly, powf for REAL exponents, integer powers as products, REAL4-folded runtime constants (EP1, cp/g, g·g, xlv·xlv, phifac·karman, 4·karman) | throughout | ULP-level |
| 8 | `ysu_topdown_pblmix` not bound from namelist.input (silently ignored) | Registry `ysu_topdown_pblmix` 1 | binding |

Callers: generic SF1 adapter (`scan_adapters.ysu_pbl_adapter`, REAL ABI under `enable_x64(False)`),
BEP/BEM urban (`urban_pbl_hook`, `urban_bep_pbl.ysu_urban_columns`) and SSiB (`ssib_ysu_step`) pass
QC3D/QI3D, RTHRATEN and the namelist switch; RQC/RQIBLTEN enter the RK-source path (generic and urban:
entry qc/qi restored, raw rates coupled once) or the State path.

## Result [M CPU]
- Literal path (op-by-op, `jax.disable_jit`) with exp/tanh evaluated by glibc (`--glibc`): BYTE on
  u/v/θ/qv/qc/qi rates, EXCH_H/EXCH_M, PBLH/KPBL/WSTAR/DELTA/U10/V10 for all 218 cases (qc/qi only
  cells with |rate| < 1e-30 next to subnormal inputs, E117). With XLA's own exp/tanh (1 ULP off glibc
  in 9 % / 58 % of float32 inputs; pow is bitwise glibc) a handful of cells differ by 1 ULP
  (`COMPARE_eager2.json`).
- Compiled XLA:CPU path: every cell of every output inside WRF's own ±1-ULP input envelope
  (`ulp_envelope.py`, 96 random ±1 ULP perturbations of every input per case = 20,928 original calls,
  plus the FMA build) widened by 2 output ULPs; KPBL identical (`COMPARE_jit2.json`).
- Before (old port, its caller: no qc/qi, top-down off; `COMPARE_before_oldport.json`): worst column
  |port−WRF|/column max u 1.85, v 1.68, θ 1.10 (1.62 vs the top-down-0 rows), qv 1.17 (1.67), EXCH_H/M
  1.00, RQC/RQIBLTEN missing (1.0); KPBL wrong in 11 of 109 top-down columns.
- Cost (`COST_old_vs_new_cpu.json`, CPU lowering, 8 columns): sequential level scans 11 → 6 (HLO while
  loops 10 → 6), temps 74 → 71 KB, ~34 % more fused elementwise work (top-down, cloud branch, TKE/get_pblh,
  qc/qi rates); new inputs qc/qi/rthraten and outputs qc/qi rates are (ncol, nz) REAL.
- GPU compiled path not measured (no GPU in this lane).

## Gates
`tests/v040/test_ysu_wrf_oracle.py` (fixture `fixtures/ysu_wrf_oracle.npz`, 48 rows = 24 columns × 2
modes, SHA in the JSON): literal BYTE (glibc exp/tanh), compiled envelope, census of the ported
branches, literal get_pblh loop reference (real columns never reach its exit safeguards), adapter
operands + RQC/RQIBLTEN application, operational call (RTHRATEN + namelist switch), namelist binding
and default treedef. `MUTANTS.json`: 14/14 genuine assertion kills (top-down off, cloud-top
entrainment off, RQCBLTEN, in-cloud Ri, thlix, RTHBLTEN/PI, CTOPO diagonal, get_pblh safeguard,
tridiag division form, xkzm(kpbl-1) rule, adapter qc, adapter rate application, operational
RTHRATEN, Python-double EP1), 0 setup errors. Legacy gates kept: `tests/test_v060_pbl_ysu.py`
(v060 oracle, topdown 0, FLAG_QC false, now passed explicitly) and the BEP urban PBL gate.

## Critic FIX round (rv-t-ysuqc, verdict on f68ca7296)
1. **Legacy radiation order (c02 LW1/SW1):** without the RRTMG driver carry the held-rate refresh ran in the
   radiation slot *after* the PBL call, so YSU's top-down entrainment read last step's RTHRATEN on refresh steps.
   WRF `module_first_rk_step_part1.F:264` calls radiation_driver before surface/PBL (`:1113`). The slot's
   refresh/hold is now one local helper; for bl=1 it runs once on the step-entry state before the PBL call and
   the radiation slot reuses the result (one SW + one LW solve; held steps unchanged).
2. **SSiB+YSU once-RK (c40):** `ssib_ysu_step` Euler-updated qc/qi (and u/v/θ/qv) in the pre-RK State. It now
   also returns its original dry/A-grid rates and the operational SSiB branch uses the generic branch's entry
   restore + raw-rate seam (shared `_raw_pbl_sources`, no second PBL solve).
- Tests: refresh step (call order, single solve, entry-state radiation input, YSU = carried rate), held step,
  SSiB six raw rates at the RK source builder + restored prognostics + retained PBLH. Mutants M15–M19 killed.
- `PROGRAM_IDENTITY_NONYSU.json` (`jaxpr_identity_probe.py`): location-stripped `_physics_step_forcing` jaxprs of
  MYNN/Shin-Hong/ACM2/BouLac with legacy RRTM-Dudhia / RRTMG-coupler / no radiation, refresh and held steps,
  RK-source on/off: 20/20 identical to f68ca7296. YSU changes only on refresh steps (by design).
- Same seam, not fixed here (other owners, E86): EEPS/KEPS (bl=16/17) and CAM-UW (bl=9) also pass
  `next_carry.rthraten` to their PBL before the legacy radiation refresh.

## Reproduce
`python3 build_oracle.py <lane>`; `JAX_PLATFORMS=cpu python3 compare_port.py <repo> <lane> <tag> [--eager] [--glibc]`;
`python3 ulp_envelope.py <lane> 96`; `python3 make_fixture.py <lane>`;
`JAX_PLATFORMS=cpu python3 mutants.py <repo> <out.json>` (source-mutating, sequential).
