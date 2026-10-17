# Shin-Hong PBL (`bl_pbl_physics=11`) vs pristine WRF 4.7.1 — lane t-shinhong

## Oracle
- Source: unmodified `phys/module_bl_shinhong.F` (SHA256 `99f44dbe…`), lane-dir copy (E104),
  built with `shinhong_oracle_wrapper.f90`, a thin public `shinhong(...)` call mirroring
  `module_pbl_driver.F:1301-1329` (QC3D/QI3D with RQCBLTEN/RQIBLTEN present, CORF=f, CTOPO,
  module_model_constants values). Builds (`build_oracle.py`): `zero` -O0 no-FMA init zero
  (reference), `inf` init inf, `fma` -O2 -mfma (116 FMA instructions).
- Columns: the predecessor's 4 synthetic 16-level regimes (s2-cammg CM78-CM89) and 32 real
  44-level columns of the original CPU-WRF c07 run (Swiss, mp9/bl11, 00/06/12/15 UTC: max/min HFX,
  cloud water, cloud ice, max PBLH, 3 random); modes 0, 1 cold (TKE .005), 1 warm (TKE = the
  oracle's own cold diagnostic). 108 cases, 0 nonfinite.
- Uninitialised reads: `zero` vs `inf` builds BYTE-identical on all 108 cases → no
  uninitialised value reaches any output (nothing for UPSTREAM_WRF_BUGS).

## Deviations found and fixed (`src/gpuwrf/physics/bl_shinhong.py`, `coupling/scan_adapters.py`)
| # | deviation | WRF | class |
|---|---|---|---|
| 1 | mixlen/prodq2/vdifq used interface k+2 for k+1 | :1883, :2040, :2163 | TKE operator (s2-cammg 036ff1891) |
| 2 | tflux_e accumulated theta rate, not TTEND = rate·PI | :1202-1212 | TKE operator (s2-cammg 4f45910a6) |
| 3 | adapter passed no QC3D/QI3D, no CORF (kernel default 1e-4) | pbl_driver :1301-1329 | caller (E11): mixlen cwm, in-cloud Ri, stable mixing length |
| 4 | RQCBLTEN/RQIBLTEN never produced (ndiff=3 shared moisture matrix) | :1238-1368, :253-256 | forecast-driving |
| 5 | local pthl/pq/pu scale factors applied for k<kpbl without pblflg | :1179, :1285, :1429 | forecast-driving (stable kpbl>1: ~1 % momentum) |
| 6 | literal REAL4 order: tridin reciprocal pivot, ad/f assembly order, -dtodsd·(tem1·rdz), RTHBLTEN=TTEND/PI, sequential zq/flux/szq/sq sums, eloq2x reciprocal, s2l association, REAL exponents (powf, not integer powers), gfortran-folded REAL4 constants (29 bit-exact, `real4_constants.f90`; EP1, g·g, cp/g, 1-a7 were 1 ULP off), nlfrac·sflux0 and a7·num/den association | throughout | ULP-level, cancellation-amplified up to 3 % of small tendencies |

## Result [M CPU] (`PROGRESSION.json`)
Worst |port−WRF| / column max, mode 1 warm, 36 columns:

| stage | TKE | EL | u | θ | qv |
|---|---|---|---|---|---|
| parked fixes, old caller args | 0.43 | 0.92 | 0.94 | 1.04 | 1.67 |
| + WRF caller args (qc/qi/corf) | 1.5e-2 | 0.92 | 2.6e-2 | 3.2e-2 | 1.7e-2 |
| + pblflg gate, RQC/RQI, cwm | 1.0e-3 | 1.8e-6 | 8.3e-4 | 3.2e-2 | 1.7e-2 |
| final, op-by-op REAL4 | **0 (BYTE)** | **0** | **0** | **0** | **0** |
| final, XLA:CPU jit | 5.4e-7 | 2.7e-6 | 4.0e-5 | 0 | 1.1e-4 |

- Literal path (op-by-op, `jax.disable_jit`): BYTE on TKE, EL_PBL, EXCH_H, u/v/θ/qv rates,
  PBLH, KPBL for all 108 cases; qc/qi rates BYTE except cells fed by subnormal inputs
  (XLA:CPU flush-to-zero, FINDINGS E117; all |diff| < 3e-38).
- Compiled path: WRF's own ±1-ULP input envelope (`ulp_envelope.py`, 67,216 original calls,
  plus the FMA build) widened by 2 output ULPs contains every normal-range cell except one:
  synthetic regime 1, v tendency level 9 (cold + warm rows), 3.3e-9 m s⁻² outside = one ULP of
  the solved wind (1.72 m/s) / 2dt. Disclosed, not relaxed; algsimp off or op-by-op → 0 outside.
  GPU compiled path not measured (no GPU in this lane).

## Gates
`tests/v040/test_shinhong_wrf_oracle.py` (fixture `fixtures/shinhong_wrf_oracle.npz`, 36 rows =
12 columns × 3 modes, SHA in the JSON): literal BYTE, compiled envelope (one disclosed cell),
adapter RQC/RQIBLTEN application; `tests/v040/test_shinhong_tke_caller.py` asserts qc/qi/corf reach
the kernel. Compiled rule (after review rv-t-shinhong): every candidate/reference/bound finite; a cell
is compared when reference OR candidate is normal (both-subnormal pairs within FLT_MIN); Registry-default
mode-0 rows check held TKE / zero EL bytes and their driving rates against the identical-input mode-1-cold
sibling's bounds; sentinel regressions (zeroed driving outputs, normal candidate vs zero truth, NaN) must
raise. The one disclosed cell then appears in rows 3/4/5 (same inputs). `MUTANTS.json`: 11/11 genuine
assertion kills (pblflg gate, cwm, PI, k+2 index, RQCBLTEN, tridiag division form, Python-double EP1,
caller corf, caller qc; M10 erased driving outputs and M11 pblflg vs the compiled gate alone), 0 setup errors.

## Reproduce
`python build_oracle.py <lane>`; `JAX_PLATFORMS=cpu python compare_port.py <repo> <lane> <tag> --wrf-args [--eager]`;
`python ulp_envelope.py <lane>`; `python check_envelope.py <lane> <tag>`; `python make_fixture.py <lane>`;
`observe_tke.py` / `observe_momentum.py` are WRITE-only first-divergence observers (public bytes asserted unchanged).
