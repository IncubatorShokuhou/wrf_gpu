# cu6 modified Tiedtke vs pristine WRF on REAL c06 columns (c06 RAINNC release blocker, 2026-10-16)

## Why
c06 (Swiss, mp6 / cu6 Tiedtke / bl8 / sf1 / Noah-MP / RRTMG) FINAL on 04cb09f816: D6 RAINNC rmse 1.06-1.11 > 1.0
at leads 16-24. The PRE56 bisect arm (a366702b5c) fails too -> batch56 is not the cause (manager 05:24Z).
GPU fields of four source versions [M]: RAINC (Tiedtke rain) fires in 49 cells in CPU-WRF by lead 16 but only
5-8 in the port, in every version -> a pre-existing cu6 defect.

## Reference (all CPU)
* Private pristine serial c06 `wrf.exe` (`pristine_run/CPU_executable.json`: original 52f914cb with only the two
  E104 dump-path literals patched), original c06 inputs (`pristine_run/inputs_outputs.sha256`), namelist diff
  `pristine_run/namelist_vs_c06orig.diff`: run to 15:04, restart at 15:00, auxhist2 every 18 s step 15:00-15:03
  (`pristine_run/iofields_cu6.txt`). Its hourly history 01-04 h is bitwise identical to v-prep's CPU reference.
* `phyprep_from_restart.py`: WRF phy_prep (REAL4 transcription of module_big_step_utilities_em.F) of the
  restart + WRF's own held RQVFTEN, this step's RQVBLTEN/QFX -> the UNMODIFIED `module_cu_tiedtke.F`
  (`build.sh`, WRF FCOPTIM flags) reproduces WRF's own RTH/RQV/RQC/RQI/RU/RVCUTEN and RAINCV of that step to
  <= 1.6e-4 rel (RAINCV 27/27 columns, sum equal) -> the inputs are WRF's.

## Findings
1. Every port caller input equals WRF phy_prep (`adapter_inputs_vs_phyprep.json`: T/PI/P_HYD/P_HYD_W/DZ8W/W/U/V
   exact or 3e-8, RHO 1.1e-5 rel). The defects are in the kernel.
2. **ROOT 1** (`cumulus_tiedtke_jax.py` + numpy `cumulus_tiedtke.py`): cuasc_new section 3.1
   (module_cu_tiedtke.F:1633-1649) sets `ldcum=.false.` for EVERY column before the ascent (convection is
   re-confirmed by finding a cloud top) and starts `zbuoy` at the cloud-base buoyancy for ktype 1. Without the
   reset CUBASMC (`.not.ldcum` trigger) never starts mid-level convection where CUBASE found a base: WRF ktype 3
   in 1349 of 1764 columns at 15:00, port 0. (The zbuoy start changes results by <= 1e-4 rel on the real deep
   columns -- pristine literal, below this oracle's REAL4-vs-f64 resolution, not separately mutant-killable.)
3. **ROOT 2**: `if(icum.eq.0) return` (:959) is decided per TILE ROW: CU_TIEDTKE calls TIECNV once per j with
   i = its..ite, and icum counts the row's convective columns after the first ascent. A column whose own first
   ascent finds nothing still gets the second ascent (where mid-level convection can start) when any column of
   its row convects. Pristine single-column calls differ from row calls in 85 strict icum = 0 columns; with
   the return commented out (diagnostic copy only) single-column == row call for KTYPE in all 1764 columns.
   The port now masks by any() over each x-row = the serial original CPU-WRF tile
   (`scan_adapters.tiedtke_tile_rows`). **Disclosure:** this reproduces WRF's per-tile decision for the serial
   reference; CPU-WRF with an MPI/OpenMP patch split of a row can decide differently itself (WN3 references
   run 4 ranks).
4. **ROOT 3** (JAX kernel only; the numpy reference was right): the ihmin search tested the INITIAL ihmin
   (`ihmin0 == kcbot`) instead of the carried one (:906), so it never stopped at the first hit -> wrong
   organized-detrainment limit in deep columns (RTH 39-74 % off on the 4 real deep columns).

## Results [M CPU]
| port vs WRF, c06 15:00, full domain through `tiedtke_adapter` | main 04cb/f0df (before) | fixed |
|---|---|---|
| raining columns (WRF 27) | 4 | 27 (all the same columns) |
| RAINCV sum (WRF .02967 mm) | .00327 | .02967 (rel rms 1e-5) |
| RTH/RQV/RQC/RQI/RU/RVCUTEN interior rel rms | .46/.45/.56/.62/.56/.60 | 9.7e-4/4.2e-4/3.7e-4/3.5e-4/1.0e-3/8.3e-4 |
| same, all columns | .22/.18/.26/.41/.19/.12 | 4.2e-3/1.7e-3/3.9e-3/3.2e-3/2.2e-3/8.2e-4 |

`adapter_oracle_main_before.json`, `adapter_oracle_fixed.json` (`adapter_oracle.py`). Input-sensitivity controls
(not mutants): zero RQVFTEN -> rel .84, zero RQVBLTEN -> rel .63.

## Test + mutants
`tests/test_t_cfix_cu_tiedtke_c06_rows.py` on `c06_tiedtke_rows.npz` (`build_fixture.py`, 783 KB): 9 real rows
(3, 21, 29, 36-41: icum = 0 columns, mid-level and all 4 deep columns, 18 raining) + 1 ASSEMBLED row of 42 real
icum = 0 columns (WRF returns early there although each column convects in its own row). Gates pre-registered
before the port ran on the fixture: raining set identical; RAINCV sum within 1 %, rel rms <= .10; every R*CUTEN
rel rms <= .02; assembled row exactly zero. First run with ROOT 1+2 only: RTHCUTEN .0201 > .02 MISS (not
relaxed) -> led to ROOT 3; final: max rel 6.6e-4. Numpy reference vs pristine single-column calls: same gates.
`mutants.py` -> `mutants_result.json`: 7/7 genuine AssertionError kills (ROOT1 JAX, ROOT1 numpy, ROOT2 per-column
icum, ROOT2 wrong axis, ROOT2 mask dropped, ROOT3, adapter wiring).

## Identity + consumers
Release identity (official probe, `identity/`): Swiss 7379c7d7 / PROD 6487f5b0 EXACT (defaults untouched); c06 release
program e33ca43e (cu6 path changes, as intended). Consumer tests `consumer_tests.txt`: 14 files PASS; four other-lane
tests stub `tiedtke_column_jax` and got the ICUM1 key (test-only ABI, convective row = previous behaviour).

## Not covered / open
* GPU c06 (and cu6/cu16 composites) on the fixed tree -- the decisive RAINNC arm (manager 05:24Z).
* NTiedtke (cu16, physics_mmm/cu_ntiedtke.F90) does not share these constructs [I, code reading]: no all-column
  ldcum reset (only ktype 3 -> .false., present in the port :771), no row-level icum return, no ihmin search;
  its row `is` gates only skip work when no column is active. No real-data cu16 oracle run.
* Cost: no new kernel work (the kernel already ran both ascents for every column and masked at the end); one
  extra (ny,nx) any() reduction per call [I]; GPU timing unmeasured.
