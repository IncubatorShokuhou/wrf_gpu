# w41-aerosol AE01 — pristine WRF V4.7.1 aer_opt=1 oracles

Unchanged WRF V4.7.1 `libwrflib.a` (`<USER_HOME>/src/wrf_pristine/WRF`, gfortran serial). Same build recipe
as RE01/TH08.

## Programs

* **`ae01_units.f90`**
  * Dumps `swaerpr` (`rsrtaua/rsrpiza/rsrasya`, after `rrtmg_swinit`) and `pina`.
  * Runs `aerosol_in` on 51 lat × 53 lon points, including table nodes, ±89.x° and both
    periodic lon edges.
  * Runs `aer_time_int` for 520 julians on a unit-vector `aerodm`, so row k = fact1 at nm and
    fact2 at np.
  * Runs `aer_p_int` on 37 columns: 31 real WN3 0227 d02/d03 columns plus 6 synthetic ones.
    The synthetic set covers above-table, below-table and exact table-level hits, and one
    monotone column through all 12 levels.
* **`ae01_driver.F90`**
  * Calls RRTMG_SWRAD with the RE01 arm-A caller wiring: Thompson radii `has_reqc/i/s=1`,
    `cldovrlp=2`, the CAM o3 parent column, and RE01's operands.
  * Covers the 186 real WN3 0227 columns, 8 cloud categories, d02/d03, tau 12–36 h.
  * Runs 11 arms of (julian, aer_opt, aerod scale): own date, 0.3, 135, 196.6 and 352.2 d
    (each with aer_opt 0 and 1), plus own date with aer_opt=1 × 10.
  * `aerosol_in` is called once for all columns (WRF calls it once per domain). Then each arm
    calls `aer_time_int` + `aer_p_int` on the column's P/P8W, as radiation_driver does.
* **`read_tegen.f90`**: gfortran list-directed read of `aerosol.formatted`. The table parse
  in `scripts/extract_wrf_tegen_aerosol.py` equals it bitwise: 0 of 2,861,568 values differ.

## Regenerate

```
python3 ae01_build_inputs.py $AE01_DIR && bash build.sh && python3 ae01_parse.py $AE01_DIR tests/v025/w41_aerosol/fixtures
```

`ae01_summary.json` records the oracle self-checks and fixture/raw SHA-256s:

* arm 0 (aer_opt 0, own date) is bitwise RE01 arm A on 12 outputs plus coszen and o3;
* the aer_opt=1 clear-day surface SW reduction is 15.7 / 21.6 / 25.2 W/m² (min/median/max);
* the median column AOD is 0.099, of which 48 % is dust.

The `aerosol_night_gsw_zero` entry uses arm 0's day mask for all arms. It is not a physics
check: the other julians move the terminator.
