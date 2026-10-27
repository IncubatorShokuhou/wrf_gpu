# Radiation aerosol (`aer_opt`)

| `aer_opt` | WRF 4.7.1 meaning | Port status |
| --- | --- | --- |
| 0 | no aerosol in RRTMG (Registry default) | Released default program, unchanged |
| 1 | Tegen/ECMWF six-type monthly climatology in RRTMG SW | Experimental. CPU-oracle-qualified; GPU 24 h forecast matches original CPU-WRF (Tenerife three nests, 27 February 2026, frozen D6 PASS, every field and frame) |
| 2 | `aer_type` + AOD550 inputs | Refused (not ported) |
| 3 | Thompson mp28 water/ice-friendly aerosol optics in RRTMG SW | Experimental. CPU-oracle-qualified; GPU 24 h forecast matches original CPU-WRF (Swiss mp28, frozen D6 PASS, every field and frame) |

`aer_opt` is a single-entry `&physics` key. Both implemented options need `ra_sw_physics=4`
on every domain: WRF reads the aerosol only in RRTMG SW (and FARMS), so any other SW scheme
would run without aerosol. `aer_opt=3` also needs `mp_physics=28`, the only scheme here with
QNWFA/QNIFA. WRF silently skips the aerosol without them. The port's mp28 is single-domain
with `wif_input_opt=1`. The CLI honour check refuses every other combination. Longwave
carries no aerosol, as in WRF without Chem.

## What runs (same chain as WRF)

* **`aerosol_in`** runs on domain 1 only (module_physics_init.F:2214). It reads the
  `aerosol.formatted` table (46 lat × 72 lon × 12 months × 12 levels × 6 types) and does a
  bilinear lat/lon interpolation (`interp_vec`, periodic in both axes). The port does this
  host-side once at domain load on the wrfinput XLAT/XLONG. The result rides as a traced
  child of the root `OperationalNamelist` (`tegen_aerosol`), so it is never a compile constant.
  The table ships as `data/fixtures/wrf-tegen-aerosol-v1.npz` (6.0 MB, SHA-pinned keyed
  asset). It is a bitwise parse of the WRF text file, checked against gfortran list-directed
  reads: 0 of 2,861,568 values differ.
* **At each radiation call**:
  * `aer_time_int` interpolates monthly from mid-month dates, with intJULIAN = JULIAN+1 and a
    Dec–Jan wrap.
  * `aer_p_int` does the vertical interpolation on `P = p_hyd`, `P8W = p_hyd_w`. It covers
    the extrapolation branches and multiplies by the layer dp.
* **RRTMG_SWRAD**: `ecaer` with an extra zero top layer feeds `iaer=6`, using the `swaerpr`
  band tables (`rsrtaua/rsrpiza/rsrasya`). The `spcvmc_sw` clear-sky optics are then:
  * `ztauc = ztaur + ztaug + ptaua`
  * `zomcc = ztaur + ptaua*pomga`
  * `zgcc = pasya*pomga*ptaua/zomcc`
  * aerosol delta scaling on top of these.

  The emitted direct beam is WRF's `ztdbt_nodel`: the unscaled clear plus unscaled cloud
  optical depth. All of this is literal REAL arithmetic.
* **Nests** never call `aerosol_in`. `aerod` is Registry `df=(p2c)`, so after every parent
  step the force-down interpolates the parent's held `aerod (6, z, y, x)` with SINT, per
  species and level, without re-interpolating to the nest pressure (nesting/nest_aerosol.py,
  same mechanism as `o3rad`). The root holds the field it radiated with. The leaf exists only
  on nested `aer_opt=1` runs and is restart state (pickle and WRF-NetCDF carry).

## aer_opt=3 (Thompson aerosol-aware, mp28)

At every radiation call:

* **`gt_aod`** (module_radiation_driver.F:5507) turns QNWFA/QNIFA into per-layer 550 nm AOD:
  * inputs: `P = p_hyd`, `T = t_phy`, `QV`, `dz8w`;
  * RH from the Thompson `RSLF` lookup, temperature-binned and RH-interpolated extinction table;
  * a per-layer floor of `1e-3`.
* **`TAOD5502D`** is the column sum. It is held between radiation calls (history `h` field,
  restart state).
* **`calc_aerosol_rrtmg_sw`** (module_ra_aerosol.F:397) runs with the radiation driver's
  PARAMETERs: rural type, 3-D AOD, Angström/SSA/asymmetry from RH. It maps the AOD to 14 band
  optical depths, single-scattering albedos and asymmetries. This uses Bolton RH and WRF's
  windowed Lagrange interpolation over the eight RH nodes of the SF79 tables.
* **RRTMG_SWRAD** receives them as `iaer=10` operands, with the extra top layer at
  tau 0 / ssa 1 / asy 0. It then uses the same `spcvmc_sw` clear-sky optics and `ztdbt_nodel`
  direct beam as `aer_opt=1`.

The tables are generated from the pristine DATA statements
(scripts/extract_wrf_aer_opt3_tables.py → physics/wrf_aer_opt3_tables.py; a test re-parses
the pristine source).

## Performance and memory

* `aer_opt=0` traces exactly the released program.
* `aer_opt=1` and `aer_opt=3` select a separate variant of the fused SW band kernel
  (kernels/rad_sw_band_sums_aer.py) with three extra per-layer, per-band operands. It is a new
  file, so the released kernel's TTIR is unchanged.
* `aer_opt=3` adds per-call column work (gt_aod + 14-band Lagrange tables) and one REAL 2-D
  carry leaf (TAOD5502D).
* `aer_opt=1`:
  * adds a root-domain static of 864 floats per column.
  * on nested runs, adds a 6·nz·ny·nx REAL carry leaf per domain.
  * the aerosol interpolation runs only inside radiation calls.

## Evidence

The pristine oracle (proofs/w41_aerosol/) is unchanged WRF V4.7.1 built from
`libwrflib.a`:

* **Units**: `aerosol_in` on 2,703 points (table nodes and both periodic edges);
  `aer_time_int` on 520 julians; `aer_p_int` on 37 columns (31 real Tenerife columns of 27 February 2026, plus
  columns above and below the table and exact table-level hits). All are bitwise equal in the
  port.
* **RRTMG_SWRAD** (caller wiring with Thompson radii and `cldovrlp=2`) on 186 real Tenerife
  d02/d03 columns of 27 February 2026:
  * 8 cloud categories, day and night;
  * 5 dates across the year, aer_opt 0/1;
  * an aerod × 10 stress arm.
* **Result**: the release path stays inside the frozen tier-1 RRTMG bounds on every
  aer_opt=1 arm, for surface/TOA/direct, clear-sky fluxes, interface profiles and heating.
  The tests are tests/v025/w41_aerosol/ and include genuine mutants of each wiring step.
* **aer_opt=3 oracle** (the same 186 columns, 6 arms: own date, mid-May, mid-July; QNWFA/QNIFA ×0.1, ×1, ×10):
  * `gt_aod` is bitwise; the band optics agree to ≤ 3e-6 relative;
  * the release path stays inside the frozen tier-1 bounds for every output, with one exception, explained next.
* **Where WRF itself is not reproducible**: gt_aod's per-layer AOD floor of 1e-3 also applies to the thin
  top layers. In one column at high sun (coszen 0.97, mid-May) the RRTMG two-stream for such a layer sits on an
  ill-conditioned branch. A 32-member ensemble of the unchanged WRF chain, with T and QNWFA moved by one REAL
  ulp, moves WRF's own heating rate in that layer between −7.8e-4 and +4.8e-4 K/s. It also moves WRF's own
  clear-sky TOA upward flux between 124 and 143 W/m² (with aer_opt=0 the same cells move by less than 1e-8).
  This affects 5 of ~157,000 compared cells, all in that column; the port shows the same instability. In
  exactly those cells the port is checked against WRF's ensemble range (widened by the same bound) instead of
  the single unperturbed WRF value. Everywhere else the frozen bound applies unchanged, and the plain miss is
  kept as a pinned test result.
* **aer_opt=3 coupled forecast** (Swiss 42×42, mp28, 24 h, RTX 5090) against an original CPU-WRF run of the same
  namelist: the frozen D6 gate passes, with all 369 CPU history fields present in all 25 frames, including
  TAOD5502D. The worst-hour errors are T2 0.23 K, U10/V10 0.51/0.56 m/s and RAINNC 0.81 mm (the mp28 run without
  aerosol shows the same RAINNC, 0.80 mm). TAOD5502D matches CPU-WRF to 1e-4, and clear-sky surface SW at noon to
  0.05 W/m².
  * **Compiled-kernel resource check on that run: FAIL on the conservative kInput rule.** Three optimized-HLO kInput transpose
    fusions exceed 500 instructions (658, 879 and 1,370). They come from the Thompson mp28 source families
    (DOUBLE-island saturation/RSLF/RSIF code), which are pre-existing. These are not byte-identical kernels: the
    v0.4.0 mp28 run without aerosol had two such flags (741 and 1,439). Maximum STACK is 16 B, and no non-leaf
    callee exceeds 1,000 SASS. The aerosol code adds no flagged fusion. Its optics fusion (720 instructions in
    the first run) is kept out of the transpose by value-neutral barriers.
* **aer_opt=1 coupled forecast** (Tenerife, 27 February 2026, three nests, 24 h, RTX 5090) against an original CPU-WRF run of the same
  namelist: the frozen D6 gate passes on d01, d02 and d03, with all CPU history fields present in all 25 frames.
  The worst-hour T2 error is 0.04/0.03/0.08 K. Clear-sky surface SW (which carries the aerosol) agrees to
  0.15/0.16/0.35 W/m² RMSE; for scale, the aerosol itself lowers it by 20–23 W/m² against the original aer_opt=0
  run. This also shows that the nests receive the parent's aerosol as WRF does.
  * The compiled-kernel resource check for this configuration was measured on a 3 h re-compile of the same tree and environment, not on the
    24 h run's own binary, whose cache was lost (cold compiles are not byte-deterministic). Maximum STACK
    is 32 B, with no non-leaf or kInput flags. The scanner verdict is INCOMPLETE because slim AOT metadata
    hides kInput sizes, so this result is a re-compile diagnostic only.
