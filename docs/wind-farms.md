# Fitch wind farms

Select `windfarm_opt=1` in `&physics` with MYNN `bl_pbl_physics=5`,
`sf_sfclay_physics=5` and `bldt=0`. The native and CPU-history pipelines load
turbines separately for each domain. `windfarm_opt=0` keeps the release program
unchanged. MAV (`windfarm_opt=2`) remains refused.

Specify per-domain selections, for example `windfarm_opt=1,0,1`. As in WRF,
a single `windfarm_opt=1` selects d01 only; omitted domain entries stay zero.
`windfarm_ij` and `windfarm_tke_factor` are scalars shared by all domains.

Place `windturbines.txt` in the case directory: one `latitude longitude type`
record per turbine. WRF Lambert, polar stereographic and Mercator projections
use WRF's nearest-grid NINT placement. With `windfarm_ij=1`, use
`windturbines-ij.txt` instead: `i j type`, **one-based mass-grid** indices.
Turbines outside the selected domain are ignored as in WRF; colocated turbines
are accumulated in file order. Supply each `wind-turbine-N.tbl`:

```text
6
80 80 .12 2000
2 .9 0
4 .85 60
8 .8 500
12 .7 1600
20 .4 2000
25 .1 2000
```

The first record is the number of curve rows (2–100). The second contains hub
height (m above ground), rotor diameter (m), standing thrust coefficient and
rated power (kW; WRF reads this but does not use it). Curve rows contain wind
speed (m/s), thrust coefficient and power (kW). Speeds must be positive and
strictly increasing; all inputs finite, coefficients and power nonnegative.
Values must remain finite and speeds strictly increasing after WRF REAL32
rounding. The last curve thrust coefficient is used at/above cut-out; below/equal cut-in
uses standing thrust. Power is zero outside cut-in/cut-out. Within the curve,
WRF interpolates **Cp**, including its fixed 1.23 kg/m³ density, and applies its
hub-power/layer-power energy normalization. This can differ from interpolating
the table's power directly.

The full rotor and hub interpolation bracket must fit the initial model column;
hub heights below the first mass level or above the last are refused because
pristine WRF's hub indices are undefined there. Degenerate, unordered and
nonfinite tables fail before stepping. Blank records are not supported.
`windfarm_tke_factor` defaults to WRF's 0.25 and scales the nonnegative
Ct−Cp source. QKE is twice TKE (m²/s²). Drag joins MYNN's mass-point DU/DV
sources at WRF RK cadence; QKE is updated after MYNN from the pre-PBL winds.

History `POWER` is instantaneous power (W) summed across turbines per grid cell,
not W/m² or accumulated energy. It starts at zero before the first PBL call.
QKE and held POWER survive both native and NetCDF restarts. Restart checks bind
the turbine input fingerprint, preventing resume with changed turbine tables.

`gpuwrf doctor CASE` reports turbine count/density and an advisory drag time
scale based on initial wind, minimum layer thickness, full rotor area, maximum
tabulated thrust and dt/dx². It warns above 0.5; this excludes energy
normalization and future flow and is **not** a stability bound. Reduce dt and
inspect farm density when the advice warns; no additional physical clamp is
applied by the solver.

CPU-oracle-qualified against pristine `phys/module_wind_fitch.F` computational
routines; GPU/coupled-forecast qualification pending. Fixtures and source seals:
`proofs/u2_windfarm/`. CPU coupled smoke is integration evidence only, not a
CPU-WRF coupled forecast fidelity claim.
