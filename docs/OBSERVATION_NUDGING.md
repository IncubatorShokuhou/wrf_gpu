# Station observation nudging

`obs_nudge_opt=1` selects WRF station nudging on the root domain. The current
qualified path uses MYNN PBL and surface layer (`bl_pbl_physics=5`,
`sf_sfclay_physics=5`), Lambert conformal, polar stereographic or Mercator maps.
Single-level surface/upper-air reports and multi-level soundings share the
original WRF error calculation and spreading rules. GPU and long coupled
forecast qualification remain pending.

Place `OBS_DOMAIN101` beside `wrfinput_d01` and the namelist. This is the
**processed WRF observation-nudging file**, produced by WRF OBSGRID/little_r
processing; a raw little_r file is not interchangeable. Reports must be ordered
by UTC observation time. The model start is the timestamp in `wrfinput_d01`.

A minimal selection is:

```fortran
&fdda
 obs_nudge_opt = 1,
 max_obs = 10000,
 fdda_end = 360.,
 obs_nudge_wind = 1, obs_nudge_temp = 1, obs_nudge_mois = 1,
 obs_coef_wind = 0.0003, obs_coef_temp = 0.0003, obs_coef_mois = 0.0003,
 obs_rinxy = 30., obs_twindo = 0.5,
 obs_ionf = 1,
 obs_ipf_in4dob = .false., obs_ipf_errob = .false., obs_ipf_nudob = .false.,
/
```

`fdda_end` is minutes, `obs_twindo` the half-window in hours, `obs_rinxy` the
horizontal radius in kilometres and coefficients inverse seconds. WRF compares
observation times with `XTIME-DT/60` when weighting. The inner half of the
half-window has full weight; the outer half ramps to zero. `max_obs` must hold
the rolling station levels, including WRF's one future-report lookahead.

The original `obs_sfcfact`, `obs_sfcfacr`, `obs_dpsmx`, surface horizontal/vertical
scheme switches, regime-specific `obs_nudgezfullr*`/`obs_nudgezrampr*`, minimum
and maximum influence heights, maximum sounding gap, per-variable PBL exclusion,
`obs_scl_neg_qv_innov` and dynamic-initialization ramp are bound. `obs_ionf` holds
saved innovations between refreshes, while the current mass, map factors and
ramp factor still apply each step. WRF removed pressure nudging: its pressure
switch/coefficient and `obs_rinsig` remain inactive, as in pristine WRF.

All reports are read, projected and QC-processed once during initialization.
The model keeps a rolling VAROBS/RKO frame and four SAVWT arrays on device.
Checkpoints preserve these fields and bind them to the input/option fingerprint;
missing carry or changed inputs are refused. There is no file read or host
callback within the timestep. Grid FDDA (`grid_fdda=1`) has a separate carry and
its sources precede the observation source, matching the WRF caller.

The daily pipeline keeps the full operational carry and global timestep across
hourly and auxiliary-history outputs. Its restart probe writes and restores that
same carry, including held observation innovations and the next timestep. Output
boundaries therefore preserve the observation window and `obs_ionf` cadence.

For each fixed-width report, the pristine format is:

1. One leading blank and 14-character `YYYYMMDDHHMMSS` UTC timestamp.
2. Latitude/longitude: `2x,f9.4,1x,f9.4`.
3. Station ID/name: `2x,a40,3x,a40`.
4. Platform/source/elevation/sounding/bogus/level count:
   `2x,a16,2x,a16,2x,f8.0,2x,l4,2x,l4,2x,i5`.
5. A surface row has nine value/QC pairs, each `f11.3,1x,f11.3,1x` after one
   leading blank: sea-level pressure, reference pressure, height, temperature,
   U, V, RH, station pressure and precipitation. Each sounding row has six
   pairs: pressure, height, temperature, U, V and RH.

Pressures in the file are Pa, temperatures K, heights m, winds m/s and RH percent.
The decoder refuses NaN/Inf and malformed rows; WRF numeric missing/QC sentinels
remain meaningful. The original QC, humidity/pressure conversion, wind rotation,
platform and duplicate-report rules run before ingestion. See the tracked
[Swiss OBS_DOMAIN example](../proofs/s1_obsnudge/full/savepoints/OBS_DOMAIN101)
and [proof packet](../proofs/s1_obsnudge/README.md).

Child-domain observation nudging, latitude/longitude maps, other PBL/SFCLAY
combinations, RUC surface-ratio capture and per-step print diagnostics are refused.
`obs_ipf_init` controls the initialization summary.
