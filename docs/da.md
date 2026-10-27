# Ensemble data assimilation (`gpuwrf da`, experimental)

`gpuwrf da` combines an ensemble of short forecasts with observations and writes
an analysis you can start the next forecast from. It uses a Local Ensemble
Transform Kalman Filter (LETKF; Hunt, Kostelich and Szunyogh 2007). The analysis
runs on the CPU between forecasts; the forecasts themselves are ordinary
`gpuwrf run` (or CPU-WRF) runs.

**Status: experimental, opt-in, outermost domain (d01) only.** Nothing changes
for `gpuwrf run` unless you start it from a file this command wrote. The
algebra, the moisture rules, the hydrostatic rebalance and the file writer are
tested (see [Evidence](#evidence)); a better forecast is **not** established.
Three development cases showed lower 12-hour station errors for some variables
and higher for others, and the forecast model lost an analysed trade-wind
inversion within a few hours. Verify against independent observations for your
own region before you rely on it.

| Command | Class and delivered scope |
| --- | --- |
| `da check/analyse` | Experimental, CPU/backend-dark offline LETKF analysis for d01 (opt-in). Analytic Kalman, Lorenz-96 and real-file gates; the written wrfinput passes the port's case preflight. Forecast skill not established; nests, cloud observations and MU analysis are not supported. [Guide](da.md). |
| `da cycle` / `da members` | Experimental (opt-in, d01): an ensemble of member forecasts run in ONE process on one shared step executable, with a zero-update control arm and LETKF arms between windows; file seam (next window from the analysis wrfinput) or restart seam (analysis increment injected into each member's restart). In-process members are byte-identical to separate `gpuwrf run` processes (CPU and GPU). [Guide](da-cycle.md). |

## Quick start

1. Run an ensemble of forecasts that are valid at the analysis time `t_a`
   (one `wrfout_d01_<t_a>` per member), plus a control (deterministic)
   forecast. All must share the grid.
2. Put your observations in one of the [supported formats](#observations).
3. Write a configuration:

```json
{
  "schema": "gpuwrf.da.v1",
  "t_a": "2023-01-15T00:00:00Z",
  "template_wrfinput": "case/wrfinput_d01",
  "control_file": "control/wrfout_d01_2023-01-15_00:00:00",
  "ensemble": {"kind": "members", "member_glob": "members/m*/wrfout_d01_2023-01-15_00:00:00"},
  "obs": {"sources": [
    {"format": "csv_surface", "path": "obs/stations.csv"},
    {"format": "igra2", "path": "obs/SZM00006610-data.txt", "station": "SZM00006610"}
  ]},
  "experiments": [
    {"name": "all", "assimilate": ["surface", "sonde"], "write_mean": true}
  ]
}
```

4. Check it, then analyse:

```bash
gpuwrf da check   --config da.json          # validates; lists every input; opens nothing
gpuwrf da analyse --config da.json --out analysis_00z
```

`analysis_00z/wrfinput_d01` is the analysis-mean initial state. Copy the
case's `namelist.input`, `wrfbdy_d01` (and `wrflowinp_d01` if used) next to it
and run the next forecast as usual. The analysis keeps the specified and
relaxation boundary zone exactly as the control, so the existing boundary file
stays consistent.

Exit codes: `0` done (every requested file written), `1` done but a requested
output was refused (the receipt says why), `2` invalid configuration or a
missing/forbidden input (nothing was read).

## What the analysis does

1. **Background ensemble** (`ensemble`). `kind: "members"` reads one wrfout
   frame per member at exactly `t_a` (recommended: a forecast ensemble).
   `kind: "same_phase"` builds a static ensemble from archived frames of
   earlier runs at the same UTC hour (causal: only frames valid at or before
   `t_a`), centred on the control; it carries no flow-dependent error of the day.
2. **Observation operators.** Sondes/profiles: horizontal bilinear on each
   model level, linear in ln p; temperature from WRF's dry theta; winds rotated
   to earth-relative. Surface T2/Q2/U10/V10: land-aware bilinear, T2 lapse-rate
   corrected to the station height; stations more than 150 m from the model
   terrain are rejected. Observations outside the domain are rejected and
   counted.
3. **Quality control.** Non-finite values, upstream flags and a gross check
   (|O−B| > 4·sqrt(σo² + spread²)) are counted in the receipt.
4. **LETKF.** Symmetric square-root transform per local analysis, Gaspari–Cohn
   R-localisation in the horizontal (default L = 100 km) and in ln p (0.1 above
   700 hPa, 0.04 below, 0.04 for surface observations). L is a Gaussian-equivalent
   length; the taper is zero beyond 3.65 L. Weights are computed every second
   column (`stride_h`) and interpolated.
5. **Posterior spread.** Relaxation to prior spread (RTPS, default α = 0.8).
6. **Boundary taper.** Increments are zero in the specified and relaxation
   zone and ramp to full strength over `taper_ramp` rows. The zone width is the
   case's `spec_bdy_width`, read from the namelist next to the frames; the same
   width is used by the analysis, the mean writer (which refuses an output that
   changes a zone cell) and the member writer. An explicit
   `params.spec_bdy_width` must equal the case value; a namelist that leaves
   the key unset runs WRF's default 5; without a namelist the parameter is
   required. A contradiction or an unknown width is refused before the analysis
   runs, and the receipt records the width and its source.
7. **Moisture policy and gates** (next section).
8. **Hydrostatic rebalance.** Dry column mass MU is held fixed. Pressure and
   geopotential are recomputed incrementally with real.exe's moisture-loading
   recursion and the model's hypsometric form, so a zero increment reproduces
   the background bit for bit.
9. **Analysis mean** for the next forecast = control + mean member increment,
   rebalanced once against the control (averaging member pressures would break
   hydrostatic balance).

Analysed fields: U, V, T, QVAPOR (configurable subset). Other prognostic and
land fields are copied from the control frame. `land: "cycled"` (default) marks
the file so the port restores the control's evolved Noah-MP snow/canopy state;
`land: "cold"` writes no marker (use it when the frames are themselves
`wrfinput` files).

## Moisture policy

A linear ensemble update can make water vapour negative where the prior spread
is large compared with the value. The policy is declared, counted and gated
(`moisture` section):

| key | default | meaning |
| --- | --- | --- |
| `negative` | `"clip"` | `clip`: negative QVAPOR → 0; the vapour this adds is integrated with the dry-air mass of each layer. `refuse`: any negative cell makes the analysis inadmissible. `log`: analyse ln(qv + floor) instead (positive by construction). |
| `max_added_column_kg_m2` | 0.1 | clip budget per column (kg m⁻², i.e. mm of precipitable water) |
| `max_added_fraction` | 0.001 | clip budget per member, as a fraction of its domain vapour mass |
| `log_floor_kgkg` | 1e-7 | floor of the `log` transform |
| `supersaturation` | `"keep"` | `keep` only: supersaturated analysed cells are counted in the receipt, never changed. `cap` and `to_cloud` are refused in this build: a saturation adjustment would act on the protected boundary zone, the fixed-QV rebalance loads only the QVAPOR increment (a QV→QCLOUD transfer would break hydrostatic balance) and the member files do not carry QCLOUD |

Only cells the analysis changed are treated; a negative value the background
already had is kept and counted separately. An analysis is refused (with the
reason in the receipt) when the added water exceeds either budget, when a
value is non-finite, when the rebalance fails, or when the stored float32
field is negative. The budgets are engineering limits, not values fitted to a
verification case; tighten or relax them knowingly. The earlier rule (refuse
on any negative cell) is `"negative": "refuse"`; with it, one negative cell can
flip the decision between two almost identical analyses.

## Observations

Times must carry a UTC offset (`2023-01-15T00:00:00Z`). For each station the
record nearest to `t_a` within `window_minutes` is used. Humidity is converted
to the water-vapour mixing ratio with WRF's saturation formula; when no
station pressure is given, the standard atmosphere at the station height is
used (counted). Winds are converted from meteorological direction (blowing
from) to earth-relative components. Nothing is dropped silently: skips are
counted in `summary.json`.

| `format` | content | keys |
| --- | --- | --- |
| `igra2` | NOAA IGRA2 v2 station file, any station; mandatory and significant levels; missing heights filled hypsometrically | `station` (11-character IGRA ID), optional `valid_time`, `drift`, `include_surface`, `ascent_mps` |
| `isd_lite` | NOAA ISD-Lite hourly file (T, dew point, wind; sea-level pressure is not used) | `station` {`id`, `lat`, `lon`, `elevation_m`} or `isd_history` + `station_id` ("USAF-WBAN"), `window_minutes` (30) |
| `csv_surface` | one row per station and time: `station_id, lat, lon, elevation_m, time` and any of `t2_k`/`t2_c`, `q2_kgkg`/`td2_k`/`td2_c`/`rh2_pct`, `u10_ms`+`v10_ms` or `wspd10_ms`+`wdir10_deg`, `psfc_pa`/`psfc_hpa` | `window_minutes` (30), `include_pressure` (false) |
| `csv_profile` | upper-air levels: `station_id, lat, lon, time` + `p_pa`/`p_hpa` and any of `height_m`, `t_k`/`t_c`, `qv_kgkg`/`td_k`/`td_c`/`rh_pct`, `u_ms`+`v_ms` or `wspd_ms`+`wdir_deg` | `window_minutes` (90) |
| `obsbatch_json` | this package's own observation table (`ObsBatch.to_dict`) | |
| `obsdb_parquet`, `msg_clm_npz` | ALISIOS ObsDB extracts and the MSG cloud mask (cloud can only be withheld, never assimilated) | site-specific |

Unknown CSV columns are refused. `obs.sonde_superob` (default true) averages
sonde levels to at most one value per model layer. Observation errors default
to T 1 K / wind 2 m/s (sonde) and T2 1.75 K / wind 2.5 m/s (surface), humidity
10 % / 15 % relative humidity converted to mixing ratio; override them with
`obs.error_overrides`, e.g. `{"SURFACE": {"T2": {"sigma": 1.2}}}`.

Observation rows also carry a role. A row whose role is `withhold` (e.g. a
held-out station in an `obsbatch_json` table) is never assimilated and never
used to calibrate the ensemble scale, even when its kind is listed under
`assimilate`; it is verified like a withheld kind (O−B/O−A in the receipt).
`assimilate: ["cloud"]` is refused (there is no cloud analysis); list cloud
under `withhold` for its diagnostics.

## Configuration reference

The configuration is strict: an unknown key, a wrong type, an out-of-range
number or a time without offset is refused before any file is read.
Relative paths are relative to the configuration file.

| key | required | meaning |
| --- | --- | --- |
| `schema` | yes | `"gpuwrf.da.v1"` |
| `t_a` | yes | analysis time, ISO 8601 with offset |
| `domain` | no | `"d01"` (only value supported) |
| `template_wrfinput` | when writing | the case's real.exe `wrfinput_d01` (static fields) |
| `control_file` | `same_phase`, or `write_mean` with `members` | control forecast frame at `t_a` |
| `ensemble` | yes | `kind` (`members` or `same_phase`), `member_glob` or `member_files`, `runs_glob`, `max_age_days`, `max_members`, `names`, `scale` (fixed number or innovation-calibrated), `inflation_rho` |
| `obs` | yes | `sources` (list), `sonde_superob`, `error_overrides` |
| `params` | no | `loc_h_km` (100), `loc_v_lnp` (0.1), `loc_v_lnp_low` (0.04), `p_low_pa` (70000), `loc_v_sfc_lnp` (0.04), `stride_h` (2), `stride_v` (1), `pmax` (400 obs per local analysis), `inflation` (1, prior), `rtps_alpha` (0.8), `gross_k` (4), `spec_bdy_width` (the case's; must match it), `taper_ramp` (5), `variables`, `diagnostic_variables`, `rebalance` (true), `workers` (4) |
| `moisture` | no | see [Moisture policy](#moisture-policy) |
| `localisation` | no | `variables`: per analysed variable, factors in [0, 1] for observation kinds (`"*"`, `"sonde"`, `"surface:T2"`); the most specific key wins |
| `governance` | no | `plugin` (`none`, `alisios` or `package.module:factory`), `options`, `hash_inputs` (true) |
| `experiments` | yes | list of `name`, `assimilate` (kinds), `withhold` (kinds verified but not assimilated), `params` (overrides), `write_mean`, `land`, `write_members` {`source_case_dir`, `hours`}, `profile_site` {`lat`, `lon`} |
| `inflation`, `cycle` | no | provided by the ensemble and cycle components; refused when the build lacks them |

## Outputs and provenance

`--out` must be a new directory. It receives `config.normalised.json` (the
validated configuration and its SHA-256), `read_plan.json` (every input with
its role), `receipt_<experiment>.json` (QC counts, O−B/O−A statistics for
assimilated and withheld observations, localisation, moisture counters and
budget, refusal reasons, timings), `summary.json`, `governance/` and the
written files. The analysis `wrfinput_d01` carries a `GPUWRF_LETKF_ANALYSIS`
attribute with the input SHA-256s, the configuration hash and the list of
analysed, recomputed, copied and template fields. Inputs are never modified.

**Governance.** Before anything is opened, every input file is listed; the
default plug-in (`none`) records size and SHA-256 of each and every file
actually opened, and flags opens that were not planned. Sites with data rules
can supply their own plug-in (`package.module:factory` returning a
`gpuwrf.assimilation.letkf.governance.Policy`). The `alisios` plug-in
implements the ALISIOS data rule (guard admission per valid day, embargoed days
dropped before any open, raw observation files refused in favour of guarded
extracts). The valid days of every ALISIOS file are taken from its run's
`namelist.input` (read through a guarded metadata session first): a `wrfinput`
is valid on its domain's start day, which can be the day after the issue day; a
`wrfout` covers its first frame up to `frames_per_outfile` history intervals
(WRF defaults when unset); other case files cover the whole run. Every covered
day is guard-checked before anything is opened: an archive frame touching a
non-ALLOW day is dropped, any other input is refused, and unknown coverage
(no namelist, no start or run length) is refused. For a `same_phase` ensemble the `max_members` limit is applied after
the plug-in has filtered the candidates, so the admitted read set is exactly the
set of frames that is opened. It prefers ALISIOS's own reader facade
(`alisios.guarded_read`, imported from a copy sealed out of the ALISIOS git
history at a pinned commit): every ALISIOS input is scanned, admitted and read
once into a private stage inside one facade session (logged in its
`READ_SPY.jsonl`), the analysis then reads only the stage while every ALISIOS
open is refused, and the stage is deleted at the end. Where the facade is not
available, or a planned ALISIOS file is used outside its planned path (source
case files for member cycles, 4D frames, inflation state), the plug-in falls
back to its own guard session and records why; `governance.options.facade`
(`auto`, `require`, `off`) controls the choice.

## Evidence

All CPU, all repeatable with `pytest tests/da tests/a1_da`:

- **Analytic Kalman limits:** the transform equals the state-space Kalman
  update with the ensemble covariance (also with prior inflation), the scalar
  gain for one observation, R-localisation as scaled observation errors.
- **Lorenz-96 OSSE** (40 variables, 20 members, every variable observed):
  analysis error 0.21–0.22 σo over three seeds against a free-run error near
  3.6, spread/error 1.2. The test setting (RTPS 0.3) was chosen from a sweep; the
  probe's pre-registered RTPS 0.6 run missed its 0.30 bar (0.302).
- **Hooks and gates:** default hooks reproduce the plain analysis bit for bit;
  variable localisation, per-point inflation and posterior maintenance change
  exactly what they claim; moisture budgets checked against hand-computed dry-air
  masses; one negative cell no longer flips admissibility.
- **Rebalance:** fixed-MU increments against a verbatim real.exe REAL*8 column
  oracle (relative pressure error 2e-11 in the development probe receipt); zero increment
  reproduces the background bit for bit.
- **Switzerland d01 end to end:** synthetic members and observations on the
  shipped case: O−A < O−B, withheld observations stay withheld, the boundary zone
  is unchanged, the written analysis passes the port's own case preflight
  (`gpuwrf run --dry-run`).
- **Development cases (ALISIOS, three cases, descriptive):** a
  flow-dependent 10-member LETKF lowered pooled 12-hour station mean-squared
  error for T2 by 15 % (90 % interval −21…−8 %), Q2 by 29 % (two cases) and U10
  by 27 %, while V10 rose by 6 % (−1…+12 %); the gain was concentrated in the
  first three hours, one sonde comparison got worse, and the analysed trade-wind
  inversion decayed within 2–3 hours in the forecast model. These are serially
  correlated development numbers, not a confirmatory skill result.

## Limits

- d01 only: nested domains have a different base state after WRF's nest
  initialisation, and a consistent parent/child analysis is not implemented.
- MU (dry column mass) is not analysed. By default all members share one
  boundary file; the opt-in ensemble options ([da_ensemble.md](da_ensemble.md))
  add per-member perturbed boundary files, adaptive and additive inflation.
- With the default settings the ensemble loses spread quickly in those development cases
  (28–76 % per 6 h in the interior).
- **Known limitation of the opt-in ensemble options: too much spread.** In their
  prospectively registered real-model test (Swiss CPU-WRF twin experiment, final
  code, a truth run not used before, 20 members, analyses at 03, 06 and 09 UTC)
  the prior spread was 2.59, 2.32 and 2.53 times the actual error at 06, 09 and
  12 UTC. That fails the registered calibration check, which required 0.5–2.0.
  The other registered checks passed: the spread did not decay, the written
  members carry the analysis spread, and the analysis error was lower than the
  prior's (0.81 vs 1.31). The excess spread was already present before the first
  analysis, which points at the size of the initial and boundary perturbations.
  Nothing has been retuned after this result.
- No cloud or radiance assimilation (cloud masks are withheld diagnostics);
  one sonde site in the development cases. No saturation adjustment of the
  analysis (`supersaturation: keep` only).
- The analysis is host-only: about 11 s of analysis phases for 10 members on
  the 120 × 70 × 44 ALISIOS d01 grid on two CPU cores in the probe (16.7 s for
  the whole command including the mean write); memory grows with members ×
  analysed fields.
