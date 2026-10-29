# Joint forecast scoring under an ALISIOS evaluation contract

`gpuwrf optimize assess --assessment assessment.json --output impact.json`
reads aligned exports on the CPU and writes JSON plus an adjacent Markdown
impact matrix. `multiobjective.assess(...)` exposes the same calculation to
wave report producers. The caller must use admissibility before ranking loss;
this offline command does not launch models or update EKI. Wave execution,
wallclock/retry caps and persistence of the best admissible incumbent belong
to the wave controller.

The ALISIOS evaluation-contract envelope below is a proposed consumer interface
for the Astra addendum. Its authoritative values and freeze come from ALISIOS;
the optimizer cannot create an admission, native reference, tolerance, cohort
or gate. Until ALISIOS supplies and freezes this contract and complete reference
values, an assessment cannot support a scientific win. All examples/tests are
synthetic, not admissions. No ALISIOS production paths are read; stage admitted
facade copies first. No ALISIOS data are committed.

The assessment JSON uses schema `gpuwrf.optimize.multiobjective.v1` and has:

- `observations`: truth CSV path; `control`: fresh control CSV path.
- `references`: **both** `wn3_0p05` and `wn3_0p1`, each a Parquet selection object
  with `path`, `role`, `issues`, and `contracts` (described below).
- `evaluation_contract`: `{ "path": "frozen_contract.json", "sha256": "<64 hex>" }`.
- `candidates`: map of arm IDs to `{ "forecast": "arm.csv", "parameters": { "knob": 1.1 } }`.
- `regions`: point ID to region name, exactly as in the evaluation contract.
- Optional `points`: variable to its explicit point ID registry. This supports
  cloud pixels and a distinct station network. Otherwise every variable must
  cover every region-map point. Optional `seed` (default 0) and `repetitions`
  (default 400) affect descriptive bootstrap CIs only, never the gate.

Unknown assessment fields fail, preventing a misspelled gate from silently
selecting defaults. The configuration, frozen contract and all value files
are SHA-256 bound and checked again after scoring. Existing reports are refused.
Errors emit machine-readable `status: UNAVAILABLE`, `recommended: null`, and
`admissible: false`, plus the reason, with CLI exit 2.

Successful reports have `status: EVALUATED`, `scores: [{id, loss, admissible,
status, mae}]`, where candidate status is `ADMISSIBLE`, `HARM_VETO`, or
`INSUFFICIENT_DATES`. `objective_identity` binds the canonical frozen contract,
truth and all three baseline errors; candidate IDs/settings do not change it.
The wave controller must bind the report to its run/study/wave and compare
incumbent losses only with the same objective identity. No admissible result
must leave the prior admissible incumbent untouched.

## Frozen evaluation envelope

The external contract explicitly supplies **every** gate/weight and fixes the
pair denominator before looking at candidates:

```json
{
  "authority": "ALISIOS",
  "contract_id": "<externally issued freeze ID>",
  "frozen": true,
  "support_sha256": "<SHA of canonical sorted case/point/time/variable keys>",
  "cases": {
    "20260110_18z": {"issue": "2026-01-10T18:00:00Z", "episode": "episode_1", "regime": "trade_cloud"}
  },
  "episode_weights": {"episode_1": 1.0},
  "regime_weights": {"trade_cloud": 1.0},
  "lead_bins": [{"name": "h00_24", "start_hour": 0, "end_hour": 24}],
  "regions": {"station_n": "north", "msg3:1234": "north"},
  "scales": {"cloud": 0.1, "wind": 1.0, "t2": 1.0, "rh": 5.0, "gust": 1.0},
  "weights": {"cloud": 1.0, "wind": 1.0, "t2": 1.0, "rh": 1.0, "gust": 1.0},
  "margins": {"cloud": 0.0, "wind": 0.0, "t2": 0.0, "rh": 0.0, "gust": 0.0},
  "alpha": 0.05
}
```

The numbers above illustrate fields, not authorized gates. `support_digest`
computes the canonical key SHA independent of row order and timestamp storage
precision. An upstream ALISIOS adapter freezes it with the valid/QC/day masks.
The exact expected support cannot shrink even if truth, control, candidates
and references all drop the same rows. Every declared case and cohort remains
present. Region assignments, episodes, regimes, positive weights and sorted,
nonoverlapping `[start_hour,end_hour)` lead bins are immutable. Observations
outside the frozen lead bins fail. Episode weights normalize **within** each
regime, then fixed regime weights combine those means; neither observed skill
nor data loss changes these denominators.

## Values and native WN3 references

Truth/model CSVs contain `case_id,station_uid,valid_time,variable,value`.
Times require explicit UTC/offset and exact matching; select issues and apply
the frozen masks upstream. The in-memory API supports one omitted case_id as
`case` for a preselected issue. Overlapping issues require distinct case IDs.
Instantaneous variables are `cloud` (cover fraction 0–1), `wind`/`gust` (m/s),
`t2` (°C), `rh` (0–100%). Scales, positive variable weights and nonnegative
physical-unit margins must equal the external contract. Every variable must
cover its full registered point set. NaN, Inf, duplicates, missing stations,
missing timestamps and extra keys fail before aggregation. There is no inner
join, interpolation, implicit QC filtering, missing=0 or normalization in
response to missing observations.

The reference follows ALISIOS's delivered point/footprint Parquet contract.
Install `gpuwrf[optimize]` for PyArrow. Required columns are `case_id,valid_time,
location_id,variable,reference_role,value,units,native_statistic,native_interval,
native_lead_hour,operator_sha256,calibrator_sha256,available,unavailable_reason`.
The unique key is case/time/location/variable/role/statistic/interval. Each
resolution arm selects `role: wn3_raw` or `wn3_calibrated`; raw never substitutes
for calibrated. `issues` maps original case IDs to issue timestamps; native
lead must exactly reproduce valid_time. `contracts` maps each native variable
to exactly `native_statistic,native_interval,operator_sha256,calibrator_sha256`.

Delivered native names/units are `cloud_occurrence`/`1`, `wind_speed_ms`/`m s-1`,
`t2_c`/`degC`, `rh_pct`/`%`. The consumer also admits `wind_gust_ms`/`m s-1` for
an explicit upstream gust contract; **no gust reference was supplied in the
structure package**, so none is inferred. The delivered instantaneous interval
is exactly `instant_valid_time; ensemble mean not time average`. Examples of
native statistics are `WN3_total_cloud_cover_mean`, `N0_scalar_ensemble_mean`,
`temperature_2m_mean`; native hourly RH remains unavailable in the supplied
package. No N1=norm(vector means) alias, implicit K→°C conversion, time-average
alias or adjacent-hour fill is allowed. Other interval products require an
interval-aware consumer and are refused here. Operators require their full
SHA256; calibrated roles require a matching calibrator SHA and ALISIOS owns
the disjoint freeze. The upstream contract must establish comparability of
model/truth/reference sampling, statistics and spatial operators.

Missing selected rows, `available=false`, non-finite available values and
unit/operator/calibrator mismatches prevent a win. The ALISIOS FORMAT_ONLY
example has all null/unavailable values and is **refused**, never fit/scored.
There are no real values hidden in that package. Both spatial resolutions and
the control remain independently binding; absence of either reference is an
explicit unavailable result.

The cloud API is a stub for the cloud scorer's co-located fractional/binary
cover. MSG code 255 is invalid, never clear. Footprints, exact daylight slots,
parallax limitations and CWP/subgrid/radiative-cloud definitions belong to
that scorer's frozen operator. CSI/ETS/FSS/displacement are separate controls,
not interchangeable with MAE. Native ensemble cloud cover and CWP footprint
occurrence are not silently relabelled as calibrated cloud probabilities.

## Score, veto and simultaneous uncertainty

The joint loss averages absolute error over time per point, points per region,
regions per episode, episodes with frozen weights within regime, then regimes
with fixed weights. Finally it averages `MAE / scale` across variables using
fixed variable weights. Admissible candidates sort first, then loss and ID;
no admissible candidate gives `recommended: null`. Even a large cloud gain
cannot average away wind/T/RH/gust harm: every variable has an independent
point **and** region veto, separately in each lead bin and regime, versus
**each of the two WN3 arms and control**.

The paired effect is candidate minus baseline absolute error on identical
observations. Each UTC date is one block; region daily effects give points
equal weight. A one-sided exact paired sign test asks whether the median daily
error difference exceeds the predeclared margin. Exact ties count as nonpositive
trials (discarding a large atom at zero could falsely reject a zero median); no
symmetry assumption is needed. Holm adjustment controls family-wise false
discoveries at external `alpha` over all candidates × variables × points/regions
× leads × regimes × three baselines. Positive effects mean degradation in
physical units. A candidate is inadmissible if any cell is significantly worse
or has fewer than two date blocks, irrespective of its joint loss.
This control applies to the complete submitted assessment family; sequential
adaptive waves require an ALISIOS-issued testing/alpha-spending plan, and this
consumer makes no global false-discovery claim across repeated assessments.

`simultaneous_interval` uses exact binomial sign/order-statistic median CIs with
Bonferroni correction across the **same full family**, giving at least
`1-alpha` simultaneous coverage. Null endpoints explicitly mean unbounded
uncertainty due to too few dates. `interval95` separately preserves a seeded,
pointwise descriptive date-bootstrap interval; it does not decide admissibility.
Independent dates are an assumption: correlated adjacent dates can invalidate
both uncertainty claims, so ALISIOS must freeze appropriate independent cases
or aggregated blocks upstream. A single 72 h case has little statistical power.
**Non-rejection is a screening result, not proof of non-inferiority or a
guarantee of being nowhere worse.** No release promotion is inferred.

The JSON and Markdown impact matrix includes parameter × variable × region ×
lead × regime, baseline, effect, simultaneous CI, adjusted p and admissibility.
Multiple changed parameters are labelled `joint_parameter_change: true`:
these are associations, not isolated causal parameter effects. Counterfactual
parameter attribution requires one-at-a-time or designed factorial arms.

This code only runs offline: it changes no timestep arithmetic, compiled keys,
GPU launches or model allocations. Memory is linear in aligned input rows and
the output test family; bootstrap workspace is bounded to 128 × dates per test.
The ordinary forecast parser registers the command lazily without importing
the NumPy/pandas scorer. CPU cost is measured separately on synthetic exports;
no model/GPU speed claim is made.
