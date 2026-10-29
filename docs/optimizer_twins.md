# Station twins for the regional optimizer

`gpuwrf.optimize.twin` is a CPU harness around the existing host Tikhonov EKI.
It generates observations from a known log-parameter truth at caller-supplied
station coordinates and times. Complete support and the registered outage mask
use the same synthetic observations and noise realization. Reports compare
parameter error, sensitivity rank, singular values, collinear columns,
information matrices and observation misfit. A perfect twin alone proves no
real identification or station improvement.

The module is an experimental Python API, with no new `gpuwrf optimize` CLI
command. It does not run forecasts or access observation services. Existing
continuous-study behavior and forecast stepping source are unchanged. GPU
execution, readiness and memory have not been measured by this harness.

## Private inputs and adapter

Get station geometry and outage support through the manager and the ALISIOS
facade. Keep those files, receipts and their generated predictions outside Git.
`load_design(station_path, support_path)` rejects protected roots before reading.
The station file uses the existing verification station schema:
`station_uid,lat,lon,elevation_m` (plus optional sampling metadata).
The support CSV uses `station_uid,valid_time,variable,available`, with explicitly
offset/UTC times and exactly `true` or `false` in `available`. Supply the intended
complete sampling grid, including unavailable rows. A missing row is not an
inferred outage. Real measured observation values are never synthetic truth.

```python
from gpuwrf.optimize.twin import (
    ForwardIdentity, PredictionBatch, TwinConfig, design_keys, load_design, run_twin,
)

design, input_receipts = load_design(private_stations_csv, private_support_csv)
identity = ForwardIdentity(
    forcing_sha256=actual_forcing_digest,
    configuration_sha256=actual_model_scheme_nest_sampling_digest,
    observation_schema="t2:degC;wind:m/s;instantaneous;sampling-contract-v1",
)
config = TwinConfig(
    names=("admitted_knob",), prior_sd_log=(0.3,), bounds_log=((-0.4, 0.4),),
)

def forward(theta_log, ordered_design):
    # Run/reuse each member with exp(theta_log) times its admitted base value;
    # sample every complete-design row at these real coordinates and times.
    # Derive identity from ACTUAL member inputs, units and sampling metadata.
    values, actual_identity = member_predictions(theta_log, ordered_design)
    return PredictionBatch(values, design_keys(ordered_design), actual_identity)

report = run_twin(
    design, forward, truth_log=[0.2], config=config, identity=identity,
    sigma={"t2": 0.5, "wind": 0.5},
)
# Persist report + input_receipts privately with allow_nan=False.
```

These example numbers define a synthetic experiment, not admitted physical knob
ranges, uncertainty estimates or scientific acceptance tolerances. The knob
contract supplies physical bounds and the log-multiplier conversion. At most
two continuous parameters may be active. Discrete schemes need separate studies.

The adapter returns finite predictions with shape `(members, complete_rows)`
and exactly the ordered keys supplied by `design_keys`. The forcing digest binds
all initial/boundary inputs. The configuration digest binds model revision,
admitted scheme/nest tuple and station-sampling method; runtime parameter values
are excluded. The schema names variables, units, time intervals and conventions.
Derive identities from the actual member products. Copying the expected identity
into every return would hide a broken forcing/schema experiment.

`sample_histories(private_paths, design, **sampling)` uses the actual WRF
verification extractor and strictly aligns its output to every registered row.
Its canonical units are T2 in **degC**, wind in m/s. Wind-height, roughness and
landmask metadata survive design construction; omitted fields receive verify's
defaults (10 m, 0.03 m, land). Freeze these assumptions in the sampling contract.
The returned audit records interpolation, elevation and wind-height settings.
Missing finite predictions for any requested row fail even if that row is in
an outage. This is a structural adapter check, not a WRF physics fidelity claim.

## Interpreting results

`complete` and `outages` are diagnostic scenarios. Null and central ± responses
at the table prior run before EKI. Rank uses the covariance-whitened response,
scaled by prior standard deviations. Near-collinear pairs have absolute cosine
at least 0.99. Sensitivities, singular values and the registered rank cutoff are
reported. This is a **local** sensitivity check; nonlinear/global uniqueness is
not established. Rank loss freezes the experiment (`UNIDENTIFIABLE_FROZEN`), and
an empty outage mask reports `NO_SUPPORT`. There is no fallback that fits the
remaining parameters and labels all parameters recovered.

Full-rank experiments use tempered EKI (sum dt = 1), registered priors/bounds,
and deterministic ensemble updates. `RECOVERED_SYNTHETIC_ONLY` requires every
known-truth error to pass the registered log tolerance and residual misfit to
pass the registered diagnostic threshold. Otherwise the status is
`NOT_RECOVERED` or `MODEL_MISMATCH`. Small residual alone is insufficient.
Ensemble spread is not a calibrated posterior uncertainty.

Declared wrong forcing, configuration or observation schema raises before any
EKI update. Wrong keys, order, dimensions and nonfinite predictions also fail.
An undeclared forcing error outside the parameter-response span can produce a
large residual. An error **inside** that span can be indistinguishable from a
parameter change. The harness cannot certify its detection from observations
alone; the report states this limitation. Independent forcing contrasts and
held-out real observations remain necessary.

Sigma is specified per variable. The default diagonal covariance explicitly
assumes independent errors. Provide a positive-definite full covariance to
represent correlation, or preregister independent blocks/aggregates. Do not
treat repeated correlated pixels/hours as independent evidence. The default
cap is 256 rows, refused before dense allocation. Any support reduction must be
preregistered; the harness never silently thins observations.

Covariance admission checks symmetry and correlation bounds in units of
`sqrt(diagonal_i * diagonal_j)`, then factors the normalized correlation matrix
to check positive definiteness. It rejects material asymmetry at tiny scales.
Roundoff asymmetry within `1e-12` in correlation units uses the same lower
triangle for every subsequent solve and for sampling; the caller's input is
preserved. Sigma squared must be representable as finite positive variances.

No work is added to the forecast timestep. Host EKI's existing solve scales
quadratically in retained rows in memory and cubically in solve work; finite
differences add `1+2p` forward members, batched in one adapter call. Complete and
outage fits each add `members × iterations` forecasts plus one evaluation.
Truth needs one forecast. The harness does not claim measured GPU runtime,
memory savings, compilation reuse or WRF fidelity from its CPU stub tests.

## Small real-forward pilot — plan only

The pilot is **not armed**. Real geometry/outage metadata and a frozen admitted
knob/configuration tuple must arrive through the manager first. No ALISIOS
payload has been copied into the repository. The CPU unit tests use explicitly
invented stations.

1. Freeze one WN3 three-nest training case and one different issue/date held out
   through the facade. Bind input SHA digests, all sampling adjustments, scheme
   tuple, source/bundle revision and complete support before any score. Restrict
   the pilot to 3 forecast hours per date and at most 32 preregistered independent
   station/variable/time blocks; verify real outage coverage. Use one continuous
   cloud-relevant runtime knob that the cloudknobs lane admits. Freeze all other
   parameters and every discrete scheme. Sigma, correlation, truth, prior,
   bounds and the recovery cutoff are registered before predictions.
2. Generate one truth member, table-prior null and ± perturbations (4 forecasts).
   Use the actual WRF station extractor on the real coordinates, with the frozen
   wind-height/elevation/domain sampling settings. Convert to the exact full
   design keys and declared units. Preserve unmatched keys as a hard failure.
   Compare rank/information for complete support and real outage support before
   fitting. A rank failure ends that fit and is an informative pilot result.
3. If rank survives, use 4 members × 2 tempered iterations plus one final-mean
   evaluation for each support arm: up to 18 forecasts after the initial 4.
   The maximum training cost is 22 × 3 = 66 case-hours. Outage and complete arms
   must share the truth/noise, actual forcing identity and parameter definitions.
   Reuse identical parameter-vector predictions by content identity only.
4. Evaluate both fitted means, the null and known truth on the held-out date
   (at most 4 × 3 = 12 case-hours). Held-out truth is withheld from fitting.
   Register a wrong-forcing input and wrong-schema adapter test: each must fail
   the identity gate before EKI, with zero parameter-recovery claim. Add an
   unlabelled forcing contrast as a diagnostic; report confounding explicitly.
   Total plan ceiling: 26 forecasts, 78 case-hours, plus host-only rejection tests.
5. Before requesting GPU time, the manager pins the composed reviewed source,
   validates the adapter's actual provenance, and supplies a night_seq launcher
   with official GPU lock/lease, backend assertion, WRF-run registry start/end,
   an exact all-phase SEQ_SMOKE receipt and private caches. Cold setup has its
   separate ≤60-minute cap and real executable cubin gate. Split measured arms
   to ≤30 minutes each; stop at the registered case budget. Measure actual cold
   readiness, cache hits/new compiles, wall, peak RSS/VRAM and forward calls;
   costs are currently [U], not inferred from the CPU stub. Follow current disk
   admission, clean own disposable products immediately, retain small SHA-bound
   receipts and all reference evidence. No GPU booking is implied by this plan.

The real-forward pilot tests optimizer plumbing and synthetic identifiability.
It does not replace pristine CPU-WRF fidelity gates, multi-date real station
validation, or the independent forcing/held-out tests required for promotion.
