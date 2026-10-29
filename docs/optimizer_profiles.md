# Profiles for optimizer waves

The optimizer can extract small profiles at explicitly selected north-windward
and sea grid cells from **completed** history files. This is a read-only,
out-of-loop consumer. It does not enable radiation or MYNN output switches,
change model inputs, rerun physics, or change prognostics. Missing histories
are `UNAVAILABLE`, with a reason; they are never interpreted as zero.

```python
from gpuwrf.optimize.diagnostics import write_profile_report, wave_report_entry

points = [
    dict(id="north", region="north_windward", domain="d03", j=30, i=70),
    dict(id="sea", region="sea", domain="d03", j=40, i=10),
]
# Example indices only: select the actual cells from the admitted case geometry.
write_profile_report(files, points, "profiles.json")
wave_report["diagnostics"] = wave_report_entry("profiles.json")
```

`files` is an explicit list of original CPU-WRF or GPU history paths from one
forecast issue. Points use zero-based mass-grid indices; their ids are unique.
Every point has an explicit nest and region. A present LANDMASK must agree
with the sea/land selection; north-windward geography is chosen by the caller,
not inferred from latitude. Absent nests are listed in `missing_points` with
report status `UNAVAILABLE`. Other reports say
`COMPLETE_WITH_UNAVAILABLE_TERMS`, meaning extraction completed, not scientific
qualification or optimizer admissibility. The wave entry carries the full
report's SHA256, closure maximum, timing, sampled bytes and missing terms.
The wave-entry reader rejects NaN, either infinity and JSON exponent overflow
anywhere in the report before reduction/export; `UNAVAILABLE` and null values
retain their meanings.

An offline command is also available with the same point list in JSON:

```bash
JAX_PLATFORMS=cpu python -m gpuwrf.optimize.diagnostics \
  --files run/wrfout_d03_2026-02-27_01:00:00 \
  --points points.json --output profiles.json
```

Outputs must be new files. Protected ALISIOS paths are refused before payload
access; use their admitted facade copies. The reader checks dimensions, units,
finite values, time uniqueness, issue, fixed sampled geometry and source stat
identity before/after extraction. It opens histories in read-only mode and
reads only selected columns. Source size/mtime/inode and sampled-byte digests
are recorded; these are not full-history content hashes. `sample_bytes` counts
logical uncompressed column values, not physical NetCDF/HDF5 chunk IO.

## Quantities and limits

- Height is mass-level `(PH+PHB)/9.81-HGT`. Theta is `T+300` K and water
  fields retain WRF's mixing ratios per kg dry air. Inversion height is the
  midpoint with the largest positive theta gradient between 100 and 4000 m
  AGL; this diagnostic definition is fixed, not a verified inversion tracker.
- Prognostic LWP integrates QCLOUD; CWP integrates QCLOUD+QICE. Both are in
  g/m², with layer thickness and ideal dry-air density from P+PB, T and QVAPOR
  (`Rd=287`, `Rv=461.6`, `Cp=1004.5`). Neither quantity substitutes for the
  separately contracted radiatively effective cloud operator/process score.
- Cloud top is the highest mass level with QCLOUD+QICE ≥ 1e-6 kg/kg.
  `cloud_top_lw_cooling` is minus the held **theta** LW rate at that level,
  multiplied by Exner to give temperature cooling in K/s. Negative cooling
  means warming; it is not clipped. Missing heating or a cloud-free column
  gives `UNAVAILABLE`. This is the prognostic cloud top, not a radiative top.
- HFX, LH, QFX and GRDFLX retain history units and signs. All existing MYNN
  budget/EDMF/raw PBL rates and eight `rad_diag` volumes are copied, if present.
  Z-staggered MYNN budget histories contain an already destaggered mass prefix
  and unused zero top: the reader trims the top without averaging again.
- Local TKE closure is `Dtke-(qSHEAR+qBUOY+qWT-qDISS)`, in m²/s³ even though
  the original Registry labels the raw rates m²/s². The unused last mass level
  is excluded. The profile, RMS and maximum absolute residual are reported.
  The original WRF residual is nonzero; there is **no invented zero-closure
  tolerance**. The existing pristine day/night residuals span maximum absolute
  0.0004031–0.013866 m²/s³ ([existing evidence](../proofs/w44_budget/BUDGET.md)).
- `pbl_water_tendency` integrates RQVBLTEN+RQCBLTEN+RQIBLTEN in kg/m²/s.
  `pbl_radiation_sensible_energy_tendency` integrates Cp·Exner times
  RTHBLTEN+RTHRATLW+RTHRATSW in W/m². These are **partial, instantaneous held
  sources**, sampled at the history time. Different physics packages may have
  different last-call times. They are not time integrals or total budgets;
  latent energy, resolved transport and boundary sources are not included.

Every row has a history valid time and forecast issue. Radiation/MYNN **call
times** and total water/energy **budget residuals** are always `UNAVAILABLE`:
the required operands are not recorded by ordinary history. The extraction
wall time is a separate host measurement, not a physics call time.

## Existing producer qualification

For output-only consumption, the four held RTHRATLW/RTHRATSW/RTHRATLWC/RTHRATSWC
rates have pristine-WRF column-oracle evidence. QC_RAD/QI_RAD/QS_RAD/QG_RAD
expose the actual mediated solver inputs; they have no ordinary CPU-WRF
history equivalents. Their source/merge checks and the radiation rates are
documented in [radiation diagnostics](radiation_diagnostics.md).
The historical three-nest GPU check preserved 378 shared history variables
over 12 frames (4,536 byte comparisons) and passed the actual native resource
gate. This permits interpreting those recorded fields within that scope;
it does not prove every hidden carry leaf or general day/night forecast
identity. This consumer does not newly enable `rad_diag`.

MYNN budgets have original-WRF day/night component oracles, but their coupled
GPU ON/OFF identity **failed** (E242). CPU equality did not predict GPU fusion
equality. The consumer can inspect such research histories and CPU-WRF
reference histories; it does not qualify that in-loop producer or authorize
using it in an unchanged-control optimizer arm. See [MYNN budgets](mynn-budgets.md).
All forecast/runtime/compile source remains unchanged for this consumer, so
CPU and GPU programs remain identical by construction when it is used after
publication with the same forecast options.

## Deferred taps and effort

These are engineering estimates [I], not booked GPU windows. Do not enable
an observer simply to fill an unavailable value.

| Missing quantity | Required change | Estimate |
| --- | --- | --- |
| Radiation and MYNN producer call timestamps | Carry each actual per-nest call clock to history, preserving held-value provenance | 1–2 days together |
| Byte-neutral MYNN terms in an unchanged GPU control | Separately compiled out-of-loop replay from exact prepared operands, or a new in-loop design with physical identity | 2–4 days plus compile/GPU window |
| Total water residual | Exact interval accumulators for dynamics, microphysics, surface, nesting and boundary exchange, with storage convention | 3–5 days |
| Total energy residual | Corresponding sensible/latent/geopotential/kinetic storage and all transport/source terms | 3–5 days, some shared with water |

Every future in-loop variant requires CPU physical BYTE identity **and** an
E156-faithful deviceless GPU comparison, followed by the actual GPU-compiled
ON/OFF carry/history identity, original-WRF term gates, E41, kernel/copy counts,
peak memory and timing. A barrier or CPU-only success cannot waive E242.
For this consumer, forecast kernel launches, device transfers, GPU allocations
and compile/readiness cost are unchanged [I, source identity]. Extraction costs
host column IO and a small JSON artifact; no full-domain arrays are allocated.
