# RW3 accumulated-rain floor policy — Owner and ALISIOS review

The binding 72 h WN3 0227 RW3 rule remains the pre-registered A3 rule:
for each raw D6 RMSE breach **X > limit**, classify it **FLOOR-LIMITED** if
**C > limit OR C/X ≥ 0.75**. C is the original CPU-WRF reference versus its
own ±1 float32 ULP initial-theta twin, at the same domain, field and lead.
The D6 limits and the A3 threshold do not change.

Accumulated precipitation at long leads integrates displaced weather events;
for this case the CPU twin itself exceeds the 1 mm RAINNC limit at d03 leads
68–72. This empirical spread supports the existing floor interpretation beyond
this case's useful predictability horizon. It does not establish a universal
horizon, confidence interval, or permission to use this reference on other dates.
Strict 24 h validation, independent station checks and early-hour twin gates stay
binding. Nonfinite values, missing fields/leads and integrity defects cannot be
classified as floor-limited.

The [full frozen reference](evidence/rw3/wn3_0227_full_floor.json) preserves all
2,406 available original CPU-pair RMSE values (d01–d03, 11 fields, leads 0–72).
Three lead-0 hourly-rain statistics were absent in the source and remain null.
The [generator](../../scripts/wn3_floor.py) verifies the SHA256 of the
[original archive](evidence/v033/w2/W2bR/metrics.json), then reproduces all six
historically frozen C values byte-for-byte **before** extracting further leads.
This is recovery of archived measurements, not a new CPU-twin simulation.

For d03 RAINNC at lead 67, archived **C = 0.8905983272943488 mm** and v0.4.4
**X = 1.0600553865983122 mm**, so **C/X = 0.8401432024719517**. A3 therefore
classifies this exceedance as FLOOR-LIMITED. All six rain exceedances at leads
67–72 satisfy A3. Raw D6 remains FAIL; the QGRAUP trace-zero integrity issue
remains a separate gate. The full reference prevents a lead from being treated
as “C absent” merely because it did not exceed D6 in an earlier release.

For future 0227 RW3 scoring, use `scripts/wn3_score.py --hours 72 --rw3-floor`
with the usual CPU/GPU/output arguments. `summary.json` preserves raw `pass`
and adds `binding72_A3`; `FLOOR72.json` records each breach and the reference
hash. An integrity failure still produces a failing process exit status. To
rescore existing metadata without running a model, use
`scripts/wn3_floor.py score --summary <summary.json> --out <new-report.json>`.
This report command does not adjudicate or modify an earlier release receipt.

The rise in X67 from approximately 0.50 mm in v0.4.3 to 1.06 mm in v0.4.4
still needs causal attribution. A3 addresses the floor interpretation; the
[pre-registered bisection](../../proofs/v45_rainfloor/GPU_BISECTION.md) investigates
TOPO, QSQ and E84 separately against original CPU-WRF. No corrected physics or
speed claim follows from the metadata recovery.
