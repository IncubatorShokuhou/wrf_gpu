# v0.4.0 original-WRF validation figures

These figures reuse sealed full-array producer scores against original CPU-WRF.
No forecast rerun, archive rewrite, tolerance change or GPU-vs-GPU fidelity
comparison was used. [PROVENANCE.json](PROVENANCE.json) binds every original
report, source commit, actual bounds, input/reference identity, plot-data SHA
and rendered image SHA. [Final validation ledger](../FINAL_COMPOSITES_v0.4.0.md).

- [Composite matrix](composite_matrix.png):66 configs ×62 bounded fields;
  ten forecast RMSE tests and52 grid/static max-absolute checks shown separately.
 48 static rules require exactzero; four retain their nonzero frozen limits.
  c45's6h scope is labelled; other 65 cases cover 24 hours. [Data](composites.csv).
- [WN3 bands](wn3_deviation_bands.png):0227 source353a85f8a, all three domains,
  25 frames each,10 forecast fields. RMSE and spatial 95th-percentile absolute
  errors are reused from complete-field scores; they are not confidence
  intervals or pointwise gates. [Data](wn3_bands.csv).
- [Tiedtke before/after](c06_tiedtke_rain.png):same original CPU receipt and
  exact actual-input hashes,25 frames before04cb09f816/afterdb78f9bd8c.
  Worst hourly RAINNC RMSE1.1108564→0.0296773mm (37.431×), not a general
  forecast-skill or speed claim. [Data](c06_rain.csv).

SVG versions accompany every PNG. All images are under 1 MB. Full observed
scores are retained in CSV; colors do not replace the scientific/reporting
scope. REPORT_ONLY fields have no invented bound and E41/F2 are not encoded
as a scientific PASS by these charts.

Reproduce from the compact CSV data with Python, NumPy and Matplotlib:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/your/scratch python render.py
```

Run in this directory or pass the path to render.py. The renderer reads its
adjacent CSV files, generates the six PNG/SVG artifacts and updates their
hashes. It does not read scientific archives. Matplotlib/NumPy versions are
recorded; image bytes can vary with plotting-library/font versions.

## Public README variant

[readme_validation.png](readme_validation.png) and its
[dark-mode twin](readme_validation_dark.png) show the same WN3-0227 data as
the band figure in a compact 2 × 5 layout with plain-language titles for the
project README. [render_readme.py](render_readme.py) draws them from the same
`wn3_bands.csv` (SHA-256 `51ce71fb9499d4fc47beb829817251b8098f12454dc2f3b4ff89e94d9ac5e1f3`); only layout, labels and colours differ.
Rendered with Matplotlib 3.10.8: light `8b9359465a31f3370f2d349d44aeaeb6b4684dde1edcdfbb3b9be517d73a96aa`, dark `e9f2477f8ce1a8b9632cdc3bfa6987155eb768e0646e2ec0edb23bd62aa00bfe`.

```bash
MPLCONFIGDIR=/your/scratch python render_readme.py
```
