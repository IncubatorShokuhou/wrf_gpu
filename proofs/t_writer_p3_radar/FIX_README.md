# Same-critic FIX: daily relay and reproducible P348 reference

The shared daily output helper now overlays the existing radar plane after
selecting either `operational_carry` or `ruc_carry`. Every applicable return
uses that overlay, including RUC, SSiB, lake, TEMF and their combinations.
There is no scheme check: P3 and NSSL use the same forwarding path.

Thirteen tests invoke the real `_run_forecast_sequence` routing and real
output helper, stubbing only forecast stepping and unrelated surface producers.
They verify the chosen carry argument, exact object/byte retention, preservation
of other diagnostics, and delivery to the writer. All thirteen plus twelve
existing daily/clock consumers pass. Restoring the reviewed old helper gives
ten genuine value assertions for the affected routes and three passing controls,
with zero setup errors. The targeted RUC revert also gives one genuine assertion.

The original P348 literal/FMA arrays and report are now tracked byte-for-byte
with provenance and SHA256s. The test verifies all three hashes, the accepted
P348 report hash and the registered bound. The strict 0.0003 dBZ band and
accepted 4.766279220581055 dBZ WRF contraction spread are unchanged. The actual
three-step oracle passes from a clean git archive using its tracked references.

Only one host output function changes relative to the reviewed candidate.
The daily route selector, P3 mathematics, timestep, kernel, carry, restart and
other IO code remain unchanged. Prior numerical, native CPU and full default
identity evidence is reused. The helper adds no array computation or copy;
previously omitted routes now incur the normal transfer and IO for publishing
the existing radar plane. No new forecast-performance claim is made.

`FIX_QUALIFICATION.json` seals this evidence. Actual 27-output GPU/E41/full-c23
fidelity, VRAM and runtime qualification remain pending. Prior strict residuals,
inherited allocation-count failures and unrelated full-smoke resource failures
retain their existing classifications.
