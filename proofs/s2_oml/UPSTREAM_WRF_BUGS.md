# Pristine WRF behavior and REAL arithmetic

All WRF OML/MYNN/old-MM5/revised-MM5 source bodies remain unmodified; hashes
and compiler flags are frozen in the fixture manifests. Compiler initialization
was varied from zero/0/false to snan/-999/true with FP traps off. The 24 OML
columns and all six hurricane arms are byte-equal between those modes. No
initialization-sensitive upstream defect was found on these active regimes.

The old-MM5 outer wrapper slices USTM/CK/CKA/CD/CDA when forwarding its optional
arguments. An initial thin driver omitted those arrays and failed at that slice;
the corrected driver supplies the actual surface_driver argument shapes and
INTEGER SCM_FORCE_FLUX=0. This was a driver wiring error, not a WRF body fix.

OML1D literal REAL4 outputs are byte-exact against the original no-FMA source
for all 24 columns (including eight-step trajectories). Compiled results are
gated by pristine-only +/-one-input-ULP envelopes and the original source's
FMA-vs-no-FMA spread. The same conditioning procedure governs complete hurricane
surface-layer columns. The old-MM5 third high-wind column differs by one ULP in
Davis ZNT (1.1641532182693481e-10 m); the pristine native FMA build exhibits
exactly that spread. No numerical tolerance is derived from port output.

The independent bound exposed a real legacy-flag seam: MYNN must always repeat
the moisture-resistance computation in module_sf_mynn.F:1023–1025 before QFX/Q2.
The new hurricane path explicitly executes that WRF operation and REAL constants,
even when the legacy default precision/fidelity flags are off. The zero-option
release path retains its existing flag behavior.
