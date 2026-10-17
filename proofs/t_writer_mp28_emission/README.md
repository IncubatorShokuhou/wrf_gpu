# Thompson surface emission: original CPU-WRF frame-zero proof

WRF `start_em.F:873` passes interface `(PH+PHB)/g` to `thompson_init`.
The old emission producer used mass-level averages. At cell j21/i18 this
changed the initialization height from 2479.18 to 2501.91 m, crossing the
2500 m aerosol branch: CPU 32030.0977 versus old GPU 9518.2725.

The correction uses the two bottom interfaces and WRF REAL4 operands.
An optimization barrier on gravity preserves division rounding under JIT;
reciprocal multiplication otherwise changes the thin first-layer depth.
The input has no QNWFA/QNWFA2D, so WRF computes the fallback once. There is
no grid-area or timestep scaling in this initialization. Existing aerosol
inputs retain their emission, and NIFA2D retains its input/default value.

The actual native standalone CPU initializer matches the original c19/mp28
frame-zero QNWFA2D within 3 float32 ULPs, maximum absolute error 0.015625;
QNIFA2D is byte-exact. The compiled producer differs by at most 19 ULPs,
relative 1.408e-6 within the existing 32eps32 FMA bound (3.815e-6).
These bounds were recorded before final producer qualification. The original
c19 and mp28 CPU files are byte-identical. Eight focused tests pass; reverting
the real producer to mass heights causes one value assertion and zero errors.

`QUALIFICATION.json` seals the sources, raw receipts and frozen original-file
fixture. `SOURCE_SCOPE.json` proves that only the emission function differs
from main and that its measured helper ASTs survive the rebase unchanged.
Default program identity is reused from accepted batch15, with the dormant
mp28 selector/source proof; no new default tracing claim is made.

Measured CPU native load plus qualification: 9.04 s, peak RSS 1.951 GiB in a
6 GiB/no-swap scope. Carry size, restart schema and initialized timestep work
are unchanged. The startup calculation uses two float32 interface planes;
the previous eager path materialized whole-column float64 mass heights.
GPU launch count, initialization cost, compilation and E41 remain unmeasured.

The old GPU output remains defective for QNWFA2D. This packet qualifies only
the CPU frame-zero fix. Full affected GPU qualification is pending. The
separate 3-D aerosol fallback still uses mass heights; that source issue was
reported to the manager and remains outside this requested small correction.
QNBCA and TAOD5502D remain uncomputed/disclosed.
