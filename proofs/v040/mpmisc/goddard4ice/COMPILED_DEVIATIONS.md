# Goddard4ice compiled REAL32 qualification

Measured CPU source 7c71fcbaa: 9 tests pass (81.65s), including all 18
original-WRF columns, all 36 output groups, a genuine hail-sedimentation
mutant and the active-hail Pallas interpreter. Literal no-FMA arithmetic
matches all 22,410 public REAL values BYTE. Zero/snan compiler initialization
produces BYTE-identical original outputs in six complete regimes.

The independent compiled bound sums original-WRF +/-1ULP responses for 14
recorded input vectors and adds original physics-only native-FMA spread.
504 perturbations and 18 native-FMA columns were run; generic-driver inputs
remain BYTE equal. All 5,760 compiled prognostic values lie within that bound.

27 of 22,410 public values exceed the strict independent bound: 20 categorical
radius values, 6 Registry '-' precipitation-flux values and one ACPHYSF value.
Maximum radius excess is 0.0001220703125 microns; maximum flux excess is
2.3283064365386963e-10 kg/m2/s. The one ACPHYSF value has absolute error
2.9802322387695312e-8 K and excess 1.862645149230957e-9 K. Exact field/cell
counts are in RESULTS.json. These are disclosed small compiled deviations
under the standing policy; this is not a claim of strict compiled-envelope
parity. A separately fixed 128-epsilon REAL regression budget passes.

CUDA/sm120 deviceless IR lowering passes (0.755s, 1,540,278 IR characters).
There was no GPU execution, cubin/E41 scan, GPU timing, VRAM measurement or
24h coupled D6 validation. Those qualifications remain pending. Default
native output elides Registry '-' latent/flux fields and adds no domain
scratch arrays; exact physical state arrays are retained by the root lane.


Performance caveat: the native path has no explicit full-domain scratch or
transpose, but serial downward `_point` access currently uses masked register
reductions to avoid unsupported dynamic gathers in Pallas. This adds roughly
four reductions per level per substep, with quadratic arithmetic in column
levels. CUDA IR compatibility does not prove efficient GPU execution or a
roofline result. GPU performance, STACK/spills and E41 remain unmeasured.
This is a CPU-oracle-qualified opt-in substrate.

The subsequent rain_other output enabler (01626b100) adds precisely the
REAL sum pptrain+pptice and one unheld surface output (4 bytes/column), with
no scratch. RAIN_OTHER_SOURCE_PROOF.json projects away that single dictionary
keyword and native SURFACE append and proves all prior equation ASTs exactly
match the validated 7c71fcbaa source. Prior numerical qualification is reused.
The changed native ABI was independently lowered again on immutable7f2ab2bdc:
CUDA/sm120 PASS, 0.842 seconds, 1,541,232 IR characters; no GPU execution.


PHYS_TOT history enabler ddc43c72c adds one REAL4 mass-grid output/held
field (4B/cell, no scratch or extra launch), while individual six instantaneous
rate outputs stay elided from production. It is the exact original
microphysics_driver left-associated c+e+d+s+m+f sum. A tiny independent
Fortran program executes that verbatim original assignment with 4,320
already-qualified original rate operands; all input bytes survive its I/O and
720/720 new totals match the production sum AST BYTE. Only the added expression
is evaluated; old microphysics science is not repeated. Prior output equation
ASTs remain exact. Changed38-output CUDA/sm120 IR passes0.351s,1,543,470chars.
Original Registry PHYS_TOT is history-only h, so it is not a required original
WRF restart prognostic. GPU execution and coupled forecast qualification remain
pending; existing compiled diagnostic deviations are unchanged disclosures.
