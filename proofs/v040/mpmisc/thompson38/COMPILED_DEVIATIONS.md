# Compiled Thompson38 deviations and resource scope

H66 (`beabc51c6`) has a GREEN literal primary gate: 6,160/6,160 column and
77/77 public surface values BYTE against eleven original-WRF callers.
The evolving three-own-step literal gate is also BYTE: 1,680 column and
21 surface values. These are independent original-reference comparisons.

The compiled one-step result is **not a strict interval PASS**:
115/6,237 values lie outside the independently measured original intervals.
[compiled_final_bound.json](compiled_final_bound.json) records every case,
field, count, maximum excess and unit. The standing wave policy permits
small documented excess as a disclosed deviation; full coupled D6 decides.
No candidate output was used to widen an interval or choose a tolerance.

The reference ensemble contains 407 original calls: baseline plus +/-1
NEAREST perturbations of each of 18 input fields, independently per field
across the column, with zeros held zero. Baseline inputs and outputs are
BYTE for all eleven cases. Original native-FMA outputs for cases 1-9 are
reused. An additional 74 original calls cover native FMA plus the same
37 input variants for each threshold case 10/11. Original bodies and cached
table records remain unchanged. These are measured response intervals,
not a mathematical interval enclosure of every simultaneous perturbation.

All compiled threshold-case column and surface values fall inside that
combined original interval, including the large branch-sensitive ice-number
change. Cases 10/11 deliberately straddle the 999e3 working-number gate;
each has one negative incoming NI1D level because the constructed snow
number exceeds the target. This is an audited stress edge, not a forecast
input claim. Original raw NI1D baseline cancellation is preserved.

The remaining strict excesses are in ordinary/fresh cases. Representative
maxima beyond the interval are:

| Public quantity | Maximum excess | Unit |
| --- | ---: | --- |
| Water mixing ratio | 1.1641532183e-10 | kg/kg |
| Ni | 595.625 | number/kg |
| Nc | 6.875 | number/kg |
| NWFA | 576 | number/kg |
| qvolg | 1.1368683772e-13 | m3/kg |
| Snow accumulation | 2.3283064365e-9 | mm |
| SR | 1.1175870895e-8 | dimensionless |

Public Ni/Nc/NWFA are carried per kg; private working ni/nc/nwfa are per m3.
Every compiled prognostic and surface output is finite and REAL32. The
compiler-reported CPU artifact has 19 REAL32 input arrays and 5 REAL32
surface outputs. For a 6x40 batch, temporary storage is 314,536 bytes.
This excludes the 929,643,480 bytes of new fixed table/coefficient payload,
host caches, retained legacy global buffers and process overhead. In
particular, legacy Aero tables may remain allocated alongside the private
255 MB freezing binding; no unchanged-memory or GPU VRAM claim is made.

Measured CPU compile-plus-first-call walls are 25.696/26.101/21.422 seconds
for the 6/3/2 case groups. These exclude prior module/table loading and are
not warm forecast throughput. The optimized CPU HLO inventory is retained
with source attribution; the new cloud DOUBLE reciprocal matches the
original declared island and its regression passes. The implementation is
shared JAX code with private native-REAL bindings, not a Pallas kernel.
GPU HLO/SASS/E41, launches, roofline, peak VRAM and full coupled D6 are pending.
