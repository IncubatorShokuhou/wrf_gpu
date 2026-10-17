P3 compiled qualification under the wave's 2026-10-13 contraction policy

The literal REAL4 path uses the port's generated formulas, control flow, table
lookups and adaptive sedimentation, with host libm's REAL/DOUBLE intrinsics and
no fused array arithmetic. P353b matches all saved fields BYTE for mp50–53,
8 columns × 12 levels × 3 calls, against the independently compiled original
WRF wrappers. It invokes no original P3 routine. Its inactive gather indexing
matches JAX's indexing semantics; all published values are checked without masks.

For the compiled path the bound is registered before measuring the final
composed intrinsic candidate. P348 separately builds the byte-identical original
module with contraction off/fast, compiler-defined zero initialization, and the
EXACT same frozen frame0 operands. The control without contraction is BYTE to
all 140 original fixture fields; changing initialization to SNAN/integer42/
logicalTrue is also BYTE. Thus this corpus has no observed uninitialized-output
dependency. Each field's independent L-infinity FMA-vs-no-FMA spread over the
three calls supplies the compiled absolute bound. Original frozen 32eps/absolute
scores remain reported separately; no fixture is regenerated from the port.

There is one additional independently derived discontinuity bound. WRF's cloud
sink limiter subtracts nearly equal QC source and dt*scaled sink. REAL
operator roundings per source/sink path are conservatively bounded by 32
(two-category sums, additions, ratio, scale, dt and final subtraction). The
two compared paths give 2*gamma32=2*32u/(1-32u), u=2^-24. The initial
density*dz weighted column-water inventory divided by the column's minimum
layer mass bounds the mixing-ratio residue. Only when BOTH saved
reference and candidate final QC lie within that residue can the diagnosed
radius cross WRF's qsmall occupancy branch. The no-cloud fallback is 10um;
the cloudy branch has mu>=2 and lambda>=(mu+1)*25000, so
re=.5*(mu+3)/lambda <= 1/30000 m. The maximum radius difference is therefore
bounded by 1/30000 m at these near-zero-water points. This changes no physics
cutoff, state value, cleanup, finite check, or mass-field bound.

These columns qualify the implementation and caller substrate only. Long
coupled forecast accuracy is governed by D6 against original CPU-WRF; GPU
execution, memory/resource and performance qualification remain pending.
