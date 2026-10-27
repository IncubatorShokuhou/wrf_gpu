# S2 inventory: RN-emulation f64 island

Authority: manager MAIN MAILBOX ruling `2026-10-25T14:10:01Z` and cost
clarification `2026-10-25T18:26:04Z`. This is a deliberately named
**RN-emulation f64 island**, not a general precision promotion.

`wrf_cam_ozone._real4_div_rn` first rounds both inputs to REAL4, widens them
exactly to binary64, divides, and rounds the result back to REAL4. Binary64
has 53 significant bits, exceeding `2*24+2`, so double rounding of division
does not change the REAL4 RN result. The opaque divisor prevents XLA
constant reciprocal folding. The real operand-range probe compares
14,398,452 quotients against NumPy REAL4 true division with zero differing
bits. The product replay separately checks 9,390,776 protected quotient
values over the 677 frozen radiation clocks.

Two compile-time quotient families are host REAL4 constants, without new
f64 device operations: `pi/180 = 0x3c8efa35` and
`360/365 = 0x3f7c7e3f` (binary32 bits).

The seven dynamic solar families are lead/60, absolute-minute/1440,
EOT-phase/365, midpoint/60, local-solar-minute/60, longitude/15 and
SOLCON-phase/365. The five dynamic ozone families are lead/60,
cyclic-factor1 early, cyclic-factor2 early, linear-factor1 and
linear-factor2. Child domains consume parent-held ozone and have no local
ozone temporal node. Root solar and ozone lead/60 may CSE to one divide;
the report requires the direct ozone-clock consumer of that same narrowed
solar result as evidence.

Counts are static **optimized instruction sites**, not cadence-weighted
device operations or measured execution time. Each `RN_AUDIT.json` records
the divide, operand conversions, result conversions, field shapes and
source-family lineage. Every divide and narrowing is tied through the
native thunk's profile annotation to its exact emitted PTX kernel, where
`div.rn.f64` and `cvt.rn.f32.f64` are required. Full-constant raw/optimized
protobuf hashes bind folded and large constant values.

No State leaf, domain-sized carry, host transfer, explicit custom call,
Pallas kernel or host synchronization is added by this source patch. XLA still
chooses fusion boundaries around the rounding barriers. Additional scalar
fusion kernels and the longitude field's binary64 temporary are counted
against unchanged frozen-source CompileOnly baselines in the final gate
packet; the source alone is not performance evidence. Actual GPU timing,
VRAM, readiness and composed-RC RW1–RW4 remain required release evidence.

## Measured CUDA CompileOnly inventory

All counts below are [M CPU, CUDA target], with native-first tracing and no GPU execution.

| Program | Trace Pallas / optimized native / unique bodies | RN f64 divides | f64 operand converts | f32 result converts | Divide elements |
|---|---:|---:|---:|---:|---:|
| Swiss d01 | 159 / 144 / 62 | 11 | 10 | 11 | 1774 |
| PROD d01 | 161 / 137 / 61 | 11 | 10 | 11 | 8410 |
| PROD d02 | 151 / 135 / 59 | 7 | 4 | 7 | 31245 |
| WN3 d01 | 161 / 137 / 61 | 11 | 10 | 11 | 8410 |
| WN3 fused d02+d03 | 329 / 297 / 118 | 14 | 8 | 14 | 41574 |
