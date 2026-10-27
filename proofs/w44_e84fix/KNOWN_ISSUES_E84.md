# E84 remaining scope for the v0.4.4 RC

The RC repair protects **14 quotient families: nine shared solar and five
ozone temporal families**. It does not classify every physical division as
IEEE exact. Final GPU forecast, cost and RW1–RW4 evidence belongs to the
composed RC. Source ranking: `lane/w44-topo` commit `7808ad63f8`,
`proofs/w44_topo/E84_DEFAULT_RANK.md`.

The following **13 deferred families remain unchanged**. The ranking used
actual released native program provenance and the static real-input clock
cohorts; these limited observations do not clear other dates or operands.

| Deferred family | Ranked observation / remaining limit |
|---|---|
| Solar UTC-minute / 60 (first local-time divide) | All six audited starts have UTC minute zero; reassess non-midnight starts. |
| Solar SOLCON / 1368.22 source scaling | Up to one ULP in the scale; no sampled temporal predicate flip. |
| Ozone absolute-minute / 1440 | Emitted constant-reciprocal model has zero effect on the sampled clocks. |
| Ozone cyclic factor1, late branch | Native but unselected in the audited weeks. |
| Ozone cyclic factor2, late branch | Native but unselected in the audited weeks. |
| Nest boundary package clock | Up to one quotient ULP; positive closed-interval endpoint and monotonicity preserve the ceil window. |
| Root boundary record clock (two source spans) | Up to one quotient ULP; actual record times and sampled own-step endpoints have zero ceil-record flips. |
| Noah-MP phenology month coordinate | Up to one quotient ULP; no sampled month-index flip, continuous weights. |
| Noah-MP water-hydro dt / 86400 | Zero effect for audited dt 54/18/6 seconds. |
| Noah-MP energy-radiation CLASS exponential argument | Zero exponential-output effect for audited dt 54/18/6. |
| Noah-MP glacier exponential argument | Zero exponential-output effect for Swiss dt18. |
| CLWRF gas interpolation factor1 | Consumed VMR at most one ULP; existing relative bound passes; no sampled gas-floor/calendar/knot flip. |
| CLWRF gas interpolation factor2 | Same isolated bound; both factors together at most two consumed ULP, within the existing relative bound. |

Ordinary physical rate/unit divisions and KF convective substep divisions
remain **un-cleared**: the ranking did not archive their evolving prognostic
operands, and it provides no harmlessness or full-forecast-error bound.
The 55 separately identified source clock/unit spans in eight disabled
feature groups are default-unreachable; that source-span count is not a
quotient-family count and does not clear other configurations.

The CPU product solar implementation also retains pre-existing
transcendental/FMA representation differences against the original-WRF
504-row fixture. Unchanged base and candidate are byte-identical in both
execution modes. Both have 116 declination bit mismatches (maximum
2.98e-8 rad), 35 eager / 87 JIT COSZEN bit mismatches (maximum 5.96e-8),
and exact hour angle and SOLCON. The frozen declination <=1 ULP and
COSZEN <=spacing(float32(1)) bounds pass. The manager amended gate 1 at
2026-10-25T18:09:44Z to separate these existing representation differences
from protected-quotient exactness; no model tolerance changed.
