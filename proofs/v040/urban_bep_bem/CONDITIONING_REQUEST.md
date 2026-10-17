# APPROVED — fixed ROUND01 conditioned R4 gate

Manager approved the pre-registered, candidate-independent ROUND01 packet in this session. The immutable original proposal below is historical; use evidence/ROUND01_APPROVAL.json and its sealed checksums. D6, state, geometry and REAL8 limits remain unchanged.

# Component-gate conditioning decision (no gate changed)

The registered full-column fp32 gate remains OPEN. This request concerns newly
registered component/driver comparisons; no project D6, station, drift, NaN,
resource or release limit is proposed to change.

The first gate used twice the pristine REAL4–REAL8 output difference plus eight
float32 epsilons times the **output** magnitude. That is not a roundoff bound
for a difference of large values. Both reference builds can round to the same
output while another machine-precision evaluation differs by one input ULP.

Measured independent WRF evidence:

- Instrumented pristine BEP soil_temp retains all 86 original full-column arrays
  BYTE. It captures 1,152 material calls. The dynamic JAX helper matches 11,519 of
  11,520 temperatures BYTE; the remaining value differs by one ULP. Potential
  temperature matches 1,151 of 1,152 values BYTE; its remaining difference is one
  ULP. This is source/operator evidence, not a JAX-vs-JAX fidelity comparison.
- With literal REAL arithmetic, WRF's recorded temperatures reproduce all 72
  first-step values per opposing wall set (144 total) BYTE. Remaining full-driver wall-heat differences
  are ≤0.00687 W/m², despite temperatures already passing the fixed state gate.
- The BouLac neutral case has a 5.0863e-7 K/s tendency difference: exactly one
  float32 ULP near 300 K divided by the 60 s step. Its near-zero reference
  tendency makes the current output-relative bound ~1e-14 K/s. The double
  precision BouLac gate passes without any violations.

Requested fixed, input-conditioned budget, derived before a new candidate arm:

1. Keep the existing full-driver REAL8 gate and all state/geometry/diagnostic
   fp32 limits. Keep comparison against pristine WRF, every admitted regime,
   deletion-sensitive mutants, and complete-step checks.
2. For an update tendency `(new-old)/dt`, add a fixed eight-ULP state-roundoff
   term `8*eps32*(abs(new_WRF)+abs(old_WRF))/dt`, using recorded WRF operands.
3. For a flux formed from `A*Twall - B*Tair`, add the fixed forward-error term
   `8*eps32*(abs(A_WRF*Twall_WRF)+abs(B_WRF*Tair_WRF))`, multiplied by WRF's
   density/heat-capacity conversion when the reported field is W/m². Use the
   calibrated WRF operands and no candidate-dependent scale or fitted constant.
4. Require state-roundoff injection controls within this bound, and physical
   perturbation mutants (lost urban drag, missing AC heat, wrong latent INOUT,
   dropped heat/moisture sources) outside it. Report both raw errors and bounds.

The reference operands and coefficients must be recorded and source-bound before
adopting this budget. This request is not a passing verdict; the current gate and
raw failures remain in BEP22/PBL02 until a manager decision and independent review.

Calibrated operand follow-up (BEP25/BEP26, 2026-10-13): the WRITE-only observer
preserves all 87 original arrays BYTE for BEP, BEM AC-on and BEM AC-off. Its
864/1296/1296 wall calls join BYTE to every saved wall/window heat value using
literal WRF REAL4 formulas. The fixed requested eight-epsilon budget is
0.00860–0.04352 W/m². A one-ULP wall-temperature perturbation uses at most
0.1595 of that budget; reversing physical heat sign fails at 6,046 active
entries. This control reads only WRF operands, never candidate outputs.
Receipt: lane artifacts BEP26.json/.npz and observations/flux_stages.json
(SHA256 415a3bded1bf59180eeb442a16f10829a719b6f4042b887c2a679b7479d6b71c).
The current gates are still unchanged. BEM also needs the same fixed bound
propagated through its positive heat-source/Noah-MP sums; its roof/latent and
BouLac gradient-flux differences require their WRF subtraction operands too.
